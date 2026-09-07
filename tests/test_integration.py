"""통합 경계 회귀 검사. 가짜 모델/학습 데이터 파일은 생성하지 않는다."""
import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gesture import config
from gesture.actions import ActionMapper
from gesture.gate import StaticGate
from gesture.model import build_model
from gesture.pipeline import Pipeline
from gesture_model import GestureDetector
from yolo.yolo import prepared_dataset, resolve_dataset

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = ROOT / "models/static/checkpoints/last_100.pt"


@contextmanager
def cwd(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class IntegrationTests(unittest.TestCase):
    def test_project_paths_ignore_cwd_and_inference_does_not_require_dataset(self):
        with tempfile.TemporaryDirectory() as directory, cwd(directory):
            paths = config.runtime_files(static_model="models/static/checkpoints/last_100.pt")
            self.assertEqual(paths["static YOLO"], WEIGHT)
            self.assertFalse(any("dataset" in str(p) for p in paths.values()))
            self.assertEqual(config.project_path("gestures.json"), ROOT / "gestures.json")

    def test_missing_models_are_reported_without_camera(self):
        with patch.object(Path, "is_file", autospec=True, side_effect=lambda p: p.suffix == ".json"), \
                self.assertRaisesRegex(FileNotFoundError, "GRU/LSTM weight"):
            config.require_runtime_files()

    def test_yolo_yaml_and_ultralytics_agree_from_other_cwd(self):
        from ultralytics.data.utils import check_det_dataset, img2label_paths
        with tempfile.TemporaryDirectory() as directory, cwd(directory):
            resolved = resolve_dataset(ROOT / "dataset/data.yaml")
            with prepared_dataset(str(ROOT / "dataset/data.yaml"), ()) as prepared, \
                    patch("ultralytics.data.utils.check_font"):
                actual = check_det_dataset(prepared, autodownload=False)
            for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
                expected = ROOT / "dataset" / folder / "images"
                self.assertEqual(Path(resolved[key]), expected)
                self.assertEqual(Path(actual[key]), expected)
                # 경로 변환만 검사한다. 이미지/라벨 파일을 만들지 않는다.
                labels = img2label_paths([str(expected / "sample.jpg")])
                self.assertEqual(Path(labels[0]), expected.parent / "labels/sample.txt")

    def test_empty_training_data_is_actionable(self):
        with tempfile.TemporaryDirectory() as empty, \
                patch("yolo.yolo.resolve_dataset", return_value={"train": empty, "names": ["start"]}):
            with self.assertRaisesRegex(FileNotFoundError, "학습/평가 이미지"):
                with prepared_dataset(str(ROOT / "dataset/data.yaml"), ("train",)):
                    pass

    def test_yolo_predict_never_builds_training_arguments(self):
        import yolo.yolo as entry
        settings = {"paths": {"trained_model": str(WEIGHT), "project": "../runs/yolo"}}
        args = argparse.Namespace(config=Path("unused.yaml"), mode="predict", source="0", device="cpu")
        with patch.object(entry, "parse_args", return_value=args), \
                patch.object(entry, "load_config", return_value=(settings, ROOT / "yolo")), \
                patch.object(entry, "build_common_args", side_effect=AssertionError("training dependency")), \
                patch.object(entry, "YOLO"), patch.object(entry, "run_predict") as predict:
            entry.main()
            predict.assert_called_once()

    def test_hold_miss_tolerance_one_shot_and_rearm(self):
        with patch("gesture_model.gesture_detector.YOLO") as model:
            model.return_value.names = {0: "start", 1: "stop", 2: "cancel"}
            detector = GestureDetector(WEIGHT, hold_seconds=1, miss_tolerance_seconds=0.3)
        def step(t, pose):
            with patch("gesture_model.gesture_detector.time.monotonic", return_value=t):
                return detector._update_state(pose, 0.9)
        self.assertFalse(step(0, "start").confirmed)
        self.assertFalse(step(0.8, "start").confirmed)
        self.assertEqual(step(1.0, None).gesture, "start")
        self.assertTrue(step(1.05, "start").confirmed)
        self.assertFalse(step(1.1, "start").confirmed)
        self.assertIsNone(step(1.5, None).gesture)
        self.assertFalse(step(2, "start").confirmed)
        self.assertTrue(step(3, "start").confirmed)

    def test_gate_error_closes_and_clears_pending_action(self):
        with patch("gesture.pipeline.GestureClassifier") as classifier, \
                patch("gesture.pipeline.HandTracker") as tracker, patch("gesture.pipeline.SessionLog"):
            classifier.return_value.labels = config.LABELS
            tracker.return_value.process.return_value = (None, "")
            gate = MagicMock(hold_seconds=2.0)
            gate.process.side_effect = RuntimeError("test inference error")
            pipe = Pipeline(gate=gate)
            pipe.set_gate(True)
            pipe._pending = ("make_fist", 1, 0, 0)
            state = pipe.step(np.zeros((64, 64, 3), dtype=np.uint8))
            self.assertFalse(state.gate_open)
            self.assertIsNone(pipe._pending)
            self.assertIn("test inference error", state.gate_error)
            self.assertEqual(pipe.mapper.log, [])
            pipe.set_gate(True, "manual")
            self.assertFalse(pipe.gate_open)
            pipe.close()

    def test_static_events_share_gate_and_action_cooldown(self):
        with patch("gesture.pipeline.GestureClassifier") as classifier, \
                patch("gesture.pipeline.HandTracker"), patch("gesture.pipeline.SessionLog"):
            classifier.return_value.labels = config.LABELS
            gate = MagicMock(hold_seconds=2.0)
            pipe = Pipeline(gate=gate, dry_run=True)
            self.assertFalse(pipe._on_pose("cancel", 0.9, 1).executed)
            pipe._on_pose("start", 0.9, 2)
            self.assertTrue(pipe._on_pose("cancel", 0.9, 3).executed)
            self.assertFalse(pipe._on_pose("cancel", 0.9, 3.1).executed)
            pipe._on_pose("stop", 0.9, 4)
            self.assertFalse(pipe.gate_open)
            pipe.close()

    def test_dynamic_rnn_shapes_without_creating_weights(self):
        for arch in ("gru", "lstm"):
            model = build_model(arch).cpu().eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.zeros(2, 30, 63)).shape), (2, 4))

    def test_real_static_model_load_and_blank_inference(self):
        gate = StaticGate(WEIGHT, device="cpu", imgsz=64)
        gate.every = 1
        result = gate.process(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertIsNone(result.confirmed)
        self.assertIsNotNone(gate.det.last_prediction)
        self.assertEqual(gate.det.last_prediction.plot().shape, (64, 64, 3))

    def test_gradio_and_ui_import_without_starting_devices(self):
        from programs.data_preprocessing import build_app
        app = build_app()
        self.assertTrue(app.blocks)
        self.assertTrue(hasattr(importlib.import_module("scripts.gesture_app"), "App"))


if __name__ == "__main__":
    unittest.main()
