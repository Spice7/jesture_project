"""오른손 복원 및 새 라벨 수집 회귀 검사. 실제 카메라나 학습 데이터는 사용하지 않습니다."""

from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

if __package__:
    from . import prepare_dataset as preparation
    from .gesture_schema import schema_from_config, validate_label_map
    from .test_prepare_dataset import fixture
else:
    import prepare_dataset as preparation
    from gesture_schema import schema_from_config, validate_label_map
    from test_prepare_dataset import fixture


class RightOnlyTests(unittest.TestCase):
    def test_removed_augmentation_options_are_rejected(self):
        for option in (["--augment-horizontal"], ["--gesture-schema", "bilateral"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                preparation.make_parser().parse_args(["--output-dir", "unused", *option])
        self.assertFalse(hasattr(preparation, "horizontal_flip"))

    def test_only_original_right_hand_config_is_valid(self):
        args = preparation.make_parser().parse_args(["--output-dir", "unused"])
        config = preparation.build_config(args, preparation.QualityConfig(), {}, {})
        self.assertEqual(schema_from_config(config), "right-only")
        with self.assertRaises(ValueError):
            validate_label_map({**preparation.LABEL_MAP, "swipe_right": 3})
        for change in ({"augmentation": {"type": "horizontal_flip"}},
                       {"preprocessing_version": "2.0.0"},
                       {"gesture_schema": "bilateral"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                schema_from_config({**config, **change})

    def test_new_label_collection_and_right_hand_quality_gate(self):
        from programs import collect_gesture as collector
        data = fixture(label="finger_snap")
        count = len(data["timestamps"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(collector, "DATASET_DIR", root), redirect_stdout(io.StringIO()):
                kwargs = dict(
                    participant_id="p001", label=collector.sanitize_name("finger_snap"), sample_id=1,
                    landmarks_buffer=list(data["landmarks"]),
                    world_landmarks_buffer=list(data["landmarks"]),
                    timestamps_buffer=list(data["timestamps"]),
                    detected_buffer=list(data["detected"]),
                    handedness_buffer=["Right"] * count,
                    handedness_confidence_buffer=[.99] * count,
                )
                saved, path = collector.save_sequence(**kwargs)
                self.assertTrue(saved)
                self.assertTrue(Path(path).is_relative_to(root))
                with np.load(path, allow_pickle=False) as result:
                    self.assertEqual(result["label"].item(), "finger_snap")
                    self.assertEqual(result["handedness"].item(), "Right")
                    self.assertFalse(result["mirrored"].item())
                self.assertEqual(collector.get_next_sample_id("p001", "finger_snap"), 2)
                # 새 클래스도 기존 좌표 처리와 동일하게 전처리됩니다.
                row, feature = preparation.scan_file(Path(path), root, {}, 32, preparation.QualityConfig())
                self.assertEqual(row["status"], "accepted")
                self.assertEqual(feature.shape, (32, 66))
                kwargs.update(sample_id=2, handedness_buffer=["Left"] * count)
                self.assertFalse(collector.save_sequence(**kwargs)[0])
                self.assertEqual(len(list(root.rglob("*.npz"))), 1)

    def test_legacy_gui_shortcut_removed_without_mutating_input(self):
        from service.settings import defaults, validate_settings
        old = defaults()
        old["shortcuts"]["swipe_right"] = {"key": "Right", "modifiers": []}
        old["notifications"] = False
        result = validate_settings(old)
        self.assertNotIn("swipe_right", result["shortcuts"])
        self.assertIn("swipe_right", old["shortcuts"])
        self.assertFalse(result["notifications"])


if __name__ == "__main__":
    unittest.main()
