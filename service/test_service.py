"""카메라·실제 키보드·실제 학습 자료 없이 합성 입력으로 서비스 계약을 검증합니다."""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from service import actions
from service import main as app
from service.gesture_controller import GestureController, ServiceConfig
from service.hand_tracker import HandTracker, HandFrame, missing_hand
from service.predictor import GesturePredictor, SequenceQualityError
from training import prepare_dataset as preparation
from training.models import LSTMClassifier
from training.gesture_schema import LEGACY_RIGHT_LABELS


def probabilities(label=0, confidence=.95):
    result = np.full(3, (1-confidence)/2, dtype=np.float32)
    result[label] = confidence
    return label, result


def points():
    data = np.zeros((21, 3), dtype=np.float32)
    data[:, :2] = [.3, .4]
    data[9, 1] += .1
    return data


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, state)
        torch.manual_seed(13)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "small.pt"
        quality = preparation.QualityConfig(min_frames=4, min_duration=.2, max_duration=2)
        args = preparation.make_parser().parse_args(["--output-dir", "unused", "--seq-len", "8"])
        config = preparation.build_config(args, quality, {}, {})
        config["label_map"] = dict(LEGACY_RIGHT_LABELS)
        model_config = dict(input_size=66, hidden_size=5, num_layers=2, num_classes=3, dropout=.3)
        self.checkpoint = dict(model_config=model_config, model_state_dict=LSTMClassifier(**model_config).state_dict(),
                               label_map=dict(LEGACY_RIGHT_LABELS), preprocessing_config=config,
                               best_epoch=1, metrics=dict(epoch=1, train_loss=1.1, val_loss=1.0,
                                                          train_accuracy=.5, val_accuracy=.5))
        torch.save(self.checkpoint, self.path)

    def controller(self, gesture_schema="right-only", **changes):
        predictor = mock.Mock()
        predictor.gesture_schema = gesture_schema
        predictor.labels = tuple(LEGACY_RIGHT_LABELS)
        predictor.quality = preparation.QualityConfig(min_frames=2, min_duration=.1, max_duration=2)
        predictor.predict.return_value = probabilities()
        config = ServiceConfig(**{**dict(window_seconds=.3, inference_interval=.1, stable_predictions=3,
                                        rearm_predictions=2, cooldown_seconds=.3, reset_gap_seconds=.6), **changes})
        return GestureController(predictor, config)

    def feed(self, controller, start, end, detected=True):
        events = []
        for tick in range(start, end+1):
            event = controller.step(round(tick*.1, 6), points(), detected)
            if event is not None:
                events.append(event)
        return events

    def test_import_has_no_runtime_io(self):
        # 새 프로세스에서 최초 import를 검사합니다. 실제 카메라 모듈 대신 모의 객체를 넣습니다.
        script = """
import importlib
import sys
from unittest import mock
import torch
cv, mp = mock.MagicMock(), mock.MagicMock()
sys.modules['cv2'], sys.modules['mediapipe'] = cv, mp
with mock.patch('torch.load', side_effect=AssertionError('model load on import')), \\
     mock.patch('pathlib.Path.write_text', side_effect=AssertionError('write on import')):
    for name in ('service', 'service.main', 'service.predictor', 'service.gesture_controller',
                 'service.hand_tracker', 'service.actions'):
        importlib.import_module(name)
cv.VideoCapture.assert_not_called()
mp.tasks.vision.HandLandmarker.create_from_options.assert_not_called()
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_predictor_preprocessing_restore_and_weights_unchanged(self):
        before_file = self.path.read_bytes()
        with mock.patch("torch.load", wraps=torch.load) as load:
            predictor = GesturePredictor(self.path, "cpu")
            self.assertEqual(load.call_count, 1)
            self.assertEqual(load.call_args.kwargs, dict(map_location="cpu", weights_only=True))
        self.assertEqual(predictor.model.num_layers, 2)
        self.assertEqual(predictor.model.hidden_size, 5)
        self.assertEqual(predictor.seq_len, 8)
        raw = np.stack([points() for _ in range(12)])
        times = np.linspace(0, 1.1, 12)
        mask = np.ones(12, dtype=bool)
        mask[4] = False
        raw[4] = np.nan
        expected, _ = preparation.preprocess_sequence(raw, times, mask, 8, predictor.quality)
        np.testing.assert_array_equal(predictor.preprocess(raw, times, mask), expected)
        before = {key: value.clone() for key, value in predictor.model.state_dict().items()}
        seen = []
        handle = predictor.model.register_forward_pre_hook(
            lambda model, inputs: seen.append((inputs[0].shape, inputs[0].dtype, torch.is_grad_enabled())))
        try:
            label, scores = predictor.predict(raw, times, mask)
            other_label, other_scores = predictor.predict(raw, times, mask)
        finally:
            handle.remove()
        self.assertEqual(seen, [(torch.Size([1, 8, 66]), torch.float32, False)] * 2)
        self.assertEqual(label, other_label)
        np.testing.assert_array_equal(scores, other_scores)
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.sum()), 1, places=6)
        self.assertFalse(predictor.model.training)
        self.assertTrue(all(p.grad is None for p in predictor.model.parameters()))
        for key, value in predictor.model.state_dict().items():
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)
        self.assertEqual(self.path.read_bytes(), before_file)

    def test_invalid_checkpoint_contracts(self):
        mutations = [lambda v: v["model_config"].update(input_size=63),
                     lambda v: v["model_config"].update(num_classes=4),
                     lambda v: v["model_config"].update(hidden_size=0),
                     lambda v: v["label_map"].update(swipe_left=1),
                     lambda v: v["preprocessing_config"].update(seq_len=True),
                     lambda v: v["preprocessing_config"].update(preprocessing_version="unknown"),
                     lambda v: v["preprocessing_config"].update(global_standardization={"mean": 0}),
                     lambda v: v["preprocessing_config"].update(mirrored=True),
                     lambda v: v["preprocessing_config"]["features"].update({"65": "different"}),
                     lambda v: v["preprocessing_config"]["quality"].update(min_duration=float("nan")),
                     lambda v: v["preprocessing_config"]["quality"].update(min_frames=True),
                     lambda v: v["model_state_dict"].pop("classifier.bias"),
                     lambda v: v["model_state_dict"].update({"classifier.bias": torch.full((3,), float("inf"))}),
                     lambda v: v.pop("metrics"), lambda v: v.update(best_epoch=True),
                     lambda v: v["metrics"].update(val_accuracy=float("nan"))]
        for mutate in mutations:
            value = copy.deepcopy(self.checkpoint)
            mutate(value)
            torch.save(value, self.path)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, TypeError, RuntimeError)):
                GesturePredictor(self.path, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False), self.assertRaises(ValueError):
            GesturePredictor(self.path, "cuda")

    def test_nonfinite_model_output_and_quality(self):
        predictor = GesturePredictor(self.path, "cpu")
        raw = np.stack([points()]*12)
        times, mask = np.linspace(0, 1.1, 12), np.ones(12, dtype=bool)
        for value in (float("nan"), float("inf")):
            with mock.patch.object(predictor.model, "forward", return_value=torch.full((1, 3), value)):
                with self.assertRaises(FloatingPointError):
                    predictor.predict(raw, times, mask)
        with self.assertRaises(ValueError):
            predictor.preprocess(raw[:1], times[:1], mask[:1])

    def test_real_preprocess_lstm_controller_and_action_smoke(self):
        # 테스트 전용 상수 출력 모델: 실제 성능 측정이 아니라 구성 요소 연결을 검사합니다.
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["model_state_dict"] = {key: torch.zeros_like(value)
                                          for key, value in checkpoint["model_state_dict"].items()}
        checkpoint["model_state_dict"]["classifier.bias"][0] = 8
        torch.save(checkpoint, self.path)
        original = self.path.read_bytes()
        predictor = GesturePredictor(self.path, "cpu")
        controller = GestureController(predictor, ServiceConfig(window_seconds=.5, inference_interval=.1,
                                                                 stable_predictions=2, reset_gap_seconds=.6))
        controller.start()
        output = io.StringIO()
        with redirect_stdout(output):
            for event in self.feed(controller, 0, 20):
                actions.dispatch(event)
        self.assertEqual(output.getvalue().count("[DRY RUN]"), 1)
        self.assertIn("swipe_left", output.getvalue())
        self.assertEqual(original, self.path.read_bytes())

    def test_tracker_initialization_failure_still_cleans_window_and_state(self):
        controller = self.controller()
        controller.start()
        cv = self.fake_cv([])
        with self.assertRaises(FileNotFoundError):
            app.run_camera_loop(controller, cv=cv, tracker_factory=mock.Mock(side_effect=FileNotFoundError("asset")))
        cv.VideoCapture.assert_not_called()
        cv.destroyAllWindows.assert_called_once()
        self.assertFalse(controller.enabled)

    def test_off_start_stop_repeat_and_fresh_warmup(self):
        controller = self.controller()
        self.assertFalse(controller.enabled)
        self.assertEqual(self.feed(controller, 0, 10), [])
        controller.predictor.predict.assert_not_called()
        controller.start()
        self.assertTrue(controller.armed)
        self.feed(controller, 11, 13)
        before = len(controller.buffer)
        controller.start()
        self.assertEqual(len(controller.buffer), before)
        controller.predictor.predict.assert_not_called()
        self.assertEqual(self.feed(controller, 14, 16), ["swipe_left"])
        controller.stop()
        self.assertFalse(controller.enabled)
        self.assertFalse(controller.buffer)
        self.assertIsNone(controller.latest)
        self.assertIsNone(controller.last_event)
        self.assertEqual(controller.candidate_count, 0)
        self.assertEqual(controller.neutral_count, 0)
        self.assertEqual(self.feed(controller, 17, 20), [])
        controller.start()
        self.feed(controller, 21, 22)
        self.assertIsNone(controller.latest)

    def test_time_window_interval_and_bounded_buffer(self):
        controller = self.controller(inference_interval=.2)
        controller.start()
        self.feed(controller, 0, 2)
        controller.predictor.predict.assert_not_called()
        self.feed(controller, 3, 3)
        self.assertEqual(controller.candidate_count, 1)
        self.feed(controller, 4, 4)
        self.assertEqual(controller.candidate_count, 1)
        self.feed(controller, 5, 5)
        self.assertEqual(controller.candidate_count, 2)
        self.feed(controller, 6, 100)
        self.assertLessEqual(len(controller.buffer), 4)
        self.assertGreaterEqual(controller.buffer[0][0], 9.7)
        self.assertEqual(controller.predictor.predict.call_count, 49)

    def test_duplicate_suppression_and_neutral_rearm(self):
        controller = self.controller()
        controller.start()
        self.assertEqual(self.feed(controller, 0, 20), ["swipe_left"])
        self.assertFalse(controller.armed)  # cooldown이 끝나도 같은 손동작은 다시 실행하지 않습니다.
        controller.predictor.predict.return_value = probabilities(2)
        self.assertEqual(self.feed(controller, 21, 22), [])
        self.assertTrue(controller.armed)
        controller.predictor.predict.return_value = probabilities(1)
        self.assertEqual(self.feed(controller, 23, 27), ["make_fist"])

    def test_neutral_requires_cooldown_and_consecutive_predictions(self):
        controller = self.controller(cooldown_seconds=1.0)
        controller.start()
        self.assertEqual(self.feed(controller, 0, 5), ["swipe_left"])
        controller.predictor.predict.return_value = probabilities(2)
        self.feed(controller, 6, 7)
        self.assertFalse(controller.armed)
        self.assertEqual(controller.neutral_count, 2)
        controller.predictor.predict.return_value = probabilities(0, .6)
        self.feed(controller, 8, 8)
        self.assertEqual(controller.neutral_count, 0)
        controller.predictor.predict.return_value = probabilities(2)
        self.feed(controller, 9, 15)
        self.assertTrue(controller.armed)

    def test_suppressed_command_is_deferred_then_fires_when_released(self):
        """YOLO 주먹 유지가 진행 중이면 make_fist 확정을 미루고, 손을 풀면 바로 실행합니다."""
        controller = self.controller()
        controller.start()
        controller.predictor.predict.return_value = probabilities(1)
        controller.suppressed = frozenset({"make_fist"})
        self.assertEqual(self.feed(controller, 0, 8), [], "유예 중에는 확정하지 않습니다.")
        self.assertEqual(controller.candidate, "make_fist")
        self.assertEqual(controller.candidate_count, controller.config.stable_predictions)
        self.assertTrue(controller.armed, "유예는 재무장 상태를 소비하지 않습니다.")
        self.assertIn("deferred", controller.reason)
        controller.suppressed = frozenset()
        self.assertEqual(self.feed(controller, 9, 9), ["make_fist"], "보류가 풀리면 즉시 확정합니다.")

    def test_suppression_does_not_block_other_commands(self):
        controller = self.controller()
        controller.start()
        controller.suppressed = frozenset({"make_fist"})
        self.assertEqual(self.feed(controller, 0, 5), ["swipe_left"])

    def test_low_confidence_class_changes_and_quality_reset(self):
        controller = self.controller()
        controller.start()
        self.feed(controller, 0, 3)
        controller.predictor.predict.return_value = probabilities(1)
        self.feed(controller, 4, 4)
        self.assertEqual(controller.candidate_count, 1)
        controller.predictor.predict.return_value = probabilities(1, .5)
        self.assertEqual(self.feed(controller, 5, 5), [])
        self.assertEqual(controller.candidate_count, 0)
        controller.predictor.predict.side_effect = SequenceQualityError("too_few_frames")
        self.assertEqual(self.feed(controller, 6, 6), [])
        self.assertIsNone(controller.latest)
        self.assertIn("Quality hold", controller.reason)

    def test_missing_frames_kept_but_not_counted_as_neutral(self):
        controller = self.controller()
        controller.start()
        self.feed(controller, 0, 5)
        controller.predictor.predict.return_value = probabilities(2)
        self.feed(controller, 6, 6)
        self.assertEqual(controller.neutral_count, 1)
        self.feed(controller, 7, 7, detected=False)
        self.assertEqual(controller.neutral_count, 0)
        self.assertIsNone(controller.latest)
        self.feed(controller, 8, 8)
        raw, times, mask = controller.predictor.predict.call_args.args
        self.assertTrue((~mask).any())
        self.assertTrue(np.isnan(raw[~mask]).all())
        self.assertIn(.7, times)
        self.assertFalse(controller.armed)

    def test_long_loss_and_time_gap_require_neutral(self):
        controller = self.controller()
        controller.start()
        self.feed(controller, 0, 5)
        self.feed(controller, 6, 13, detected=False)
        self.assertFalse(controller.buffer)
        self.assertIsNone(controller.last_event)
        self.assertIsNone(controller.latest)
        self.assertFalse(controller.armed)
        self.assertEqual(self.feed(controller, 14, 22), [])
        controller.predictor.predict.return_value = probabilities(2)
        self.feed(controller, 23, 24)
        self.assertTrue(controller.armed)
        controller.step(5.0, points(), True)
        self.assertEqual(len(controller.buffer), 1)
        self.assertFalse(controller.armed)
        self.assertIsNone(controller.latest)

    def test_fatal_input_and_inference_errors_turn_off(self):
        for error in (RuntimeError("model failed"), FloatingPointError("NaN")):
            controller = self.controller()
            controller.start()
            controller.predictor.predict.side_effect = error
            with self.assertRaises(type(error)):
                self.feed(controller, 0, 3)
            self.assertFalse(controller.enabled)
            self.assertFalse(controller.buffer)
        controller = self.controller()
        controller.start()
        controller.step(0, points(), True)
        with self.assertRaises(ValueError):
            controller.step(0, points(), True)
        self.assertFalse(controller.enabled)
        controller.start()
        with self.assertRaises(ValueError):
            controller.step(1, np.zeros((20, 3)), True)
        controller.start()
        controller.predictor.predict.return_value = (0, np.array([np.nan]*3))
        with self.assertRaises(ValueError):
            self.feed(controller, 0, 3)

    def test_invalid_settings_and_cli(self):
        for changes in (dict(window_seconds=0), dict(inference_interval=float("nan")),
                        dict(confidence_threshold=1.1), dict(cooldown_seconds=-1),
                        dict(stable_predictions=True), dict(rearm_predictions=0), dict(reset_gap_seconds=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ServiceConfig(**changes).validate()
        with self.assertRaises(ValueError):
            self.controller(window_seconds=3)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            app.make_parser().parse_args([])
        with mock.patch.object(app, "run_camera_loop") as camera, redirect_stderr(io.StringIO()):
            self.assertEqual(app.main(["--checkpoint", str(self.path), "--camera-index", "-1"]), 1)
            self.assertEqual(app.main(["--checkpoint", str(self.path), "--window-seconds", "nan"]), 1)
            self.assertEqual(app.main(["--checkpoint", str(self.root / "missing.pt")]), 1)
            camera.assert_not_called()

    def test_action_logs_only(self):
        output = io.StringIO()
        with redirect_stdout(output):
            actions.dispatch("no_gesture")
            self.assertEqual(output.getvalue(), "")
            actions.dispatch("swipe_left")
            actions.dispatch("make_fist")
        self.assertEqual(output.getvalue().splitlines(), [
            "[DRY RUN] gesture=swipe_left, action=SWIPE_LEFT_ACTION",
            "[DRY RUN] gesture=make_fist, action=MAKE_FIST_ACTION"])
        with self.assertRaises(ValueError):
            actions.dispatch("unknown")

    def fake_cv(self, keys):
        cv = mock.Mock()
        cv.CAP_DSHOW, cv.WND_PROP_VISIBLE, cv.FONT_HERSHEY_SIMPLEX = 1, 2, 3
        cv.VideoCapture.return_value.isOpened.return_value = True
        cv.VideoCapture.return_value.read.side_effect = lambda: (True, np.zeros((480, 640, 3), dtype=np.uint8))
        cv.waitKey.side_effect = keys
        cv.getWindowProperty.return_value = 1
        return cv

    def test_camera_mock_keys_and_cleanup(self):
        controller = self.controller(window_seconds=.1, stable_predictions=1)
        tracker = mock.Mock()
        tracker.process.return_value = HandFrame(points(), True, "Right detected", "Right")
        cv = self.fake_cv([-1, ord("s"), ord("s"), ord("x"), -1, 27])
        clock = iter([0, 0, .1, .1, .1, .2, .2, .2, .3, .3, .4, .4, .5])
        action = mock.Mock()
        app.run_camera_loop(controller, cv=cv, tracker_factory=lambda: tracker,
                            clock=lambda: next(clock), action=action)
        self.assertEqual(tracker.process.call_count, 2)
        action.assert_called_once_with("swipe_left")
        tracker.close.assert_called_once()
        cv.VideoCapture.return_value.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()
        self.assertFalse(controller.enabled)
        self.assertIsNone(controller.latest)

    def test_camera_failure_tracker_failure_and_interrupt_cleanup(self):
        for error in (RuntimeError("tracker failure"), KeyboardInterrupt()):
            controller = self.controller()
            tracker = mock.Mock()
            tracker.process.side_effect = error
            cv = self.fake_cv([ord("s")])
            with self.assertRaises(type(error)):
                app.run_camera_loop(controller, cv=cv, tracker_factory=lambda: tracker, clock=lambda: 0)
            tracker.close.assert_called_once()
            cv.VideoCapture.return_value.release.assert_called_once()
            cv.destroyAllWindows.assert_called_once()
            self.assertFalse(controller.enabled)
        cv = self.fake_cv([27])
        tracker = mock.Mock()
        cv.VideoCapture.return_value.read.side_effect = lambda: (False, None)
        with self.assertRaisesRegex(RuntimeError, "프레임"):
            app.run_camera_loop(self.controller(), cv=cv, tracker_factory=lambda: tracker)
        tracker.close.assert_called_once()
        cv.VideoCapture.return_value.release.assert_called_once()
        cv = self.fake_cv([27])
        first, second = mock.Mock(), mock.Mock()
        first.isOpened.return_value = second.isOpened.return_value = False
        cv.VideoCapture.side_effect = [first, second]
        with self.assertRaises(RuntimeError):
            app.run_camera_loop(self.controller(), cv=cv, tracker_factory=lambda: tracker)
        first.release.assert_called_once()
        second.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()

    def test_late_inference_event_is_dropped(self):
        controller = self.controller()
        with mock.patch.object(controller, "step", return_value="swipe_left"):
            tracker = mock.Mock()
            tracker.process.return_value = HandFrame(points(), True, "Right detected", "Right")
            cv = self.fake_cv([ord("s"), 27])
            times = iter([0, 1, 1, 1.1])
            action = mock.Mock()
            app.run_camera_loop(controller, cv=cv, tracker_factory=lambda: tracker,
                                clock=lambda: next(times), action=action)
            action.assert_not_called()

    def test_tracker_timestamps_handedness_and_missing_model(self):
        with self.assertRaises(FileNotFoundError):
            HandTracker(self.root / "missing.task")
        for hands in (("Left",), ("Right", "Left")):
            with self.assertRaisesRegex(ValueError, "Right"):
                HandTracker(self.root / "missing.task", allowed_hands=hands)
        tracker = object.__new__(HandTracker)
        tracker.allowed_hands = ("Right",)
        tracker.cv2 = mock.Mock()
        tracker.mp = mock.Mock()
        tracker.detector = mock.Mock()
        tracker.origin, tracker.last_timestamp_ms = None, -1
        landmark = SimpleNamespace(x=.3, y=.4, z=0)
        category = SimpleNamespace(category_name="Right")
        result = SimpleNamespace(hand_landmarks=[[landmark]*21], handedness=[[category]])
        tracker.detector.detect_for_video.return_value = result
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for timestamp in (10.0, 10.0001, 10.0002):
            self.assertTrue(tracker.process(frame, timestamp).detected)
        self.assertEqual([call.args[1] for call in tracker.detector.detect_for_video.call_args_list], [0, 1, 2])
        category.category_name = "Left"
        hand = tracker.process(frame, 10.1)
        self.assertFalse(hand.detected)
        self.assertTrue(np.isnan(hand.landmarks).all())
        self.assertIn("Left", hand.status)
        tracker.allowed_hands = ("Right", "Left")
        hand = tracker.process(frame, 10.15)
        self.assertFalse(hand.detected)  # 과거 설정을 넣어도 왼손 허용으로 우회하지 못합니다.
        self.assertEqual(hand.handedness, "Left")
        category.category_name = "Right"
        hand = tracker.process(frame, 10.16)
        self.assertTrue(hand.detected)
        np.testing.assert_allclose(hand.landmarks[:, 0], .3)  # 서비스에서 다시 반전하지 않습니다.
        result.hand_landmarks = []
        self.assertFalse(tracker.process(frame, 10.2).detected)
        detector = tracker.detector
        tracker.close()
        tracker.close()
        detector.close.assert_called_once()

    def test_old_left_hand_four_class_checkpoint_rejected(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["label_map"]["swipe_right"] = 3
        checkpoint["model_config"]["num_classes"] = 4
        checkpoint["model_state_dict"] = LSTMClassifier(**checkpoint["model_config"]).state_dict()
        torch.save(checkpoint, self.path)
        with self.assertRaises(ValueError):
            GesturePredictor(self.path, "cpu")

    def test_finger_snap_checkpoint_predictor_controller_and_action(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["label_map"] = dict(preparation.LABEL_MAP)
        checkpoint["preprocessing_config"]["label_map"] = dict(preparation.LABEL_MAP)
        checkpoint["model_config"]["num_classes"] = 4
        model = LSTMClassifier(**checkpoint["model_config"])
        # 합성 가중치로 클래스 3을 강제합니다. 실제 분류 성능 검사가 아닙니다.
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([-10., -10., -10., 10.]))
        checkpoint["model_state_dict"] = model.state_dict()
        torch.save(checkpoint, self.path)
        predictor = GesturePredictor(self.path, "cpu")
        self.assertEqual(predictor.labels, ("swipe_left", "make_fist", "no_gesture", "finger_snap"))
        controller = GestureController(predictor, ServiceConfig(
            window_seconds=.5, inference_interval=.1, stable_predictions=1))
        controller.start()
        self.assertEqual(self.feed(controller, 0, 12), ["finger_snap"])
        output = io.StringIO()
        with redirect_stdout(output):
            actions.dispatch("finger_snap")
        self.assertIn("FINGER_SNAP_ACTION", output.getvalue())
        controller.step(1.3, points(), True, "Left")
        self.assertFalse(controller.buffer)
        self.assertIsNone(controller.latest)

    def test_right_hand_rearm_after_left_input(self):
        controller = self.controller(stable_predictions=1)
        def set_label(label):
            scores = np.full(3, .02, dtype=np.float32)
            scores[label] = .96
            controller.predictor.predict.return_value = label, scores
        def feed(start, end, hand):
            return [event for tick in range(start, end+1)
                    if (event := controller.step(round(tick*.1, 6), points(), True, hand)) is not None]
        controller.start()
        set_label(0)
        self.assertEqual(feed(0, 5, "Right"), ["swipe_left"])
        calls = controller.predictor.predict.call_count
        self.assertEqual(feed(6, 6, "Left"), [])
        self.assertEqual(len(controller.buffer), 0)
        self.assertIsNone(controller.latest)
        self.assertIsNone(controller.last_event)
        self.assertFalse(controller.armed)
        set_label(1)
        self.assertEqual(feed(7, 10, "Left"), [])
        self.assertEqual(controller.predictor.predict.call_count, calls)
        self.assertEqual(feed(11, 16, "Right"), [])  # 새 구간 + 오른손 중립 확인 필요
        self.assertEqual(controller.neutral_count, 0)
        set_label(2)
        self.assertEqual(feed(17, 18, "Right"), [])
        self.assertTrue(controller.armed)
        set_label(1)
        self.assertEqual(feed(19, 19, "Right"), ["make_fist"])
        for label in range(3):
            controller.stop()
            controller.start()
            set_label(label)
            calls = controller.predictor.predict.call_count
            self.assertEqual(feed(25, 32, "Left"), [])
            self.assertEqual(controller.predictor.predict.call_count, calls)
            self.assertFalse(controller.buffer)
            self.assertEqual(controller.neutral_count, 0)
        controller.stop()
        controller.start()
        self.assertEqual(feed(33, 35, ""), [])
        with self.assertRaises(ValueError):
            actions.dispatch("unsupported_gesture")

    def test_three_class_left_flushes_existing_right_sequence(self):
        controller = self.controller()
        controller.start()
        self.feed(controller, 0, 3)
        self.assertTrue(controller.buffer)
        calls = controller.predictor.predict.call_count
        controller.step(.4, np.full((21, 3), np.nan, dtype=np.float32), False, "Left")
        self.assertFalse(controller.buffer)
        self.assertIsNone(controller.latest)
        self.assertEqual(controller.candidate_count, 0)
        self.assertEqual(controller.predictor.predict.call_count, calls)
        self.assertFalse(controller.armed)

    def test_cli_uses_right_only_tracker_and_blocks_left(self):
        controller = self.controller()
        tracker = mock.Mock()
        tracker.process.return_value = HandFrame(points(), True, "Left detected", "Left")
        action = mock.Mock()
        cv = self.fake_cv([ord("s"), 27])
        with mock.patch.object(app, "HandTracker", return_value=tracker) as factory:
            app.run_camera_loop(controller, cv=cv, clock=lambda: 0, action=action)
        factory.assert_called_once_with()
        action.assert_not_called()
        controller.predictor.predict.assert_not_called()
        tracker.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
