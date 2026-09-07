"""가중치와 카메라 없이 유지 판정만 검사합니다. 실제 손모양 인식 성능과는 무관합니다."""

import unittest

from service.yolo_gate import (DEFAULT_WEIGHTS, GATE_ACTIONS, GATE_LABELS, GateConfig,
                               HoldDetector, suppressed_commands)


def feed(detector, start, label, seconds, step=.1):
    """label을 seconds 동안 step 간격으로 넣고 확정된 동작들을 모읍니다."""
    fired, now = [], start
    end = start + seconds
    while now <= end + 1e-9:
        result = detector.update(now, label)
        if result is not None:
            fired.append((round(now - start, 3), result))
        now += step
    return fired, now - step


class ImportTests(unittest.TestCase):
    def test_import_does_not_load_ultralytics_or_weights(self):
        """기존 모듈과 같이 import만으로 모델을 올리지 않는지 새 프로세스에서 확인합니다."""
        import subprocess
        import sys
        from pathlib import Path
        script = """
import sys
import importlib
importlib.import_module('service.yolo_gate')
assert 'ultralytics' not in sys.modules, 'import만으로 ultralytics를 올렸습니다.'
assert 'torch' not in sys.modules, 'import만으로 torch를 올렸습니다.'
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


class ConfigTests(unittest.TestCase):
    def test_rejects_impossible_settings(self):
        for bad in (dict(hold_seconds=0), dict(hold_seconds=float("nan")), dict(confidence=1.5),
                    dict(release_seconds=-1), dict(imgsz=16), dict(imgsz=640.0),
                    dict(suppress_after_seconds=5.0)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GateConfig(**bad).validate()

    def test_actions_cover_every_label(self):
        self.assertEqual(set(GATE_ACTIONS), set(GATE_LABELS))
        self.assertEqual(GATE_ACTIONS["start"], "arm")
        self.assertEqual(GATE_ACTIONS["stop"], "disarm")
        self.assertEqual(GATE_ACTIONS["cancel"], "toggle_window")


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.detector = HoldDetector(GateConfig(hold_seconds=3., release_seconds=.4))

    def test_confirms_once_after_three_seconds(self):
        fired, _ = feed(self.detector, 100., "start", 5.)
        self.assertEqual(len(fired), 1, f"3초 유지에 한 번만 확정해야 합니다: {fired}")
        self.assertEqual(fired[0][1], "start")
        self.assertGreaterEqual(fired[0][0], 3.)

    def test_short_hold_does_not_confirm(self):
        fired, _ = feed(self.detector, 100., "start", 2.5)
        self.assertEqual(fired, [])
        self.assertAlmostEqual(self.detector.progress(102.5), 2.5 / 3., places=2)

    def test_release_is_required_before_confirming_again(self):
        fired, now = feed(self.detector, 100., "start", 4.)
        self.assertEqual(len(fired), 1)
        # 계속 유지해도 재확정하지 않습니다.
        more, now = feed(self.detector, now + .1, "start", 4.)
        self.assertEqual(more, [])
        # 손을 충분히 놓았다가 다시 하면 새로 확정합니다.
        self.detector.update(now + .1, None)
        self.detector.update(now + .6, None)
        again, _ = feed(self.detector, now + 1., "start", 3.5)
        self.assertEqual(len(again), 1)

    def test_brief_dropout_keeps_the_hold(self):
        now = 100.
        for _ in range(20):  # 2.0초 유지
            self.detector.update(now, "start")
            now += .1
        self.detector.update(now, None)  # 한 프레임 놓침 (0.1초 < release 0.4초)
        now += .1
        fired, _ = feed(self.detector, now, "start", 1.2)
        self.assertEqual(len(fired), 1, "짧은 끊김은 유지가 이어져야 합니다.")

    def test_long_dropout_restarts_the_hold(self):
        now = 100.
        for _ in range(25):  # 2.5초 유지
            self.detector.update(now, "start")
            now += .1
        for _ in range(6):  # 0.6초 미검출 (> release 0.4초)
            self.detector.update(now, None)
            now += .1
        fired, _ = feed(self.detector, now, "start", 2.5)
        self.assertEqual(fired, [], "길게 놓친 뒤에는 처음부터 다시 세야 합니다.")

    def test_switching_pose_restarts_the_hold(self):
        now = 100.
        for _ in range(25):
            self.detector.update(now, "start")
            now += .1
        fired, _ = feed(self.detector, now, "stop", 2.5)
        self.assertEqual(fired, [], "다른 손모양으로 바뀌면 누적을 이어받지 않습니다.")
        more, _ = feed(self.detector, now + 2.6, "stop", 1.)
        self.assertEqual([label for _, label in more], ["stop"])

    def test_rejects_unknown_label_and_backward_time(self):
        with self.assertRaises(ValueError):
            self.detector.update(100., "make_fist")
        self.detector.update(100., "start")
        with self.assertRaises(ValueError):
            self.detector.update(99., "start")
        with self.assertRaises(ValueError):
            self.detector.update(float("inf"), "start")

    def test_progress_and_pending_report_the_candidate(self):
        self.assertEqual(self.detector.pending(100.), (None, 0.))
        self.assertEqual(self.detector.progress(100.), 0.)
        now = 100.
        for _ in range(10):  # 100.0 ~ 100.9를 넣고 now는 101.0이 됩니다.
            self.detector.update(now, "cancel")
            now += .1
        label, elapsed = self.detector.pending(now)
        self.assertEqual(label, "cancel")
        self.assertAlmostEqual(elapsed, 1., places=2)
        feed(self.detector, now, "cancel", 2.5)
        # 확정 뒤 계속 유지하는 동안은 후보가 아니고 진행률은 1입니다.
        self.assertEqual(self.detector.pending(now + 2.5)[0], None)
        self.assertEqual(self.detector.progress(now + 2.5), 1.)


class SuppressionTests(unittest.TestCase):
    def setUp(self):
        self.detector = HoldDetector(GateConfig(hold_seconds=3., suppress_after_seconds=1.))

    def test_make_fist_is_free_until_stop_persists(self):
        now = 100.
        self.assertEqual(suppressed_commands(self.detector, now), frozenset())
        for _ in range(9):  # 0.9초 — 아직 유예하지 않습니다.
            self.detector.update(now, "stop")
            now += .1
        self.assertEqual(suppressed_commands(self.detector, now), frozenset())

    def test_make_fist_is_held_while_stop_is_building_up(self):
        now = 100.
        for _ in range(15):  # 1.5초 — 유예 구간입니다.
            self.detector.update(now, "stop")
            now += .1
        self.assertEqual(suppressed_commands(self.detector, now), frozenset({"make_fist"}))

    def test_releasing_before_three_seconds_frees_make_fist(self):
        now = 100.
        for _ in range(20):
            self.detector.update(now, "stop")
            now += .1
        self.assertIn("make_fist", suppressed_commands(self.detector, now))
        for _ in range(5):  # 손을 폄 → 0.5초 미검출로 후보 해제
            self.detector.update(now, None)
            now += .1
        self.assertEqual(suppressed_commands(self.detector, now), frozenset())

    def test_other_poses_never_suppress_commands(self):
        now = 100.
        for label in ("start", "cancel"):
            self.detector.reset()
            for _ in range(20):
                self.detector.update(now, label)
                now += .1
            with self.subTest(label=label):
                self.assertEqual(suppressed_commands(self.detector, now), frozenset())
            now += 1.


@unittest.skipUnless(DEFAULT_WEIGHTS.is_file(), f"게이트 가중치 없음: {DEFAULT_WEIGHTS}")
class WeightsTests(unittest.TestCase):
    """실제 가중치의 클래스 계약과 반환 형식만 확인합니다. 인식 정확도 평가가 아닙니다."""

    def test_real_weights_match_the_label_contract(self):
        import numpy as np
        from service.yolo_gate import YoloGate

        gate = YoloGate(DEFAULT_WEIGHTS, "cpu", GateConfig(imgsz=320))
        self.assertEqual(set(gate.names.values()), set(GATE_LABELS))
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        label, score = gate.predict(blank)
        self.assertIsNone(label, "빈 프레임에서는 손모양을 확정하지 않아야 합니다.")
        self.assertEqual(score, 0.)
        action, label, score, progress = gate.step(100., blank)
        self.assertIsNone(action)
        self.assertEqual((label, score, progress), (None, 0., 0.))

    def test_missing_weights_file_is_reported(self):
        from service.yolo_gate import YoloGate
        with self.assertRaises(FileNotFoundError):
            YoloGate(DEFAULT_WEIGHTS.parent / "does-not-exist.pt", "cpu")


if __name__ == "__main__":
    unittest.main()
