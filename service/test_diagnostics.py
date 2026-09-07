"""진단 로그는 임시 폴더와 합성 입력으로만 검증합니다."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from service.diagnostics import DiagnosticLog
from service.gesture_controller import GestureController, ServiceConfig
from training.prepare_dataset import QualityConfig


class DiagnosticTests(unittest.TestCase):
    def test_file_roundtrip_no_overwrite_and_no_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            log = DiagnosticLog(path)
            log.emit("prediction", frame_s=1., label="swipe_left", confidence=.9)
            log.close()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["event"], "prediction")
            self.assertIn("utc", rows[0])
            self.assertIn("elapsed_s", rows[0])
            self.assertEqual(rows[-1]["event"], "diagnostics_closed")
            self.assertNotIn("landmarks", rows[0])
            with self.assertRaises(FileExistsError):
                DiagnosticLog(path)

    def test_predictions_and_events_unchanged_and_confirmation_logged(self):
        def run(trace):
            predictor = SimpleNamespace(labels=("swipe_left", "make_fist", "no_gesture", "finger_snap"),
                gesture_schema="right-only", quality=QualityConfig(min_frames=2, min_duration=.1),
                predict=lambda *args: (0, np.array([.97, .01, .01, .01], dtype=np.float32)))
            controller = GestureController(predictor, ServiceConfig(window_seconds=.3, inference_interval=.1))
            controller.diagnostics = trace
            controller.start()
            events = [controller.step(round(i*.1, 6), np.zeros((21, 3), np.float32), True, "Right")
                      for i in range(12)]
            return events
        rows = []
        baseline = run(None)
        self.assertEqual(run(lambda event, **fields: rows.append(dict(event=event, **fields))), baseline)
        predictions = [row for row in rows if row["event"] == "prediction"]
        self.assertEqual([row["candidate_count"] for row in predictions[:2]], [1, 2])
        self.assertTrue(all(row["stable_required"] == 2 and row["neutral_required"] == 2
                            for row in predictions))
        self.assertTrue(all(row["inference_ms"] >= 0 for row in predictions))
        self.assertEqual(sum(row["event"] == "confirmed" for row in rows), 1)
        self.assertEqual(predictions[0]["label"], "swipe_left")
        def failed(*args, **kwargs):
            raise OSError("diagnostics unavailable")
        self.assertEqual(run(failed), baseline)

    def test_writer_size_limit_does_not_raise_in_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            log = DiagnosticLog(Path(directory) / "bounded.jsonl", max_bytes=1)
            with mock.patch("sys.stderr"):
                log.emit("test")
                log.close()
            self.assertTrue(log.failed)
            log.emit("ignored")


if __name__ == "__main__":
    unittest.main()
