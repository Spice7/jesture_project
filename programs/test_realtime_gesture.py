import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from training.models import LSTMClassifier
from training.prepare_dataset import QualityConfig
from programs.realtime_gesture import (
    FrameWindow,
    ProbabilitySmoother,
    RuntimeBundle,
    draw_dashboard,
    load_runtime_bundle,
    predict_probabilities,
)


class FrameWindowTests(unittest.TestCase):
    def test_arrays_keep_only_frames_inside_recent_window(self):
        window = FrameWindow(max_seconds=1.0)
        landmarks = np.zeros((21, 3), dtype=np.float32)

        window.append(0.0, landmarks, True)
        window.append(0.5, landmarks + 1.0, True)
        window.append(1.2, landmarks + 2.0, False)

        points, timestamps, detected = window.arrays()

        self.assertEqual(points.shape, (2, 21, 3))
        np.testing.assert_allclose(timestamps, [0.5, 1.2])
        np.testing.assert_array_equal(detected, [True, False])
        self.assertEqual(points.dtype, np.float32)
        self.assertEqual(detected.dtype, np.bool_)

    def test_append_rejects_non_increasing_timestamp(self):
        window = FrameWindow(max_seconds=1.0)
        landmarks = np.zeros((21, 3), dtype=np.float32)
        window.append(1.0, landmarks, True)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            window.append(1.0, landmarks, True)


class ProbabilitySmootherTests(unittest.TestCase):
    def test_update_returns_recent_probability_mean(self):
        smoother = ProbabilitySmoother(window_size=2, num_classes=3)

        first = smoother.update(np.array([0.8, 0.1, 0.1]))
        second = smoother.update(np.array([0.2, 0.6, 0.2]))
        third = smoother.update(np.array([0.1, 0.1, 0.8]))

        np.testing.assert_allclose(first, [0.8, 0.1, 0.1])
        np.testing.assert_allclose(second, [0.5, 0.35, 0.15])
        np.testing.assert_allclose(third, [0.15, 0.35, 0.5])

    def test_update_rejects_invalid_probability_vector(self):
        smoother = ProbabilitySmoother(window_size=2, num_classes=3)

        with self.assertRaises(ValueError):
            smoother.update(np.array([0.4, 0.4, 0.4]))


class RuntimeBundleTests(unittest.TestCase):
    def _save_checkpoint(self, path: Path, input_size: int = 66) -> None:
        label_map = {"swipe_left": 0, "make_fist": 1, "no_gesture": 2}
        model_config = {
            "input_size": input_size,
            "hidden_size": 8,
            "num_layers": 1,
            "num_classes": 3,
            "dropout": 0.0,
        }
        model = LSTMClassifier(**model_config)
        checkpoint = {
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "preprocessing_config": {
                "seq_len": 32,
                "feature_dim": 66,
                "label_map": label_map,
                "quality": asdict(QualityConfig()),
            },
            "best_val_metrics": {"accuracy": 0.75, "macro_f1": 0.7},
        }
        torch.save(checkpoint, path)

    def test_load_runtime_bundle_restores_model_and_training_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            self._save_checkpoint(checkpoint_path)

            bundle = load_runtime_bundle(checkpoint_path, "cpu")

            self.assertFalse(bundle.model.training)
            self.assertEqual(bundle.class_names, (
                "swipe_left", "make_fist", "no_gesture"
            ))
            self.assertEqual(bundle.seq_len, 32)
            self.assertEqual(bundle.quality.min_frames, 20)
            self.assertEqual(bundle.device.type, "cpu")

    def test_load_runtime_bundle_rejects_feature_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            self._save_checkpoint(checkpoint_path, input_size=65)

            with self.assertRaisesRegex(ValueError, "feature_dim"):
                load_runtime_bundle(checkpoint_path, "cpu")

    def test_predict_probabilities_returns_normalized_class_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            self._save_checkpoint(checkpoint_path)
            bundle = load_runtime_bundle(checkpoint_path, "cpu")
            features = np.zeros((32, 66), dtype=np.float32)

            probabilities = predict_probabilities(bundle, features)

            self.assertEqual(probabilities.shape, (3,))
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
            self.assertTrue(np.isfinite(probabilities).all())


class DashboardTests(unittest.TestCase):
    def test_480p_frame_is_padded_without_shape_error(self):
        bundle = RuntimeBundle(
            model=torch.nn.Identity(),
            device=torch.device("cpu"),
            class_names=("swipe_left", "make_fist", "no_gesture"),
            seq_len=32,
            feature_dim=66,
            quality=QualityConfig(),
            best_val_metrics={},
            test_metrics={"accuracy": 0.9, "macro_f1": 0.8},
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        dashboard = draw_dashboard(
            frame,
            bundle,
            np.array([0.7, 0.2, 0.1]),
            "swipe_left",
            "Stable score 70.0%",
            "Right",
            1.0,
            1.5,
            30.0,
            2.0,
            0.65,
        )

        self.assertEqual(dashboard.shape, (620, 1030, 3))


if __name__ == "__main__":
    unittest.main()
