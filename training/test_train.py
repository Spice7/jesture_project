import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import training.train as train_module
from training.train import build_loader, evaluate, load_split, main, train_one_epoch


class LoadSplitTests(unittest.TestCase):
    def test_load_split_reads_valid_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            np.save(
                data_dir / "X_train.npy",
                np.zeros((4, 32, 66), dtype=np.float32),
            )
            np.save(
                data_dir / "y_train.npy",
                np.array([0, 1, 2, 0], dtype=np.int64),
            )

            x, y = load_split(
                data_dir,
                "train",
                seq_len=32,
                feature_dim=66,
                num_classes=3,
            )

            self.assertEqual(tuple(x.shape), (4, 32, 66))
            self.assertEqual(tuple(y.shape), (4,))

    def test_load_split_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            np.save(
                data_dir / "X_train.npy",
                np.zeros((4, 66), dtype=np.float32),
            )
            np.save(
                data_dir / "y_train.npy",
                np.array([0, 1, 2, 0], dtype=np.int64),
            )

            with self.assertRaises(ValueError):
                load_split(
                    data_dir,
                    "train",
                    seq_len=32,
                    feature_dim=66,
                    num_classes=3,
                )


class WeightedLossAggregationTests(unittest.TestCase):
    @staticmethod
    def _data() -> tuple[torch.Tensor, torch.Tensor, nn.CrossEntropyLoss]:
        logits = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0, 10.0])
        )
        return logits, labels, criterion

    def test_evaluate_matches_full_weighted_cross_entropy(self):
        logits, labels, criterion = self._data()
        loader = build_loader(
            logits, labels, batch_size=2, shuffle=False, seed=42,
            pin_memory=False,
        )
        expected = float(criterion(logits, labels))

        result = evaluate(
            nn.Identity(), loader, criterion, torch.device("cpu"), 3
        )

        self.assertAlmostEqual(result["loss"], expected, places=6)

    def test_train_epoch_matches_full_weighted_cross_entropy(self):
        logits, labels, criterion = self._data()
        loader = build_loader(
            logits, labels, batch_size=2, shuffle=False, seed=42,
            pin_memory=False,
        )

        class ScaledLogits(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return values * self.scale

        model = ScaledLogits()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        expected = float(criterion(logits, labels))

        result = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            torch.device("cpu"),
            max_grad_norm=1.0,
        )

        self.assertAlmostEqual(result, expected, places=6)


class CriterionConstructionTests(unittest.TestCase):
    def test_label_smoothing_is_used_only_for_training_loss(self):
        self.assertTrue(
            hasattr(train_module, "build_criteria"),
            "training.train.build_criteria is required",
        )
        weights = torch.tensor([1.0, 2.0, 3.0])

        train_criterion, evaluation_criterion = train_module.build_criteria(
            weights,
            label_smoothing=0.05,
        )

        self.assertIs(train_criterion.weight, weights)
        self.assertIs(evaluation_criterion.weight, weights)
        self.assertEqual(train_criterion.label_smoothing, 0.05)
        self.assertEqual(evaluation_criterion.label_smoothing, 0.0)

    def test_invalid_label_smoothing_is_rejected(self):
        self.assertTrue(
            hasattr(train_module, "build_criteria"),
            "training.train.build_criteria is required",
        )

        for value in (-0.01, 1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    train_module.build_criteria(None, value)


class MainSideEffectTests(unittest.TestCase):
    def test_invalid_epoch_count_does_not_leave_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "result"
            arguments = [
                "training.train",
                "--data-dir", str(root / "data"),
                "--output-dir", str(output_dir),
                "--epochs", "0",
            ]

            with patch("sys.argv", arguments):
                with self.assertRaisesRegex(ValueError, "epochs"):
                    main()

            self.assertFalse(output_dir.exists())
