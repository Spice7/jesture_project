"""임시 합성 NPZ의 실제 전처리 결과로 학습 계약을 검사합니다. 실제 수집 자료는 사용하지 않습니다."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from . import train as t
    from . import prepare_dataset as p
    from .test_prepare_dataset import fixture, save_fixture
else:
    import train as t
    import prepare_dataset as p
    from test_prepare_dataset import fixture, save_fixture


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 작은 합성 테스트의 과도한 스레드 비용을 줄이고 종료 후 원래 설정을 복원합니다.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.addCleanup(random.setstate, random_state)
        self.addCleanup(np.random.set_state, numpy_state)
        self.addCleanup(torch.set_rng_state, torch_state)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "prepared"
        self.output = self.root / "run"
        source = self.root / "source"
        for person_index, person in enumerate(("p001", "p002", "p003")):
            for label_index, label in enumerate(t.LABEL_MAP):
                for sample_id in (1, 2):
                    data = fixture(person, label, sample_id=sample_id,
                                   seed=person_index * 10 + label_index * 2 + sample_id)
                    save_fixture(source, data)
        with redirect_stdout(io.StringIO()):
            code = p.main(["--input-dir", str(source), "--output-dir", str(self.data), "--seq-len", "8",
                           "--train-subjects", "p001", "--val-subjects", "p002", "--test-subjects", "p003"])
        self.assertEqual(code, 0)

    def invoke(self, *extra, expected=0, output=None):
        target = output or self.output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = t.main(["--data-dir", str(self.data), "--output-dir", str(target), "--device", "cpu",
                           "--epochs", "2", "--hidden-size", "4", "--batch-size", "4", *extra])
        self.assertEqual(code, expected)
        summary = target / "training_summary.json"
        return t.read_json(summary) if summary.exists() else None

    def mutate_json(self, name, function):
        path = self.data / name
        value = t.read_json(path)
        function(value)
        t.write_json(path, value)

    def rows(self):
        with (self.data / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def write_rows(self, rows):
        with (self.data / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=p.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_contract_and_test_never_loaded(self):
        load = np.load
        opened = []
        def guarded(path, *args, **kwargs):
            opened.append(Path(path).name)
            self.assertNotIn(Path(path).name, ("X_test.npy", "y_test.npy"))
            self.assertIs(kwargs.get("allow_pickle"), False)
            return load(path, *args, **kwargs)
        with mock.patch.object(t.np, "load", side_effect=guarded):
            summary = self.invoke()
        self.assertEqual(set(opened), {"X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"})
        self.assertEqual(summary["status"], "success")
        self.assertFalse(summary["test_evaluated"])

    def test_reject_failed_audit_and_conflicts(self):
        original = t.read_json(self.data / "report.json")
        for changes in ({"mode": "audit-only"}, {"success": False}, {"training_arrays_written": False},
                        {"conflicts": ["conflict"]}, {"errors": ["bad"]}, {"split_errors": ["bad"]}):
            t.write_json(self.data / "report.json", {**original, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                t.load_prepared_data(self.data)

    def test_missing_file_and_failure_summary(self):
        (self.data / "X_val.npy").unlink()
        summary = self.invoke(expected=1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["epochs_completed"], 0)
        self.assertIn("필수 파일 누락", summary["error"])
        self.assertTrue(summary["partial_run"])

    def test_array_validation(self):
        x_path, y_path = self.data / "X_train.npy", self.data / "y_train.npy"
        x, y = np.load(x_path), np.load(y_path)
        bad_x = [x.astype(np.float64), x.reshape(len(x), -1), x[:, :7], x[:, :, :65], x[:0],
                 np.full_like(x, np.nan), np.full_like(x, np.inf), x[:5]]
        for bad in bad_x:
            np.save(x_path, bad)
            with self.subTest(x_shape=bad.shape), self.assertRaises(ValueError):
                t.load_prepared_data(self.data)
        np.save(x_path, x)
        for bad in (y.astype(np.int32), y[:, None], y[:5], np.full_like(y, -1),
                    np.full_like(y, 4), np.zeros_like(y), np.roll(y, 1)):
            np.save(y_path, bad)
            with self.subTest(y=bad.tolist()), self.assertRaises(ValueError):
                t.load_prepared_data(self.data)
        np.save(y_path, y)

    def test_config_label_and_shape_mismatch(self):
        config = t.read_json(self.data / "preprocessing_config.json")
        for changes in ({"seq_len": 9}, {"feature_dim": 63}, {"X_dtype": "float64"},
                        {"label_map": {"swipe_left": 1, "make_fist": 0, "no_gesture": 2}}):
            t.write_json(self.data / "preprocessing_config.json", {**config, **changes})
            with self.assertRaises(ValueError):
                t.load_prepared_data(self.data)
        t.write_json(self.data / "preprocessing_config.json", config)
        t.write_json(self.data / "label_map.json", {"swipe_left": False, "make_fist": 1, "no_gesture": 2})
        with self.assertRaises(ValueError):
            t.load_prepared_data(self.data)

    def test_manifest_indices_labels_and_counts(self):
        original = self.rows()
        for field, value in (("output_index", "99"), ("output_index", "1"), ("label", "unknown"),
                             ("status", "conflict"), ("canonical_participant_id", "")):
            rows = [dict(row) for row in original]
            rows[0][field] = value
            self.write_rows(rows)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                t.load_prepared_data(self.data)
        self.write_rows(original[1:])
        with self.assertRaises(ValueError):
            t.load_prepared_data(self.data)

    def test_leakage_config_assignment_and_missing_classes(self):
        rows = self.rows()
        for row in rows:
            if row["split"] == "test":
                row["canonical_participant_id"] = "p001"
        self.write_rows(rows)
        self.mutate_json("preprocessing_config.json", lambda c: c["canonical_split_subjects"].update(test=["p001"]))
        with self.assertRaisesRegex(ValueError, "참가자 누수"):
            t.load_prepared_data(self.data)
        self.mutate_json("preprocessing_config.json", lambda c: c["canonical_split_subjects"].update(test=["ghost"]))
        with self.assertRaisesRegex(ValueError, "배정 불일치"):
            t.load_prepared_data(self.data)
        for row in rows:
            if row["split"] == "train":
                row["label"] = "swipe_left"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "클래스 누락"):
            t.load_prepared_data(self.data)

    def test_cli_bounds_path_and_device(self):
        for flag, value in (("--epochs", "0"), ("--batch-size", "0"), ("--num-layers", "0"),
                            ("--hidden-size", "0"), ("--dropout", "1"), ("--patience", "0"),
                            ("--seed", "-1"), ("--num-workers", "-1"), ("--learning-rate", "0"),
                            ("--max-grad-norm", "0"), ("--min-delta", "-1"), ("--weight-decay", "nan")):
            self.invoke(flag, value, expected=2)
            self.assertFalse(self.output.exists())
        self.invoke(output=self.data / "inside", expected=2)
        self.invoke(output=t.PROJECT_ROOT / "dataset" / "must_not_create_training_output", expected=2)
        self.output.mkdir()
        self.invoke(expected=2)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(ValueError):
                t.choose_device("cuda")
            self.assertEqual(t.choose_device("auto").type, "cpu")

    def test_smoke_history_checkpoint_and_reproducibility(self):
        first = self.invoke()
        second_path = self.root / "second"
        second = self.invoke(output=second_path)
        self.assertEqual(first, second)
        self.assertEqual(first["epochs_completed"], 2)
        self.assertFalse(first["partial_run"])
        with (self.output / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
            history = list(csv.DictReader(stream))
        checkpoint = torch.load(self.output / "best_model.pt", weights_only=True, map_location="cpu")
        best = min(history, key=lambda row: float(row["val_loss"]))
        self.assertEqual(checkpoint["best_epoch"], int(best["epoch"]))
        self.assertEqual(first["best_epoch"], checkpoint["best_epoch"])
        self.assertEqual(first["best_metrics"], checkpoint["metrics"])
        self.assertEqual(first["best_metrics"]["val_loss"], float(best["val_loss"]))
        other = torch.load(second_path / "best_model.pt", weights_only=True, map_location="cpu")
        for name, value in checkpoint["model_state_dict"].items():
            torch.testing.assert_close(value, other["model_state_dict"][name], rtol=0, atol=0)
        self.assertEqual((self.output / "history.csv").read_bytes(), (second_path / "history.csv").read_bytes())
        settings = t.read_json(self.output / "training_config.json")
        self.assertEqual(settings["actual_device"], "cpu")
        self.assertEqual(len(settings["input_sha256"]), 4)

    def test_epoch_update_validation_and_weighted_metrics(self):
        torch.manual_seed(4)
        x, y = torch.randn(5, 2, 66), torch.tensor([0, 1, 2, 0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
        model = t.LSTMClassifier(hidden_size=4, dropout=0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        t.run_epoch(model, loader, torch.device("cpu"), torch.optim.AdamW(model.parameters()))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.zero_grad(set_to_none=True)
        result = t.run_epoch(model, loader, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)
        with torch.no_grad():
            logits = model(x)
            expected_loss = nn.CrossEntropyLoss()(logits, y).item()
        self.assertAlmostEqual(result["loss"], expected_loss, places=6)
        self.assertEqual(result["accuracy"], (logits.argmax(1) == y).sum().item() / 5)
        # 테스트에서만 lr=0으로 가중치를 고정해 학습 경로의 샘플 가중 평균도 비교합니다.
        fixed_optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0)
        train_metrics = t.run_epoch(model, loader, torch.device("cpu"), fixed_optimizer)
        self.assertAlmostEqual(train_metrics["loss"], expected_loss, places=6)
        self.assertEqual(train_metrics["accuracy"], result["accuracy"])

    def test_early_stopping_boundary_and_best_are_separate(self):
        stopping = t.EarlyStopping(patience=2, min_delta=.125)
        decisions = [stopping.update(v) for v in (1, .9375, .875, .875, .8671875)]
        self.assertEqual(decisions, [(True, False), (True, False), (True, False), (False, False), (True, True)])
        self.assertEqual(stopping.reference, .875)
        zero = t.EarlyStopping(1, 0)
        self.assertEqual(zero.update(1), (True, False))
        self.assertEqual(zero.update(1), (False, True))
        with self.assertRaises(FloatingPointError):
            stopping.update(float("nan"))

    def test_controlled_losses_select_best_not_last(self):
        values = iter([1.0, .9375, .875, .875, .8671875])
        def epoch(model, loader, device, optimizer=None, max_grad_norm=1):
            return {"loss": 1.1 if optimizer is not None else next(values), "accuracy": .5}
        with mock.patch.object(t, "run_epoch", side_effect=epoch):
            summary = self.invoke("--epochs", "10", "--patience", "2", "--min-delta", ".125")
        self.assertTrue(summary["early_stopped"])
        self.assertEqual(summary["best_epoch"], 5)
        self.assertEqual(summary["epochs_completed"], 5)
        values = iter([1.0, .5, .8])
        with mock.patch.object(t, "run_epoch", side_effect=epoch):
            summary = self.invoke("--epochs", "3", output=self.root / "later_worse")
        self.assertEqual(summary["best_epoch"], 2)

    def test_checkpoint_independent_roundtrip(self):
        data = t.load_prepared_data(self.data)
        config = dict(input_size=66, hidden_size=4, num_layers=1, num_classes=4, dropout=.2)
        model = t.LSTMClassifier(**config).eval()
        x = torch.randn(2, 8, 66)
        with torch.no_grad():
            expected = model(x)
        path = self.root / "best.pt"
        t.save_checkpoint(path, model, config, data, dict(epoch=1, train_loss=1, train_accuracy=.5, val_loss=1, val_accuracy=.5))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        restored = t.LSTMClassifier(**checkpoint["model_config"]).eval()
        restored.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            actual = restored(x)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_nan_loss_gradient_failures(self):
        model = t.LSTMClassifier(hidden_size=4)
        x, y = torch.randn(3, 8, 66), torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        optimizer = torch.optim.AdamW(model.parameters())
        handle = next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
        with self.assertRaisesRegex(FloatingPointError, "gradient"):
            t.run_epoch(model, loader, torch.device("cpu"), optimizer)
        handle.remove()
        x[:] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            t.run_epoch(model, loader, torch.device("cpu"))
        for index, error in enumerate((FloatingPointError("loss NaN"), FloatingPointError("gradient Inf"))):
            with mock.patch.object(t, "run_epoch", side_effect=error):
                summary = self.invoke(expected=1, output=self.root / f"failed{index}")
            self.assertEqual(summary["status"], "failed")
            self.assertIn(str(error), summary["error"])

    def test_interruption_preserves_partial_status(self):
        with mock.patch.object(t, "run_epoch", side_effect=[
            {"loss": 1.0, "accuracy": .5}, {"loss": .9, "accuracy": .5}, KeyboardInterrupt()
        ]):
            summary = self.invoke(expected=130)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["epochs_completed"], 1)
        self.assertEqual(summary["best_epoch"], 1)
        self.assertTrue(summary["partial_run"])

    def test_atomic_save_failure_keeps_previous_file(self):
        target = self.root / "result.json"
        t.write_json(target, {"original": True})
        def fail(path):
            path.write_text("partial", encoding="utf-8")
            raise OSError("simulated disk error")
        with self.assertRaises(OSError):
            t.atomic_write(target, fail)
        self.assertEqual(t.read_json(target), {"original": True})
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
