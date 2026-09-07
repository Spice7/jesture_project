"""임시 합성 NPZ/체크포인트만 사용합니다. 실제 validation/test는 절대 열지 않습니다."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

if __package__:
    from . import evaluate as e, prepare_dataset as p, train as t
    from .test_prepare_dataset import fixture, save_fixture
else:
    import evaluate as e
    import prepare_dataset as p
    import train as t
    from test_prepare_dataset import fixture, save_fixture


class EvaluationTests(unittest.TestCase):
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
        torch.manual_seed(7)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.run_dir, self.output = (self.root / name for name in ("prepared", "run", "eval"))
        source = self.root / "source"
        for person_index, person in enumerate(("p001", "p002", "p003")):
            for label_index, label in enumerate(e.LABEL_MAP):
                for sample_id in (1, 2):
                    save_fixture(source, fixture(person, label, sample_id=sample_id,
                                 seed=person_index * 10 + label_index * 2 + sample_id))
        with redirect_stdout(io.StringIO()):
            code = p.main(["--input-dir", str(source), "--output-dir", str(self.data), "--seq-len", "8",
                           "--train-subjects", "p001", "--val-subjects", "p002", "--test-subjects", "p003"])
        self.assertEqual(code, 0)
        # 실제 train.py 저장 함수를 사용하되 학습하지 않습니다. 지표도 테스트 전용 가상 값입니다.
        self.run_dir.mkdir()
        data = t.load_prepared_data(self.data)
        self.model_config = dict(input_size=66, hidden_size=5, num_layers=2, num_classes=4, dropout=.3)
        model = e.LSTMClassifier(**self.model_config)
        metrics = dict(epoch=2, train_loss=1.2, train_accuracy=.5, val_loss=1.1, val_accuracy=.5)
        t.save_checkpoint(self.run_dir / "best_model.pt", model, self.model_config, data, metrics)
        settings = {key: data[key] for key in ("label_map", "preprocessing_config", "input_sha256", "split_counts")}
        settings.update(model_config=self.model_config, data_dir="old/location")
        t.write_json(self.run_dir / "training_config.json", settings)
        t.write_json(self.run_dir / "training_summary.json",
                     dict(status="success", partial_run=False, best_epoch=2, best_metrics=metrics))

    def invoke(self, *extra, expected=0, output=None, split="val"):
        target = output or self.output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = e.main(["--run-dir", str(self.run_dir), "--data-dir", str(self.data),
                           "--output-dir", str(target), "--split", split, "--device", "cpu",
                           "--batch-size", "4", *extra])
        self.assertEqual(code, expected)
        path = target / "evaluation_metrics.json"
        return t.read_json(path) if path.is_file() else None

    def metadata(self):
        return e.load_metadata(self.run_dir, self.data)

    def read_rows(self, path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def change_json(self, path, function):
        data = t.read_json(path)
        function(data)
        t.write_json(path, data)

    def snapshot(self):
        return {str(path.relative_to(self.root)): e.sha256(path)
                for directory in (self.data, self.run_dir) for path in directory.rglob("*") if path.is_file()}

    def test_smoke_only_selected_arrays_loaded_and_hashed_inputs_unchanged(self):
        before = self.snapshot()
        original_load, original_hash, original_torch_load = np.load, e.sha256, torch.load
        for split in ("val", "test"):
            loads, hashes = [], []
            def guarded_load(path, *args, **kwargs):
                loads.append(Path(path).name)
                self.assertIn(Path(path).name, (f"X_{split}.npy", f"y_{split}.npy"))
                self.assertIs(kwargs.get("allow_pickle"), False)
                return original_load(path, *args, **kwargs)
            def guarded_hash(path):
                if path.suffix == ".npy":
                    hashes.append(path.name)
                    self.assertIn(path.name, (f"X_{split}.npy", f"y_{split}.npy"))
                return original_hash(path)
            def guarded_checkpoint(path, *args, **kwargs):
                self.assertIs(kwargs.get("weights_only"), True)
                self.assertEqual(kwargs.get("map_location"), "cpu")
                return original_torch_load(path, *args, **kwargs)
            target = self.root / f"eval_{split}"
            with mock.patch.object(e.np, "load", side_effect=guarded_load), \
                 mock.patch.object(e, "sha256", side_effect=guarded_hash), \
                 mock.patch.object(e.torch, "load", side_effect=guarded_checkpoint) as checkpoint_load:
                status = self.invoke(split=split, output=target)
            self.assertEqual(checkpoint_load.call_count, 1)
            self.assertEqual(loads, [f"X_{split}.npy", f"y_{split}.npy"])
            self.assertCountEqual(hashes, loads)
            self.assertTrue(status["completed"])
            self.assertEqual(status["status"], "success")
            self.assertEqual(status["sample_count"], 8)
            settings = t.read_json(target / "evaluation_config.json")
            self.assertEqual(settings["model_config"], self.model_config)
            self.assertEqual(settings["actual_device"], "cpu")
            self.assertEqual(set(settings["array_sha256"]), set(loads))
            self.assertEqual(set(settings["input_sha256"]), set(e.METADATA_FILES))
            self.assertEqual(settings["checkpoint_sha256"], original_hash(self.run_dir / "best_model.pt"))
            self.assertEqual(len(list(target.iterdir())), 6)
            report = (target / "classification_report.txt").read_text(encoding="utf-8")
            self.assertIn(f"split: {split}", report)
            for label in e.LABEL_ORDER:
                line = next(line for line in report.splitlines() if line.strip().startswith(label))
                fields = line.split()
                values = status["per_class"][label]
                self.assertEqual(fields[1:4], [f"{values[k]:.4f}" for k in ("precision", "recall", "f1")])
                self.assertEqual(int(fields[4]), values["support"])
        self.assertEqual(before, self.snapshot())

    def test_restore_saved_structure_strict_state_and_no_grad(self):
        metadata = self.metadata()
        model = e.restore_model(metadata, torch.device("cpu"))
        self.assertEqual(model.hidden_size, 5)
        self.assertEqual(model.num_layers, 2)
        x, y, _ = e.load_split(self.data, "val", metadata)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        seen = []
        handle = model.register_forward_hook(lambda *_: seen.append(torch.is_grad_enabled()))
        try:
            e.predict_batches(model, x, y, torch.device("cpu"), batch_size=4)
        finally:
            handle.remove()
        self.assertEqual(seen, [False, False])
        self.assertFalse(model.training)
        self.assertTrue(all(v.grad is None for v in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        metadata["checkpoint"]["model_state_dict"].pop("classifier.bias")
        with self.assertRaises(RuntimeError):
            e.restore_model(metadata, torch.device("cpu"))

    def test_known_metrics_and_zero_division(self):
        y = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        predicted = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
        overall, per_class, matrix = e.classification_metrics(y, predicted, .7)
        np.testing.assert_array_equal(matrix, [[1, 2, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        expected = [(1, 1/3, 1/2, 3), (1/3, 1/2, 2/5, 2), (1/2, 1, 2/3, 1)]
        for name, values in zip(e.LABEL_ORDER, expected):
            for key, value in zip(("precision", "recall", "f1", "support"), values):
                self.assertAlmostEqual(per_class[name][key], value)
        self.assertEqual(overall["accuracy"], .5)
        self.assertAlmostEqual(overall["macro_precision"], (1+1/3+1/2)/4)
        self.assertAlmostEqual(overall["macro_recall"], (1/3+1/2+1)/4)
        self.assertAlmostEqual(overall["macro_f1"], (1/2+2/5+2/3)/4)
        self.assertAlmostEqual(overall["weighted_f1"], (3/2+4/5+2/3)/6)
        _, classes, _ = e.classification_metrics(y, np.zeros_like(y), .9)
        self.assertEqual(classes["make_fist"]["precision"], 0)
        self.assertEqual(classes["make_fist"]["f1"], 0)
        # 독립 지표 함수의 support=0도 안전하게 처리하며 macro에서 클래스를 빼지 않습니다.
        overall, classes, _ = e.classification_metrics(np.array([0]), np.array([0]), .1)
        self.assertEqual(classes["no_gesture"]["recall"], 0)
        self.assertAlmostEqual(overall["macro_f1"], 1/4)

    def test_weighted_loss_last_batch_and_probabilities(self):
        class Scores(nn.Module):
            def forward(self, x):
                return x[:, 0, :3]
        x = np.zeros((5, 2, 66), dtype=np.float32)
        x[:, 0, :3] = [[4, 0, 0], [0, 4, 0], [0, 0, 4], [4, 0, 0], [10, 0, 0]]
        y = np.array([0, 1, 2, 0, 2], dtype=np.int64)
        loss, predicted, probs = e.predict_batches(Scores(), x, y, torch.device("cpu"), 4, num_classes=3)
        logits = torch.from_numpy(x[:, 0, :3])
        self.assertAlmostEqual(loss, nn.CrossEntropyLoss()(logits, torch.from_numpy(y)).item(), places=6)
        np.testing.assert_array_equal(predicted, logits.argmax(1).numpy())
        np.testing.assert_allclose(probs, logits.softmax(1).numpy(), rtol=1e-6)
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)

    def test_classification_report_averages_and_zero_division(self):
        y = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        for predicted in (np.array([0, 1, 1, 1, 2, 2]), np.zeros_like(y)):
            overall, classes, _ = e.classification_metrics(y, predicted, .7)
            path = self.root / "classification_report.txt"
            e.save_classification_report(path, "val", overall, classes)
            lines = path.read_text(encoding="utf-8").splitlines()
            macro = next(line for line in lines if line.strip().startswith("macro avg")).split()
            weighted = next(line for line in lines if line.strip().startswith("weighted avg")).split()
            self.assertEqual(macro[2:5], [f"{overall[k]:.4f}" for k in
                                         ("macro_precision", "macro_recall", "macro_f1")])
            for i, key in enumerate(("precision", "recall", "f1")):
                expected = sum(v[key] * v["support"] for v in classes.values()) / len(y)
                self.assertEqual(weighted[i + 2], f"{expected:.4f}")
            self.assertEqual(int(weighted[-1]), len(y))
            self.assertIn("CrossEntropyLoss: 0.700000", lines)

    def test_classification_report_save_failure(self):
        with mock.patch.object(e, "save_classification_report", side_effect=OSError("report save failed")):
            status = self.invoke(expected=1)
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["completed"])
        self.assertIn("report save failed", status["error"])

    def test_prediction_manifest_mapping_and_misclassified(self):
        metadata = self.metadata()
        x, y, _ = e.load_split(self.data, "val", metadata)
        predicted = y.copy()
        predicted[1] = (predicted[1] + 1) % 3
        probs = np.eye(4, dtype=np.float32)[predicted]
        # CSV 행 순서를 바꿔도 검증된 output_index 기준으로 연결되어야 합니다.
        rows = self.read_rows(self.data / "manifest.csv")
        e.write_csv(self.data / "manifest.csv", p.FIELDS, reversed(rows))
        self.change_json(self.run_dir / "training_config.json",
                         lambda v: v["input_sha256"].update({"manifest.csv": e.sha256(self.data / "manifest.csv")}))
        with mock.patch.object(e, "predict_batches", return_value=(.5, predicted, probs)):
            self.invoke()
        predictions = self.read_rows(self.output / "predictions.csv")
        wrong = self.read_rows(self.output / "misclassified.csv")
        self.assertEqual(wrong, [r for r in predictions if r["correct"] == "False"])
        self.assertEqual(len(wrong), 1)
        self.assertEqual(len(predictions), len(x))
        for index, source in metadata["grouped"]["val"]:
            row = predictions[index]
            for key in ("source_path", "participant_id", "canonical_participant_id"):
                self.assertEqual(row[key], source[key])
            self.assertEqual(int(row["output_index"]), index)
            self.assertEqual(int(row["true_label_id"]), y[index])
        matrix = self.read_rows(self.output / "confusion_matrix.csv")
        self.assertEqual([r["true_label / predicted_label"] for r in matrix], list(e.LABEL_ORDER))
        self.assertEqual(sum(int(r[label]) for r in matrix for label in e.LABEL_ORDER), len(y))

    def test_all_correct_header_only(self):
        _, y, _ = e.load_split(self.data, "val", self.metadata())
        with mock.patch.object(e, "predict_batches", return_value=(.01, y, np.eye(4)[y])):
            status = self.invoke()
        self.assertEqual(status["overall"]["accuracy"], 1)
        self.assertEqual(self.read_rows(self.output / "misclassified.csv"), [])
        self.assertEqual(len((self.output / "misclassified.csv").read_text(encoding="utf-8-sig").splitlines()), 1)

    def test_invalid_arrays(self):
        metadata = self.metadata()
        x_path, y_path = self.data / "X_val.npy", self.data / "y_val.npy"
        x, y = np.load(x_path), np.load(y_path)
        for bad in (x.astype(np.float64), x.astype(np.int64), x.reshape(len(x), -1), x[:, :7],
                    x[:, :, :65], x[:0], np.full_like(x, np.nan), np.full_like(x, np.inf), x[:5]):
            np.save(x_path, bad)
            with self.subTest(x=bad.shape), self.assertRaises(ValueError):
                e.load_split(self.data, "val", metadata)
        np.save(x_path, x)
        for bad in (y.astype(np.int32), y.astype(np.float32), y[:, None], y[:0], y[:5],
                    np.full_like(y, -1), np.full_like(y, 4), np.zeros_like(y), np.roll(y, 1)):
            np.save(y_path, bad)
            with self.subTest(y=bad.tolist()), self.assertRaises(ValueError):
                e.load_split(self.data, "val", metadata)

    def test_invalid_manifest(self):
        metadata = self.metadata()
        config, report = metadata["preprocessing_config"], t.read_json(self.data / "report.json")
        original = self.read_rows(self.data / "manifest.csv")
        for field, value in (("output_index", "99"), ("output_index", "1"), ("output_index", "oops"),
                             ("canonical_participant_id", "ghost"), ("label", "unknown"),
                             ("participant_id", ""), ("source_path", ""), ("status", "conflict"),
                             ("status", "excluded"), ("split", "unknown")):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            e.write_csv(self.data / "manifest.csv", p.FIELDS, rows)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                e.validate_manifest(self.data, config, report)
        e.write_csv(self.data / "manifest.csv", p.FIELDS, original[1:])
        with self.assertRaises(ValueError):
            e.validate_manifest(self.data, config, report)
        rows = copy.deepcopy(original)
        for row in rows:
            if row["split"] == "test":
                row["canonical_participant_id"] = "p001"
        e.write_csv(self.data / "manifest.csv", p.FIELDS, rows)
        config["canonical_split_subjects"]["test"] = ["p001"]
        with self.assertRaisesRegex(ValueError, "참가자 누수"):
            e.validate_manifest(self.data, config, report)

    def test_checkpoint_and_run_mismatches(self):
        path = self.run_dir / "best_model.pt"
        original = torch.load(path, map_location="cpu", weights_only=True)
        mutations = [lambda v: v["model_config"].update(hidden_size=9),
                     lambda v: v.update(best_epoch=99), lambda v: v["metrics"].update(val_loss=.1),
                     lambda v: v["label_map"].update(swipe_left=1),
                     lambda v: v["preprocessing_config"].update(seq_len=9)]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value)
            torch.save(value, path)
            with self.assertRaises(ValueError):
                self.metadata()
        torch.save(original, path)
        settings_path = self.run_dir / "training_config.json"
        settings = t.read_json(settings_path)
        for key, value in (("model_config", {}), ("preprocessing_config", {}),
                           ("label_map", {}), ("input_sha256", {}), ("split_counts", {})):
            t.write_json(settings_path, {**settings, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.metadata()

    def test_metadata_hashes_and_contracts(self):
        for name in e.METADATA_FILES:
            path = self.data / name
            original = path.read_bytes()
            # 내용 의미가 같아도 바이트 지문이 다르면 거부합니다.
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "SHA-256"):
                self.metadata()
            path.write_bytes(original)
        path = self.data / "report.json"
        report = t.read_json(path)
        for change in ({"mode": "audit-only"}, {"success": False}, {"training_arrays_written": False},
                       {"conflicts": ["bad"]}, {"errors": ["bad"]}, {"split_errors": ["bad"]}):
            t.write_json(path, {**report, **change})
            with self.assertRaises(ValueError):
                self.metadata()
        t.write_json(path, report)
        self.change_json(self.data / "label_map.json", lambda v: v.update(swipe_left=False))
        with self.assertRaises(ValueError):
            self.metadata()

    def test_unsuccessful_training_and_missing_files(self):
        path = self.run_dir / "training_summary.json"
        summary = t.read_json(path)
        for change in ({"status": "failed"}, {"status": "interrupted"}, {"status": "running"},
                       {"partial_run": True}, {"best_epoch": 99}, {"best_metrics": {}}):
            t.write_json(path, {**summary, **change})
            with self.assertRaises(ValueError):
                self.metadata()
        t.write_json(path, summary)
        (self.data / "X_val.npy").unlink()
        status = self.invoke(expected=1)
        self.assertFalse(status["completed"])
        self.assertEqual(status["status"], "failed")

    def test_unselected_arrays_need_not_exist(self):
        # 선택하지 않은 파일은 열기/해시뿐 아니라 존재 자체에도 의존하지 않습니다.
        for split in ("train", "test"):
            for prefix in ("X", "y"):
                (self.data / f"{prefix}_{split}.npy").unlink()
        self.invoke()

    def test_invalid_consistent_configs_and_nonfinite_state(self):
        original = self.metadata()
        for changes in ({"hidden_size": 0}, {"num_layers": True}, {"dropout": 1}):
            metadata = copy.deepcopy(original)
            metadata["model_config"].update(changes)
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                e.restore_model(metadata, torch.device("cpu"))
        for change in (torch.full((3,), float("nan")), torch.zeros(3, dtype=torch.float64)):
            metadata = copy.deepcopy(original)
            metadata["checkpoint"]["model_state_dict"]["classifier.bias"] = change
            with self.assertRaises(ValueError):
                e.restore_model(metadata, torch.device("cpu"))
        path = self.run_dir / "best_model.pt"
        checkpoint = copy.deepcopy(original["checkpoint"])
        checkpoint["preprocessing_config"]["label_map"]["swipe_left"] = False
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "preprocessing_config"):
            self.metadata()
        torch.save(original["checkpoint"], path)
        # 3개 설정의 내용이 모두 같더라도 전처리 입력 계약을 위반하면 거부합니다.
        initial_settings = t.read_json(self.run_dir / "training_config.json")
        for changes in ({"seq_len": 0}, {"feature_dim": 65}, {"X_dtype": "float64"}):
            config = {**original["preprocessing_config"], **changes}
            t.write_json(self.data / "preprocessing_config.json", config)
            t.write_json(self.run_dir / "training_config.json", {**initial_settings, "preprocessing_config": config})
            checkpoint = {**original["checkpoint"], "preprocessing_config": config}
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "전처리 seq_len/feature_dim/dtype"):
                self.metadata()

    def test_cli_paths_and_device(self):
        for flag, value in (("--batch-size", "0"), ("--batch-size", "-1"), ("--num-workers", "-1")):
            self.invoke(flag, value, expected=2)
            self.assertFalse(self.output.exists())
        self.invoke(output=self.data / "inside", expected=2)
        self.invoke(output=e.PROJECT_ROOT / "dataset" / "must_not_create_evaluation_output", expected=2)
        self.output.mkdir()
        self.invoke(expected=2)
        self.assertEqual(list(self.output.iterdir()), [])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            e.make_parser().parse_args(["--run-dir", "x", "--data-dir", "y", "--output-dir", "z"])
        self.assertEqual(caught.exception.code, 2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            e.make_parser().parse_args(["--run-dir", "x", "--data-dir", "y", "--output-dir", "z", "--split", "train"])
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(ValueError):
                e.choose_device("cuda")
            self.assertEqual(e.choose_device("auto").type, "cpu")
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(e.choose_device("auto").type, "cuda")

    def test_nan_logits_loss_probability_rejected(self):
        metadata = self.metadata()
        x, y, _ = e.load_split(self.data, "val", metadata)
        model = e.restore_model(metadata, torch.device("cpu"))
        for value in (float("nan"), float("inf")):
            with mock.patch.object(model, "forward", return_value=torch.full((6, 3), value)):
                with self.assertRaisesRegex(FloatingPointError, "logits"):
                    e.predict_batches(model, x, y, torch.device("cpu"))
        with mock.patch.object(nn.CrossEntropyLoss, "forward", return_value=torch.tensor(float("inf"))):
            with self.assertRaisesRegex(FloatingPointError, "loss"):
                e.predict_batches(model, x, y, torch.device("cpu"))
        with mock.patch.object(torch.Tensor, "softmax", return_value=torch.full((6, 3), float("nan"))):
            with self.assertRaisesRegex(FloatingPointError, "확률"):
                e.predict_batches(model, x, y, torch.device("cpu"))

    def test_failure_interrupt_and_partial_save_status(self):
        for index, error in enumerate((RuntimeError("synthetic failure"), KeyboardInterrupt())):
            target = self.root / f"failure{index}"
            with mock.patch.object(e, "predict_batches", side_effect=error):
                status = self.invoke(expected=130 if index else 1, output=target)
            self.assertEqual(status["status"], "interrupted" if index else "failed")
            self.assertFalse(status["completed"])
            self.assertIsNone(status["overall"])
            self.assertIsNone(status["sample_count"])
            self.assertFalse((target / "predictions.csv").exists())
        real_write = e.write_csv
        def fail_second(path, fields, rows):
            if path.name == "misclassified.csv":
                raise OSError("synthetic disk error")
            real_write(path, fields, rows)
        with mock.patch.object(e, "write_csv", side_effect=fail_second):
            status = self.invoke(expected=1)
        self.assertTrue((self.output / "predictions.csv").exists())
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["completed"])
        self.assertIsNone(status["overall"])


if __name__ == "__main__":
    unittest.main()
