"""Temporary synthetic fixtures test preprocessing contracts, not model quality."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np

if __package__:
    from . import prepare_dataset as p
else:
    import prepare_dataset as p


def fixture(participant="p001", label="swipe_left", sample_id=1, version=2, seed=0):
    times = np.linspace(10, 11, 24)
    points = np.zeros((24, 21, 3), dtype=np.float32)
    points[:, :, 0] = 0.2 + seed * 0.001
    points[:, :, 1] = 0.3
    points[:, 9, 1] += 0.1
    points[:, 5, 2] = -0.02
    if label == "swipe_left":
        points[:, :, 0] += np.linspace(0, 0.2, 24)[:, None]
    if label == "make_fist":
        points[:, 8, 1] += np.linspace(0.1, 0.01, 24)
    result = dict(landmarks=points, timestamps=times, detected=np.ones(24, dtype=bool),
                  participant_id=np.array(participant), label=np.array(label),
                  sample_id=np.array(sample_id, dtype=np.int32), handedness=np.array("Right"),
                  low_motion_warning=np.array(True))
    if version == 2:
        result.update(collection_schema_version=np.array(2), mirrored=np.array(False),
                      detection_rate_ratio=np.array(0.01), duration=np.array(999.0))
    return result


def save_fixture(root, data, nested=""):
    directory = root / nested / data["label"].item()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['participant_id'].item()}_{data['label'].item()}_{data['sample_id'].item():04d}.npz"
    np.savez_compressed(path, **data)
    return path


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "dataset"
        self.source.mkdir()
        self.output = self.root / "result"

    def invoke(self, *extra, expected=0, output=None):
        target = output or self.output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = p.main(["--input-dir", str(self.source), "--output-dir", str(target), *extra])
        self.assertEqual(code, expected)
        if target.is_dir():
            report = json.loads((target / "report.json").read_text(encoding="utf-8"))
            with (target / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            return report, rows
        return None, None

    def scan(self, data):
        path = save_fixture(self.source, data)
        return p.scan_file(path, self.source, {}, 32, p.QualityConfig())

    def add_complete_dataset(self):
        for person_index, person in enumerate(("p001", "p002", "p003")):
            for class_index, label in enumerate(p.LABEL_MAP):
                save_fixture(self.source, fixture(person, label, seed=person_index * 3 + class_index), person)

    def splits(self):
        return ["--train-subjects", "p001", "--val-subjects", "p002", "--test-subjects", "p003"]

    def assert_metadata_only(self, output=None):
        self.assertEqual({f.name for f in (output or self.output).iterdir()}, {
            "manifest.csv", "report.json", "preprocessing_config.json", "label_map.json"})

    def test_v1_v2_optional_fields_and_recomputed_quality(self):
        for version in (1, 2):
            with self.subTest(version=version):
                data = fixture(sample_id=version, version=version)
                if version == 1:
                    data.update(world_landmarks=data["landmarks"].copy(),
                                handedness_per_frame=np.full(24, "Right"),
                                handedness_confidence=np.ones(24, dtype=np.float32))
                row, feature = self.scan(data)
                self.assertEqual(row["status"], "accepted")
                self.assertEqual(row["schema"], version)
                self.assertEqual(row["duration"], 1.0)
                self.assertEqual(row["valid_detection_ratio"], 1.0)
                self.assertEqual(bool(row["assumptions"]), version == 1)
                self.assertEqual(feature.dtype, np.float32)
                self.assertEqual(feature.shape, (32, 66))

    def test_irregular_timestamps_and_feature_formula(self):
        times = np.array([100, 100.1, 100.3, 101.0])
        points = np.zeros((4, 21, 3), dtype=np.float32)
        points[:, :, 0] = (times - times[0])[:, None]
        points[:, 9, 1] = 0.25
        points[:, 8, 2] = -0.125
        x, metrics = p.preprocess_sequence(points, times, np.ones(4, bool), 5,
                                          p.QualityConfig(min_frames=2))
        np.testing.assert_allclose(x[:, 63], [0, 1, 2, 3, 4], atol=1e-6)
        np.testing.assert_allclose(x[:, 64], 0)
        np.testing.assert_allclose(x[:, 0:3], 0)
        np.testing.assert_allclose(x[:, 28], 1)
        np.testing.assert_allclose(x[:, 26], -0.5)
        np.testing.assert_allclose(x[:, 65], 1)
        self.assertEqual(metrics["scale"], 0.25)
        # Index-based resampling would yield x[1,63] = 0.3, not 1.

    def test_short_and_edge_missing_use_time_and_nearest(self):
        times = np.array([0, .1, .2, .4, .9, 1.0])
        points = np.zeros((6, 21, 3), np.float32)
        points[:, :, 0] = times[:, None]
        points[:, 9, 1] = 0.1
        valid = np.array([False, True, False, True, True, False])
        points[~valid] = np.nan
        filled = p.interpolate_landmarks(points, times, valid)
        np.testing.assert_allclose(filled[:, 0, 0], [.1, .1, .2, .4, .9, .9], atol=1e-7)
        x, metrics = p.preprocess_sequence(points, times, valid, 11,
            p.QualityConfig(min_frames=2, min_detection_rate=.5))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(metrics["longest_missing_run"], 1)
        np.testing.assert_allclose(x[-1, 63], 8.0, atol=1e-5)

    def test_all_missing_and_detected_true_nan(self):
        data = fixture()
        data["landmarks"][[0, 5, 23], 2, 1] = np.nan
        row, x = self.scan(data)
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["valid_frames"], 21)
        self.assertTrue(np.isfinite(x).all())
        data["landmarks"][:] = np.nan
        row, x = self.scan(data)
        self.assertIn("all_missing", row["reason"])
        self.assertIsNone(x)

    def test_invalid_timestamps(self):
        for value in (10.0, 9.0, np.nan, np.inf):
            with self.subTest(value=value):
                data = fixture()
                data["timestamps"][1] = value
                row, _ = self.scan(data)
                self.assertIn("invalid_timestamps", row["reason"])

    def test_quality_limits(self):
        cases = [("low_detection_rate", lambda d: d["detected"].__setitem__(slice(0, 6), False)),
                 ("duration_too_short", lambda d: d.__setitem__("timestamps", np.linspace(0, .5, 24))),
                 ("duration_too_long", lambda d: d.__setitem__("timestamps", np.linspace(0, 3, 24)))]
        for reason, modify in cases:
            with self.subTest(reason=reason):
                data = fixture()
                modify(data)
                row, _ = self.scan(data)
                self.assertIn(reason, row["reason"])
        data = fixture()
        for key in ("landmarks", "timestamps", "detected"):
            data[key] = data[key][:19]
        self.assertIn("too_few_frames", self.scan(data)[0]["reason"])

    def test_zero_scale_and_positive_median(self):
        data = fixture()
        data["landmarks"][:, 9] = data["landmarks"][:, 0]
        self.assertIn("degenerate_scale", self.scan(data)[0]["reason"])
        points = data["landmarks"]
        points[0, 9, 1] += .2
        scale = p.sequence_scale(points, np.ones(24, bool))
        self.assertAlmostEqual(scale, .2, places=6)
        with self.assertRaisesRegex(ValueError, "degenerate_scale"):
            p.sequence_scale(points, np.ones(24, bool), epsilon=1)

    def test_stationary_classes_preserved_and_make_fist_scale_fixed(self):
        for label in ("make_fist", "no_gesture"):
            row, x = self.scan(fixture(label=label))
            self.assertEqual(row["status"], "accepted")
            np.testing.assert_allclose(x[:, 63:65], 0)
        data = fixture(label="make_fist")
        data["landmarks"][:, 9, 1] = np.linspace(.4, .35, 24)
        row, x = self.scan(data)
        self.assertEqual(row["status"], "accepted")
        self.assertGreater(x[0, 65], x[-1, 65])
        self.assertGreater(abs(x[0, 25] - x[-1, 25]), .1)

    def test_malformed_arrays_metadata_and_optional_fields(self):
        cases = [
            ("detected", np.ones(24, dtype=np.int32), "invalid_array:detected"),
            ("landmarks", np.zeros((24, 63)), "invalid_array:landmarks"),
            ("landmarks", np.zeros((24, 21, 3), dtype=object), "Object arrays"),
            ("timestamps", np.zeros(23), "invalid_array:timestamps"),
            ("label", np.array(["swipe_left"]), "invalid_scalar"),
            ("participant_id", np.array(""), "invalid_participant_id"),
            ("label", np.array("other"), "unknown_label"),
            ("world_landmarks", np.zeros((2, 21, 3)), "invalid_array:world"),
            ("handedness_per_frame", np.full(23, "Right"), "invalid_array:handedness"),
            ("handedness_confidence", np.ones((24, 1)), "invalid_array:handedness"),
            ("collection_schema_version", np.array(3), "unknown_schema"),
            ("handedness", np.array("Left"), "unexpected_handedness"),
            ("mirrored", np.array(True), "mirrored_sample"),
            ("synthetic", np.array(True), "synthetic_sample"),
            ("augmentation_type", np.array("horizontal_flip"), "synthetic_sample"),
        ]
        # Keep filename fixed so malformed metadata itself is tested.
        path = save_fixture(self.source, fixture())
        for key, value, reason in cases:
            with self.subTest(key=key, reason=reason):
                data = fixture()
                data[key] = value
                np.savez_compressed(path, **data)
                row, _ = p.scan_file(path, self.source, {}, 32, p.QualityConfig())
                self.assertEqual(row["status"], "excluded")
                self.assertIn(reason, row["reason"])
        data = fixture()
        del data["mirrored"]
        self.assertIn("missing_mirrored_in_v2", self.scan(data)[0]["reason"])
        data["landmarks"] = np.full((24, 21, 3), 1e300)
        self.assertIn("unsafe_dtype_conversion", self.scan(data)[0]["reason"])

    def test_nested_underscore_id_and_name_folder_checks(self):
        data = fixture("person_one")
        path = save_fixture(self.source, data, "person_one")
        row, _ = p.scan_file(path, self.source, {}, 32, p.QualityConfig())
        self.assertEqual(row["status"], "accepted")
        wrong = path.with_name("wrong.npz")
        path.rename(wrong)
        self.assertIn("filename_mismatch", p.scan_file(wrong, self.source, {}, 32, p.QualityConfig())[0]["reason"])
        wrong_folder = self.source / path.name
        wrong.rename(wrong_folder)
        self.assertIn("label_folder_mismatch", p.scan_file(wrong_folder, self.source, {}, 32, p.QualityConfig())[0]["reason"])

    def test_audit_is_read_only_and_continues_after_bad_file(self):
        save_fixture(self.source, fixture(version=1))
        bad = self.source / "broken.npz"
        bad.write_bytes(b"not an NPZ")
        index = self.source / "index.csv"
        index.write_text("must not be read as input or changed", encoding="utf-8")
        before = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in self.source.rglob("*") if f.is_file()}
        report, rows = self.invoke("--audit-only")
        after = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in before}
        self.assertEqual(before, after)
        self.assertEqual(report["discovered_files"], 2)
        self.assertEqual(report["counts"], {"excluded": 1, "accepted": 1})
        self.assertTrue(report["split_errors"])
        self.assertTrue(all(row["output_index"] == "" for row in rows))
        self.assert_metadata_only()

    def test_normal_arrays_rows_order_and_dtypes(self):
        self.add_complete_dataset()
        report, rows = self.invoke(*self.splits())
        self.assertTrue(report["training_arrays_written"])
        self.assertEqual([r["source_path"] for r in rows], sorted(r["source_path"] for r in rows))
        self.assertEqual(json.loads((self.output / "label_map.json").read_text()), p.LABEL_MAP)
        for split in p.SPLITS:
            x = np.load(self.output / f"X_{split}.npy", allow_pickle=False)
            y = np.load(self.output / f"y_{split}.npy", allow_pickle=False)
            self.assertEqual(x.shape, (3, 32, 66))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(y.dtype, np.int64)
            self.assertEqual(set(y), {0, 1, 2})
            self.assertTrue(np.isfinite(x).all())
            for row in [r for r in rows if r["split"] == split]:
                index = int(row["output_index"])
                self.assertEqual(y[index], p.LABEL_MAP[row["label"]])
                with np.load(self.source / row["source_path"], allow_pickle=False) as data:
                    expected, _ = p.preprocess_sequence(data["landmarks"], data["timestamps"], data["detected"])
                np.testing.assert_array_equal(x[index], expected)

    def test_missing_classes_normal_stops_with_reports(self):
        save_fixture(self.source, fixture())
        report, rows = self.invoke("--train-subjects", "p001", expected=1)
        self.assertFalse(report["success"])
        self.assertTrue(any("missing_classes" in e for e in report["errors"]))
        self.assertTrue(all(row["output_index"] == "" for row in rows))
        self.assert_metadata_only()

    def test_leakage_repeated_unknown_unassigned_and_empty_splits(self):
        self.add_complete_dataset()
        report, _ = self.invoke("--train-subjects", "p001", "p001", "ghost",
                               "--val-subjects", "p001", expected=1)
        for reason in ("participant_leakage", "duplicate_subject_in_split", "unknown_or_no_accepted_subject",
                       "unassigned_subject", "empty_split"):
            self.assertTrue(any(reason in e for e in report["errors"]), reason)
        self.assert_metadata_only()

    def test_duplicates_different_compression_and_same_sha(self):
        data = fixture()
        path = save_fixture(self.source, data, "a")
        second = save_fixture(self.source, data, "b")
        np.savez(second, **data)
        third = save_fixture(self.source, data, "c")
        third.write_bytes(path.read_bytes())
        report, rows = self.invoke("--audit-only")
        self.assertEqual(report["counts"], {"accepted": 1, "duplicate": 2})
        self.assertNotEqual(rows[0]["sha256"], rows[1]["sha256"])
        self.assertEqual(rows[0]["arrays_sha256"], rows[1]["arrays_sha256"])
        self.assertEqual(rows[0]["sha256"], rows[2]["sha256"])

    def test_conflict_same_identifier_different_arrays(self):
        save_fixture(self.source, fixture(), "a")
        save_fixture(self.source, fixture(seed=1), "b")
        report, rows = self.invoke("--audit-only", expected=1)
        self.assertEqual(report["conflicts"][0]["reason"], "same_identity_different_arrays")
        self.assertTrue(all(row["status"] == "conflict" for row in rows))
        self.assert_metadata_only()

    def test_conflict_same_arrays_different_identity_blocks_normal(self):
        self.add_complete_dataset()
        data = fixture()
        data["participant_id"] = np.array("different_person")
        data["label"] = np.array("make_fist")
        save_fixture(self.source, data)
        report, _ = self.invoke(*self.splits(), expected=1)
        self.assertTrue(any(c["reason"] == "same_arrays_different_identity" for c in report["conflicts"]))
        self.assert_metadata_only()

    def test_aliases_applied_before_dedup_and_split(self):
        self.add_complete_dataset()
        data = fixture(participant="old_user")
        save_fixture(self.source, data)
        mapping = self.root / "aliases.json"
        mapping.write_text('{"old_user":"p001"}', encoding="utf-8")
        report, rows = self.invoke(*self.splits(), "--participant-map", str(mapping))
        self.assertEqual(report["counts"], {"accepted": 9, "duplicate": 1})
        row = next(r for r in rows if r["participant_id"] == "old_user")
        self.assertEqual(row["canonical_participant_id"], "p001")
        report, _ = self.invoke("--train-subjects", "old_user", "--val-subjects", "p001",
                               "--test-subjects", "p003", "--participant-map", str(mapping),
                               output=self.root / "leak", expected=1)
        self.assertIn("participant_leakage:p001", report["errors"])

    def test_alias_conflict_and_invalid_mapping(self):
        save_fixture(self.source, fixture())
        save_fixture(self.source, fixture("old_user", seed=10))
        mapping = self.root / "aliases.json"
        mapping.write_text('{"old_user":"p001"}', encoding="utf-8")
        report, _ = self.invoke("--audit-only", "--participant-map", str(mapping), expected=1)
        self.assertTrue(report["conflicts"])
        for content in ('{"a":"b","b":"c"}', '{"a":"b","b":"a"}',
                        '{"a":"b","a":"c"}', '[]', '{"a":2}'):
            mapping.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                p.load_participant_map(mapping)

    def test_cli_safety_bounds_empty_input_and_defaults(self):
        self.invoke("--audit-only", expected=1)
        self.assert_metadata_only()
        self.invoke("--audit-only", expected=2)  # Existing output, no overwrite.
        inside = self.source / "results"
        self.invoke("--audit-only", output=inside, expected=2)
        self.assertFalse(inside.exists())
        for options in (("--seq-len", "1"), ("--min-frames", "0"),
                        ("--min-detection-rate", "1.1"), ("--max-missing-run", "-1"),
                        ("--min-duration", "0"), ("--min-duration", "3"),
                        ("--scale-epsilon", "0"), ("--max-duration", "nan")):
            self.invoke("--audit-only", *options, output=self.root / "invalid", expected=2)
        defaults = p.make_parser().parse_args(["--output-dir", "unused"])
        self.assertEqual(defaults.input_dir, p.PROJECT_ROOT / "dataset")

    def test_bad_compression_is_a_per_file_error(self):
        path = save_fixture(self.source, fixture())
        with mock.patch.object(p.np, "load", side_effect=zlib.error("invalid compressed stream")):
            row, feature = p.scan_file(path, self.source, {}, 32, p.QualityConfig())
        self.assertEqual(row["status"], "excluded")
        self.assertIn("invalid compressed stream", row["reason"])
        self.assertIsNone(feature)

    def test_output_io_failure_leaves_no_partial_bundle(self):
        self.add_complete_dataset()
        save = np.save
        calls = 0
        def failing_save(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("simulated disk error")
            return save(*args, **kwargs)
        with mock.patch.object(p.np, "save", side_effect=failing_save):
            self.invoke(*self.splits(), expected=2)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".prepare_dataset_*")))


if __name__ == "__main__":
    unittest.main()
