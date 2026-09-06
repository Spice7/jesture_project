import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.train import load_split


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

            x, y = load_split(data_dir, "train")

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
                load_split(data_dir, "train")
