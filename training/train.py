"""
전처리된 NPY 파일을 받아 LSTM을 학습하고 가장 좋은 validation 모델을 저장한다.

평가 기준 : validation macro F1이 가장 높은 epoch의 모델 저장
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_split(data_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_
    data_dir에서 split에 해당하는 NPY 파일을 불러와 torch.Tensor로 변환합니다.
    Args:
        data_dir (Path): _description_
        split (str): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        tuple[torch.Tensor, torch.Tensor]: _description_
    """
    x_path = data_dir / f"x_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"

    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)

    if x.dtype != np.float32:
        raise ValueError(
            f"{x_path}의 dtype은 float32여야 합니다. 현재 dtype: {x.dtype}"
        )
    if y.dtype != np.int64:
        raise ValueError(f"{y_path}의 dtype은 int64여야 합니다. 현재 dtype: {y.dtype}")

    if x.ndim != 3:
        raise ValueError(
            f"{x_path}의 shape은 (batch_size, seq_len, input_size)의 3차원이어야 합니다. 현재 shape: {x.shape}"
        )
    if y.ndim != 1:
        raise ValueError(
            f"{y_path}의 shape은 (batch_size,)의 1차원이어야 합니다. 현재 shape: {y.shape}"
        )

    if len(x) != len(y):
        raise ValueError(f"{split}: x와 y 샘플 수가 다릅니다.")

    if not np.isfinite(x).all():
        raise ValueError(f"{split}: X에 NaN 또는 Infinity가 있습니다.")

    return torch.from_numpy(x), torch.from_numpy(y)


def load_metadata(data_dir: Path) -> tuple[dict[str, int], dict]:
    import json

    with (data_dir / "label_map.json").open(encoding="utf-8") as f:
        label_map = json.load(f)

    with (data_dir / "preprocessing_config.json").open(encoding="utf-8") as f:
        preprocessing_config = json.load(f)
    return label_map, preprocessing_config


def build_loader(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


from torch import nn


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_samples = len(y_batch)

    return total_loss / total_samples


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []

    for class_id in range(num_classes):
        true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
        false_positive = np.sum((y_true != class_id) & (y_pred == class_id))
        false_negative = np.sum((y_true == class_id) & (y_pred != class_id))

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)

        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        scores.append(f1)

    return float(np.mean(scores))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

        all_true.append(y_batch.cpu().numpy())
        all_pred.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return {
        "loss": total_loss / total_samples,
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }
