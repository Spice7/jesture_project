"""전처리된 제스처 시퀀스로 LSTM을 학습하고 최적 모델을 평가합니다."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from training.models import LSTMClassifier


def set_seed(seed: int) -> None:
    """가능한 범위에서 실험 재현성을 확보합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)


def load_metadata(data_dir: Path) -> tuple[dict[str, int], dict]:
    label_path = data_dir / "label_map.json"
    config_path = data_dir / "preprocessing_config.json"

    if not label_path.is_file():
        raise FileNotFoundError(f"label_map.json이 없습니다: {label_path}")

    if not config_path.is_file():
        raise FileNotFoundError(
            f"preprocessing_config.json이 없습니다: {config_path}"
        )

    with label_path.open(encoding="utf-8") as stream:
        label_map = json.load(stream)

    with config_path.open(encoding="utf-8") as stream:
        preprocessing_config = json.load(stream)

    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("label_map은 비어 있지 않은 dictionary여야 합니다.")

    class_ids = sorted(label_map.values())
    expected_ids = list(range(len(label_map)))

    if class_ids != expected_ids:
        raise ValueError(
            f"label id는 0부터 연속이어야 합니다: {label_map}"
        )

    if preprocessing_config.get("label_map") != label_map:
        raise ValueError(
            "label_map.json과 preprocessing_config.json의 label map이 다릅니다."
        )

    seq_len = preprocessing_config.get("seq_len")
    feature_dim = preprocessing_config.get("feature_dim")

    if not isinstance(seq_len, int) or seq_len < 2:
        raise ValueError(f"잘못된 seq_len: {seq_len}")

    if not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError(f"잘못된 feature_dim: {feature_dim}")

    return label_map, preprocessing_config


def load_split(
    data_dir: Path,
    split: str,
    seq_len: int,
    feature_dim: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """전처리 결과 한 split을 읽고 학습 계약을 검증합니다."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"알 수 없는 split: {split}")

    # prepare_dataset.py가 생성하는 파일명과 대소문자를 맞춥니다.
    x_path = data_dir / f"X_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"

    if not x_path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {x_path}")

    if not y_path.is_file():
        raise FileNotFoundError(f"정답 파일이 없습니다: {y_path}")

    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)

    if x.dtype != np.float32:
        raise ValueError(f"{x_path} dtype은 float32여야 합니다: {x.dtype}")

    if y.dtype != np.int64:
        raise ValueError(f"{y_path} dtype은 int64여야 합니다: {y.dtype}")

    expected_x_shape = (len(y), seq_len, feature_dim)

    if x.shape != expected_x_shape:
        raise ValueError(
            f"{split} X shape 오류: expected={expected_x_shape}, actual={x.shape}"
        )

    if y.ndim != 1:
        raise ValueError(f"{split} y는 1차원이어야 합니다: {y.shape}")

    if len(y) == 0:
        raise ValueError(f"{split} 데이터가 비어 있습니다.")

    if not np.isfinite(x).all():
        raise ValueError(f"{split} X에 NaN 또는 Infinity가 있습니다.")

    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError(
            f"{split} y에 범위를 벗어난 label이 있습니다: "
            f"min={y.min()}, max={y.max()}"
        )

    return torch.from_numpy(x), torch.from_numpy(y)


def build_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    workers: int = 0,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def compute_class_weights(
    y_train: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """클래스 수가 적은 no_gesture 등이 무시되지 않도록 가중치를 계산합니다."""
    counts = torch.bincount(y_train, minlength=num_classes).float()

    if torch.any(counts == 0):
        raise ValueError(
            f"train split에 없는 클래스가 있습니다: counts={counts.tolist()}"
        )

    weights = len(y_train) / (num_classes * counts)
    return weights


def build_criteria(
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss]:
    """학습용 smoothing loss와 비교 가능한 평가용 원래 loss를 만듭니다."""
    if not np.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing은 0 이상 1 미만이어야 합니다.")

    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    evaluation_criterion = nn.CrossEntropyLoss(weight=class_weights)
    return train_criterion, evaluation_criterion


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> float:
    scores: list[float] = []

    for class_id in range(num_classes):
        true_positive = np.sum(
            (y_true == class_id) & (y_pred == class_id)
        )
        false_positive = np.sum(
            (y_true != class_id) & (y_pred == class_id)
        )
        false_negative = np.sum(
            (y_true == class_id) & (y_pred != class_id)
        )

        precision = true_positive / max(
            true_positive + false_positive, 1
        )
        recall = true_positive / max(
            true_positive + false_negative, 1
        )

        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(
                float(2 * precision * recall / (precision + recall))
            )

    return float(np.mean(scores))


def mean_loss_weight(
    criterion: nn.Module,
    targets: torch.Tensor,
) -> float:
    """배치 평균 loss를 전체 데이터 평균으로 합칠 때 사용할 분모를 반환합니다.

    가중 CrossEntropyLoss의 ``mean`` 분모는 샘플 수가 아니라 정답 클래스
    가중치의 합입니다. 이 값을 사용해야 평가 loss가 배치 크기에 의존하지 않습니다.
    """
    if (
        isinstance(criterion, nn.CrossEntropyLoss)
        and criterion.reduction == "mean"
    ):
        valid = targets != criterion.ignore_index
        if criterion.weight is None:
            return float(valid.sum().item())
        return float(criterion.weight[targets[valid]].sum().item())

    return float(targets.size(0))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_weight = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        # LSTM에서 매우 큰 gradient가 발생하는 것을 제한합니다.
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        batch_weight = mean_loss_weight(criterion, y_batch)
        total_loss += loss.item() * batch_weight
        total_weight += batch_weight

    if total_weight == 0:
        raise ValueError("train loader가 비어 있습니다.")

    return total_loss / total_weight


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
    total_weight = 0.0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        predictions = logits.argmax(dim=1)

        batch_weight = mean_loss_weight(criterion, y_batch)
        total_loss += loss.item() * batch_weight
        total_weight += batch_weight

        all_true.append(y_batch.cpu().numpy())
        all_pred.append(predictions.cpu().numpy())

    if total_weight == 0:
        raise ValueError("평가 loader가 비어 있습니다.")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return {
        "loss": total_loss / total_weight,
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notebook 분석용 정답, 예측, softmax 값을 반환합니다."""
    model.eval()

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)

        logits = model(x_batch)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)

        all_true.append(y_batch.numpy())
        all_pred.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    if not all_true:
        raise ValueError("예측할 데이터가 없습니다.")

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_probabilities),
    )


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="학습 loss에만 적용할 label smoothing 비율 (0 이상 1 미만)",
    )

    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
    )

    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용 가능한 CUDA 장치가 없습니다.")

    return torch.device(requested)


def main() -> int:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"기존 학습 결과를 덮어쓰지 않습니다: {output_dir}"
        )

    if args.epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")

    if args.early_stopping_patience < 1:
        raise ValueError("early stopping patience는 1 이상이어야 합니다.")

    set_seed(args.seed)
    device = select_device(args.device)

    label_map, preprocessing_config = load_metadata(data_dir)

    seq_len = int(preprocessing_config["seq_len"])
    feature_dim = int(preprocessing_config["feature_dim"])
    num_classes = len(label_map)

    x_train, y_train = load_split(
        data_dir, "train", seq_len, feature_dim, num_classes
    )
    x_val, y_val = load_split(
        data_dir, "val", seq_len, feature_dim, num_classes
    )
    x_test, y_test = load_split(
        data_dir, "test", seq_len, feature_dim, num_classes
    )

    pin_memory = device.type == "cuda"

    train_loader = build_loader(
        x_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        pin_memory=pin_memory,
        workers=args.workers,
    )
    val_loader = build_loader(
        x_val,
        y_val,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=pin_memory,
        workers=args.workers,
    )
    test_loader = build_loader(
        x_test,
        y_test,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=pin_memory,
        workers=args.workers,
    )

    model_config = {
        "input_size": feature_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_classes": num_classes,
        "dropout": args.dropout,
    }

    model = LSTMClassifier(**model_config).to(device)

    class_weights = None

    if not args.no_class_weights:
        class_weights = compute_class_weights(
            y_train, num_classes
        ).to(device)

    train_criterion, evaluation_criterion = build_criteria(
        class_weights,
        args.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
    )

    # 옵션, 데이터, 모델 구성이 모두 유효한 것을 확인한 뒤 결과 폴더를 만듭니다.
    output_dir.mkdir(parents=True)

    history: list[dict[str, float | int]] = []

    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=train_criterion,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )

        train_metrics = evaluate(
            model, train_loader, evaluation_criterion, device, num_classes
        )
        val_metrics = evaluate(
            model, val_loader, evaluation_criterion, device, num_classes
        )

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": epoch,
            "learning_rate": float(current_lr),
            "train_loss": float(train_loss),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(epoch_result)
        save_json(history_path, history)

        improved = (
            val_metrics["macro_f1"] > best_macro_f1
            or (
                np.isclose(
                    val_metrics["macro_f1"],
                    best_macro_f1,
                )
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "label_map": label_map,
                "preprocessing_config": preprocessing_config,
                "class_weights": (
                    class_weights.detach().cpu().tolist()
                    if class_weights is not None
                    else None
                ),
                "best_val_metrics": val_metrics,
                "training_config": {
                    "label_smoothing": float(args.label_smoothing),
                },
                "seed": args.seed,
            }

            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"lr={current_lr:.6f}"
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model = LSTMClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        evaluation_criterion,
        device,
        num_classes,
    )

    result = {
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["best_val_metrics"],
        "test": test_metrics,
        "model_config": checkpoint["model_config"],
        "training_config": checkpoint.get("training_config", {}),
        "label_map": label_map,
        "device": str(device),
    }

    save_json(output_dir / "test_metrics.json", result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"best checkpoint: {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
