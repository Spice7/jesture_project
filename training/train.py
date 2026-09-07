"""전처리 결과 검증 → LSTM 학습/validation → 최적 체크포인트 저장.

실제 실행은 main()에서만 시작합니다. test 배열은 열지 않으며, 최종 test 평가는
별도 단계입니다. 원본 NPZ, 전처리 코드, 모델 구조는 변경하지 않습니다.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .models import LSTMClassifier
    from .gesture_schema import validate_label_map, schema_from_config, validate_provenance, RIGHT_LABELS
else:
    from models import LSTMClassifier
    from gesture_schema import validate_label_map, schema_from_config, validate_provenance, RIGHT_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = dict(RIGHT_LABELS)
SPLITS = ("train", "val", "test")
HISTORY_FIELDS = ("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy")


def read_json(path: Path) -> dict:
    """중복 키/NaN을 허용하지 않고 JSON 객체를 읽습니다."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 중복 키: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"JSON 비정상 상수: {value}")

    with path.open(encoding="utf-8-sig") as stream:
        result = json.load(stream, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError(f"JSON 객체가 필요합니다: {path.name}")
    return result


def load_prepared_data(directory: Path) -> dict:
    """파일/분리 계약을 확인하고 train/val만 로딩합니다. test 내용은 접근하지 않습니다."""
    names = [f"{prefix}_{split}.npy" for split in SPLITS for prefix in ("X", "y")]
    names += ["label_map.json", "preprocessing_config.json", "manifest.csv", "report.json"]
    for name in names:
        if not (directory / name).is_file():
            raise ValueError(f"필수 파일 누락: {name}")
    report = read_json(directory / "report.json")
    if (report.get("mode") != "normal" or report.get("success") is not True
            or report.get("training_arrays_written") is not True):
        raise ValueError("정상 normal 전처리 성공 결과가 아닙니다. audit-only 결과는 학습할 수 없습니다.")
    if (report.get("conflicts") != [] or report.get("errors") != []
            or report.get("split_errors") != []):
        raise ValueError("전처리 보고서에 충돌/오류가 있거나 필수 오류 정보가 없습니다.")
    label_map = read_json(directory / "label_map.json")
    validate_label_map(label_map)
    config = read_json(directory / "preprocessing_config.json")
    validate_label_map(config.get("label_map"))
    schema_from_config(config)
    if label_map != config["label_map"]:
        raise ValueError("label_map과 preprocessing_config 불일치")
    length = config.get("seq_len")
    if (type(length) is not int or length < 2 or type(config.get("feature_dim")) is not int
            or config["feature_dim"] != 66 or config.get("X_dtype") != "float32"
            or config.get("y_dtype") != "int64"):
        raise ValueError("전처리 seq_len/feature_dim/dtype 계약이 잘못되었습니다.")

    with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"status", "split", "output_index", "canonical_participant_id", "label", "source_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("manifest 필수 열 누락")
        rows = list(reader)
    grouped = {split: [] for split in SPLITS}
    for row in rows:
        if row["status"] not in ("accepted", "excluded", "duplicate"):
            raise ValueError("manifest에 충돌 또는 알 수 없는 상태가 있습니다.")
        if row["status"] != "accepted":
            if row["split"] or row["output_index"]:
                raise ValueError("제외/중복 행에 배열 인덱스가 있습니다.")
            continue
        if row["split"] not in SPLITS or row["label"] not in label_map or not row["canonical_participant_id"]:
            raise ValueError("manifest의 split/label/참가자 정보가 잘못되었습니다.")
        try:
            index = int(row["output_index"])
        except (ValueError, TypeError) as exc:
            raise ValueError("manifest output_index는 정수여야 합니다.") from exc
        grouped[row["split"]].append((index, row))

    subjects = config.get("canonical_split_subjects")
    if not isinstance(subjects, dict) or set(subjects) != set(SPLITS):
        raise ValueError("config 참가자 배정 누락/오류")
    seen, distributions = set(), {}
    for split in SPLITS:
        entries = sorted(grouped[split], key=lambda item: item[0])
        grouped[split] = entries
        if not entries or [i for i, _ in entries] != list(range(len(entries))):
            raise ValueError(f"{split}: output_index 중복/누락 또는 빈 split")
        actual = {row["canonical_participant_id"] for _, row in entries}
        specified = subjects[split]
        if (not isinstance(specified, list) or not all(isinstance(v, str) and v for v in specified)
                or len(specified) != len(set(specified)) or set(specified) != actual):
            raise ValueError(f"{split}: config와 manifest 참가자 배정 불일치")
        if seen & actual:
            raise ValueError("참가자 누수: train/val/test에 같은 canonical ID가 있습니다.")
        seen.update(actual)
        counts = Counter(row["label"] for _, row in entries)
        if set(counts) != set(label_map):
            raise ValueError(f"{split}: 클래스 누락")
        distributions[split] = dict(counts)
    if report.get("split_class_distribution") != distributions:
        raise ValueError("report와 manifest의 split 클래스 분포 불일치")
    if report.get("counts") != dict(Counter(row["status"] for row in rows)):
        raise ValueError("report와 manifest 상태 개수 불일치")
    validate_provenance(rows, config, report)

    arrays = {}
    for split in ("train", "val"):
        # test는 파일 존재/메타데이터 분리만 검사합니다. 배열 크기/값 검사는 평가 단계의 역할입니다.
        x = np.load(directory / f"X_{split}.npy", allow_pickle=False)
        y = np.load(directory / f"y_{split}.npy", allow_pickle=False)
        if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 3
                or x.shape[1:] != (length, 66) or len(x) == 0 or not np.isfinite(x).all()):
            raise ValueError(f"{split}: X dtype/shape/유한값 오류")
        if (not isinstance(y, np.ndarray) or y.dtype != np.int64 or y.shape != (len(x),)
                or set(y.tolist()) != set(label_map.values())):
            raise ValueError(f"{split}: y dtype/shape/라벨 오류 또는 클래스 누락")
        entries = grouped[split]
        if len(entries) != len(x):
            raise ValueError(f"{split}: 배열과 manifest 행 개수 불일치")
        if any(y[index] != label_map[row["label"]] for index, row in entries):
            raise ValueError(f"{split}: manifest 라벨과 y 불일치")
        arrays[split] = (x, y)
    hashes = {}
    for name in ("manifest.csv", "preprocessing_config.json", "label_map.json", "report.json"):
        with (directory / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"arrays": arrays, "preprocessing_config": config, "label_map": label_map,
            "input_sha256": hashes, "split_counts": {s: len(grouped[s]) for s in SPLITS}}


def seed_everything(seed: int) -> None:
    """난수 초기화는 실행 시에만 수행합니다. 다른 환경까지 완전 동일함을 보장하지 않습니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Windows worker가 가져올 수 있는 최상위 함수로 worker별 난수를 설정합니다."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(request: str) -> torch.device:
    available = torch.cuda.is_available()
    if request == "cuda" and not available:
        raise ValueError("CUDA를 요청했지만 사용할 수 없습니다. --device cpu를 사용하세요.")
    return torch.device("cuda" if request == "cuda" or (request == "auto" and available) else "cpu")


def make_loaders(data: dict, batch_size: int, workers: int, seed: int) -> dict:
    """TensorDataset은 X/y를 묶고 DataLoader는 배치로 나눕니다. train만 섞습니다."""
    result = {}
    for offset, split in enumerate(("train", "val")):
        x, y = data["arrays"][split]
        generator = torch.Generator().manual_seed(seed + offset)
        result[split] = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=batch_size, shuffle=split == "train", drop_last=False,
            num_workers=workers, worker_init_fn=seed_worker, generator=generator,
        )
    return result


def run_epoch(model, loader, device, optimizer=None, max_grad_norm: float = 1.0) -> dict:
    """optimizer가 있으면 학습, 없으면 validation. 손실은 샘플 수로 가중 평균합니다."""
    training = optimizer is not None
    # train/eval은 Dropout 동작을 바꿉니다. validation에서는 no_grad로 그래프도 만들지 않습니다.
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)  # softmax 대신 logits를 직접 전달합니다.
            if not torch.isfinite(loss).item():
                raise FloatingPointError("loss가 NaN/Inf입니다.")
            if training:
                loss.backward()  # 미분으로 각 파라미터의 기울기를 계산합니다.
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                        raise FloatingPointError("gradient가 NaN/Inf입니다.")
                # 큰 기울기의 전체 norm을 제한해 급격한 업데이트를 줄입니다.
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
                optimizer.step()  # 계산된 기울기를 사용해 실제 가중치를 갱신합니다.
                if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                    raise FloatingPointError("업데이트 후 파라미터가 NaN/Inf입니다.")
            size = y.shape[0]
            total_loss += loss.item() * size
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += size
    if not count or not math.isfinite(total_loss):
        raise ValueError("빈 DataLoader 또는 비정상 누적 손실입니다.")
    return {"loss": total_loss / count, "accuracy": correct / count}


@dataclass
class EarlyStopping:
    """최저 손실(best)과 충분한 개선(reference)을 따로 추적합니다.

    첫 값은 두 기준을 초기화합니다. 이후 value < reference 이면서
    reference - value >= min_delta이면 기준점을 갱신하고 대기를 0으로 만듭니다.
    미세 개선은 누적되어 나중에 기준을 넘을 수 있습니다. 동률은 개선이 아닙니다.
    """
    patience: int
    min_delta: float
    best: float = math.inf
    reference: float = math.inf
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if not math.isfinite(value):
            raise FloatingPointError("val_loss가 NaN/Inf입니다.")
        is_best = value < self.best
        if is_best:
            self.best = value
        if math.isinf(self.reference) or (value < self.reference and self.reference - value >= self.min_delta):
            self.reference = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return is_best, self.bad_epochs >= self.patience


def atomic_write(path: Path, writer) -> None:
    """동일 폴더의 임시 파일을 완성한 뒤 교체합니다. 작업 중 오류면 임시 파일만 정리합니다."""
    with NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    def writer(temporary):
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    atomic_write(path, writer)


def write_history(path: Path, rows: list) -> None:
    def writer(temporary):
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            output = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            output.writeheader()
            output.writerows(rows)
    atomic_write(path, writer)


def save_checkpoint(path, model, model_config, data, metrics) -> None:
    # CPU의 독립 복사본을 저장해 이후 학습이 best 가중치까지 바꾸는 참조 문제를 방지합니다.
    checkpoint = {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": model_config, "best_epoch": metrics["epoch"],
        "metrics": dict(metrics), "label_map": data["label_map"],
        "preprocessing_config": data["preprocessing_config"],
    }
    atomic_write(path, lambda temporary: torch.save(checkpoint, temporary))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in (("hidden-size", 64), ("num-layers", 1), ("batch-size", 32),
                          ("epochs", 100), ("patience", 10), ("seed", 42), ("num-workers", 0)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("dropout", .2), ("learning-rate", .001), ("weight-decay", .0001),
                          ("min-delta", .0001), ("max-grad-norm", 1.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def validate_args(args) -> None:
    for name in ("hidden_size", "num_layers", "batch_size", "epochs", "patience"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name}는 1 이상이어야 합니다.")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("num_workers는 0 이상, seed는 0~2**32-1이어야 합니다.")
    for name in ("dropout", "learning_rate", "weight_decay", "min_delta", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name}는 유한한 값이어야 합니다.")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("dropout은 [0,1), weight_decay/min_delta는 0 이상이어야 합니다.")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning_rate와 max_grad_norm은 양수여야 합니다.")


def run(args) -> int:
    """설정/경로 검사 → 데이터 검증 → 학습/validation → 결과 확정. test 평가는 없습니다."""
    validate_args(args)
    directory, output = args.data_dir.resolve(), args.output_dir.resolve()
    if not directory.is_dir():
        raise ValueError("data-dir 폴더가 없습니다.")
    if output.is_relative_to(directory) or output.is_relative_to((PROJECT_ROOT / "dataset").resolve()):
        raise ValueError("output-dir는 입력 전처리 폴더와 원본 dataset 밖이어야 합니다.")
    if output.exists():
        raise ValueError("기존 output-dir는 덮어쓰지 않습니다. 새 폴더를 지정하세요.")
    device = choose_device(args.device)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "epochs_completed": 0, "best_epoch": None,
               "best_metrics": None, "early_stopped": False, "error": None,
               "test_evaluated": False, "partial_run": True}
    try:
        write_json(output / "training_summary.json", summary)
        data = load_prepared_data(directory)
        seed_everything(args.seed)
        model_config = {"input_size": data["preprocessing_config"]["feature_dim"],
                        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
                        "num_classes": len(data["label_map"]), "dropout": args.dropout}
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        settings.update(data_dir=str(directory), output_dir=str(output), actual_device=str(device),
                        model_config=model_config, label_map=data["label_map"],
                        preprocessing_config=data["preprocessing_config"], input_sha256=data["input_sha256"],
                        split_counts=data["split_counts"], versions={"python": platform.python_version(),
                        "numpy": str(np.__version__), "torch": str(torch.__version__)},
                        optimizer="AdamW", loss="CrossEntropyLoss", accuracy_unit="ratio_0_to_1",
                        loader_config={"train_shuffle": True, "val_shuffle": False, "drop_last": False},
                        cuda_seeded=torch.cuda.is_available(),
                        early_stopping_rule="strict decrease from reference AND reference-value >= min_delta; reference updates only on sufficient improvement; best checkpoint uses any strict minimum")
        write_json(output / "training_config.json", settings)
        loaders = make_loaders(data, args.batch_size, args.num_workers, args.seed)
        model = LSTMClassifier(**model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        stopping = EarlyStopping(args.patience, args.min_delta)
        history = []
        write_history(output / "history.csv", history)
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, loaders["train"], device, optimizer, args.max_grad_norm)
            val = run_epoch(model, loaders["val"], device)
            metrics = {"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                       "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
            is_best, should_stop = stopping.update(val["loss"])
            if is_best:
                save_checkpoint(output / "best_model.pt", model, model_config, data, metrics)
                summary.update(best_epoch=epoch, best_metrics=dict(metrics))
            history.append(metrics)
            write_history(output / "history.csv", history)
            summary.update(epochs_completed=epoch, early_stopped=should_stop)
            write_json(output / "training_summary.json", summary)
            print(f"epoch {epoch}: train_loss={train['loss']:.6f}, val_loss={val['loss']:.6f}, "
                  f"train_acc={train['accuracy']:.4f}, val_acc={val['accuracy']:.4f}")
            if should_stop:
                break
        summary.update(status="success", partial_run=False)
        write_json(output / "training_summary.json", summary)
        print(f"학습 완료: best_epoch={summary['best_epoch']}, 결과={output}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        summary.update(status="interrupted" if interrupted else "failed", partial_run=True,
                       error=f"{type(exc).__name__}: {exc}")
        try:
            write_json(output / "training_summary.json", summary)
        except OSError as write_error:
            print(f"상태 기록도 실패했습니다: {write_error}. 이 실행을 성공 결과로 사용하지 마세요.", file=sys.stderr)
        print(f"{summary['status']}: {summary['error']}", file=sys.stderr)
        return 130 if interrupted else 1


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print(f"실행 설정 오류: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("출력 초기화 전 실행이 중단되었습니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
