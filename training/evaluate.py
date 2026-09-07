"""저장된 LSTM의 잘라진 시퀀스 분류 평가. 학습/튜닝/실시간 추론은 하지 않습니다.

흐름: 옵션 검사 → 학습 결과/메타데이터 검증 → 선택 split 로딩 → 모델 복원
→ 배치 추론 → 지표/CSV 저장. import만으로 파일을 읽거나 평가하지 않습니다.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .models import LSTMClassifier
    from .train import read_json, validate_label_map, choose_device, atomic_write, write_json
    from .gesture_schema import schema_from_config, validate_provenance, RIGHT_LABELS
else:
    from models import LSTMClassifier
    from train import read_json, validate_label_map, choose_device, atomic_write, write_json
    from gesture_schema import schema_from_config, validate_provenance, RIGHT_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = dict(RIGHT_LABELS)
LABEL_ORDER = tuple(LABEL_MAP)
SPLITS = ("train", "val", "test")
METADATA_FILES = ("manifest.csv", "preprocessing_config.json", "label_map.json", "report.json")
PREDICTION_FIELDS = (
    "output_index", "source_path", "participant_id", "canonical_participant_id",
    "true_label_id", "true_label", "predicted_label_id", "predicted_label",
    *(f"probability_{label}" for label in LABEL_ORDER), "correct",
)


def sha256(path: Path) -> str:
    """지정한 파일 하나만 읽어 내용 지문을 계산합니다."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_metadata(left, right) -> bool:
    """JSON형 설정을 자료형까지 비교합니다. False와 0을 같은 설정으로 취급하지 않습니다."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_metadata(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_metadata(a, b) for a, b in zip(left, right))
    return left == right


def validate_manifest(directory, config, report):
    """전체 split의 참가자 누수/인덱스를 메타데이터로 검사합니다. NPY는 열지 않습니다."""
    schema_from_config(config)
    label_map = config["label_map"]
    with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"status", "split", "output_index", "label", "source_path",
                    "participant_id", "canonical_participant_id"}
        fields = reader.fieldnames or []
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError("manifest 필수 열 누락/중복")
        rows = list(reader)
    grouped = {split: [] for split in SPLITS}
    for row in rows:
        if None in row or any(row.get(key) is None for key in required):
            raise ValueError("manifest 행의 열 개수가 잘못되었습니다.")
        if row["status"] not in ("accepted", "excluded", "duplicate"):
            raise ValueError("manifest 충돌 또는 알 수 없는 상태")
        if row["status"] != "accepted":
            if row["split"] or row["output_index"]:
                raise ValueError("제외/중복 행에 배열 인덱스가 있습니다.")
            continue
        if (row["split"] not in SPLITS or row["label"] not in label_map
                or any(not row[key].strip() for key in
                       ("source_path", "participant_id", "canonical_participant_id"))):
            raise ValueError("manifest split/label/경로/참가자 정보 오류")
        try:
            index = int(row["output_index"])
        except ValueError as exc:
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
    if (report.get("split_class_distribution") != distributions
            or report.get("counts") != dict(Counter(row["status"] for row in rows))):
        raise ValueError("report와 manifest 개수/분포 불일치")
    validate_provenance(rows, config, report)
    return grouped


def load_metadata(run_dir: Path, directory: Path) -> dict:
    """학습 완료 여부와 체크포인트/설정/입력 지문을 교차 검증합니다."""
    settings = read_json(run_dir / "training_config.json")
    summary = read_json(run_dir / "training_summary.json")
    if summary.get("status") != "success" or summary.get("partial_run") is not False:
        raise ValueError("성공하고 완료된 학습 결과만 평가할 수 있습니다.")
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("체크포인트는 텐서/설정 dictionary여야 합니다.")
    validate_label_map(checkpoint.get("label_map"))
    model_config = checkpoint.get("model_config")
    if (not isinstance(model_config, dict) or set(model_config) !=
            {"input_size", "hidden_size", "num_layers", "num_classes", "dropout"}
            or not same_metadata(model_config, settings.get("model_config"))
            or model_config["input_size"] != 66 or model_config["num_classes"] != len(checkpoint["label_map"])):
        raise ValueError("체크포인트 model_config 불일치/계약 오류")
    epoch = checkpoint.get("best_epoch")
    metrics = checkpoint.get("metrics")
    if (type(epoch) is not int or epoch < 1 or not same_metadata(epoch, summary.get("best_epoch"))
            or not isinstance(metrics, dict) or not same_metadata(metrics, summary.get("best_metrics"))
            or not same_metadata(metrics.get("epoch"), epoch)):
        raise ValueError("체크포인트 best_epoch/metrics와 학습 summary 불일치")
    for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
        value = metrics.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or (key.endswith("accuracy") and value > 1)):
            raise ValueError(f"학습 metrics 오류: {key}")
    report = read_json(directory / "report.json")
    if (report.get("mode") != "normal" or report.get("success") is not True
            or report.get("training_arrays_written") is not True
            or any(report.get(key) != [] for key in ("conflicts", "errors", "split_errors"))):
        raise ValueError("성공한 normal 전처리 결과가 아니거나 충돌/오류가 있습니다.")
    label_map = read_json(directory / "label_map.json")
    config = read_json(directory / "preprocessing_config.json")
    for value in (label_map, config.get("label_map"), settings.get("label_map"), checkpoint.get("label_map")):
        validate_label_map(value)
        if value != label_map:
            raise ValueError("학습/체크포인트/전처리 label_map 불일치")
    schema_from_config(config)
    if (not same_metadata(config, settings.get("preprocessing_config"))
            or not same_metadata(config, checkpoint.get("preprocessing_config"))):
        raise ValueError("preprocessing_config 불일치")
    if (type(config.get("seq_len")) is not int or config["seq_len"] < 2
            or type(config.get("feature_dim")) is not int or config["feature_dim"] != 66
            or config.get("X_dtype") != "float32" or config.get("y_dtype") != "int64"):
        raise ValueError("전처리 seq_len/feature_dim/dtype 계약 오류")
    hashes = {name: sha256(directory / name) for name in METADATA_FILES}
    saved_hashes = settings.get("input_sha256")
    if not isinstance(saved_hashes, dict) or any(saved_hashes.get(k) != v for k, v in hashes.items()):
        raise ValueError("학습 당시 입력 메타데이터 SHA-256과 현재 파일 불일치")
    grouped = validate_manifest(directory, config, report)
    if settings.get("split_counts") != {s: len(grouped[s]) for s in SPLITS}:
        raise ValueError("학습 split_counts와 manifest 개수 불일치")
    return {"checkpoint": checkpoint, "checkpoint_sha256": checkpoint_hash,
            "model_config": model_config, "label_map": label_map,
            "preprocessing_config": config, "input_sha256": hashes, "grouped": grouped}


def load_split(directory: Path, split: str, metadata: dict):
    """선택한 X/y만 검사합니다. 재전처리/정렬/자료형 변환이나 이동량 필터는 없습니다."""
    if split not in ("val", "test"):
        raise ValueError("평가 split은 val 또는 test여야 합니다.")
    x = np.load(directory / f"X_{split}.npy", allow_pickle=False)
    y = np.load(directory / f"y_{split}.npy", allow_pickle=False)
    length = metadata["preprocessing_config"]["seq_len"]
    label_map = metadata["label_map"]
    if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 3
            or x.shape[1:] != (length, 66) or len(x) == 0 or not np.isfinite(x).all()):
        raise ValueError(f"{split}: X dtype/shape/유한값 오류 또는 빈 데이터")
    if (not isinstance(y, np.ndarray) or y.dtype != np.int64 or y.shape != (len(x),)
            or set(y.tolist()) != set(label_map.values())):
        raise ValueError(f"{split}: y dtype/shape/라벨 오류 또는 클래스 누락")
    entries = metadata["grouped"][split]
    if len(entries) != len(x):
        raise ValueError(f"{split}: 배열과 manifest 행 개수 불일치")
    if any(y[index] != label_map[row["label"]] for index, row in entries):
        raise ValueError(f"{split}: manifest 라벨과 y 불일치")
    hashes = {f"{prefix}_{split}.npy": sha256(directory / f"{prefix}_{split}.npy") for prefix in ("X", "y")}
    return x, y, hashes


def restore_model(metadata: dict, device: torch.device):
    """기본 구조를 추측하지 않고 저장된 구조/가중치를 엄격하게 복원합니다."""
    model = LSTMClassifier(**metadata["model_config"])
    state = metadata["checkpoint"].get("model_state_dict")
    if (not isinstance(state, dict) or any(not isinstance(v, torch.Tensor)
            or v.dtype != torch.float32 or not torch.isfinite(v).all().item() for v in state.values())):
        raise ValueError("model_state_dict는 유한한 float32 텐서여야 합니다.")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict_batches(model, x, y, device, batch_size=32, num_workers=0, num_classes=4):
    """가중치를 갱신하지 않고 모든 배치의 logits에서 예측/확률/평균 손실을 계산합니다."""
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    model.eval()  # Dropout을 끕니다. inference_mode는 별도로 미분 그래프 생성을 막습니다.
    criterion = nn.CrossEntropyLoss()
    total_loss, count = 0.0, 0
    predicted, probabilities = [], []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            if logits.shape != (len(batch_y), num_classes) or not torch.isfinite(logits).all().item():
                raise FloatingPointError("logits shape 오류 또는 NaN/Inf")
            # logits는 점수, argmax는 최대 점수의 클래스, softmax는 확률 표현입니다.
            # CrossEntropyLoss 자체가 logits를 받으므로 확률을 loss에 넣지 않습니다.
            loss = criterion(logits, batch_y)
            probs = logits.softmax(dim=1)
            if not torch.isfinite(loss).item() or not torch.isfinite(probs).all().item():
                raise FloatingPointError("loss/확률이 NaN/Inf입니다.")
            size = len(batch_y)
            total_loss += loss.item() * size  # 마지막 작은 배치도 샘플 수만큼만 가중합니다.
            count += size
            predicted.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    if count != len(y) or count == 0 or not math.isfinite(total_loss):
        raise ValueError("전체 샘플 평가 누락 또는 비정상 누적 손실")
    return total_loss / count, np.concatenate(predicted), np.concatenate(probabilities)


def classification_metrics(y, predicted, loss, label_map=None):
    """행=실제, 열=예측. 전달받은 라벨 계약 전체로 지표를 계산합니다."""
    label_map = LABEL_MAP if label_map is None else label_map
    validate_label_map(label_map)
    labels = tuple(sorted(label_map, key=label_map.get))
    count = len(labels)
    if (y.ndim != 1 or predicted.shape != y.shape or len(y) == 0
            or y.dtype.kind not in "iu" or predicted.dtype.kind not in "iu"
            or not np.isin(y, list(label_map.values())).all() or not np.isin(predicted, list(label_map.values())).all()
            or not math.isfinite(loss) or loss < 0):
        raise ValueError("지표 계산 입력 오류")
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    if int(matrix.sum()) != len(y):
        raise ValueError("confusion matrix 합과 샘플 수 불일치")
    tp = matrix.diagonal().astype(np.float64)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    # precision=TP/(TP+FP), recall=TP/(TP+FN). 분모 0은 0으로 정의합니다.
    precision = np.divide(tp, predicted_count, out=np.zeros(count), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(count), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(count), where=(precision + recall) != 0)
    per_class = {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                        "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)}
    # accuracy는 전체 샘플의 정답 비율. macro F1은 작은 클래스도 동일 비중으로 봅니다.
    # macro F1은 클래스별 F1의 평균이며 macro precision/recall로 재계산하지 않습니다.
    overall = {"loss": float(loss), "accuracy": float(tp.sum() / len(y)),
               "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
               "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(f1, support) / len(y))}
    return overall, per_class, matrix


def save_classification_report(path, split, overall, per_class):
    """이미 계산한 지표를 읽기 쉬운 텍스트 표로 저장합니다. 추가 추론은 하지 않습니다."""
    labels = tuple(per_class)
    count = sum(per_class[label]["support"] for label in labels)
    # weighted 평균은 실제 클래스 샘플 수를 가중치로 사용합니다.
    weighted = {key: sum(per_class[label][key] * per_class[label]["support"]
                         for label in labels) / count for key in ("precision", "recall")}

    def row(name, precision, recall, f1, support):
        return f"{name:>14} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}"

    lines = ["Classification Report", f"split: {split}", "",
             f"{'':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}", ""]
    for label in labels:
        values = per_class[label]
        lines.append(row(label, values["precision"], values["recall"], values["f1"], values["support"]))
    lines += ["", f"{'accuracy':>14} {'':>10} {'':>10} {overall['accuracy']:>10.4f} {count:>10d}",
              row("macro avg", overall["macro_precision"], overall["macro_recall"], overall["macro_f1"], count),
              row("weighted avg", weighted["precision"], weighted["recall"], overall["weighted_f1"], count),
              "", f"CrossEntropyLoss: {overall['loss']:.6f}",
              "점수는 0~1 비율이며 소수점 4자리로 표시합니다. 분모가 0인 지표는 0입니다.",
              "완료 여부는 evaluation_metrics.json의 status=success, completed=true를 확인하세요."]
    # 임시 파일을 완성한 뒤 교체하며, 실패하면 전체 평가도 실패 상태로 기록됩니다.
    atomic_write(path, lambda temporary: temporary.write_text("\n".join(lines) + "\n", encoding="utf-8"))


def write_csv(path, fields, rows):
    """CSV 하나를 완성한 후 교체합니다. 빈 rows도 헤더를 저장합니다."""
    def writer(temporary):
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            output = csv.DictWriter(stream, fieldnames=fields)
            output.writeheader()
            output.writerows(rows)
    atomic_write(path, writer)


def save_predictions(output, entries, y, predicted, probabilities, matrix, label_map=None):
    """검증된 output_index로 manifest와 예측을 연결합니다. 원본 NPZ는 열지 않습니다."""
    label_map = LABEL_MAP if label_map is None else label_map
    validate_label_map(label_map)
    labels = tuple(sorted(label_map, key=label_map.get))
    fields = (*PREDICTION_FIELDS[:8], *(f"probability_{label}" for label in labels), "correct")
    rows = []
    for index, source in entries:
        row = {key: source[key] for key in ("source_path", "participant_id", "canonical_participant_id")}
        row.update(output_index=index, true_label_id=int(y[index]), true_label=labels[y[index]],
                   predicted_label_id=int(predicted[index]), predicted_label=labels[predicted[index]],
                   correct=bool(y[index] == predicted[index]))
        row.update({f"probability_{label}": float(probabilities[index, i]) for i, label in enumerate(labels)})
        rows.append(row)
    write_csv(output / "predictions.csv", fields, rows)
    write_csv(output / "misclassified.csv", fields, (r for r in rows if not r["correct"]))
    axis = "true_label / predicted_label"
    write_csv(output / "confusion_matrix.csv", (axis, *labels),
              ({axis: label, **{name: int(matrix[i, j]) for j, name in enumerate(labels)}}
               for i, label in enumerate(labels)))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "data-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def validate_args(args):
    """출력을 만들기 전에 옵션/경로/device를 확인합니다."""
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size는 양수, num-workers는 0 이상이어야 합니다.")
    run_dir, directory, output = args.run_dir.resolve(), args.data_dir.resolve(), args.output_dir.resolve()
    if not run_dir.is_dir() or not directory.is_dir():
        raise ValueError("run-dir/data-dir 폴더가 없습니다.")
    if output.is_relative_to(directory) or output.is_relative_to((PROJECT_ROOT / "dataset").resolve()):
        raise ValueError("output-dir는 전처리 폴더와 원본 dataset 밖이어야 합니다.")
    if output.exists():
        raise ValueError("기존 output-dir는 비어 있어도 덮어쓰지 않습니다.")
    return run_dir, directory, output, choose_device(args.device)


def run(args):
    """마지막 성공 상태가 기록되어야 완료 결과입니다. 실패 시 부분 지표를 공개하지 않습니다."""
    run_dir, directory, output, device = validate_args(args)
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "split": args.split, "completed": False, "sample_count": None,
              "overall": None, "per_class": None, "label_order": list(LABEL_ORDER),
              "zero_division_policy": "분모가 0이면 0; macro는 모델의 전체 클래스 포함",
              "score_unit": "ratio_0_to_1", "error": None,
              "split_note": ("validation은 이미 모델/epoch 선택에 사용된 자료입니다." if args.split == "val"
                             else "별도의 held-out test 평가입니다. 이 결과로 반복 튜닝하지 마세요.")}
    try:
        write_json(output / "evaluation_metrics.json", status)
        metadata = load_metadata(run_dir, directory)
        label_map = metadata["label_map"]
        status["label_order"] = sorted(label_map, key=label_map.get)
        x, y, array_hashes = load_split(directory, args.split, metadata)
        model = restore_model(metadata, device)
        config = {"run_dir": str(run_dir), "data_dir": str(directory), "output_dir": str(output),
                  "checkpoint_path": str(run_dir / "best_model.pt"), "split": args.split,
                  "batch_size": args.batch_size, "requested_device": args.device,
                  "actual_device": str(device), "num_workers": args.num_workers,
                  **{key: metadata[key] for key in ("model_config", "label_map", "preprocessing_config",
                                                  "checkpoint_sha256", "input_sha256")},
                  "array_sha256": array_hashes,
                  "versions": {"python": platform.python_version(), "numpy": str(np.__version__),
                               "torch": str(torch.__version__)},
                  "loader_config": {"shuffle": False, "drop_last": False},
                  "hash_limit": "NPY 해시는 평가 시점의 지문이며 학습 당시 NPY 동일성을 보증하지 않습니다."}
        write_json(output / "evaluation_config.json", config)
        loss, predicted, probabilities = predict_batches(model, x, y, device, args.batch_size, args.num_workers, len(label_map))
        overall, per_class, matrix = classification_metrics(y, predicted, loss, label_map)
        save_predictions(output, metadata["grouped"][args.split], y, predicted, probabilities, matrix, label_map)
        save_classification_report(output / "classification_report.txt", args.split, overall, per_class)
        # 개별 파일 저장 후 마지막으로 완료 표시: 중간 파일만 보고 성공으로 판단하면 안 됩니다.
        status.update(status="success", completed=True, sample_count=len(y), overall=overall, per_class=per_class)
        write_json(output / "evaluation_metrics.json", status)
        print(f"평가 완료 ({args.split}): N={len(y)}, accuracy={overall['accuracy']:.4f}, "
              f"macro_f1={overall['macro_f1']:.4f}, 결과={output}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        status.update(status="interrupted" if interrupted else "failed", completed=False,
                      sample_count=None, overall=None, per_class=None, error=f"{type(exc).__name__}: {exc}")
        try:
            write_json(output / "evaluation_metrics.json", status)
        except OSError as write_error:
            print(f"상태 기록도 실패: {write_error}. 이 출력은 완료 결과로 사용하지 마세요.", file=sys.stderr)
        print(f"{status['status']}: {status['error']}", file=sys.stderr)
        return 130 if interrupted else 1


def main(argv=None):
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
