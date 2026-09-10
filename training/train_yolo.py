"""start/stop/cancle YOLO detection 모델 학습 진입점."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = (
    PROJECT_ROOT
    / "start-stop-cancle.v1i.yolov8"
    / "data.yaml"
)
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "runs" / "gesture_detect"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRECTORIES = {
    "train": "train",
    "val": "valid",
    "test": "test",
}


def resolve_path(path: Path) -> Path:
    """상대경로는 프로젝트 루트를 기준으로 해석합니다."""
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def validate_yolo_dataset(data_yaml: Path) -> dict[str, dict[str, int]]:
    """현재 Roboflow YOLO export의 디렉터리와 라벨을 검사합니다."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {data_yaml}")

    dataset_root = data_yaml.parent
    summary: dict[str, dict[str, int]] = {}

    for split, directory_name in SPLIT_DIRECTORIES.items():
        image_dir = dataset_root / directory_name / "images"
        label_dir = dataset_root / directory_name / "labels"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"{split} 이미지 폴더가 없습니다: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"{split} 라벨 폴더가 없습니다: {label_dir}")

        image_files = [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        label_files = list(label_dir.rglob("*.txt"))

        if not image_files:
            raise ValueError(f"{split} 이미지가 비어 있습니다.")

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}

        # 라벨만 있고 대응 이미지가 없는 경우는 명백한 오류입니다.
        orphan_labels = sorted(label_stems - image_stems)
        if orphan_labels:
            raise ValueError(
                f"{split}에 대응 이미지가 없는 라벨이 있습니다: "
                f"{orphan_labels[:5]}"
            )

        object_count = 0

        for label_path in label_files:
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            # 빈 라벨 파일은 background 이미지로 허용합니다.
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()

                if len(fields) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "detect 라벨은 5개 값이어야 합니다."
                    )

                try:
                    class_id = int(fields[0])
                    x_center, y_center, width, height = map(float, fields[1:])
                except ValueError as exc:
                    raise ValueError(
                        f"{label_path}:{line_number}: 숫자 형식이 아닙니다."
                    ) from exc

                if class_id not in {0, 1, 2}:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        f"알 수 없는 class id {class_id}"
                    )

                if not (
                    0.0 <= x_center <= 1.0
                    and 0.0 <= y_center <= 1.0
                    and 0.0 < width <= 1.0
                    and 0.0 < height <= 1.0
                ):
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "bounding box 좌표는 0~1 범위여야 합니다."
                    )

                object_count += 1

        summary[split] = {
            "images": len(image_files),
            "labels": len(label_files),
            "objects": object_count,
            # YOLO에서는 라벨 파일이 없는 이미지도 background로 허용합니다.
            "background_candidates": len(image_stems - label_stems),
        }

    return summary


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested

    return "0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="YOLO data.yaml 경로",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="사전 학습 모델 또는 체크포인트 경로",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="yolov8n_gesture")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="전체 학습 전 1 epoch 소량 실행",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_yaml = resolve_path(args.data)
    device = select_device(args.device)

    dataset_summary = validate_yolo_dataset(data_yaml)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print(f"device={device}")

    epochs = 1 if args.smoke_test else args.epochs
    imgsz = 320 if args.smoke_test else args.imgsz
    fraction = 0.05 if args.smoke_test else 1.0
    run_name = f"{args.name}_smoke" if args.smoke_test else args.name

    # 첫 실행에서는 yolov8n.pt가 없으면 Ultralytics가 다운로드합니다.
    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=args.batch,
        imgsz=imgsz,
        device=device,
        workers=args.workers,
        project=str(DEFAULT_PROJECT_DIR),
        name=run_name,
        exist_ok=False,

        # 모델 선택과 재현성
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        save=True,
        plots=True,
        val=True,

        # 현재 데이터가 이미 Roboflow에서 증강되었으므로 보수적으로 설정
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.20,
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.20,

        cache=False,
        amp=device != "cpu",
        fraction=fraction,
    )

    if model.trainer is None:
        raise RuntimeError("YOLO trainer가 생성되지 않았습니다.")

    best_path = Path(model.trainer.best)

    if not best_path.is_file():
        raise FileNotFoundError(f"best.pt가 생성되지 않았습니다: {best_path}")

    print(f"best weights: {best_path}")

    # validation으로 선택된 best 모델을 test split에서 한 번만 평가합니다.
    best_model = YOLO(str(best_path))

    test_metrics = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(DEFAULT_PROJECT_DIR),
        name=f"{run_name}_test",
        plots=True,
    )

    test_summary = {
        key: float(value)
        for key, value in test_metrics.results_dict.items()
    }
    test_summary["best_weights"] = str(best_path)
    test_summary["device"] = device

    run_dir = best_path.parent.parent
    metrics_path = run_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(test_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(test_summary, ensure_ascii=False, indent=2))
    print(f"test metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    # Windows에서 DataLoader multiprocessing을 안전하게 실행하기 위해 필요합니다.
    raise SystemExit(main())