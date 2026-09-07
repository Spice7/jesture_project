"""YOLOv8 object detection training, tuning, and inference entry point."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "yolo_param.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, tune, or run an Ultralytics YOLOv8 model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML parameter file (default: model_yolo/yolo_param.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "tune", "test", "predict"),
        default="train",
        help="Run training, hyperparameter tuning, test evaluation, or inference.",
    )
    parser.add_argument(
        "--source",
        help="Inference input used with --mode predict: image, video, directory, URL, or webcam index (0).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("The top level of the parameter file must be a mapping.")

    return config, config_path.parent


def resolve_local_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def resolve_model_source(config_dir: Path, value: str | Path) -> str:
    """Resolve explicit model paths while leaving weight names such as yolov8n.pt unchanged."""
    model_text = str(value)
    if Path(model_text).is_absolute() or len(Path(model_text).parts) > 1:
        return str(resolve_local_path(config_dir, model_text))
    return model_text


def build_common_args(config: dict[str, Any], config_dir: Path) -> tuple[str, dict[str, Any]]:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("'paths' must be defined as a mapping in yolo_param.yaml.")

    model_value = paths.get("model")
    data_value = paths.get("data")
    project_value = paths.get("project")
    if not all((model_value, data_value, project_value)):
        raise ValueError("'paths.model', 'paths.data', and 'paths.project' are required.")

    # A simple weight name such as yolov8n.pt is left unchanged so Ultralytics can
    # find or download it. Explicit relative/absolute paths are resolved locally.
    model_text = resolve_model_source(config_dir, model_value)

    data_path = resolve_local_path(config_dir, data_value)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    common_args = {
        "data": str(data_path),
        "project": str(resolve_local_path(config_dir, project_value)),
    }
    return model_text, common_args


def run_train(model: YOLO, config: dict[str, Any], common_args: dict[str, Any]) -> None:
    train_args = config.get("train", {})
    if not isinstance(train_args, dict):
        raise ValueError("'train' must be a mapping in yolo_param.yaml.")

    results = model.train(**common_args, **train_args)

    validation_args = config.get("validation", {})
    if validation_args.get("enabled", True):
        val_args = {key: value for key, value in validation_args.items() if key != "enabled"}
        validation_run_args = {**common_args, **val_args}
        model.val(**validation_run_args)

    print(f"Training complete. Results saved to: {results.save_dir}")


def run_tune(model: YOLO, config: dict[str, Any], common_args: dict[str, Any]) -> None:
    tune_args = config.get("tune", {})
    if not isinstance(tune_args, dict):
        raise ValueError("'tune' must be a mapping in yolo_param.yaml.")

    tune_args = dict(tune_args)
    enabled = tune_args.pop("enabled", True)
    if not enabled:
        raise ValueError("Set 'tune.enabled: true' before running with --mode tune.")

    # Training settings are also used as the fixed baseline for each tuning trial.
    train_args = dict(config.get("train", {}))
    train_args.pop("name", None)
    tuning_run_args = {**common_args, **train_args, **tune_args}
    model.tune(**tuning_run_args)


def get_split_image_paths(data_yaml: str, split: str) -> list[Path]:
    data_yaml_path = Path(data_yaml)
    with data_yaml_path.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file) or {}

    dataset_root = data_yaml_path.parent
    if dataset_config.get("path"):
        configured_root = Path(dataset_config["path"]).expanduser()
        dataset_root = configured_root if configured_root.is_absolute() else dataset_root / configured_root

    split_value = dataset_config.get(split)
    if not isinstance(split_value, str):
        raise ValueError(f"'{split}' must be a directory path in {data_yaml_path}.")

    split_path = Path(split_value).expanduser()
    split_path = split_path if split_path.is_absolute() else dataset_root / split_path
    split_path = split_path.resolve()
    if not split_path.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_path}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = sorted(path for path in split_path.rglob("*") if path.suffix.lower() in extensions)
    if not image_paths:
        raise ValueError(f"No test images found in: {split_path}")
    return image_paths


def read_yolo_labels(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    path_parts = list(image_path.parts)
    try:
        images_index = max(index for index, part in enumerate(path_parts) if part.lower() == "images")
    except ValueError as error:
        raise ValueError(f"Expected an 'images' directory in path: {image_path}") from error
    path_parts[images_index] = "labels"
    label_path = Path(*path_parts).with_suffix(".txt")
    if not label_path.is_file() or not label_path.read_text(encoding="utf-8").strip():
        return np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)

    classes: list[int] = []
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid detection label in {label_path}: {line}")
        class_id = int(values[0])
        x_center, y_center, width, height = map(float, values[1:])
        classes.append(class_id)
        boxes.append(
            [
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            ]
        )
    return np.asarray(classes, dtype=int), np.asarray(boxes, dtype=float)


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=float)
    intersection_top_left = np.maximum(box[:2], boxes[:, :2])
    intersection_bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection_size = np.clip(intersection_bottom_right - intersection_top_left, 0, None)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return np.array([]), np.array([]), float("nan")

    order = np.argsort(-y_score, kind="stable")
    sorted_true = y_true[order]
    sorted_scores = y_score[order]
    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, len(sorted_true) - 1]
    true_positives = np.cumsum(sorted_true)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    tpr = np.r_[0.0, true_positives / positives, 1.0]
    fpr = np.r_[0.0, false_positives / negatives, 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def calculate_r2(y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
    if not y_true:
        return float("nan")
    true_values = np.concatenate(y_true)
    predicted_values = np.concatenate(y_pred)
    total_sum_of_squares = float(np.sum((true_values - true_values.mean()) ** 2))
    if total_sum_of_squares == 0:
        return float("nan")
    residual_sum_of_squares = float(np.sum((true_values - predicted_values) ** 2))
    return 1.0 - residual_sum_of_squares / total_sum_of_squares


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def flatten_metrics(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def calculate_custom_test_metrics(
    model: YOLO,
    image_paths: list[Path],
    class_names: dict[int, str],
    test_args: dict[str, Any],
    metric_conf: float,
    match_iou: float,
    roc_min_conf: float,
    save_dir: Path,
) -> dict[str, Any]:
    prediction_args = {
        "source": [str(path) for path in image_paths],
        "stream": True,
        "save": False,
        "verbose": False,
        "conf": min(metric_conf, roc_min_conf),
        "iou": test_args.get("iou", 0.7),
        "imgsz": test_args.get("imgsz", 640),
        "device": test_args.get("device"),
        "max_det": test_args.get("max_det", 300),
    }

    total_tp = total_fp = total_fn = exact_images = 0
    matched_true_boxes: list[np.ndarray] = []
    matched_predicted_boxes: list[np.ndarray] = []
    roc_true: dict[int, list[int]] = {class_id: [] for class_id in class_names}
    roc_scores: dict[int, list[float]] = {class_id: [] for class_id in class_names}

    for result in model.predict(**prediction_args):
        image_path = Path(result.path).resolve()
        true_classes, true_boxes = read_yolo_labels(image_path)

        if result.boxes is None or len(result.boxes) == 0:
            predicted_classes = np.empty((0,), dtype=int)
            predicted_boxes = np.empty((0, 4), dtype=float)
            predicted_scores = np.empty((0,), dtype=float)
        else:
            predicted_classes = result.boxes.cls.cpu().numpy().astype(int)
            predicted_boxes = result.boxes.xyxyn.cpu().numpy()
            predicted_scores = result.boxes.conf.cpu().numpy()

        for class_id in class_names:
            roc_true[class_id].append(int(class_id in true_classes))
            class_scores = predicted_scores[predicted_classes == class_id]
            roc_scores[class_id].append(float(class_scores.max()) if len(class_scores) else 0.0)

        selected = predicted_scores >= metric_conf
        selected_classes = predicted_classes[selected]
        selected_boxes = predicted_boxes[selected]
        selected_scores = predicted_scores[selected]
        order = np.argsort(-selected_scores)
        unmatched_true = set(range(len(true_classes)))
        image_tp = image_fp = 0

        for prediction_index in order:
            same_class_indices = [
                index for index in unmatched_true if true_classes[index] == selected_classes[prediction_index]
            ]
            if not same_class_indices:
                image_fp += 1
                continue

            ious = box_iou_one_to_many(selected_boxes[prediction_index], true_boxes[same_class_indices])
            best_local_index = int(np.argmax(ious))
            if ious[best_local_index] < match_iou:
                image_fp += 1
                continue

            true_index = same_class_indices[best_local_index]
            unmatched_true.remove(true_index)
            image_tp += 1
            matched_true_boxes.append(true_boxes[true_index])
            matched_predicted_boxes.append(selected_boxes[prediction_index])

        image_fn = len(unmatched_true)
        total_tp += image_tp
        total_fp += image_fp
        total_fn += image_fn
        exact_images += int(image_fp == 0 and image_fn == 0)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    detection_accuracy = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn else 0.0

    auc_by_class: dict[str, float] = {}
    figure, axis = plt.subplots(figsize=(8, 6))
    for class_id, class_name in class_names.items():
        fpr, tpr, auc = binary_roc_curve(
            np.asarray(roc_true[class_id], dtype=int), np.asarray(roc_scores[class_id], dtype=float)
        )
        auc_by_class[class_name] = auc
        if len(fpr):
            axis.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], "k--", label="Random")
    axis.set(title="Image-level One-vs-Rest ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(save_dir / "roc_curve.png", dpi=200)
    plt.close(figure)

    finite_auc_values = [value for value in auc_by_class.values() if math.isfinite(value)]
    return {
        "definitions": {
            "confidence_threshold": metric_conf,
            "true_positive_match_iou": match_iou,
            "accuracy": "TP / (TP + FP + FN); object detection has no well-defined true-negative count",
            "image_accuracy": "fraction of images with every GT matched and no false-positive detections",
            "roc_auc": "image-level one-vs-rest class-presence AUC; localization is ignored",
            "bbox_r2": "R2 of normalized xyxy coordinates for correctly matched boxes only",
        },
        "counts": {"images": len(image_paths), "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detection_accuracy": detection_accuracy,
        "image_accuracy": exact_images / len(image_paths),
        "macro_roc_auc": float(np.mean(finite_auc_values)) if finite_auc_values else float("nan"),
        "roc_auc_by_class": auc_by_class,
        "bbox_r2_score": calculate_r2(matched_true_boxes, matched_predicted_boxes),
    }


def run_test(model: YOLO, config: dict[str, Any], common_args: dict[str, Any]) -> None:
    test_args = config.get("test", {})
    if not isinstance(test_args, dict):
        raise ValueError("'test' must be a mapping in yolo_param.yaml.")

    test_args = dict(test_args)
    enabled = test_args.pop("enabled", True)
    metric_conf = float(test_args.pop("metric_conf", 0.5))
    match_iou = float(test_args.pop("match_iou", 0.5))
    roc_min_conf = float(test_args.pop("roc_min_conf", 0.001))
    if not enabled:
        raise ValueError("Set 'test.enabled: true' before running with --mode test.")
    if not 0 <= roc_min_conf <= metric_conf <= 1:
        raise ValueError("Require 0 <= test.roc_min_conf <= test.metric_conf <= 1.")
    if not 0 <= match_iou <= 1:
        raise ValueError("'test.match_iou' must be between 0 and 1.")

    validation_args = {**common_args, "split": "test", **test_args}
    metrics = model.val(**validation_args)
    save_dir = Path(metrics.save_dir)

    dataset_config = yaml.safe_load(Path(common_args["data"]).read_text(encoding="utf-8")) or {}
    raw_names = dataset_config.get("names", {})
    class_names = (
        {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, dict)
        else {index: str(value) for index, value in enumerate(raw_names)}
    )
    image_paths = get_split_image_paths(common_args["data"], "test")
    custom_metrics = calculate_custom_test_metrics(
        model, image_paths, class_names, test_args, metric_conf, match_iou, roc_min_conf, save_dir
    )
    official_metrics = {key: float(value) for key, value in metrics.results_dict.items()}
    report = make_json_safe({"ultralytics": official_metrics, "custom": custom_metrics})

    with (save_dir / "test_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(report, file, allow_unicode=True, sort_keys=False)
    with (save_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (save_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(flatten_metrics(report).items())

    print("\nFinal test metrics")
    for key, value in flatten_metrics(report).items():
        if not key.startswith("custom.definitions"):
            print(f"{key}: {value}")
    print(f"Test results saved to: {save_dir}")


def resolve_inference_source(config_dir: Path, source: str) -> str | int:
    if source.isdecimal():
        return int(source)
    if source.startswith(("http://", "https://", "rtsp://", "rtmp://", "tcp://")):
        return source

    source_path = resolve_local_path(config_dir, source)
    if not source_path.exists():
        raise FileNotFoundError(f"Inference source not found: {source_path}")
    return str(source_path)


def run_predict(
    model: YOLO,
    config: dict[str, Any],
    config_dir: Path,
    project: str,
    source: str | None,
) -> None:
    if source is None:
        raise ValueError("--mode predict requires --source (for example: --source 0 or --source ../video/test.mp4).")

    inference_args = config.get("inference", {})
    if not isinstance(inference_args, dict):
        raise ValueError("'inference' must be a mapping in yolo_param.yaml.")

    prediction_args = {
        "source": resolve_inference_source(config_dir, source),
        "project": project,
        **inference_args,
    }
    results = model.predict(**prediction_args)

    frame_count = 0
    no_detection_count = 0
    detection_counts: Counter[str] = Counter()
    save_dir = None
    for result in results:
        frame_count += 1
        save_dir = result.save_dir
        if result.boxes is None or len(result.boxes) == 0:
            no_detection_count += 1
            continue
        detection_counts.update(model.names[int(class_id)] for class_id in result.boxes.cls.tolist())

    print(f"Processed frames/images: {frame_count}")
    print(f"Frames/images with no detection: {no_detection_count}")
    print(f"Detections by class: {dict(detection_counts)}")
    if save_dir is not None and inference_args.get("save", False):
        print(f"Prediction results saved to: {save_dir}")


def main() -> None:
    args = parse_args()
    config, config_dir = load_config(args.config)
    model_source, common_args = build_common_args(config, config_dir)

    if args.mode == "predict":
        paths = config["paths"]
        trained_model_value = paths.get("trained_model")
        if not trained_model_value:
            raise ValueError("'paths.trained_model' is required for prediction.")
        trained_model = resolve_model_source(config_dir, trained_model_value)
        if not Path(trained_model).is_file():
            raise FileNotFoundError(f"Trained model not found: {trained_model}")
        run_predict(YOLO(trained_model), config, config_dir, common_args["project"], args.source)
    elif args.mode == "test":
        paths = config["paths"]
        trained_model_value = paths.get("trained_model")
        if not trained_model_value:
            raise ValueError("'paths.trained_model' is required for testing.")
        trained_model = resolve_model_source(config_dir, trained_model_value)
        if not Path(trained_model).is_file():
            raise FileNotFoundError(f"Trained model not found: {trained_model}")
        run_test(YOLO(trained_model), config, common_args)
    elif args.mode == "tune":
        model = YOLO(model_source)
        run_tune(model, config, common_args)
    else:
        model = YOLO(model_source)
        run_train(model, config, common_args)


if __name__ == "__main__":
    main()
