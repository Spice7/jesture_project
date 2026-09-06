"""YOLOv8 object detection training, tuning, and inference entry point."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

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
        choices=("train", "tune", "predict"),
        default="train",
        help="Run training, hyperparameter tuning, or inference.",
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
    elif args.mode == "tune":
        model = YOLO(model_source)
        run_tune(model, config, common_args)
    else:
        model = YOLO(model_source)
        run_train(model, config, common_args)


if __name__ == "__main__":
    main()
