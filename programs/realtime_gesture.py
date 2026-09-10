"""학습된 LSTM 제스처 모델을 웹캠 영상에 연속 적용하는 OpenCV 프로그램.

실행 예시::

    uv run python -m programs.realtime_gesture \
        --checkpoint runs/lstm/experiment_001/best.pt

학습 때 사용한 전처리 설정을 체크포인트에서 읽고, 최근 프레임의 sliding
window에 ``training.prepare_dataset.preprocess_sequence``를 그대로 적용한다.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from training.models import LSTMClassifier
from training.prepare_dataset import QualityConfig, preprocess_sequence
from training.train import select_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "lstm" / "experiment_001" / "best.pt"


class FrameWindow:
    """최근 ``max_seconds`` 동안의 원본 landmark 프레임을 보관한다."""

    def __init__(self, max_seconds: float) -> None:
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be a positive finite value")
        self.max_seconds = float(max_seconds)
        self._frames: deque[tuple[float, np.ndarray, bool]] = deque()

    def append(
        self,
        timestamp: float,
        landmarks: np.ndarray,
        detected: bool,
    ) -> None:
        timestamp = float(timestamp)
        points = np.asarray(landmarks, dtype=np.float32)

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if points.shape != (21, 3):
            raise ValueError(f"landmarks must have shape (21, 3): {points.shape}")
        if self._frames and timestamp <= self._frames[-1][0]:
            raise ValueError("timestamps must be strictly increasing")

        self._frames.append((timestamp, points.copy(), bool(detected)))
        cutoff = timestamp - self.max_seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._frames:
            return (
                np.empty((0, 21, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.bool_),
            )

        points = np.stack([item[1] for item in self._frames]).astype(
            np.float32, copy=False
        )
        timestamps = np.asarray(
            [item[0] for item in self._frames], dtype=np.float64
        )
        detected = np.asarray(
            [item[2] for item in self._frames], dtype=np.bool_
        )
        return points, timestamps, detected

    def clear(self) -> None:
        self._frames.clear()


class ProbabilitySmoother:
    """최근 추론 확률의 이동 평균으로 화면 결과의 떨림을 줄인다."""

    def __init__(self, window_size: int, num_classes: int) -> None:
        if window_size < 1 or num_classes < 1:
            raise ValueError("window_size and num_classes must be >= 1")
        self.num_classes = int(num_classes)
        self._values: deque[np.ndarray] = deque(maxlen=int(window_size))

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (self.num_classes,):
            raise ValueError(
                f"probabilities must have shape ({self.num_classes},): {values.shape}"
            )
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.isclose(values.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("probabilities must be finite, non-negative, and sum to 1")

        self._values.append(values.copy())
        return np.mean(np.stack(self._values), axis=0)

    def clear(self) -> None:
        self._values.clear()


@dataclass(frozen=True)
class RuntimeBundle:
    model: nn.Module
    device: torch.device
    class_names: tuple[str, ...]
    seq_len: int
    feature_dim: int
    quality: QualityConfig
    best_val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None


def _ordered_class_names(label_map: object) -> tuple[str, ...]:
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("checkpoint label_map must be a non-empty dictionary")
    if not all(isinstance(name, str) and isinstance(index, int)
               for name, index in label_map.items()):
        raise ValueError("checkpoint label_map must map strings to integers")

    expected = list(range(len(label_map)))
    if sorted(label_map.values()) != expected:
        raise ValueError("checkpoint label ids must be contiguous from zero")
    return tuple(name for name, _ in sorted(label_map.items(), key=lambda item: item[1]))


def _load_test_metrics(checkpoint_path: Path) -> dict[str, float] | None:
    path = checkpoint_path.parent / "test_metrics.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("test")
        if not isinstance(metrics, dict):
            return None
        return {
            key: float(metrics[key])
            for key in ("accuracy", "macro_f1", "loss")
            if key in metrics
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def load_runtime_bundle(
    checkpoint_path: Path | str,
    requested_device: str = "auto",
) -> RuntimeBundle:
    """체크포인트를 읽고 모델과 학습 전처리 계약의 일관성을 검증한다."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    device = select_device(requested_device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")

    required = {
        "model_config", "model_state_dict", "label_map", "preprocessing_config"
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"checkpoint is missing keys: {missing}")

    model_config = checkpoint["model_config"]
    preprocessing = checkpoint["preprocessing_config"]
    label_map = checkpoint["label_map"]
    if not isinstance(model_config, dict) or not isinstance(preprocessing, dict):
        raise ValueError("model_config and preprocessing_config must be dictionaries")

    class_names = _ordered_class_names(label_map)
    if preprocessing.get("label_map") != label_map:
        raise ValueError("checkpoint and preprocessing label_map values differ")

    seq_len = preprocessing.get("seq_len")
    feature_dim = preprocessing.get("feature_dim")
    if not isinstance(seq_len, int) or seq_len < 2:
        raise ValueError(f"invalid seq_len: {seq_len}")
    if not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError(f"invalid feature_dim: {feature_dim}")
    if model_config.get("input_size") != feature_dim:
        raise ValueError(
            "model input_size and preprocessing feature_dim do not match: "
            f"{model_config.get('input_size')} != {feature_dim}"
        )
    if model_config.get("num_classes") != len(class_names):
        raise ValueError("model num_classes and label_map size do not match")

    quality_values = preprocessing.get("quality")
    if not isinstance(quality_values, dict):
        raise ValueError("preprocessing_config.quality must be a dictionary")
    try:
        quality = QualityConfig(**quality_values)
    except TypeError as error:
        raise ValueError(f"invalid quality configuration: {error}") from error
    quality.validate()

    model = LSTMClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_val = checkpoint.get("best_val_metrics", {})
    if not isinstance(best_val, dict):
        best_val = {}

    return RuntimeBundle(
        model=model,
        device=device,
        class_names=class_names,
        seq_len=seq_len,
        feature_dim=feature_dim,
        quality=quality,
        best_val_metrics={
            key: float(best_val[key])
            for key in ("accuracy", "macro_f1", "loss")
            if key in best_val
        },
        test_metrics=_load_test_metrics(path),
    )


@torch.inference_mode()
def predict_probabilities(
    bundle: RuntimeBundle,
    features: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    expected = (bundle.seq_len, bundle.feature_dim)
    if values.shape != expected:
        raise ValueError(f"features must have shape {expected}: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("features contain NaN or Infinity")

    inputs = torch.from_numpy(values).unsqueeze(0).to(bundle.device)
    logits = bundle.model(inputs)
    return logits.softmax(dim=1)[0].detach().cpu().numpy()


def _put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def draw_dashboard(
    camera_frame: np.ndarray,
    bundle: RuntimeBundle,
    probabilities: np.ndarray | None,
    recognition: str,
    status: str,
    handedness: str,
    detection_rate: float,
    buffer_seconds: float,
    fps: float,
    inference_ms: float,
    threshold: float,
) -> np.ndarray:
    import cv2

    height, width = camera_frame.shape[:2]
    panel_width = 390
    canvas_height = max(height, 620)
    canvas = np.zeros((canvas_height, width + panel_width, 3), dtype=np.uint8)
    canvas[:height, :width] = camera_frame
    if canvas_height > height:
        canvas[height:, :width] = (18, 19, 22)
    panel_x = width
    canvas[:, panel_x:] = (30, 32, 38)

    _put_text(canvas, "CONTINUOUS GESTURE", panel_x + 20, 35, 0.7,
              (255, 210, 80), 2)
    _put_text(canvas, "Sliding-window LSTM", panel_x + 20, 61, 0.47,
              (175, 180, 190))

    label_color = (80, 220, 120) if probabilities is not None else (80, 190, 255)
    _put_text(canvas, "Recognition", panel_x + 20, 102, 0.48,
              (160, 165, 175))
    _put_text(canvas, recognition, panel_x + 20, 137, 0.82, label_color, 2)
    _put_text(canvas, status[:48], panel_x + 20, 164, 0.43,
              (190, 195, 205))

    y = 205
    shown = probabilities if probabilities is not None else np.zeros(len(bundle.class_names))
    for name, score in zip(bundle.class_names, shown):
        _put_text(canvas, name, panel_x + 20, y, 0.48)
        _put_text(canvas, f"{score * 100:5.1f}%", panel_x + 300, y, 0.48)
        bar_y = y + 9
        cv2.rectangle(canvas, (panel_x + 20, bar_y),
                      (panel_x + 350, bar_y + 13), (70, 73, 82), -1)
        filled = int(330 * float(np.clip(score, 0.0, 1.0)))
        cv2.rectangle(canvas, (panel_x + 20, bar_y),
                      (panel_x + 20 + filled, bar_y + 13), (70, 185, 245), -1)
        y += 50

    _put_text(canvas, "Scores are confidence, not measured accuracy.",
              panel_x + 20, y + 2, 0.38, (145, 150, 160))
    y += 35

    metrics = bundle.test_metrics
    if metrics and "accuracy" in metrics and "macro_f1" in metrics:
        _put_text(canvas, "OFFLINE HELD-OUT TEST", panel_x + 20, y, 0.46,
                  (255, 210, 80), 1)
        y += 25
        _put_text(canvas, f"Accuracy {metrics['accuracy'] * 100:.1f}%   "
                  f"Macro F1 {metrics['macro_f1'] * 100:.1f}%",
                  panel_x + 20, y, 0.45)
        y += 31

    _put_text(canvas, f"Hand: {handedness or 'not detected'}", panel_x + 20, y, 0.45)
    _put_text(canvas, f"Valid frames: {detection_rate * 100:.1f}%", panel_x + 20, y + 23, 0.45)
    _put_text(canvas, f"Buffer: {buffer_seconds:.2f}s  Threshold: {threshold:.2f}",
              panel_x + 20, y + 46, 0.45)
    _put_text(canvas, f"FPS: {fps:.1f}  Inference: {inference_ms:.1f}ms",
              panel_x + 20, y + 69, 0.45)
    _put_text(canvas, f"Device: {bundle.device}", panel_x + 20, y + 92, 0.45)
    _put_text(canvas, "Q/ESC quit   R reset", panel_x + 20, canvas_height - 20,
              0.46, (175, 180, 190))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continuous webcam recognition with a trained LSTM model."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=1.5)
    parser.add_argument("--inference-interval", type=int, default=3,
                        help="Run inference every N camera frames.")
    parser.add_argument("--smoothing", type=int, default=5,
                        help="Number of recent predictions to average.")
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mirror-display", action="store_true",
                        help="Mirror only the displayed camera image, not model input.")
    return parser.parse_args()


def _validate_runtime_args(args: argparse.Namespace, quality: QualityConfig) -> None:
    if not quality.min_duration <= args.window_seconds <= quality.max_duration:
        raise ValueError(
            "window-seconds must be inside the training quality duration range "
            f"[{quality.min_duration}, {quality.max_duration}]"
        )
    if args.inference_interval < 1 or args.smoothing < 1:
        raise ValueError("inference-interval and smoothing must be >= 1")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold must be in [0, 1]")


def main() -> int:
    args = parse_args()
    bundle = load_runtime_bundle(args.checkpoint, args.device)
    _validate_runtime_args(args, bundle.quality)

    import cv2
    import mediapipe as mp

    from programs.collect_gesture import (
        draw_hand_landmarks,
        extract_landmarks,
        initialize_hand_detector,
        open_camera,
    )

    frame_window = FrameWindow(args.window_seconds)
    smoother = ProbabilitySmoother(args.smoothing, len(bundle.class_names))
    detector = initialize_hand_detector()
    capture = open_camera(args.camera)

    probabilities: np.ndarray | None = None
    recognition = "WARMING UP"
    status = "Collecting a valid Right-hand window"
    inference_ms = 0.0
    fps = 0.0
    frame_index = 0
    last_frame_time = time.perf_counter()
    start_time = last_frame_time
    last_media_timestamp_ms = -1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            now = time.perf_counter()
            elapsed = max(now - last_frame_time, 1e-9)
            instant_fps = 1.0 / elapsed
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            last_frame_time = now

            timestamp_ms = max(
                last_media_timestamp_ms + 1,
                int((now - start_time) * 1000),
            )
            last_media_timestamp_ms = timestamp_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, timestamp_ms)
            landmarks, _, detected, handedness, _, selected = extract_landmarks(result)

            right_hand = bool(detected and handedness == "Right")
            if not right_hand:
                landmarks = np.full((21, 3), np.nan, dtype=np.float32)
            frame_window.append(timestamp_ms / 1000.0, landmarks, right_hand)
            draw_hand_landmarks(frame, selected)

            points, timestamps, detected_mask = frame_window.arrays()
            duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
            detection_rate = float(detected_mask.mean()) if len(detected_mask) else 0.0

            if frame_index % args.inference_interval == 0:
                started = time.perf_counter()
                try:
                    features, _ = preprocess_sequence(
                        points,
                        timestamps,
                        detected_mask,
                        seq_len=bundle.seq_len,
                        config=bundle.quality,
                    )
                    raw_probabilities = predict_probabilities(bundle, features)
                    probabilities = smoother.update(raw_probabilities)
                    best_index = int(np.argmax(probabilities))
                    best_score = float(probabilities[best_index])
                    if best_score >= args.confidence_threshold:
                        recognition = bundle.class_names[best_index]
                        status = f"Stable score {best_score * 100:.1f}%"
                    else:
                        recognition = "UNCERTAIN"
                        status = f"Top score {best_score * 100:.1f}% is below threshold"
                except ValueError as error:
                    probabilities = None
                    smoother.clear()
                    recognition = "WARMING UP" if duration < bundle.quality.min_duration else "NO VALID WINDOW"
                    status = str(error)
                inference_ms = (time.perf_counter() - started) * 1000.0

            display_frame = cv2.flip(frame, 1) if args.mirror_display else frame
            dashboard = draw_dashboard(
                display_frame,
                bundle,
                probabilities,
                recognition,
                status,
                handedness,
                detection_rate,
                duration,
                fps,
                inference_ms,
                args.confidence_threshold,
            )
            cv2.imshow("Continuous Gesture Recognition", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                frame_window.clear()
                smoother.clear()
                probabilities = None
                recognition = "WARMING UP"
                status = "Buffer reset"

            frame_index += 1
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
