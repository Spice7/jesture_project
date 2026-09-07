"""Run webcam gesture detection with ckpoint/best.pt."""

import argparse
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "ckpoint" / "best.pt"
WINDOW_NAME = "Gesture detection - Q / Esc to quit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.70, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="Inference device: 0, cpu (default: auto)")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        parser.error(f"Model not found: {weights}. Copy your trained best.pt here or use --weights.")

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))
    print(f"Model: {weights}\nDevice: {device}\nPress Q or Esc in the video window to quit.")
    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}. Check camera permissions or try --camera 1.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            started = time.perf_counter()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Cannot read a frame from the camera. Check its connection.")
            result = model.predict(
                source=frame, device=device, conf=args.conf, iou=0.5,
                imgsz=640, max_det=10, verbose=False,
            )[0]
            annotated = result.plot()
            fps = 1.0 / max(time.perf_counter() - started, 1e-6)
            cv2.putText(annotated, f"FPS: {fps:.1f} | Q / Esc: quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if result.boxes is None or len(result.boxes) == 0:
                cv2.putText(annotated, "No detection", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
