"""YOLO 정적 손모양 모델의 클래스 이름과 실제 손모양 대응을 눈으로 확인합니다.

카메라 영상에 예측 클래스·확신도와 3초 유지 진행률만 표시합니다.
서비스 상태를 바꾸거나 단축키를 전송하지 않으며 영상/좌표를 저장하지 않습니다.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 스크립트로 실행하면 sys.path[0]이 programs/라서 프로젝트 루트가 빠집니다.
# 맨 앞에 넣어 같은 이름의 외부 패키지보다 저장소의 service/를 먼저 찾게 합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo" / "v2" / "best.pt"
# 화면 표시는 OpenCV 기본 폰트 제약 때문에 기존 CLI와 같이 영어로 남깁니다.
COLORS = {0: (80, 180, 250), 1: (110, 220, 130), 2: (120, 130, 240)}


def make_parser():
    parser = argparse.ArgumentParser(description="YOLO 클래스 확인 도구 (읽기 전용)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="확인할 YOLO 가중치. 기본값은 yolo/v2/best.pt")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=416, help="추론 입력 크기 (기본 416)")
    parser.add_argument("--conf", type=float, default=0.5, help="표시할 최소 확신도")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto는 CUDA를 쓸 수 있으면 GPU, 아니면 CPU를 사용합니다.")
    parser.add_argument("--hold-seconds", type=float, default=3.0, help="유지 판정 시간 (기본 3초)")
    return parser


def best_detection(result, conf):
    """가장 확신도가 높은 상자 하나만 씁니다. 여러 손을 동시에 추적하지 않습니다."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    top = max(range(len(boxes)), key=lambda i: float(boxes.conf[i]))
    score = float(boxes.conf[top])
    if score < conf:
        return None
    return int(boxes.cls[top]), score, [int(v) for v in boxes.xyxy[top].tolist()]


def main(argv=None):
    args = make_parser().parse_args(argv)
    if not args.weights.is_file():
        raise SystemExit(f"가중치 파일이 없습니다: {args.weights}")
    import cv2
    import torch
    from ultralytics import YOLO

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다. --device cpu로 실행하세요.")
    model = YOLO(str(args.weights))
    model.to(device)
    names = model.names
    if not isinstance(names, dict) or not names:
        raise SystemExit(f"클래스 이름이 없는 가중치입니다: {args.weights}\n"
                         "yolo/v2/best.pt처럼 names가 저장된 파일을 사용하세요.")
    print(f"가중치 : {args.weights}")
    print(f"클래스 : {names}")
    print(f"장치   : {device} / imgsz={args.imgsz} / conf>={args.conf}")
    print("\n손모양을 하나씩 보여주고 화면의 클래스 이름을 적어 두세요. Q 또는 Esc로 종료합니다.\n")

    from service.main import open_camera
    camera = open_camera(cv2, args.camera_index)
    window = "YOLO class check (read-only)"
    held, held_since = None, None
    seen = Counter()
    latencies = []
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise SystemExit("카메라 프레임 읽기 실패")
            started = time.perf_counter()
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                   verbose=False, device=device)[0]
            latencies.append((time.perf_counter() - started) * 1000)
            detection = best_detection(result, args.conf)
            now = time.monotonic()
            if detection is None:
                held, held_since = None, None
                label, line = None, "no detection"
            else:
                index, score, (x1, y1, x2, y2) = detection
                label = names.get(index, str(index))
                seen[label] += 1
                color = COLORS.get(index, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 유지 시간은 같은 클래스가 끊기지 않고 이어질 때만 누적합니다.
                if held != label:
                    held, held_since = label, now
                elapsed = now - held_since
                line = f"{label} {score:.2f}   hold {min(elapsed, args.hold_seconds):.1f}/{args.hold_seconds:.0f}s"
                if elapsed >= args.hold_seconds:
                    line += "   -> HOLD CONFIRMED"
                cv2.putText(frame, f"{label} {score:.2f}", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
            cv2.putText(frame, line, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.putText(frame, f"{latencies[-1]:.0f} ms   Q/Esc: quit", (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 200), 1)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    if latencies:
        ordered = sorted(latencies)
        print(f"\n프레임 {len(latencies)}개 · 추론 중앙값 {ordered[len(ordered)//2]:.1f} ms "
              f"· 최대 {ordered[-1]:.1f} ms")
    print("클래스별 검출 프레임 수:", dict(seen) or "없음")


if __name__ == "__main__":
    raise SystemExit(main())
