"""키보드 ON/OFF 서비스 진입점. 실제 단축키 전송 없이 확정 이벤트만 출력합니다."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time

# -m service.main을 권장하며 python service/main.py 직접 실행도 지원합니다.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.actions import dispatch
from service.gesture_controller import GestureController, ServiceConfig
from service.hand_tracker import HandTracker, missing_hand
from service.predictor import GesturePredictor

WINDOW_NAME = "Gesture service - DRY RUN"


def open_camera(cv, index):
    """Windows DirectShow 우선, 실패하면 기본 backend. 실패한 핸들도 반드시 해제합니다."""
    camera = cv.VideoCapture(index, cv.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"카메라 {index}를 열 수 없습니다.")
    return camera


def draw_status(cv, frame, controller, hand, now):
    """카메라 원본 좌표는 유지합니다. UI 표시는 추론 입력으로 다시 들어가지 않습니다."""
    if hand.detected:
        height, width = frame.shape[:2]
        for point in hand.landmarks:
            x, y = int(point[0] * width), int(point[1] * height)
            if 0 <= x < width and 0 <= y < height:
                cv.circle(frame, (x, y), 3, (0, 255, 0), -1)
    lines = [f"Hand: {hand.status}", *controller.status_lines(now)]
    for index, text in enumerate(lines):
        position = (10, 25 + index * 25)
        cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3)
        cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)


def run_camera_loop(controller, camera_index=0, *, cv=None, tracker_factory=None,
                    clock=time.monotonic, action=dispatch):
    """단일 루프입니다. 테스트에서는 cv/tracker/clock/action을 모의 객체로 주입합니다."""
    if cv is None:
        import cv2 as cv
    # 이후 초기화 중 오류가 발생해도 등록된 자원을 역순으로 모두 해제합니다.
    with ExitStack() as resources:
        resources.callback(controller.stop)
        resources.callback(cv.destroyAllWindows)
        tracker = tracker_factory() if tracker_factory is not None else HandTracker()
        resources.callback(tracker.close)
        camera = open_camera(cv, camera_index)
        resources.callback(camera.release)
        cv.namedWindow(WINDOW_NAME)
        while True:
            ok, frame = camera.read()
            timestamp = clock()  # 프레임 획득 직후의 monotonic 시각입니다.
            if not ok or frame is None:
                raise RuntimeError("카메라 프레임 읽기에 실패했습니다.")
            key = cv.waitKey(1) & 0xFF
            if key == 27 or cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                controller.start()
            elif key in (ord("x"), ord("X")):
                controller.stop()
            hand = missing_hand("Tracking paused (OFF)")
            if controller.enabled:
                hand = tracker.process(frame, timestamp)
                event = controller.step(timestamp, hand.landmarks, hand.detected, hand.handedness)
                # 오래 걸린 동기 추론 결과를 뒤늦게 실행하지 않습니다. 다음 구간은 중립부터 확인합니다.
                if clock() - timestamp >= controller.config.reset_gap_seconds:
                    controller.reset_after_gap()
                elif event is not None and controller.enabled:
                    action(event)
            draw_status(cv, frame, controller, hand, clock())
            cv.imshow(WINDOW_NAME, frame)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    defaults = ServiceConfig()
    for name in defaults.__dataclass_fields__:
        parser.add_argument("--" + name.replace("_", "-"), type=int if name in
                            ("stable_predictions", "rearm_predictions") else float, default=getattr(defaults, name))
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        config = ServiceConfig(**{name: getattr(args, name) for name in ServiceConfig.__dataclass_fields__})
        config.validate()
        if args.camera_index < 0:
            raise ValueError("camera-index는 0 이상이어야 합니다.")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"체크포인트 파일 누락: {args.checkpoint}")
        predictor = GesturePredictor(args.checkpoint, args.device)
        controller = GestureController(predictor, config)
        print(f"DRY RUN ONLY / device={predictor.device} / seq_len={predictor.seq_len}")
        run_camera_loop(controller, args.camera_index)
        return 0
    except KeyboardInterrupt:
        print("사용자가 서비스를 중단했습니다.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"서비스 오류 ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
