from pathlib import Path
import time

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n_v2_null"

# ============================================================
# 설정
# ============================================================

# 같은 제스처가 몇 초 동안 유지되어야 확정할 것인지
RECOGNITION_HOLD_SECONDS = 3.0

# YOLO 검출 신뢰도 기준
# 기초 설정 0.7
CONFIDENCE_THRESHOLD = 0.7

# YOLO 입력 이미지 크기
# 기본 640
IMAGE_SIZE = 640

# 카메라 번호
# 기본 웹캠: 0
# 스마트폰 웹캠 등이면 1, 2 등을 시도
CAMERA_INDEX = 0

# GPU 사용
DEVICE = 0

# 우리가 허용하는 클래스
VALID_GESTURES = {"start", "stop", "cancel"}


def main():
    project_root = Path(__file__).resolve().parent.parent

    model_path = (
        project_root
        / "runs"
        / "gesture"
        / MODEL_NAME
        / "weights"
        / "best.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"best.pt를 찾을 수 없습니다.\n"
            f"확인 경로: {model_path}"
        )

    # 학습된 모델 로드
    model = YOLO(str(model_path))

    # 웹캠 연결
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"카메라를 열 수 없습니다. CAMERA_INDEX={CAMERA_INDEX}"
        )

    # 현재 연속으로 인식 중인 제스처
    current_gesture = None

    # 해당 제스처가 처음 인식된 시간
    gesture_start_time = None

    # 이미 action이 확정되었는지
    action_triggered = False

    print("Webcam gesture test started.")
    print("Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("카메라 프레임을 가져오지 못했습니다.")
                break

            # ------------------------------------------------
            # YOLO 추론
            # ------------------------------------------------
            result = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )[0]

            # YOLO가 그린 Bounding Box가 포함된 프레임
            display_frame = result.plot()

            detected_gesture = None
            detected_confidence = 0.0

            # ------------------------------------------------
            # 여러 검출 결과 중 confidence가 가장 높은 것 선택
            # ------------------------------------------------
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                best_index = max(
                    range(len(confidences)),
                    key=lambda i: confidences[i],
                )

                class_id = class_ids[best_index]
                confidence = confidences[best_index]

                class_name = result.names[class_id]

                if class_name in VALID_GESTURES:
                    detected_gesture = class_name
                    detected_confidence = confidence

            now = time.monotonic()

            # ------------------------------------------------
            # 연속 인식 시간 계산
            # ------------------------------------------------
            if detected_gesture is None:
                # 아무 제스처도 검출되지 않음
                current_gesture = None
                gesture_start_time = None
                action_triggered = False

            elif detected_gesture != current_gesture:
                # 기존과 다른 제스처가 들어옴
                current_gesture = detected_gesture
                gesture_start_time = now
                action_triggered = False

            else:
                # 같은 제스처가 계속 유지되고 있음
                elapsed_time = now - gesture_start_time

                if (
                    elapsed_time >= RECOGNITION_HOLD_SECONDS
                    and not action_triggered
                ):
                    action_triggered = True

                    print(
                        f"[ACTION] {current_gesture.upper()} "
                        f"({detected_confidence:.2f})"
                    )

            # ------------------------------------------------
            # 화면 출력 정보
            # ------------------------------------------------
            if current_gesture is not None:
                elapsed_time = now - gesture_start_time

                cv2.putText(
                    display_frame,
                    f"Gesture: {current_gesture}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    display_frame,
                    (
                        f"Hold: "
                        f"{min(elapsed_time, RECOGNITION_HOLD_SECONDS):.1f}"
                        f" / {RECOGNITION_HOLD_SECONDS:.1f} sec"
                    ),
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"Confidence: {detected_confidence:.2f}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # 3초 이상 유지되었을 때
                if action_triggered:
                    action_text = (
                        f"ACTION: {current_gesture.upper()}"
                    )

                    cv2.putText(
                        display_frame,
                        action_text,
                        (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                    )

            else:
                cv2.putText(
                    display_frame,
                    "Gesture: None",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(
                "YOLO Gesture Test",
                display_frame,
            )

            # Q를 누르면 종료
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()