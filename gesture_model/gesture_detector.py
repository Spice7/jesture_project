from dataclasses import dataclass
from pathlib import Path
import time

from ultralytics import YOLO

VALID_GESTURES = {"start", "stop", "cancel"}


@dataclass
class GestureResult:
    gesture: str | None
    confidence: float
    hold_time: float
    confirmed: bool

class GestureDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.7,      # 신뢰도 임계값    # 예측 confidence가 이 수치 이상이어야 제스처 후보로 인정함
        hold_seconds: float = 3.0,              # 동일한 제스처를 이 수치 이상 유지해야 해당 제스처로 간주함
        imgsz: int = 640,                       # yolo추론 시 사용할 이미지 입력 크기   # 높이면 멀리 있는 것도 인식 할 수 있지만 추론속도와 GPU 사용량 늘어남
        device: int | str = 0,                  # 첫 번째 gpu 사용
    ):
        self.model = YOLO(str(model_path))

        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.imgsz = imgsz
        self.device = device

        self.current_gesture: str | None = None
        self.gesture_start_time: float | None = None

        # 동일 제스처가 계속 유지될 때
        # 한 번만 confirmed=True를 반환하기 위한 값
        self.triggered = False

    def process(self, frame) -> GestureResult:
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        detected_gesture = None
        detected_confidence = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            # 여러 객체가 검출되었다면
            # confidence가 가장 높은 결과 사용
            best_index = max(
                range(len(confidences)),
                key=lambda i: confidences[i],
            )

            class_id = class_ids[best_index]
            class_name = result.names[class_id]

            if class_name in VALID_GESTURES:
                detected_gesture = class_name
                detected_confidence = confidences[best_index]

        return self._update_state(
            detected_gesture,
            detected_confidence,
        )

    def _update_state(
        self,
        detected_gesture: str | None,
        confidence: float,
    ) -> GestureResult:
        now = time.monotonic()

        # 아무것도 검출되지 않은 경우
        if detected_gesture is None:
            self.reset()

            return GestureResult(
                gesture=None,
                confidence=0.0,
                hold_time=0.0,
                confirmed=False,
            )

        # 새로운 제스처가 들어온 경우
        if detected_gesture != self.current_gesture:
            self.current_gesture = detected_gesture
            self.gesture_start_time = now
            self.triggered = False

            return GestureResult(
                gesture=detected_gesture,
                confidence=confidence,
                hold_time=0.0,
                confirmed=False,
            )

        # 기존 제스처가 계속 유지되는 경우
        hold_time = now - self.gesture_start_time

        confirmed = False

        if (
            hold_time >= self.hold_seconds
            and not self.triggered
        ):
            confirmed = True
            self.triggered = True

        return GestureResult(
            gesture=detected_gesture,
            confidence=confidence,
            hold_time=hold_time,
            confirmed=confirmed,
        )

    def reset(self):
        self.current_gesture = None
        self.gesture_start_time = None
        self.triggered = False

    def set_confidence_threshold(self, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "confidence_threshold는 0.0 ~ 1.0 사이여야 합니다."
            )

        self.confidence_threshold = value

    def set_hold_seconds(self, seconds: float):
        if seconds < 0:
            raise ValueError(
                "hold_seconds는 0 이상이어야 합니다."
            )

        self.hold_seconds = seconds