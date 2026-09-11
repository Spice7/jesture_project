"""정적 포즈 게이트용 YOLO 검출기 (원본: 팀 lkh, 구 `gesture_model/gesture_detector.py`).

동적 GRU 파이프라인(같은 `gesture/` 패키지의 model·pipeline·segmenter 등)과 역할이 다르다.
이쪽은 손바닥/주먹/취소 포즈로 인식을 켜고 끄는 정적 게이트(YOLOv8)이며, `gesture/gate.py`가
이 GestureDetector를 얇게 감싼다. ultralytics(YOLO)는 무거우므로 이 모듈은 게이트가 실제로
필요할 때만 지연 import 된다 — 그래서 `from gesture import config` 같은 경량 사용은 영향받지 않는다.
"""
from dataclasses import dataclass
from pathlib import Path
import time

from ultralytics import YOLO


VALID_GESTURES = {"start", "stop", "cancel"}

@dataclass
class GestureResult:
    """
    한 프레임에 대한 제스처 인식 결과
    """

    gesture: str | None
    confidence: float
    hold_time: float
    confirmed: bool

class GestureDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.7,
        hold_seconds: float = 3.0,
        miss_tolerance_seconds: float = 0.3,
        imgsz: int = 640,
        device: int | str | None = None,
        iou: float = 0.5,
        max_det: int = 10,
    ):
        """
        Parameters
        ----------
        model_path
            YOLO best.pt 모델 경로

        confidence_threshold
            신뢰도 임계값.
            예측 confidence가 이 값 이상인 경우에만
            제스처 후보로 인정한다.

        hold_seconds
            동일한 제스처가 이 시간(초) 이상 유지되어야
            최종 제스처로 확정한다.

        miss_tolerance_seconds
            순간적으로 제스처 검출에 실패하더라도
            이 시간(초) 이내라면 기존 제스처가
            계속 유지되고 있는 것으로 판단한다.

        imgsz
            YOLO 추론 시 사용할 입력 이미지 크기.
            값을 키우면 작은 손 검출에 도움이 될 수 있지만
            추론 속도와 GPU 메모리 사용량이 증가한다.

        device
            추론에 사용할 장치.
            0       : 첫 번째 GPU
            1       : 두 번째 GPU
            "cpu"   : CPU
        """

        self._validate_confidence_threshold(confidence_threshold)
        self._validate_hold_seconds(hold_seconds)
        self._validate_miss_tolerance_seconds(
            miss_tolerance_seconds
        )

        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"정적 제스처 모델을 복사하세요: {model_path}")
        self.model = YOLO(str(model_path))
        if not VALID_GESTURES.issubset(set(self.model.names.values())):
            raise ValueError(f"정적 모델에 start/stop/cancel 클래스가 필요합니다: {model_path}")
        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        self.iou = iou
        self.max_det = max_det
        self.last_prediction = None

        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.miss_tolerance_seconds = miss_tolerance_seconds
        self.imgsz = imgsz
        self.device = device

        # 현재 유지 중인 제스처
        self.current_gesture: str | None = None

        # 현재 제스처가 처음 인식된 시간
        self.gesture_start_time: float | None = None

        # 마지막으로 정상적으로 검출된 시간
        self.last_detected_time: float | None = None

        # 최근 정상 검출 confidence
        self.last_confidence: float = 0.0

        # 현재 제스처가 이미 confirmed 되었는지
        self.triggered: bool = False

    def process(self, frame) -> GestureResult:
        """
        OpenCV frame 하나를 입력받아 제스처를 판별한다.

        Parameters
        ----------
        frame
            cv2.VideoCapture 등으로 얻은 이미지 프레임

        Returns
        -------
        GestureResult
            gesture
                현재 인식 중인 제스처
                "start", "stop", "cancel", None

            confidence
                현재 제스처의 confidence

            hold_time
                현재 제스처가 유지된 시간(초)

            confirmed
                hold_seconds 조건을 충족한 순간에만 True
        """

        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            device=self.device,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )[0]
        self.last_prediction = result

        detected_gesture = None
        detected_confidence = 0.0

        # --------------------------------------------------
        # 검출된 객체가 존재하는 경우
        # --------------------------------------------------
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            # 여러 손/객체가 검출되었다면
            # confidence가 가장 높은 결과 하나를 사용
            best_index = max(
                range(len(confidences)),
                key=lambda i: confidences[i],
            )

            class_id = class_ids[best_index]
            class_name = result.names[class_id]
            confidence = confidences[best_index]

            if class_name in VALID_GESTURES:
                detected_gesture = class_name
                detected_confidence = confidence

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

        # ==================================================
        # 1. 이번 프레임에서 제스처를 검출하지 못함
        # ==================================================
        if detected_gesture is None:

            # 이전에 유지 중이던 제스처가 존재한다면
            if (
                self.current_gesture is not None
                and self.last_detected_time is not None
                and self.gesture_start_time is not None
            ):
                missing_time = (
                    now - self.last_detected_time
                )

                # ------------------------------------------
                # 허용된 시간 안의 순간 미검출
                # ------------------------------------------
                if (
                    missing_time
                    <= self.miss_tolerance_seconds
                ):
                    hold_time = (
                        now - self.gesture_start_time
                    )

                    # 순간 미검출이어도 현재 제스처 상태는
                    # 유지한다.
                    return GestureResult(
                        gesture=self.current_gesture,
                        confidence=self.last_confidence,
                        hold_time=hold_time,
                        confirmed=False,
                    )

            # 허용 시간을 초과했거나
            # 기존에 인식 중인 제스처가 없으면 초기화
            self.reset()

            return GestureResult(
                gesture=None,
                confidence=0.0,
                hold_time=0.0,
                confirmed=False,
            )

        # ==================================================
        # 2. 이전과 다른 새로운 제스처가 검출됨
        # ==================================================
        if detected_gesture != self.current_gesture:
            self.current_gesture = detected_gesture
            self.gesture_start_time = now
            self.last_detected_time = now
            self.last_confidence = confidence
            self.triggered = False

            return GestureResult(
                gesture=detected_gesture,
                confidence=confidence,
                hold_time=0.0,
                confirmed=False,
            )

        # ==================================================
        # 3. 이전과 같은 제스처가 계속 검출됨
        # ==================================================
        self.last_detected_time = now
        self.last_confidence = confidence

        if self.gesture_start_time is None:
            self.gesture_start_time = now

        hold_time = now - self.gesture_start_time

        confirmed = False

        # 설정한 시간 이상 유지되었고
        # 아직 실행되지 않았다면 한 번만 확정
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
        """
        현재 제스처 인식 상태를 완전히 초기화한다.
        """

        self.current_gesture = None
        self.gesture_start_time = None
        self.last_detected_time = None
        self.last_confidence = 0.0
        self.triggered = False

    # ======================================================
    # 설정 변경
    # ======================================================

    def set_confidence_threshold(
        self,
        value: float,
    ):
        """
        Confidence Threshold를 변경한다.
        """

        self._validate_confidence_threshold(value)
        self.confidence_threshold = value

    def set_hold_seconds(
        self,
        seconds: float,
    ):
        """
        제스처 확정까지 필요한 유지 시간을 변경한다.
        """

        self._validate_hold_seconds(seconds)
        self.hold_seconds = seconds

    def set_miss_tolerance_seconds(
        self,
        seconds: float,
    ):
        """
        순간 미검출을 허용할 시간을 변경한다.
        """

        self._validate_miss_tolerance_seconds(
            seconds
        )

        self.miss_tolerance_seconds = seconds

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def _validate_confidence_threshold(
        value: float,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "confidence_threshold는 "
                "0.0 ~ 1.0 사이여야 합니다."
            )

    @staticmethod
    def _validate_hold_seconds(
        seconds: float,
    ):
        if seconds < 0:
            raise ValueError(
                "hold_seconds는 0 이상이어야 합니다."
            )

    @staticmethod
    def _validate_miss_tolerance_seconds(
        seconds: float,
    ):
        if seconds < 0:
            raise ValueError(
                "miss_tolerance_seconds는 "
                "0 이상이어야 합니다."
            )
