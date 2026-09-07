"""수집기와 동일한 원본 좌표 및 MediaPipe VIDEO 모드. 카메라 제어는 main의 역할입니다."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from .policy import SERVICE_HAND
from .resources import resource_path

DEFAULT_MODEL = resource_path("models", "hand_landmarker.task")


@dataclass
class HandFrame:
    landmarks: np.ndarray
    detected: bool
    status: str
    handedness: str = ""


def missing_hand(status="No hand", handedness=""):
    return HandFrame(np.full((21, 3), np.nan, dtype=np.float32), False, status, handedness)


class HandTracker:
    """오른손 하나만 지원합니다. handedness를 뒤집거나 영상을 반전하지 않습니다."""

    def __init__(self, model_path=DEFAULT_MODEL, allowed_hands=("Right",)):
        if tuple(allowed_hands) != (SERVICE_HAND,):
            raise ValueError("현재 서비스는 오른손(Right)만 허용합니다.")
        self.allowed_hands = tuple(allowed_hands)
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe 모델 파일 누락: {path}")
        # 모듈 import만으로 MediaPipe 런타임이나 detector를 초기화하지 않습니다.
        import cv2
        import mediapipe as mp
        self.cv2, self.mp = cv2, mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
            min_tracking_confidence=.5)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.origin = None
        self.last_timestamp_ms = -1

    def process(self, frame, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError("프레임 시각은 유한해야 합니다.")
        if self.origin is None:
            self.origin = timestamp
        # 같은 밀리초 안의 프레임도 MediaPipe timestamp는 반드시 증가시킵니다.
        timestamp_ms = max(int((timestamp - self.origin) * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return missing_hand()
        if len(result.hand_landmarks) != 1:
            return missing_hand("Multiple hands unsupported")
        handedness = result.handedness
        if not handedness or not handedness[0]:
            return missing_hand("Unknown handedness")
        # 오른손으로 판정한 좌표만 제스처 분류에 전달합니다.
        name = handedness[0][0].category_name
        if name != SERVICE_HAND:
            # Left였다는 정보만 전달해 controller가 기존 오른손 구간을 즉시 폐기하게 합니다.
            return missing_hand(f"Expected Right, got {name} (unsupported)", name or "")
        points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return missing_hand("Invalid landmarks")
        return HandFrame(points, True, f"{name} detected", name)

    def close(self):
        if self.detector is not None:
            detector, self.detector = self.detector, None
            detector.close()
