"""MediaPipe Tasks API(HandLandmarker) 래퍼. mediapipe>=1.0 은 mp.solutions 가 없다."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from . import config

# 이 모듈은 실시간 추론 루프용이다. 데이터 수집은 programs/collect_gesture.py 가 담당한다.

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model():
    if not config.HAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"MediaPipe 모델을 복사하세요: {config.HAND_MODEL_PATH}")


class HandTracker:
    """VIDEO 모드 HandLandmarker. 타임스탬프는 객체 수명 전체에서 단조 증가해야 하므로
    내부에서 보정한다. process()의 ts_ms가 None이거나 역행하면 자동으로 +1 처리."""

    def __init__(self, num_hands=1, det_conf=0.5, presence_conf=0.5, track_conf=0.5):
        ensure_model()
        opts = vision.HandLandmarkerOptions(
            # delegate=CPU 명시: Windows/macOS 어디서나 같은 경로로 돌게 한다. 이 모델은 CPU로 실시간 충분.
            # 참고: mediapipe 1.0.1 은 Apple Silicon 맥에서 여기서 크래시한다 (Windows 는 정상). 맥에서 쓰려면 1.0.0.
            base_options=BaseOptions(model_asset_path=str(config.HAND_MODEL_PATH),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, bgr: np.ndarray, ts_ms: int | None = None):
        """반환 (landmarks (21,3) float32 또는 None, handedness "Left"/"Right"/"")
        bgr 은 이미 미러 처리된 프레임이어야 한다 (config.MIRROR 규약)."""
        if ts_ms is None or ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(img, int(ts_ms))
        if not res.hand_landmarks:
            return None, ""
        lm = np.array([(p.x, p.y, p.z) for p in res.hand_landmarks[0]], dtype=np.float32)
        hand = res.handedness[0][0].category_name if res.handedness else ""
        return lm, hand

    def close(self):
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def draw_landmarks(bgr: np.ndarray, lm: np.ndarray, color=(0, 255, 0)):
    h, w = bgr.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(bgr, p, 3, (0, 0, 255), -1)
