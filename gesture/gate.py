"""YOLO 정적 포즈 게이트: 팀(lkh) 의 gesture_model.GestureDetector 를 우리 파이프라인에 잇는 얇은 층.

포즈(클래스) 3개, 전부 "3초 유지" 가 트리거:
  start  (손바닥)  → 게이트 열림  = 동적 제스처 인식 시작
  stop   (주먹)    → 게이트 닫힘  = 인식 끝. 취소 없음
  cancel (세 번째) → 정적 명령 하나 (gestures.json 의 "cancel" 에 매핑된 키). 게이트가 열려 있을 때만

GestureDetector 는 팀 코드 그대로 쓴다(gesture_model/gesture_detector.py, 브랜치 lkh). 여기서는
  - GPU 없으면 CPU 로, CPU 면 프레임을 솎아 부하를 줄인다
  - 프레임마다 (포즈, 유지 시간, 확정 이벤트) 를 돌려준다
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import config

POSE_NAMES = {"start": "손바닥 (시작)", "stop": "주먹 (끝)", "cancel": "세 번째 포즈"}


@dataclass
class GateStatus:
    pose: str | None          # 지금 보이는 포즈 (start/stop/cancel) 또는 None
    confidence: float
    hold: float               # 같은 포즈가 유지된 시간 (초)
    hold_needed: float        # 트리거까지 필요한 유지 시간
    confirmed: str | None     # 이 프레임에 트리거된 포즈 (한 번만)


class StaticGate:
    def __init__(self, model_path: str | Path | None = None, confidence: float = 0.5, hold_seconds: float = 3.0,
                 imgsz: int = 640, device: int | str | None = None,
                 miss_tolerance_seconds: float = 0.3, iou: float = 0.5, max_det: int = 10):
        import torch  # 무거운 import 는 필요할 때
        from gesture_model.gesture_detector import GestureDetector

        path = Path(model_path) if model_path else config.STATIC_MODEL_PATH
        if not path.is_absolute():
            path = config.ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"정적 제스처 모델을 복사하세요: {path}")
        self.path = path
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.on_gpu = device != "cpu"
        self.det = GestureDetector(path, confidence_threshold=confidence, hold_seconds=hold_seconds,
                                   imgsz=imgsz, device=device, miss_tolerance_seconds=miss_tolerance_seconds,
                                   iou=iou, max_det=max_det)
        # CPU 는 프레임당 ~25ms 라 3프레임에 한 번만 (유지 시간 판단은 시각 기준이라 문제 없음)
        self.every = 1 if self.on_gpu else 3
        self._n = 0
        self._last = GateStatus(None, 0.0, 0.0, hold_seconds, None)

    @property
    def hold_seconds(self) -> float:
        return self.det.hold_seconds

    def set_hold_seconds(self, sec: float):
        self.det.set_hold_seconds(sec)

    def set_confidence(self, c: float):
        self.det.set_confidence_threshold(c)

    def process(self, frame_bgr) -> GateStatus:
        self._n += 1
        if self._n % self.every:
            # 건너뛴 프레임: 직전 상태 유지, confirmed 는 한 번만
            return GateStatus(self._last.pose, self._last.confidence, self._last.hold, self.det.hold_seconds, None)
        r = self.det.process(frame_bgr)
        st = GateStatus(r.gesture, float(r.confidence), float(r.hold_time), self.det.hold_seconds,
                        r.gesture if r.confirmed else None)
        self._last = st
        return st

    def reset(self):
        self.det.reset()
        self._last = GateStatus(None, 0.0, 0.0, self.det.hold_seconds, None)


class FakeGate:
    """카메라 없이 시험할 때: 포즈 이름을 직접 넣어 주면 GestureDetector 와 같은 규칙(3초 유지)으로 확정한다."""

    def __init__(self, hold_seconds: float = 3.0):
        self.hold_seconds = hold_seconds
        self.on_gpu = False
        self.path = Path("(fake)")
        self._pose = None
        self._since = None
        self._fired = False
        self.next_pose: str | None = None

    def set_hold_seconds(self, sec: float):
        self.hold_seconds = sec

    def set_confidence(self, c: float):
        pass

    def process(self, frame_bgr, now: float | None = None) -> GateStatus:
        now = time.monotonic() if now is None else now
        pose = self.next_pose
        if pose is None:
            self._pose = None; self._since = None; self._fired = False
            return GateStatus(None, 0.0, 0.0, self.hold_seconds, None)
        if pose != self._pose:
            self._pose, self._since, self._fired = pose, now, False
            return GateStatus(pose, 1.0, 0.0, self.hold_seconds, None)
        hold = now - self._since
        confirmed = None
        if hold >= self.hold_seconds and not self._fired:
            self._fired = True; confirmed = pose
        return GateStatus(pose, 1.0, hold, self.hold_seconds, confirmed)

    def reset(self):
        self._pose = None; self._since = None; self._fired = False
