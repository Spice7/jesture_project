"""정적 손모양(YOLO)으로 LSTM 인식을 켜고 끄는 게이트. 카메라와 GUI 없이 판정만 담당합니다.

모델 라벨은 `start`/`stop`/`cancel`이고, 서비스 동작 이름은 여기서만 연결합니다.
import만으로 가중치나 카메라를 열지 않습니다.
"""

from dataclasses import dataclass
import math
from pathlib import Path

# 학습된 가중치의 클래스 계약입니다. 순서가 아니라 이름으로 대응합니다.
GATE_LABELS = ("cancel", "start", "stop")
# 손모양 → 서비스 동작. 보자기=start, 주먹=stop, 총 모양=cancel로 확인했습니다.
GATE_ACTIONS = {"start": "arm", "stop": "disarm", "cancel": "toggle_window"}
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "yolo" / "v2" / "best.pt"


@dataclass(frozen=True)
class GateConfig:
    """초기 실험값입니다. 실제 연속 사용에서 검증한 최적값이 아닙니다."""

    hold_seconds: float = 3.0
    confidence: float = .5
    imgsz: int = 640
    # 이 시간보다 짧은 끊김은 같은 유지로 봅니다. 손떨림/한두 프레임 미검출 대응입니다.
    release_seconds: float = .4
    # stop 후보가 이만큼 이어지면 make_fist 확정을 유예합니다.
    suppress_after_seconds: float = 1.0

    def validate(self):
        for name in ("hold_seconds", "confidence", "release_seconds", "suppress_after_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}는 유한한 수여야 합니다.")
        if min(self.hold_seconds, self.release_seconds) <= 0:
            raise ValueError("hold/release seconds는 양수여야 합니다.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence는 0~1이어야 합니다.")
        if self.suppress_after_seconds < 0 or self.suppress_after_seconds > self.hold_seconds:
            raise ValueError("suppress_after_seconds는 0 이상 hold_seconds 이하여야 합니다.")
        if type(self.imgsz) is not int or self.imgsz < 32:
            raise ValueError("imgsz는 32 이상의 정수여야 합니다.")


class HoldDetector:
    """같은 손모양이 hold_seconds 동안 이어지면 한 번만 확정합니다.

    확정 후에는 그 포즈를 놓거나 다른 포즈로 바꾸기 전까지 다시 확정하지 않습니다.
    시간이 지났다는 이유만으로 반복 실행되지 않게 하려는 기존 cooldown과 같은 취지입니다.
    """

    def __init__(self, config=None):
        self.config = config or GateConfig()
        self.config.validate()
        self.reset()

    def reset(self):
        self.label = None
        self.since = None
        self.last_seen = None
        self.fired = None
        self.last_timestamp = None

    def update(self, timestamp, label):
        """프레임 한 개의 판정 결과를 넣고 확정된 라벨 하나 또는 None을 받습니다."""
        if not math.isfinite(timestamp) or (self.last_timestamp is not None
                                            and timestamp < self.last_timestamp):
            raise ValueError("게이트 프레임 시각은 유한하고 감소하지 않아야 합니다.")
        self.last_timestamp = timestamp
        if label is not None and label not in GATE_LABELS:
            raise ValueError(f"게이트 라벨 계약 오류: {label}")
        if label is None:
            # 짧은 끊김은 유지로 인정하고, 충분히 길면 포즈를 놓은 것으로 봅니다.
            if self.label is not None and timestamp - self.last_seen >= self.config.release_seconds:
                self.reset()
            return None
        if label != self.label:
            self.label, self.since, self.fired = label, timestamp, None
        self.last_seen = timestamp
        if self.fired == label:
            return None  # 이미 확정한 포즈를 계속 유지하는 중입니다.
        if timestamp - self.since + 1e-9 < self.config.hold_seconds:
            return None
        self.fired = label
        return label

    def pending(self, timestamp):
        """아직 확정되지 않은 진행 중인 후보와 지속 시간입니다. UI/유예 판정용입니다."""
        if self.label is None or self.since is None or self.fired == self.label:
            return None, 0.
        return self.label, max(0., timestamp - self.since)

    def progress(self, timestamp):
        """0~1 진행률입니다. 확정 후 유지 중이면 1로 표시합니다."""
        if self.label is None or self.since is None:
            return 0.
        if self.fired == self.label:
            return 1.
        return min(1., max(0., timestamp - self.since) / self.config.hold_seconds)


def suppressed_commands(detector, timestamp):
    """stop 유지가 진행 중이면 주먹 명령을 잠시 보류할 라벨을 알려줍니다.

    3초를 채우면 명령 없이 인식이 꺼지고, 그 전에 손을 펴면 보류가 풀려 실행됩니다.
    """
    label, elapsed = detector.pending(timestamp)
    if label == "stop" and elapsed >= detector.config.suppress_after_seconds:
        return frozenset({"make_fist"})
    return frozenset()


class YoloGate:
    """가중치 하나를 시작 시 로딩합니다. 프레임을 저장하거나 학습하지 않습니다."""

    def __init__(self, weights=DEFAULT_WEIGHTS, device="auto", config=None):
        self.config = config or GateConfig()
        self.config.validate()
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO 가중치 파일 누락: {path}")
        # 모듈 import만으로 ultralytics/torch 모델을 올리지 않습니다.
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA를 사용할 수 없습니다. device를 cpu로 지정하세요.")
        self.device = device
        self.model = YOLO(str(path))
        self.model.to(device)
        names = getattr(self.model, "names", None)
        if not isinstance(names, dict) or set(names.values()) != set(GATE_LABELS):
            raise ValueError(f"게이트 모델은 {GATE_LABELS} 클래스여야 합니다: {names}")
        self.names = dict(names)
        self.detector = HoldDetector(self.config)

    def predict(self, frame):
        """가장 확신도가 높은 손모양 하나를 돌려줍니다. 없으면 (None, 0.0)입니다."""
        result = self.model.predict(frame, imgsz=self.config.imgsz, conf=self.config.confidence,
                                    verbose=False, device=self.device)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None, 0.
        top = max(range(len(boxes)), key=lambda index: float(boxes.conf[index]))
        score = float(boxes.conf[top])
        if not math.isfinite(score) or score < self.config.confidence:
            return None, 0.
        label = self.names.get(int(boxes.cls[top]))
        if label is None:
            raise ValueError("게이트 모델이 계약 밖의 클래스 번호를 반환했습니다.")
        return label, score

    def step(self, timestamp, frame):
        """프레임 한 개 → (확정 동작 또는 None, 현재 라벨, 확신도, 진행률)."""
        label, score = self.predict(frame)
        confirmed = self.detector.update(timestamp, label)
        return (GATE_ACTIONS[confirmed] if confirmed else None,
                label, score, self.detector.progress(timestamp))

    def reset(self):
        self.detector.reset()
