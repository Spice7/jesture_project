"""프레임 버퍼와 이벤트 판정. UI/카메라/키보드 없이 제어 가능한 시각으로 테스트합니다."""

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from .predictor import SequenceQualityError
from training.gesture_schema import validate_label_map
from .policy import SERVICE_HAND, SERVICE_LABELS


@dataclass(frozen=True)
class ServiceConfig:
    """초기 실험값입니다. 실제 연속 동작에서 검증된 최적값이 아닙니다."""

    window_seconds: float = 1.5
    inference_interval: float = .15
    confidence_threshold: float = .8
    stable_predictions: int = 2
    cooldown_seconds: float = 1.0
    rearm_predictions: int = 2
    reset_gap_seconds: float = .75

    def validate(self):
        for name in ("window_seconds", "inference_interval", "confidence_threshold", "cooldown_seconds", "reset_gap_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}는 유한한 수여야 합니다.")
        if min(self.window_seconds, self.inference_interval, self.reset_gap_seconds) <= 0:
            raise ValueError("window/inference interval/reset gap은 양수여야 합니다.")
        if self.cooldown_seconds < 0 or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("cooldown은 0 이상, confidence는 0~1이어야 합니다.")
        for name in ("stable_predictions", "rearm_predictions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name}는 양의 정수여야 합니다.")


class GestureController:
    """ON/OFF는 입력 장치와 무관합니다. 미래의 YOLO도 start()/stop()을 호출하면 됩니다."""

    def __init__(self, predictor, config=None):
        self.predictor = predictor
        self.gesture_schema = predictor.gesture_schema
        # 저장된 모델의 실제 출력 순서를 따릅니다. 구형 모델에 새 출력을 붙이지 않습니다.
        self.labels = tuple(predictor.labels)
        validate_label_map({name: index for index, name in enumerate(self.labels)})
        self.config = config or ServiceConfig()
        self.config.validate()
        quality = predictor.quality
        if not quality.min_duration <= self.config.window_seconds <= quality.max_duration:
            raise ValueError("window-seconds가 체크포인트의 min/max_duration 범위를 벗어납니다.")
        self.enabled = False
        self.diagnostics = None
        self._diagnostic_state = None
        # YOLO 게이트가 매 프레임 갱신합니다. 여기 있는 명령은 확정을 미룹니다.
        self.suppressed = frozenset()
        self._reset(armed=True)
        self.reason = "Recognition OFF"

    def _reset(self, armed):
        self.active_hand = None
        self.buffer = deque()
        self.segment_start = None
        self.last_frame = None
        self.last_attempt = None
        self.missing_since = None
        self.waiting_for_hand = False
        self.latest = None
        self.candidate = None
        self.candidate_count = 0
        self.neutral_count = 0
        self.armed = armed
        self.cooldown_until = 0.0
        self.last_event = None
        self.reason = "Collecting fresh window"

    def start(self):
        if not self.enabled:
            self.enabled = True
            self._reset(armed=True)

    def stop(self):
        self.enabled = False
        self._reset(armed=True)
        self.reason = "Recognition OFF"

    def reset_after_gap(self):
        """외부 입력 지연도 버퍼/판정을 초기화합니다. OFF 상태를 임의로 켜지 않습니다."""
        self._reset(armed=False)
        self.reason = "Input latency/gap: fresh window and neutral required"

    def _invalidate_prediction(self, reason):
        # 품질/검출 실패를 no_gesture로 취급하지 않으며 중립 연속 횟수도 끊습니다.
        self.latest = None
        self.candidate = None
        self.candidate_count = 0
        self.neutral_count = 0
        self.reason = reason

    def _accept_prediction(self, label_id, probabilities, timestamp):
        if (type(label_id) is not int or not 0 <= label_id < len(self.labels)
                or not isinstance(probabilities, np.ndarray) or probabilities.shape != (len(self.labels),)
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1, atol=1e-5)
                or label_id != int(probabilities.argmax())):
            raise ValueError("추론 결과 클래스/확률 계약 오류")
        label, confidence = self.labels[label_id], float(probabilities[label_id])
        self.latest = (label, confidence, timestamp)
        if self.active_hand != SERVICE_HAND or label not in SERVICE_LABELS:
            self._invalidate_prediction("Hand/class mismatch: command blocked")
            return None
        if confidence < self.config.confidence_threshold:
            self.candidate, self.candidate_count, self.neutral_count = None, 0, 0
            self.reason = "Low confidence"
            return None
        if label == "no_gesture":
            self.candidate, self.candidate_count = None, 0
            if not self.armed:
                self.neutral_count = min(self.neutral_count + 1, self.config.rearm_predictions)
                if self.neutral_count >= self.config.rearm_predictions and timestamp >= self.cooldown_until:
                    self.armed = True
            self.reason = "Neutral / no command"
            return None
        self.neutral_count = 0
        if not self.armed or timestamp < self.cooldown_until:
            self.candidate, self.candidate_count = None, 0
            self.reason = "Wait for neutral before next command"
            return None
        self.candidate_count = self.candidate_count + 1 if self.candidate == label else 1
        self.candidate = label
        self.reason = "Stabilizing prediction"
        if self.candidate_count < self.config.stable_predictions:
            return None
        if label in self.suppressed:
            # 같은 손모양의 유지 판정이 진행 중입니다. 3초를 채우면 게이트가 인식을 끄고,
            # 그 전에 손을 풀면 보류가 사라져 바로 다음 추론에서 확정합니다.
            self.candidate_count = self.config.stable_predictions
            self.reason = "Hold gesture in progress: command deferred"
            return None
        self.armed = False
        self.cooldown_until = timestamp + self.config.cooldown_seconds
        self.last_event = label
        self.candidate, self.candidate_count = None, 0
        self.reason = "Event confirmed (DRY RUN)"
        return label

    def step(self, timestamp, landmarks, detected, handedness=None):
        """프레임 한 개를 처리하고 확정 이벤트 하나 또는 None을 반환합니다.

        timestamp는 프레임 획득 시각(초)입니다. 짧은 결측은 버퍼에 남겨 보간하며,
        긴 결측/시간 공백은 세션 내 단절로 보고 중립 재확인부터 시작합니다.
        """
        if not self.enabled:
            return None
        if handedness is None and self.gesture_schema == "right-only":
            handedness = "Right"  # 기존 오른손 전용 호출과 호환합니다.
        try:
            return self._step(timestamp, landmarks, detected, handedness)
        except Exception:
            # 예상된 품질 보류 외의 오류는 즉시 OFF로 전환하고 호출 측에 알립니다.
            self.stop()
            self.reason = "Input or inference error; recognition OFF"
            raise
        finally:
            state = (self.enabled, self.reason, self.armed)
            if state != self._diagnostic_state:
                self._trace("state", frame_s=timestamp, reason=self.reason, armed=self.armed)
                self._diagnostic_state = state

    def _trace(self, event, **fields):
        if self.diagnostics is not None:
            try:
                self.diagnostics(event, **fields)
            except Exception:
                pass  # 진단 출력 실패 때문에 제스처 동작을 바꾸지 않습니다.

    def _step(self, timestamp, landmarks, detected, handedness):
        if not math.isfinite(timestamp) or (self.last_frame is not None and timestamp <= self.last_frame):
            raise ValueError("프레임 시각은 유한하고 엄격히 증가해야 합니다.")
        if not isinstance(landmarks, np.ndarray) or landmarks.shape != (21, 3) or landmarks.dtype.kind != "f":
            raise ValueError("landmarks는 부동소수점 (21,3) 배열이어야 합니다.")
        if type(detected) not in (bool, np.bool_):
            raise ValueError("detected는 bool이어야 합니다.")
        if handedness == "Left":
            # 왼손은 추론/중립 판정에 쓰지 않습니다. 잠깐 등장해도 이전 오른손과 보간하지 않습니다.
            self._reset(armed=False)
            self.last_frame = timestamp
            self.reason = "Left hand unsupported: show Right hand and return to neutral"
            return None
        detected = bool(detected and np.isfinite(landmarks).all() and handedness == SERVICE_HAND)
        if self.last_frame is not None and timestamp - self.last_frame >= self.config.reset_gap_seconds:
            self._reset(armed=False)
            self.reason = "Camera time gap: fresh window required"
        self.last_frame = timestamp
        if detected:
            if self.active_hand is not None and self.active_hand != handedness:
                self._reset(armed=False)
                self.last_frame = timestamp
                self.reason = "Hand changed: fresh window and neutral required"
            self.active_hand = handedness
        if not detected:
            self._invalidate_prediction("No valid/allowed hand")
            if self.waiting_for_hand:
                return None
            if self.missing_since is None:
                self.missing_since = timestamp
            if timestamp - self.missing_since >= self.config.reset_gap_seconds:
                self._reset(armed=False)
                self.last_frame = timestamp
                self.waiting_for_hand = True
                self.reason = "Hand lost: buffer cleared, wait for neutral"
                return None
        else:
            self.missing_since = None
            self.waiting_for_hand = False
        if self.segment_start is None:
            self.segment_start = timestamp
        points = landmarks.copy() if detected else np.full((21, 3), np.nan, dtype=np.float32)
        self.buffer.append((timestamp, points, detected))
        cutoff = timestamp - self.config.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()
        if len(self.buffer) > 10000:
            raise ValueError("버퍼 프레임 상한 초과: 입력 시각/주기를 확인하세요.")
        if not detected:
            return None
        if timestamp - self.segment_start + 1e-9 < self.config.window_seconds:
            self.reason = "Collecting fresh window"
            return None
        if self.last_attempt is not None and timestamp - self.last_attempt + 1e-9 < self.config.inference_interval:
            return None  # 기존 예측을 다시 세지 않습니다.
        self.last_attempt = timestamp
        times = np.array([frame[0] for frame in self.buffer], dtype=np.float64)
        points = np.stack([frame[1] for frame in self.buffer])
        mask = np.array([frame[2] for frame in self.buffer], dtype=bool)
        started = time.monotonic()
        result = None
        raw_label, raw_scores = None, None
        try:
            label_id, probabilities = self.predictor.predict(points, times, mask)
            finished = time.monotonic()
            result = self._accept_prediction(label_id, probabilities, timestamp)
            raw_label = self.labels[label_id]
            raw_scores = {label: float(value) for label, value in zip(self.labels, probabilities)}
            return result
        except SequenceQualityError as exc:
            self._invalidate_prediction(f"Quality hold: {exc}")
            return None
        finally:
            ended = time.monotonic()
            self._trace("prediction", frame_s=timestamp, inference_started_s=started,
                        inference_finished_s=finished if raw_label is not None else ended,
                        inference_ms=1000*((finished if raw_label is not None else ended)-started),
                        label=raw_label, probabilities=raw_scores,
                        confidence=raw_scores.get(raw_label) if raw_scores else None, reason=self.reason,
                        candidate_count=self.config.stable_predictions if result else self.candidate_count,
                        stable_required=self.config.stable_predictions, neutral_count=self.neutral_count,
                        neutral_required=self.config.rearm_predictions, armed=self.armed,
                        cooldown_remaining_s=max(0., self.cooldown_until-timestamp),
                        window_frames=len(times), window_span_s=float(times[-1]-times[0]),
                        confirmed=result)
            if result:
                self._trace("confirmed", frame_s=timestamp, label=result)

    def status_lines(self, now):
        if not self.enabled:
            mode = "OFF"
        elif self.segment_start is None or now - self.segment_start < self.config.window_seconds:
            mode = "WARMUP"
        else:
            mode = "ACTIVE"
        span = self.buffer[-1][0] - self.buffer[0][0] if self.buffer else 0
        prediction = "none"
        if self.latest is not None:
            label, confidence, timestamp = self.latest
            prediction = f"{label} {confidence:.3f} (age {max(0, now-timestamp):.2f}s)"
        gate = "READY" if self.armed else "WAIT NEUTRAL"
        if not self.enabled:
            gate = "OFF"
        elif now < self.cooldown_until:
            gate = f"COOLDOWN {self.cooldown_until-now:.2f}s / WAIT NEUTRAL"
        return [f"Recognition: {mode}   DRY RUN ONLY", f"Buffer: {len(self.buffer)} frames / {span:.2f}s",
                f"Latest prediction: {prediction}",
                f"Candidate: {self.candidate or '-'} {self.candidate_count}/{self.config.stable_predictions}",
                f"Gate: {gate}   Neutral: {self.neutral_count}/{self.config.rearm_predictions}",
                f"Last event: {self.last_event or '-'}", self.reason,
                "S: ON   X: OFF   Esc: Exit (focus this window)"]
