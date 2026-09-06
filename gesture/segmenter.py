"""실시간 스트림에서 "정지 → 동작 → 정지" 구간을 잘라내는 감지기.

학습 데이터(수집기 npz)가 정확히 이 모양(앞뒤 짧은 정지 + 동작 한 번)으로 찍혔으므로,
실시간에서도 같은 모양으로 잘라 모델에 넘겨야 잘 맞는다.

움직임 에너지 = 21개 관절의 프레임 간 평균 이동거리 / 손바닥 크기.
손목만 보면 make_fist(손목 고정)를 놓치므로 손가락을 포함한 전체 관절을 쓴다.
기준값은 녹화 데이터에서 잼: 정지 0.004~0.03, make_fist 피크 ≥0.04, swipe 피크 0.1~0.3.
09-06 웹캠 실측 후 조정: OFF 0.02→0.015 (느린 스와이프가 중간에 끊기던 문제),
구간 끝의 미검출(손이 화면 밖) 프레임은 잘라낸 뒤 품질 검사 (꼬리 NaN 때문에 검출률 80% 미달로 버려지던 문제).
3차 실측 후: off_frames 10→8 로 복귀 (주먹 뒤 짧은 멈춤에도 구간이 닫히게. 10 이면 주먹+내리기가 한 구간으로 합쳐져
확신도가 떨어짐), max_sec 2.5→3.0 (화면 가장자리에서 진입해 쓸면 2.4초까지 나옴).

상태:
  IDLE   손이 없거나 가만히 있음. 링버퍼에 프레임만 쌓음(pre-roll 용).
  ACTIVE 에너지가 ON 을 연속 on_frames 이상 넘어 동작 시작. 시작점은 그보다 pre_roll 앞.
  → 에너지가 OFF 아래로 연속 off_frames 이상이면 동작 끝 → 구간 반환.
  → max_sec 을 넘으면 강제 종료(구간 반환 안 함 = 너무 긴 건 버림).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config

WRIST, MIDDLE_MCP = 0, 9


@dataclass
class Segment:
    landmarks: np.ndarray      # (T,21,3) NaN = 미검출
    timestamps_ms: np.ndarray  # (T,)
    handedness: str            # 구간 내 다수 손 ("Right"/"Left"/"")
    duration_sec: float
    detection_ratio: float
    peak_energy: float = 0.0   # 구간 중 최대 에너지 (진단용)
    palm: float = 0.0          # 구간 중 손바닥 크기 중간값, 정규화 좌표 (거리 진단용: 가까우면 0.3, 멀면 0.1 이하)


class MotionSegmenter:
    """에너지 = 프레임 간 이동거리가 큰 상위 top_k 개 관절의 평균 / 손바닥 크기.
    (09-06 4차 실측 후 21개 평균 → 상위 8개 평균으로 변경. 주먹은 손가락 12개만 움직여 21개 평균이 깎이는데,
    상위 8개 평균이면 주먹 피크 하위10% 0.10 vs 정지 잡음 0.012 로 여유가 두 배. 스와이프는 전 관절이 움직여 무관.)"""

    def __init__(self, fps_hint: float = 30.0,
                 on_thresh: float = 0.05, off_thresh: float = 0.025,
                 on_frames: int = 3, off_frames: int = 8,
                 pre_roll_sec: float = 0.25, min_sec: float = 0.4, max_sec: float = 3.0,
                 buffer_sec: float = 4.0, smooth: int = 3, top_k: int = 8,
                 floor_alpha: float = 0.05, on_over_floor: float = 2.5, off_over_floor: float = 1.6):
        self.on_thresh, self.off_thresh = on_thresh, off_thresh
        # ── 떨림 바닥값(noise floor) 적응 ─────────────────────────────────────────
        # 09-06 5차 실측: 손이 멀면(손바닥 0.06~0.1) 좌표 떨림이 손바닥 대비 커져 고정 멈춤 기준 0.025 아래로
        # 못 내려가고, 구간이 3초를 채운 뒤 통째로 버려졌다(13건 = "멀리서 주먹 안 됨").
        # 해결: IDLE 상태에서 손이 보일 때의 에너지를 느린 지수평균으로 추적해 '지금 이 거리의 떨림 바닥값'을 재고,
        #   시작 기준 = max(on_thresh,  on_over_floor  × 바닥값)
        #   멈춤 기준 = max(off_thresh, off_over_floor × 바닥값)
        # 가까운 손(바닥 ~0.01)은 고정값이 그대로 적용되고, 먼 손(바닥 0.03~0.05)은 기준이 자동으로 올라간다.
        # (구간 최대 에너지에 비례시키는 방식은 순간 스파이크가 기준을 끌어올려 느린 동작을 중간에 끊는 부작용이 있어 폐기.)
        self.floor_alpha = floor_alpha
        self.on_over_floor, self.off_over_floor = on_over_floor, off_over_floor
        self.noise_floor = 0.0
        self.on_frames, self.off_frames = on_frames, off_frames
        self.pre_roll = int(round(pre_roll_sec * fps_hint))
        self.min_sec, self.max_sec = min_sec, max_sec
        self.smooth = smooth
        self.top_k = top_k
        self.last_drop: str | None = None   # 마지막으로 구간이 버려진 이유 (진단용, 읽고 나면 None 으로)
        # ── 주먹 조기 판정 ────────────────────────────────────────────────
        # 09-06 22시: "동작이 끝나고 멈춘 뒤" 판정하면 빨리 쥔 주먹은 0.5~0.8초 뒤에야 실행돼 느리게 느껴짐.
        # 손가락이 접힌 상태가 early_fist_frames 프레임 이어지면 멈춤을 기다리지 않고 구간을 바로 닫는다.
        # 스와이프는 방향을 끝까지 봐야 하므로 해당 없음(펴짐이 안 떨어짐).
        self.early_fist = True
        self.early_fist_frames = 4
        self._closed_count = 0
        self.last_early = False              # 마지막 반환 구간이 조기 판정이었나 (로그용)
        # 조기 판정 뒤에는 손이 한 번 '쉬어야'(에너지 < 멈춤 기준이 off_frames 연속) 다음 구간을 시작한다.
        # 안 그러면 쥐었다 폈다를 반복하는 동작이 닫힐 때마다 새 구간으로 잡혀 매번 주먹으로 실행된다.
        self._need_rest = False
        self._rest_count = 0
        n = int(buffer_sec * fps_hint) + 10
        self._ts = deque(maxlen=n)
        self._lm = deque(maxlen=n)
        self._hand = deque(maxlen=n)
        self._energy = deque(maxlen=n)
        self.reset()

    # ── 상태 ─────────────────────────────────────────────────────────
    def reset(self):
        self.state = "IDLE"
        self._on_count = 0
        self._off_count = 0
        self._start_idx = None       # ACTIVE 시작 프레임의 절대 인덱스
        self._abs = 0                # 지금까지 넣은 프레임 수 (절대 인덱스)
        self._prev_lm = None
        self.last_energy = 0.0
        self._peak = 0.0
        self._palms: list[float] = []
        self.noise_floor = 0.0

    @property
    def on_level(self) -> float:
        """현재 적용 중인 시작 기준"""
        return max(self.on_thresh, self.on_over_floor * self.noise_floor)

    @property
    def off_level(self) -> float:
        """현재 적용 중인 멈춤 기준"""
        return max(self.off_thresh, self.off_over_floor * self.noise_floor)

    @property
    def active_sec(self) -> float:
        if self.state != "ACTIVE" or not self._ts:
            return 0.0
        return (self._ts[-1] - self._ts[self._rel(self._start_idx)]) / 1000.0

    def _rel(self, abs_idx: int) -> int:
        """절대 인덱스 → 링버퍼 내 상대 인덱스 (버퍼에서 밀려났으면 0)"""
        first_abs = self._abs - len(self._ts)
        return max(0, abs_idx - first_abs)

    # ── 에너지 ───────────────────────────────────────────────────────
    def _frame_energy(self, lm: np.ndarray | None) -> float:
        if lm is None or self._prev_lm is None:
            return 0.0
        palm = float(np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2]))
        if palm < 1e-4:
            return 0.0
        d = np.linalg.norm(lm[:, :2] - self._prev_lm[:, :2], axis=1) / palm
        return float(np.sort(d)[-self.top_k:].mean())

    # ── 입력 ─────────────────────────────────────────────────────────
    def push(self, lm: np.ndarray | None, ts_ms: float, handedness: str = "") -> Segment | None:
        """프레임 하나 넣기. 동작 구간이 끝났으면 Segment 반환, 아니면 None."""
        e_raw = self._frame_energy(lm)
        if lm is not None:
            self._prev_lm = lm
        self._energy.append(e_raw)
        # 최근 smooth 프레임의 중간값으로 잡음 완화
        recent = list(self._energy)[-self.smooth:]
        e = float(np.median(recent)) if recent else 0.0
        self.last_energy = e

        self._ts.append(float(ts_ms))
        self._lm.append(lm if lm is not None else np.full((config.N_LANDMARKS, 3), np.nan, np.float32))
        self._hand.append(handedness)
        self._abs += 1

        if lm is None:
            # 손이 사라짐: 동작 중이었으면 끝난 것으로 처리 시도, 아니면 대기
            if self.state == "ACTIVE":
                self._off_count += 1
                if self._off_count >= self.off_frames:
                    return self._finish()
            else:
                self._on_count = 0
            return None

        if self.state == "IDLE":
            # 바닥값 갱신: 손이 보이고 동작 시작 후보가 아닐 때만 (동작 자체가 바닥값을 올리지 않게)
            if e < self.on_level:
                self.noise_floor = e if self.noise_floor == 0.0 else \
                    (1 - self.floor_alpha) * self.noise_floor + self.floor_alpha * e
            if self._need_rest:
                # 조기 판정 직후: 손이 쉴 때까지 새 구간을 열지 않는다
                self._rest_count = self._rest_count + 1 if e < self.off_level else 0
                if self._rest_count >= self.off_frames:
                    self._need_rest = False
                    self._rest_count = 0
                self._on_count = 0
                return None
            if e >= self.on_level:
                self._on_count += 1
                if self._on_count >= self.on_frames:
                    self.state = "ACTIVE"
                    self._start_idx = max(0, self._abs - self._on_count - self.pre_roll)
                    self._off_count = 0
                    self._peak = e
                    self._palms = []
            else:
                self._on_count = 0
            return None

        # ACTIVE
        self._peak = max(self._peak, e)
        self._palms.append(float(np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2])))
        if self.early_fist:
            from . import sanity   # 순환 import 방지용 지연 import
            s = self._rel(self._start_idx)
            window = np.stack(list(self._lm)[s:], axis=0)
            if sanity.fist_closed_now(window):
                self._closed_count += 1
                if self._closed_count >= self.early_fist_frames:
                    self._closed_count = 0
                    self.last_early = True
                    self._need_rest = True
                    self._rest_count = 0
                    return self._finish()
            else:
                self._closed_count = 0
        if e < self.off_level:
            self._off_count += 1
            if self._off_count >= self.off_frames:
                self.last_early = False
                return self._finish()
        else:
            self._off_count = 0
        if self.active_sec > self.max_sec:
            palm = float(np.median(self._palms)) if self._palms else 0.0
            self.last_drop = (f"too long > {self.max_sec}s (peak {self._peak:.3f}, palm {palm:.3f}, "
                              f"floor {self.noise_floor:.3f}, off_level {self.off_level:.3f}, last_e {e:.3f})")
            self._abort()
        return None

    def _abort(self):
        self.state = "IDLE"
        self._on_count = self._off_count = 0
        self._closed_count = 0
        self._start_idx = None

    def _finish(self) -> Segment | None:
        s = self._rel(self._start_idx)
        ts = np.array(list(self._ts)[s:], dtype=np.float64)
        lm = np.stack(list(self._lm)[s:], axis=0)
        hands_all = list(self._hand)[s:]
        peak, palms = self._peak, self._palms
        self._abort()
        det = ~np.isnan(lm[:, 0, 0])
        if det.sum() < 2:
            self.last_drop = "no detected frames"
            return None
        # 손이 화면 밖으로 나가 끝난 경우: 꼬리의 미검출 프레임을 잘라낸다 (동작 자체는 그 앞에 다 들어 있음)
        last = int(np.where(det)[0][-1]) + 1
        ts, lm, det = ts[:last], lm[:last], det[:last]
        hands = [h for h in hands_all[:last] if h]
        if len(ts) < 2:
            self.last_drop = "too few frames"
            return None
        dur = (ts[-1] - ts[0]) / 1000.0
        ratio = float(det.mean())
        palm = float(np.median(palms)) if palms else 0.0
        if dur < self.min_sec or dur > self.max_sec:
            self.last_drop = f"duration {dur:.2f}s out of [{self.min_sec}, {self.max_sec}] (peak {peak:.3f}, palm {palm:.3f})"
            return None
        hand = max(set(hands), key=hands.count) if hands else ""
        return Segment(landmarks=lm, timestamps_ms=ts, handedness=hand,
                       duration_sec=float(dur), detection_ratio=ratio, peak_energy=peak, palm=palm)
