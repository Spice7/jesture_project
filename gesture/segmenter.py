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
        # ── 스냅 조기 판정 (09-07) ──────────────────────────────────────────
        # 엄지·중지가 붙어 있다가 떨어진 상태가 early_snap_frames 프레임 이어지면 바로 닫는다 (주먹과 같은 구조).
        # 튕긴 뒤 손이 멈추길 0.27초 기다리던 지연을 없앤다. 스냅 라벨이 없는 설정에서는 꺼진다.
        self.early_snap = "finger_snap" in config.LABELS
        self.early_snap_frames = 4          # 09-07 스냅 50개 실측: 3f 는 46/50, 4f 부터 49/50(조기판정 없을 때와 동일), 판정 -293ms
        self._released_count = 0
        # 09-07 오후: 손목이 크게 움직인 구간(=스와이프)에서는 조기 판정을 하지 않는다. 스와이프 끝에서 손이 기울면 2D 에서
        # 손가락이 접힌 듯 보여 주먹 규칙이 구간을 중간에 끊었다(실행된 스와이프 26개 중 13개가 EARLY, 잘린 구간은
        # 상식검사 '손 안 펴짐 1.00~1.09' 에 걸림). 구간 내 손목 최대 이동(손바닥 단위): 스와이프 p5 1.57, 주먹 p95 0.43, 스냅 p95 0.67.
        self.early_max_wrist_move = 1.0
        # ── 스와이프 조기 판정 (09-07 오후) ─────────────────────────────────
        # 웹캠 실측: 스와이프 뒤 손을 바로 되돌리면 "가기+돌아오기"가 한 구간(1.5~2.4s)이 되어 순이동≈0 → 상식검사 차단(12건 전부).
        # 손목이 swipe_peak 이상 나아간 뒤 swipe_reverse 만큼 되돌아오면, 가장 멀리 간 지점 직후에서 구간을 닫는다.
        # 되돌리는 동작은 다음 구간이 되고 모델이 복귀(no_gesture)로 배운 것이다.
        self.early_swipe = True
        self.swipe_peak = 1.5               # 손바닥 단위. 스와이프 순이동 p5 1.47, 주먹/스냅 최대 이동 p95 0.43/0.67.
                                            # 1.2~1.8 어디든 결과 동일(바로 되돌리기 118/120, no_gesture 307 중 새는 것 1개 = 옆으로 갔다 오는 클립)
        self.swipe_reverse = 0.5            # 최고점에서 이만큼 되돌아오면 반전으로 본다
        self.swipe_tail = 3                 # 최고점 뒤에 남길 프레임 수 (학습 클립의 짧은 멈춤을 흉내)
        self._peak_disp = 0.0
        self._peak_rel = -1
        # (A) 동작 중 손을 놓치면(빠른 스와이프의 모션 블러) 바로 닫지 않고 lost_frames 까지 기다린다. 다시 잡히면 같은 구간.
        #     09-07 11:48 실측: DROP 10건 중 8건이 "0.32s, 검출 0" = 출발 직후 추적 끊김 → 짧고 빠른 스와이프가 통째로 버려짐.
        #     이미 스와이프가 완성된 구간(최고 이동 ≥ swipe_peak)이면 예전처럼 off_frames 만 기다린다(손이 화면 밖으로 나간 경우).
        self.lost_frames = 12               # = 실시간 품질 기준 max_gap 12 (보간 가능한 최대 공백)
        self._lost_count = 0
        # (B) 감속 종료: 손목이 swipe_peak 이상 나아간 뒤 속도가 최고 속도의 decel_ratio 아래로 decel_frames 연속이면 멈춤을 안 기다리고 닫는다.
        #     11:48 실측: 스와이프 30건 중 18건이 '정지 8프레임' 으로 닫혀 끝 자세를 유지해야 했음.
        self.decel_ratio = 0.3
        self.decel_frames = 3
        self._peak_speed = 0.0
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
                self._lost_count += 1
                limit = self.off_frames if self._peak_disp >= self.swipe_peak else self.lost_frames
                if self._lost_count >= limit:
                    return self._finish()
            else:
                self._on_count = 0
            return None
        self._lost_count = 0

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
        early_allowed = self.early_fist or self.early_snap
        if early_allowed:
            from . import sanity   # 순환 import 방지용 지연 import
            s = self._rel(self._start_idx)
            window = np.stack(list(self._lm)[s:], axis=0)
            wdet = ~np.isnan(window[:, 0, 0])
            palm_now = float(np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2])) or 1e-4
            wrist_move = float(np.max(np.linalg.norm(window[wdet, 0, :2] - window[wdet][0, 0, :2], axis=1))) / palm_now
            if wrist_move > self.early_max_wrist_move:      # 손목이 크게 움직임 = 스와이프 → 주먹/스냅 조기 판정은 끔
                early_allowed = False
                self._closed_count = self._released_count = 0
            if self.early_swipe and wdet.sum() >= 2:
                disp = np.linalg.norm(window[wdet, 0, :2] - window[wdet][0, 0, :2], axis=1) / palm_now
                idx_det = np.where(wdet)[0]
                k = int(np.argmax(disp))
                if disp[k] > self._peak_disp:
                    self._peak_disp, self._peak_rel = float(disp[k]), int(idx_det[k])
                if self._peak_disp >= self.swipe_peak and self._peak_disp - float(disp[-1]) >= self.swipe_reverse:
                    self.last_early = True
                    self._need_rest = True
                    self._rest_count = 0
                    return self._finish(end_rel=self._peak_rel + 1 + self.swipe_tail)
                # (B) 감속 종료
                w = window[wdet, 0, :2]
                if len(w) >= self.decel_frames + 2:
                    speed = np.linalg.norm(np.diff(w, axis=0), axis=1) / palm_now
                    self._peak_speed = max(self._peak_speed, float(speed.max()))
                    if (self._peak_disp >= self.swipe_peak and self._peak_speed > 0
                            and bool(np.all(speed[-self.decel_frames:] < self.decel_ratio * self._peak_speed))):
                        self.last_early = True
                        self._need_rest = True
                        self._rest_count = 0
                        return self._finish()
        if early_allowed and self.early_fist:
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
        if early_allowed and self.early_snap:
            if sanity.snap_released_now(window):
                self._released_count += 1
                if self._released_count >= self.early_snap_frames:
                    self._released_count = 0
                    self.last_early = True
                    self._need_rest = True
                    self._rest_count = 0
                    return self._finish()
            else:
                self._released_count = 0
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
        self._released_count = 0
        self._peak_disp, self._peak_rel = 0.0, -1
        self._lost_count = 0
        self._peak_speed = 0.0
        self._start_idx = None

    def _finish(self, end_rel: int | None = None) -> Segment | None:
        """end_rel: 구간 시작 기준 이 프레임 수까지만 잘라서 반환 (스와이프 조기 판정용). None 이면 지금까지 전부."""
        s = self._rel(self._start_idx)
        e = None if end_rel is None else s + end_rel
        ts = np.array(list(self._ts)[s:e], dtype=np.float64)
        lm = np.stack(list(self._lm)[s:e], axis=0)
        hands_all = list(self._hand)[s:e]
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
