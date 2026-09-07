"""실시간 인식 파이프라인을 한 덩어리로: 프레임 → 손 관절 → 구간 감지 → GRU → 상식 검사 → 게이트 → 기능 실행.

scripts/realtime_demo.py 의 루프에서 화면 그리기를 뺀 것. 시연 UI(scripts/gesture_app.py)가 쓴다.
realtime_demo.py 는 검증된 그대로 두었고, 판정 규칙·기준값은 여기서도 동일하다.

게이트(YOLO 정적 포즈 ON/OFF)는 아직 연결 전이다. YOLO 팀 코드가 오면 `Pipeline.set_gate(True/False)` 만 불러 주면
된다. 게이트가 닫혀 있으면 판정은 기록만 하고 키는 누르지 않는다.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, preprocess, sanity
from .actions import ActionMapper
from .landmarks import HandTracker
from .model import GestureClassifier
from .segmenter import MotionSegmenter

# realtime_demo.py 와 같은 값
REALTIME_MIN_DET = 0.5
REALTIME_MAX_GAP = 12
TRACK_CONF = 0.3
PRESENCE_CONF = 0.4


class SessionLog:
    """판정·버림·실행을 한 줄씩 파일에 남긴다. reports/realtime/session_YYYYMMDD_HHMMSS.log"""

    def __init__(self, root: Path = config.REPORTS_DIR / "realtime", echo: bool = False):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / time.strftime("session_%Y%m%d_%H%M%S.log")
        self._f = open(self.path, "a", encoding="utf-8")
        self.echo = echo

    def write(self, line: str):
        stamp = time.strftime("%H:%M:%S")
        if self.echo:
            print(f"[{stamp}] {line}")
        self._f.write(f"[{stamp}] {line}\n")
        self._f.flush()

    def close(self):
        self._f.close()


@dataclass
class Event:
    """구간 하나가 끝났을 때의 결과 (UI 표시용)."""
    time: float                      # perf_counter
    label: str                       # 판정 라벨 ("skipped" 면 품질 미달)
    confidence: float
    duration_sec: float
    probs: np.ndarray | None
    executed: bool                   # 키를 실제로 눌렀거나(LIVE) 누를 뻔했나(DRY)
    message: str                     # 사람이 읽는 한 줄 (GUARD/cooldown/ignored/[DRY]/[FIRE]/GATE)
    early: bool = False


@dataclass
class FrameState:
    """매 프레임 갱신되는 상태 (UI 가 폴링)."""
    landmarks: np.ndarray | None = None
    hand: str = ""
    seg_state: str = "IDLE"
    active_sec: float = 0.0
    energy: float = 0.0
    on_level: float = 0.0
    off_level: float = 0.0
    floor: float = 0.0
    fps: float = 0.0
    in_cooldown: bool = False
    last_event: Event | None = None
    n_executed: int = 0
    events: list[Event] = field(default_factory=list)


def gap_runs(det: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, ok in enumerate(det):
        if not ok and start is None:
            start = i
        if ok and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(det) - 1))
    return runs


class Pipeline:
    """한 프레임씩 `step(frame)` 을 부르면 된다. 스레드 안전: `state` 는 lock 으로 복사해 읽는다."""

    def __init__(self, model_file: Path | None = None, gestures_path: Path | None = None,
                 dry_run: bool = True, log_echo: bool = False):
        self.clf = GestureClassifier(model_file=model_file) if model_file else GestureClassifier()
        self.mapper = ActionMapper(gestures_path) if gestures_path else ActionMapper()
        self.mapper.dry_run = dry_run
        self.seg = MotionSegmenter()
        self.log = SessionLog(echo=log_echo)
        self.tracker = HandTracker(track_conf=TRACK_CONF, presence_conf=PRESENCE_CONF)
        self._lock = threading.Lock()
        self.state = FrameState()
        self.gate_open = True            # YOLO 게이트 자리. 연결 전에는 항상 열림.
        self.gate_connected = False      # YOLO 팀 코드가 set_gate 를 부르기 시작하면 True
        self._t0 = time.perf_counter()
        self._fps_n, self._fps_t = 0, self._t0
        self.log.write(f"모델 {self.clf.arch} | 라벨 {self.clf.labels} | 매핑: "
                       + ", ".join(f"{k}->{self.mapper.describe(k)}" for k in self.clf.labels))
        s = self.seg
        self.log.write(f"segmenter on=max({s.on_thresh},{s.on_over_floor}*floor) off=max({s.off_thresh},{s.off_over_floor}*floor) "
                       f"on_frames={s.on_frames} off_frames={s.off_frames} min_sec={s.min_sec} "
                       f"| quality det>={REALTIME_MIN_DET} gap<={REALTIME_MAX_GAP} | track_conf={TRACK_CONF} presence={PRESENCE_CONF} "
                       f"| min_conf={self.mapper.min_confidence} | mode={'DRY' if self.mapper.dry_run else 'LIVE'}")

    # ── 외부에서 바꾸는 것 ──
    @property
    def dry_run(self) -> bool:
        return self.mapper.dry_run

    def set_dry_run(self, dry: bool):
        if dry == self.mapper.dry_run:
            return
        self.mapper.dry_run = dry
        self.log.write("MODE  " + ("DRY RUN (연습)" if dry else "LIVE (실제 키 입력)"))

    def set_gate(self, is_open: bool, source: str = "yolo"):
        """YOLO 게이트 연결 지점. 닫히면 명령을 실행하지 않는다."""
        self.gate_connected = self.gate_connected or source == "yolo"
        if is_open != self.gate_open:
            self.gate_open = is_open
            self.log.write(f"GATE  {'OPEN' if is_open else 'CLOSED'} ({source})")

    def reset(self):
        self.seg.reset()
        self.log.write("RESET")

    def reload_mapping(self):
        self.mapper.reload()
        self.log.write("MAP   " + ", ".join(f"{k}->{self.mapper.describe(k)}" for k in self.clf.labels))

    # ── 매 프레임 ──
    def step(self, frame_bgr: np.ndarray) -> FrameState:
        now = time.perf_counter()
        ts_ms = (now - self._t0) * 1000.0
        lm, hand = self.tracker.process(frame_bgr, int(ts_ms))
        seg = self.seg
        segment = seg.push(lm, ts_ms, hand)
        event = None
        if seg.last_drop:
            self.log.write(f"DROP  {seg.last_drop}")
            seg.last_drop = None
        if segment is not None:
            event = self._judge(segment, now)

        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            fps = self._fps_n / (now - self._fps_t)
            self._fps_n, self._fps_t = 0, now
        else:
            fps = None
        with self._lock:
            st = self.state
            st.landmarks = lm
            st.hand = hand or ""
            st.seg_state = seg.state
            st.active_sec = seg.active_sec if seg.state == "ACTIVE" else 0.0
            st.energy = seg.last_energy
            st.on_level, st.off_level, st.floor = seg.on_level, seg.off_level, seg.noise_floor
            if fps is not None:
                st.fps = fps
            st.in_cooldown = self.mapper.in_cooldown(now)
            if event is not None:
                st.last_event = event
                st.events.append(event)
                if event.executed:
                    st.n_executed += 1
                if len(st.events) > 500:
                    del st.events[:250]
        return st

    def snapshot(self) -> FrameState:
        with self._lock:
            st = self.state
            copy = FrameState(landmarks=st.landmarks, hand=st.hand, seg_state=st.seg_state, active_sec=st.active_sec,
                              energy=st.energy, on_level=st.on_level, off_level=st.off_level, floor=st.floor,
                              fps=st.fps, in_cooldown=st.in_cooldown, last_event=st.last_event,
                              n_executed=st.n_executed, events=list(st.events))
        return copy

    def _judge(self, segment, now: float) -> Event:
        feats = preprocess.sample_to_features(segment.landmarks, segment.timestamps_ms,
                                              min_ratio=REALTIME_MIN_DET, max_gap=REALTIME_MAX_GAP)
        det = ~np.isnan(segment.landmarks[:, 0, 0])
        runs = gap_runs(det)
        gap_txt = ("gaps " + ",".join(f"{a}-{b}" for a, b in runs) + f" of {len(det)}f") if runs else "no gaps"
        if feats is None:
            gap = preprocess.longest_gap(det)
            why = (f"quality: det {segment.detection_ratio:.0%} (min {REALTIME_MIN_DET:.0%}), gap {gap} (max {REALTIME_MAX_GAP}), "
                   f"{segment.duration_sec:.2f}s, {gap_txt}")
            self.log.write(f"SKIP  {why} | palm={segment.palm:.3f} peak={segment.peak_energy:.3f} hand={segment.handedness or '-'}")
            return Event(now, "skipped", 0.0, segment.duration_sec, None, False, why.split(",")[0])

        label, conf, probs = self.clf.predict(feats)
        seg_start = self._t0 + segment.timestamps_ms[0] / 1000.0
        ok, reason = sanity.check(label, segment.landmarks)
        if label == "no_gesture":
            b = sanity._basic(segment.landmarks)
            if b:
                reason = (f"dx {b['dx']:+.2f} dy {b['dy']:+.2f} ext {b['ext_mean']:.2f} "
                          f"pinch {b['pinch_min']:.2f}->{b['pinch_end']:.2f}")
        executed = False
        if label != "no_gesture" and not ok:
            msg = f"GUARD {reason} -> ignored"
        elif label != "no_gesture" and not self.gate_open:
            msg = f"GATE closed -> {label} ignored"
        else:
            msg = self.mapper.handle(label, conf, now, start=seg_start)
            executed = msg.startswith("[DRY]") or msg.startswith("[FIRE]")
        pr = " ".join(f"{l[:5]}={p:.2f}" for l, p in zip(self.clf.labels, probs))
        self.log.write(f"{label:10s} {conf:.2f} {segment.duration_sec:.2f}s{' EARLY' if self.seg.last_early else ''} "
                       f"det={segment.detection_ratio:.0%} palm={segment.palm:.3f} peak={segment.peak_energy:.3f} "
                       f"floor={self.seg.noise_floor:.3f} hand={segment.handedness or '-'} [{pr}] {gap_txt} | {reason} -> {msg}")
        return Event(now, label, float(conf), segment.duration_sec, probs, executed, msg, early=bool(self.seg.last_early))

    def close(self):
        self.log.write(f"END  실행 {len(self.mapper.log)}건")
        self.log.close()
        try:
            self.tracker.close()
        except Exception:
            pass
