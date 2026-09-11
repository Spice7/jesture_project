"""실시간 인식 파이프라인을 한 덩어리로: 프레임 → 손 관절 → 구간 감지 → GRU → 상식 검사 → 게이트 → 기능 실행.

통합 UI(scripts/gesture_app.py)가 사용하는 공통 인식 루프.

YOLO 정적 포즈 게이트 (09-07 오후 연결, gesture/gate.py + 정적 검출기 gesture/static_detector.py):
  손바닥 3초 → 게이트 열림(동적 제스처 실행 시작) / 주먹 3초 → 닫힘 / 세 번째 포즈 3초 → "cancel" 에 매핑된 키.
  게이트가 닫혀 있으면 동적 제스처는 판정·기록만 하고 키는 누르지 않는다.
  gestures.json 의 "gate" 로 켜고 끈다. 모델이 없으면 시작하지 않으며 실행 중 오류가 나면 인식을 중지한다.

주먹 명령 vs 주먹 3초(게이트 닫기) 충돌:
  게이트가 연결돼 있으면 make_fist 실행을 FIST_DEFER_SEC 만큼 미룬다. 그때까지 손이 계속 주먹이면
  "게이트를 닫으려는 주먹" 으로 보고 버리고, 손을 내렸거나 폈으면 실행한다.
  → 주먹 명령 습관: 쥐고 0.3초 멈춘 뒤 바로 내리기 (0.8초 안).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, preprocess, sanity
from .actions import ActionMapper, _IS_WIN
from .landmarks import HandTracker
from .model import GestureClassifier
from .segmenter import MotionSegmenter

# realtime_demo.py 와 같은 값
REALTIME_MIN_DET = 0.5
REALTIME_MAX_GAP = 12
TRACK_CONF = 0.3
PRESENCE_CONF = 0.4
FIST_DEFER_SEC = 0.8        # 게이트 연결 시 주먹 명령 유예
FIST_STILL_CLOSED_EXT = 1.1  # 손가락 펴짐 평균이 이 아래면 아직 주먹 (편 손 1.6~2.4, 주먹 0.6~0.9)


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
    """구간 하나가 끝났을 때(또는 게이트 포즈가 확정됐을 때)의 결과 (UI 표시용)."""
    time: float                      # perf_counter
    label: str                       # 판정 라벨. "skipped"=품질 미달, "gate_open"/"gate_close"=게이트, "cancel"=정적 명령
    confidence: float
    duration_sec: float
    probs: np.ndarray | None
    executed: bool                   # 키를 실제로 눌렀거나(LIVE) 누를 뻔했나(DRY)
    message: str                     # 사람이 읽는 한 줄 (GUARD/cooldown/ignored/[DRY]/[FIRE]/GATE/PENDING)
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
    # 게이트 (YOLO)
    gate_connected: bool = False
    gate_open: bool = True
    gate_error: str = ""
    pose: str | None = None          # 지금 보이는 정적 포즈 (start/stop/cancel)
    pose_hold: float = 0.0
    pose_needed: float = 3.0
    pose_conf: float = 0.0
    pending_label: str | None = None  # 유예 중인 명령 (make_fist)


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


def finger_extension(lm: np.ndarray) -> float:
    """한 프레임의 손가락 펴짐(네 손끝-손목 평균 / 손바닥). 주먹 0.6~0.9, 편 손 1.6~2.4."""
    palm = float(np.linalg.norm(lm[9, :2] - lm[0, :2]))
    if palm < 1e-4:
        return 0.0
    return float(np.linalg.norm(lm[[8, 12, 16, 20], :2] - lm[0, :2], axis=1).mean() / palm)


class Pipeline:
    """한 프레임씩 `step(frame)` 을 부르면 된다. 스레드 안전: `state` 는 lock 으로 복사해 읽는다."""

    def __init__(self, model_file: Path | None = None, gestures_path: Path | None = None,
                 dry_run: bool = True, log_echo: bool = False, gate=None, use_gate: bool | None = None,
                 static_model: str | Path | None = None, device: int | str | None = None):
        self.clf = GestureClassifier(model_file=model_file) if model_file else GestureClassifier()
        self.mapper = ActionMapper(gestures_path) if gestures_path else ActionMapper()
        self.mapper.dry_run = dry_run or not _IS_WIN
        self.seg = MotionSegmenter()
        self._lock = threading.Lock()
        self.state = FrameState()
        self._t0 = time.perf_counter()
        self._fps_n, self._fps_t = 0, self._t0
        self._pending = None            # (label, conf, seg_start, deadline)
        self._last_lm = None

        # ── 게이트 ──
        self.gate = gate                # StaticGate / FakeGate / None
        self.gate_error = ""
        cfg = self.mapper.gate
        if static_model is not None:
            cfg["model"] = str(config.project_path(static_model))
        if device is not None:
            cfg["device"] = device
        if self.gate is None and (cfg.get("enabled", True) if use_gate is None else use_gate):
            from .gate import StaticGate
            self.gate = StaticGate(cfg.get("model"), confidence=float(cfg.get("confidence", 0.7)),
                                   hold_seconds=float(cfg.get("hold_seconds", 3.0)),
                                   miss_tolerance_seconds=float(cfg.get("miss_tolerance_seconds", 0.3)),
                                   imgsz=int(cfg.get("imgsz", 640)), device=cfg.get("device"),
                                   iou=float(cfg.get("iou", 0.5)), max_det=int(cfg.get("max_det", 10)))
        self.gate_connected = self.gate is not None
        self.gate_open = not self.gate_connected    # 게이트가 있으면 손바닥 3초로 열어야 시작
        self.state.gate_connected = self.gate_connected
        self.state.gate_open = self.gate_open
        self.tracker = HandTracker(track_conf=TRACK_CONF, presence_conf=PRESENCE_CONF)
        self.log = SessionLog(echo=log_echo)

        self.log.write(f"모델 {self.clf.arch} | 라벨 {self.clf.labels} | 매핑: "
                       + ", ".join(f"{k}->{self.mapper.describe(k)}" for k in list(self.clf.labels) + ["cancel"]))
        s = self.seg
        self.log.write(f"segmenter on=max({s.on_thresh},{s.on_over_floor}*floor) off=max({s.off_thresh},{s.off_over_floor}*floor) "
                       f"on_frames={s.on_frames} off_frames={s.off_frames} min_sec={s.min_sec} "
                       f"| quality det>={REALTIME_MIN_DET} gap<={REALTIME_MAX_GAP} | track_conf={TRACK_CONF} presence={PRESENCE_CONF} "
                       f"| min_conf={self.mapper.min_confidence} | mode={'DRY' if self.mapper.dry_run else 'LIVE'}")
        if self.gate is not None:
            self.log.write(f"GATE  YOLO {getattr(self.gate, 'path', '?')} hold={self.gate.hold_seconds}s "
                           f"{'GPU' if getattr(self.gate, 'on_gpu', False) else 'CPU'} | 시작=닫힘(손바닥 {self.gate.hold_seconds:.0f}초로 열기)")
        else:
            self.log.write(f"GATE  없음(항상 열림){' | ' + self.gate_error if self.gate_error else ''}")

    # ── 외부에서 바꾸는 것 ──
    @property
    def dry_run(self) -> bool:
        return self.mapper.dry_run

    def set_dry_run(self, dry: bool):
        if dry == self.mapper.dry_run:
            return
        self.mapper.dry_run = dry or not _IS_WIN
        self.log.write("MODE  " + ("DRY RUN (연습)" if dry else "LIVE (실제 키 입력)"))

    def set_gate(self, is_open: bool, source: str = "yolo"):
        """게이트 열기/닫기. YOLO 가 부르거나(source='yolo') UI 수동 테스트(source='manual')."""
        if is_open and self.gate_error:
            self.log.write("GATE  오류 해결 후 프로그램을 다시 시작하세요")
            return
        if is_open != self.gate_open:
            self.gate_open = is_open
            self.log.write(f"GATE  {'OPEN' if is_open else 'CLOSED'} ({source})")
        if not is_open:
            self._pending = None
            self.seg.reset()

    def set_gate_hold(self, sec: float):
        if self.gate is not None:
            self.gate.set_hold_seconds(sec)
        self.mapper.gate["hold_seconds"] = float(sec)

    def reset(self):
        self.seg.reset()
        self._pending = None
        if self.gate is not None:
            self.gate.reset()
        self.log.write("RESET")

    def reload_mapping(self):
        self.mapper.reload()
        self.log.write("MAP   " + ", ".join(f"{k}->{self.mapper.describe(k)}" for k in self.clf.labels))

    # ── 매 프레임 ──
    def step(self, frame_bgr: np.ndarray) -> FrameState:
        now = time.perf_counter()
        ts_ms = (now - self._t0) * 1000.0
        lm, hand = self.tracker.process(frame_bgr, int(ts_ms))
        self._last_lm = lm
        seg = self.seg
        events: list[Event] = []

        # 1) 정적 포즈 게이트
        gs = None
        if self.gate is not None:
            try:
                gs = self.gate.process(frame_bgr)
            except Exception as e:
                self.gate_error = f"{type(e).__name__}: {e}"
                self.log.write(f"GATE  error {self.gate_error} → 인식 중지")
                self.set_gate(False, "error")
                self.gate = None
                self.gate_connected = False
            if gs is not None and gs.confirmed:
                events.append(self._on_pose(gs.confirmed, gs.confidence, now))

        # 2) 동적 제스처
        segment = seg.push(lm, ts_ms, hand)
        if seg.last_drop:
            self.log.write(f"DROP  {seg.last_drop}")
            seg.last_drop = None
        if segment is not None:
            events.append(self._judge(segment, now))

        # 3) 유예 중인 주먹 명령
        if self._pending is not None and now >= self._pending[3]:
            events.append(self._resolve_pending(now))

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
            st.gate_connected, st.gate_open, st.gate_error = self.gate_connected, self.gate_open, self.gate_error
            if gs is not None:
                st.pose, st.pose_hold, st.pose_needed, st.pose_conf = gs.pose, gs.hold, gs.hold_needed, gs.confidence
            else:
                st.pose, st.pose_hold, st.pose_conf = None, 0.0, 0.0
            st.pending_label = self._pending[0] if self._pending else None
            for event in events:
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
                              n_executed=st.n_executed, events=list(st.events),
                              gate_connected=st.gate_connected, gate_open=st.gate_open, gate_error=st.gate_error,
                              pose=st.pose, pose_hold=st.pose_hold, pose_needed=st.pose_needed, pose_conf=st.pose_conf,
                              pending_label=st.pending_label)
        return copy

    # ── 내부 ──
    def _on_pose(self, pose: str, conf: float, now: float) -> Event:
        """YOLO 포즈가 3초 유지되어 확정됐을 때."""
        if pose == "start":
            self.set_gate(True, "yolo")
            self.log.write(f"POSE  start {conf:.2f} -> 게이트 열림")
            return Event(now, "gate_open", conf, 0.0, None, False, f"손바닥 {self.gate.hold_seconds:.0f}초 → 인식 시작")
        if pose == "stop":
            self.set_gate(False, "yolo")
            self.log.write(f"POSE  stop {conf:.2f} -> 게이트 닫힘")
            return Event(now, "gate_close", conf, 0.0, None, False, f"주먹 {self.gate.hold_seconds:.0f}초 → 인식 끝")
        # cancel = 정적 명령
        if not self.gate_open:
            msg = "GATE closed -> cancel ignored"
        else:
            msg = self.mapper.handle("cancel", conf, now)
        executed = msg.startswith("[DRY]") or msg.startswith("[FIRE]")
        self.log.write(f"POSE  cancel {conf:.2f} -> {msg}")
        return Event(now, "cancel", conf, 0.0, None, executed, msg)

    def _resolve_pending(self, now: float) -> Event:
        label, conf, seg_start, _deadline = self._pending
        self._pending = None
        lm = self._last_lm
        still_fist = lm is not None and finger_extension(lm) < FIST_STILL_CLOSED_EXT
        if still_fist:
            msg = f"held fist {FIST_DEFER_SEC}s -> treated as gate-stop intent, ignored"
            executed = False
        else:
            msg = self.mapper.handle(label, conf, now, start=seg_start)
            executed = msg.startswith("[DRY]") or msg.startswith("[FIRE]")
        self.log.write(f"DEFER {label} -> {msg}")
        return Event(now, label, conf, 0.0, None, executed, msg)

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
        elif (label == "make_fist" and self.gate_connected and conf >= self.mapper.min_confidence
              and not self.mapper.in_cooldown(seg_start)):
            # 게이트 닫기용 주먹(3초 유지)과 구분하려고 잠깐 미룬다
            self._pending = (label, float(conf), seg_start, now + FIST_DEFER_SEC)
            msg = f"PENDING {FIST_DEFER_SEC}s (fist held? -> gate stop)"
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
