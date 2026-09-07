"""GUI 비종속 단일 작업자. 모델/카메라/추적기 소유권과 수명은 이 스레드 하나에 있습니다.

카메라와 YOLO 게이트는 앱이 켜져 있는 동안 계속 돌고, MediaPipe와 LSTM은 armed일 때만
실행합니다. 게이트가 확정한 동작은 직접 처리하지 않고 GUI 스레드로 올려 권한 판단을 맡깁니다.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import threading
import time
from .policy import COMMAND_LABELS, supported_labels
from .yolo_gate import suppressed_commands


@dataclass(frozen=True)
class CommandEvent:
    session: int
    serial: int
    gesture: str
    target: tuple[int, int]
    expires: float
    frame_s: float | None = None


@dataclass(frozen=True)
class GateEvent:
    """정적 손모양 유지로 확정한 동작입니다. 실행 여부는 GUI 스레드가 정합니다."""

    serial: int
    action: str
    frame_s: float


class SessionGate:
    """GUI 스레드의 최종 권한 검사. 중지 시 이미 계산 중인 과거 결과도 무효화합니다."""

    def __init__(self):
        self.session = 0
        self.enabled = False
        self.settings_open = False
        self.closing = False
        self.last_serial = 0

    def start(self):
        if self.settings_open or self.closing or self.enabled:
            return False
        self.session += 1
        self.enabled = True
        self.last_serial = 0
        return True

    def stop(self):
        self.enabled = False
        self.session += 1
        self.last_serial = 0

    def accept(self, event, now, labels):
        if (not self.enabled or self.settings_open or self.closing or event.session != self.session
                or event.serial <= self.last_serial):
            return False
        self.last_serial = event.serial  # 오래된/미지원 명령도 소비하여 재시도하지 않습니다.
        return now < event.expires and event.gesture in labels and event.gesture in COMMAND_LABELS


def default_predictor(path, device):
    from .predictor import GesturePredictor
    return GesturePredictor(path, device)


def default_controller(predictor):
    from .gesture_controller import GestureController
    return GestureController(predictor)


def default_camera(index):
    import cv2
    from .main import open_camera
    return open_camera(cv2, index)


def default_tracker(schema):
    from .hand_tracker import HandTracker
    return HandTracker()  # 모델 스키마와 무관하게 오른손 전용입니다.


def default_gate(weights, device):
    from .yolo_gate import DEFAULT_WEIGHTS, YoloGate
    return YoloGate(weights or DEFAULT_WEIGHTS, device)


class RecognitionRuntime:
    """호출 측이 launch()할 때만 실행. UI는 poll()로 최신 상태 한 개를 가져갑니다.

    프레임/명령/게이트 이벤트도 슬롯 한 개만 보관하므로 GUI가 지연되어도 큐가 늘지 않습니다.
    폴링은 GUI 스레드에서 위젯을 갱신하며 worker는 Qt 객체를 건드리지 않습니다.
    """

    def __init__(self, checkpoint, device, foreground, *, camera_index=0, gate_weights=None,
                 predictor_factory=default_predictor, controller_factory=default_controller,
                 camera_factory=default_camera, tracker_factory=default_tracker,
                 gate_factory=default_gate, clock=time.monotonic):
        self.checkpoint, self.device, self.foreground = checkpoint, device, foreground
        self.predictor_factory, self.controller_factory = predictor_factory, controller_factory
        self.camera_factory, self.tracker_factory, self.clock = camera_factory, tracker_factory, clock
        self.gate_factory, self.gate_weights = gate_factory, gate_weights
        self.condition = threading.Condition()
        self.thread = None
        self.request = None
        self.quitting = False
        self.preview = True
        self.diagnostics = None
        self.camera_index = camera_index
        self.camera_generation = 0
        self.frame = self.command = self.gate_event = None
        self.status = dict(ready=False, model_status="모델 준비 전", labels=(), schema="", camera="해제",
                           recognition="OFF", detail="", gate=None, gate_ready=False, gate_error="",
                           error="", session=None)

    def launch(self):
        if self.thread is not None:
            raise RuntimeError("작업자는 한 번만 생성할 수 있습니다.")
        self.thread = threading.Thread(target=self._run, name="gesture-worker", daemon=False)
        self.thread.start()

    def start(self, session, camera_index=None):
        """LSTM 인식을 켭니다. 카메라와 게이트는 이미 돌고 있습니다."""
        with self.condition:
            if self.quitting or not self.status["ready"]:
                raise RuntimeError("모델이 준비되지 않았거나 종료 중입니다.")
            if camera_index is not None and camera_index != self.camera_index:
                self.camera_index = camera_index  # 다음 루프에서 카메라를 다시 엽니다.
            self.request = (session, self.camera_index)
            self.frame = self.command = None
            self.status["error"] = ""
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            if self.diagnostics is not None and self.request is not None:
                self.diagnostics.emit("session_stop", session=self.request[0])
            self.request = None
            self.frame = self.command = None
            self.condition.notify_all()

    def reopen(self, camera_index=None):
        """카메라 오류 후 다시 시도하거나 카메라 번호를 바꿉니다."""
        with self.condition:
            if camera_index is not None:
                self.camera_index = camera_index
            self.camera_generation += 1
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.quitting = True
            self.request = None
            self.frame = self.command = self.gate_event = None
            self.condition.notify_all()

    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def set_preview(self, visible):
        with self.condition:
            self.preview = visible
            if not visible:
                self.frame = None

    def poll(self):
        with self.condition:
            packet = dict(self.status), self.frame, self.command, self.gate_event
            self.frame = self.command = self.gate_event = None
            return packet

    def _status(self, **values):
        with self.condition:
            self.status.update(values)

    def _running(self, index, generation):
        with self.condition:
            return (not self.quitting and self.camera_index == index
                    and self.camera_generation == generation)

    def _run(self):
        try:
            self._status(model_status="모델 로딩 중")
            predictor = self.predictor_factory(self.checkpoint, self.device)
            controller = self.controller_factory(predictor)
            gate = None
            if self.gate_factory is not None:
                # 게이트 로딩 실패로 서비스 전체를 막지 않습니다. 수동 시작/중지는 그대로 됩니다.
                try:
                    gate = self.gate_factory(self.gate_weights, self.device)
                except Exception as exc:
                    self._status(gate_error=f"손모양 게이트를 쓸 수 없습니다: {exc}")
            self._status(ready=True, model_status="모델 준비 완료", gate_ready=gate is not None,
                         labels=supported_labels(predictor.labels), schema=predictor.gesture_schema)
            while True:
                with self.condition:
                    if self.quitting:
                        break
                    index, generation = self.camera_index, self.camera_generation
                try:
                    self._camera_loop(controller, predictor, gate, index, generation)
                except Exception as exc:
                    self._status(error=f"카메라/인식 오류: {exc}", camera="해제", recognition="OFF")
                    with self.condition:
                        self.request = None
                        self.frame = self.command = self.gate_event = None
                        # 같은 오류로 즉시 재시도하지 않고 재시도 요청이나 종료를 기다립니다.
                        self.condition.wait_for(lambda: self.quitting or self.camera_index != index
                                                or self.camera_generation != generation)
        except Exception as exc:
            self._status(ready=False, model_status="모델 로딩/작업자 실패", error=str(exc), recognition="OFF")
        finally:
            self._status(camera="해제", recognition="OFF")

    def _camera_loop(self, controller, predictor, gate, index, generation):
        with ExitStack() as resources:
            resources.callback(controller.stop)
            self._status(camera="연결 중", recognition="OFF", error="")
            camera = self.camera_factory(index)
            resources.callback(camera.release)
            if not self._running(index, generation):
                return
            tracker = self.tracker_factory(predictor.gesture_schema)
            resources.callback(tracker.close)
            if gate is not None:
                gate.reset()
            self._status(camera="사용 중")
            last_target, serial, gate_serial, session = None, 0, 0, None
            while self._running(index, generation):
                target = self.foreground()
                ok, frame = camera.read()
                timestamp = self.clock()
                if not self._running(index, generation):
                    return
                if not ok or frame is None:
                    raise RuntimeError("카메라 프레임 읽기 실패")
                action, gate_label, gate_score, progress = self._gate_step(gate, controller, timestamp, frame)
                session, last_target = self._sync_session(controller, session, last_target)
                gesture, detail, mode = None, "", "OFF"
                if session is not None:
                    if last_target is not None and target != last_target:
                        controller.reset_after_gap()  # 앱 전환 시 새 구간/중립부터 다시 판정합니다.
                    last_target = target
                    hand = tracker.process(frame, timestamp)
                    gesture = controller.step(timestamp, hand.landmarks, hand.detected, hand.handedness)
                    now = self.clock()
                    if now - timestamp >= controller.config.reset_gap_seconds or self.foreground() != target:
                        if self.diagnostics is not None:
                            self.diagnostics.emit("runtime_hold", session=session[0], frame_s=timestamp,
                                                  reason="processing_delay_or_target_change", discarded=gesture)
                        controller.reset_after_gap()
                        gesture = None
                    mode = ("WARMUP" if controller.segment_start is None or
                            now-controller.segment_start < controller.config.window_seconds else "ACTIVE")
                    detail = f"{hand.status} · {controller.reason}"
                # 추론/카메라 읽기 도중 상태가 바뀌었으면 아무 결과도 발행하지 않습니다.
                with self.condition:
                    if self.quitting or self.camera_index != index or self.camera_generation != generation:
                        return
                    self.status.update(recognition=mode, detail=detail, camera="사용 중",
                                       gate=(gate_label, gate_score, progress) if gate is not None else None,
                                       session=session[0] if session is not None else None)
                    if self.preview:
                        self.frame = frame.copy()
                    if gesture is not None and self.request == session:
                        serial += 1
                        if self.diagnostics is not None and self.command is not None:
                            self.diagnostics.emit("command_replaced", session=session[0],
                                                  serial=self.command.serial, reason="latest_slot_overwritten")
                        self.command = CommandEvent(session[0], serial, gesture, target,
                                                    timestamp+controller.config.reset_gap_seconds, timestamp)
                        if self.diagnostics is not None:
                            self.diagnostics.emit("command_queued", session=session[0], serial=serial,
                                                  frame_s=timestamp, label=gesture)
                    if action is not None:
                        gate_serial += 1
                        self.gate_event = GateEvent(gate_serial, action, timestamp)
                        if self.diagnostics is not None:
                            self.diagnostics.emit("gate_confirmed", serial=gate_serial,
                                                  frame_s=timestamp, action=action, label=gate_label)

    def _gate_step(self, gate, controller, timestamp, frame):
        """게이트는 armed 여부와 무관하게 매 프레임 확인합니다."""
        if gate is None:
            controller.suppressed = frozenset()
            return None, None, 0., 0.
        action, label, score, progress = gate.step(timestamp, frame)
        # 주먹 유지가 진행 중이면 make_fist 확정을 미룹니다. 손을 풀면 바로 실행됩니다.
        controller.suppressed = suppressed_commands(gate.detector, timestamp)
        return action, label, score, progress

    def _sync_session(self, controller, session, last_target):
        """GUI가 요청한 armed 상태를 컨트롤러에 반영합니다."""
        with self.condition:
            request = self.request
        if request == session:
            return session, last_target
        if request is None:
            controller.stop()
            return None, None
        controller.start()
        if self.diagnostics is not None:
            self.diagnostics.emit("session_start", session=request[0],
                                  config=vars(controller.config))
        return request, None
