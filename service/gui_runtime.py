"""GUI 비종속 단일 작업자. 모델/카메라/추적기 소유권과 수명은 이 스레드 하나에 있습니다."""

from contextlib import ExitStack
from dataclasses import dataclass
import threading
import time
from .policy import COMMAND_LABELS, supported_labels


@dataclass(frozen=True)
class CommandEvent:
    session: int
    serial: int
    gesture: str
    target: tuple[int, int]
    expires: float
    frame_s: float | None = None


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


class RecognitionRuntime:
    """호출 측이 launch()할 때만 실행. UI는 poll()로 최신 상태 한 개를 가져갑니다.

    프레임/명령도 슬롯 한 개만 보관하므로 GUI가 지연되어도 큐가 늘지 않습니다.
    폴링은 GUI 스레드에서 위젯을 갱신하며 worker는 Qt 객체를 건드리지 않습니다.
    """

    def __init__(self, checkpoint, device, foreground, *, predictor_factory=default_predictor,
                 controller_factory=default_controller, camera_factory=default_camera,
                 tracker_factory=default_tracker, clock=time.monotonic):
        self.checkpoint, self.device, self.foreground = checkpoint, device, foreground
        self.predictor_factory, self.controller_factory = predictor_factory, controller_factory
        self.camera_factory, self.tracker_factory, self.clock = camera_factory, tracker_factory, clock
        self.condition = threading.Condition()
        self.thread = None
        self.request = None
        self.quitting = False
        self.preview = True
        self.diagnostics = None
        self.frame = self.command = None
        self.status = dict(ready=False, model_status="모델 준비 전", labels=(), schema="", camera="해제",
                           recognition="OFF", detail="", error="", session=None)

    def launch(self):
        if self.thread is not None:
            raise RuntimeError("작업자는 한 번만 생성할 수 있습니다.")
        self.thread = threading.Thread(target=self._run, name="gesture-worker", daemon=False)
        self.thread.start()

    def start(self, session, camera_index):
        with self.condition:
            if self.quitting or not self.status["ready"]:
                raise RuntimeError("모델이 준비되지 않았거나 종료 중입니다.")
            self.request = (session, camera_index)
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

    def close(self):
        with self.condition:
            self.quitting = True
            self.request = None
            self.frame = self.command = None
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
            packet = dict(self.status), self.frame, self.command
            self.frame = self.command = None
            return packet

    def _status(self, **values):
        with self.condition:
            self.status.update(values)

    def _current(self, request):
        with self.condition:
            return not self.quitting and self.request == request

    def _run(self):
        try:
            self._status(model_status="모델 로딩 중")
            predictor = self.predictor_factory(self.checkpoint, self.device)
            controller = self.controller_factory(predictor)
            self._status(ready=True, model_status="모델 준비 완료", labels=supported_labels(predictor.labels),
                         schema=predictor.gesture_schema)
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.quitting or self.request is not None)
                    if self.quitting:
                        break
                    request = self.request
                try:
                    self._session(controller, predictor, request)
                except Exception as exc:
                    self._status(error=f"카메라/인식 오류: {exc}")
                    with self.condition:
                        if self.request == request:
                            self.request = None
                finally:
                    self._status(camera="해제", recognition="OFF", session=request[0])
                    with self.condition:
                        self.frame = self.command = None
        except Exception as exc:
            self._status(ready=False, model_status="모델 로딩/작업자 실패", error=str(exc), recognition="OFF")
        finally:
            self._status(camera="해제", recognition="OFF")

    def _session(self, controller, predictor, request):
        if self.diagnostics is not None:
            controller.diagnostics = lambda event, **fields: self.diagnostics.emit(
                event, session=request[0], **fields)
            self.diagnostics.emit("session_start", session=request[0], labels=list(predictor.labels),
                                  config=vars(controller.config))
        with ExitStack() as resources:
            resources.callback(controller.stop)
            self._status(camera="연결 중", recognition="WARMUP", session=request[0], error="")
            camera = self.camera_factory(request[1])
            resources.callback(camera.release)
            if not self._current(request):
                return
            tracker = self.tracker_factory(predictor.gesture_schema)
            resources.callback(tracker.close)
            controller.start()
            self._status(camera="사용 중")
            last_target, serial = None, 0
            while self._current(request):
                target = self.foreground()
                ok, frame = camera.read()
                timestamp = self.clock()
                if not self._current(request):
                    return
                if not ok or frame is None:
                    raise RuntimeError("카메라 프레임 읽기 실패")
                if last_target is not None and target != last_target:
                    controller.reset_after_gap()  # 앱 전환 시 새 구간/중립부터 다시 판정합니다.
                last_target = target
                hand = tracker.process(frame, timestamp)
                gesture = controller.step(timestamp, hand.landmarks, hand.detected, hand.handedness)
                now = self.clock()
                if now - timestamp >= controller.config.reset_gap_seconds or self.foreground() != target:
                    if self.diagnostics is not None:
                        self.diagnostics.emit("runtime_hold", session=request[0], frame_s=timestamp,
                                              reason="processing_delay_or_target_change", discarded=gesture)
                    controller.reset_after_gap()
                    gesture = None
                mode = ("WARMUP" if controller.segment_start is None or
                        now-controller.segment_start < controller.config.window_seconds else "ACTIVE")
                # 추론/카메라 읽기 도중 OFF가 되었으면 아무 결과도 발행하지 않습니다.
                with self.condition:
                    if self.quitting or self.request != request:
                        return
                    self.status.update(recognition=mode, detail=f"{hand.status} · {controller.reason}",
                                       camera="사용 중", session=request[0])
                    if self.preview:
                        self.frame = frame.copy()
                    if gesture is not None:
                        serial += 1
                        if self.diagnostics is not None and self.command is not None:
                            self.diagnostics.emit("command_replaced", session=request[0],
                                                  serial=self.command.serial, reason="latest_slot_overwritten")
                        self.command = CommandEvent(request[0], serial, gesture, target,
                                                    timestamp+controller.config.reset_gap_seconds, timestamp)
                        if self.diagnostics is not None:
                            self.diagnostics.emit("command_queued", session=request[0], serial=serial,
                                                  frame_s=timestamp, label=gesture)
