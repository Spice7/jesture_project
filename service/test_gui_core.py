"""실제 카메라/Win32 등록/키 전송 없이 GUI의 비시각적 계약을 검사합니다."""

import copy
import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from service.settings import defaults, validate_settings, SettingsStore, DEFAULT_CHECKPOINT, resolve_checkpoint
from service.shortcuts import Shortcut, ChordCapture, HotkeyManager, KeySender
from service.gui_runtime import RecognitionRuntime, SessionGate, CommandEvent
from service.windows_input import INPUT, input_array, VK, WindowsInput


class FakeInput:
    def __init__(self, hwnd=0):
        self.process_id = 1
        self.target = (100, 2)
        self.pressed = False
        self.events = []
        self.registered = {}
        self.registration_failure = False
        self.removed = []
        self.send_count = None

    def foreground(self):
        return self.target

    def any_pressed(self, keys):
        return self.pressed

    def send_events(self, events):
        self.events.extend(events)
        return len(events) if self.send_count is None else self.send_count

    def register_hotkey(self, identifier, shortcut):
        if self.registration_failure:
            raise OSError("registered elsewhere")
        self.registered[identifier] = shortcut

    def unregister_hotkey(self, identifier):
        self.removed.append(identifier)
        del self.registered[identifier]


class CoreTests(unittest.TestCase):
    def test_requested_windows_actions_save_send_and_cannot_be_toggle(self):
        for modifiers, key in ((("Win",), "D"), (("Win", "Shift"), "S")):
            with self.subTest(key=key):
                chord = Shortcut(modifiers, key)
                data = defaults()
                data["shortcuts"]["make_fist"] = chord.to_dict()
                self.assertEqual(validate_settings(data)["shortcuts"]["make_fist"], chord.to_dict())
                backend = FakeInput()
                sender = KeySender(backend)
                self.assertTrue(sender.send(chord, backend.target)[0])
                keys = [*chord.modifiers, key]
                self.assertEqual(backend.events,
                                 [(k, True) for k in keys] + [(k, False) for k in reversed(keys)])
                self.assertFalse(sender.owned)
                data["toggle"] = chord.to_dict()
                with self.assertRaises(ValueError):
                    validate_settings(data)
                with self.assertRaises(ValueError):
                    HotkeyManager(backend).replace(chord)
                self.assertFalse(backend.registered)
        for modifiers, key in ((("Win",), "S"), (("Win", "Ctrl"), "D"), (("Win",), "L")):
            with self.assertRaises(ValueError):
                Shortcut(modifiers, key)

    def test_default_model_and_explicit_saved_precedence(self):
        # 사용자가 기본 모델을 교체해도 경로 선택 우선순위 계약은 동일합니다.
        self.assertTrue(DEFAULT_CHECKPOINT.is_absolute())
        self.assertEqual(DEFAULT_CHECKPOINT.name, "best_model.pt")
        self.assertEqual(resolve_checkpoint(), DEFAULT_CHECKPOINT)
        self.assertEqual(resolve_checkpoint(saved=""), DEFAULT_CHECKPOINT)
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "saved.pt"
            explicit = Path(directory) / "explicit.pt"
            self.assertEqual(resolve_checkpoint(saved=str(saved)), saved)
            self.assertEqual(resolve_checkpoint(explicit, str(saved)), explicit)
            # 해석 단계에서 모델을 읽거나 잘못 지정한 경로를 기본값으로 숨기지 않습니다.
            self.assertFalse(saved.exists())
            with mock.patch("pathlib.Path.cwd", return_value=Path(directory)):
                self.assertEqual(resolve_checkpoint(), DEFAULT_CHECKPOINT)

    def test_import_without_runtime_effects(self):
        script = """
import importlib
from unittest import mock
with mock.patch('ctypes.WinDLL', create=True, side_effect=AssertionError('native API')), \\
     mock.patch('threading.Thread.start', side_effect=AssertionError('worker')), \\
     mock.patch('pathlib.Path.write_text', side_effect=AssertionError('settings write')):
    for name in ('service.gui', 'service.gui_runtime', 'service.settings', 'service.shortcuts', 'service.windows_input'):
        importlib.import_module(name)
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_and_partial_settings(self):
        data = validate_settings({"notifications": False, "shortcuts": {"make_fist": Shortcut(("Ctrl",), "C").to_dict()}})
        self.assertEqual(Shortcut.from_dict(data["shortcuts"]["swipe_left"]).key, "Left")
        self.assertNotIn("swipe_right", data["shortcuts"])
        self.assertEqual(Shortcut.from_dict(defaults()["shortcuts"]["make_fist"]).key, "Space")
        self.assertEqual(Shortcut.from_dict(data["toggle"]).text(), "Ctrl+Alt+G")
        self.assertFalse(data["notifications"])
        self.assertNotIn("no_gesture", data["shortcuts"])

    def test_reject_settings_and_conflict(self):
        changes = [{"version": True}, {"version": 99}, {"camera_index": -1}, {"camera_index": True},
                   {"preview": 1}, {"model_path": None}, {"unknown": 2},
                   {"shortcuts": {"no_gesture": Shortcut((), "A").to_dict()}},
                   {"toggle": Shortcut((), "G").to_dict()},
                   {"shortcuts": {"make_fist": defaults()["toggle"]}}]
        for value in changes:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_settings(value)

    def test_capture_chord_release_repeat_and_cancel(self):
        capture = ChordCapture()
        capture.press("Ctrl")
        capture.press("C")
        self.assertEqual(capture.value.text(), "Ctrl+C")
        capture.press("C", repeat=True)
        self.assertEqual(capture.value.text(), "Ctrl+C")
        capture.release("C")
        capture.release("Ctrl")
        capture.press("C")
        self.assertEqual(capture.value.text(), "C")
        capture.press("Esc")
        self.assertTrue(capture.cancelled)
        self.assertIsNone(capture.value)
        capture.reset()
        capture.press("Ctrl")
        self.assertIsNone(capture.value)
        capture.release("Ctrl")
        capture.press("A")
        with self.assertRaises(ValueError):
            capture.press("B")
        self.assertIsNone(capture.value)

    def test_shortcut_reserved_and_structured_validation(self):
        for mods, key in [((), "Esc"), ((), "unknown"), (("Ctrl", "Ctrl"), "C"),
                          (("Win",), "L"), (("Alt",), "F4"), (("Ctrl", "Alt"), "Delete")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                Shortcut(mods, key)
        self.assertEqual(Shortcut(("Shift", "Ctrl"), "S").text(), "Ctrl+Shift+S")
        self.assertEqual(Shortcut(("Win",), "F2").text(), "Win+F2")
        with self.assertRaises(ValueError):
            Shortcut.from_dict({"modifiers": "Ctrl", "key": "C"})

    def test_atomic_settings_roundtrip_and_failed_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = SettingsStore(path)
            self.assertEqual(store.load(), defaults())
            self.assertFalse(path.exists())
            store.save(defaults())
            before = path.read_bytes()
            data = defaults()
            data["preview"] = False
            with mock.patch("service.settings.os.replace", side_effect=OSError("disk error")), self.assertRaises(OSError):
                store.save(data)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            store.save(data)
            self.assertEqual(SettingsStore(path).load(), data)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "다른"):
                store.save(defaults())

    def test_corrupt_settings_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for payload in (b"broken", b'{"version": 1, "version": 2}', b'{"version": 99}'):
                path.write_bytes(payload)
                store = SettingsStore(path)
                with self.assertRaises(ValueError):
                    store.load()
                with self.assertRaises(ValueError):
                    store.save(defaults())
                self.assertEqual(path.read_bytes(), payload)

    def test_hotkey_replace_conflict_and_save_rollback(self):
        backend = FakeInput()
        manager = HotkeyManager(backend)
        first, second = Shortcut(("Ctrl", "Alt"), "G"), Shortcut(("Ctrl", "Alt"), "J")
        manager.replace(first)
        old = manager.active_id
        backend.registration_failure = True
        with self.assertRaises(OSError):
            manager.replace(second)
        self.assertEqual(backend.registered, {old: first})
        backend.registration_failure = False
        with self.assertRaises(OSError):
            manager.replace(second, mock.Mock(side_effect=OSError("disk")))
        self.assertEqual(backend.registered, {old: first})
        self.assertEqual(manager.current, first)
        manager.replace(second)
        self.assertEqual(list(backend.registered.values()), [second])
        manager.close()
        manager.close()
        self.assertEqual(backend.registered, {})

    def test_sender_order_and_physical_foreground_gates(self):
        backend = FakeInput()
        sender = KeySender(backend)
        chord = Shortcut(("Ctrl", "Shift"), "S")
        self.assertTrue(sender.send(chord, (100, 2))[0])
        self.assertEqual(backend.events, [("Ctrl", True), ("Shift", True), ("S", True),
                                           ("S", False), ("Shift", False), ("Ctrl", False)])
        self.assertEqual(sender.owned, [])
        backend.events.clear()
        backend.pressed = True
        self.assertFalse(sender.send(chord, (100, 2))[0])
        backend.pressed = False
        self.assertFalse(sender.send(chord, (999, 2))[0])
        backend.target = (100, 1)
        self.assertFalse(sender.send(chord, (100, 1))[0])
        self.assertEqual(backend.events, [])

    def test_partial_send_releases_only_successful_owned_keys(self):
        backend = FakeInput()
        sender = KeySender(backend)
        results = iter([2, 1, 1])
        backend.send_events = mock.Mock(side_effect=lambda events: next(results))
        with self.assertRaisesRegex(RuntimeError, "전송 실패"):
            sender.send(Shortcut(("Ctrl", "Shift"), "S"), (100, 2))
        self.assertEqual(backend.send_events.call_args_list[1:], [mock.call([("Shift", False)]), mock.call([("Ctrl", False)])])
        self.assertEqual(sender.owned, [])

    def test_zero_send_and_release_failure(self):
        backend = FakeInput()
        sender = KeySender(backend)
        backend.send_count = 0
        with self.assertRaises(RuntimeError):
            sender.send(Shortcut((), "A"), (100, 2))
        self.assertEqual(sender.owned, [])
        sender.owned = ["Ctrl"]
        with self.assertRaises(RuntimeError):
            sender.release_owned()
        self.assertEqual(sender.owned, ["Ctrl"])
        backend.send_count = None
        sender.release_owned()
        self.assertEqual(sender.owned, [])

    def test_native_structures_and_no_repeat_using_fake_dll(self):
        if sys.platform != "win32":
            self.skipTest("Windows ABI test")
        self.assertEqual(ctypes.sizeof(INPUT), 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        inputs = input_array([("Left", True), ("Left", False)])
        self.assertEqual(inputs[0].ki.wVk, VK["Left"])
        self.assertEqual(inputs[0].ki.dwFlags, 1)
        self.assertEqual(inputs[1].ki.dwFlags, 3)
        dll = mock.Mock()
        dll.SendInput.return_value = 2
        with mock.patch("ctypes.WinDLL", return_value=dll):
            native = WindowsInput(100)
        native.register_hotkey(7, Shortcut(("Ctrl", "Alt"), "G"))
        dll.RegisterHotKey.assert_called_once_with(100, 7, 0x4000 | 3, ord("G"))
        native.send_events([("A", True), ("A", False)])
        self.assertEqual(dll.SendInput.call_args.args[0], 2)
        self.assertEqual(dll.SendInput.call_args.args[2], ctypes.sizeof(INPUT))

    def test_session_gates_duplicate_late_off_settings_and_labels(self):
        gate = SessionGate()
        self.assertTrue(gate.start())
        event = CommandEvent(gate.session, 1, "swipe_left", (100, 2), 10)
        self.assertTrue(gate.accept(event, 1, ("swipe_left",)))
        self.assertFalse(gate.accept(event, 1, ("swipe_left",)))
        self.assertFalse(gate.accept(CommandEvent(gate.session, 2, "no_gesture", (100, 2), 10), 1, ("no_gesture",)))
        self.assertFalse(gate.accept(CommandEvent(gate.session, 3, "swipe_right", (100, 2), 10), 1, ("swipe_left",)))
        self.assertFalse(gate.accept(CommandEvent(gate.session, 4, "swipe_right", (100, 2), 10), 1, ("swipe_left", "swipe_right")))
        gate.stop()
        self.assertFalse(gate.accept(event, 1, ("swipe_left",)))
        gate.settings_open = True
        self.assertFalse(gate.start())
        gate.settings_open = False
        gate.start()
        late = CommandEvent(gate.session, 1, "swipe_left", (100, 2), 2)
        self.assertFalse(gate.accept(late, 2, ("swipe_left",)))
        gate.closing = True
        self.assertFalse(gate.accept(CommandEvent(gate.session, 2, "swipe_left", (100, 2), 10), 1, ("swipe_left",)))


class RuntimeTests(unittest.TestCase):
    def test_runtime_exposes_right_hand_service_labels(self):
        self.runtime.launch()
        self.wait_for(lambda: self.runtime.poll()[0]["ready"])
        status = self.runtime.poll()[0]
        self.assertEqual(status["labels"], ("swipe_left", "make_fist", "no_gesture"))
        from service.gui_runtime import default_tracker
        with mock.patch("service.hand_tracker.HandTracker") as tracker:
            default_tracker("right-only")
        tracker.assert_called_once_with()

    def setUp(self):
        self.predictor = SimpleNamespace(labels=("swipe_left", "make_fist", "no_gesture"), gesture_schema="right-only")
        self.controller = mock.Mock()
        self.controller.segment_start = 0
        self.controller.reason = "test"
        self.controller.config = SimpleNamespace(reset_gap_seconds=.75, window_seconds=1.5)
        self.controller.step.return_value = "swipe_left"
        self.camera, self.tracker = mock.Mock(), mock.Mock()
        self.camera.read.return_value = True, np.zeros((10, 20, 3), dtype=np.uint8)
        self.tracker.process.return_value = SimpleNamespace(landmarks=np.zeros((21, 3)), detected=True, handedness="Right", status="Right")
        # 게이트는 별도 테스트에서 다루고 여기서는 기존 LSTM 경로만 확인합니다.
        self.runtime = RecognitionRuntime("synthetic-only", "cpu", lambda: (100, 2),
            predictor_factory=lambda *a: self.predictor, controller_factory=lambda p: self.controller,
            camera_factory=mock.Mock(return_value=self.camera), tracker_factory=mock.Mock(return_value=self.tracker),
            gate_factory=None)
        self.addCleanup(self.cleanup)

    def loop(self, request=None):
        """카메라 루프를 현재 세대로 한 번 실행합니다."""
        self.runtime.request = request
        return self.runtime._camera_loop(self.controller, self.predictor, None,
                                         self.runtime.camera_index, self.runtime.camera_generation)

    def cleanup(self):
        self.runtime.close()
        if self.runtime.thread:
            self.runtime.thread.join(2)
            self.assertFalse(self.runtime.alive())

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("worker timeout")
            threading.Event().wait(.005)

    def test_launch_opens_camera_once_and_rejects_duplicate_launch(self):
        factory = mock.Mock(return_value=self.predictor)
        self.runtime.predictor_factory = factory
        self.runtime.launch()
        # 인식 OFF여도 카메라는 앱 수명 동안 열려 있습니다. 게이트를 항상 확인하기 때문입니다.
        self.wait_for(lambda: self.runtime.poll()[0]["camera"] == "사용 중")
        self.assertTrue(self.runtime.poll()[0]["ready"])
        factory.assert_called_once()
        self.runtime.camera_factory.assert_called_once_with(0)
        self.controller.start.assert_not_called()
        with self.assertRaises(RuntimeError):
            self.runtime.launch()

    def test_disarmed_loop_skips_tracker_and_controller(self):
        calls = 0
        def read():
            nonlocal calls
            calls += 1
            if calls >= 3:
                self.runtime.close()
            return True, np.zeros((10, 20, 3), dtype=np.uint8)
        self.camera.read.side_effect = read
        self.loop(None)
        self.tracker.process.assert_not_called()
        self.controller.step.assert_not_called()
        self.assertIsNone(self.runtime.poll()[2])
        self.camera.release.assert_called_once()
        self.tracker.close.assert_called_once()

    def test_session_preview_hidden_and_resource_cleanup(self):
        self.runtime.preview = False
        def step(*args):
            # 첫 프레임은 발행, 다음 프레임에서 중지하여 뒤늦은 명령이 발행되지 않음을 검사.
            if self.controller.step.call_count == 2:
                self.assertIsNone(self.runtime.poll()[1])
                self.runtime.close()
            return "swipe_left"
        self.controller.step.side_effect = step
        self.loop((1, 0))
        self.assertEqual(self.controller.step.call_count, 2)
        self.assertIsNone(self.runtime.poll()[2])
        self.camera.release.assert_called_once()
        self.tracker.close.assert_called_once()
        self.controller.stop.assert_called()

    def test_late_frame_error_and_tracker_failure_release(self):
        self.camera.read.return_value = False, None
        with self.assertRaises(RuntimeError):
            self.loop((1, 0))
        self.camera.release.assert_called_once()
        self.tracker.close.assert_called_once()
        self.camera.reset_mock()
        self.runtime.tracker_factory.side_effect = RuntimeError("detector failed")
        with self.assertRaises(RuntimeError):
            self.loop((1, 0))
        self.camera.release.assert_called_once()

    def test_worker_load_failure_never_opens_camera(self):
        self.runtime.predictor_factory = mock.Mock(side_effect=ValueError("invalid checkpoint"))
        self.runtime.launch()
        self.runtime.thread.join(2)
        status = self.runtime.poll()[0]
        self.assertFalse(status["ready"])
        self.assertIn("invalid checkpoint", status["error"])
        self.runtime.camera_factory.assert_not_called()

    def gate_stub(self, actions):
        """확정 동작을 순서대로 내보내는 가짜 게이트입니다. YOLO 가중치를 쓰지 않습니다."""
        from service.yolo_gate import GateConfig, HoldDetector
        stub = mock.Mock()
        stub.detector = HoldDetector(GateConfig())
        stub.step.side_effect = [(action, "stop" if action == "disarm" else "start", .9, 1.)
                                 for action in actions]
        return stub

    def test_gate_runs_while_disarmed_and_publishes_confirmed_action(self):
        gate = self.gate_stub([None, "arm"])
        def read():
            # close()는 대기 중인 슬롯을 지우므로, 카메라 세대를 올려 루프만 끝냅니다.
            if gate.step.call_count >= 2:
                self.runtime.camera_generation += 1
            return True, np.zeros((10, 20, 3), dtype=np.uint8)
        self.camera.read.side_effect = read
        self.runtime.request = None
        self.runtime._camera_loop(self.controller, self.predictor, gate, 0, 0)
        # 인식이 꺼져 있어도 게이트는 매 프레임 확인합니다.
        self.assertEqual(gate.step.call_count, 2)
        self.tracker.process.assert_not_called()
        event = self.runtime.poll()[3]
        self.assertIsNotNone(event, "확정된 손모양 동작이 GUI로 올라와야 합니다.")
        self.assertEqual(event.action, "arm")
        self.assertEqual(event.serial, 1)

    def test_gate_sets_command_suppression_on_the_controller(self):
        from service.yolo_gate import GateConfig, HoldDetector
        gate = mock.Mock()
        gate.detector = HoldDetector(GateConfig(hold_seconds=3., suppress_after_seconds=1.))
        for _ in range(15):  # 주먹 1.5초 유지 → 유예 구간
            gate.detector.update(gate.detector.last_timestamp or 100., "stop")
            gate.detector.last_timestamp = (gate.detector.last_timestamp or 100.) + .1
        gate.step.return_value = (None, "stop", .9, .5)
        self.runtime._gate_step(gate, self.controller, gate.detector.last_timestamp, None)
        self.assertEqual(self.controller.suppressed, frozenset({"make_fist"}))

    def test_no_gate_clears_suppression(self):
        self.runtime._gate_step(None, self.controller, 100., None)
        self.assertEqual(self.controller.suppressed, frozenset())

    def test_focus_change_or_slow_inference_drops_event(self):
        for slow in (False, True):
            with self.subTest(slow=slow):
                self.controller.reset_mock()
                self.runtime.clock = mock.Mock(side_effect=[0, 1 if slow else .1])
                targets = iter([(100, 2), (101, 2)])
                self.runtime.foreground = lambda: next(targets)
                self.camera.read.side_effect = None
                def step(*args):
                    self.runtime.close()  # 이 프레임까지만 처리하고 루프를 끝냅니다.
                    return "swipe_left"
                self.controller.step.side_effect = step
                self.loop((1, 0))
                self.assertIsNone(self.runtime.poll()[2])
                self.controller.reset_after_gap.assert_called()
                self.runtime.quitting = False

    def test_stop_during_read_and_restart_no_overlapping_camera(self):
        entered, release = threading.Event(), threading.Event()
        def read():
            entered.set()
            release.wait(2)
            return True, np.zeros((10, 20, 3), dtype=np.uint8)
        self.camera.read.side_effect = read
        self.runtime.launch()
        self.wait_for(lambda: self.runtime.poll()[0]["ready"])
        self.runtime.start(1, 0)
        self.assertTrue(entered.wait(2))
        self.runtime.stop()
        self.runtime.start(3, 0)
        self.runtime.close()
        release.set()
        self.runtime.thread.join(2)
        self.controller.step.assert_not_called()
        self.camera.release.assert_called_once()
        self.runtime.camera_factory.assert_called_once()
        self.assertIsNone(self.runtime.poll()[2])


if __name__ == "__main__":
    unittest.main()
