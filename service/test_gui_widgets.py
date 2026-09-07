"""PySide6 설치 시에만 offscreen으로 실행. 실제 Win32/카메라를 생성하지 않습니다."""

import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

from service.settings import defaults, SettingsStore, DEFAULT_CHECKPOINT
from service.test_gui_core import FakeInput
from service.gui_runtime import CommandEvent

HAS_QT = importlib.util.find_spec("PySide6") is not None


class FakeRuntime:
    def __init__(self, *args, camera_index=0, **kwargs):
        # 카메라와 게이트는 실행 직후부터 돌아갑니다. 인식 ON/OFF와는 별개입니다.
        self.status = dict(ready=True, labels=("swipe_left", "make_fist", "no_gesture"),
                          schema="right-only", model_status="모의 모델", camera="사용 중", recognition="OFF",
                          session=None, error="", detail="test", gate=None, gate_ready=True, gate_error="")
        self.launched = False
        self.starts = []
        self.closed = False
        self.preview = True
        self.camera_index = camera_index
        self.reopened = []
        self.frame = self.event = self.gate_event = None

    def launch(self):
        self.launched = True

    def start(self, session, camera=None):
        self.starts.append((session, camera))
        self.status.update(recognition="ACTIVE", session=session)

    def stop(self):
        self.status.update(recognition="OFF")

    def reopen(self, camera_index=None):
        self.reopened.append(camera_index)
        if camera_index is not None:
            self.camera_index = camera_index

    def close(self):
        self.closed = True
        self.status.update(camera="해제")

    def alive(self):
        return not self.closed

    def set_preview(self, value):
        self.preview = value

    def poll(self):
        packet = dict(self.status), self.frame, self.event, self.gate_event
        self.frame = self.event = self.gate_event = None
        return packet


@unittest.skipUnless(HAS_QT, "PySide6 미설치: 실제 GUI/offscreen 검증 미실행")
class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_platform = os.environ.get("QT_QPA_PLATFORM")
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt, QEvent
        from PySide6.QtTest import QTest
        from service import gui_widgets
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        cls.widgets, cls.Qt, cls.QTest, cls.QEvent = gui_widgets, Qt, QTest, QEvent

    @classmethod
    def tearDownClass(cls):
        if cls.old_platform is None:
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = cls.old_platform

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = SettingsStore(Path(self.temporary.name) / "settings.json")
        settings = self.store.load()
        self.window = self.widgets.MainWindow(settings, self.store, checkpoint="synthetic.pt",
            backend_factory=FakeInput, runtime_factory=FakeRuntime, enable_tray=False)
        self.window.poll()

    def tearDown(self):
        self.window.close()
        self.window.poll()
        self.app.processEvents()
        self.temporary.cleanup()

    def test_off_default_and_right_only_buttons(self):
        self.assertFalse(self.window.gate.enabled)
        self.window.open_settings()
        self.assertEqual(set(self.window.dialog.buttons), {"swipe_left", "make_fist", "finger_snap", "toggle"})
        self.assertNotIn("no_gesture", self.window.dialog.buttons)
        self.assertIn("오른손 전용", self.window.mode.text())
        self.assertNotIn("swipe_right", self.window.labels)
        self.assertFalse(self.window.dialog.buttons["finger_snap"].isEnabled())

    def test_finger_snap_unassigned_then_configured_send(self):
        import time
        from unittest import mock
        self.window.runtime.status["labels"] += ("finger_snap",)
        diagnostic = mock.Mock()
        self.window.diagnostics = diagnostic
        self.window.poll()
        self.window.toggle()
        with mock.patch.object(self.window.sender, "send", return_value=(True, "sent")) as send:
            self.window.runtime.event = CommandEvent(
                self.window.gate.session, 1, "finger_snap", (100, 2), time.monotonic()+10)
            self.window.poll()
            send.assert_not_called()
            self.assertIn("미지정", self.window.last_command.text())
            result = [call for call in diagnostic.emit.call_args_list if call.args == ("key_send_result",)]
            self.assertFalse(result[-1].kwargs["sent"])
            self.window.open_settings()
            dialog = self.window.dialog
            self.assertTrue(dialog.buttons["finger_snap"].isEnabled())
            dialog.select("finger_snap")
            self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_K, self.Qt.KeyboardModifier.ControlModifier)
            dialog.save()
            self.assertEqual(self.window.settings["shortcuts"]["finger_snap"]["key"], "K")
            self.window.poll()  # 설정 진입 때 해제한 카메라 상태를 반영한 뒤 다시 시작합니다.
            self.window.toggle()
            self.assertTrue(self.window.gate.enabled)
            self.window.runtime.event = CommandEvent(
                self.window.gate.session, 2, "finger_snap", (100, 2), time.monotonic()+10)
            self.window.poll()
            send.assert_called_once()
            self.assertEqual(send.call_args.args[0].text(), "Ctrl+K")
            result = [call for call in diagnostic.emit.call_args_list if call.args == ("key_send_result",)]
            self.assertTrue(result[-1].kwargs["sent"])
            self.assertEqual(result[-1].kwargs["serial"], 2)

    def gate_event(self, action, serial=1):
        from service.gui_runtime import GateEvent
        self.window.runtime.gate_event = GateEvent(serial, action, 100.)
        self.window.poll()

    def test_hand_shape_gate_arms_and_disarms_recognition(self):
        self.assertFalse(self.window.gate.enabled)
        self.gate_event("arm")
        self.assertTrue(self.window.gate.enabled, "보자기 확정이 인식을 켜야 합니다.")
        self.assertEqual(len(self.window.runtime.starts), 1)
        self.gate_event("arm", 2)  # 이미 켜져 있으면 다시 시작하지 않습니다.
        self.assertEqual(len(self.window.runtime.starts), 1)
        self.gate_event("disarm", 3)
        self.assertFalse(self.window.gate.enabled, "주먹 확정이 인식을 꺼야 합니다.")

    def test_gate_is_ignored_while_settings_are_open_or_closing(self):
        self.window.open_settings()
        self.gate_event("arm")
        self.assertFalse(self.window.gate.enabled, "설정 중에는 손모양으로 켜지지 않습니다.")
        self.window.dialog.reject()
        self.window.gate.closing = True
        self.gate_event("arm", 2)
        self.assertFalse(self.window.gate.enabled)
        self.window.gate.closing = False

    def test_gun_shape_toggles_window_without_stealing_focus(self):
        self.window.show()
        self.assertTrue(self.window.isVisible())
        self.gate_event("toggle_window")
        self.assertFalse(self.window.isVisible(), "총 모양 확정이 창을 숨겨야 합니다.")
        self.gate_event("toggle_window", 2)
        self.assertTrue(self.window.isVisible(), "다시 확정하면 창이 보여야 합니다.")
        # 활성 창을 바꾸면 KeySender가 단축키 전송을 막으므로 포커스를 가져오지 않습니다.
        self.assertTrue(self.window.testAttribute(self.Qt.WidgetAttribute.WA_ShowWithoutActivating))

    def test_camera_and_gate_keep_running_after_recognition_stops(self):
        self.window.toggle()
        self.assertTrue(self.window.gate.enabled)
        self.window.stop()
        self.assertFalse(self.window.gate.enabled)
        self.window.poll()
        self.assertEqual(self.window.worker_camera, "사용 중")
        self.assertIn("손모양 게이트", self.window.gate_status.text())

    def test_changing_camera_number_reopens_the_camera(self):
        self.window.open_settings()
        dialog = self.window.dialog
        dialog.camera.setValue(2)
        dialog.save()
        self.assertEqual(self.window.runtime.reopened, [2])

    def test_default_model_without_cli_or_saved_path(self):
        self.window.close()
        self.window = self.widgets.MainWindow(defaults(), self.store,
            backend_factory=FakeInput, runtime_factory=FakeRuntime, enable_tray=False)
        self.assertEqual(Path(self.window.checkpoint), DEFAULT_CHECKPOINT)
        self.assertTrue(self.window.runtime.launched)
        self.assertFalse(self.window.gate.enabled)

    def test_preview_hide_does_not_stop_and_settings_does(self):
        self.window.toggle()
        self.window.poll()
        self.window.change_preference("preview", False)
        self.assertTrue(self.window.gate.enabled)
        self.assertFalse(self.window.runtime.preview)
        self.assertTrue(self.window.preview_label.isHidden())
        self.assertEqual(len(self.window.runtime.starts), 1)
        self.window.open_settings()
        self.assertFalse(self.window.gate.enabled)
        self.assertIsNone(self.window.hotkeys.active_id)
        self.window.toggle()
        self.assertFalse(self.window.gate.enabled)
        self.window.dialog.reject()
        self.assertFalse(self.window.gate.enabled)
        self.assertIsNotNone(self.window.hotkeys.active_id)

    def test_capture_cancel_save_and_restore(self):
        self.window.open_settings()
        dialog = self.window.dialog
        dialog.select("make_fist")
        self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_C, self.Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(dialog.draft["shortcuts"]["make_fist"]["modifiers"], ["Ctrl"])
        self.assertEqual(self.window.settings["shortcuts"]["make_fist"]["key"], "Space")
        dialog.save()
        self.assertEqual(self.window.settings["shortcuts"]["make_fist"]["key"], "C")
        self.assertEqual(SettingsStore(self.store.path).load(), self.window.settings)
        self.window.open_settings()
        dialog = self.window.dialog
        dialog.restore_shortcuts()
        self.assertEqual(dialog.draft["shortcuts"]["make_fist"]["key"], "Space")
        dialog.reject()
        self.assertEqual(self.window.settings["shortcuts"]["make_fist"]["key"], "C")

    def test_invalid_capture_does_not_save(self):
        self.window.open_settings()
        dialog = self.window.dialog
        dialog.select("make_fist")
        self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_F12)
        dialog.save()
        self.assertIs(self.window.dialog, dialog)
        self.assertFalse(dialog.pending_valid)
        self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_Escape)
        self.assertTrue(dialog.pending_valid)

    def test_windows_modifier_checkbox_capture_save_and_reserved_rejection(self):
        self.window.open_settings()
        dialog = self.window.dialog
        dialog.select("make_fist")
        dialog.win_modifier.setChecked(True)
        self.assertFalse(dialog.pending_valid)
        self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_F2)
        self.assertEqual(dialog.draft["shortcuts"]["make_fist"], {"modifiers": ["Win"], "key": "F2"})
        dialog.save()
        self.assertEqual(self.window.settings["shortcuts"]["make_fist"]["modifiers"], ["Win"])
        self.assertEqual(self.store.load()["shortcuts"]["make_fist"]["key"], "F2")
        self.window.open_settings()
        dialog = self.window.dialog
        dialog.select("make_fist")
        self.assertTrue(dialog.win_modifier.isChecked())
        self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_L)
        self.assertFalse(dialog.pending_valid)
        self.assertIn("예약", dialog.error.text())
        dialog.save()
        self.assertIs(self.window.dialog, dialog)
        dialog.win_modifier.setChecked(False)
        self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_C,
                            self.Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(dialog.draft["shortcuts"]["make_fist"]["modifiers"], ["Ctrl"])

    def test_physical_meta_modifier_capture(self):
        self.window.open_settings()
        dialog = self.window.dialog
        dialog.select("make_fist")
        self.QTest.keyClick(dialog.capture_edit, self.Qt.Key.Key_F2,
                            self.Qt.KeyboardModifier.MetaModifier)
        self.assertEqual(dialog.draft["shortcuts"]["make_fist"], {"modifiers": ["Win"], "key": "F2"})

    def test_requested_windows_actions_capture_and_save(self):
        for key, modifiers, expected in (
            (self.Qt.Key.Key_D, self.Qt.KeyboardModifier.NoModifier, ["Win"]),
            (self.Qt.Key.Key_S, self.Qt.KeyboardModifier.ShiftModifier, ["Shift", "Win"]),
        ):
            self.window.open_settings()
            dialog = self.window.dialog
            dialog.select("make_fist")
            dialog.win_modifier.setChecked(True)
            self.QTest.keyClick(dialog.capture_edit, key, modifiers)
            self.assertTrue(dialog.pending_valid)
            dialog.save()
            self.assertIsNone(self.window.dialog)
            chord = self.store.load()["shortcuts"]["make_fist"]
            self.assertEqual(chord["modifiers"], expected)
            self.assertEqual(chord["key"], "D" if key == self.Qt.Key.Key_D else "S")

    def test_notification_flags_off_and_close_cleanup(self):
        toast = self.window.toast
        self.assertTrue(toast.testAttribute(self.Qt.WidgetAttribute.WA_ShowWithoutActivating))
        self.assertTrue(toast.windowFlags() & self.Qt.WindowType.WindowDoesNotAcceptFocus)
        self.assertTrue(toast.windowFlags() & self.Qt.WindowType.WindowTransparentForInput)
        self.window.change_preference("notifications", False)
        self.window.notify("should hide")
        self.assertFalse(toast.isVisible())
        self.window.close()
        self.window.poll()
        self.assertTrue(self.window.runtime.closed)
        self.assertEqual(self.window.backend.registered, {})

    def test_late_events_error_and_minimize(self):
        import time
        from unittest import mock
        self.window.toggle()
        session = self.window.gate.session
        self.window.stop()
        self.window.runtime.event = CommandEvent(session, 1, "swipe_left", (100, 2), time.monotonic()+1)
        self.window.poll()
        self.assertEqual(self.window.backend.events, [])
        self.window.toggle()
        self.window.runtime.status["error"] = "mock camera error"
        self.window.poll()
        self.assertFalse(self.window.gate.enabled)
        self.assertIn("mock camera error", self.window.error.text())
        # 실제 생성 경로는 트레이와 두 메뉴 항목을 함께 만듭니다. fake도 같은 구성을
        # 제공하고 종료까지 검사합니다. 범위 밖에서는 원래의 트레이 없음 상태로 복원합니다.
        tray, status, toggle = mock.Mock(), mock.Mock(), mock.Mock()
        with mock.patch.multiple(self.window, tray=tray, tray_status=status, tray_toggle=toggle, create=True):
            with mock.patch.object(self.window, "isMinimized", return_value=True), mock.patch.object(self.window, "hide") as hide:
                self.window.changeEvent(self.QEvent(self.QEvent.Type.WindowStateChange))
                self.app.processEvents()
                hide.assert_called()
            self.window.poll()
            # 인식만 꺼지고 카메라와 손모양 게이트는 계속 돌아갑니다.
            status.setText.assert_called_with("인식 OFF · 카메라 사용 중")
            toggle.setEnabled.assert_called()
            self.window.close()
            self.window.poll()
            tray.hide.assert_called_once()
            self.assertTrue(self.window.cleanup_done)
            self.assertTrue(self.window.runtime.closed)
            self.assertEqual(self.window.backend.registered, {})


if __name__ == "__main__":
    unittest.main()
