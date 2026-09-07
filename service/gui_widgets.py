"""PySide6 화면 계층. 위젯·설정·키 전송은 GUI 스레드, 추론은 gui_runtime이 담당합니다."""

import copy
from ctypes import wintypes
from pathlib import Path
import time

from PySide6.QtCore import Qt, QEvent, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QFrame, QDialog, QLineEdit, QSpinBox, QFileDialog, QSystemTrayIcon, QMenu,
    QScrollArea)

from .gui_runtime import RecognitionRuntime, SessionGate, default_gate
from .settings import SettingsStore, defaults, validate_settings, default_settings_path, resolve_checkpoint
from .shortcuts import ChordCapture, Shortcut, HotkeyManager, KeySender, KEYS, MODIFIERS
from .windows_input import WindowsInput, WM_HOTKEY
from .policy import COMMAND_LABELS, supported_labels
from .ui_text import detail_text, gate_action_name, gate_line, gesture_name, model_line

STYLE = """
QWidget { font-family: 'Malgun Gothic'; font-size: 13px; color: #203454; }
QMainWindow, QDialog { background: #f3f7fd; }
QFrame#card { background: white; border: 1px solid #e0e9f6; border-radius: 16px; }
QLabel#title { font-size: 25px; font-weight: 700; color: #175dcc; }
QLabel#muted { color: #58708e; }
QLabel#error { color: #ae283e; }
QPushButton { background: #eaf2ff; border: 1px solid #cedef5; border-radius: 10px;
 padding: 10px 16px; color: #174b99; }
QPushButton:hover { background: #dceaff; }
QPushButton:pressed { background: #c9ddfc; }
QPushButton#primary { background: #236bea; color: white; border: none; font-weight: 600; }
QPushButton:disabled { color: #8b99ab; background: #eff2f6; border-color: #e0e5ec; }
QLineEdit, QSpinBox { background: white; border: 1px solid #cbdaf0; border-radius: 8px; padding: 8px; }
QLineEdit:focus { border: 2px solid #397bea; }
QCheckBox { spacing: 8px; padding: 4px; }
QMenu { background: white; padding: 6px; }
QMenu::item { padding: 8px 20px; }
QMenu::item:selected { background: #eaf2ff; }
"""


def icon_for(enabled=False):
    pixmap = QPixmap(48, 48)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#236bea" if enabled else "#8b99ab"))
    painter.drawRoundedRect(3, 3, 42, 42, 13, 13)
    painter.setPen(QColor("white"))
    font = painter.font()
    font.setPixelSize(27)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "G")
    painter.end()
    return QIcon(pixmap)


def text_label(text, name=None):
    label = QLabel(text)
    label.setWordWrap(True)
    if name:
        label.setObjectName(name)
    return label


class Toast(QWidget):
    """알림 슬롯 하나를 재사용합니다. show()만 사용하며 activateWindow/raise를 호출하지 않습니다."""

    def __init__(self):
        flags = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                 Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus |
                 Qt.WindowType.WindowTransparentForInput)
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet("background: #edf4ff; color: #174b99; border-radius: 12px; font: 13px 'Malgun Gothic';")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        self.message = text_label("")
        layout.addWidget(self.message)
        self.setFixedWidth(320)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def notify(self, message, enabled=True):
        if not enabled:
            self.timer.stop()
            self.hide()
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        self.message.setText(message)
        self.adjustSize()
        area = screen.availableGeometry()
        self.move(area.right()-self.width()-18, area.bottom()-self.height()-18)
        self.show()
        self.timer.start(2300)


QT_NAMES = {int(getattr(Qt.Key, "Key_" + key)): key for key in KEYS if hasattr(Qt.Key, "Key_" + key)}
QT_NAMES.update({int(Qt.Key.Key_Return): "Enter", int(Qt.Key.Key_Escape): "Esc",
                 int(Qt.Key.Key_Control): "Ctrl", int(Qt.Key.Key_Alt): "Alt",
                 int(Qt.Key.Key_Shift): "Shift", int(Qt.Key.Key_Meta): "Win"})
QT_MODS = {"Ctrl": Qt.KeyboardModifier.ControlModifier, "Alt": Qt.KeyboardModifier.AltModifier,
           "Shift": Qt.KeyboardModifier.ShiftModifier, "Win": Qt.KeyboardModifier.MetaModifier}


class CaptureEdit(QLineEdit):
    changed = Signal(object)
    cancelled = Signal()
    invalid = Signal(str)

    def __init__(self):
        super().__init__()
        self.capture = ChordCapture()
        self.include_win = False
        self.setReadOnly(True)
        self.setPlaceholderText("여기를 선택하고 조합키를 눌러 주세요")

    def reset_capture(self):
        self.capture.reset()
        self.clear()
        self.setFocus()

    def event(self, event):
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        # Tab도 일반 포커스 이동 대신 이 위젯 안에서만 조합키로 수집합니다.
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            key = QT_NAMES.get(event.key(), "unsupported")
            if event.type() == QEvent.Type.KeyRelease:
                self.capture.release(key)
            elif not event.isAutoRepeat():
                self.capture.held = (self.capture.held - set(MODIFIERS)) | {
                    name for name, flag in QT_MODS.items() if event.modifiers() & flag}
                # 체크박스로 선택한 Win은 실제 키를 누르지 않아도 조합에 포함합니다.
                # 시스템에 키를 보내거나 전역 키 입력을 가로채는 기능은 아닙니다.
                if self.include_win:
                    self.capture.held.add("Win")
                try:
                    self.capture.press(key)
                    if self.capture.cancelled:
                        self.cancelled.emit()
                    elif self.capture.value is not None and key not in MODIFIERS:
                        self.setText(self.capture.value.text())
                        self.changed.emit(self.capture.value)
                except ValueError as exc:
                    self.clear()
                    self.invalid.emit(str(exc))
            event.accept()
            return True
        return super().event(event)

    def focusOutEvent(self, event):
        self.capture.held.clear()
        super().focusOutEvent(event)


class SettingsDialog(QDialog):
    """draft만 편집하고 저장 콜백 성공 후 닫습니다. 취소는 실행 중 설정을 건드리지 않습니다."""

    def __init__(self, settings, labels, save, parent=None):
        super().__init__(parent)
        self.draft, self.save_callback = copy.deepcopy(settings), save
        self.active = None
        self.pending_valid = True
        self.setWindowTitle("제스처 · 단축키 설정")
        self.resize(590, 650)
        self.setStyleSheet(STYLE)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        layout.addWidget(text_label("단축키 설정", "title"))
        layout.addWidget(text_label("설정 중에는 인식이 OFF입니다. 닫은 뒤 직접 다시 시작하세요.", "muted"))
        self.buttons = {}
        for name in (*COMMAND_LABELS, "toggle"):
            button = QPushButton()
            supported = name == "toggle" or (name in labels and name in COMMAND_LABELS)
            button.setEnabled(supported)
            button.clicked.connect(lambda checked=False, key=name: self.select(key))
            self.buttons[name] = button
            layout.addWidget(button)
        self.refresh_buttons()
        self.capture_title = text_label("변경할 라벨을 선택하세요")
        layout.addWidget(self.capture_title)
        self.capture_edit = CaptureEdit()
        self.capture_edit.setEnabled(False)
        self.capture_edit.changed.connect(self.captured)
        self.capture_edit.cancelled.connect(self.cancel_capture)
        self.capture_edit.invalid.connect(self.invalid_capture)
        layout.addWidget(self.capture_edit)
        self.win_modifier = QCheckBox("Windows 키 포함 (Win)")
        self.win_modifier.setEnabled(False)
        self.win_modifier.toggled.connect(self.change_win_modifier)
        layout.addWidget(self.win_modifier)
        again = QPushButton("다시 입력")
        again.clicked.connect(lambda: self.select(self.active) if self.active else None)
        layout.addWidget(again)
        layout.addWidget(text_label("수정키 + 일반 키 하나 · Esc: 입력 취소\n"
            "Win 체크 후 D → Win+D / Shift+S → Win+Shift+S (제스처 실행용).\n"
            "Win 단독 및 일부 OS 예약 조합(Win+L, Win+방향키 등)/F12는 지원하지 않습니다.", "muted"))
        self.preview = QCheckBox("카메라 미리보기 표시")
        self.preview.setChecked(settings["preview"])
        self.notifications = QCheckBox("우측 하단 알림 표시")
        self.notifications.setChecked(settings["notifications"])
        layout.addWidget(self.preview)
        layout.addWidget(self.notifications)
        layout.addWidget(text_label("카메라 번호 (저장하면 카메라를 다시 엽니다)"))
        self.camera = QSpinBox()
        self.camera.setRange(0, 99)
        self.camera.setValue(settings["camera_index"])
        layout.addWidget(self.camera)
        layout.addWidget(text_label("외부 모델 경로 (변경 후 앱 재시작 필요)"))
        self.model = QLineEdit(settings["model_path"])
        self.model.setPlaceholderText("비우면 기본 모델: lstm_gpu_2layers_001/best_model.pt")
        layout.addWidget(self.model)
        choose = QPushButton("모델 파일 선택…")
        choose.clicked.connect(self.choose_model)
        layout.addWidget(choose)
        self.error = text_label("", "error")
        layout.addWidget(self.error)
        row = QHBoxLayout()
        restore, cancel, save_button = QPushButton("단축키 기본값"), QPushButton("취소"), QPushButton("저장")
        save_button.setObjectName("primary")
        restore.clicked.connect(self.restore_shortcuts)
        cancel.clicked.connect(self.reject)
        save_button.clicked.connect(self.save)
        for button in (restore, cancel, save_button):
            row.addWidget(button)
        outer.addLayout(row)

    def refresh_buttons(self):
        for name, button in self.buttons.items():
            chord = self.draft["toggle"] if name == "toggle" else self.draft["shortcuts"][name]
            title = "전역 인식 시작/중지" if name == "toggle" else gesture_name(name)
            suffix = " · 현재 모델 미지원" if not button.isEnabled() else ""
            chord_text = "미지정 (키 전송 없음)" if chord is None else Shortcut.from_dict(chord).text()
            button.setText(f"{title}    {chord_text}" + suffix)

    def select(self, name):
        self.active = name
        self.pending_valid = False
        title = "전역 인식 시작/중지" if name == "toggle" else gesture_name(name)
        self.capture_title.setText(f"{title} — 조합키 입력 중")
        self.capture_edit.setEnabled(True)
        chord = self.draft["toggle"] if name == "toggle" else self.draft["shortcuts"][name]
        include_win = chord is not None and "Win" in chord["modifiers"]
        self.win_modifier.blockSignals(True)
        self.win_modifier.setChecked(include_win)
        self.win_modifier.blockSignals(False)
        self.win_modifier.setEnabled(True)
        self.capture_edit.include_win = include_win
        self.capture_edit.reset_capture()
        self.error.clear()

    def change_win_modifier(self, checked):
        self.capture_edit.include_win = checked
        if self.active is not None:
            # 기존 조합을 암묵적으로 변경/저장하지 않고 새 일반 키 입력을 요구합니다.
            self.pending_valid = False
            self.capture_edit.reset_capture()
            self.error.clear()

    def captured(self, chord):
        if self.active == "toggle":
            self.draft["toggle"] = chord.to_dict()
        elif self.active:
            self.draft["shortcuts"][self.active] = chord.to_dict()
        self.pending_valid = True
        self.refresh_buttons()
        self.error.clear()

    def invalid_capture(self, message):
        self.pending_valid = False
        self.error.setText(message)

    def cancel_capture(self):
        self.active = None
        self.pending_valid = True
        self.capture_edit.setEnabled(False)
        self.win_modifier.setEnabled(False)
        self.capture_title.setText("입력 취소 · 이전에 확정한 조합 유지")

    def restore_shortcuts(self):
        self.draft["shortcuts"], self.draft["toggle"] = defaults()["shortcuts"], defaults()["toggle"]
        self.cancel_capture()
        self.refresh_buttons()

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "PyTorch checkpoint (*.pt)")
        if path:
            self.model.setText(path)

    def save(self):
        if not self.pending_valid:
            self.error.setText("유효한 조합을 입력하거나 Esc로 입력을 취소하세요.")
            return
        self.draft.update(preview=self.preview.isChecked(), notifications=self.notifications.isChecked(),
                          camera_index=self.camera.value(), model_path=self.model.text().strip())
        try:
            self.save_callback(validate_settings(self.draft))
        except Exception as exc:
            self.error.setText(f"저장 실패: {exc}")
            return
        self.accept()


class MainWindow(QMainWindow):
    """UI 이벤트는 하나의 스레드에서 처리하여 stop→명령 전송 사이의 권한 경쟁을 줄입니다."""

    def __init__(self, settings, store, *, checkpoint=None, device="auto", load_error="",
                 backend_factory=WindowsInput, runtime_factory=RecognitionRuntime, enable_tray=True,
                 diagnostics=None, gate_weights=None, use_gate=True):
        super().__init__()
        self.settings, self.store = copy.deepcopy(settings), store
        self.checkpoint = str(resolve_checkpoint(checkpoint, settings["model_path"]))
        self.gate = SessionGate()
        self.diagnostics = diagnostics
        self.dialog = None
        self.runtime = None
        self.labels = ()
        self.ready = False
        self.worker_camera = "해제"
        self.last_error = ""
        self.cleanup_done = False
        self.setWindowTitle("Gesture · 제스처 컨트롤")
        self.setWindowIcon(icon_for())
        self.resize(820, 780)
        self.setMinimumSize(540, 480)
        self.setStyleSheet(STYLE)
        self.toast = Toast()
        self._build()
        self.backend = backend_factory(int(self.winId()))
        self.hotkeys = HotkeyManager(self.backend)
        self.sender = KeySender(self.backend)
        self.tray = None
        if enable_tray and QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(icon_for(), self)
            menu = QMenu(self)
            self.tray_status = menu.addAction("인식 OFF · 카메라 해제")
            self.tray_status.setEnabled(False)
            self.tray_toggle = menu.addAction("인식 시작/중지", self.toggle)
            menu.addAction("메인 창 열기", self.restore_window)
            menu.addSeparator()
            menu.addAction("종료", self.close)
            self.tray.setContextMenu(menu)
            self.tray.activated.connect(lambda reason: self.restore_window()
                if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick) else None)
            self.tray.show()
        try:
            self.hotkeys.replace(Shortcut.from_dict(settings["toggle"]))
        except Exception as exc:
            self.show_error(str(exc))
        self.runtime = runtime_factory(Path(self.checkpoint), device, self.backend.foreground,
                                       camera_index=settings["camera_index"],
                                       gate_weights=gate_weights,
                                       gate_factory=default_gate if use_gate else None)
        if diagnostics is not None:
            self.runtime.diagnostics = diagnostics
        self.runtime.set_preview(settings["preview"])
        self.runtime.launch()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self._show_preferences()
        if load_error:
            self.show_error(load_error)
        self.update_controls()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(text_label("Gesture Control", "title"))
        layout.addWidget(text_label("손동작으로 제어하고, 키보드로 언제든 멈추세요.", "muted"))
        card = QFrame()
        card.setObjectName("card")
        box = QVBoxLayout(card)
        box.setContentsMargins(20, 18, 20, 18)
        self.state = text_label("인식 OFF · 카메라 해제")
        self.model_status = text_label("모델 준비 중", "muted")
        self.mode = text_label("지원 모드: 모델 로딩 전", "muted")
        self.gate_status = text_label("손모양 게이트: 준비 중", "muted")
        self.path_label = text_label(f"현재 모델: {self.checkpoint or '미선택'}", "muted")
        self.toggle_button = QPushButton("인식 시작")
        self.toggle_button.setObjectName("primary")
        self.toggle_button.clicked.connect(self.toggle)
        self.hotkey_label = text_label("", "muted")
        for widget in (self.state, self.model_status, self.mode, self.gate_status, self.path_label,
                       self.toggle_button, self.hotkey_label):
            box.addWidget(widget)
        layout.addWidget(card)
        controls = QHBoxLayout()
        self.preview_check = QCheckBox("카메라 미리보기")
        self.notification_check = QCheckBox("알림 표시")
        self.options = QPushButton("단축키 · 설정")
        self.preview_check.toggled.connect(lambda value: self.change_preference("preview", value))
        self.notification_check.toggled.connect(lambda value: self.change_preference("notifications", value))
        self.options.clicked.connect(self.open_settings)
        for widget in (self.preview_check, self.notification_check, self.options):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.preview_label = QLabel("인식을 시작하면 카메라 미리보기가 표시됩니다.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(220)
        self.preview_label.setStyleSheet("background: #e6edf8; border-radius: 16px; color: #58708e;")
        layout.addWidget(self.preview_label)
        self.detail = text_label("미리보기 숨김은 인식을 중지하지 않습니다.", "muted")
        self.last_command = text_label("마지막 명령: 없음")
        self.error = text_label("", "error")
        for widget in (self.detail, self.last_command, self.error):
            layout.addWidget(widget)
        layout.addStretch()

    def notify(self, message):
        if not self.gate.closing:
            self.toast.notify(message, self.settings["notifications"])

    def show_error(self, message):
        self.last_error = message
        self.error.setText(message)

    def _show_preferences(self):
        for widget, key in ((self.preview_check, "preview"), (self.notification_check, "notifications")):
            widget.blockSignals(True)
            widget.setChecked(self.settings[key])
            widget.blockSignals(False)
        self.preview_label.setVisible(self.settings["preview"])
        self.hotkey_label.setText("전역 시작/중지: " + Shortcut.from_dict(self.settings["toggle"]).text() +
                                 (" (등록 실패)" if self.hotkeys.active_id is None else ""))
        if not self.settings["notifications"]:
            self.toast.notify("", False)

    def change_preference(self, key, value):
        changed = copy.deepcopy(self.settings)
        changed[key] = value
        try:
            self.store.save(changed)
        except Exception as exc:
            self.show_error(f"설정 저장 실패: {exc}")
            self._show_preferences()
            return
        self.settings = changed
        if self.runtime:
            self.runtime.set_preview(changed["preview"])
        self._show_preferences()

    def commit_settings(self, changed):
        self.hotkeys.replace(Shortcut.from_dict(changed["toggle"]), lambda: self.store.save(changed))
        moved = changed["camera_index"] != self.settings["camera_index"]
        self.settings = copy.deepcopy(changed)
        if self.runtime:
            self.runtime.set_preview(changed["preview"])
            if moved:
                self.runtime.reopen(changed["camera_index"])
        self._show_preferences()
        self.show_error("모델 경로 변경은 앱 재시작 후 적용됩니다." if
                        str(resolve_checkpoint(saved=changed["model_path"])) != self.checkpoint else "")

    def open_settings(self):
        if self.gate.closing or self.gate.settings_open:
            return
        self.stop()
        self.gate.settings_open = True
        # RegisterHotKey가 현재 전역 조합을 설정 위젯에서 가로채지 않도록 잠시 해제합니다.
        try:
            self.hotkeys.close()
        except Exception as exc:
            self.gate.settings_open = False
            self.show_error(f"설정 진입 전 전역 단축키 해제 실패: {exc}")
            return
        self.dialog = SettingsDialog(self.settings, self.labels, self.commit_settings, self)
        self.dialog.finished.connect(self.settings_finished)
        self.dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.dialog.open()
        self.update_controls()

    def settings_finished(self, result):
        self.gate.settings_open = False
        self.dialog = None
        if not self.gate.closing:
            try:
                self.hotkeys.replace(Shortcut.from_dict(self.settings["toggle"]))
            except Exception as exc:
                self.show_error(f"전역 단축키 복원 실패: {exc}")
        self._show_preferences()
        self.detail.setText("설정 종료 · 인식은 OFF입니다. 직접 다시 시작하세요.")
        self.update_controls()

    def toggle(self):
        if self.gate.closing or self.gate.settings_open:
            return
        if self.gate.enabled:
            self.stop()
        elif self.ready and self.worker_camera == "사용 중" and self.runtime and self.runtime.alive():
            if self.gate.start():
                try:
                    self.runtime.start(self.gate.session, self.settings["camera_index"])
                    self.notify("제스처 인식을 시작합니다.")
                except Exception as exc:
                    self.stop()
                    self.show_error(str(exc))
        self.update_controls()

    def handle_gate(self, event):
        """손모양 유지로 확정된 동작입니다. 실행 권한은 여기서 다시 검사합니다."""
        if self.gate.closing or self.gate.settings_open:
            return
        if event.action == "arm" and not self.gate.enabled:
            self.toggle()
        elif event.action == "disarm" and self.gate.enabled:
            self.stop()
        elif event.action == "toggle_window":
            self.toggle_window()
        else:
            return
        self.notify(gate_action_name(event.action))

    def toggle_window(self):
        """창을 보이거나 숨깁니다. 활성 창을 바꾸면 단축키 전송이 막히므로 포커스는 두지 않습니다."""
        if self.isVisible() and not self.isMinimized():
            self.hide()
            return
        # 이후 트레이 조작은 restore_window의 명시적 activateWindow로 계속 활성화됩니다.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.showNormal()

    def stop(self):
        was_on = self.gate.enabled
        self.gate.stop()  # worker의 작업 완료를 기다리기 전에 UI 전송 권한부터 취소합니다.
        if self.runtime:
            self.runtime.stop()
        self.last_command.setText("마지막 명령: 없음")
        # 카메라와 손모양 게이트는 계속 돌아가므로 미리보기를 지우지 않습니다.
        if was_on:
            self.notify("제스처 인식을 중지했습니다.")
        self.update_controls()

    def update_controls(self, mode=None):
        self.toggle_button.setText("인식 중지" if self.gate.enabled else "인식 시작")
        self.toggle_button.setEnabled(not self.gate.closing and not self.gate.settings_open and
            (self.gate.enabled or (self.ready and self.worker_camera == "해제")))
        self.options.setEnabled(not self.gate.closing and not self.gate.settings_open)
        self.preview_check.setEnabled(not self.gate.closing)
        self.notification_check.setEnabled(not self.gate.closing)
        state = (mode or "WARMUP") if self.gate.enabled else "OFF"
        if self.gate.closing:
            state = "종료 중"
        text = f"인식 {state} · 카메라 {self.worker_camera}"
        self.state.setText(text)
        if self.tray:
            self.tray_status.setText(text)
            self.tray.setToolTip(text)
            self.tray.setIcon(icon_for(self.gate.enabled))
            self.tray_toggle.setEnabled(self.toggle_button.isEnabled())

    def poll(self):
        if self.gate.closing:
            if self.runtime is None or not self.runtime.alive():
                self.timer.stop()
                self.close()
            return
        if self.runtime is None:
            return
        status, frame, event, gate_event = self.runtime.poll()
        self.ready, self.labels = status["ready"], supported_labels(status["labels"])
        self.worker_camera = status["camera"]
        self.model_status.setText(status["model_status"])
        self.mode.setText(model_line(status["schema"], self.labels))
        self.gate_status.setText(gate_line(status["gate"], status["gate_ready"], status["gate_error"]))
        if status["error"]:
            self.show_error(status["error"])
            if self.gate.enabled:
                self.stop()
        if self.gate.enabled and status["session"] == self.gate.session:
            self.detail.setText(detail_text(status["detail"]))
        # 카메라와 게이트는 인식 OFF에서도 돌아가므로 미리보기는 항상 갱신합니다.
        if frame is not None and self.settings["preview"]:
            height, width = frame.shape[:2]
            image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888).copy()
            self.preview_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.preview_label.width(), 300, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        if gate_event is not None:
            self.handle_gate(gate_event)
        accepted = event is not None and self.gate.accept(event, time.monotonic(), self.labels)
        if event is not None and self.diagnostics is not None:
            self.diagnostics.emit("gui_command", session=event.session, serial=event.serial,
                                  frame_s=event.frame_s, label=event.gesture, accepted=accepted)
        if accepted:
            try:
                if self.diagnostics is not None:
                    self.diagnostics.emit("key_send_start", session=event.session, serial=event.serial,
                                          frame_s=event.frame_s, label=event.gesture)
                chord = self.settings["shortcuts"][event.gesture]
                if chord is None:
                    sent, message = False, "단축키 미지정 · 키 전송 없음"
                else:
                    sent, message = self.sender.send(Shortcut.from_dict(chord), event.target)
                self.last_command.setText(f"마지막 명령: {gesture_name(event.gesture)} · {message}")
                if self.diagnostics is not None:
                    self.diagnostics.emit("key_send_result", session=event.session, serial=event.serial,
                                          frame_s=event.frame_s, label=event.gesture, sent=sent,
                                          reason=message if not sent else "SendInput accepted")
                if sent:
                    self.notify(f"{gesture_name(event.gesture)} · {message}")
            except Exception as exc:
                if self.diagnostics is not None:
                    self.diagnostics.emit("key_send_error", session=event.session, serial=event.serial,
                                          frame_s=event.frame_s, error=str(exc))
                self.stop()
                self.show_error(f"입력 전송 실패: {exc}")
        # 손모양으로 방금 켠 직후에는 작업자가 아직 이전 세션을 보고하고 있습니다.
        # 다른 세션의 mode를 그대로 쓰면 ON 상태를 OFF로 표시하므로 무시합니다.
        matched = status["session"] == self.gate.session
        self.update_controls(status["recognition"] if matched else None)

    def nativeEvent(self, event_type, message):
        if bytes(event_type) in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            native = wintypes.MSG.from_address(int(message))
            if (native.message == WM_HOTKEY and hasattr(self, "hotkeys")
                    and native.wParam == self.hotkeys.active_id):
                self.toggle()
                return True, 0
        return False, 0

    def restore_window(self):
        self.showNormal()
        self.activateWindow()  # 사용자가 트레이를 눌렀을 때만 명시적으로 활성화합니다.

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange and self.isMinimized() and self.tray:
            QTimer.singleShot(0, self.hide)
        super().changeEvent(event)

    def closeEvent(self, event):
        if not self.gate.closing:
            self.gate.closing = True
            self.stop()
            if self.dialog:
                self.dialog.reject()
            self.toast.timer.stop()
            self.toast.close()
            for cleanup in (self.hotkeys.close, self.sender.release_owned):
                try:
                    cleanup()
                except Exception as exc:
                    self.show_error(f"종료 정리 오류: {exc}")
            if self.runtime:
                self.runtime.close()
            self.update_controls()
        if self.runtime and self.runtime.alive():
            # 드라이버의 blocking read를 다른 스레드에서 강제 release/terminate하지 않습니다.
            self.detail.setText("카메라/추론 작업이 끝나기를 기다립니다. 정리 중에는 명령이 실행되지 않습니다.")
            event.ignore()
            return
        self.timer.stop()
        if self.tray:
            self.tray.hide()
        self.cleanup_done = True
        event.accept()
        QApplication.instance().quit()


def run_gui(args):
    app = QApplication([])
    app.setApplicationName("JestureService")
    app.setQuitOnLastWindowClosed(False)
    store = SettingsStore(args.settings or default_settings_path())
    error = ""
    try:
        settings = store.load()
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        settings = defaults()
        error = f"설정 읽기 실패: {exc}\n기존 파일은 보존됩니다: {store.path}\n파일을 별도 이동/보관하고 재시작하세요."
    from .diagnostics import DiagnosticLog
    diagnostics = DiagnosticLog(args.diagnostics) if getattr(args, "diagnostics", None) else None
    if diagnostics is not None:
        print(f"진단 로그: {diagnostics.path.resolve()} (영상/좌표 저장 없음)")
    window = None
    try:
        window = MainWindow(settings, store, checkpoint=args.checkpoint, device=args.device,
                            load_error=error, diagnostics=diagnostics,
                            gate_weights=getattr(args, "gate_weights", None),
                            use_gate=not getattr(args, "no_gate", False))
        window.show()
        return app.exec()
    finally:
        # 이벤트 루프가 외부 quit/예외로 끝난 경우에도 worker와 키 등록을 정리합니다.
        if window is not None and not window.cleanup_done:
            window.close()
            if window.runtime and window.runtime.thread:
                window.runtime.thread.join(timeout=2)
        if diagnostics is not None:
            diagnostics.close()
