"""지연 초기화되는 Win32 어댑터. 전역 타이핑 훅 없이 RegisterHotKey와 SendInput을 사용합니다."""

import ctypes
from ctypes import wintypes
import os
import sys

VK = {**{k: ord(k) for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"},
      **{f"F{i}": 0x6F+i for i in range(1, 13)},
      "Ctrl": 0x11, "Alt": 0x12, "Shift": 0x10, "Win": 0x5B,
      "Left": 0x25, "Up": 0x26, "Right": 0x27, "Down": 0x28,
      "Space": 0x20, "Enter": 0x0D, "Tab": 9, "Backspace": 8, "Delete": 0x2E,
      "Insert": 0x2D, "Home": 0x24, "End": 0x23, "PageUp": 0x21, "PageDown": 0x22}
EXTENDED = {"Left", "Up", "Right", "Down", "Delete", "Insert", "Home", "End", "PageUp", "PageDown", "Win"}
MOD_FLAGS = {"Alt": 1, "Ctrl": 2, "Shift": 4, "Win": 8}
WM_HOTKEY = 0x0312


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", wintypes.DWORD), ("data", INPUTUNION)]


def input_array(events):
    """32/64비트 포인터 크기를 보존하는 INPUT 배열 생성. API 호출은 하지 않습니다."""
    array = (INPUT * len(events))()
    for item, (key, down) in zip(array, events):
        item.type = 1
        item.ki = KEYBDINPUT(VK[key], 0, (1 if key in EXTENDED else 0) | (0 if down else 2), 0, 0)
    return array


class WindowsInput:
    def __init__(self, hwnd):
        if sys.platform != "win32":
            raise RuntimeError("GUI의 전역 제어/입력 전송은 Windows 전용입니다.")
        self.hwnd = int(hwnd)
        self.process_id = os.getpid()
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        definitions = {
            "RegisterHotKey": ([wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT], wintypes.BOOL),
            "UnregisterHotKey": ([wintypes.HWND, ctypes.c_int], wintypes.BOOL),
            "GetForegroundWindow": ([], wintypes.HWND),
            "GetWindowThreadProcessId": ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            "GetAsyncKeyState": ([ctypes.c_int], ctypes.c_short),
            "SendInput": ([wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int], wintypes.UINT),
        }
        for name, (args, result) in definitions.items():
            function = getattr(self.user32, name)
            function.argtypes, function.restype = args, result

    def register_hotkey(self, identifier, shortcut):
        flags = 0x4000 | sum(MOD_FLAGS[m] for m in shortcut.modifiers)  # MOD_NOREPEAT
        if not self.user32.RegisterHotKey(self.hwnd, identifier, flags, VK[shortcut.key]):
            raise OSError(ctypes.get_last_error(), "전역 단축키 등록 실패: 다른 앱의 사용/예약 조합을 확인하세요.")

    def unregister_hotkey(self, identifier):
        if not self.user32.UnregisterHotKey(self.hwnd, identifier):
            raise OSError(ctypes.get_last_error(), "전역 단축키 해제 실패")

    def foreground(self):
        hwnd = self.user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        if hwnd:
            self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(hwnd or 0), int(pid.value)

    def any_pressed(self, keys):
        # Win은 오른쪽 키도 검사. Ctrl/Alt/Shift의 일반 VK는 좌우를 포함합니다.
        codes = {VK[key] for key in keys}
        if "Win" in keys:
            codes.add(0x5C)
        return any(self.user32.GetAsyncKeyState(code) & 0x8000 for code in codes)

    def send_events(self, events):
        array = input_array(events)
        return int(self.user32.SendInput(len(array), array, ctypes.sizeof(INPUT)))
