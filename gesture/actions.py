"""제스처 → PC 기능 매핑(gestures.json) + Windows 키 입력.

추가 라이브러리 없이 Windows 기본 API(SendInput)로 키를 누른다. Windows 가 아니면 항상 연습 모드.
연습 모드(dry_run=True)에서는 실제로 키를 누르지 않고 무엇을 누를지만 알려준다.

gestures.json 예:
{
  "min_confidence": 0.8,
  "cooldown_sec": 1.0,
  "actions": {
    "swipe_left": {"keys": ["alt", "tab"], "label": "Next window (Alt+Tab)"},
    "make_fist":  {"keys": ["win", "tab"], "label": "Task view (Win+Tab)"},
    "finger_snap": {"keys": ["media_play_pause"], "label": "Play/Pause"},
    "no_gesture": null
  }
}
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

from . import config

GESTURES_JSON = config.ROOT / "gestures.json"

# 가상 키 코드 (Windows)
VK = {
    "alt": 0x12, "ctrl": 0x11, "shift": 0x10, "win": 0x5B,
    "tab": 0x09, "enter": 0x0D, "esc": 0x1B, "space": 0x20,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "pageup": 0x21, "pagedown": 0x22, "home": 0x24, "end": 0x23,
    "f5": 0x74, "d": 0x44, "f4": 0x73,
    "volume_mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "media_play_pause": 0xB3, "media_next": 0xB0, "media_prev": 0xB1,
}
_IS_WIN = sys.platform == "win32"

if _IS_WIN:
    _user32 = ctypes.windll.user32
    _KEYEVENTF_KEYUP = 0x0002

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("_pad", ctypes.c_byte * 32)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]

    def _send_key(vk: int, up: bool = False):
        inp = _INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.u.ki = _KEYBDINPUT(vk, 0, _KEYEVENTF_KEYUP if up else 0, 0, None)
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def press_combo(keys: list[str], hold_sec: float = 0.05):
    """keys 를 순서대로 누르고 역순으로 뗀다. 예: ["alt","tab"]"""
    if not _IS_WIN:
        raise RuntimeError("키 입력은 Windows 에서만 지원")
    vks = [VK[k.lower()] for k in keys]
    for vk in vks:
        _send_key(vk)
        time.sleep(0.02)
    time.sleep(hold_sec)
    for vk in reversed(vks):
        _send_key(vk, up=True)
        time.sleep(0.02)


class ActionMapper:
    def __init__(self, path: Path = GESTURES_JSON, dry_run: bool = True):
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        self.min_confidence = float(cfg.get("min_confidence", 0.8))
        self.cooldown_sec = float(cfg.get("cooldown_sec", 1.0))
        self.actions: dict[str, dict | None] = cfg.get("actions", {})
        self.dry_run = dry_run or not _IS_WIN
        self._cooldown_until = -1e9      # 이 시각 전에 시작된 구간은 무시
        self.log: list[str] = []

    def cooldown_for(self, label: str) -> float:
        """동작별 cooldown (gestures.json 의 actions.<label>.cooldown_sec, 없으면 공통값).
        09-06 4차 실측: 공통 1.2초는 연속 스와이프를 절반이나 놓쳤다. 상식 검사가 '주먹 뒤 내리기'를 막아 주므로
        cooldown 은 중복 실행 방지 정도로 짧게 두고, 주먹만 조금 길게."""
        a = self.actions.get(label) or {}
        return float(a.get("cooldown_sec", self.cooldown_sec))

    def describe(self, label: str) -> str:
        a = self.actions.get(label)
        if not a:
            return "-"
        return a.get("label") or "+".join(a["keys"])

    def in_cooldown(self, t: float) -> bool:
        return t < self._cooldown_until

    def handle(self, label: str, confidence: float, now: float | None = None,
               start: float | None = None) -> str:
        """판정 결과 하나를 받아 실행 여부 결정. 반환: 사람이 읽는 한 줄 상태.

        start: 구간이 *시작된* 시각. 주어지면 cooldown 은 시작 시각으로 판단한다.
        (09-06: 주먹 실행 뒤 손을 내리는 동작이 새 구간으로 잡혀 실행되던 문제. 그 구간은 실행 직후에 시작되므로
        시작 시각 기준이면 걸러진다. 끝 시각 기준이면 1초가 지나 통과했다.)"""
        now = time.perf_counter() if now is None else now
        a = self.actions.get(label)
        if not a:
            return f"{label}: no action"
        if confidence < self.min_confidence:
            return f"{label} {confidence:.2f} < {self.min_confidence} -> ignored"
        if self.in_cooldown(start if start is not None else now):
            return f"{label} -> started in cooldown, ignored"
        self._cooldown_until = now + self.cooldown_for(label)
        what = self.describe(label)
        if self.dry_run:
            msg = f"[DRY] {label} -> {what}"
        else:
            press_combo(a["keys"])
            msg = f"[FIRE] {label} -> {what}"
        self.log.append(msg)
        return msg
