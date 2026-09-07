"""제스처 → PC 기능 매핑(gestures.json) + Windows 키 입력.

추가 라이브러리 없이 Windows 기본 API(SendInput)로 키를 누른다. Windows 가 아니면 항상 연습 모드.
연습 모드(dry_run=True)에서는 실제로 키를 누르지 않고 무엇을 누를지만 알려준다.

gestures.json 예:
{
  "min_confidence": 0.8,
  "cooldown_sec": 1.0,
  "actions": {
    "swipe_left": {"keys": ["tab"], "label": "Switcher: next (Tab)"},
    "make_fist":  {"keys": ["ctrl", "alt", "tab"], "label": "Open switcher (Ctrl+Alt+Tab)"},
    "finger_snap": {"keys": ["enter"], "label": "Switcher: select (Enter)"},
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

# 가상 키 코드 (Windows). 이름은 gestures.json 의 keys 에 그대로 쓴다.
VK = {
    "alt": 0x12, "ctrl": 0x11, "shift": 0x10, "win": 0x5B,
    "tab": 0x09, "enter": 0x0D, "esc": 0x1B, "space": 0x20, "backspace": 0x08, "delete": 0x2E, "insert": 0x2D,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "pageup": 0x21, "pagedown": 0x22, "home": 0x24, "end": 0x23, "printscreen": 0x2C,
    "volume_mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "media_play_pause": 0xB3, "media_next": 0xB0, "media_prev": 0xB1, "media_stop": 0xB2,
    "browser_back": 0xA6, "browser_forward": 0xA7, "browser_refresh": 0xA8,
    "comma": 0xBC, "period": 0xBE, "minus": 0xBD, "equal": 0xBB, "slash": 0xBF,
}
VK.update({f"f{i}": 0x6F + i for i in range(1, 13)})          # f1..f12
VK.update({chr(c): c for c in range(0x41, 0x5B)})              # A..Z (대문자 코드)
VK.update({chr(c).lower(): c for c in range(0x41, 0x5B)})      # a..z
VK.update({str(d): 0x30 + d for d in range(10)})               # 0..9

# 확장 키(방향키·Ins/Del/Home/End/PgUp/PgDn·Win): 일부 프로그램은 KEYEVENTF_EXTENDEDKEY 가 있어야 알아듣는다.
_EXTENDED = {"left", "up", "right", "down", "pageup", "pagedown", "home", "end", "insert", "delete", "win",
             "printscreen"}
MODIFIERS = ["ctrl", "alt", "shift", "win"]
# UI 에서 고를 수 있는 특수 키(키보드로 잡기 어려운 것). (이름, 설명)
SPECIAL_KEYS = [
    ("media_play_pause", "재생/일시정지"), ("media_next", "다음 곡"), ("media_prev", "이전 곡"), ("media_stop", "정지"),
    ("volume_up", "소리 크게"), ("volume_down", "소리 작게"), ("volume_mute", "음소거"),
    ("browser_back", "브라우저 뒤로"), ("browser_forward", "브라우저 앞으로"), ("browser_refresh", "새로고침"),
    ("printscreen", "화면 캡처"),
]
_DISPLAY = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win", "tab": "Tab", "enter": "Enter",
            "esc": "Esc", "space": "Space", "backspace": "Backspace", "delete": "Delete", "insert": "Insert",
            "left": "←", "up": "↑", "right": "→", "down": "↓", "pageup": "PageUp", "pagedown": "PageDown",
            "home": "Home", "end": "End", "printscreen": "PrintScreen",
            "comma": ",", "period": ".", "minus": "-", "equal": "=", "slash": "/"}
_DISPLAY.update({k: v for k, v in SPECIAL_KEYS})


def key_display(keys: list[str] | None) -> str:
    """["ctrl","alt","tab"] → "Ctrl+Alt+Tab". 비어 있으면 "(없음)"."""
    if not keys:
        return "(없음)"
    return "+".join(_DISPLAY.get(k, k.upper() if len(k) == 1 else k.replace("_", " ").title()) for k in keys)


def normalize_keys(keys: list[str]) -> list[str]:
    """소문자·중복 제거·보조키(ctrl,alt,shift,win) 먼저 정렬. 모르는 키 이름이면 ValueError."""
    ks = []
    for k in keys:
        k = k.strip().lower()
        if k not in VK:
            raise ValueError(f"모르는 키 이름: {k}")
        if k not in ks:
            ks.append(k)
    mods = [m for m in MODIFIERS if m in ks]
    rest = [k for k in ks if k not in MODIFIERS]
    return mods + rest


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

    _KEYEVENTF_EXTENDEDKEY = 0x0001

    def _send_key(vk: int, up: bool = False, extended: bool = False):
        inp = _INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        flags = (_KEYEVENTF_KEYUP if up else 0) | (_KEYEVENTF_EXTENDEDKEY if extended else 0)
        inp.u.ki = _KEYBDINPUT(vk, 0, flags, 0, None)
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def press_combo(keys: list[str], hold_sec: float = 0.05):
    """keys 를 순서대로 누르고 역순으로 뗀다. 예: ["alt","tab"]"""
    if not _IS_WIN:
        raise RuntimeError("키 입력은 Windows 에서만 지원")
    ks = [k.lower() for k in keys]
    for k in ks:
        _send_key(VK[k], extended=k in _EXTENDED)
        time.sleep(0.02)
    time.sleep(hold_sec)
    for k in reversed(ks):
        _send_key(VK[k], up=True, extended=k in _EXTENDED)
        time.sleep(0.02)


class ActionMapper:
    def __init__(self, path: Path = GESTURES_JSON, dry_run: bool = True):
        self.path = config.project_path(path)
        self.dry_run = dry_run or not _IS_WIN
        self._cooldown_until = -1e9      # 이 시각 전에 시작된 구간은 무시
        self.log: list[str] = []
        self.reload()

    # ── 설정 읽기/쓰기 (UI 에서 사용자가 키를 바꿀 수 있게) ──
    def reload(self):
        cfg = json.loads(self.path.read_text(encoding="utf-8"))
        self.min_confidence = float(cfg.get("min_confidence", 0.8))
        self.cooldown_sec = float(cfg.get("cooldown_sec", 1.0))
        self.actions: dict[str, dict | None] = cfg.get("actions", {})
        # YOLO 정적 포즈 게이트 설정 (09-07 오후 연결). 없으면 기본값
        self.gate: dict = {"enabled": True, "model": "models/static/v3.pt", "confidence": 0.7,
                           "hold_seconds": 3.0, "miss_tolerance_seconds": 0.3, "imgsz": 640,
                           "iou": 0.5, "max_det": 10, "device": None}
        self.gate.update(cfg.get("gate") or {})
        self._extra = {k: v for k, v in cfg.items() if k not in ("min_confidence", "cooldown_sec", "actions", "gate")}

    def to_config(self) -> dict:
        out = dict(self._extra)
        out.update({"min_confidence": self.min_confidence, "cooldown_sec": self.cooldown_sec, "actions": self.actions,
                    "gate": self.gate})
        return out

    def save(self, path: Path | None = None):
        p = Path(path) if path else self.path
        p.write_text(json.dumps(self.to_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def set_action(self, label: str, keys: list[str] | None, name: str | None = None,
                   cooldown_sec: float | None = None):
        """label 에 키 조합을 배정. keys 가 비면 '아무것도 안 함'. name 은 화면 표시용."""
        if not keys:
            self.actions[label] = None
            return
        ks = normalize_keys(keys)
        prev = self.actions.get(label) or {}
        a = {"keys": ks, "label": name or key_display(ks)}
        cd = cooldown_sec if cooldown_sec is not None else prev.get("cooldown_sec")
        if cd is not None:
            a["cooldown_sec"] = float(cd)
        self.actions[label] = a

    def keys_for(self, label: str) -> list[str]:
        a = self.actions.get(label)
        return list(a["keys"]) if a else []

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
