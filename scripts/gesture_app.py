"""시연용 서비스 UI (2차): 사이드바 + 페이지(홈 / 키 매핑 / 설정 / 기록), 다크 테마. customtkinter 사용.

  uv run python scripts/gesture_app.py            # 연습 모드(키 안 누름)로 시작
  uv run python scripts/gesture_app.py --live     # 처음부터 실제 키 입력
  uv run python scripts/gesture_app.py --camera 1

참고한 앱들의 패턴
  - Logi Options+ / PowerToys : 왼쪽 사이드바로 페이지 이동, 설정은 카드 단위, 상태는 항상 보이는 자리에
  - Elgato Stream Deck        : 동작 하나 = 카드 하나, 키는 칩(chip)으로 크게, 프로필(프리셋) 한 번에 바꾸기
  - Project Gameface          : 카메라 화면을 크게, 판정 문턱(확신도)은 슬라이더, 무엇이 인식됐는지 즉시 표시

YOLO 게이트는 `app.pipeline.set_gate(True/False)` 로 연결한다 (docs/gesture_app.md).
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import customtkinter as ctk  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageTk  # noqa: E402

from gesture import config  # noqa: E402
from gesture.actions import MODIFIERS, key_display, normalize_keys, press_combo  # noqa: E402
from gesture.landmarks import draw_landmarks  # noqa: E402
from gesture.presets import FUNCTIONS, GESTURE_NAMES, PRESETS, STATIC_LABELS  # noqa: E402

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_FAMILY = "Malgun Gothic"
ACCENT = {"swipe_left": "#3aa0ff", "make_fist": "#ff9f43", "finger_snap": "#2ecc71", "no_gesture": "#8d99ae",
          "skipped": "#8d99ae", "cancel": "#b478ff", "gate_open": "#2ecc71", "gate_close": "#e74c3c"}
ICON = {"swipe_left": "⇦", "make_fist": "✊", "finger_snap": "✦", "cancel": "⚑"}
POSE_KR = {"start": "손바닥", "stop": "주먹", "cancel": "세 번째 포즈"}
DYNAMIC_LABELS = [l for l in config.LABELS if l != "no_gesture"]
COMMAND_LABELS = DYNAMIC_LABELS + STATIC_LABELS      # 키 매핑 대상 (동적 3 + 정적 1)
CARD = ("#f2f3f5", "#26282e")          # 카드 배경 (light, dark)
CARD2 = ("#e6e8ec", "#31343b")         # 카드 안 요소
SIDEBAR = ("#e9ebef", "#1b1d22")
MUTED = ("#5f6572", "#9aa3b2")
TEXT = ("#1c1e22", "#e8eaf0")

# Tk keysym → 우리 키 이름
KEYSYM = {
    "Control_L": "ctrl", "Control_R": "ctrl", "Alt_L": "alt", "Alt_R": "alt", "Shift_L": "shift", "Shift_R": "shift",
    "Win_L": "win", "Win_R": "win", "Super_L": "win", "Super_R": "win",
    "Tab": "tab", "Return": "enter", "Escape": "esc", "space": "space", "BackSpace": "backspace",
    "Delete": "delete", "Insert": "insert", "Left": "left", "Up": "up", "Right": "right", "Down": "down",
    "Prior": "pageup", "Next": "pagedown", "Home": "home", "End": "end", "Print": "printscreen",
    "comma": "comma", "period": "period", "minus": "minus", "equal": "equal", "slash": "slash",
}
KEYSYM.update({f"F{i}": f"f{i}" for i in range(1, 13)})
KEYSYM.update({c: c for c in "abcdefghijklmnopqrstuvwxyz0123456789"})
KEYSYM.update({c.upper(): c for c in "abcdefghijklmnopqrstuvwxyz"})


def font(size=13, weight="normal"):
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)


def _pil_font(size):
    for name in ("malgunbd.ttf", "malgun.ttf", "arial.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


PIL_FONT = _pil_font(18)
PIL_FONT_S = _pil_font(14)


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def key_chips(parent, keys: list[str], size=15, color="#3a7bd5"):
    """['ctrl','alt','tab'] → [Ctrl] + [Alt] + [Tab] 모양의 칩 묶음. 비어 있으면 회색 '동작 없음'."""
    box = ctk.CTkFrame(parent, fg_color="transparent")
    if not keys:
        ctk.CTkLabel(box, text="동작 없음", font=font(size), text_color=MUTED).pack(side="left")
        return box
    for i, k in enumerate(keys):
        if i:
            ctk.CTkLabel(box, text="+", font=font(size), text_color=MUTED).pack(side="left", padx=4)
        ctk.CTkLabel(box, text=key_display([k]), font=font(size, "bold"), fg_color=color, corner_radius=6,
                     text_color="#ffffff", padx=10, pady=3).pack(side="left")
    return box


class KeyRecorder(ctk.CTkFrame):
    """카드 안에서 바로 녹화하는 키 칸 (OBS·Discord·PowerToys 방식).
    클릭 → '키를 누르세요' 상태 → 조합을 누르면 바로 확정. Esc 취소, Backspace 지우기, 다른 곳 클릭하면 취소."""

    def __init__(self, master, color: str, on_change, on_clear):
        super().__init__(master, fg_color=CARD2, corner_radius=10, height=48, cursor="hand2")
        self.color, self.on_change, self.on_clear = color, on_change, on_clear
        self.keys: list[str] = []
        self.recording = False
        self._held: list[str] = []
        self.pack_propagate(False)
        self._inner = ctk.CTkFrame(self, fg_color="transparent")
        self._inner.pack(side="left", padx=12)
        self._hint = ctk.CTkLabel(self, text="클릭해서 바꾸기", font=font(11), text_color=MUTED)
        self._hint.pack(side="right", padx=12)
        # 키 이벤트를 받는 보이지 않는 위젯 (포커스용)
        self._focus = tk.Label(self, width=0, height=0, bd=0, highlightthickness=0, takefocus=1)
        self._focus.place(x=0, y=0)
        self._focus.bind("<KeyPress>", self._on_press)
        self._focus.bind("<KeyRelease>", self._on_release)
        self._focus.bind("<FocusOut>", lambda e: self.after(50, self._cancel_if_recording))
        for w in (self, self._inner, self._hint):
            w.bind("<Button-1>", self.start)
        self.set_keys([])

    def set_keys(self, keys: list[str]):
        self.keys = list(keys)
        for w in self._inner.winfo_children():
            w.destroy()
        if self.recording:
            return
        if not keys:
            ctk.CTkLabel(self._inner, text="동작 없음", font=font(14), text_color=MUTED).pack(side="left")
        else:
            for k, key in enumerate(keys):
                if k:
                    ctk.CTkLabel(self._inner, text="+", font=font(14), text_color=MUTED).pack(side="left", padx=4)
                ctk.CTkLabel(self._inner, text=key_display([key]), font=font(14, "bold"), fg_color=self.color,
                             corner_radius=6, text_color="#ffffff", padx=10, pady=3).pack(side="left")
        for w in self._inner.winfo_children():
            w.bind("<Button-1>", self.start)
        self._hint.configure(text="클릭해서 바꾸기")
        self.configure(fg_color=CARD2, border_width=0)

    def start(self, _e=None):
        if self.recording:
            return
        self.recording = True
        self._held = []
        for w in self._inner.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._inner, text="키 조합을 누르세요…", font=font(14, "bold"), text_color=self.color).pack(side="left")
        self._hint.configure(text="Esc 취소 · Backspace 지우기 · Win 은 오른쪽 체크")
        self.configure(fg_color=("#dfe9f5", "#1f2a3a"), border_width=2, border_color=self.color)
        self._focus.focus_set()

    def _finish(self, keys):
        self.recording = False
        self._held = []
        self.set_keys(keys)
        self.on_change(keys)

    def _cancel_if_recording(self):
        if self.recording:
            self.recording = False
            self._held = []
            self.set_keys(self.keys)

    def _on_press(self, e):
        if not self.recording:
            return "break"
        if e.keysym == "Escape":
            self._cancel_if_recording(); return "break"
        if e.keysym == "BackSpace":
            self.recording = False; self.set_keys([]); self.on_clear(); return "break"
        name = KEYSYM.get(e.keysym)
        if name is None:
            self._hint.configure(text=f"쓸 수 없는 키: {e.keysym}"); return "break"
        if name in MODIFIERS:
            if name not in self._held:
                self._held.append(name)
            self._hint.configure(text=key_display(self._held) + " + …")
            return "break"
        mods = [m for m in MODIFIERS if m in self._held]
        # 보조키를 이벤트 상태로도 보완 (Ctrl 0x4, Shift 0x1, Alt 0x20000 on Windows)
        if e.state & 0x4 and "ctrl" not in mods: mods.append("ctrl")
        if e.state & 0x1 and "shift" not in mods: mods.append("shift")
        if e.state & 0x20000 and "alt" not in mods: mods.append("alt")
        self._finish(normalize_keys(mods + [name]))
        return "break"

    def _on_release(self, e):
        if not self.recording:
            return "break"
        name = KEYSYM.get(e.keysym)
        if name in MODIFIERS and name in self._held:
            self._held.remove(name)
            if not self._held:
                # 보조키만 눌렀다 뗀 경우 (예: Win 하나) → 그 조합으로 확정
                self._finish(normalize_keys([name]))
        return "break"


class App(ctk.CTk):
    def __init__(self, camera: int, live: bool, model: str | None, gestures: str | None):
        super().__init__()
        self.camera_index = camera
        self.model_file = model
        self.gestures_path = gestures
        self.pipeline = None
        self.cap = None
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame = None
        self._worker = None
        self._msgq: queue.Queue = queue.Queue()
        self._flash_until = 0.0
        self._card_glow: dict[str, float] = {}
        self._toast_until = 0.0
        self._n_events_shown = 0
        self._last_ev = None
        self._want_live = live
        self._mirror = tk.BooleanVar(value=False)

        self.title("Jesture — 손 제스처 PC 제어")
        self.geometry("1280x800")
        self.minsize(1180, 740)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_pages()
        self.show_page("home")
        self._start_worker()
        self.after(30, self._tick)

    # ───────────────────────── 사이드바 ─────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=SIDEBAR)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(7, weight=1)
        ctk.CTkLabel(sb, text="✋  Jesture", font=font(22, "bold")).grid(row=0, column=0, padx=22, pady=(24, 2), sticky="w")
        ctk.CTkLabel(sb, text="손 제스처로 PC 제어 · 4팀", font=font(12), text_color=MUTED).grid(row=1, column=0, padx=24, pady=(0, 18), sticky="w")
        self.nav = {}
        for i, (key, text) in enumerate([("home", "●  홈"), ("map", "⌨  키 매핑"), ("settings", "⚙  설정"), ("log", "≡  기록"),
                                         ("help", "?  도움말")]):
            b = ctk.CTkButton(sb, text=text, font=font(14), anchor="w", height=40, corner_radius=8,
                              fg_color="transparent", text_color=TEXT, hover_color=CARD2,
                              command=lambda k=key: self.show_page(k))
            b.grid(row=2 + i, column=0, padx=14, pady=3, sticky="ew")
            self.nav[key] = b

        status = ctk.CTkFrame(sb, fg_color=CARD, corner_radius=10)
        status.grid(row=8, column=0, padx=14, pady=(0, 14), sticky="ew")
        self.cam_lbl = ctk.CTkLabel(status, text="모델 로딩 중...", font=font(12), text_color=MUTED, anchor="w")
        self.cam_lbl.pack(fill="x", padx=12, pady=(10, 0))
        self.gate_lbl = ctk.CTkLabel(status, text="게이트: 미연결 (항상 열림)", font=font(12), text_color=MUTED, anchor="w")
        self.gate_lbl.pack(fill="x", padx=12, pady=(2, 8))
        self.mode_switch = ctk.CTkSwitch(status, text="실제 키 입력", font=font(14, "bold"), command=self.toggle_mode,
                                         progress_color="#e74c3c")
        self.mode_switch.pack(anchor="w", padx=12, pady=(4, 2))
        self.mode_lbl = ctk.CTkLabel(status, text="연습 모드 — 키를 누르지 않음", font=font(11), text_color=MUTED, anchor="w")
        self.mode_lbl.pack(fill="x", padx=12, pady=(0, 10))

    def show_page(self, key: str):
        for k, page in self.pages.items():
            page.grid_remove()
        self.pages[key].grid()
        for k, b in self.nav.items():
            b.configure(fg_color=CARD2 if k == key else "transparent")

    # ───────────────────────── 페이지 ─────────────────────────
    def _build_pages(self):
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=(8, 18), pady=14)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)
        # 토스트 (실행 알림)
        self.toast = ctk.CTkLabel(self.content, text="", font=font(14, "bold"), fg_color="#1e7e34", corner_radius=8,
                                  text_color="#ffffff", padx=16, pady=8)
        self.pages = {}
        for key, builder in [("home", self._build_home), ("map", self._build_map), ("settings", self._build_settings),
                             ("log", self._build_log), ("help", self._build_help)]:
            page = ctk.CTkFrame(self.content, fg_color="transparent")
            page.grid(row=1, column=0, sticky="nsew")
            page.grid_columnconfigure(0, weight=1)
            builder(page)
            self.pages[key] = page

    def _header(self, page, title, sub):
        ctk.CTkLabel(page, text=title, font=font(24, "bold")).grid(row=0, column=0, sticky="w", padx=6)
        ctk.CTkLabel(page, text=sub, font=font(13), text_color=MUTED).grid(row=1, column=0, sticky="w", padx=6, pady=(0, 12))

    # 홈
    def _build_home(self, page):
        self._header(page, "홈", "카메라를 향해 제스처를 해 보세요. 인식되면 아래 카드가 빛나고 실행 내용이 표시됩니다.")
        ctk.CTkButton(page, text="처음이면 도움말 →", width=130, height=26, font=font(12), fg_color="transparent",
                      hover_color=CARD2, text_color=MUTED, command=lambda: self.show_page("help")
                      ).grid(row=1, column=0, sticky="e", padx=6, pady=(0, 12))
        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(1, weight=1)

        cam = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
        cam.grid(row=0, column=0, sticky="n")
        self.video = tk.Label(cam, bg="#111318", width=640, height=480, bd=0)
        self.video.pack(padx=10, pady=10)
        bar = ctk.CTkFrame(cam, fg_color="transparent"); bar.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(bar, text="움직임", font=font(11), text_color=MUTED).pack(side="left")
        self.energy = ctk.CTkProgressBar(bar, width=200, height=8, progress_color="#3aa0ff")
        self.energy.set(0); self.energy.pack(side="left", padx=8)
        self.energy_lbl = ctk.CTkLabel(bar, text="", font=font(11), text_color=MUTED); self.energy_lbl.pack(side="left")
        ctk.CTkCheckBox(bar, text="거울처럼 보기", variable=self._mirror, font=font(12)).pack(side="right")

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
        right.grid_columnconfigure(0, weight=1)
        gate = ctk.CTkFrame(right, fg_color=CARD, corner_radius=14, border_width=2, border_color=CARD)
        gate.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.gate_card = gate
        ctk.CTkLabel(gate, text="게이트 (정적 포즈, YOLO)", font=font(12), text_color=MUTED).pack(anchor="w", padx=16, pady=(12, 0))
        self.gate_big = ctk.CTkLabel(gate, text="연결 중...", font=font(20, "bold"))
        self.gate_big.pack(anchor="w", padx=16)
        self.gate_hint = ctk.CTkLabel(gate, text="", font=font(12), text_color=MUTED, anchor="w", wraplength=380, justify="left")
        self.gate_hint.pack(fill="x", padx=16)
        hb = ctk.CTkFrame(gate, fg_color="transparent"); hb.pack(fill="x", padx=16, pady=(6, 14))
        self.pose_bar = ctk.CTkProgressBar(hb, height=10, progress_color="#b478ff"); self.pose_bar.set(0)
        self.pose_bar.pack(side="left", fill="x", expand=True)
        self.pose_lbl = ctk.CTkLabel(hb, text="", font=font(12), width=150, anchor="e"); self.pose_lbl.pack(side="left", padx=(8, 0))

        now = ctk.CTkFrame(right, fg_color=CARD, corner_radius=14)
        now.grid(row=1, column=0, sticky="ew")
        ctk.CTkLabel(now, text="지금 인식", font=font(12), text_color=MUTED).pack(anchor="w", padx=16, pady=(12, 0))
        self.last_lbl = ctk.CTkLabel(now, text="—", font=font(30, "bold"))
        self.last_lbl.pack(anchor="w", padx=16)
        self.conf_bar = ctk.CTkProgressBar(now, height=10, progress_color="#8d99ae")
        self.conf_bar.set(0); self.conf_bar.pack(fill="x", padx=16, pady=(4, 2))
        self.conf_lbl = ctk.CTkLabel(now, text="확신도 —", font=font(12), text_color=MUTED)
        self.conf_lbl.pack(anchor="w", padx=16)
        self.msg_lbl = ctk.CTkLabel(now, text="손바닥이 카메라를 향하게 하고, 동작 사이에 잠깐 멈추세요.", font=font(13),
                                    wraplength=380, justify="left", anchor="w")
        self.msg_lbl.pack(fill="x", padx=16, pady=(6, 14))

        self.prob_rows = {}
        pr = ctk.CTkFrame(right, fg_color=CARD, corner_radius=14)
        pr.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ctk.CTkLabel(pr, text="모델 확률", font=font(12), text_color=MUTED).pack(anchor="w", padx=16, pady=(12, 4))
        for lab in config.LABELS:
            r = ctk.CTkFrame(pr, fg_color="transparent"); r.pack(fill="x", padx=16, pady=2)
            name = GESTURE_NAMES.get(lab, (lab, ""))[0] if lab != "no_gesture" else "동작 아님"
            ctk.CTkLabel(r, text=name, font=font(12), width=100, anchor="w").pack(side="left")
            b = ctk.CTkProgressBar(r, height=8, progress_color=ACCENT.get(lab, "#999")); b.set(0)
            b.pack(side="left", fill="x", expand=True, padx=8)
            v = ctk.CTkLabel(r, text="0.00", font=font(11), width=52, anchor="e", text_color=MUTED); v.pack(side="left")
            self.prob_rows[lab] = (b, v)
        ctk.CTkLabel(pr, text="", font=font(4)).pack(pady=2)

        cards = ctk.CTkFrame(body, fg_color="transparent")
        cards.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        for i in range(len(COMMAND_LABELS)):
            cards.grid_columnconfigure(i, weight=1)
        self.home_cards = {}
        for i, lab in enumerate(COMMAND_LABELS):
            name, desc = GESTURE_NAMES.get(lab, (lab, ""))
            c = ctk.CTkFrame(cards, fg_color=CARD, corner_radius=14, border_width=2, border_color=CARD)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 12, 0))
            top = ctk.CTkFrame(c, fg_color="transparent"); top.pack(fill="x", padx=16, pady=(12, 0))
            ctk.CTkLabel(top, text=ICON.get(lab, ""), font=font(22, "bold"), text_color=ACCENT[lab]).pack(side="left")
            ctk.CTkLabel(top, text=name, font=font(15, "bold")).pack(side="left", padx=8)
            ctk.CTkLabel(c, text=desc, font=font(11), text_color=MUTED, anchor="w").pack(fill="x", padx=16)
            holder = ctk.CTkFrame(c, fg_color="transparent"); holder.pack(fill="x", padx=16, pady=(8, 12))
            self.home_cards[lab] = {"frame": c, "holder": holder, "chips": None}

    # 키 매핑
    def _build_map(self, page):
        self._header(page, "키 매핑", "제스처마다 누를 키를 정합니다. 키 칸을 클릭하고 조합을 누르거나, 기능을 이름으로 고르세요. 바꾸면 바로 저장됩니다.")
        page.grid_rowconfigure(2, weight=1)
        body = ctk.CTkScrollableFrame(page, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        pf = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
        pf.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(pf, text="프리셋 — 용도에 맞는 조합을 한 번에", font=font(12), text_color=MUTED).pack(anchor="w", padx=16, pady=(12, 2))
        row = ctk.CTkFrame(pf, fg_color="transparent"); row.pack(fill="x", padx=16, pady=(0, 12))
        self.preset_seg = ctk.CTkSegmentedButton(row, values=list(PRESETS), font=font(13), height=34)
        self.preset_seg.set(list(PRESETS)[0])
        self.preset_seg.pack(side="left")
        ctk.CTkButton(row, text="적용", width=90, font=font(13, "bold"), command=self.apply_preset).pack(side="left", padx=12)
        self.saved_lbl = ctk.CTkLabel(row, text="", font=font(12), text_color="#2ecc71")
        self.saved_lbl.pack(side="left")
        self.conflict_lbl = ctk.CTkLabel(pf, text="", font=font(12), text_color="#f39c12")
        self.conflict_lbl.pack(anchor="w", padx=16, pady=(0, 8))

        func_values = [f"{cat} · {name}" for cat, name, _ in FUNCTIONS]
        self.map_cards = {}
        for i, lab in enumerate(COMMAND_LABELS):
            name, desc = GESTURE_NAMES.get(lab, (lab, ""))
            c = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
            c.grid(row=1 + i, column=0, sticky="ew", pady=6)
            c.grid_columnconfigure(2, weight=1)
            ctk.CTkFrame(c, width=6, height=10, fg_color=ACCENT[lab], corner_radius=3).grid(row=0, column=0, rowspan=4, sticky="ns", padx=(14, 0), pady=14)
            ctk.CTkLabel(c, text=ICON.get(lab, ""), font=font(30, "bold"), text_color=ACCENT[lab], width=50).grid(row=0, column=1, rowspan=4, padx=(12, 6))
            head = ctk.CTkFrame(c, fg_color="transparent"); head.grid(row=0, column=2, columnspan=2, sticky="ew", pady=(12, 0))
            ctk.CTkLabel(head, text=name, font=font(17, "bold")).pack(side="left")
            ctk.CTkLabel(head, text="   " + desc, font=font(12), text_color=MUTED).pack(side="left")
            sw = ctk.CTkSwitch(head, text="사용", font=font(12), command=lambda l=lab: self.toggle_enabled(l))
            sw.pack(side="right", padx=(0, 16))

            rec_row = ctk.CTkFrame(c, fg_color="transparent"); rec_row.grid(row=1, column=2, columnspan=2, sticky="ew", padx=(0, 16), pady=(8, 0))
            rec_row.grid_columnconfigure(0, weight=1)
            rec = KeyRecorder(rec_row, ACCENT[lab], on_change=lambda ks, l=lab: self.apply_key(l, ks, ""),
                              on_clear=lambda l=lab: self.apply_key(l, [], ""))
            rec.grid(row=0, column=0, sticky="ew")
            win_var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(rec_row, text="Win", variable=win_var, width=60, font=font(12),
                            command=lambda l=lab: self._toggle_win(l)).grid(row=0, column=1, padx=(10, 0))
            fn = ctk.CTkOptionMenu(rec_row, values=func_values, width=200, font=font(12), dynamic_resizing=False,
                                   command=lambda v, l=lab: self._pick_function(l, v))
            fn.set("기능 고르기 ▾")
            fn.grid(row=0, column=2, padx=(10, 0))

            tools = ctk.CTkFrame(c, fg_color="transparent"); tools.grid(row=2, column=2, columnspan=2, sticky="ew", padx=(0, 16), pady=(8, 14))
            ctk.CTkLabel(tools, text="이름", font=font(12), text_color=MUTED).pack(side="left")
            name_var = tk.StringVar()
            ent = ctk.CTkEntry(tools, textvariable=name_var, width=200, font=font(12), placeholder_text="예: 다음 슬라이드")
            ent.pack(side="left", padx=(6, 12))
            ent.bind("<Return>", lambda e, l=lab: self._rename(l))
            ent.bind("<FocusOut>", lambda e, l=lab: self._rename(l))
            ctk.CTkButton(tools, text="눌러보기", width=84, font=font(12), fg_color="transparent", border_width=1,
                          text_color=TEXT, command=lambda l=lab: self.try_key(l)).pack(side="left")
            ctk.CTkButton(tools, text="기본값", width=70, font=font(12), fg_color="transparent", border_width=1,
                          text_color=MUTED, command=lambda l=lab: self.reset_key(l)).pack(side="left", padx=6)
            ctk.CTkButton(tools, text="지우기", width=70, font=font(12), fg_color="transparent", border_width=1,
                          text_color=MUTED, command=lambda l=lab: self.apply_key(l, [], "")).pack(side="left")
            self.map_cards[lab] = {"rec": rec, "switch": sw, "last_keys": [], "win": win_var, "name": name_var, "fn": fn}
        ctk.CTkLabel(body, text=f"저장 위치: {config.ROOT / 'gestures.json'}  ·  Win 키는 키보드로 잡히지 않아 체크로 붙입니다.",
                     font=font(11), text_color=MUTED).grid(row=10, column=0, sticky="w", padx=6, pady=(8, 0))

    # 설정
    def _build_settings(self, page):
        self._header(page, "설정", "인식 민감도, 화면, 카메라, 게이트")
        body = ctk.CTkScrollableFrame(page, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew")
        page.grid_rowconfigure(2, weight=1)
        body.grid_columnconfigure(0, weight=1)

        def card(title, sub=None):
            c = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
            c.grid(sticky="ew", pady=6)
            ctk.CTkLabel(c, text=title, font=font(15, "bold")).pack(anchor="w", padx=16, pady=(12, 0))
            if sub:
                ctk.CTkLabel(c, text=sub, font=font(12), text_color=MUTED, wraplength=760, justify="left").pack(anchor="w", padx=16)
            inner = ctk.CTkFrame(c, fg_color="transparent"); inner.pack(fill="x", padx=16, pady=(8, 14))
            return inner

        r = card("확신도 문턱", "모델이 이 값보다 확신할 때만 실행합니다. 낮추면 잘 잡히지만 오작동이 늘 수 있습니다. (기본 0.80)")
        self.conf_var = tk.DoubleVar(value=0.8)
        self.conf_slider = ctk.CTkSlider(r, from_=0.5, to=1.0, number_of_steps=10, variable=self.conf_var,
                                         command=self._on_conf, width=320)
        self.conf_slider.pack(side="left")
        self.conf_slider.bind("<ButtonRelease-1>", lambda e: self._save_conf())
        self.conf_val = ctk.CTkLabel(r, text="0.80", font=font(16, "bold"), width=60); self.conf_val.pack(side="left", padx=12)

        r = card("화면")
        ctk.CTkSwitch(r, text="거울처럼 보기 (판정에는 영향 없음)", variable=self._mirror, font=font(13)).pack(anchor="w")
        th = ctk.CTkFrame(r, fg_color="transparent"); th.pack(anchor="w", pady=(10, 0))
        ctk.CTkLabel(th, text="테마", font=font(13)).pack(side="left", padx=(0, 10))
        seg = ctk.CTkSegmentedButton(th, values=["어둡게", "밝게"], font=font(12),
                                     command=lambda v: ctk.set_appearance_mode("dark" if v == "어둡게" else "light"))
        seg.set("어둡게"); seg.pack(side="left")

        r = card("카메라")
        ctk.CTkLabel(r, text="카메라 번호", font=font(13)).pack(side="left")
        self.cam_menu = ctk.CTkOptionMenu(r, values=[str(i) for i in range(5)], width=80, font=font(13))
        self.cam_menu.set(str(self.camera_index)); self.cam_menu.pack(side="left", padx=10)
        ctk.CTkButton(r, text="다시 연결", width=100, font=font(13), command=self.reconnect_camera).pack(side="left")

        r = card("YOLO 게이트 (정적 포즈, 유지 시간이 트리거)",
                 "손바닥 → 인식 시작 · 주먹 → 인식 끝 · 세 번째 포즈 → 키 매핑의 '세 번째 포즈' 키. "
                 "게이트가 닫혀 있으면 동적 제스처는 판정만 하고 키를 누르지 않습니다.")
        self.gate_use = ctk.CTkSwitch(r, text="YOLO 게이트 사용 (끄면 항상 열림, 다시 시작 후 적용)", font=font(13),
                                      command=self._on_gate_use)
        self.gate_use.pack(anchor="w")
        hs = ctk.CTkFrame(r, fg_color="transparent"); hs.pack(anchor="w", fill="x", pady=(10, 0))
        ctk.CTkLabel(hs, text="유지 시간", font=font(13)).pack(side="left")
        self.hold_var = tk.DoubleVar(value=3.0)
        self.hold_slider = ctk.CTkSlider(hs, from_=1.0, to=5.0, number_of_steps=8, variable=self.hold_var,
                                         command=self._on_hold, width=240)
        self.hold_slider.pack(side="left", padx=10)
        self.hold_slider.bind("<ButtonRelease-1>", lambda e: self._save_hold())
        self.hold_val = ctk.CTkLabel(hs, text="3.0초", font=font(14, "bold"), width=60); self.hold_val.pack(side="left")
        self.gate_info = ctk.CTkLabel(r, text="", font=font(11), text_color=MUTED, wraplength=760, justify="left")
        self.gate_info.pack(anchor="w", pady=(8, 0))
        self.gate_test = ctk.CTkSwitch(r, text="게이트 수동 테스트: 닫기 (YOLO 없이 돌 때만)", font=font(13),
                                       command=self._on_gate_test)
        self.gate_test.pack(anchor="w", pady=(8, 0))

        r = card("진단")
        ctk.CTkButton(r, text="인식 상태 초기화", width=140, font=font(13), command=self.reset).pack(side="left")
        ctk.CTkButton(r, text="기록 폴더 열기", width=120, font=font(13), fg_color="transparent", border_width=1,
                      text_color=TEXT, command=self.open_log_dir).pack(side="left", padx=10)
        self.log_path_lbl = ctk.CTkLabel(r, text="", font=font(11), text_color=MUTED); self.log_path_lbl.pack(side="left")

    # 기록
    def _build_log(self, page):
        self._header(page, "기록", "최근 판정과 실행. 전체 기록은 reports/realtime/ 의 세션 파일에 남습니다.")
        page.grid_rowconfigure(2, weight=1)
        self.log_box = ctk.CTkTextbox(page, font=ctk.CTkFont(family="Consolas", size=12), fg_color=CARD, corner_radius=14,
                                      wrap="none")
        self.log_box.grid(row=2, column=0, sticky="nsew")
        self.log_box.configure(state="disabled")
        foot = ctk.CTkFrame(page, fg_color="transparent"); foot.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.count_lbl = ctk.CTkLabel(foot, text="실행 0건", font=font(13, "bold")); self.count_lbl.pack(side="left", padx=6)
        ctk.CTkButton(foot, text="지우기", width=80, font=font(12), fg_color="transparent", border_width=1, text_color=MUTED,
                      command=self._clear_log).pack(side="right")

    # 도움말
    def _hold_seconds(self) -> float:
        p = self.pipeline
        if p is None:
            return 2.0
        return float(p.mapper.gate.get("hold_seconds", 2.0))

    def _refresh_help_hold(self):
        """도움말에 적힌 포즈 유지 시간을 현재 설정값으로 다시 쓴다 (pipeline 준비 후, 슬라이더 조작 시)."""
        h = f"{self._hold_seconds():g}"
        for lbl, tmpl in getattr(self, "_help_hold_labels", []):
            lbl.configure(text=tmpl.replace("{h}", h))

    def _build_help(self, page):
        self._header(page, "도움말", "처음 쓰는 분을 위한 요약입니다. 더 자세한 내용은 맨 아래 설명서를 여세요.")
        page.grid_rowconfigure(2, weight=1)
        body = ctk.CTkScrollableFrame(page, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        self.help_body = body
        self._help_hold_labels: list[tuple[ctk.CTkLabel, str]] = []
        BLUE = "#3aa0ff"

        def card(title, sub=None):
            c = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
            c.grid(sticky="ew", pady=6)
            ctk.CTkLabel(c, text=title, font=font(15, "bold")).pack(anchor="w", padx=16, pady=(12, 0))
            if sub:
                ctk.CTkLabel(c, text=sub, font=font(12), text_color=MUTED, wraplength=900, justify="left").pack(anchor="w", padx=16)
            inner = ctk.CTkFrame(c, fg_color="transparent"); inner.pack(fill="x", padx=16, pady=(8, 14))
            return inner

        def hold_label(parent, tmpl, **kw):
            """'{h}' 자리에 유지 시간이 들어가는 라벨. 값이 바뀌면 _refresh_help_hold 가 다시 쓴다."""
            lbl = ctk.CTkLabel(parent, text=tmpl.replace("{h}", "2"), **kw)
            self._help_hold_labels.append((lbl, tmpl))
            return lbl

        def badge(parent, text, color=BLUE):
            return ctk.CTkLabel(parent, text=text, font=font(13, "bold"), fg_color=color, text_color="#ffffff",
                                corner_radius=14, width=28, height=28)

        def numbered(parent, items, color=BLUE):
            for i, (main, sub) in enumerate(items):
                row = ctk.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", pady=4)
                badge(row, str(i + 1), color).pack(side="left", padx=(0, 12), anchor="n", pady=2)
                txt = ctk.CTkFrame(row, fg_color="transparent"); txt.pack(side="left", fill="x", expand=True)
                hold_label(txt, main, font=font(14, "bold"), anchor="w", justify="left", wraplength=880).pack(anchor="w")
                hold_label(txt, sub, font=font(12), text_color=MUTED, anchor="w", justify="left", wraplength=880).pack(anchor="w")

        # ① 3단계로 시작
        r = card("3단계로 시작")
        numbered(r, [
            ("사이드바 아래 '실제 키 입력' 스위치를 켭니다", "연습만 하려면 끈 채로 — 인식은 되지만 키는 누르지 않습니다."),
            ("손바닥을 카메라에 {h}초 보여 줍니다", "게이트가 열리고 홈의 게이트 카드가 초록 '열림 — 인식 중' 으로 바뀝니다. 게이트 모델이 없으면 건너뜁니다."),
            ("제스처를 합니다", "홈 아래 카드가 빛나고 오른쪽 위에 '실행: …' 알림이 뜹니다. 키는 앞에 있는 창에 들어갑니다."),
        ])

        # ② 제스처 하는 법 (2×2) + 정적 포즈
        r = card("제스처 하는 법", "손바닥이 카메라를 향하게 · 손 전체가 화면 안에 · 동작 사이 0.5초 이상 멈추기")
        grid = ctk.CTkFrame(r, fg_color="transparent"); grid.pack(fill="x")
        grid.grid_columnconfigure((0, 1), weight=1, uniform="help")
        how = {
            "swipe_left": ("편 손을 자기 왼쪽으로 손바닥 하나 이상 휙",
                           "화면을 보고 화면 왼쪽으로 밀면 반대 방향이라 안 잡힙니다. 헷갈리면 '거울처럼 보기'를 켜세요."),
            "make_fist": ("편 손을 화면 안에 멈춘 뒤 쥐고, 쥐자마자 내리기",
                          "0.8초 넘게 쥐고 있으면 '끝내려는 주먹'으로 보고 실행하지 않습니다."),
            "finger_snap": ("엄지+중지를 붙였다가 튕겨서 떨어뜨리기",
                            "튕긴 뒤 검지는 편 채로. 튕기는 순간 바로 판정됩니다."),
            "cancel": ("정적 포즈를 {h}초 유지 (YOLO)",
                       "게이트가 열려 있을 때만 동작하며, 키 매핑의 '세 번째 포즈' 키가 눌립니다."),
        }
        for i, lab in enumerate(COMMAND_LABELS):
            name, _desc = GESTURE_NAMES.get(lab, (lab, ""))
            main, tip = how.get(lab, ("", ""))
            c = ctk.CTkFrame(grid, fg_color=CARD2, corner_radius=12)
            c.grid(row=i // 2, column=i % 2, sticky="nsew", padx=(0, 10) if i % 2 == 0 else (0, 0), pady=5)
            top = ctk.CTkFrame(c, fg_color="transparent"); top.pack(fill="x", padx=14, pady=(10, 0))
            ctk.CTkLabel(top, text=ICON.get(lab, ""), font=font(22, "bold"), text_color=ACCENT[lab]).pack(side="left")
            ctk.CTkLabel(top, text=name, font=font(15, "bold")).pack(side="left", padx=8)
            hold_label(c, "이렇게:  " + main, font=font(13), anchor="w", justify="left", wraplength=400).pack(fill="x", padx=14, pady=(4, 0))
            hold_label(c, "팁:  " + tip, font=font(12), text_color=MUTED, anchor="w", justify="left", wraplength=400).pack(fill="x", padx=14, pady=(2, 12))
        poses = ctk.CTkFrame(r, fg_color="transparent"); poses.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(poses, text="정적 포즈 (YOLO)", font=font(12), text_color=MUTED).pack(side="left", padx=(0, 10))
        for tmpl, color in [("✋  손바닥 {h}초 → 인식 시작", ACCENT["gate_open"]), ("✊  주먹 {h}초 → 인식 끝", ACCENT["gate_close"]),
                            ("⚑  세 번째 포즈 {h}초 → 매핑된 키", ACCENT["cancel"])]:
            hold_label(poses, tmpl, font=font(12, "bold"), fg_color=color, text_color="#ffffff", corner_radius=6,
                       padx=10, pady=3).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(r, text="유지 시간은 설정 페이지에서 1~5초로 바꿉니다. 앱을 켜면 게이트는 닫힘으로 시작하고, 닫혀 있는 동안은 판정만 하고 키를 누르지 않습니다.",
                     font=font(11), text_color=MUTED, anchor="w", justify="left", wraplength=900).pack(fill="x", pady=(8, 0))

        # ③ 키 바꾸기
        r = card("키 바꾸기", "키 매핑 페이지에서. 바꾸는 즉시 gestures.json 에 저장되고 인식에 반영됩니다.")
        numbered(r, [
            ("직접 녹화", "키 칸을 클릭하고 조합을 누릅니다 (예: Ctrl+Shift+T). Esc 취소 · Backspace 지우기 · Win 키는 옆의 체크박스로."),
            ("기능 고르기", "단축키를 몰라도 '다음 슬라이드' 처럼 이름으로 고릅니다 (창 / 미디어 / 발표 / 브라우저 / 편집 / 기타)."),
            ("프리셋", "창 전환 · 발표 · 음악·영상 · 브라우저 · 바탕화면 — 동작 4개를 한 번에 바꾸고 '적용'을 누릅니다."),
        ], color=ACCENT["cancel"])
        ctk.CTkButton(r, text="키 매핑 페이지로 →", width=150, font=font(12), fg_color="transparent", border_width=1,
                      text_color=TEXT, command=lambda: self.show_page("map")).pack(anchor="w", pady=(6, 0))

        # ④ 잘 안 될 때
        r = card("잘 안 될 때")
        hdr = ctk.CTkFrame(r, fg_color="transparent"); hdr.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(hdr, text="증상", font=font(11), text_color=MUTED, width=280, anchor="w").pack(side="left", padx=(12, 8))
        ctk.CTkLabel(hdr, text="해결", font=font(11), text_color=MUTED, anchor="w").pack(side="left")
        fixes = [
            ("카메라가 안 열리거나 1 fps", "다른 프로그램(수집기·Teams 등)이 카메라를 쓰는지 확인. 설정 → 카메라 번호를 바꾸고 '다시 연결'."),
            ("스와이프가 '동작 아님'", "자기 왼쪽으로, 손바닥 하나 이상 거리. 화면에서는 손이 오른쪽으로 흘러야 맞습니다."),
            ("주먹 명령이 안 됨 (held fist)", "0.8초 넘게 쥐면 끝내려는 주먹으로 봅니다 → 쥐자마자 내리기. 편 손이 먼저 화면 안에 보여야 합니다."),
            ("튕기기가 '동작 아님'", "엄지·중지를 확실히 떨어뜨리고 검지는 편 채로."),
            ("게이트가 안 열림", "손바닥을 정면으로 크게, 보라 막대가 찰 때까지 유지. 설정에서 유지 시간을 줄여 보세요."),
            ("키가 엉뚱한 창에 눌림", "키는 앞에 있는 창에 들어갑니다 → 제어할 창을 클릭해 앞에 두고 제스처. 이 앱 창은 옆으로."),
        ]
        for i, (sym, fix) in enumerate(fixes):
            row = ctk.CTkFrame(r, fg_color=CARD2 if i % 2 == 0 else "transparent", corner_radius=8)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=sym, font=font(13, "bold"), width=280, anchor="w", justify="left", wraplength=270).pack(side="left", padx=(12, 8), pady=6)
            ctk.CTkLabel(row, text=fix, font=font(12), anchor="w", justify="left", wraplength=600).pack(side="left", fill="x", expand=True, padx=(0, 12), pady=6)

        # ⑤ 설명서 · 기록
        foot = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
        foot.grid(sticky="ew", pady=6)
        row = ctk.CTkFrame(foot, fg_color="transparent"); row.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(row, text="자세한 설명서:  docs/user_guide.md", font=font(13)).pack(side="left")
        ctk.CTkButton(row, text="설명서 열기", width=100, font=font(12, "bold"), command=self.open_guide).pack(side="right")
        ctk.CTkButton(row, text="기록 폴더 열기", width=110, font=font(12), fg_color="transparent", border_width=1,
                      text_color=TEXT, command=self.open_log_dir).pack(side="right", padx=8)

    # ───────────────────────── 워커 스레드 ─────────────────────────
    def _start_worker(self):
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        try:
            if self.pipeline is None:
                from gesture.pipeline import Pipeline
                self.pipeline = Pipeline(model_file=self.model_file, gestures_path=self.gestures_path,
                                         dry_run=not self._want_live)
                self._msgq.put(("ready", None))
            cap = open_camera(self.camera_index)
            if cap is None:
                self._msgq.put(("error", f"카메라 {self.camera_index} 를 열 수 없습니다. 다른 프로그램이 쓰고 있는지, 번호가 맞는지 확인하세요."))
                return
            self.cap = cap
            self._msgq.put(("camera", self.camera_index))
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._msgq.put(("error", "카메라 프레임을 읽지 못했습니다."))
                    break
                self.pipeline.step(frame)
                with self._frame_lock:
                    self._frame = frame
        except Exception as e:
            self._msgq.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            if self.cap is not None:
                self.cap.release(); self.cap = None

    # ───────────────────────── 화면 갱신 ─────────────────────────
    def _tick(self):
        while not self._msgq.empty():
            kind, payload = self._msgq.get()
            if kind == "ready":
                self._on_ready()
            elif kind == "camera":
                self.cam_lbl.configure(text=f"카메라 {payload} 연결됨")
            elif kind == "error":
                self.cam_lbl.configure(text="오류", text_color="#e74c3c")
                self.msg_lbl.configure(text=payload)
                self._show_toast("⚠ " + payload, color="#c0392b", sec=6)
        if self.pipeline is not None:
            self._render()
        now = time.perf_counter()
        if self._toast_until and now > self._toast_until:
            self.toast.place_forget(); self._toast_until = 0.0
        self.after(30, self._tick)

    def _on_ready(self):
        p = self.pipeline
        self.conf_var.set(p.mapper.min_confidence)
        self.conf_val.configure(text=f"{p.mapper.min_confidence:.2f}")
        self.log_path_lbl.configure(text=p.log.path.name)
        if p.mapper.gate.get("enabled", True):
            self.gate_use.select()
        hold = float(p.mapper.gate.get("hold_seconds", 3.0))
        self.hold_var.set(hold); self.hold_val.configure(text=f"{hold:.1f}초")
        self._refresh_help_hold()
        if p.gate_connected:
            self.gate_info.configure(text=f"모델 {getattr(p.gate, 'path', '')} · {'GPU' if getattr(p.gate, 'on_gpu', False) else 'CPU'}")
        else:
            self.gate_info.configure(text="YOLO 게이트 없이 실행 중 (항상 열림)" + (f" — {p.gate_error}" if p.gate_error else ""))
        if not p.dry_run:
            self.mode_switch.select()
        self._refresh_mode()
        self._refresh_map()

    def _render(self):
        p = self.pipeline
        st = p.snapshot()
        with self._frame_lock:
            frame = self._frame
        now = time.perf_counter()
        if frame is not None:
            vis = frame.copy()
            if st.landmarks is not None:
                draw_landmarks(vis, st.landmarks)
            if self._mirror.get():
                vis = cv2.flip(vis, 1)
            img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(img, "RGBA")
            if st.seg_state == "ACTIVE":
                chip, col = f"동작 중 {st.active_sec:.1f}s", (231, 76, 60, 220)
            elif st.in_cooldown:
                chip, col = "쿨다운", (243, 156, 18, 220)
            else:
                chip, col = ("대기 · 손 보임" if st.landmarks is not None else "대기 · 손 없음"), (40, 44, 52, 200)
            tw = d.textlength(chip, font=PIL_FONT)
            d.rounded_rectangle((10, 10, 10 + tw + 24, 42), radius=8, fill=col)
            d.text((22, 15), chip, font=PIL_FONT, fill=(255, 255, 255, 255))
            fps = f"{st.fps:.0f} fps"
            d.rounded_rectangle((img.width - 80, 12, img.width - 10, 38), radius=8, fill=(40, 44, 52, 180))
            d.text((img.width - 72, 16), fps, font=PIL_FONT_S, fill=(220, 220, 220, 255))
            if st.gate_connected and not st.gate_open:
                d.rounded_rectangle((img.width // 2 - 150, img.height - 48, img.width // 2 + 150, img.height - 12), radius=10,
                                    fill=(30, 33, 40, 200))
                d.text((img.width // 2 - 120, img.height - 41), f"손바닥을 {st.pose_needed:.0f}초 보여 주면 시작", font=PIL_FONT, fill=(255, 255, 255, 255))
            if st.pose:
                txt = f"{POSE_KR.get(st.pose, st.pose)} {min(st.pose_hold, st.pose_needed):.1f}/{st.pose_needed:.0f}s"
                tw = d.textlength(txt, font=PIL_FONT)
                d.rounded_rectangle((10, 50, 10 + tw + 24, 82), radius=8, fill=(180, 120, 255, 220))
                d.text((22, 55), txt, font=PIL_FONT, fill=(255, 255, 255, 255))
            if now < self._flash_until:
                d.rounded_rectangle((3, 3, img.width - 4, img.height - 4), radius=12, outline=(46, 204, 113, 255), width=8)
            tkimg = ImageTk.PhotoImage(img)
            self.video.config(image=tkimg); self.video.image = tkimg

        self.energy.set(min(st.energy, 0.3) / 0.3)
        self.energy_lbl.configure(text=f"{st.energy:.3f}  (시작 {st.on_level:.3f} / 멈춤 {st.off_level:.3f})")

        ev = st.last_event
        if ev is not None and ev is not self._last_ev:
            self._last_ev = ev
            self._on_event(ev, now)
        for lab, card in self.home_cards.items():
            glow = self._card_glow.get(lab, 0.0)
            card["frame"].configure(border_color=ACCENT[lab] if now < glow else CARD)
        self.count_lbl.configure(text=f"실행 {st.n_executed}건")
        if len(st.events) != self._n_events_shown:
            new = st.events[self._n_events_shown:] if len(st.events) >= self._n_events_shown else st.events
            self.log_box.configure(state="normal")
            for e in new:
                name = {"no_gesture": "동작 아님", "gate_open": "게이트", "gate_close": "게이트", "skipped": "건너뜀"}.get(
                    e.label, GESTURE_NAMES.get(e.label, (e.label, ""))[0])
                mark = "●" if e.executed else " "
                self.log_box.insert("end", f"{time.strftime('%H:%M:%S')} {mark} {name:7s} {e.confidence:4.0%} {e.duration_sec:.2f}s  {self._friendly(e)}\n")
            self.log_box.see("end"); self.log_box.configure(state="disabled")
            self._n_events_shown = len(st.events)
        self._refresh_gate(st)

    def _on_event(self, ev, now):
        if ev.label in ("gate_open", "gate_close"):
            opened = ev.label == "gate_open"
            hold = float(self.pipeline.mapper.gate.get("hold_seconds", 3.0))
            self._show_toast((f"✋ 손바닥 {hold:.0f}초 → 인식 시작" if opened else f"✊ 주먹 {hold:.0f}초 → 인식 끝"),
                             color="#1e7e34" if opened else "#8e2b2b", sec=2.5)
            self.last_lbl.configure(text="인식 시작" if opened else "인식 끝", text_color=ACCENT[ev.label])
            self.conf_bar.set(0); self.conf_lbl.configure(text="—"); self.msg_lbl.configure(text=self._friendly(ev))
            return
        if ev.label == "skipped":
            name = "건너뜀"
        elif ev.label == "no_gesture":
            name = "동작 아님"
        else:
            name = GESTURE_NAMES.get(ev.label, (ev.label, ""))[0]
        self.last_lbl.configure(text=name, text_color=ACCENT.get(ev.label, "#e8eaf0"))
        self.conf_bar.configure(progress_color=ACCENT.get(ev.label, "#8d99ae"))
        self.conf_bar.set(ev.confidence if ev.probs is not None else 0)
        self.conf_lbl.configure(text=f"확신도 {ev.confidence:.0%} · {ev.duration_sec:.2f}초" if ev.probs is not None else "—")
        self.msg_lbl.configure(text=self._friendly(ev))
        if ev.probs is not None:
            for lab, pv in zip(config.LABELS, ev.probs):
                b, v = self.prob_rows[lab]; b.set(float(pv)); v.configure(text=f"{float(pv):.2f}")
        if ev.executed:
            self._flash_until = now + 0.5
            self._card_glow[ev.label] = now + 1.2
            what = ev.message.split("->", 1)[1].strip()
            prefix = "실행" if ev.message.startswith("[FIRE]") else "연습"
            self._show_toast(f"{ICON.get(ev.label, '')} {name}  →  {what}   ({prefix})",
                             color="#1e7e34" if prefix == "실행" else "#2f5d8a")

    def _show_toast(self, text, color="#1e7e34", sec=1.6):
        self.toast.configure(text=text, fg_color=color)
        # place() 로 띄워 레이아웃에 끼어들지 않게 한다 (grid 로 넣으면 뜰 때마다 아래 화면이 밀렸다 올라온다)
        self.toast.place(relx=1.0, y=4, x=-6, anchor="ne")
        self.toast.lift()
        self._toast_until = time.perf_counter() + sec

    def _friendly(self, ev) -> str:
        m = ev.message
        if ev.label in ("gate_open", "gate_close"):
            return m
        if m.startswith("PENDING"):
            return "주먹 확인 중… 내리면 실행됩니다"
        if m.startswith("held fist"):
            return "주먹을 계속 쥐고 있어 명령으로 보지 않음 (끝내기 의도)"
        if m.startswith("[FIRE]"):
            return "실행: " + m.split("->", 1)[1].strip()
        if m.startswith("[DRY]"):
            return "(연습) 누를 키: " + m.split("->", 1)[1].strip()
        if m.startswith("GUARD"):
            return "상식 검사에서 걸러짐: " + m[6:].replace(" -> ignored", "")
        if m.startswith("GATE"):
            return "게이트 닫힘 → 실행 안 함"
        if "cooldown" in m:
            return "직전 명령 직후라 무시"
        if "<" in m and "ignored" in m:
            return "확신도가 낮아 무시"
        if ev.label == "no_gesture":
            return "명령이 아닌 움직임"
        if ev.label == "skipped":
            return "손을 놓쳐 건너뜀"
        return m

    def _refresh_gate(self, st=None):
        p = self.pipeline
        st = st or p.snapshot()
        if st.gate_connected:
            self.gate_test.configure(state="disabled")
            if st.gate_open:
                self.gate_lbl.configure(text="게이트: 열림 — 인식 중", text_color="#2ecc71")
                self.gate_big.configure(text="열림 — 인식 중", text_color="#2ecc71")
                self.gate_hint.configure(text=f"주먹을 {st.pose_needed:.0f}초 유지하면 끝납니다. 세 번째 포즈 {st.pose_needed:.0f}초 = 정적 명령.")
                self.gate_card.configure(border_color="#2ecc71")
            else:
                self.gate_lbl.configure(text=f"게이트: 닫힘 — 손바닥 {st.pose_needed:.0f}초로 시작", text_color="#e74c3c")
                self.gate_big.configure(text="닫힘 — 대기", text_color="#e74c3c")
                self.gate_hint.configure(text=f"손바닥을 카메라에 {st.pose_needed:.0f}초 보여 주면 인식이 시작됩니다.")
                self.gate_card.configure(border_color=CARD)
            if st.pose:
                self.pose_bar.set(min(st.pose_hold / max(st.pose_needed, 0.1), 1.0))
                self.pose_lbl.configure(text=f"{POSE_KR.get(st.pose, st.pose)} {min(st.pose_hold, st.pose_needed):.1f} / {st.pose_needed:.0f}초")
            else:
                self.pose_bar.set(0); self.pose_lbl.configure(text="포즈 없음")
        elif not st.gate_open:
            self.gate_lbl.configure(text="게이트: 닫힘 (수동 테스트)", text_color="#e74c3c")
            self.gate_big.configure(text="닫힘 (수동 테스트)", text_color="#e74c3c")
            self.gate_hint.configure(text="설정에서 수동 테스트를 끄면 다시 열립니다."); self.pose_lbl.configure(text="")
        else:
            self.gate_lbl.configure(text="게이트: 없음 (항상 열림)", text_color=MUTED)
            self.gate_big.configure(text="YOLO 없음 — 항상 열림", text_color=TEXT)
            self.gate_hint.configure(text=st.gate_error or "models/yolo_gate.pt 가 있으면 자동으로 켜집니다."); self.pose_lbl.configure(text="")
        if st.pending_label:
            self.msg_lbl.configure(text="주먹 확인 중… (내리면 실행, 계속 쥐고 있으면 끝내기로 봄)")

    # ───────────────────────── 사용자 조작 ─────────────────────────
    def toggle_mode(self):
        p = self.pipeline
        if p is None:
            self.mode_switch.deselect(); return
        want_live = bool(self.mode_switch.get())
        if want_live:
            self._show_toast("● 실제 키 입력이 켜졌습니다. 제어할 창을 앞에 두세요.", color="#c0392b", sec=3)
        p.set_dry_run(not want_live)
        self._refresh_mode()

    def _refresh_mode(self):
        p = self.pipeline
        if p.dry_run:
            self.mode_lbl.configure(text="연습 모드 — 키를 누르지 않음", text_color=MUTED)
        else:
            self.mode_lbl.configure(text="● 실행 중 — 제스처가 키를 누릅니다", text_color="#e74c3c")

    def _refresh_map(self):
        p = self.pipeline
        used: dict[str, list[str]] = {}
        for lab, card in self.map_cards.items():
            keys = p.mapper.keys_for(lab)
            a = p.mapper.actions.get(lab) or {}
            card["rec"].set_keys(keys)
            card["win"].set("win" in keys)
            card["name"].set(a.get("label", "") if keys else "")
            card["fn"].set("기능 고르기 ▾")
            if keys:
                card["switch"].select(); card["last_keys"] = keys
                used.setdefault(key_display(keys), []).append(GESTURE_NAMES.get(lab, (lab,))[0])
            else:
                card["switch"].deselect()
        dup = [f"{k} ← {' / '.join(v)}" for k, v in used.items() if len(v) > 1]
        self.conflict_lbl.configure(text=("⚠ 같은 키가 겹칩니다: " + " ; ".join(dup)) if dup else "")
        for lab, card in self.home_cards.items():
            keys = p.mapper.keys_for(lab)
            if card["chips"] is not None:
                card["chips"].destroy()
            card["chips"] = key_chips(card["holder"], keys, size=13, color=ACCENT[lab] if keys else "#555")
            card["chips"].pack(side="left")

    def _toggle_win(self, lab: str):
        p = self.pipeline
        if p is None:
            return
        keys = [k for k in p.mapper.keys_for(lab) if k != "win"]
        if self.map_cards[lab]["win"].get():
            keys = ["win"] + keys
        a = p.mapper.actions.get(lab) or {}
        self.apply_key(lab, keys, a.get("label", ""))

    def _pick_function(self, lab: str, value: str):
        for cat, name, keys in FUNCTIONS:
            if f"{cat} · {name}" == value:
                self.apply_key(lab, keys, name)
                return

    def _rename(self, lab: str):
        p = self.pipeline
        if p is None:
            return
        keys = p.mapper.keys_for(lab)
        a = p.mapper.actions.get(lab) or {}
        new = self.map_cards[lab]["name"].get().strip()
        if keys and new != a.get("label", ""):
            self.apply_key(lab, keys, new)

    def try_key(self, lab: str):
        p = self.pipeline
        if p is None:
            return
        keys = p.mapper.keys_for(lab)
        if not keys:
            self._show_toast("지정된 키가 없습니다", color="#c0392b"); return
        try:
            press_combo(keys)
            self._show_toast(f"눌렀습니다: {key_display(keys)}", color="#2f5d8a")
        except Exception as e:
            self._show_toast(f"⚠ {e}", color="#c0392b", sec=4)

    def reset_key(self, lab: str):
        keys, disp = PRESETS[list(PRESETS)[0]][lab]
        self.apply_key(lab, keys, disp)

    def toggle_enabled(self, lab: str):
        p = self.pipeline
        card = self.map_cards[lab]
        if card["switch"].get():
            keys = card["last_keys"] or PRESETS[list(PRESETS)[0]][lab][0]
            self.apply_key(lab, keys, "")
        else:
            card["last_keys"] = p.mapper.keys_for(lab)
            self.apply_key(lab, [], "")

    def apply_key(self, lab: str, keys: list[str], name: str):
        p = self.pipeline
        if p is None:
            return
        try:
            p.mapper.set_action(lab, keys, name or None)
            p.mapper.save()
        except ValueError as e:
            self._show_toast("⚠ " + str(e), color="#c0392b", sec=4)
            return
        p.log.write(f"MAP   {lab} -> {key_display(keys)}")
        self._refresh_map()
        gname = GESTURE_NAMES.get(lab, (lab,))[0]
        self.saved_lbl.configure(text=f"✔ 저장됨 {time.strftime('%H:%M:%S')} — {gname} → {key_display(keys)}")

    def apply_preset(self):
        p = self.pipeline
        if p is None:
            return
        name = self.preset_seg.get()
        for lab, (keys, disp) in PRESETS[name].items():
            p.mapper.set_action(lab, keys, disp)
        p.mapper.save()
        p.log.write(f"MAP   preset '{name}'")
        self._refresh_map()
        self.saved_lbl.configure(text=f"✔ 저장됨 {time.strftime('%H:%M:%S')} — 프리셋 '{name}'")
        self._show_toast(f"프리셋 '{name}' 적용", color="#2f5d8a")

    def _on_conf(self, _v):
        v = round(float(self.conf_var.get()) / 0.05) * 0.05
        self.conf_val.configure(text=f"{v:.2f}")
        if self.pipeline is not None:
            self.pipeline.mapper.min_confidence = v

    def _save_conf(self):
        if self.pipeline is not None:
            self.pipeline.mapper.save()
            self._show_toast(f"확신도 문턱 {self.pipeline.mapper.min_confidence:.2f} 저장", color="#2f5d8a")

    def _on_gate_use(self):
        p = self.pipeline
        if p is None:
            return
        p.mapper.gate["enabled"] = bool(self.gate_use.get())
        p.mapper.save()
        self._show_toast("게이트 설정 저장 — 다시 시작하면 적용됩니다", color="#2f5d8a", sec=2.5)

    def _on_hold(self, _v):
        v = round(float(self.hold_var.get()) * 2) / 2
        self.hold_val.configure(text=f"{v:.1f}초")
        if self.pipeline is not None:
            self.pipeline.set_gate_hold(v)
            self._refresh_help_hold()

    def _save_hold(self):
        if self.pipeline is not None:
            self.pipeline.mapper.save()
            self._show_toast(f"포즈 유지 시간 {self.pipeline.mapper.gate['hold_seconds']:.1f}초 저장", color="#2f5d8a")

    def _on_gate_test(self):
        if self.pipeline is not None:
            self.pipeline.set_gate(not bool(self.gate_test.get()), source="manual")

    def reset(self):
        if self.pipeline is not None:
            self.pipeline.reset()
            self.last_lbl.configure(text="—", text_color=TEXT)
            self.msg_lbl.configure(text="초기화됨")

    def open_log_dir(self):
        d = config.REPORTS_DIR / "realtime"
        d.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(d)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(d)])

    def open_guide(self):
        p = config.ROOT / "docs" / "user_guide.md"
        if not p.exists():
            self._show_toast("⚠ docs/user_guide.md 가 없습니다", color="#c0392b", sec=4); return
        try:
            if sys.platform == "win32":
                os.startfile(p)  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except OSError:
            # .md 에 연결된 프로그램이 없으면 폴더라도 연다
            if sys.platform == "win32":
                os.startfile(p.parent)  # noqa: S606
            self._show_toast("연결된 프로그램이 없어 docs 폴더를 열었습니다 (메모장으로 여세요)", color="#2f5d8a", sec=4)

    def _clear_log(self):
        self.log_box.configure(state="normal"); self.log_box.delete("1.0", "end"); self.log_box.configure(state="disabled")

    def reconnect_camera(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)
        self.camera_index = int(self.cam_menu.get())
        self.cam_lbl.configure(text=f"카메라 {self.camera_index} 연결 중...", text_color=MUTED)
        self._start_worker()

    def on_close(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)
        if self.pipeline is not None:
            self.pipeline.close()
        self.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="시작부터 실제 키 입력")
    ap.add_argument("--model", default=None, help="기본 models/gru_gesture.pt")
    ap.add_argument("--gestures", default=None, help="기본 gestures.json")
    args = ap.parse_args()
    app = App(args.camera, args.live, args.model, args.gestures)
    app.mainloop()


if __name__ == "__main__":
    main()
