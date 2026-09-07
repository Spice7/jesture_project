"""시연용 서비스 UI: 웹캠 화면 + 제스처→키 매핑 편집 + 실행 기록. (Tkinter, 추가 라이브러리 없음)

  uv run python scripts/gesture_app.py            # 연습 모드(키 안 누름)로 시작
  uv run python scripts/gesture_app.py --live     # 처음부터 실제 키 입력
  uv run python scripts/gesture_app.py --camera 1

화면 구성
  왼쪽  : 카메라 화면(손 관절 표시), 상태(대기/동작 중/쿨다운), 마지막 판정과 확신도
  오른쪽: [제스처 → 키] 각 제스처의 키 조합을 '변경' 으로 바꾼다(키를 직접 누르거나 특수 키 선택). 바꾸면 바로 저장.
          [프리셋] 창 전환 / 발표 / 음악·영상 / 브라우저 / 바탕화면 을 한 번에 적용
          [설정] 확신도 문턱, 화면 좌우반전, 카메라 번호
          [기록] 최근 판정·실행

YOLO 게이트(정적 포즈 ON/OFF)는 아직 연결 전. YOLO 팀 코드에서 `app.pipeline.set_gate(True/False)` 를 부르면 된다.
연결 전에는 게이트가 항상 열려 있고, 설정 탭의 '게이트 수동 테스트' 로 닫힘 동작만 확인할 수 있다.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from gesture import config  # noqa: E402
from gesture.actions import MODIFIERS, SPECIAL_KEYS, VK, key_display, normalize_keys  # noqa: E402
from gesture.landmarks import draw_landmarks  # noqa: E402
from gesture.presets import GESTURE_NAMES, PRESETS  # noqa: E402

FONT = ("Malgun Gothic", 10)
FONT_B = ("Malgun Gothic", 10, "bold")
FONT_BIG = ("Malgun Gothic", 16, "bold")
FONT_TITLE = ("Malgun Gothic", 13, "bold")
COLORS = {"swipe_left": "#0aa2e6", "make_fist": "#e67e22", "finger_snap": "#27ae60", "no_gesture": "#8a8a8a",
          "skipped": "#8a8a8a"}
BGR_COLORS = {"swipe_left": (230, 162, 10), "make_fist": (34, 126, 230), "finger_snap": (96, 174, 39)}
COMMAND_LABELS = [l for l in config.LABELS if l != "no_gesture"]

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


class KeyCaptureDialog(tk.Toplevel):
    """키 조합 입력 창. 키를 직접 누르거나(보조키 체크 + 키 하나), 특수 키(재생/음량 등)를 고른다."""

    def __init__(self, master, gesture_label: str, current: list[str], current_name: str):
        super().__init__(master)
        self.title("키 바꾸기 — " + GESTURE_NAMES.get(gesture_label, (gesture_label, ""))[0])
        self.resizable(False, False)
        self.result: tuple[list[str], str] | None = None
        self.transient(master)
        self._mods = {m: tk.BooleanVar(value=(m in current)) for m in MODIFIERS}
        self._key = next((k for k in current if k not in MODIFIERS), "")
        self._name = tk.StringVar(value=current_name)

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="① 함께 누를 보조키를 고르고  ② 아래 칸을 클릭한 뒤 키 하나를 누르세요",
                  font=FONT).grid(row=0, column=0, columnspan=4, sticky="w")
        row = ttk.Frame(body); row.grid(row=1, column=0, columnspan=4, pady=(8, 4), sticky="w")
        for m in MODIFIERS:
            ttk.Checkbutton(row, text=key_display([m]), variable=self._mods[m], command=self._refresh).pack(side="left", padx=4)

        self._capture = tk.Label(body, text="여기를 클릭하고 키를 누르세요", font=FONT_BIG, relief="groove",
                                 width=26, height=2, bg="#f3f3f3", cursor="hand2")
        self._capture.grid(row=2, column=0, columnspan=4, pady=6, sticky="we")
        self._capture.bind("<Button-1>", lambda e: self._capture.focus_set())
        self._capture.bind("<FocusIn>", lambda e: self._capture.config(bg="#e3f2fd"))
        self._capture.bind("<FocusOut>", lambda e: self._capture.config(bg="#f3f3f3"))
        self._capture.bind("<KeyPress>", self._on_key)

        ttk.Label(body, text="키보드로 못 누르는 키:", font=FONT).grid(row=3, column=0, sticky="w", pady=(8, 0))
        self._special = ttk.Combobox(body, state="readonly", width=22, font=FONT,
                                     values=[f"{desc} ({name})" for name, desc in SPECIAL_KEYS])
        self._special.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(body, text="이 키 쓰기", command=self._use_special).grid(row=3, column=3, sticky="w", pady=(8, 0), padx=4)

        ttk.Label(body, text="표시 이름(선택):", font=FONT).grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(body, textvariable=self._name, width=30, font=FONT).grid(row=4, column=1, columnspan=3, sticky="w", pady=(10, 0))

        self._preview = ttk.Label(body, text="", font=FONT_B, foreground="#1565c0")
        self._preview.grid(row=5, column=0, columnspan=4, pady=(10, 0), sticky="w")

        btns = ttk.Frame(body); btns.grid(row=6, column=0, columnspan=4, pady=(12, 0), sticky="e")
        ttk.Button(btns, text="지우기(동작 없음)", command=self._clear).pack(side="left", padx=4)
        ttk.Button(btns, text="취소", command=self.destroy).pack(side="left", padx=4)
        ttk.Button(btns, text="확인", command=self._ok).pack(side="left", padx=4)
        self._refresh()
        self.grab_set()
        self.after(100, self._capture.focus_set)
        self.bind("<Escape>", lambda e: self.destroy())

    def _combo(self) -> list[str]:
        ks = [m for m in MODIFIERS if self._mods[m].get()]
        if self._key:
            ks.append(self._key)
        return ks

    def _refresh(self):
        ks = self._combo()
        self._capture.config(text=key_display(ks) if ks else "여기를 클릭하고 키를 누르세요")
        self._preview.config(text=("지정될 키: " + key_display(ks)) if ks else "")

    def _on_key(self, e):
        name = KEYSYM.get(e.keysym)
        if name is None:
            self._preview.config(text=f"이 키는 쓸 수 없습니다: {e.keysym}")
            return "break"
        if name in MODIFIERS:
            self._mods[name].set(True)
        else:
            self._key = name
        self._refresh()
        return "break"

    def _use_special(self):
        i = self._special.current()
        if i < 0:
            return
        self._key = SPECIAL_KEYS[i][0]
        self._refresh()

    def _clear(self):
        self.result = ([], "")
        self.destroy()

    def _ok(self):
        ks = self._combo()
        if not ks:
            messagebox.showinfo("키 바꾸기", "키를 하나 이상 지정하세요. 동작을 없애려면 '지우기' 를 누르세요.", parent=self)
            return
        self.result = (normalize_keys(ks), self._name.get().strip())
        self.destroy()


class App:
    def __init__(self, root: tk.Tk, camera: int, live: bool, model: str | None, gestures: str | None):
        self.root = root
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
        self._n_events_shown = 0
        self._want_live = live

        root.title("손 제스처 PC 제어 — 4팀")
        root.minsize(1120, 700)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure(".", font=FONT)
        style.configure("TNotebook.Tab", font=FONT_B, padding=(12, 6))
        self._build()
        self._start_worker()
        self.root.after(30, self._tick)

    # ── 화면 구성 ──
    def _build(self):
        r = self.root
        top = ttk.Frame(r, padding=(12, 8)); top.pack(fill="x")
        ttk.Label(top, text="손 제스처 PC 제어", font=FONT_TITLE).pack(side="left")
        self.mode_btn = tk.Button(top, text="연습 모드 (키 안 누름)", font=FONT_B, bg="#dfe6e9", width=30,
                                  command=self.toggle_mode, relief="raised")
        self.mode_btn.pack(side="right", padx=4)
        self.gate_lbl = tk.Label(top, text="게이트: 미연결(항상 열림)", font=FONT, fg="#666")
        self.gate_lbl.pack(side="right", padx=12)
        self.status_lbl = ttk.Label(top, text="모델 로딩 중...", font=FONT)
        self.status_lbl.pack(side="right", padx=12)

        main = ttk.Frame(r, padding=(12, 0, 12, 8)); main.pack(fill="both", expand=True)
        left = ttk.Frame(main); left.pack(side="left", fill="both", expand=False)
        right = ttk.Frame(main, padding=(14, 0, 0, 0)); right.pack(side="left", fill="both", expand=True)

        # 왼쪽: 영상
        self.video = tk.Label(left, bg="#111", width=640, height=480)
        self.video.pack()
        st = ttk.Frame(left, padding=(0, 6)); st.pack(fill="x")
        self.state_lbl = tk.Label(st, text="대기", font=FONT_B, fg="#555", width=14, anchor="w")
        self.state_lbl.pack(side="left")
        self.energy = tk.Canvas(st, width=220, height=14, bg="#eee", highlightthickness=0)
        self.energy.pack(side="left", padx=8)
        self.fps_lbl = ttk.Label(st, text="", font=FONT); self.fps_lbl.pack(side="right")
        self.last_lbl = tk.Label(left, text="—", font=FONT_BIG, fg="#555", anchor="w")
        self.last_lbl.pack(fill="x")
        self.msg_lbl = tk.Label(left, text="손바닥이 카메라를 향하게 하고 동작 사이에 잠깐 멈추세요.", font=FONT, fg="#555",
                                anchor="w", wraplength=640, justify="left")
        self.msg_lbl.pack(fill="x")
        self.probs = tk.Canvas(left, width=640, height=len(config.LABELS) * 20 + 4, bg="white", highlightthickness=0)
        self.probs.pack(fill="x", pady=(4, 0))

        # 오른쪽: 탭
        nb = ttk.Notebook(right); nb.pack(fill="both", expand=True)
        self.tab_map = ttk.Frame(nb, padding=12); nb.add(self.tab_map, text="제스처 → 키")
        self.tab_set = ttk.Frame(nb, padding=12); nb.add(self.tab_set, text="설정")
        self.tab_log = ttk.Frame(nb, padding=12); nb.add(self.tab_log, text="기록")
        self._build_map_tab()
        self._build_settings_tab()
        self._build_log_tab()

        bottom = ttk.Frame(r, padding=(12, 0, 12, 8)); bottom.pack(fill="x")
        ttk.Label(bottom, text="팁: 주먹은 쥔 뒤 0.3초 멈추고 내리기 · 스와이프는 자기 왼쪽으로 · 명령 사이 0.5초 이상",
                  foreground="#666").pack(side="left")

    def _build_map_tab(self):
        t = self.tab_map
        ttk.Label(t, text="제스처를 하면 아래 키를 누릅니다. '변경' 으로 원하는 키로 바꾸세요. 바꾸면 바로 저장됩니다.",
                  wraplength=420).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.map_rows: dict[str, dict] = {}
        for i, lab in enumerate(COMMAND_LABELS):
            name, desc = GESTURE_NAMES.get(lab, (lab, ""))
            fr = ttk.LabelFrame(t, text=f" {name} ", padding=(10, 6))
            fr.grid(row=1 + i, column=0, columnspan=4, sticky="we", pady=4)
            fr.columnconfigure(1, weight=1)
            sw = tk.Frame(fr, width=8, bg=COLORS.get(lab, "#999")); sw.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))
            ttk.Label(fr, text=desc, foreground="#666").grid(row=0, column=1, sticky="w")
            keys_lbl = tk.Label(fr, text="", font=FONT_BIG, fg="#1565c0", anchor="w")
            keys_lbl.grid(row=1, column=1, sticky="w")
            name_lbl = ttk.Label(fr, text="", foreground="#444")
            name_lbl.grid(row=2, column=1, sticky="w")
            ttk.Button(fr, text="변경", width=8, command=lambda l=lab: self.edit_key(l)).grid(row=0, column=2, rowspan=2, padx=4)
            ttk.Button(fr, text="끄기", width=6, command=lambda l=lab: self.apply_key(l, [], "")).grid(row=0, column=3, rowspan=2)
            self.map_rows[lab] = {"keys": keys_lbl, "name": name_lbl}

        pf = ttk.Frame(t); pf.grid(row=10, column=0, columnspan=4, sticky="we", pady=(14, 0))
        ttk.Label(pf, text="프리셋:").pack(side="left")
        self.preset = ttk.Combobox(pf, state="readonly", values=list(PRESETS), width=18)
        self.preset.current(0)
        self.preset.pack(side="left", padx=6)
        ttk.Button(pf, text="적용", command=self.apply_preset).pack(side="left")
        self.saved_lbl = ttk.Label(t, text="", foreground="#2e7d32")
        self.saved_lbl.grid(row=11, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(t, text=f"저장 위치: {config.ROOT / 'gestures.json'}", foreground="#888").grid(
            row=12, column=0, columnspan=4, sticky="w", pady=(2, 0))
        t.columnconfigure(0, weight=1)

    def _build_settings_tab(self):
        t = self.tab_set
        ttk.Label(t, text="확신도 문턱 (이보다 낮으면 실행 안 함)").grid(row=0, column=0, sticky="w")
        self.conf_var = tk.DoubleVar(value=0.8)
        self.conf_val = ttk.Label(t, text="0.80", width=5)
        self.conf_val.grid(row=0, column=2, sticky="w")
        sc = ttk.Scale(t, from_=0.5, to=1.0, variable=self.conf_var, command=self._on_conf, length=220)
        sc.grid(row=0, column=1, sticky="w", padx=8)
        sc.bind("<ButtonRelease-1>", lambda e: self._save_conf())

        self.mirror_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(t, text="화면 좌우반전 (거울처럼 보기, 판정에는 영향 없음)", variable=self.mirror_var).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(12, 0))

        cf = ttk.Frame(t); cf.grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Label(cf, text="카메라 번호:").pack(side="left")
        self.cam_var = tk.IntVar(value=self.camera_index)
        ttk.Spinbox(cf, from_=0, to=5, width=4, textvariable=self.cam_var).pack(side="left", padx=6)
        ttk.Button(cf, text="다시 연결", command=self.reconnect_camera).pack(side="left")

        ttk.Separator(t).grid(row=3, column=0, columnspan=3, sticky="we", pady=14)
        ttk.Label(t, text="YOLO 게이트 (정적 포즈로 ON/OFF)", font=FONT_B).grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Label(t, text="YOLO 팀 코드가 연결되면 여기 상태가 자동으로 바뀝니다. 지금은 연결 전이라 항상 열려 있습니다.",
                  wraplength=420, foreground="#666").grid(row=5, column=0, columnspan=3, sticky="w")
        self.gate_test_var = tk.BooleanVar(value=False)
        self.gate_test_chk = ttk.Checkbutton(t, text="게이트 수동 테스트: 닫기 (제스처를 해도 실행 안 됨)",
                                             variable=self.gate_test_var, command=self._on_gate_test)
        self.gate_test_chk.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Separator(t).grid(row=7, column=0, columnspan=3, sticky="we", pady=14)
        ttk.Button(t, text="인식 상태 초기화 (R)", command=self.reset).grid(row=8, column=0, sticky="w")
        self.log_path_lbl = ttk.Label(t, text="", foreground="#888", wraplength=420)
        self.log_path_lbl.grid(row=9, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _build_log_tab(self):
        t = self.tab_log
        ttk.Label(t, text="최근 판정 (시각 · 제스처 · 확신도 · 결과)").pack(anchor="w")
        fr = ttk.Frame(t); fr.pack(fill="both", expand=True, pady=(6, 0))
        self.log_list = tk.Listbox(fr, font=("Consolas", 10), activestyle="none")
        sb = ttk.Scrollbar(fr, command=self.log_list.yview)
        self.log_list.config(yscrollcommand=sb.set)
        self.log_list.pack(side="left", fill="both", expand=True); sb.pack(side="left", fill="y")
        self.count_lbl = ttk.Label(t, text="실행 0건"); self.count_lbl.pack(anchor="w", pady=(6, 0))

    # ── 워커 스레드: 카메라 + 파이프라인 ──
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
                self._msgq.put(("error", f"카메라 {self.camera_index} 를 열 수 없습니다. 다른 앱이 쓰고 있는지, 번호가 맞는지 확인하세요."))
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
        except Exception as e:  # 모델 파일 없음 등
            self._msgq.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            if self.cap is not None:
                self.cap.release(); self.cap = None

    # ── 주기적 화면 갱신 ──
    def _tick(self):
        while not self._msgq.empty():
            kind, payload = self._msgq.get()
            if kind == "ready":
                self._on_ready()
            elif kind == "camera":
                self.status_lbl.config(text=f"카메라 {payload} 연결됨")
            elif kind == "error":
                self.status_lbl.config(text="오류")
                messagebox.showerror("오류", payload)
        if self.pipeline is not None:
            self._render()
        self.root.after(30, self._tick)

    def _on_ready(self):
        p = self.pipeline
        self.conf_var.set(p.mapper.min_confidence)
        self.conf_val.config(text=f"{p.mapper.min_confidence:.2f}")
        self.log_path_lbl.config(text=f"세션 기록: {p.log.path}")
        self._refresh_map()
        self._refresh_mode()

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
            if self.mirror_var.get():
                vis = cv2.flip(vis, 1)
            if now < self._flash_until:
                cv2.rectangle(vis, (2, 2), (vis.shape[1] - 3, vis.shape[0] - 3), (0, 200, 0), 8)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.config(image=img); self.video.image = img

        if st.seg_state == "ACTIVE":
            self.state_lbl.config(text=f"동작 중 {st.active_sec:.1f}s", fg="#c62828")
        elif st.in_cooldown:
            self.state_lbl.config(text="쿨다운", fg="#f9a825")
        else:
            self.state_lbl.config(text="대기 (손 " + ("보임" if st.landmarks is not None else "없음") + ")", fg="#555")
        self.fps_lbl.config(text=f"{st.fps:.0f} fps")
        c = self.energy; c.delete("all")
        w = 220
        c.create_rectangle(0, 0, int(min(st.energy, 0.3) / 0.3 * w), 14, fill="#0aa2e6", width=0)
        for lvl, col in ((st.on_level, "#333"), (st.off_level, "#f9a825")):
            x = int(min(lvl, 0.3) / 0.3 * w)
            c.create_line(x, 0, x, 14, fill=col, width=2)

        ev = st.last_event
        if ev is not None and ev is not getattr(self, "_last_ev", None):
            self._last_ev = ev
            if ev.executed:
                self._flash_until = now + 0.5
            name = GESTURE_NAMES.get(ev.label, (ev.label, ""))[0] if ev.label != "no_gesture" else "동작 아님"
            if ev.label == "skipped":
                name = "건너뜀"
            self.last_lbl.config(text=f"{name}  {ev.confidence:.0%}" if ev.probs is not None else name,
                                 fg=COLORS.get(ev.label, "#555"))
            self.msg_lbl.config(text=self._friendly(ev))
            self._draw_probs(ev.probs)
        self.count_lbl.config(text=f"실행 {st.n_executed}건")
        if len(st.events) != self._n_events_shown:
            new = st.events[self._n_events_shown:] if len(st.events) >= self._n_events_shown else st.events
            for e in new:
                name = GESTURE_NAMES.get(e.label, (e.label, ""))[0]
                self.log_list.insert("end", f"{time.strftime('%H:%M:%S')}  {name:8s} {e.confidence:4.0%}  {self._friendly(e)}")
                self.log_list.see("end")
            self._n_events_shown = len(st.events)
        self._refresh_gate(st)

    def _friendly(self, ev) -> str:
        m = ev.message
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

    def _draw_probs(self, probs):
        c = self.probs; c.delete("all")
        if probs is None:
            return
        for i, (lab, pv) in enumerate(zip(config.LABELS, probs)):
            y = 2 + i * 20
            name = GESTURE_NAMES.get(lab, (lab, ""))[0] if lab != "no_gesture" else "동작 아님"
            c.create_text(4, y + 9, text=name, anchor="w", font=FONT, fill="#333")
            c.create_rectangle(120, y + 2, 120 + int(float(pv) * 400), y + 16, fill=COLORS.get(lab, "#999"), width=0)
            c.create_text(530, y + 9, text=f"{float(pv):.2f}", anchor="w", font=FONT, fill="#333")

    def _refresh_gate(self, st):
        p = self.pipeline
        if p.gate_connected:
            self.gate_lbl.config(text="게이트: " + ("열림 (YOLO)" if p.gate_open else "닫힘 (YOLO)"),
                                 fg="#2e7d32" if p.gate_open else "#c62828")
            self.gate_test_chk.state(["disabled"])
        elif not p.gate_open:
            self.gate_lbl.config(text="게이트: 닫힘 (수동 테스트)", fg="#c62828")
        else:
            self.gate_lbl.config(text="게이트: 미연결(항상 열림)", fg="#666")

    # ── 사용자 조작 ──
    def toggle_mode(self):
        p = self.pipeline
        if p is None:
            return
        if p.dry_run:
            if not messagebox.askyesno("실제 실행", "이제부터 제스처를 하면 실제로 키를 누릅니다.\n"
                                                    "다른 창을 앞에 두고 쓰세요. 계속할까요?"):
                return
        p.set_dry_run(not p.dry_run)
        self._refresh_mode()

    def _refresh_mode(self):
        p = self.pipeline
        if p.dry_run:
            self.mode_btn.config(text="연습 모드 (키 안 누름)  ▶ 실행 켜기", bg="#dfe6e9", fg="#000")
        else:
            self.mode_btn.config(text="● 실행 중 (키 누름)  ■ 연습으로", bg="#e53935", fg="#fff")

    def _refresh_map(self):
        p = self.pipeline
        for lab, row in self.map_rows.items():
            keys = p.mapper.keys_for(lab)
            a = p.mapper.actions.get(lab) or {}
            row["keys"].config(text=key_display(keys), fg="#1565c0" if keys else "#999")
            row["name"].config(text=a.get("label", "") if keys else "동작 없음")

    def edit_key(self, lab: str):
        p = self.pipeline
        if p is None:
            return
        a = p.mapper.actions.get(lab) or {}
        dlg = KeyCaptureDialog(self.root, lab, p.mapper.keys_for(lab), a.get("label", ""))
        self.root.wait_window(dlg)
        if dlg.result is None:
            return
        keys, name = dlg.result
        self.apply_key(lab, keys, name)

    def apply_key(self, lab: str, keys: list[str], name: str):
        p = self.pipeline
        if p is None:
            return
        try:
            p.mapper.set_action(lab, keys, name or None)
            p.mapper.save()
        except ValueError as e:
            messagebox.showerror("키 바꾸기", str(e))
            return
        p.log.write(f"MAP   {lab} -> {key_display(keys)}")
        self._refresh_map()
        self.saved_lbl.config(text=f"저장됨 {time.strftime('%H:%M:%S')}  ({GESTURE_NAMES.get(lab, (lab,))[0]} → {key_display(keys)})")

    def apply_preset(self):
        p = self.pipeline
        if p is None:
            return
        name = self.preset.get()
        for lab, (keys, disp) in PRESETS[name].items():
            p.mapper.set_action(lab, keys, disp)
        p.mapper.save()
        p.log.write(f"MAP   preset '{name}'")
        self._refresh_map()
        self.saved_lbl.config(text=f"저장됨 {time.strftime('%H:%M:%S')}  (프리셋 '{name}')")

    def _on_conf(self, _v):
        v = round(float(self.conf_var.get()) / 0.05) * 0.05
        self.conf_val.config(text=f"{v:.2f}")
        if self.pipeline is not None:
            self.pipeline.mapper.min_confidence = v

    def _save_conf(self):
        if self.pipeline is not None:
            self.pipeline.mapper.save()
            self.saved_lbl.config(text=f"저장됨 {time.strftime('%H:%M:%S')}  (확신도 {self.pipeline.mapper.min_confidence:.2f})")

    def _on_gate_test(self):
        if self.pipeline is not None:
            self.pipeline.set_gate(not self.gate_test_var.get(), source="manual")

    def reset(self):
        if self.pipeline is not None:
            self.pipeline.reset()
            self.last_lbl.config(text="—", fg="#555")
            self.msg_lbl.config(text="초기화됨")

    def reconnect_camera(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)
        self.camera_index = int(self.cam_var.get())
        self.status_lbl.config(text=f"카메라 {self.camera_index} 연결 중...")
        self._start_worker()

    def on_close(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3)
        if self.pipeline is not None:
            self.pipeline.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="시작부터 실제 키 입력")
    ap.add_argument("--model", default=None, help="기본 models/gru_gesture.pt")
    ap.add_argument("--gestures", default=None, help="기본 gestures.json")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args.camera, args.live, args.model, args.gestures)
    root.mainloop()


if __name__ == "__main__":
    main()
