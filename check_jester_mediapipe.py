#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_jester_mediapipe.py
=========================
20BN-Jester 프레임에 MediaPipe Hand Landmarker를 적용했을 때
"손 랜드마크가 실제로 잡히는가"를 정량 측정하는 사전 검증 스크립트.

이 스크립트가 답해야 하는 질문
------------------------------
1. Jester의 저해상도(높이 100px) 프레임에서 손 검출률이 몇 %인가?
2. 업스케일 배율 / 검출 신뢰도 임계값 / IMAGE vs VIDEO(트래킹) 모드를
   바꾸면 검출률이 얼마나 개선되는가?
3. LSTM 학습에 쓸 만한 시퀀스(연속 결측 구간이 짧은 시퀀스)가 몇 % 확보되는가?
4. 결론: Jester를 그대로 쓸 것인가, 보정해서 쓸 것인가, 자체 녹화로 갈 것인가?

사용 예
-------
python check_jester_mediapipe.py ^
    --frames-root  D:/data/20bn-jester-v1 ^
    --labels-csv   D:/data/jester-v1-train.csv ^
    --classes      "Swiping Left,Swiping Right,Swiping Down,Swiping Up,Stop Sign,No gesture" ^
    --n-per-class  10 ^
    --upscale      1 2 3 ^
    --det-conf     0.3 0.5 ^
    --mode         video image ^
    --out-dir      ./mp_check

의존성
------
pip install mediapipe opencv-python numpy pandas tqdm
(mediapipe 0.10.10 이상 권장. 구버전 solutions API도 자동 폴백 지원)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("opencv가 필요합니다:  pip install opencv-python")

try:
    from tqdm import tqdm
except ImportError:  # tqdm 없으면 그냥 통과
    def tqdm(x, **kw):  # type: ignore
        return x


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")

# MediaPipe 21 랜드마크 연결 구조 (시각화용)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# --------------------------------------------------------------------------
# 1. MediaPipe 백엔드 어댑터
#    - 신버전(Tasks API)과 구버전(solutions.hands) 양쪽을 같은 인터페이스로 감싼다.
# --------------------------------------------------------------------------

@dataclass
class HandResult:
    """한 프레임 결과. detected=False면 landmarks는 None."""
    detected: bool
    landmarks: Optional[np.ndarray]   # (n_hands, 21, 3) 정규화 좌표(0~1, z는 상대깊이)
    handedness: list[str]
    n_hands: int


def ensure_model(model_path: Path) -> Path:
    """hand_landmarker.task 모델이 없으면 내려받는다."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[모델] {model_path} 없음 → 다운로드 시도")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[모델] 다운로드 완료 ({model_path.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        sys.exit(
            f"[모델] 자동 다운로드 실패: {e}\n"
            f"      아래 URL을 브라우저로 받아 {model_path} 위치에 두고 다시 실행하세요.\n"
            f"      {MODEL_URL}"
        )
    return model_path


class TasksBackend:
    """mediapipe.tasks (0.10.x 이상) 기반 백엔드."""

    name = "tasks"

    def __init__(self, model_path: Path, mode: str, det_conf: float,
                 presence_conf: float, track_conf: float, num_hands: int):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp_python = mp_python
        self._vision = vision
        self._model_path = str(model_path)
        self._mode = mode
        self._cfg = dict(
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf,
        )
        self._landmarker = None
        self.reset()

    def reset(self):
        """영상 하나가 끝나면 트래킹 상태를 버리고 새로 만든다."""
        self.close()
        vision = self._vision
        running_mode = (vision.RunningMode.VIDEO if self._mode == "video"
                        else vision.RunningMode.IMAGE)
        base = self._mp_python.BaseOptions(model_asset_path=self._model_path)
        opts = vision.HandLandmarkerOptions(base_options=base,
                                            running_mode=running_mode,
                                            **self._cfg)
        self._landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> HandResult:
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._mode == "video":
            res = self._landmarker.detect_for_video(image, timestamp_ms)
        else:
            res = self._landmarker.detect(image)
        return _pack(res.hand_landmarks, res.handedness)

    def close(self):
        if getattr(self, "_landmarker", None) is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


class SolutionsBackend:
    """구버전 mediapipe.solutions.hands 기반 백엔드(폴백)."""

    name = "solutions"

    def __init__(self, model_path: Path, mode: str, det_conf: float,
                 presence_conf: float, track_conf: float, num_hands: int):
        import mediapipe as mp
        self._mp = mp
        self._kw = dict(
            static_image_mode=(mode == "image"),
            max_num_hands=num_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            model_complexity=1,
        )
        self._hands = None
        self.reset()

    def reset(self):
        self.close()
        self._hands = self._mp.solutions.hands.Hands(**self._kw)

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> HandResult:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return HandResult(False, None, [], 0)
        lms = np.array([[[p.x, p.y, p.z] for p in h.landmark]
                        for h in res.multi_hand_landmarks], dtype=np.float32)
        handed = ([c.classification[0].label for c in res.multi_handedness]
                  if res.multi_handedness else ["?"] * len(lms))
        return HandResult(True, lms, handed, len(lms))

    def close(self):
        if getattr(self, "_hands", None) is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None


def _pack(hand_landmarks, handedness) -> HandResult:
    if not hand_landmarks:
        return HandResult(False, None, [], 0)
    lms = np.array([[[p.x, p.y, p.z] for p in hand] for hand in hand_landmarks],
                   dtype=np.float32)
    labels = [h[0].category_name for h in handedness] if handedness else ["?"] * len(lms)
    return HandResult(True, lms, labels, len(lms))


def make_backend(model_path: Path, mode: str, det_conf: float,
                 presence_conf: float, track_conf: float, num_hands: int):
    """설치된 mediapipe에 맞는 백엔드를 고른다."""
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        # 구버전 API가 살아있으면 그쪽이 저해상도에서 더 관대한 경우가 많다.
        try:
            return SolutionsBackend(model_path, mode, det_conf, presence_conf,
                                    track_conf, num_hands)
        except Exception as e:
            print(f"[백엔드] solutions 초기화 실패({e}) → tasks로 전환")
    return TasksBackend(model_path, mode, det_conf, presence_conf,
                        track_conf, num_hands)


# --------------------------------------------------------------------------
# 2. 데이터 로딩
# --------------------------------------------------------------------------

def load_labels(labels_csv: Optional[Path]) -> dict[str, str]:
    """jester-v1-train.csv (video_id;label) → {video_id: label}"""
    if labels_csv is None:
        return {}
    mapping: dict[str, str] = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        sep = ";" if sample.count(";") >= sample.count(",") else ","
        for row in csv.reader(f, delimiter=sep):
            if len(row) >= 2 and row[0].strip():
                mapping[row[0].strip()] = row[1].strip()
    return mapping


def sample_videos(frames_root: Path, labels: dict[str, str],
                  classes: Optional[list[str]], n_per_class: int,
                  seed: int) -> list[tuple[str, str, Path]]:
    """(video_id, label, dir) 목록을 클래스별로 균등 샘플링."""
    rng = np.random.default_rng(seed)
    available = {p.name: p for p in frames_root.iterdir() if p.is_dir()}
    if not available:
        sys.exit(f"[데이터] {frames_root} 아래에 프레임 폴더가 없습니다.")

    buckets: dict[str, list[str]] = {}
    for vid, path in available.items():
        label = labels.get(vid, "unknown")
        if classes and label not in classes:
            continue
        buckets.setdefault(label, []).append(vid)

    if not buckets:
        sys.exit("[데이터] --classes 조건에 맞는 영상이 없습니다. 클래스명을 확인하세요.")

    picked: list[tuple[str, str, Path]] = []
    for label, vids in sorted(buckets.items()):
        vids = sorted(vids)
        idx = rng.permutation(len(vids))[:n_per_class]
        picked += [(vids[i], label, available[vids[i]]) for i in idx]
    return picked


def read_frames(video_dir: Path, glob_pat: str, max_frames: int) -> list[Path]:
    frames = sorted(video_dir.glob(glob_pat))
    if max_frames > 0 and len(frames) > max_frames:
        # 균등 간격으로 솎아냄 (앞뒤 잘라내지 않음)
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


# --------------------------------------------------------------------------
# 3. 전처리 옵션
# --------------------------------------------------------------------------

def enhance(img: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return img
    if method == "unsharp":
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        return cv2.addWeighted(img, 1.6, blur, -0.6, 0)
    if method == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    raise ValueError(method)


def preprocess(img: np.ndarray, scale: float, method: str) -> np.ndarray:
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return enhance(img, method)


# --------------------------------------------------------------------------
# 4. 측정 로직
# --------------------------------------------------------------------------

def longest_gap(flags: Sequence[bool]) -> int:
    """가장 긴 연속 미검출 구간 길이."""
    best = cur = 0
    for f in flags:
        cur = 0 if f else cur + 1
        best = max(best, cur)
    return best


@dataclass
class VideoStat:
    config: str
    label: str
    video_id: str
    n_frames: int
    n_detected: int
    det_rate: float
    longest_gap: int
    mean_hands: float
    usable: bool


def run_config(videos, backend_factory, scale: float, method: str,
               usable_rate: float, max_gap: int, config_name: str,
               frame_rows: list, sample_dir: Optional[Path],
               fps_ms: int) -> tuple[list[VideoStat], float]:
    stats: list[VideoStat] = []
    backend = backend_factory()
    t_total, n_total = 0.0, 0
    saved_labels: set[str] = set()

    for video_id, label, frames in tqdm(videos, desc=config_name, ncols=80):
        backend.reset()
        flags, hand_counts = [], []
        for i, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                flags.append(False)
                hand_counts.append(0)
                continue
            img = preprocess(img, scale, method)
            t0 = time.perf_counter()
            res = backend.detect(img, timestamp_ms=i * fps_ms)
            t_total += time.perf_counter() - t0
            n_total += 1
            flags.append(res.detected)
            hand_counts.append(res.n_hands)
            frame_rows.append(dict(config=config_name, label=label,
                                   video_id=video_id, frame=i,
                                   detected=int(res.detected),
                                   n_hands=res.n_hands))
            # 클래스별 대표 프레임 1장 저장(눈으로 확인용)
            if (sample_dir is not None and res.detected
                    and label not in saved_labels and i > len(frames) // 2):
                _save_overlay(sample_dir / f"{config_name}__{_slug(label)}__{video_id}.jpg",
                              img, res)
                saved_labels.add(label)

        n = len(flags)
        det = sum(flags)
        rate = det / n if n else 0.0
        gap = longest_gap(flags)
        stats.append(VideoStat(
            config=config_name, label=label, video_id=video_id,
            n_frames=n, n_detected=det, det_rate=rate, longest_gap=gap,
            mean_hands=float(np.mean(hand_counts)) if hand_counts else 0.0,
            usable=bool(rate >= usable_rate and gap <= max_gap),
        ))

    backend.close()
    fps = n_total / t_total if t_total > 0 else float("nan")
    return stats, fps


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:30]


def _save_overlay(path: Path, img: np.ndarray, res: HandResult):
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = img.copy()
    h, w = vis.shape[:2]
    for hand in res.landmarks:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(vis, pts[a], pts[b], (0, 255, 0), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(vis, p, 2, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


# --------------------------------------------------------------------------
# 5. 리포트
# --------------------------------------------------------------------------

def summarize(all_stats: list[VideoStat], fps_map: dict[str, float]) -> list[dict]:
    rows = []
    by_cfg: dict[str, list[VideoStat]] = {}
    for s in all_stats:
        by_cfg.setdefault(s.config, []).append(s)
    for cfg, group in by_cfg.items():
        frames = sum(g.n_frames for g in group)
        det = sum(g.n_detected for g in group)
        rows.append(dict(
            config=cfg,
            n_videos=len(group),
            frame_det_rate=det / frames if frames else 0.0,
            video_usable_rate=float(np.mean([g.usable for g in group])),
            median_video_det_rate=float(np.median([g.det_rate for g in group])),
            median_longest_gap=float(np.median([g.longest_gap for g in group])),
            proc_fps=fps_map.get(cfg, float("nan")),
        ))
    rows.sort(key=lambda r: (-r["video_usable_rate"], -r["frame_det_rate"]))
    return rows


def print_table(rows: list[dict], title: str, cols: list[tuple[str, str, int]]):
    print(f"\n{title}")
    header = "".join(name.ljust(w) for name, _, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = ""
        for _, key, w in cols:
            v = r[key]
            line += (f"{v:.3f}" if isinstance(v, float) else str(v)).ljust(w)
        print(line)


def verdict(best: dict, usable_rate: float, max_gap: int) -> str:
    u = best["video_usable_rate"]
    if u >= 0.80:
        return ("판정: GO — Jester를 학습 데이터로 그대로 사용 가능합니다.\n"
                f"       최적 설정 [{best['config']}] 기준 사용 가능 영상 {u:.1%}.\n"
                "       다음 단계: extract_landmark_sequences.py 로 좌표 시퀀스 추출.")
    if u >= 0.50:
        return ("판정: 조건부 GO — 절반 이상은 쓸 수 있으나 손실이 큽니다.\n"
                f"       최적 설정 [{best['config']}] 기준 사용 가능 영상 {u:.1%}.\n"
                "       권장: (a) 검출률이 낮은 클래스를 제외하고 클래스 수를 줄일 것,\n"
                "             (b) 결측 프레임 보간을 반드시 적용할 것,\n"
                "             (c) 실사용 도메인(웹캠 30fps, 근거리)과의 갭을 메우기 위해\n"
                "                 팀 자체 녹화 데이터를 소량 섞어 파인튜닝할 것.")
    return ("판정: NO-GO — Jester 경로는 위험합니다.\n"
            f"       최적 설정에서도 사용 가능 영상이 {u:.1%}에 불과합니다.\n"
            "       권장: 팀원 6명이 클래스당 30~50회씩 직접 녹화하는 방식으로 전환.\n"
            "             6클래스 × 40회 × 6명 ≈ 1,440 시퀀스면 LSTM 학습에 충분하고,\n"
            "             실사용 카메라와 도메인이 동일해 오히려 실시간 성능이 좋습니다.")


# --------------------------------------------------------------------------
# 6. main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Jester × MediaPipe 손 검출률 사전 검증",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--frames-root", required=True, type=Path,
                    help="video_id 폴더들이 들어있는 루트 (예: 20bn-jester-v1)")
    ap.add_argument("--labels-csv", type=Path, default=None,
                    help="jester-v1-train.csv (video_id;label)")
    ap.add_argument("--classes", type=str, default="",
                    help="쉼표로 구분한 대상 클래스. 비우면 전체")
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=40,
                    help="영상당 최대 프레임 수(0이면 전체)")
    ap.add_argument("--frame-glob", type=str, default="*.jpg")

    ap.add_argument("--upscale", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--det-conf", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--mode", type=str, nargs="+", default=["video"],
                    choices=["video", "image"],
                    help="video=프레임 간 트래킹 사용, image=매 프레임 독립 검출")
    ap.add_argument("--enhance", type=str, nargs="+", default=["none"],
                    choices=["none", "unsharp", "clahe"])
    ap.add_argument("--presence-conf", type=float, default=0.3)
    ap.add_argument("--track-conf", type=float, default=0.3)
    ap.add_argument("--num-hands", type=int, default=2)

    ap.add_argument("--usable-rate", type=float, default=0.70,
                    help="영상이 '사용 가능'으로 인정되는 최소 프레임 검출률")
    ap.add_argument("--max-gap", type=int, default=5,
                    help="허용하는 최대 연속 미검출 프레임 수")
    ap.add_argument("--fps", type=float, default=12.0,
                    help="원본 데이터 fps (Jester는 12)")

    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--out-dir", type=Path, default=Path("mp_check"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-samples", action="store_true",
                    help="확인용 오버레이 이미지 저장 안 함")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = None if args.no_samples else args.out_dir / "samples"

    model_path = ensure_model(args.model)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None

    labels = load_labels(args.labels_csv)
    picked = sample_videos(args.frames_root, labels, classes,
                           args.n_per_class, args.seed)

    videos = []
    for vid, label, path in picked:
        frames = read_frames(path, args.frame_glob, args.max_frames)
        if frames:
            videos.append((vid, label, frames))
    if not videos:
        sys.exit("[데이터] 프레임 파일을 찾지 못했습니다. --frame-glob 확인.")

    n_cls = len({v[1] for v in videos})
    print(f"[데이터] 영상 {len(videos)}개 / 클래스 {n_cls}개 / "
          f"프레임 총 {sum(len(v[2]) for v in videos)}장")

    grid = list(itertools.product(args.upscale, args.det_conf,
                                  args.mode, args.enhance))
    print(f"[설정] 조합 {len(grid)}개 실행")

    frame_rows: list[dict] = []
    all_stats: list[VideoStat] = []
    fps_map: dict[str, float] = {}

    for scale, conf, mode, method in grid:
        cfg = f"x{scale:g}_conf{conf:g}_{mode}_{method}"

        def factory(conf=conf, mode=mode):
            return make_backend(model_path, mode, conf, args.presence_conf,
                                args.track_conf, args.num_hands)

        stats, fps = run_config(
            videos, factory, scale, method, args.usable_rate, args.max_gap,
            cfg, frame_rows, sample_dir, fps_ms=int(round(1000 / args.fps)))
        all_stats += stats
        fps_map[cfg] = fps

    # ---- 저장 ----
    _write_csv(args.out_dir / "frame_level.csv", frame_rows)
    _write_csv(args.out_dir / "video_level.csv", [asdict(s) for s in all_stats])

    summary = summarize(all_stats, fps_map)
    _write_csv(args.out_dir / "summary.csv", summary)

    # 클래스별(최적 설정 기준)
    best = summary[0]
    per_class = {}
    for s in all_stats:
        if s.config != best["config"]:
            continue
        d = per_class.setdefault(s.label, dict(label=s.label, n=0, det=0,
                                               frames=0, usable=0))
        d["n"] += 1
        d["det"] += s.n_detected
        d["frames"] += s.n_frames
        d["usable"] += int(s.usable)
    class_rows = [dict(label=k,
                       n_videos=v["n"],
                       frame_det_rate=v["det"] / v["frames"] if v["frames"] else 0.0,
                       usable_rate=v["usable"] / v["n"] if v["n"] else 0.0)
                  for k, v in sorted(per_class.items())]
    class_rows.sort(key=lambda r: -r["usable_rate"])
    _write_csv(args.out_dir / "per_class_best_config.csv", class_rows)

    # ---- 출력 ----
    print_table(summary, "■ 설정별 요약 (사용 가능 영상 비율 내림차순)", [
        ("config", "config", 26), ("videos", "n_videos", 8),
        ("frame_det", "frame_det_rate", 11), ("usable", "video_usable_rate", 9),
        ("med_det", "median_video_det_rate", 9),
        ("med_gap", "median_longest_gap", 9), ("proc_fps", "proc_fps", 9),
    ])
    print_table(class_rows, f"■ 클래스별 결과 (설정: {best['config']})", [
        ("label", "label", 24), ("videos", "n_videos", 8),
        ("frame_det", "frame_det_rate", 11), ("usable", "usable_rate", 9),
    ])

    print("\n" + "=" * 72)
    print(verdict(best, args.usable_rate, args.max_gap))
    print("=" * 72)
    if class_rows and class_rows[-1]["usable_rate"] < 0.5:
        weak = [r["label"] for r in class_rows if r["usable_rate"] < 0.5]
        print(f"\n주의: 아래 클래스는 검출률이 낮아 제외 검토가 필요합니다 → {weak}")
    print(f"\n결과 파일: {args.out_dir.resolve()}")
    if sample_dir and sample_dir.exists():
        print(f"오버레이 확인용 이미지: {sample_dir.resolve()}")

    with open(args.out_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f,
                  ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
