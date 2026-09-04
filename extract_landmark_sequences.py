#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_landmark_sequences.py
=============================
검증(check_jester_mediapipe.py)을 통과한 설정으로
Jester 프레임 → MediaPipe 랜드마크 → LSTM 학습용 시계열 텐서를 만든다.

핵심 설계
---------
* 특징 구성 (프레임당 기본 66차원)
    - 21개 랜드마크의 손목 기준 상대좌표 × 스케일 정규화 : 63
    - 손목 절대좌표의 '윈도우 첫 프레임 대비 이동량' (dx, dy) :  2
      → 이게 없으면 Swiping Left/Right가 구분되지 않는다. 손 모양은 같고
        움직이는 방향만 다르기 때문에 상대좌표만으로는 정보가 소실된다.
    - 손 크기(카메라 거리 대리 변수) : 1
    - (옵션) 왼손/오른손 플래그 : +1
* z좌표는 MediaPipe에서 신뢰도가 낮으므로 --no-z 로 끌 수 있게 했다.
  63차원(xyz) vs 42차원(xy) 비교 실험을 반드시 해볼 것.
* 결측 프레임은 선형 보간, 앞뒤 끝은 edge fill.
* 시퀀스 길이는 시간축 선형 보간으로 --seq-len 에 통일한다.
  (Jester 12fps / 웹캠 30fps 의 프레임 수 차이를 여기서 흡수)
* 학습과 추론이 반드시 같은 전처리를 쓰도록 feature_config.json 을 함께 저장한다.

사용 예
-------
python extract_landmark_sequences.py ^
    --frames-root D:/data/20bn-jester-v1 ^
    --labels-csv  D:/data/jester-v1-train.csv ^
    --classes "Swiping Left,Swiping Right,Swiping Down,Swiping Up,Stop Sign,No gesture" ^
    --n-per-class 400 --upscale 2 --det-conf 0.3 --mode video ^
    --seq-len 32 --out-dir ./dataset ^
    --mirror-pairs "Swiping Left:Swiping Right,Swiping Up:Swiping Up"

출력
----
dataset/X_train.npy (N,T,F), y_train.npy, X_val.npy, ... , X_test.npy
dataset/label_map.json, dataset/feature_config.json, dataset/meta.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import check_jester_mediapipe as C  # 백엔드/로딩 유틸 재사용

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


WRIST, MIDDLE_MCP = 0, 9


# --------------------------------------------------------------------------
# 특징 추출
# --------------------------------------------------------------------------

def pick_hand(landmarks: np.ndarray, handedness: list[str],
              prefer: str) -> tuple[np.ndarray, str]:
    """여러 손이 잡히면 하나를 고른다. 기본은 화면상 가장 큰 손."""
    if len(landmarks) == 1:
        return landmarks[0], (handedness[0] if handedness else "?")
    if prefer in ("Left", "Right") and prefer in handedness:
        i = handedness.index(prefer)
        return landmarks[i], prefer
    sizes = [np.ptp(h[:, 0]) * np.ptp(h[:, 1]) for h in landmarks]
    i = int(np.argmax(sizes))
    return landmarks[i], (handedness[i] if i < len(handedness) else "?")


def frame_features(hand: np.ndarray, handed: str, use_z: bool,
                   use_handedness: bool) -> np.ndarray:
    """한 프레임의 손 랜드마크 → 특징 벡터."""
    dims = 3 if use_z else 2
    pts = hand[:, :dims].astype(np.float32)

    wrist = pts[WRIST].copy()
    rel = pts - wrist                                    # 손목 기준 상대좌표

    # 손 크기: 손목→중지 MCP 거리. 0에 가까우면 전체 최대거리로 대체.
    scale = float(np.linalg.norm(rel[MIDDLE_MCP, :2]))
    if scale < 1e-6:
        scale = float(np.max(np.linalg.norm(rel[:, :2], axis=1))) or 1e-6
    rel = rel / scale                                    # 거리 불변 정규화

    feats = [rel.reshape(-1), wrist[:2], np.array([scale], np.float32)]
    if use_handedness:
        feats.append(np.array([1.0 if handed == "Right" else 0.0], np.float32))
    return np.concatenate(feats).astype(np.float32)


def feature_dim(use_z: bool, use_handedness: bool) -> int:
    return 21 * (3 if use_z else 2) + 2 + 1 + (1 if use_handedness else 0)


def fill_gaps(feats: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """검출 실패 프레임을 선형 보간으로 채운다. 전부 실패면 None."""
    if not mask.any():
        return None
    idx = np.arange(len(mask))
    good = idx[mask]
    out = feats.copy()
    for c in range(feats.shape[1]):
        out[:, c] = np.interp(idx, good, feats[good, c])  # 양 끝은 edge 값 유지
    return out


def resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    """시간축 선형 보간으로 길이를 통일."""
    t_src = np.linspace(0.0, 1.0, len(seq))
    t_dst = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(t_dst, t_src, seq[:, c])
                     for c in range(seq.shape[1])], axis=1).astype(np.float32)


def anchor_motion(seq: np.ndarray, use_z: bool) -> np.ndarray:
    """손목 절대좌표를 '첫 프레임 대비 이동량'으로 바꾼다(위치 불변, 방향 보존)."""
    d = 21 * (3 if use_z else 2)
    seq = seq.copy()
    seq[:, d:d + 2] -= seq[0, d:d + 2]
    return seq


def mirror(seq: np.ndarray, use_z: bool) -> np.ndarray:
    """좌우 반전 증강. x 성분의 부호를 뒤집는다(라벨 스왑과 함께 써야 함)."""
    dims = 3 if use_z else 2
    d = 21 * dims
    out = seq.copy()
    out[:, 0:d:dims] *= -1.0        # 상대좌표 x
    out[:, d] *= -1.0               # 손목 이동량 dx
    if out.shape[1] == feature_dim(use_z, True):
        out[:, -1] = 1.0 - out[:, -1]   # 왼손/오른손 플래그 반전
    return out


# --------------------------------------------------------------------------
# 파이프라인
# --------------------------------------------------------------------------

def extract_video(frames, backend, scale: float, method: str, fps_ms: int,
                  use_z: bool, use_handedness: bool, prefer: str):
    backend.reset()
    F = feature_dim(use_z, use_handedness)
    feats = np.zeros((len(frames), F), np.float32)
    mask = np.zeros(len(frames), bool)

    for i, fp in enumerate(frames):
        img = C.cv2.imread(str(fp))
        if img is None:
            continue
        img = C.preprocess(img, scale, method)
        res = backend.detect(img, timestamp_ms=i * fps_ms)
        if not res.detected:
            continue
        hand, handed = pick_hand(res.landmarks, res.handedness, prefer)
        feats[i] = frame_features(hand, handed, use_z, use_handedness)
        mask[i] = True
    return feats, mask


def stratified_split(labels: list[str], ratios=(0.7, 0.15, 0.15), seed=42):
    rng = np.random.default_rng(seed)
    idx_by_label: dict[str, list[int]] = {}
    for i, l in enumerate(labels):
        idx_by_label.setdefault(l, []).append(i)
    tr, va, te = [], [], []
    for l, idx in idx_by_label.items():
        idx = np.array(idx)
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        tr += idx[:n_tr].tolist()
        va += idx[n_tr:n_tr + n_va].tolist()
        te += idx[n_tr + n_va:].tolist()
    return sorted(tr), sorted(va), sorted(te)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Jester → MediaPipe 랜드마크 시계열 데이터셋 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--frames-root", required=True, type=Path)
    ap.add_argument("--labels-csv", type=Path, default=None)
    ap.add_argument("--classes", type=str, default="")
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--frame-glob", type=str, default="*.jpg")
    ap.add_argument("--max-frames", type=int, default=0)

    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--presence-conf", type=float, default=0.3)
    ap.add_argument("--track-conf", type=float, default=0.3)
    ap.add_argument("--mode", type=str, default="video", choices=["video", "image"])
    ap.add_argument("--enhance", type=str, default="none",
                    choices=["none", "unsharp", "clahe"])
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--prefer-hand", type=str, default="auto",
                    choices=["auto", "Left", "Right"])

    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--min-det-rate", type=float, default=0.6,
                    help="이 비율 미만으로 검출된 영상은 버린다")
    ap.add_argument("--max-gap", type=int, default=6,
                    help="연속 미검출이 이보다 길면 버린다")
    ap.add_argument("--no-z", action="store_true", help="z좌표 제외(42차원 실험)")
    ap.add_argument("--handedness-feature", action="store_true")
    ap.add_argument("--mirror-pairs", type=str, default="",
                    help='좌우반전 증강 라벨 매핑. 예: "Swiping Left:Swiping Right"')

    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--model", type=Path, default=C.DEFAULT_MODEL_PATH)
    ap.add_argument("--out-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    use_z = not args.no_z
    use_handed = args.handedness_feature
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_path = C.ensure_model(args.model)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
    labels_map = C.load_labels(args.labels_csv)
    picked = C.sample_videos(args.frames_root, labels_map, classes,
                             args.n_per_class, args.seed)
    print(f"[데이터] 대상 영상 {len(picked)}개")

    backend = C.make_backend(model_path, args.mode, args.det_conf,
                             args.presence_conf, args.track_conf, args.num_hands)
    fps_ms = int(round(1000 / args.fps))

    seqs, ys, meta = [], [], []
    dropped = {"no_frames": 0, "low_det": 0, "long_gap": 0, "empty": 0}

    for vid, label, path in tqdm(picked, ncols=80, desc="extract"):
        frames = C.read_frames(path, args.frame_glob, args.max_frames)
        if not frames:
            dropped["no_frames"] += 1
            continue
        feats, mask = extract_video(frames, backend, args.upscale, args.enhance,
                                    fps_ms, use_z, use_handed,
                                    args.prefer_hand)
        det_rate = float(mask.mean())
        gap = C.longest_gap(mask.tolist())
        if det_rate < args.min_det_rate:
            dropped["low_det"] += 1
            continue
        if gap > args.max_gap:
            dropped["long_gap"] += 1
            continue
        filled = fill_gaps(feats, mask)
        if filled is None:
            dropped["empty"] += 1
            continue
        seq = anchor_motion(resample(filled, args.seq_len), use_z)
        seqs.append(seq)
        ys.append(label)
        meta.append(dict(video_id=vid, label=label, n_frames=len(frames),
                         det_rate=round(det_rate, 3), longest_gap=gap))

    backend.close()

    if not seqs:
        sys.exit("[결과] 사용 가능한 시퀀스가 0개입니다. "
                 "먼저 check_jester_mediapipe.py 로 설정을 조정하세요.")

    # ---- 먼저 분할한다. 증강은 train에만 적용해야 원본/미러가
    #      서로 다른 split에 흩어지는 데이터 누수를 막을 수 있다. ----
    tr, va, te = stratified_split(ys, seed=args.seed)

    pairs = {}
    for item in args.mirror_pairs.split(","):
        if ":" in item:
            a, b = item.split(":", 1)
            pairs[a.strip()] = b.strip()
            pairs.setdefault(b.strip(), a.strip())

    aug_seqs, aug_ys = [], []
    if pairs:
        for i in tr:
            if ys[i] in pairs:
                aug_seqs.append(mirror(seqs[i], use_z))
                aug_ys.append(pairs[ys[i]])
                m = dict(meta[i]); m["video_id"] += "_mirror"
                m["label"] = pairs[ys[i]]; m["split"] = "train(aug)"
                meta.append(m)
        print(f"[증강] train에만 좌우반전 {len(aug_seqs)}개 추가")

    for name, idx in (("train", tr), ("val", va), ("test", te)):
        for i in idx:
            meta[i].setdefault("split", name)

    X = np.stack(seqs).astype(np.float32)
    label_names = sorted(set(ys) | set(aug_ys))
    label_map = {name: i for i, name in enumerate(label_names)}
    y = np.array([label_map[v] for v in ys], np.int64)

    for name, idx in (("train", tr), ("val", va), ("test", te)):
        Xs, ys_ = X[idx], y[idx]
        if name == "train" and aug_seqs:
            Xs = np.concatenate([Xs, np.stack(aug_seqs).astype(np.float32)])
            ys_ = np.concatenate([ys_, np.array([label_map[v] for v in aug_ys],
                                                np.int64)])
        np.save(args.out_dir / f"X_{name}.npy", Xs)
        np.save(args.out_dir / f"y_{name}.npy", ys_)
        if name == "train":
            n_train = len(ys_)

    with open(args.out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    cfg = dict(seq_len=args.seq_len, feature_dim=X.shape[2], use_z=use_z,
               use_handedness=use_handed, upscale=args.upscale,
               det_conf=args.det_conf, presence_conf=args.presence_conf,
               track_conf=args.track_conf, mode=args.mode,
               enhance=args.enhance, num_hands=args.num_hands,
               prefer_hand=args.prefer_hand, source_fps=args.fps,
               normalization="wrist-relative + hand-scale; wrist motion anchored to frame0",
               feature_layout=(f"[0:{21*(3 if use_z else 2)}] relative landmarks, "
                               f"[{21*(3 if use_z else 2)}:{21*(3 if use_z else 2)+2}] wrist dx,dy, "
                               f"[-{2 if use_handed else 1}] hand scale"
                               + (", [-1] handedness" if use_handed else "")))
    with open(args.out_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    C._write_csv(args.out_dir / "meta.csv", meta)

    print(f"\n[결과] 시퀀스 shape = (N, {X.shape[1]}, {X.shape[2]})  (T, F)")
    print(f"[결과] 분할 train/val/test = {n_train}/{len(va)}/{len(te)}"
          f"  (train은 증강 포함)")
    print("[결과] 원본 클래스별 개수:")
    for name in label_names:
        print(f"       {name:<24} {int((y == label_map[name]).sum())}")
    print(f"[제외] {dropped}")
    print(f"[저장] {args.out_dir.resolve()}")
    print("\n실시간 추론 시에는 feature_config.json 의 값을 그대로 읽어서 "
          "동일한 전처리를 적용해야 합니다.")


if __name__ == "__main__":
    main()
