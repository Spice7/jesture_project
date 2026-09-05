"""수집기(JIN) npz 로딩, 사람 단위 분할, 학습 배열 생성.

npz 형식 (programs/collect_gesture.py 가 저장):
  landmarks        (T,21,3) float32, 미검출 프레임은 NaN
  timestamps       (T,)     float64, 초 단위 (녹화 시작 기준)
  detected         (T,)     bool
  label            str      예: "swipe_left"
  participant_id   str      예: "p001"
  handedness, sample_id, duration, detection_rate, world_landmarks, ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, preprocess

REQUIRED_KEYS = ("landmarks", "timestamps", "label", "participant_id")


class FormatError(ValueError):
    """수집기 npz 형식이 예상과 다를 때. 수집기가 바뀌었는지 먼저 확인할 것."""


@dataclass
class Sample:
    path: Path
    person: str
    label: str
    landmarks: np.ndarray        # (T,21,3) NaN 포함 원본
    timestamps_ms: np.ndarray    # (T,) int64 밀리초
    meta: dict


def load_sample(path: Path) -> Sample:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            raise FormatError(f"{path.name}: 키 {missing} 없음. 수집기 저장 형식이 바뀌었는지 확인")
        lm = np.asarray(z["landmarks"], dtype=np.float32)
        ts = np.asarray(z["timestamps"], dtype=np.float64)
        label = str(z["label"])
        person = str(z["participant_id"]).strip().lower()
        if "detected" in z.files:
            det = np.asarray(z["detected"], dtype=bool)
            lm = lm.copy()
            lm[~det] = np.nan                 # 수집기도 NaN 으로 저장하지만 마스크를 우선한다
        meta = {k: z[k].item() if z[k].ndim == 0 else None for k in z.files
                if k not in ("landmarks", "timestamps", "detected", "world_landmarks",
                             "handedness_per_frame", "handedness_confidence", "mirrored")}
    if lm.ndim != 3 or lm.shape[1:] != (config.N_LANDMARKS, 3):
        raise FormatError(f"{path.name}: landmarks 형상 {lm.shape}, 기대 (T,21,3)")
    if ts.shape != (len(lm),):
        raise FormatError(f"{path.name}: timestamps 길이 {ts.shape} ≠ 프레임 수 {len(lm)}")
    if ts.max() > 60:                          # 초 단위여야 한다. ms 로 저장됐다면 60 을 넘는다
        raise FormatError(f"{path.name}: timestamps 최대값 {ts.max():.0f} — 초 단위가 아닌 것 같음")
    folder_label = path.parent.name
    if folder_label != label:
        raise FormatError(f"{path.name}: 파일 내부 라벨 {label!r} ≠ 폴더 {folder_label!r}")
    if not re.match(config.PARTICIPANT_PATTERN, person):
        raise FormatError(f"{path.name}: participant_id {person!r} 가 규약(p001~p006)과 다름")
    ts_ms = np.round((ts - ts[0]) * 1000).astype(np.int64)
    return Sample(path, person, label, lm, ts_ms, meta)


def load_dataset(root: Path = config.DATASET_DIR, labels=config.LABELS, strict: bool = False) -> list[Sample]:
    """dataset/<label>/*.npz 전부 로딩. 형식 오류 파일은 경고 후 건너뜀 (strict=True 면 예외)."""
    root = Path(root)
    samples, bad = [], []
    for p in sorted(root.rglob("*.npz")):
        if p.parent.name not in labels:
            continue
        try:
            samples.append(load_sample(p))
        except FormatError as e:
            if strict:
                raise
            bad.append(str(e))
    for msg in bad[:10]:
        print(f"경고(형식): {msg}")
    if len(bad) > 10:
        print(f"... 형식 오류 {len(bad)}개")
    return samples


def summarize(samples: list[Sample]) -> str:
    persons = sorted({s.person for s in samples})
    lines = [f"{'person':10s} " + " ".join(f"{l[:11]:>11s}" for l in config.LABELS) + "   total"]
    for p in persons:
        cnt = {l: sum(1 for s in samples if s.person == p and s.label == l) for l in config.LABELS}
        lines.append(f"{p:10s} " + " ".join(f"{cnt[l]:11d}" for l in config.LABELS) + f"   {sum(cnt.values())}")
    tot = {l: sum(1 for s in samples if s.label == l) for l in config.LABELS}
    lines.append(f"{'total':10s} " + " ".join(f"{tot[l]:11d}" for l in config.LABELS) + f"   {len(samples)}")
    return "\n".join(lines)


def split_by_person(samples: list[Sample], val_persons=None, test_persons=None, seed=config.RANDOM_SEED):
    """사람 단위 분할. 명시하지 않으면 사람 수에 따라 자동:
       4명 이상: 마지막 1명 test, 그 앞 1명 val / 2~3명: 마지막 1명 test, val 은 train 15% 랜덤
       1명: 랜덤 70/15/15 (성능 과대평가 경고)"""
    persons = sorted({s.person for s in samples})
    rng = np.random.default_rng(seed)
    if test_persons is None and val_persons is None:
        if len(persons) >= 4:
            test_persons, val_persons = [persons[-1]], [persons[-2]]
        elif len(persons) >= 2:
            test_persons, val_persons = [persons[-1]], []
        else:
            print("경고: 참가자가 1명뿐이라 랜덤 분할합니다. 실제 성능은 과대평가됩니다.")
            idx = rng.permutation(len(samples))
            n = len(samples)
            a, b = int(n * 0.7), int(n * 0.85)
            pick = lambda ids: [samples[i] for i in ids]
            return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])
    test_persons, val_persons = set(test_persons or []), set(val_persons or [])
    test = [s for s in samples if s.person in test_persons]
    val = [s for s in samples if s.person in val_persons]
    train = [s for s in samples if s.person not in test_persons | val_persons]
    if not val:
        idx = rng.permutation(len(train))
        k = max(1, int(len(train) * 0.15))
        val = [train[i] for i in idx[:k]]
        train = [train[i] for i in idx[k:]]
    return train, val, test


def build_arrays(samples: list[Sample], augment_times: int = 0, seed=config.RANDOM_SEED):
    """→ X (N, SEQ_LEN, FEATURE_DIM), y (N,) int, dropped(list[path])
    좌우반전 증강은 하지 않는다: swipe_left 를 뒤집으면 존재하지 않는 swipe_right 가 된다."""
    rng = np.random.default_rng(seed)
    X, y, dropped = [], [], []
    for s in samples:
        seq = preprocess.interpolate_missing(s.landmarks)
        if seq is None:
            dropped.append(s.path)
            continue
        seq = preprocess.resample(seq, s.timestamps_ms)
        variants = [seq] + [preprocess.augment(seq, rng) for _ in range(augment_times)]
        for v in variants:
            X.append(preprocess.to_features(preprocess.normalize(v)))
            y.append(config.LABEL_TO_IDX[s.label])
    if not X:
        return (np.zeros((0, config.SEQ_LEN, config.FEATURE_DIM), np.float32),
                np.zeros((0,), np.int64), dropped)
    return np.stack(X), np.array(y, dtype=np.int64), dropped
