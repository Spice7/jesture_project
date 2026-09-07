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


REVERSE_SETS = {"none": (), "fist": ("make_fist",), "both": ("make_fist", "swipe_left"),
                "all": ("make_fist", "swipe_left", "finger_snap")}   # all: 스냅 되감기(중지가 엄지로 돌아감)도 no_gesture


def reversed_negatives(samples: list[Sample], labels=("make_fist",), fraction: float = 0.5,
                       seed=config.RANDOM_SEED) -> list[Sample]:
    """명령 클립을 시간 역재생해 no_gesture 학습 샘플로 만든다 (학습 세트에만 쓸 것).

    make_fist 되감기 = "쥔 손 → 편 손"(펴기), swipe_left 되감기 = "오른쪽으로 되돌리기"(복귀).
    둘 다 실제로 자주 일어나지만 녹화 데이터에 거의 없어(4명 합쳐 펴기 6개, 복귀 0개) 모델이 명령으로 오인하던 동작.
    09-06 23시 도입. fraction 으로 비율을 제한해 no_gesture 가 과대 클래스가 되지 않게 한다."""
    rng = np.random.default_rng(seed)
    out = []
    for lab in labels:
        pool = [s for s in samples if s.label == lab]
        k = int(round(len(pool) * fraction))
        for i in rng.permutation(len(pool))[:k]:
            s = pool[i]
            lm = s.landmarks[::-1].copy()
            ts = (s.timestamps_ms[-1] - s.timestamps_ms[::-1]).astype(s.timestamps_ms.dtype)
            meta = dict(s.meta) if isinstance(s.meta, dict) else {}
            meta["reversed_from"] = lab
            out.append(Sample(s.path.with_name(s.path.stem + f"_rev.npz"), s.person, "no_gesture", lm, ts, meta))
    return out


LOWER_LABELS = ("make_fist", "finger_snap")


def lowering_negatives(samples: list[Sample], labels=LOWER_LABELS, fraction: float = 0.5,
                       fps: float = 30.0, seed=config.RANDOM_SEED) -> list[Sample]:
    """명령이 끝난 손 모양(주먹, 튕긴 손) 그대로 손을 내리는 동작을 합성해 no_gesture 학습 샘플로 만든다 (학습 세트에만).

    09-07 웹캠 실측: 스냅·주먹 실행 뒤 손을 내리는 구간을 모델이 같은 명령으로 1.00 확신 → 상식검사가 아슬아슬하게 막음.
    학습 데이터에 "명령 자세로 내리기"가 없어서다. 클립 끝 자세를 0.2~0.4초 유지한 뒤 0.4~0.7초 동안 아래(±40°)로
    손바닥 3~5개만큼 내리며, 손가락은 시작 자세 쪽으로 0~30% 풀어준다(실제로 내리면서 손이 느슨해지는 것을 흉내)."""
    rng = np.random.default_rng(seed)
    out = []
    for lab in labels:
        pool = [s for s in samples if s.label == lab]
        k = int(round(len(pool) * fraction))
        for i in rng.permutation(len(pool))[:k]:
            s = pool[i]
            det = ~np.isnan(s.landmarks[:, 0, 0])
            if det.sum() < 2:
                continue
            first, last = s.landmarks[det][0], s.landmarks[det][-1]
            palm = float(np.linalg.norm(last[9, :2] - last[0, :2]))
            n_hold = int(fps * rng.uniform(0.2, 0.4))
            n_move = int(fps * rng.uniform(0.4, 0.7))
            total = palm * rng.uniform(3.0, 5.0)                      # 총 이동량 (손바닥 단위)
            ang = np.deg2rad(90 + rng.uniform(-40, 40))               # 아래 방향 ±40°
            step = np.array([np.cos(ang), np.sin(ang)], np.float32) * (total / max(n_move, 1))
            relax = rng.uniform(0.0, 0.3)                             # 손가락 풀림 비율
            frames = [last + rng.normal(0, 0.0005, last.shape).astype(np.float32) for _ in range(n_hold)]
            cur = last.copy()
            for j in range(n_move):
                cur = cur.copy()
                cur[:, :2] += step
                t = (j + 1) / n_move
                shape = last * (1 - relax * t) + first * (relax * t)   # 손 모양만 서서히 풀림
                cur = shape + (cur[0] - shape[0])                      # 손목 위치는 이동 경로 유지
                frames.append(cur + rng.normal(0, 0.0005, cur.shape).astype(np.float32))
            lm = np.stack(frames).astype(np.float32)
            ts = (np.arange(len(lm)) * 1000.0 / fps).astype(np.int64)
            meta = dict(s.meta) if isinstance(s.meta, dict) else {}
            meta["lowered_from"] = lab
            out.append(Sample(s.path.with_name(s.path.stem + "_low.npz"), s.person, "no_gesture", lm, ts, meta))
    return out


def shortened_swipes(samples: list[Sample], fraction: float = 0.5, scale=(0.4, 0.7),
                     seed=config.RANDOM_SEED) -> list[Sample]:
    """스와이프 클립의 손목 이동 폭만 scale 배로 줄인 '짧은 스와이프'를 만든다 (라벨 유지, 학습 세트에만).

    09-07 11:48 실측: 수집 데이터의 스와이프는 순이동 최소 1.47(손바닥) 이라 0.5~0.6 짜리 짧은 스와이프는 모델이
    본 적이 없어 no_gesture 로 빠졌다. 손 모양(손목 기준 상대 좌표)은 그대로 두고 손목 궤적만 시작점 기준으로 축소한다."""
    rng = np.random.default_rng(seed)
    out = []
    pool = [s for s in samples if s.label == "swipe_left"]
    k = int(round(len(pool) * fraction))
    for i in rng.permutation(len(pool))[:k]:
        s = pool[i]
        lm = s.landmarks.copy()
        det = ~np.isnan(lm[:, 0, 0])
        if det.sum() < 2:
            continue
        f = float(rng.uniform(*scale))
        w0 = lm[det][0, 0, :2]
        wrist = lm[:, 0:1, :2]                         # (T,1,2)
        shape = lm[:, :, :2] - wrist                   # 손목 기준 상대 좌표 (손 모양)
        new_wrist = w0 + (wrist - w0) * f              # 궤적 축소
        lm[:, :, :2] = shape + new_wrist
        meta = dict(s.meta) if isinstance(s.meta, dict) else {}
        meta["shortened"] = f
        out.append(Sample(s.path.with_name(s.path.stem + "_short.npz"), s.person, s.label, lm, s.timestamps_ms, meta))
    return out


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
