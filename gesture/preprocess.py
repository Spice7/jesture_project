"""녹화 원본(가변 길이, 결측 포함) → LSTM 입력 (SEQ_LEN, FEATURE_DIM) 변환.

규약:
  1. 결측(NaN) 프레임은 시간축 선형 보간, 양 끝은 가장 가까운 값으로 채움
  2. 검출 비율 < MIN_DETECTION_RATIO 또는 연속 결측 > MAX_GAP_FRAMES → 폐기(None)
  3. 타임스탬프 기준으로 SEQ_LEN 프레임으로 리샘플 (fps가 달라도 동일 길이)
  4. 정규화: 시퀀스 전체의 손목 평균 위치를 원점으로, 평균 손바닥 크기로 스케일
     → 프레임별 손목 기준 정규화를 하면 "이동" 정보가 사라지므로 절대 그렇게 하지 말 것
"""
from __future__ import annotations

import numpy as np

from . import config

WRIST, MIDDLE_MCP = 0, 9
PALM_IDX = [0, 5, 9, 13, 17]


def detection_mask(seq: np.ndarray) -> np.ndarray:
    return ~np.isnan(seq[:, 0, 0])


def longest_gap(mask: np.ndarray) -> int:
    best = cur = 0
    for ok in mask:
        cur = 0 if ok else cur + 1
        best = max(best, cur)
    return best


def interpolate_missing(seq: np.ndarray, min_ratio: float | None = None,
                        max_gap: int | None = None) -> np.ndarray | None:
    """(T,21,3) with NaN rows → 보간된 (T,21,3). 품질 기준 미달이면 None.

    기본 기준은 수집기 규약(검출률 ≥0.8, 연속 미검출 ≤5) = 학습 데이터 기준.
    실시간 경로는 빠른 동작에서 추적이 0.2~0.3초 끊기는 일이 흔해 더 느슨한 값을 넘긴다 (09-06 실측)."""
    min_ratio = config.MIN_DETECTION_RATIO if min_ratio is None else min_ratio
    max_gap = config.MAX_GAP_FRAMES if max_gap is None else max_gap
    mask = detection_mask(seq)
    if mask.sum() < 2 or mask.mean() < min_ratio:
        return None
    if longest_gap(mask) > max_gap:
        return None
    t = np.arange(len(seq))
    out = seq.astype(np.float32).copy()
    flat = out.reshape(len(seq), -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, t[mask], flat[mask, j])   # 양끝은 np.interp가 최근접값으로 채움
    return out


def resample(seq: np.ndarray, timestamps_ms: np.ndarray, n: int = config.SEQ_LEN) -> np.ndarray:
    """시간축 기준 선형 보간으로 n프레임으로 맞춘다. seq: (T,21,3), timestamps_ms: (T,)"""
    ts = np.asarray(timestamps_ms, dtype=np.float64)
    if len(seq) == 1 or ts[-1] <= ts[0]:
        return np.repeat(seq[:1], n, axis=0).astype(np.float32)
    target = np.linspace(ts[0], ts[-1], n)
    flat = seq.reshape(len(seq), -1)
    out = np.stack([np.interp(target, ts, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape(n, seq.shape[1], seq.shape[2]).astype(np.float32)


def normalize(seq: np.ndarray) -> np.ndarray:
    """(n,21,3) → 시퀀스 단위 정규화. 이동 정보는 보존된다."""
    center = seq[:, WRIST, :].mean(axis=0)                            # (3,)
    palm = np.linalg.norm(seq[:, MIDDLE_MCP, :2] - seq[:, WRIST, :2], axis=1).mean()
    scale = max(float(palm), 1e-3)
    out = (seq - center[None, None, :]) / scale
    return out.astype(np.float32)


def to_features(seq: np.ndarray) -> np.ndarray:
    """(n,21,3) 정규화된 시퀀스 → (n, FEATURE_DIM)"""
    if not config.USE_Z:
        seq = seq[:, :, :2]
    return seq.reshape(len(seq), -1).astype(np.float32)


def sample_to_features(landmarks: np.ndarray, timestamps_ms: np.ndarray,
                       min_ratio: float | None = None, max_gap: int | None = None) -> np.ndarray | None:
    """녹화 원본 → (SEQ_LEN, FEATURE_DIM). 품질 미달이면 None. min_ratio/max_gap 은 interpolate_missing 참고."""
    seq = interpolate_missing(landmarks, min_ratio=min_ratio, max_gap=max_gap)
    if seq is None:
        return None
    seq = resample(seq, timestamps_ms)
    seq = normalize(seq)
    return to_features(seq)


# ── 증강 (학습 시에만, 정규화 전 (n,21,3) 에 적용). 좌우반전은 하지 않는다 (swipe_right 없음) ──
def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """정규화 전 (n,21,3) 에 적용: 스케일·이동·시간 왜곡·노이즈. 좌우반전은 하지 않는다."""
    out = seq.astype(np.float32).copy()
    s = rng.uniform(0.85, 1.15)
    c = out[:, WRIST, :2].mean(axis=0)
    out[:, :, :2] = (out[:, :, :2] - c) * s + c + rng.normal(0, 0.03, size=2).astype(np.float32)
    # 시간 왜곡: 구간별 속도를 바꾼 뒤 원래 길이로 다시 샘플링
    n = len(out)
    warp = np.cumsum(rng.uniform(0.7, 1.3, size=n))
    warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (n - 1)
    flat = out.reshape(n, -1)
    flat = np.stack([np.interp(np.arange(n), warp, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    out = flat.reshape(n, config.N_LANDMARKS, 3)
    out += rng.normal(0, 0.004, size=out.shape)
    return out.astype(np.float32)
