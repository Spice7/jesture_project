"""baseline: 손으로 짠 룰 분류기. GRU 가 이걸 확실히 이겨야 딥러닝을 쓴 이유가 된다.

규칙 (정규화된 좌표, 단위 = 손바닥 크기):
  1. 손바닥 중심이 MIN_MOVE 이상 이동하면 swipe_left (현재 스와이프 클래스는 하나)
  2. 아니고 손가락 끝 5개 ↔ 손목 평균 거리가 시작 대비 FIST_RATIO 이하로 줄면 make_fist
  3. 둘 다 아니면 no_gesture
이동을 먼저 보는 이유: 스와이프 끝에서 손이 옆으로 기울면 2D 에서 손가락이 짧아 보여 주먹으로 오판한다.
make_fist 는 손목이 거의 움직이지 않는다는 수집 규약을 이용한다.
"""
import numpy as np

from . import config, preprocess

FINGERTIPS = [4, 8, 12, 16, 20]
FIST_RATIO = 0.7        # 끝-손목 거리가 30% 이상 줄면 주먹
MIN_MOVE = 1.0          # 손바닥 크기 1배 이상 이동하면 스와이프
EDGE = 5                # 시작/끝 판단에 쓰는 프레임 수


def _tip_dist(seq: np.ndarray) -> np.ndarray:
    """(n,21,3) → (n,) 프레임별 손가락끝-손목 평균 거리 (xy 평면)"""
    d = seq[:, FINGERTIPS, :2] - seq[:, [preprocess.WRIST], :2]
    return np.linalg.norm(d, axis=2).mean(axis=1)


def rule_classify(seq: np.ndarray) -> str:
    """seq: 정규화된 (n,21,3)"""
    palm = seq[:, preprocess.PALM_IDX, :2].mean(axis=1)
    move = np.linalg.norm(palm[-EDGE:].mean(axis=0) - palm[:EDGE].mean(axis=0))
    if move >= MIN_MOVE:
        return "swipe_left"
    tips = _tip_dist(seq)
    start, end = tips[:EDGE].mean(), tips[-EDGE:].mean()
    if start > 1e-6 and end / start < FIST_RATIO:
        return "make_fist"
    return "no_gesture"


def predict_features(X: np.ndarray) -> np.ndarray:
    """X: (N, SEQ_LEN, FEATURE_DIM) → 라벨 인덱스 (N,)"""
    dims = 3 if config.USE_Z else 2
    out = []
    for x in X:
        seq = x.reshape(config.SEQ_LEN, config.N_LANDMARKS, dims)
        if dims == 2:
            seq = np.concatenate([seq, np.zeros_like(seq[:, :, :1])], axis=2)
        out.append(config.LABEL_TO_IDX[rule_classify(seq)])
    return np.array(out)
