"""시계열 분류기(GRU / LSTM) 정의·저장·추론.

두 아키텍처는 셀 종류만 다르고 층 수·유닛·드롭아웃은 동일하다 (공정 비교를 위해).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

ARCHS = ("lstm", "gru")
DEFAULT_ARCH = "gru"          # 우리 담당은 GRU. LSTM 은 다른 팀원이 맡되 비교용으로 남겨둔다


def model_path(arch: str = DEFAULT_ARCH) -> Path:
    return config.MODELS_DIR / f"{arch}_gesture.keras"


def meta_path(arch: str = DEFAULT_ARCH) -> Path:
    return config.MODELS_DIR / f"{arch}_gesture.json"


# 기존 코드 호환용 (LSTM 기본 경로)
LSTM_MODEL_PATH = model_path("lstm")
LSTM_META_PATH = meta_path("lstm")


def build_model(arch: str = DEFAULT_ARCH, n_classes: int = len(config.LABELS),
                units: int = 64, dropout: float = 0.3):
    """arch: 'lstm' | 'gru'. 구조는 동일: RNN(units, seq) → Drop → RNN(units) → Drop → Dense32 → softmax"""
    from tensorflow import keras
    from tensorflow.keras import layers

    arch = arch.lower()
    if arch not in ARCHS:
        raise ValueError(f"arch는 {ARCHS} 중 하나여야 합니다: {arch!r}")
    Cell = layers.LSTM if arch == "lstm" else layers.GRU

    inp = keras.Input(shape=(config.SEQ_LEN, config.FEATURE_DIM))
    x = Cell(units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = Cell(units)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name=f"gesture_{arch}")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_lstm(**kw):
    return build_model("lstm", **kw)


def build_gru(**kw):
    return build_model("gru", **kw)


def save_meta(path: Path, arch: str = DEFAULT_ARCH, **extra):
    meta = dict(arch=arch, labels=config.LABELS, seq_len=config.SEQ_LEN, feature_dim=config.FEATURE_DIM,
                use_z=config.USE_Z, mirror=config.MIRROR, **extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


class GestureClassifier:
    """실시간 루프에서 쓰는 추론 래퍼. features: (SEQ_LEN, FEATURE_DIM)

    GestureClassifier()            → models/gru_gesture.keras
    GestureClassifier(arch="lstm") → models/lstm_gesture.keras
    """

    def __init__(self, arch: str = DEFAULT_ARCH, model_file: Path | None = None, meta_file: Path | None = None):
        from tensorflow import keras
        model_file = Path(model_file) if model_file else model_path(arch)
        meta_file = Path(meta_file) if meta_file else model_file.with_suffix(".json")
        self.model = keras.models.load_model(model_file)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.arch = meta.get("arch", arch)
        self.labels = meta["labels"]
        assert meta["seq_len"] == config.SEQ_LEN and meta["feature_dim"] == config.FEATURE_DIM, \
            "모델 규약(seq_len/feature_dim)이 현재 config와 다릅니다. 다시 학습하세요."

    def predict(self, features: np.ndarray):
        probs = self.model.predict(features[None], verbose=0)[0]
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i]), probs
