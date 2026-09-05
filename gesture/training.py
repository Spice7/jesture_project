"""학습·평가 공용 로직. train_lstm.py 와 compare_models.py 가 함께 쓴다."""
from __future__ import annotations

import time

import numpy as np

from . import baseline, config
from .model import build_model


def fit(arch: str, Xtr, ytr, Xva, yva, *, seed: int = config.RANDOM_SEED, epochs: int = 80,
        batch: int = 32, init_weights: str | None = None, verbose: int = 0):
    """모델 생성 → 학습. (model, history) 반환. EarlyStopping으로 val_loss 최적 가중치 복원."""
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    model = build_model(arch)
    if init_weights:
        model.load_weights(init_weights)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    t0 = time.perf_counter()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch,
                     callbacks=cbs, verbose=verbose)
    hist.train_sec = time.perf_counter() - t0
    return model, hist


def metrics(y_true, y_pred) -> dict:
    """정확도, macro-F1, no_gesture 재현율(거부 성능), no_gesture 정밀도."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    labels = list(range(len(config.LABELS)))
    ng = config.LABEL_TO_IDX["no_gesture"]
    return dict(
        acc=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        ng_recall=recall_score(y_true, y_pred, labels=[ng], average="macro", zero_division=0),
        ng_precision=precision_score(y_true, y_pred, labels=[ng], average="macro", zero_division=0),
    )


def predict(model, X) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0,), np.int64)
    return model.predict(X, verbose=0).argmax(1)


def infer_ms_per_sample(model, X, n: int = 50) -> float:
    """실시간 루프에서 쓰이는 배치=1 추론 시간(ms). 처음 몇 번은 워밍업으로 버린다."""
    if len(X) == 0:
        return float("nan")
    xs = X[: min(n, len(X))]
    for x in xs[:3]:
        model.predict(x[None], verbose=0)
    t0 = time.perf_counter()
    for x in xs:
        model.predict(x[None], verbose=0)
    return (time.perf_counter() - t0) / len(xs) * 1000


def baseline_metrics(X, y) -> dict:
    if len(X) == 0:
        return {}
    return metrics(y, baseline.predict_features(X))


def report(name, y_true, y_pred):
    """classification_report + 혼동행렬 출력."""
    from sklearn.metrics import classification_report, confusion_matrix
    labels = list(range(len(config.LABELS)))
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=labels, target_names=config.LABELS, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion (row=true, col=pred):")
    print(f"{'':12s}" + "".join(f"{l[:7]:>8s}" for l in config.LABELS))
    for l, row in zip(config.LABELS, cm):
        print(f"{l:12s}" + "".join(f"{v:8d}" for v in row))
