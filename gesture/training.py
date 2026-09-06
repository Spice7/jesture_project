"""학습·평가 공용 로직. train_model.py 와 compare_models.py 가 함께 쓴다. PyTorch.

Keras 시절과 같은 설정: Adam 1e-3, 배치 32, 최대 80 epoch,
EarlyStopping(val_loss, patience 12, 최적 가중치 복원), ReduceLROnPlateau(factor 0.5, patience 5, min_lr 1e-5).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import baseline, config
from .model import DEVICE, build_model


@dataclass
class History:
    """Keras History 흉내: .history['loss'|'val_loss'|'accuracy'|'val_accuracy'], .train_sec"""
    history: dict = field(default_factory=lambda: {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []})
    train_sec: float = 0.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(X, y=None):
    xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    if y is None:
        return xt
    return xt, torch.as_tensor(np.asarray(y, dtype=np.int64))


@torch.no_grad()
def _evaluate(model, X, y, loss_fn, batch=256):
    model.eval()
    xt, yt = _to_tensor(X, y)
    tot_loss, correct = 0.0, 0
    for i in range(0, len(xt), batch):
        xb, yb = xt[i:i + batch].to(DEVICE), yt[i:i + batch].to(DEVICE)
        logits = model(xb)
        tot_loss += loss_fn(logits, yb).item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    n = max(len(xt), 1)
    return tot_loss / n, correct / n


def fit(arch: str, Xtr, ytr, Xva, yva, *, seed: int = config.RANDOM_SEED, epochs: int = 80,
        batch: int = 32, init_weights: str | None = None, verbose: int = 0,
        lr: float = 1e-3, patience: int = 12, lr_patience: int = 5, min_lr: float = 1e-5):
    """모델 생성 → 학습. (model, history) 반환. val_loss 최적 가중치 복원."""
    set_seed(seed)
    model = build_model(arch)
    if init_weights:
        model.load_weights(init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=lr_patience, min_lr=min_lr)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = _to_tensor(Xtr, ytr)
    g = torch.Generator().manual_seed(seed)
    hist = History()
    best_loss, best_state, bad = float("inf"), None, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        tot, correct = 0.0, 0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        tr_loss, tr_acc = tot / len(xt), correct / len(xt)
        va_loss, va_acc = _evaluate(model, Xva, yva, loss_fn)
        for k, v in zip(("loss", "accuracy", "val_loss", "val_accuracy"), (tr_loss, tr_acc, va_loss, va_acc)):
            hist.history[k].append(v)
        sched.step(va_loss)
        if verbose:
            print(f"Epoch {ep + 1}/{epochs} - loss: {tr_loss:.4f} - acc: {tr_acc:.4f} - val_loss: {va_loss:.4f} "
                  f"- val_acc: {va_acc:.4f} - lr: {opt.param_groups[0]['lr']:.1e}")
        if va_loss < best_loss - 1e-6:
            best_loss, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
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
    return model.predict_proba(X).argmax(1)


def infer_ms_per_sample(model, X, n: int = 50) -> float:
    """실시간 루프에서 쓰이는 배치=1 추론 시간(ms). 처음 몇 번은 워밍업으로 버린다."""
    if len(X) == 0:
        return float("nan")
    xs = X[: min(n, len(X))]
    for x in xs[:3]:
        model.predict_proba(x[None])
    t0 = time.perf_counter()
    for x in xs:
        model.predict_proba(x[None])
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
