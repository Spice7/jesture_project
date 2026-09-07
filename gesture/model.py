"""시계열 분류기(GRU / LSTM) 정의·저장·추론. PyTorch.

두 아키텍처는 셀 종류만 다르고 층 수·유닛·드롭아웃은 동일하다 (공정 비교를 위해).
구조: RNN(units, 시퀀스 출력) → Dropout → RNN(units, 마지막 출력) → Dropout → Linear 32 (ReLU) → Linear n_classes
파라미터 수는 Keras 판과 같다 (GRU 51,907).

09-06: TensorFlow/Keras → PyTorch 로 이식. 팀(YOLO, 팀원 LSTM)이 PyTorch 라 프레임워크를 통일했다.
저장 형식 .keras → .pt (state_dict + 구조 정보). 정보 파일(.json)은 그대로.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config

ARCHS = ("lstm", "gru")
DEFAULT_ARCH = "gru"          # 우리 담당은 GRU. LSTM 은 다른 팀원이 맡되 비교용으로 남겨둔다
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_path(arch: str = DEFAULT_ARCH) -> Path:
    return config.MODELS_DIR / f"{arch}_gesture.pt"


def meta_path(arch: str = DEFAULT_ARCH) -> Path:
    return config.MODELS_DIR / f"{arch}_gesture.json"


# 기존 코드 호환용 (LSTM 기본 경로)
LSTM_MODEL_PATH = model_path("lstm")
LSTM_META_PATH = meta_path("lstm")


class GestureRNN(nn.Module):
    def __init__(self, arch: str = DEFAULT_ARCH, n_classes: int = len(config.LABELS),
                 units: int = 64, dropout: float = 0.3, feature_dim: int = config.FEATURE_DIM):
        super().__init__()
        arch = arch.lower()
        if arch not in ARCHS:
            raise ValueError(f"arch는 {ARCHS} 중 하나여야 합니다: {arch!r}")
        Cell = nn.LSTM if arch == "lstm" else nn.GRU
        self.arch, self.units, self.n_classes, self.dropout_p, self.feature_dim = arch, units, n_classes, dropout, feature_dim
        self.rnn1 = Cell(feature_dim, units, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.rnn2 = Cell(units, units, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(units, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self._init_like_keras()

    def _init_like_keras(self):
        """Keras 기본 초기화와 맞춤: RNN 입력 가중치 Glorot(xavier) uniform, 순환 가중치 직교(orthogonal), 편향 0,
        Linear 는 Glorot uniform + 편향 0. (PyTorch 기본은 전부 U(-1/√h, 1/√h) 라 RNN 일반화가 달라짐.
        09-06 이식 직후 4인 실험에서 평균 4%p 낮게 나와 맞춤.)"""
        for rnn in (self.rnn1, self.rnn2):
            for name, p in rnn.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(p)
                elif name.startswith("bias"):
                    nn.init.zeros_(p)
        for lin in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, T, F) → logits (B, C)
        y, _ = self.rnn1(x)
        y = self.drop1(y)
        y, _ = self.rnn2(y)
        h = self.drop2(y[:, -1])
        h = torch.relu(self.fc1(h))
        return self.fc2(h)

    # ── Keras 시절 호출부 호환 ─────────────────────────────────────────
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self, line_length: int = 80):
        print(f'Model: "gesture_{self.arch}"  (PyTorch)')
        print("-" * line_length)
        for name, mod in self.named_children():
            n = sum(p.numel() for p in mod.parameters())
            print(f"{name:10s} {mod.__class__.__name__:10s} params {n:>8,d}")
        print("-" * line_length)
        print(f"Trainable params: {self.count_params():,d}")

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch: int = 256) -> np.ndarray:
        """X: (N, T, F) 또는 (T, F) → 확률 (N, C)"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        self.eval()
        dev = next(self.parameters()).device
        out = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(dev)
            out.append(torch.softmax(self(xb), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.n_classes), np.float32)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": self.arch, "units": self.units, "n_classes": self.n_classes,
                    "dropout": self.dropout_p, "feature_dim": self.feature_dim,
                    "state_dict": self.state_dict()}, path)

    def load_weights(self, path: Path):
        ck = torch.load(Path(path), map_location="cpu")
        self.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
        return self


def build_model(arch: str = DEFAULT_ARCH, n_classes: int = len(config.LABELS),
                units: int = 64, dropout: float = 0.3) -> GestureRNN:
    return GestureRNN(arch, n_classes, units, dropout).to(DEVICE)


def load_model(path: Path) -> GestureRNN:
    ck = torch.load(Path(path), map_location="cpu")
    m = GestureRNN(ck["arch"], ck["n_classes"], ck["units"], ck.get("dropout", 0.3), ck.get("feature_dim", config.FEATURE_DIM))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def build_lstm(**kw):
    return build_model("lstm", **kw)


def build_gru(**kw):
    return build_model("gru", **kw)


def save_meta(path: Path, arch: str = DEFAULT_ARCH, **extra):
    meta = dict(arch=arch, framework="pytorch", labels=config.LABELS, seq_len=config.SEQ_LEN,
                feature_dim=config.FEATURE_DIM, use_z=config.USE_Z, mirror=config.MIRROR, **extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


class GestureClassifier:
    """실시간 루프에서 쓰는 추론 래퍼. features: (SEQ_LEN, FEATURE_DIM)

    GestureClassifier()            → models/gru_gesture.pt
    GestureClassifier(arch="lstm") → models/lstm_gesture.pt
    """

    def __init__(self, arch: str = DEFAULT_ARCH, model_file: Path | None = None, meta_file: Path | None = None):
        model_file = config.project_path(model_file) if model_file else model_path(arch)
        meta_file = config.project_path(meta_file) if meta_file else model_file.with_suffix(".json")
        for path in (model_file, meta_file):
            if not path.is_file():
                raise FileNotFoundError(f"동적 제스처 모델/metadata를 복사하세요: {path}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if (meta.get("seq_len") != config.SEQ_LEN or meta.get("feature_dim") != config.FEATURE_DIM
                or meta.get("labels") != config.LABELS or meta.get("mirror") != config.MIRROR
                or meta.get("use_z") != config.USE_Z):
            raise ValueError(f"모델 metadata의 라벨/좌표/시퀀스 규약이 현재 프로그램과 다릅니다: {meta_file}")
        self.model = load_model(model_file)
        if self.model.n_classes != len(meta["labels"]) or self.model.arch != meta.get("arch", arch):
            raise ValueError(f"가중치와 metadata의 클래스 수/아키텍처가 다릅니다: {model_file}")
        self.arch = meta.get("arch", arch)
        self.labels = meta["labels"]
        assert meta["seq_len"] == config.SEQ_LEN and meta["feature_dim"] == config.FEATURE_DIM, \
            "모델 규약(seq_len/feature_dim)이 현재 config와 다릅니다. 다시 학습하세요."

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, features: np.ndarray):
        probs = self.model.predict_proba(features[None])[0]
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i]), probs
