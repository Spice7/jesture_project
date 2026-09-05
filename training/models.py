"""전처리가 끝난 제스처 시퀀스를 분류하는 단방향 LSTM 모델.

구조: (배치, 시간, 특징) → LSTM → 최상위 층 최종 hidden → Dropout → Linear.
이 파일은 모델 정의만 담당합니다. 데이터 로딩, 학습, 저장은 호출 측의 역할이며
import만으로 모델을 생성하거나 카메라/파일 입출력을 실행하지 않습니다.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """고정 길이로 전처리된 시퀀스 하나당 클래스별 logits를 반환합니다.

    Args:
        input_size: 프레임 하나의 특징 개수. 현재 전처리 결과는 66개입니다.
        hidden_size: LSTM이 각 시점의 정보를 표현하는 hidden 벡터 크기.
        num_layers: 쌓을 LSTM 층 수. 마지막 층의 최종 상태로 분류합니다.
        num_classes: 분류할 클래스 수. 기본은 swipe_left/make_fist/no_gesture의 3개.
        dropout: 학습 모드에서 특징을 무작위로 끄는 비율. 0 이상 1 미만.

    크기와 층 수는 bool이 아닌 양의 int를 받습니다. 기본값은 초기 실험용이며
    최적 설정을 의미하지 않습니다. 입력과 모델의 dtype/device는 호출 측에서 맞춥니다.
    기본 사용은 float32/CPU이며, 모델은 입력의 자료형이나 장치를 자동 변경하지 않습니다.
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # 1. 구조 설정 검사: float를 int로 잘라내거나 True를 층 수 1로 간주하지 않습니다.
        for name, value in (
            ("input_size", input_size), ("hidden_size", hidden_size),
            ("num_layers", num_layers), ("num_classes", num_classes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name}는 bool이 아닌 양의 정수여야 합니다.")
            if value < 1:
                raise ValueError(f"{name}는 1 이상이어야 합니다.")
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError("dropout은 0 이상 1 미만의 실수여야 합니다.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout은 유한한 값이며 0 이상 1 미만이어야 합니다.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = float(dropout)

        # 2. 시계열 인코더: batch_first로 (배치, 시간, 특징) 순서를 그대로 받습니다.
        # LSTM 내부 dropout은 층 사이에만 적용하므로 단층에서는 0으로 설정합니다.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout_rate if num_layers > 1 else 0.0,
        )
        # 3. 분류기: 이 Dropout은 단층 LSTM에서도 적용되고 eval()에서는 비활성화됩니다.
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, input_size) → (batch_size, num_classes).

        batch_size는 한 번에 처리하는 샘플 수, seq_len은 샘플당 시점 수입니다.
        seq_len을 32로 고정하지 않지만 한 배치의 샘플 길이는 같아야 합니다.
        정지 no_gesture도 정상 입력이며 움직임 크기를 검사하거나 재전처리하지 않습니다.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("입력은 torch.Tensor여야 합니다.")
        if not x.is_floating_point():
            raise TypeError("입력은 부동소수점 Tensor여야 합니다. 기본 dtype은 torch.float32입니다.")
        if x.ndim != 3:
            raise ValueError("입력 shape은 (batch_size, seq_len, input_size)의 3차원이어야 합니다.")
        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("batch_size와 seq_len은 1 이상이어야 합니다.")
        if x.shape[2] != self.input_size:
            raise ValueError(f"입력의 마지막 특징 차원은 {self.input_size}여야 합니다: {x.shape[2]}")

        # 초기 (h_0, c_0)를 전달하지 않으면 PyTorch가 매 호출 0 상태에서 시작합니다.
        # 이전 배치의 상태를 self에 보관하지 않아 서로 다른 녹화의 정보가 섞이지 않습니다.
        # output은 최상위 층의 모든 시점 출력 (B,T,H), h_n은 각 층의 최종 상태 (L,B,H).
        # 단방향이므로 h_n[-1]이 최상위 층에서 전체 시퀀스를 읽은 후의 요약입니다.
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (B,H): B=1이어도 squeeze하지 않아 배치 차원을 유지합니다.

        # CrossEntropyLoss가 logits를 받으므로 여기서는 softmax/argmax를 적용하지 않습니다.
        # 확률이 필요한 호출 측에서만 logits.softmax(dim=-1)을 별도로 계산합니다.
        return self.classifier(self.dropout(last_hidden))
