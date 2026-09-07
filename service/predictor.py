"""체크포인트 한 번 복원 → 기존 전처리 → 고정 가중치 추론. 데이터 파일은 읽지 않습니다."""

from dataclasses import fields
import math
from pathlib import Path

import numpy as np
import torch

from training import prepare_dataset as preparation
from training.evaluate import restore_model, same_metadata
from training.train import choose_device, validate_label_map
from training.gesture_schema import schema_from_config, RIGHT_LABELS

LABELS = tuple(RIGHT_LABELS)


class SequenceQualityError(ValueError):
    """현재 구간의 품질 부족: 프로그램 오류와 구분하여 다음 구간을 기다립니다."""


def validate_preprocessing(config):
    """현재 전처리 구현과 다른 수식/반전/표준화/버전은 조용히 무시하지 않습니다."""
    if not isinstance(config, dict):
        raise ValueError("preprocessing_config가 필요합니다.")
    length = config.get("seq_len")
    if type(length) is not int or length < 2:
        raise ValueError("seq_len은 2 이상의 정수여야 합니다.")
    quality = config.get("quality")
    required = {field.name for field in fields(preparation.QualityConfig)}
    if not isinstance(quality, dict) or set(quality) != required:
        raise ValueError("지원하지 않는 quality 설정입니다.")
    for key, value in quality.items():
        if key in ("min_frames", "max_missing_run"):
            if type(value) is not int:
                raise ValueError(f"quality.{key}는 정수여야 합니다.")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"quality.{key}는 유한한 수여야 합니다.")
    quality = preparation.QualityConfig(**quality)
    quality.validate()
    # build_config는 I/O 없는 기존 설정 생성 함수입니다. 수식 설명을 중복 복사하지 않습니다.
    args = preparation.make_parser().parse_args(["--output-dir", "unused"])
    schema_from_config(config)
    reference = preparation.build_config(args, quality, {}, {})
    # 특징 수식/버전은 동일합니다. 검증된 구형 3클래스 모델도 자기 라벨로 복원합니다.
    reference["label_map"] = dict(config["label_map"])
    if set(config) != set(reference):
        raise ValueError("지원하지 않는 전처리 설정 필드입니다.")
    variable = {"seq_len", "quality", "participant_map", "requested_split_subjects", "canonical_split_subjects"}
    for key in reference.keys() - variable:
        if not same_metadata(config[key], reference[key]):
            raise ValueError(f"지원하지 않는 전처리 계약: {key}")
    return length, quality


class GesturePredictor:
    """서비스 시작 시 생성합니다. 모델/전처리 설정은 체크포인트를 유일한 기준으로 삼습니다."""

    def __init__(self, checkpoint_path: Path, device="auto"):
        self.device = choose_device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        required = {"model_state_dict", "model_config", "label_map", "preprocessing_config", "best_epoch", "metrics"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("train.py 형식의 체크포인트가 필요합니다.")
        epoch, metrics = checkpoint["best_epoch"], checkpoint["metrics"]
        if (type(epoch) is not int or epoch < 1 or not isinstance(metrics, dict)
                or type(metrics.get("epoch")) is not int or metrics["epoch"] != epoch):
            raise ValueError("체크포인트 best_epoch/metrics 계약 오류")
        for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            value = metrics.get(key)
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or (key.endswith("accuracy") and value > 1)):
                raise ValueError(f"체크포인트 metrics 오류: {key}")
        validate_label_map(checkpoint["label_map"])
        self.labels = tuple(sorted(checkpoint["label_map"], key=checkpoint["label_map"].get))
        self.gesture_schema = schema_from_config(checkpoint["preprocessing_config"])
        if checkpoint["label_map"] != checkpoint["preprocessing_config"]["label_map"]:
            raise ValueError("체크포인트와 전처리 label_map 불일치")
        config = checkpoint["model_config"]
        if (not isinstance(config, dict) or set(config) !=
                {"input_size", "hidden_size", "num_layers", "num_classes", "dropout"}
                or type(config["input_size"]) is not int or config["input_size"] != 66
                or type(config["num_classes"]) is not int or config["num_classes"] != len(self.labels)):
            raise ValueError("LSTM input_size=66, num_classes와 label_map 일치 계약이 필요합니다.")
        self.seq_len, self.quality = validate_preprocessing(checkpoint["preprocessing_config"])
        self.model = restore_model({"model_config": config, "checkpoint": checkpoint}, self.device)
        self.model_config = dict(config)

    def preprocess(self, landmarks, timestamps, detected):
        """구간 길이를 seq_len으로 리샘플링합니다. 손 이동량 필터는 추가하지 않습니다."""
        try:
            features, _ = preparation.preprocess_sequence(
                landmarks, timestamps, detected, self.seq_len, self.quality)
        except ValueError as exc:
            raise SequenceQualityError(str(exc)) from exc
        return features

    def predict(self, landmarks, timestamps, detected):
        features = self.preprocess(landmarks, timestamps, detected)
        if features.dtype != np.float32 or features.shape != (self.seq_len, 66) or not np.isfinite(features).all():
            raise ValueError("전처리 출력 float32 (seq_len,66) 계약 오류")
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        self.model.eval()  # Dropout 비활성화와 미분 그래프 비활성화는 서로 다른 설정입니다.
        with torch.inference_mode():
            logits = self.model(x)
            if logits.shape != (1, len(self.labels)) or not torch.isfinite(logits).all().item():
                raise FloatingPointError("LSTM logits shape 오류 또는 NaN/Inf")
            probabilities = logits.softmax(dim=1)[0]
            if not torch.isfinite(probabilities).all().item():
                raise FloatingPointError("LSTM 확률에 NaN/Inf가 있습니다.")
            # argmax는 클래스 선택, softmax는 화면/판정용 점수입니다. 정확도를 보증하지 않습니다.
            return int(logits.argmax(dim=1).item()), probabilities.cpu().numpy()
