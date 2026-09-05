"""수집한 NPZ를 검사하고, 참가자별로 분리된 LSTM/GRU 학습 입력을 만드는 도구.

전체 구조를 처음 볼 때는 아래쪽 run()부터 읽으면 처리 순서를 파악하기 쉽습니다.
    main() → 실행 옵션 해석
    run() → 파일 탐색 → scan_file() → 중복 검사 → 참가자 분리 검사 → 결과 저장
    scan_file() → 파일 하나 검사 → preprocess_sequence()로 특징 생성
    preprocess_sequence() → 품질 검사 → 크기 기준 계산 → 결측 보간 → 리샘플링 → 66차원 특징

입력: landmarks (T, 21, 3), timestamps (T,), detected (T,)와 라벨/참가자 정보.
출력: 일반 모드는 X (N, seq_len, 66), y (N,) 및 보고서. audit-only는 보고서만 저장.
T는 원본 프레임 수, N은 해당 split의 샘플 수이며 시간 단위는 초입니다.

원본 NPZ는 읽기만 합니다. 이 모듈을 import해도 파일이나 카메라를 열지 않으며,
모델 학습도 실행하지 않습니다. 실행 예시와 출력 상세는 README.md를 참고하세요.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import zipfile
import zlib

import numpy as np


# -----------------------------------------------------------------------------
# 1. 공통 설정: 경로, 클래스 번호, 보고서 열, 품질 기준
# -----------------------------------------------------------------------------
# 실행한 터미널 위치가 아니라 이 파일 위치를 기준으로 프로젝트 루트를 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 일부 클래스의 데이터가 없어도 라벨 번호는 바뀌지 않습니다.
LABEL_MAP = {"swipe_left": 0, "make_fist": 1, "no_gesture": 2}
SPLITS = ("train", "val", "test")
VERSION = "1.0.0"
# manifest.csv의 열 순서. 발견한 원본 파일 하나가 보고서의 한 행에 대응합니다.
FIELDS = (
    "source_path", "sha256", "arrays_sha256", "identity_content_sha256",
    "participant_id", "canonical_participant_id", "label", "sample_id",
    "schema", "status", "reason", "split", "output_index", "original_T",
    "duration", "valid_frames", "valid_detection_ratio", "longest_missing_run",
    "scale", "assumptions",
)


@dataclass(frozen=True)
class QualityConfig:
    """시퀀스 채택 기준. CLI에서 바꿀 수 있으며 실제 사용값은 config에 저장합니다."""

    # 검출률은 0~100 퍼센트가 아니라 0~1 비율입니다.
    min_frames: int = 20
    min_detection_rate: float = 0.8
    max_missing_run: int = 5
    min_duration: float = 0.6
    max_duration: float = 2.5
    scale_epsilon: float = 1e-6  # 지나치게 작은 손 크기로 나누는 것을 방지합니다.

    def validate(self) -> None:
        """데이터 검사 전에 설정값 자체가 허용 범위인지 확인합니다."""
        if self.min_frames < 2 or self.max_missing_run < 0:
            raise ValueError("min_frames must be >= 2; max_missing_run must be >= 0")
        floats = (self.min_detection_rate, self.min_duration,
                  self.max_duration, self.scale_epsilon)
        if not all(np.isfinite(v) for v in floats):
            raise ValueError("thresholds must be finite")
        if not 0 <= self.min_detection_rate <= 1:
            raise ValueError("min_detection_rate must be in [0, 1]")
        if not 0 < self.min_duration <= self.max_duration:
            raise ValueError("require 0 < min_duration <= max_duration")
        if self.scale_epsilon <= 0:
            raise ValueError("scale_epsilon must be positive")


# -----------------------------------------------------------------------------
# 2. 입력 구조 및 품질 검사: 보간으로 오류를 덮기 전에 원본 상태를 확인
# -----------------------------------------------------------------------------
def valid_identifier(value: object) -> bool:
    """참가자 ID 형식을 검사합니다. person_one처럼 밑줄이 있는 ID도 허용합니다."""
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is not None


def scalar(data, key: str, kind: str):
    """NPZ에 저장된 단일 값을 검사한 뒤 Python 문자열/정수/bool로 꺼냅니다.

    kind는 NumPy dtype 종류입니다: U=문자열, i/u=정수, b=불리언.
    shape=()인 단일 값만 허용하며 길이 1 배열을 임의로 단일 값으로 바꾸지 않습니다.
    """
    value = data[key]
    if value.shape != () or value.dtype.kind not in kind:
        raise ValueError(f"invalid_scalar:{key}")
    return value.item()


def numeric_array(value: np.ndarray, shape: tuple, name: str,
                  dtype=np.float64) -> np.ndarray:
    """수치 배열의 모양을 검사하고 dtype을 변환하되, 변환 중 발생한 넘침은 거부합니다."""
    if value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError(f"invalid_array:{name}:expected_numeric_{shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = value.astype(dtype)
    # 원래 있던 NaN/Inf는 이후 결측 검사에 사용합니다.
    # 정상 숫자가 dtype 변환 때문에 Inf로 바뀐 경우는 별도의 입력 오류입니다.
    if np.any(np.isfinite(value) & ~np.isfinite(converted)):
        raise ValueError(f"unsafe_dtype_conversion:{name}")
    return converted


def validate_arrays(landmarks, timestamps, detected):
    """필수 배열의 형태와 자료형을 확인합니다. 잘못된 차원이나 bool을 임의로 고치지 않습니다."""
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise ValueError("invalid_array:landmarks:expected_(T,21,3)")
    count = landmarks.shape[0]
    points = numeric_array(landmarks, (count, 21, 3), "landmarks", np.float32)
    times = numeric_array(timestamps, (count,), "timestamps")
    if detected.shape != (count,) or detected.dtype.kind != "b":
        raise ValueError("invalid_array:detected:expected_bool_(T,)")
    return points, times, detected.copy()


def longest_missing_run(valid: np.ndarray) -> int:
    """유효 여부 배열에서 False가 가장 길게 연속된 프레임 수를 반환합니다."""
    longest = run = 0
    for present in valid:
        run = 0 if present else run + 1
        longest = max(longest, run)
    return longest


def quality_metrics(landmarks, timestamps, detected):
    """원본 배열로 유효 프레임 마스크와 품질 지표를 다시 계산합니다.

    NPZ에 저장된 duration/detection_rate 요약값 대신 실제 배열을 기준으로 삼습니다.
    시간 역전이나 중복 timestamp는 정렬로 감추지 않고 오류로 처리합니다.
    """
    if len(timestamps) < 2:
        raise ValueError("too_few_timestamps")
    with np.errstate(over="ignore", invalid="ignore"):
        duration = float(timestamps[-1] - timestamps[0])
        increasing = np.all(np.diff(timestamps) > 0)
    if not np.isfinite(timestamps).all() or not increasing or not np.isfinite(duration):
        raise ValueError("invalid_timestamps:require_finite_strictly_increasing")
    # 손이 검출되었어도 21개 점의 xyz 중 하나라도 NaN/Inf면 해당 프레임은 결측입니다.
    valid = detected & np.isfinite(landmarks).all(axis=(1, 2))
    metrics = {
        "original_T": len(timestamps), "duration": duration,
        "valid_frames": int(valid.sum()), "valid_detection_ratio": float(valid.mean()),
        "longest_missing_run": longest_missing_run(valid),
    }
    return valid, metrics


def quality_reasons(metrics: dict, config: QualityConfig) -> list[str]:
    """품질 기준을 못 넘긴 이유들을 반환합니다. 빈 목록이면 품질 검사를 통과한 것입니다."""
    # 이동량과 dx 부호는 검사하지 않습니다. 정지 no_gesture나 작은 make_fist도 유효합니다.
    checks = (
        (metrics["valid_frames"] == 0, "all_missing"),
        (metrics["original_T"] < config.min_frames, "too_few_frames"),
        (metrics["valid_detection_ratio"] < config.min_detection_rate, "low_detection_rate"),
        (metrics["longest_missing_run"] > config.max_missing_run, "long_missing_run"),
        (metrics["duration"] < config.min_duration, "duration_too_short"),
        (metrics["duration"] > config.max_duration, "duration_too_long"),
    )
    return [reason for failed, reason in checks if failed]


# -----------------------------------------------------------------------------
# 3. 단일 시퀀스 전처리: 파일 입출력 없이 배열만 받아 계산하는 재사용 가능 함수
# -----------------------------------------------------------------------------
def interpolate_landmarks(landmarks, timestamps, valid):
    """원래 프레임 시각을 유지하면서 결측을 채웁니다. 출력 길이는 아직 T입니다."""
    if not np.any(valid):
        raise ValueError("all_missing")
    times = timestamps - timestamps[0]
    # 21개 랜드마크 × xyz를 63개 열로 펼쳐 각 좌표를 시간에 따라 따로 보간합니다.
    # 유효 프레임만 보간의 기준점으로 쓰되, 결과에는 결측 프레임 시각도 그대로 남깁니다.
    values = landmarks[valid].reshape((-1, 63))
    # np.interp는 중간을 선형 보간하고, 유효 구간 바깥 양끝은 가장 가까운 값으로 채웁니다.
    filled = np.column_stack([
        np.interp(times, times[valid], values[:, column]) for column in range(63)
    ])
    return filled.reshape((-1, 21, 3))


def resample_landmarks(filled, timestamps, seq_len: int = 32):
    """결측을 채운 T프레임을 실제 시간 기준의 고정 길이(seq_len)로 변환합니다."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    times = timestamps - timestamps[0]
    # 예: 녹화 시간이 1초이고 seq_len=32면 0~1초에 32개의 등간격 시각을 만듭니다.
    # 프레임 번호 기준으로 고르는 것이 아니므로 불규칙한 촬영 간격도 반영됩니다.
    # 모든 동작을 같은 길이로 바꾸므로 실제 소요 시간은 특징이 아닌 메타데이터에 남습니다.
    target = np.linspace(0.0, times[-1], seq_len)
    values = filled.reshape((-1, 63))
    result = np.column_stack([
        np.interp(target, times, values[:, column]) for column in range(63)
    ])
    return result.reshape((seq_len, 21, 3))


def sequence_scale(landmarks, valid, epsilon: float = 1e-6) -> float:
    """보간 전 유효 프레임에서 손목(0)~중지 MCP(9)의 xy 거리 중앙값 s를 구합니다."""
    # s는 시퀀스 전체에 고정해서 사용합니다. 매 프레임 다른 크기로 나누면
    # 주먹을 쥘 때의 손 모양/크기 변화까지 과도하게 지워질 수 있기 때문입니다.
    original = landmarks[valid].astype(np.float64)
    distances = np.linalg.norm(original[:, 9, :2] - original[:, 0, :2], axis=1)
    positive = distances[np.isfinite(distances) & (distances > 0)]
    scale = float(np.median(positive)) if len(positive) else 0.0
    if not np.isfinite(scale) or scale <= epsilon:
        raise ValueError("degenerate_scale")
    return scale


def features_from_landmarks(resampled, scale: float):
    """리샘플링된 좌표에서 손 모양 63개 + 손목 이동 2개 + 상대 크기 1개를 만듭니다."""
    # x/y는 각각 영상 폭/높이 기준 정규화 좌표입니다. 여기서의 거리는 물리적 거리가
    # 아니며, 영상 크기 정보가 없으므로 실제 종횡비 보정도 수행하지 않습니다.
    wrist = resampled[:, 0, :]
    # [0:63] 현재 손목 기준 상대 xyz: 화면상의 위치 영향을 줄이고 손가락 모양을 보존.
    # wrist[:, None, :]는 같은 프레임 손목을 21개 랜드마크 모두에서 빼기 위한 차원입니다.
    relative = ((resampled - wrist[:, None, :]) / scale).reshape((-1, 63))
    # [63:65] 첫 출력 프레임 대비 손목의 누적 dx, dy. 직전 프레임과의 차분이 아닙니다.
    # 위 상대좌표에서는 손목 자체가 0이 되므로 스와이프 이동을 이 두 값에 따로 보존합니다.
    travel = (wrist[:, :2] - wrist[0, :2]) / scale
    # [65] 고정 크기 s에 대한 현재 손목~중지 MCP 거리의 비율.
    size = np.linalg.norm(resampled[:, 9, :2] - wrist[:, :2], axis=1) / scale
    with np.errstate(over="ignore", invalid="ignore"):
        features = np.column_stack((relative, travel, size)).astype(np.float32)
    if features.shape != (len(resampled), 66) or not np.isfinite(features).all():
        raise ValueError("nonfinite_features")
    return features


def preprocess_sequence(landmarks, timestamps, detected, seq_len: int = 32,
                        config: QualityConfig | None = None):
    """단일 시퀀스 전처리의 대표 함수. (특징 배열, 품질 메타데이터)를 반환합니다.

    특징 배열은 float32 (seq_len, 66)입니다. 입력 배열을 변경하거나 파일을 저장하지
    않으므로 이후 추론 코드에서도 재사용할 수 있습니다. 동작 구간을 찾는 기능은 아닙니다.
    라벨/참가자 ID를 특징에 넣지 않으며 전체 데이터의 평균·표준편차도 사용하지 않습니다.
    구조나 품질이 잘못된 시퀀스는 ValueError로 거부합니다.
    """
    config = config or QualityConfig()
    config.validate()
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    points, times, mask = validate_arrays(landmarks, timestamps, detected)
    valid, metrics = quality_metrics(points, times, mask)
    reasons = quality_reasons(metrics, config)
    if reasons:
        raise ValueError(";".join(reasons))
    # 순서가 중요합니다. 원본 품질 검사 → 원본에서 s 계산 → 보간 → 길이 통일 → 특징 생성.
    scale = sequence_scale(points, valid, config.scale_epsilon)
    filled = interpolate_landmarks(points, times, valid)
    sampled = resample_landmarks(filled, times, seq_len)
    return features_from_landmarks(sampled, scale), {**metrics, "scale": scale}


# -----------------------------------------------------------------------------
# 4. 파일 단위 검사와 중복 판정: 원본 ID와 확인된 대표 ID(canonical ID)를 구분
# -----------------------------------------------------------------------------
def arrays_digest(landmarks, timestamps, detected) -> str:
    """NPZ 압축 방식과 수치 저장 dtype 차이를 줄인 필수 배열 비교용 해시를 만듭니다.

    해시는 데이터 내용을 비교하기 위한 지문입니다. 좌표/시간은 float64로 통일하고,
    NaN의 내부 표현과 +0/-0도 통일합니다. 원래 절대 timestamp는 비교에 포함합니다.
    파일 전체의 SHA-256과 별개이며, 선택 메타데이터는 이 해시에 넣지 않습니다.
    """
    digest = hashlib.sha256()
    for name, array in (("landmarks", landmarks), ("timestamps", timestamps),
                        ("detected", detected)):
        normalized = array.astype("u1" if name == "detected" else "<f8", copy=True)
        if name != "detected":
            normalized[np.isnan(normalized)] = np.nan
            normalized[normalized == 0] = 0.0
        digest.update(json.dumps([name, list(array.shape)], separators=(",", ":")).encode())
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def load_participant_map(path: Path | None) -> dict[str, str]:
    """담당자가 확인한 원본 ID → 대표 ID JSON을 읽습니다. 같은 사람인지 추측하지 않습니다."""
    if path is None:
        return {}
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate participant-map key: {key}")
            result[key] = value
        return result
    with path.open(encoding="utf-8-sig") as stream:
        mapping = json.load(stream, object_pairs_hook=unique_keys)
    if not isinstance(mapping, dict) or not all(
        valid_identifier(k) and valid_identifier(v) for k, v in mapping.items()
    ):
        raise ValueError("participant-map must be a JSON object of valid string IDs")
    # a → b → c 같은 연쇄나 순환은 거부합니다. 각 별칭을 최종 대표 ID로 직접 연결해야 합니다.
    if any(v in mapping and mapping[v] != v for v in mapping.values()):
        raise ValueError("participant-map chains/cycles are not allowed; map directly to final IDs")
    return mapping


def check_optional_fields(data, count: int, schema: int, row: dict) -> None:
    """선택 필드가 있으면 구조를 검사하고, 현재 오른손 원본 데이터 조건을 확인합니다."""
    if "world_landmarks" in data:
        numeric_array(data["world_landmarks"], (count, 21, 3), "world_landmarks", np.float32)
    if "handedness_per_frame" in data:
        values = data["handedness_per_frame"]
        if values.shape != (count,) or values.dtype.kind != "U":
            raise ValueError("invalid_array:handedness_per_frame")
    if "handedness_confidence" in data:
        numeric_array(data["handedness_confidence"], (count,), "handedness_confidence", np.float32)
    if "handedness" in data and scalar(data, "handedness", "U") != "Right":
        raise ValueError("unexpected_handedness:expected_Right")
    if "mirrored" in data:
        if scalar(data, "mirrored", "b"):
            raise ValueError("mirrored_sample")
    elif schema == 1:
        # 구버전에는 반전 여부가 없었습니다. 당시 수집 코드에 근거한 가정을 보고서에 남깁니다.
        row["assumptions"] = "legacy_v1_missing_mirrored_assumed_false_from_collector"
    else:
        raise ValueError("missing_mirrored_in_v2")
    # 명시적으로 표시된 합성/증강 자료만 판별합니다. 표시 없는 합성을 자동 감지하지는 않습니다.
    for key in ("synthetic", "is_synthetic", "augmented", "is_augmented"):
        if key in data and scalar(data, key, "b"):
            raise ValueError(f"synthetic_sample:{key}")
    for key in ("augmentation", "augmentation_type"):
        if key in data and scalar(data, key, "U").strip().lower() not in ("", "none", "original"):
            raise ValueError(f"synthetic_sample:{key}")


def scan_file(path: Path, input_dir: Path, mapping: dict, seq_len: int,
              config: QualityConfig):
    """NPZ 하나를 검사해 (manifest 행, 특징 배열)을 반환합니다. 제외 시 특징은 None입니다."""
    # 먼저 제외 상태로 시작하고 모든 검사를 통과해야 accepted로 바꿉니다.
    # 없는 메타데이터는 빈 칸으로 남겨, 오류가 난 파일도 보고서에서 추적할 수 있게 합니다.
    row = dict.fromkeys(FIELDS, "")
    row.update(source_path=path.relative_to(input_dir).as_posix(), status="excluded")
    feature = None
    try:
        with path.open("rb") as stream:
            row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        # pickle 객체는 허용하지 않고, with 블록을 나갈 때 NPZ 파일 핸들을 닫습니다.
        with np.load(path, allow_pickle=False) as data:
            required = {"landmarks", "timestamps", "detected", "label", "participant_id", "sample_id"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError("missing_fields:" + ",".join(missing))
            row["label"] = scalar(data, "label", "U")
            row["participant_id"] = scalar(data, "participant_id", "U")
            row["sample_id"] = scalar(data, "sample_id", "iu")
            if not valid_identifier(row["participant_id"]):
                raise ValueError("invalid_participant_id")
            if row["label"] not in LABEL_MAP:
                raise ValueError("unknown_label")
            if row["sample_id"] < 1:
                raise ValueError("invalid_sample_id:must_be_positive")
            # 파일명 검사는 원본 ID로, 중복/참가자 분리는 별칭 적용 후 대표 ID로 처리합니다.
            row["canonical_participant_id"] = mapping.get(row["participant_id"], row["participant_id"])
            raw_points, raw_times, raw_mask = data["landmarks"], data["timestamps"], data["detected"]
            points, times, mask = validate_arrays(raw_points, raw_times, raw_mask)
            row["original_T"] = len(times)
            # 품질 때문에 나중에 제외되더라도, 구조가 확인된 배열의 충돌은 검사할 수 있게 합니다.
            row["arrays_sha256"] = arrays_digest(raw_points, raw_times, raw_mask)
            identity = [row[k] for k in ("canonical_participant_id", "label", "sample_id")]
            payload = json.dumps([identity, row["arrays_sha256"]], ensure_ascii=True)
            row["identity_content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            row["schema"] = scalar(data, "collection_schema_version", "iu") if "collection_schema_version" in data else 1
            if row["schema"] not in (1, 2):
                raise ValueError(f"unknown_schema:{row['schema']}")
            expected = f"{row['participant_id']}_{row['label']}_{row['sample_id']:04d}.npz"
            if path.name != expected:
                raise ValueError(f"filename_mismatch:expected_{expected}")
            if path.parent.name != row["label"]:
                raise ValueError("label_folder_mismatch")
            check_optional_fields(data, len(times), row["schema"], row)
            valid, metrics = quality_metrics(points, times, mask)
            row.update(metrics)
            reasons = quality_reasons(metrics, config)
            if reasons:
                raise ValueError(";".join(reasons))
            # 위 품질 계산은 제외 사유/수치를 보고서에 남기기 위한 것입니다.
            # 재사용용 전처리 함수도 독립적으로 안전하게 호출할 수 있도록 내부 검사를 수행합니다.
            feature, metrics = preprocess_sequence(points, times, mask, seq_len, config)
            row.update(metrics, status="accepted")
    except (ValueError, TypeError, KeyError, OSError, EOFError, zipfile.BadZipFile,
            zlib.error, RuntimeError, NotImplementedError, OverflowError) as exc:
        # 파일 하나가 손상되어도 전체 탐색은 계속하고, 해당 파일의 오류를 기록합니다.
        row["reason"] = str(exc) or type(exc).__name__
    return row, feature


def resolve_duplicates(rows: list[dict]) -> list[dict]:
    """rows의 상태를 중복/충돌로 갱신하고 충돌 목록을 반환합니다. 충돌 판정이 우선입니다."""
    # 두 방향으로 묶어 검사합니다.
    # identities: 같은 대표 참가자/라벨/샘플 번호에 서로 다른 데이터가 있는가?
    # contents: 같은 원본 배열이 서로 다른 식별자로 제출되었는가?
    identities, contents = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        if row["arrays_sha256"]:
            key = tuple(row[k] for k in ("canonical_participant_id", "label", "sample_id"))
            identities[key].append(index)
            contents[row["arrays_sha256"]].append(index)
    conflicts = []
    for groups, field, reason in (
        (identities, "arrays_sha256", "same_identity_different_arrays"),
        (contents, "identity_content_sha256", "same_arrays_different_identity"),
    ):
        for indices in groups.values():
            if len({rows[i][field] for i in indices}) > 1:
                conflicts.append({"reason": reason, "sources": [rows[i]["source_path"] for i in indices]})
                for index in indices:
                    row = rows[index]
                    row["status"] = "conflict"
                    row["reason"] = ";".join(filter(None, (row["reason"], reason)))
    # 충돌 없는 동일 자료는 경로 정렬상 첫 번째 유효 파일만 채택합니다.
    # 보고서 상태만 변경하며 원본 복사본을 삭제하지 않습니다.
    for indices in identities.values():
        accepted = [i for i in indices if rows[i]["status"] == "accepted"]
        for index in accepted[1:]:
            rows[index].update(status="duplicate", reason="duplicate_of:" + rows[accepted[0]]["source_path"])
    return conflicts


# -----------------------------------------------------------------------------
# 5. 참가자별 분리와 보고서 구성: 같은 사람이 학습/검증/시험에 겹치지 않도록 검사
# -----------------------------------------------------------------------------
def validate_splits(rows: list[dict], subjects: dict, mapping: dict):
    """명시적으로 지정된 참가자 배정을 검증합니다. 샘플 무작위 분리로 대체하지 않습니다.

    반환: 대표 ID 목록, 참가자별 배정, split별 클래스 개수, 오류 목록.
    오류가 있으면 배정/분포는 진단용 정보이며 학습 배열 생성에 사용하면 안 됩니다.
    """
    normalized = {split: [mapping.get(v, v) for v in subjects[split]] for split in SPLITS}
    accepted = [row for row in rows if row["status"] == "accepted"]
    participants = {row["canonical_participant_id"] for row in accepted}
    errors, assignment = [], {}
    distribution = {split: dict.fromkeys(LABEL_MAP, 0) for split in SPLITS}
    for split, ids in normalized.items():
        if not ids:
            errors.append(f"empty_split:{split}")
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate_subject_in_split:{split}")
        for participant in ids:
            if participant not in participants:
                errors.append(f"unknown_or_no_accepted_subject:{split}:{participant}")
            if participant in assignment and assignment[participant] != split:
                errors.append(f"participant_leakage:{participant}")
            assignment[participant] = split
    for participant in sorted(participants - assignment.keys()):
        errors.append(f"unassigned_subject:{participant}")
    for row in accepted:
        split = assignment.get(row["canonical_participant_id"])
        if split:
            distribution[split][row["label"]] += 1
    # 참가자가 겹치지 않는 것뿐 아니라 각 split에 세 클래스가 모두 있는지도 확인합니다.
    for split in SPLITS:
        missing = [label for label, count in distribution[split].items() if not count]
        if missing:
            errors.append(f"missing_classes:{split}:" + ",".join(missing))
    return normalized, assignment, distribution, errors


def count_groups(rows: list[dict], field: str):
    """지정한 항목(라벨/참가자 등)별 입력 개수와 채택·제외·중복·충돌 개수를 집계합니다."""
    result = {}
    for row in rows:
        group = str(row[field]) if row[field] != "" else "<unknown>"
        counts = result.setdefault(group, {"input": 0, "accepted": 0, "excluded": 0,
                                           "duplicate": 0, "conflict": 0})
        counts["input"] += 1
        counts[row["status"]] += 1
    return dict(sorted(result.items()))


def build_config(args, quality, mapping, normalized):
    """같은 전처리를 재현할 수 있도록 수식, 옵션, ID 배정을 JSON 저장용 사전으로 만듭니다."""
    return {
        "preprocessing_version": VERSION, "seq_len": args.seq_len, "feature_dim": 66,
        "X_dtype": "float32", "y_dtype": "int64", "label_map": LABEL_MAP,
        "quality": asdict(quality), "participant_map": mapping,
        "requested_split_subjects": {s: getattr(args, s + "_subjects") for s in SPLITS},
        "canonical_split_subjects": normalized,
        "interpolation": {"axis": "timestamps - timestamps[0], seconds",
                          "missing": "linear on original timestamps; nearest valid value at edges",
                          "resampling": "linear on filled sequence at linspace(0, duration, seq_len)"},
        "scale": "fixed sequence median of positive wrist(0)-middle_MCP(9) xy distances in ORIGINAL valid frames",
        "features": {"0:63": "flatten_landmark_xyz((resampled_xyz - current_wrist_xyz) / s)",
                     "63:65": "(current_wrist_xy - first_resampled_wrist_xy) / s (cumulative, not velocity)",
                     "65": "norm(current_landmark9_xy - current_wrist_xy) / s"},
        "coordinate_caveat": "image-normalized x/y use width/height respectively; approximate, not physical or aspect-ratio-corrected; z is not metric depth",
        "absolute_duration_in_features": False, "global_standardization": None,
        "augmentation": None, "world_landmarks_used": False,
        "valid_mask": "detected AND all 63 landmark coordinates finite",
        "handedness": "Right when provided; no handedness inferred when absent",
        "mirrored": "false only; legacy v1 missing flag assumed false; v2 missing flag excluded",
        "synthetic_flags_checked": ["synthetic", "is_synthetic", "augmented", "is_augmented", "augmentation", "augmentation_type"],
        "deduplication": "canonical participant/label/sample_id + normalized essential arrays; float64 bytes, canonical NaN/zero, absolute timestamps; source SHA-256 also recorded",
        "ordering": "source_path sorted; first valid duplicate retained; output_index zero-based within split",
    }


# -----------------------------------------------------------------------------
# 6. 결과 저장: 일반 모드는 배열+보고서, 점검/사전 검증 실패 시에는 보고서만 저장
# -----------------------------------------------------------------------------
def write_outputs(output_dir: Path, rows: list[dict], features: list,
                  report: dict, config: dict, emit_arrays: bool) -> None:
    """임시 폴더에서 결과를 완성하고 검증한 뒤 최종 output_dir로 옮깁니다."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # 저장 중 오류가 나도 일부 X/y가 완성된 결과처럼 최종 폴더에 남지 않도록 합니다.
    with TemporaryDirectory(prefix=".prepare_dataset_", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        if emit_arrays:
            for split in SPLITS:
                indices = [i for i, row in enumerate(rows)
                           if row["status"] == "accepted" and row["split"] == split]
                # 동일한 indices 순서로 X와 y를 쌓아 특징과 정답의 대응을 유지합니다.
                # X: (샘플 수, seq_len, 66), y: (샘플 수,).
                x = np.stack([features[i] for i in indices]).astype(np.float32)
                y = np.array([LABEL_MAP[rows[i]["label"]] for i in indices], dtype=np.int64)
                if x.shape != (len(y), config["seq_len"], 66) or not np.isfinite(x).all():
                    raise ValueError(f"output_validation_failed:{split}")
                # 원본 파일 행에 split 내부의 0부터 시작하는 배열 인덱스를 기록합니다.
                for output_index, source_index in enumerate(indices):
                    rows[source_index]["output_index"] = output_index
                np.save(staging / f"X_{split}.npy", x, allow_pickle=False)
                np.save(staging / f"y_{split}.npy", y, allow_pickle=False)
                # 저장한 파일을 다시 읽어 dtype, 내용, manifest 행 대응까지 검사합니다.
                saved_x = np.load(staging / f"X_{split}.npy", allow_pickle=False)
                saved_y = np.load(staging / f"y_{split}.npy", allow_pickle=False)
                if saved_x.dtype != np.float32 or saved_y.dtype != np.int64:
                    raise ValueError(f"output_dtype_failed:{split}")
                if not np.array_equal(saved_x, x) or not np.array_equal(saved_y, y):
                    raise ValueError(f"output_roundtrip_failed:{split}")
                for i in indices:
                    index = rows[i]["output_index"]
                    if saved_y[index] != LABEL_MAP[rows[i]["label"]] or not np.array_equal(saved_x[index], features[i]):
                        raise ValueError(f"manifest_correspondence_failed:{split}")
        # 아래 네 메타데이터 파일은 배열을 만들지 않는 audit-only에서도 저장합니다.
        with (staging / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        for name, value in (("report", report), ("preprocessing_config", config), ("label_map", LABEL_MAP)):
            with (staging / f"{name}.json").open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        if output_dir.exists():
            raise FileExistsError(f"output-dir already exists; choose a new directory: {output_dir}")
        staging.rename(output_dir)


# -----------------------------------------------------------------------------
# 7. 실행 진입점: 옵션 해석 → 전체 작업 제어 → 종료 코드 반환
# -----------------------------------------------------------------------------
def make_parser():
    """터미널에서 전달하는 --seq-len 등의 실행 옵션과 기본값을 정의합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--min-detection-rate", type=float, default=0.8)
    parser.add_argument("--max-missing-run", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=0.6)
    parser.add_argument("--max-duration", type=float, default=2.5)
    parser.add_argument("--scale-epsilon", type=float, default=1e-6)
    parser.add_argument("--participant-map", type=Path)
    for split in SPLITS:
        parser.add_argument(f"--{split}-subjects", nargs="+", default=[])
    return parser


def run(args) -> int:
    """전체 작업 흐름을 연결합니다. 성공은 0, 보고서가 남는 점검/분리 실패는 1입니다."""
    # [1단계] 품질 설정과 입출력 경로 검사: 원본 내부 저장 및 기존 결과 덮어쓰기 방지.
    quality = QualityConfig(**{key: getattr(args, key) for key in QualityConfig.__dataclass_fields__})
    quality.validate()
    if args.seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input-dir is not a directory: {input_dir}")
    if output_dir.is_relative_to(input_dir) or output_dir.is_relative_to((PROJECT_ROOT / "dataset").resolve()):
        raise ValueError("output-dir must be outside the source dataset")
    if output_dir.exists():
        raise ValueError(f"output-dir already exists; choose a new directory: {output_dir}")
    # [2단계] 별칭을 읽고 NPZ만 재귀 탐색합니다. index.csv는 입력 기준으로 사용하지 않습니다.
    # 상대경로 정렬로 중복 선택과 출력 행 순서를 재현 가능하게 유지합니다.
    mapping = load_participant_map(args.participant_map)
    paths = sorted((p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".npz"),
                   key=lambda p: p.relative_to(input_dir).as_posix())
    # [3단계] 각 파일을 검사하고 특징을 계산합니다. rows[i]와 features[i]는 같은 파일입니다.
    pairs = [scan_file(path, input_dir, mapping, args.seq_len, quality) for path in paths]
    rows, features = ([p[0] for p in pairs], [p[1] for p in pairs])
    # [4단계] 별칭 적용 후 중복/충돌을 먼저 처리하고, 채택된 참가자의 split 조건을 검사합니다.
    conflicts = resolve_duplicates(rows)
    subjects = {split: getattr(args, split + "_subjects") for split in SPLITS}
    normalized, assignment, distribution, split_errors = validate_splits(rows, subjects, mapping)
    accepted = [r for r in rows if r["status"] == "accepted"]
    # [5단계] 배열 생성 여부 결정. audit-only에서는 split 부족을 경고로만 안내하지만,
    # 충돌이나 유효 자료 없음은 여전히 실패입니다. 일반 모드는 모든 조건을 충족해야 합니다.
    errors = []
    if conflicts:
        errors.append("identity_conflicts:resolve_conflicting_sources_before_preparing")
    if not accepted:
        errors.append("no_accepted_samples")
    if not args.audit_only:
        errors.extend(split_errors)
    emit_arrays = not args.audit_only and not errors
    if emit_arrays:
        for row in accepted:
            row["split"] = assignment[row["canonical_participant_id"]]
    warnings = []
    if args.audit_only and split_errors:
        warnings.append("audit_only:split_requirements_not_met; no training arrays are produced")
    if any(row["status"] != "accepted" for row in rows):
        warnings.append("some_sources_not_accepted; inspect manifest reasons")
    unused = sorted(mapping.keys() - {r["participant_id"] for r in rows})
    if unused:
        warnings.append("unused_participant_map_keys:" + ",".join(unused))
    # [6단계] 결과 집계. accepted는 전처리 가능 상태이며 배열 생성 완료 자체를 뜻하지 않습니다.
    # 학습 배열 생성 여부는 training_arrays_written을 별도로 확인해야 합니다.
    report = {
        "mode": "audit-only" if args.audit_only else "normal", "success": not errors,
        "training_arrays_written": emit_arrays, "input_dir": str(input_dir),
        "output_dir": str(output_dir), "discovered_files": len(paths),
        "counts": dict(Counter(row["status"] for row in rows)),
        "by_label": count_groups(rows, "label"),
        "by_participant": count_groups(rows, "canonical_participant_id"),
        "by_original_participant": count_groups(rows, "participant_id"),
        "schema_counts": dict(Counter(str(r["schema"]) or "unknown" for r in rows)),
        "duplicates": [{"source_path": r["source_path"], "reason": r["reason"]} for r in rows if r["status"] == "duplicate"],
        "conflicts": conflicts, "split_class_distribution": distribution,
        "split_errors": split_errors, "errors": errors, "warnings": warnings,
        "assumptions": dict(Counter(r["assumptions"] for r in rows if r["assumptions"])),
        "accepted_ranges": {key: [min(r[key] for r in accepted), max(r[key] for r in accepted)]
                            for key in ("original_T", "duration", "valid_detection_ratio", "longest_missing_run", "scale")} if accepted else {},
    }
    # [7단계] 결과 저장이 완료된 뒤 콘솔에 최종 상태를 출력합니다.
    write_outputs(output_dir, rows, features, report, build_config(args, quality, mapping, normalized), emit_arrays)
    print(f"{report['mode']}: {'SUCCESS' if report['success'] else 'FAILED'}; "
          f"discovered={len(paths)}, accepted={len(accepted)}, arrays_written={emit_arrays}")
    print(f"Reports: {output_dir}")
    for message in errors + warnings:
        print(message)
    return 0 if report["success"] else 1


def main(argv=None) -> int:
    """CLI 진입 함수. 인자/경로/저장 오류는 콘솔에 알리고 종료 코드 2를 반환합니다."""
    args = make_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # 직접 실행할 때만 전체 작업을 시작합니다. 다른 코드에서 import하면 함수 정의만 읽습니다.
    raise SystemExit(main())
