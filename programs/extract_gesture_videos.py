from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# Configuration
# =========================================================

# 이 파일은 프로젝트의 programs 폴더에 두는 것을 기준으로 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEOS_DIR = PROJECT_ROOT / "videos"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# 팀에서 합의한 단일 gesture 영상 길이
MINIMUM_DURATION_SECONDS = 1.0
MAXIMUM_DURATION_SECONDS = 5.0

# 기존 수집기와 동일한 품질 검사 기준
MINIMUM_FRAMES = 20
MINIMUM_DETECTION_RATE = 0.8
MAX_CONSECUTIVE_MISSING_FRAMES = 5

LOW_MOTION_WARNING_THRESHOLD = 0.5

MAX_HANDS_TO_DETECT = 1
EXPECTED_HANDEDNESS = "Right"

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# 스마트폰에서 MOV로 저장되는 경우도 고려
VIDEO_EXTENSIONS = {".mp4", ".mov"}


# =========================================================
# Utility
# =========================================================

def sanitize_name(value: str) -> str:
    """
    participant ID와 gesture label을 파일명에 안전한 문자열로 변환한다.
    영문, 숫자, -, _ 만 허용한다.
    """
    value = value.strip()

    if not value:
        raise ValueError("입력값은 비어 있을 수 없습니다.")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(
            "Participant ID와 Gesture Label은 "
            "영문, 숫자, '-', '_' 만 사용할 수 있습니다."
        )

    return value


def get_longest_missing_run(
    detected_values: np.ndarray,
) -> int:
    """
    연속으로 손을 검출하지 못한 최대 프레임 수를 반환한다.
    """
    longest_run = 0
    current_run = 0

    for detected in detected_values:
        if bool(detected):
            current_run = 0
        else:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )

    return longest_run


def calculate_movement_metrics(
    landmarks: np.ndarray,
    detected: np.ndarray,
) -> dict[str, float]:
    """
    손목 이동량과 이동 경로 길이를 계산한다.

    이동량은 저장 차단 기준이 아니라 분석 및 경고용 메타데이터로 사용한다.
    """
    finite_frames = np.isfinite(
        landmarks
    ).all(axis=(1, 2))

    valid = (
        detected.astype(bool)
        & finite_frames
    )

    valid_indices = np.flatnonzero(valid)

    empty_metrics = {
        "wrist_dx": np.nan,
        "wrist_dy": np.nan,
        "net_displacement": np.nan,
        "path_length": np.nan,
        "hand_size": np.nan,
        "normalized_displacement": np.nan,
        "normalized_path_length": np.nan,
    }

    if len(valid_indices) < 2:
        return empty_metrics

    wrist_xy = landmarks[:, 0, :2]

    first_index = int(valid_indices[0])
    last_index = int(valid_indices[-1])

    displacement_vector = (
        wrist_xy[last_index]
        - wrist_xy[first_index]
    )

    wrist_dx = float(displacement_vector[0])
    wrist_dy = float(displacement_vector[1])

    net_displacement = float(
        np.linalg.norm(displacement_vector)
    )

    valid_wrist_xy = wrist_xy[valid]

    wrist_steps = np.diff(
        valid_wrist_xy,
        axis=0,
    )

    path_length = float(
        np.linalg.norm(
            wrist_steps,
            axis=1,
        ).sum()
    )

    # 손목(0) - 중지 MCP(9) 거리를 손 크기 대리값으로 사용
    hand_sizes = np.linalg.norm(
        landmarks[valid, 9, :2]
        - landmarks[valid, 0, :2],
        axis=1,
    )

    hand_sizes = hand_sizes[
        np.isfinite(hand_sizes)
        & (hand_sizes > 1e-6)
    ]

    if len(hand_sizes) == 0:
        hand_size = np.nan
        normalized_displacement = np.nan
        normalized_path_length = np.nan
    else:
        hand_size = float(
            np.median(hand_sizes)
        )

        normalized_displacement = (
            net_displacement / hand_size
        )

        normalized_path_length = (
            path_length / hand_size
        )

    return {
        "wrist_dx": wrist_dx,
        "wrist_dy": wrist_dy,
        "net_displacement": net_displacement,
        "path_length": path_length,
        "hand_size": hand_size,
        "normalized_displacement": float(
            normalized_displacement
        ),
        "normalized_path_length": float(
            normalized_path_length
        ),
    }


# =========================================================
# MediaPipe
# =========================================================

def initialize_hand_detector():
    """
    한 개의 영상 처리를 위한 MediaPipe Hand Landmarker를 생성한다.

    중요:
    VIDEO 모드에서는 timestamp가 계속 증가해야 하므로
    여러 MP4를 일괄 처리할 때 영상마다 detector를 새로 생성한다.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nMediaPipe 모델 파일을 찾을 수 없습니다.\n"
            f"Expected: {MODEL_PATH.resolve()}\n"
            "'hand_landmarker.task' 파일을 models 폴더에 넣어주세요."
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS_TO_DETECT,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


def extract_landmarks(result):
    """
    MediaPipe 결과에서 confidence가 가장 높은 손 하나를 가져온다.

    Returns
    -------
    landmarks
        shape = (21, 3)

    world_landmarks
        shape = (21, 3)

    detected
        손 검출 여부

    handedness
        "Left", "Right", 또는 ""

    handedness_confidence
        handedness confidence
    """
    nan_landmarks = np.full(
        (21, 3),
        np.nan,
        dtype=np.float32,
    )

    if not result.hand_landmarks:
        return (
            nan_landmarks.copy(),
            nan_landmarks.copy(),
            False,
            "",
            np.nan,
        )

    def get_score(index: int) -> float:
        if (
            result.handedness
            and index < len(result.handedness)
            and result.handedness[index]
        ):
            return float(
                result.handedness[index][0].score
            )

        return 0.0

    best_index = max(
        range(len(result.hand_landmarks)),
        key=get_score,
    )

    selected_landmarks = (
        result.hand_landmarks[best_index]
    )

    landmarks = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in selected_landmarks
        ],
        dtype=np.float32,
    )

    if (
        result.hand_world_landmarks
        and best_index
        < len(result.hand_world_landmarks)
    ):
        world_landmarks = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm
                in result.hand_world_landmarks[
                    best_index
                ]
            ],
            dtype=np.float32,
        )
    else:
        world_landmarks = (
            nan_landmarks.copy()
        )

    handedness = ""
    handedness_confidence = np.nan

    if (
        result.handedness
        and best_index < len(result.handedness)
        and result.handedness[best_index]
    ):
        category = (
            result.handedness[
                best_index
            ][0]
        )

        handedness = (
            category.category_name
        )

        handedness_confidence = float(
            category.score
        )

    return (
        landmarks,
        world_landmarks,
        True,
        handedness,
        handedness_confidence,
    )


# =========================================================
# Dataset Management
# =========================================================

def get_next_sample_id(
    participant_id: str,
    label: str,
) -> int:
    """
    기존 dataset을 확인하여 다음 sample 번호를 반환한다.
    기존 NPZ를 덮어쓰지 않는다.
    """
    gesture_dir = DATASET_DIR / label

    gesture_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pattern = re.compile(
        rf"^{re.escape(participant_id)}_"
        rf"{re.escape(label)}_(\d+)\.npz$"
    )

    ids = []

    for filepath in gesture_dir.glob(
        "*.npz"
    ):
        match = pattern.match(
            filepath.name
        )

        if match:
            ids.append(
                int(match.group(1))
            )

    if not ids:
        return 1

    return max(ids) + 1


def get_dominant_handedness(
    handedness_values: list[str],
) -> str:
    """
    sequence 전체에서 가장 많이 검출된 handedness를 반환한다.
    """
    valid_values = [
        value
        for value in handedness_values
        if value
    ]

    if not valid_values:
        return "Unknown"

    return Counter(
        valid_values
    ).most_common(1)[0][0]


def update_index_csv(
    filepath: Path,
    participant_id: str,
    label: str,
    sample_id: int,
    num_frames: int,
    duration: float,
    detection_rate: float,
    handedness: str,
):
    """
    dataset/index.csv에 저장된 sample 정보를 한 행 추가한다.
    """
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = (
        DATASET_DIR / "index.csv"
    )

    fieldnames = [
        "filepath",
        "participant_id",
        "label",
        "sample_id",
        "num_frames",
        "duration",
        "detection_rate",
        "handedness",
        "created_at",
    ]

    try:
        stored_filepath = (
            filepath.resolve()
            .relative_to(PROJECT_ROOT)
            .as_posix()
        )
    except ValueError:
        stored_filepath = (
            filepath.as_posix()
        )

    row = {
        "filepath": stored_filepath,
        "participant_id": participant_id,
        "label": label,
        "sample_id": sample_id,
        "num_frames": num_frames,
        "duration": round(
            duration,
            4,
        ),
        "detection_rate": round(
            detection_rate,
            4,
        ),
        "handedness": handedness,
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
    }

    file_exists = (
        index_path.exists()
        and index_path.stat().st_size > 0
    )

    with index_path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_sequence(
    participant_id: str,
    label: str,
    sample_id: int,
    landmarks_buffer: list[np.ndarray],
    world_landmarks_buffer: list[np.ndarray],
    timestamps_buffer: list[float],
    detected_buffer: list[bool],
    handedness_buffer: list[str],
    handedness_confidence_buffer: list[float],
):
    """
    하나의 MP4에서 추출한 gesture sequence를 NPZ로 저장한다.

    기존 팀원 수집기의 NPZ 구조와 품질검사 기준을 유지한다.
    """
    num_frames = len(
        landmarks_buffer
    )

    buffer_lengths = {
        "landmarks": len(
            landmarks_buffer
        ),
        "world_landmarks": len(
            world_landmarks_buffer
        ),
        "timestamps": len(
            timestamps_buffer
        ),
        "detected": len(
            detected_buffer
        ),
        "handedness": len(
            handedness_buffer
        ),
        "handedness_confidence": len(
            handedness_confidence_buffer
        ),
    }

    if len(
        set(buffer_lengths.values())
    ) != 1:
        raise ValueError(
            "Sequence buffer lengths do not match: "
            f"{buffer_lengths}"
        )

    if num_frames < MINIMUM_FRAMES:
        print(
            "  -> Discarded: "
            f"too short ({num_frames} frames)"
        )
        return False, None

    landmarks = np.asarray(
        landmarks_buffer,
        dtype=np.float32,
    )

    world_landmarks = np.asarray(
        world_landmarks_buffer,
        dtype=np.float32,
    )

    timestamps = np.asarray(
        timestamps_buffer,
        dtype=np.float64,
    )

    detected = np.asarray(
        detected_buffer,
        dtype=np.bool_,
    )

    handedness_per_frame = (
        np.asarray(
            handedness_buffer,
            dtype="U16",
        )
    )

    handedness_confidence = (
        np.asarray(
            handedness_confidence_buffer,
            dtype=np.float32,
        )
    )

    if landmarks.shape != (
        num_frames,
        21,
        3,
    ):
        raise ValueError(
            "Invalid landmark shape: "
            f"{landmarks.shape}"
        )

    if world_landmarks.shape != (
        num_frames,
        21,
        3,
    ):
        raise ValueError(
            "Invalid world landmark shape: "
            f"{world_landmarks.shape}"
        )

    if timestamps.shape != (
        num_frames,
    ):
        raise ValueError(
            "Invalid timestamp shape: "
            f"{timestamps.shape}"
        )

    if detected.shape != (
        num_frames,
    ):
        raise ValueError(
            "Invalid detected shape: "
            f"{detected.shape}"
        )

    if not np.isfinite(
        timestamps
    ).all():
        raise ValueError(
            "Timestamps contain "
            "a non-finite value."
        )

    if (
        num_frames > 1
        and not (
            np.diff(
                timestamps
            ) > 0
        ).all()
    ):
        raise ValueError(
            "Timestamps must increase strictly."
        )

    if num_frames > 1:
        duration = float(
            timestamps[-1]
            - timestamps[0]
        )
    else:
        duration = 0.0

    detection_rate_ratio = float(
        detected.mean()
    )

    detection_rate = float(
        detection_rate_ratio * 100.0
    )

    valid_frames = int(
        detected.sum()
    )

    longest_missing_run = (
        get_longest_missing_run(
            detected
        )
    )

    dominant_handedness = (
        get_dominant_handedness(
            handedness_buffer
        )
    )

    rejection_reasons = []

    if (
        detection_rate_ratio
        < MINIMUM_DETECTION_RATE
    ):
        rejection_reasons.append(
            "detection rate is too low "
            f"({detection_rate_ratio:.1%} < "
            f"{MINIMUM_DETECTION_RATE:.1%})"
        )

    if (
        longest_missing_run
        > MAX_CONSECUTIVE_MISSING_FRAMES
    ):
        rejection_reasons.append(
            "too many consecutive "
            "missing frames "
            f"({longest_missing_run} > "
            f"{MAX_CONSECUTIVE_MISSING_FRAMES})"
        )

    if (
        duration
        < MINIMUM_DURATION_SECONDS
    ):
        rejection_reasons.append(
            "duration is too short "
            f"({duration:.2f} < "
            f"{MINIMUM_DURATION_SECONDS:.2f} sec)"
        )

    if (
        duration
        > MAXIMUM_DURATION_SECONDS
    ):
        rejection_reasons.append(
            "duration is too long "
            f"({duration:.2f} > "
            f"{MAXIMUM_DURATION_SECONDS:.2f} sec)"
        )

    if (
        dominant_handedness
        != EXPECTED_HANDEDNESS
    ):
        rejection_reasons.append(
            "unexpected handedness "
            f"({dominant_handedness} != "
            f"{EXPECTED_HANDEDNESS})"
        )

    if rejection_reasons:
        print("  -> Discarded")

        for reason in rejection_reasons:
            print(
                f"     - {reason}"
            )

        return False, None

    movement_metrics = (
        calculate_movement_metrics(
            landmarks,
            detected,
        )
    )

    normalized_displacement = (
        movement_metrics[
            "normalized_displacement"
        ]
    )

    low_motion_warning = bool(
        np.isfinite(
            normalized_displacement
        )
        and normalized_displacement
        < LOW_MOTION_WARNING_THRESHOLD
    )

    if low_motion_warning:
        print(
            "  -> Warning: "
            "normalized wrist movement is small "
            f"({normalized_displacement:.3f})"
        )

    gesture_dir = (
        DATASET_DIR / label
    )

    gesture_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{participant_id}_"
        f"{label}_"
        f"{sample_id:04d}.npz"
    )

    filepath = (
        gesture_dir / filename
    )

    if filepath.exists():
        raise FileExistsError(
            f"File already exists: {filepath}"
        )

    np.savez_compressed(
        filepath,

        landmarks=landmarks,
        timestamps=timestamps,
        detected=detected,
        label=np.array(label),
        participant_id=np.array(
            participant_id
        ),
        handedness=np.array(
            dominant_handedness
        ),
        sample_id=np.array(
            sample_id,
            dtype=np.int32,
        ),

        world_landmarks=world_landmarks,
        handedness_per_frame=(
            handedness_per_frame
        ),
        handedness_confidence=(
            handedness_confidence
        ),
        duration=np.array(
            duration,
            dtype=np.float32,
        ),
        detection_rate=np.array(
            detection_rate,
            dtype=np.float32,
        ),
        detection_rate_ratio=np.array(
            detection_rate_ratio,
            dtype=np.float32,
        ),
        valid_frames=np.array(
            valid_frames,
            dtype=np.int32,
        ),
        longest_missing_run=np.array(
            longest_missing_run,
            dtype=np.int32,
        ),

        wrist_dx=np.array(
            movement_metrics[
                "wrist_dx"
            ],
            dtype=np.float32,
        ),
        wrist_dy=np.array(
            movement_metrics[
                "wrist_dy"
            ],
            dtype=np.float32,
        ),
        net_displacement=np.array(
            movement_metrics[
                "net_displacement"
            ],
            dtype=np.float32,
        ),
        path_length=np.array(
            movement_metrics[
                "path_length"
            ],
            dtype=np.float32,
        ),
        hand_size=np.array(
            movement_metrics[
                "hand_size"
            ],
            dtype=np.float32,
        ),
        normalized_displacement=np.array(
            normalized_displacement,
            dtype=np.float32,
        ),
        normalized_path_length=np.array(
            movement_metrics[
                "normalized_path_length"
            ],
            dtype=np.float32,
        ),
        low_motion_warning=np.array(
            low_motion_warning,
            dtype=np.bool_,
        ),
        mirrored=np.array(
            False,
            dtype=np.bool_,
        ),
        collection_schema_version=np.array(
            2,
            dtype=np.int32,
        ),
    )

    update_index_csv(
        filepath=filepath,
        participant_id=participant_id,
        label=label,
        sample_id=sample_id,
        num_frames=num_frames,
        duration=duration,
        detection_rate=detection_rate,
        handedness=dominant_handedness,
    )

    print(
        "  -> Saved: "
        f"{filepath.relative_to(PROJECT_ROOT)}"
    )

    print(
        "     shape="
        f"{landmarks.shape}, "
        f"duration={duration:.2f}s, "
        f"detection={detection_rate:.1f}%, "
        f"hand={dominant_handedness}"
    )

    return True, filepath


# =========================================================
# Video Processing
# =========================================================

def find_video_files() -> list[tuple[str, Path]]:
    """
    videos/<label>/ 아래의 영상 파일을 찾는다.

    반환 형식:
        [
            ("swipe_left", Path(...)),
            ("make_fist", Path(...)),
            ...
        ]
    """
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(
            "\nvideos 폴더를 찾을 수 없습니다.\n"
            f"Expected: {VIDEOS_DIR.resolve()}"
        )

    items: list[
        tuple[str, Path]
    ] = []

    for label_dir in sorted(
        VIDEOS_DIR.iterdir()
    ):
        if not label_dir.is_dir():
            continue

        # 폴더명이 그대로 label이 되므로 검증한다.
        label = sanitize_name(
            label_dir.name
        )

        for video_path in sorted(
            label_dir.iterdir()
        ):
            if (
                video_path.is_file()
                and video_path.suffix.lower()
                in VIDEO_EXTENSIONS
            ):
                items.append(
                    (
                        label,
                        video_path,
                    )
                )

    return items


def extract_video(
    video_path: Path,
):
    """
    영상 하나를 프레임별로 읽고 MediaPipe landmark sequence를 반환한다.

    MediaPipe VIDEO mode timestamp가 영상마다 0부터 시작하므로
    detector도 영상마다 새로 만든다.
    """
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "영상 파일을 열 수 없습니다: "
            f"{video_path}"
        )

    detector = None

    try:
        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            raise RuntimeError(
                "영상 FPS를 읽을 수 없습니다: "
                f"{fps}"
            )

        detector = (
            initialize_hand_detector()
        )

        landmarks_buffer = []
        world_landmarks_buffer = []
        timestamps_buffer = []
        detected_buffer = []
        handedness_buffer = []
        handedness_confidence_buffer = []

        frame_index = 0

        while True:
            success, frame = (
                cap.read()
            )

            if not success:
                break

            relative_timestamp = (
                frame_index / fps
            )

            timestamp_ms = int(
                relative_timestamp
                * 1000
            )

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb_frame,
            )

            result = (
                detector.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            (
                landmarks,
                world_landmarks,
                detected,
                handedness,
                handedness_confidence,
            ) = extract_landmarks(
                result
            )

            landmarks_buffer.append(
                landmarks
            )

            world_landmarks_buffer.append(
                world_landmarks
            )

            timestamps_buffer.append(
                relative_timestamp
            )

            detected_buffer.append(
                detected
            )

            handedness_buffer.append(
                handedness
            )

            handedness_confidence_buffer.append(
                handedness_confidence
            )

            frame_index += 1

        return {
            "landmarks_buffer": landmarks_buffer,
            "world_landmarks_buffer": world_landmarks_buffer,
            "timestamps_buffer": timestamps_buffer,
            "detected_buffer": detected_buffer,
            "handedness_buffer": handedness_buffer,
            "handedness_confidence_buffer": handedness_confidence_buffer,
            "fps": fps,
        }

    finally:
        if detector is not None:
            detector.close()

        cap.release()


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 65)
    print("Gesture MP4 -> MediaPipe NPZ Batch Extractor")
    print("=" * 65)

    try:
        participant_id = sanitize_name(
            input(
                "Participant ID: "
            )
        )

        video_items = (
            find_video_files()
        )

    except (
        ValueError,
        FileNotFoundError,
    ) as error:
        print(error)
        return

    if not video_items:
        print()
        print(
            "videos/<label>/ 폴더에서 "
            "처리할 MP4/MOV 파일을 찾지 못했습니다."
        )
        return

    print()
    print(
        f"Participant : {participant_id}"
    )
    print(
        f"Videos      : {len(video_items)}"
    )
    print(
        "Duration    : "
        f"{MINIMUM_DURATION_SECONDS:.1f}"
        " ~ "
        f"{MAXIMUM_DURATION_SECONDS:.1f} sec"
    )
    print()

    saved_count = 0
    discarded_count = 0
    error_count = 0

    for index, (
        label,
        video_path,
    ) in enumerate(
        video_items,
        start=1,
    ):
        print(
            f"[{index}/{len(video_items)}] "
            f"{label} / {video_path.name}"
        )

        try:
            sample_id = (
                get_next_sample_id(
                    participant_id,
                    label,
                )
            )

            data = extract_video(
                video_path
            )

            saved, _ = save_sequence(
                participant_id=participant_id,
                label=label,
                sample_id=sample_id,
                landmarks_buffer=(
                    data[
                        "landmarks_buffer"
                    ]
                ),
                world_landmarks_buffer=(
                    data[
                        "world_landmarks_buffer"
                    ]
                ),
                timestamps_buffer=(
                    data[
                        "timestamps_buffer"
                    ]
                ),
                detected_buffer=(
                    data[
                        "detected_buffer"
                    ]
                ),
                handedness_buffer=(
                    data[
                        "handedness_buffer"
                    ]
                ),
                handedness_confidence_buffer=(
                    data[
                        "handedness_confidence_buffer"
                    ]
                ),
            )

            if saved:
                saved_count += 1
            else:
                discarded_count += 1

        except Exception as error:
            error_count += 1

            print(
                "  -> ERROR: "
                f"{error}"
            )

        print()

    print("=" * 65)
    print("Extraction Summary")
    print("=" * 65)
    print(
        f"Saved     : {saved_count}"
    )
    print(
        f"Discarded : {discarded_count}"
    )
    print(
        f"Errors    : {error_count}"
    )
    print(
        f"Total     : {len(video_items)}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
