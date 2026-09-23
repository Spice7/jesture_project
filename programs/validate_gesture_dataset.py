from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# =========================================================
# Configuration
# =========================================================

# 이 파일은 프로젝트의 programs 폴더에 두는 것을 기준으로 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dynamic"
REPORT_PATH = DATASET_DIR / "validation_report.csv"

MINIMUM_FRAMES = 20
MINIMUM_DURATION_SECONDS = 1.0
MAXIMUM_DURATION_SECONDS = 5.0
MINIMUM_DETECTION_RATE = 0.8
MAX_CONSECUTIVE_MISSING_FRAMES = 5
EXPECTED_HANDEDNESS = "Right"

# 아래 기준은 FAIL이 아니라 WARNING 판정용 휴리스틱이다.
MAKE_FIST_MAX_LATE_TO_EARLY_RATIO = 0.85
SWIPE_LEFT_MIN_NORMALIZED_DISPLACEMENT = 0.50
NO_GESTURE_MAX_NORMALIZED_DISPLACEMENT = 0.50
NO_GESTURE_MAX_NORMALIZED_PATH_LENGTH = 1.50

WINDOW_RATIO = 0.20

REQUIRED_KEYS = {
    "landmarks",
    "timestamps",
    "detected",
    "label",
    "participant_id",
    "handedness",
    "sample_id",
    "duration",
    "detection_rate",
}


# =========================================================
# Helpers
# =========================================================

def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape={arr.shape}")
    return str(arr.reshape(-1)[0])


def scalar_float(value) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def scalar_int(value) -> int:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape={arr.shape}")
    return int(arr.reshape(-1)[0])


def scalar_bool(data, key: str, default: bool = False) -> bool:
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size != 1:
        return default
    return bool(arr.reshape(-1)[0])


def longest_missing_run(detected: np.ndarray) -> int:
    longest = 0
    current = 0

    for value in detected:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)

    return longest


def valid_frame_mask(
    landmarks: np.ndarray,
    detected: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(landmarks).all(axis=(1, 2))
    return detected.astype(bool) & finite


def hand_scale_per_frame(
    landmarks: np.ndarray,
) -> np.ndarray:
    # 손목(0) - 중지 MCP(9) 거리
    return np.linalg.norm(
        landmarks[:, 9, :2] - landmarks[:, 0, :2],
        axis=1,
    )


def wrist_metrics(
    landmarks: np.ndarray,
    detected: np.ndarray,
) -> tuple[float, float, float]:
    """
    wrist_dx, normalized displacement, normalized path length
    """
    valid = valid_frame_mask(landmarks, detected)
    indices = np.flatnonzero(valid)

    if len(indices) < 2:
        return math.nan, math.nan, math.nan

    first = int(indices[0])
    last = int(indices[-1])

    wrist = landmarks[:, 0, :2]
    displacement = wrist[last] - wrist[first]

    wrist_dx = float(displacement[0])
    net = float(np.linalg.norm(displacement))

    valid_wrist = wrist[valid]
    steps = np.diff(valid_wrist, axis=0)
    path = float(np.linalg.norm(steps, axis=1).sum())

    scales = hand_scale_per_frame(landmarks)
    scale_values = scales[
        valid & np.isfinite(scales) & (scales > 1e-6)
    ]

    if len(scale_values) == 0:
        return wrist_dx, math.nan, math.nan

    hand_scale = float(np.median(scale_values))

    return (
        wrist_dx,
        net / hand_scale,
        path / hand_scale,
    )


def make_fist_metric(
    landmarks: np.ndarray,
    detected: np.ndarray,
) -> tuple[float, float, float]:
    """
    초반/후반의 손가락 TIP-손목 거리를 손 크기로 정규화하여 비교한다.

    make_fist라면 일반적으로:
        late_score < early_score

    ratio = late_score / early_score
    값이 충분히 1보다 작을 것을 기대한다.
    """
    valid = valid_frame_mask(landmarks, detected)
    tips = [8, 12, 16, 20]

    wrist = landmarks[:, 0, :2]

    distances = np.stack(
        [
            np.linalg.norm(
                landmarks[:, tip, :2] - wrist,
                axis=1,
            )
            for tip in tips
        ],
        axis=1,
    )

    mean_tip_distance = distances.mean(axis=1)
    scale = hand_scale_per_frame(landmarks)

    usable = (
        valid
        & np.isfinite(mean_tip_distance)
        & np.isfinite(scale)
        & (scale > 1e-6)
    )

    indices = np.flatnonzero(usable)

    if len(indices) < 5:
        return math.nan, math.nan, math.nan

    score = mean_tip_distance / scale

    window_size = max(
        1,
        int(len(indices) * WINDOW_RATIO),
    )

    early_indices = indices[:window_size]
    late_indices = indices[-window_size:]

    early = float(np.median(score[early_indices]))
    late = float(np.median(score[late_indices]))

    if not np.isfinite(early) or early <= 1e-6:
        ratio = math.nan
    else:
        ratio = float(late / early)

    return early, late, ratio


# =========================================================
# Validation
# =========================================================

def validate_npz(filepath: Path) -> dict[str, object]:
    failures: list[str] = []
    warnings: list[str] = []

    row: dict[str, object] = {
        "filepath": filepath.relative_to(PROJECT_ROOT).as_posix(),
        "folder_label": filepath.parent.name,
        "stored_label": "",
        "participant_id": "",
        "sample_id": "",
        "frames": "",
        "duration": "",
        "detection_rate": "",
        "longest_missing_run": "",
        "handedness": "",
        "mirrored": "",
        "normalized_displacement": "",
        "normalized_path_length": "",
        "make_fist_early_score": "",
        "make_fist_late_score": "",
        "make_fist_ratio": "",
        "status": "",
        "reasons": "",
    }

    try:
        with np.load(filepath, allow_pickle=False) as data:
            missing = sorted(REQUIRED_KEYS - set(data.files))

            if missing:
                failures.append(
                    "missing keys: " + ", ".join(missing)
                )
                row["status"] = "FAIL"
                row["reasons"] = " | ".join(failures)
                return row

            landmarks = np.asarray(
                data["landmarks"],
                dtype=np.float32,
            )
            timestamps = np.asarray(
                data["timestamps"],
                dtype=np.float64,
            )
            detected = np.asarray(
                data["detected"],
                dtype=np.bool_,
            )

            label = scalar_str(data["label"])
            participant_id = scalar_str(data["participant_id"])
            handedness = scalar_str(data["handedness"])
            sample_id = scalar_int(data["sample_id"])
            stored_duration = scalar_float(data["duration"])
            stored_detection_rate = scalar_float(
                data["detection_rate"]
            )
            mirrored = scalar_bool(
                data,
                "mirrored",
                False,
            )

            row["stored_label"] = label
            row["participant_id"] = participant_id
            row["sample_id"] = sample_id
            row["handedness"] = handedness
            row["mirrored"] = mirrored

            # -------------------------------------------------
            # 구조 검사
            # -------------------------------------------------

            if (
                landmarks.ndim != 3
                or landmarks.shape[1:] != (21, 3)
            ):
                failures.append(
                    f"invalid landmarks shape {landmarks.shape}"
                )
                num_frames = (
                    landmarks.shape[0]
                    if landmarks.ndim >= 1
                    else 0
                )
            else:
                num_frames = landmarks.shape[0]

            row["frames"] = num_frames

            if timestamps.shape != (num_frames,):
                failures.append(
                    f"timestamps length mismatch {timestamps.shape}"
                )

            if detected.shape != (num_frames,):
                failures.append(
                    f"detected length mismatch {detected.shape}"
                )

            if num_frames < MINIMUM_FRAMES:
                failures.append(
                    f"too few frames ({num_frames} < {MINIMUM_FRAMES})"
                )

            if (
                timestamps.shape == (num_frames,)
                and num_frames > 0
            ):
                if not np.isfinite(timestamps).all():
                    failures.append(
                        "timestamps contain non-finite values"
                    )

                if (
                    num_frames > 1
                    and not (np.diff(timestamps) > 0).all()
                ):
                    failures.append(
                        "timestamps are not strictly increasing"
                    )

            # -------------------------------------------------
            # metadata 검사
            # -------------------------------------------------

            folder_label = filepath.parent.name

            if label != folder_label:
                failures.append(
                    "stored label does not match folder "
                    f"({label} != {folder_label})"
                )

            expected_filename = (
                f"{participant_id}_{label}_{sample_id:04d}.npz"
            )

            if filepath.name != expected_filename:
                warnings.append(
                    "filename does not match metadata "
                    f"(expected {expected_filename})"
                )

            if handedness != EXPECTED_HANDEDNESS:
                failures.append(
                    "unexpected handedness "
                    f"({handedness} != {EXPECTED_HANDEDNESS})"
                )

            # -------------------------------------------------
            # duration
            # -------------------------------------------------

            calculated_duration = math.nan

            if (
                timestamps.shape == (num_frames,)
                and num_frames > 1
                and np.isfinite(timestamps).all()
            ):
                calculated_duration = float(
                    timestamps[-1] - timestamps[0]
                )
                row["duration"] = round(
                    calculated_duration,
                    4,
                )

                if calculated_duration < MINIMUM_DURATION_SECONDS:
                    failures.append(
                        f"duration too short ({calculated_duration:.2f}s)"
                    )

                if calculated_duration > MAXIMUM_DURATION_SECONDS:
                    failures.append(
                        f"duration too long ({calculated_duration:.2f}s)"
                    )

                if abs(
                    calculated_duration - stored_duration
                ) > 0.05:
                    warnings.append(
                        "stored duration differs from timestamps "
                        f"({stored_duration:.3f} vs "
                        f"{calculated_duration:.3f})"
                    )

            # -------------------------------------------------
            # detection
            # -------------------------------------------------

            if detected.shape == (num_frames,):
                detection_ratio = (
                    float(detected.mean())
                    if num_frames > 0
                    else 0.0
                )
                detection_rate = detection_ratio * 100.0

                row["detection_rate"] = round(
                    detection_rate,
                    2,
                )

                if detection_ratio < MINIMUM_DETECTION_RATE:
                    failures.append(
                        "detection rate too low "
                        f"({detection_rate:.1f}%)"
                    )

                missing_run = longest_missing_run(detected)

                row["longest_missing_run"] = missing_run

                if (
                    missing_run
                    > MAX_CONSECUTIVE_MISSING_FRAMES
                ):
                    failures.append(
                        "too many consecutive missing frames "
                        f"({missing_run})"
                    )

                if abs(
                    detection_rate - stored_detection_rate
                ) > 0.1:
                    warnings.append(
                        "stored detection rate differs "
                        f"({stored_detection_rate:.2f} vs "
                        f"{detection_rate:.2f})"
                    )

            # -------------------------------------------------
            # landmark finite 검사
            # -------------------------------------------------

            valid_structure = (
                landmarks.ndim == 3
                and landmarks.shape[1:] == (21, 3)
                and detected.shape == (num_frames,)
                and num_frames > 0
            )

            if valid_structure:
                if detected.any():
                    if not np.isfinite(
                        landmarks[detected]
                    ).all():
                        failures.append(
                            "detected frame contains NaN/Inf landmarks"
                        )

                if not np.isfinite(landmarks).any():
                    failures.append(
                        "all landmark values are non-finite"
                    )

            # -------------------------------------------------
            # 라벨별 휴리스틱
            # -------------------------------------------------

            if valid_structure:
                (
                    wrist_dx,
                    normalized_displacement,
                    normalized_path_length,
                ) = wrist_metrics(
                    landmarks,
                    detected,
                )

                if np.isfinite(normalized_displacement):
                    row["normalized_displacement"] = round(
                        normalized_displacement,
                        4,
                    )

                if np.isfinite(normalized_path_length):
                    row["normalized_path_length"] = round(
                        normalized_path_length,
                        4,
                    )

                if label == "make_fist":
                    early, late, ratio = make_fist_metric(
                        landmarks,
                        detected,
                    )

                    if np.isfinite(early):
                        row["make_fist_early_score"] = round(
                            early,
                            4,
                        )

                    if np.isfinite(late):
                        row["make_fist_late_score"] = round(
                            late,
                            4,
                        )

                    if np.isfinite(ratio):
                        row["make_fist_ratio"] = round(
                            ratio,
                            4,
                        )

                        if (
                            ratio
                            > MAKE_FIST_MAX_LATE_TO_EARLY_RATIO
                        ):
                            warnings.append(
                                "make_fist finger-tip contraction "
                                "is small "
                                f"(late/early={ratio:.3f})"
                            )
                    else:
                        warnings.append(
                            "could not calculate make_fist metric"
                        )

                elif label == "swipe_left":
                    if not np.isfinite(normalized_displacement):
                        warnings.append(
                            "could not calculate swipe displacement"
                        )

                    elif (
                        normalized_displacement
                        < SWIPE_LEFT_MIN_NORMALIZED_DISPLACEMENT
                    ):
                        warnings.append(
                            "swipe displacement is small "
                            f"({normalized_displacement:.3f})"
                        )

                    # 현재 프로젝트 기준:
                    # mirrored=False일 때 수행자의 왼쪽 이동은
                    # 이미지 x 증가 방향으로 나타나는 것을 기대한다.
                    if np.isfinite(wrist_dx):
                        expected_positive = not mirrored

                        wrong_direction = (
                            expected_positive and wrist_dx <= 0
                        ) or (
                            (not expected_positive)
                            and wrist_dx >= 0
                        )

                        if wrong_direction:
                            warnings.append(
                                "swipe_left direction does not "
                                "match expected image-x direction "
                                f"(wrist_dx={wrist_dx:.4f}, "
                                f"mirrored={mirrored})"
                            )

                elif label == "no_gesture":
                    if (
                        np.isfinite(normalized_displacement)
                        and normalized_displacement
                        > NO_GESTURE_MAX_NORMALIZED_DISPLACEMENT
                    ):
                        warnings.append(
                            "no_gesture net wrist movement "
                            "is relatively large "
                            f"({normalized_displacement:.3f})"
                        )

                    if (
                        np.isfinite(normalized_path_length)
                        and normalized_path_length
                        > NO_GESTURE_MAX_NORMALIZED_PATH_LENGTH
                    ):
                        warnings.append(
                            "no_gesture wrist path length "
                            "is relatively large "
                            f"({normalized_path_length:.3f})"
                        )

            # -------------------------------------------------
            # 최종 판정
            # -------------------------------------------------

            if failures:
                status = "FAIL"
                reasons = failures + [
                    "WARNING: " + warning
                    for warning in warnings
                ]
            elif warnings:
                status = "WARNING"
                reasons = warnings
            else:
                status = "PASS"
                reasons = []

            row["status"] = status
            row["reasons"] = " | ".join(reasons)

            return row

    except Exception as error:
        row["status"] = "FAIL"
        row["reasons"] = (
            "could not read/validate file: "
            f"{type(error).__name__}: {error}"
        )
        return row


# =========================================================
# Dataset scan / report
# =========================================================

def find_npz_files() -> list[Path]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "dataset 폴더를 찾을 수 없습니다: "
            f"{DATASET_DIR.resolve()}"
        )

    return sorted(
        path
        for path in DATASET_DIR.rglob("*.npz")
        if path.is_file()
    )


def write_report(
    rows: list[dict[str, object]],
):
    if not rows:
        return

    fieldnames = [
        "filepath",
        "folder_label",
        "stored_label",
        "participant_id",
        "sample_id",
        "frames",
        "duration",
        "detection_rate",
        "longest_missing_run",
        "handedness",
        "mirrored",
        "normalized_displacement",
        "normalized_path_length",
        "make_fist_early_score",
        "make_fist_late_score",
        "make_fist_ratio",
        "status",
        "reasons",
    ]

    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: list[dict[str, object]],
):
    total = Counter(
        str(row["status"])
        for row in rows
    )

    by_label = defaultdict(Counter)

    for row in rows:
        by_label[
            str(row["folder_label"])
        ][
            str(row["status"])
        ] += 1

    print()
    print("=" * 72)
    print("Validation Summary")
    print("=" * 72)
    print(f"Total   : {len(rows)}")
    print(f"PASS    : {total['PASS']}")
    print(f"WARNING : {total['WARNING']}")
    print(f"FAIL    : {total['FAIL']}")
    print()

    print("By label")

    for label in sorted(by_label):
        counter = by_label[label]

        print(
            f"- {label}: "
            f"PASS={counter['PASS']}, "
            f"WARNING={counter['WARNING']}, "
            f"FAIL={counter['FAIL']}"
        )

    print()
    print(
        "Report  : "
        f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 72)


def main():
    print("=" * 72)
    print("Gesture NPZ Dataset Validator")
    print("=" * 72)

    try:
        npz_files = find_npz_files()
    except FileNotFoundError as error:
        print(error)
        return

    if not npz_files:
        print("검사할 NPZ 파일을 찾지 못했습니다.")
        return

    print(f"NPZ files: {len(npz_files)}")
    print()

    rows = []

    for index, filepath in enumerate(
        npz_files,
        start=1,
    ):
        row = validate_npz(filepath)
        rows.append(row)

        print(
            f"[{index:03d}/{len(npz_files):03d}] "
            f"{row['status']:<7} "
            f"{row['filepath']}"
        )

        if row["status"] != "PASS":
            print(
                f"            {row['reasons']}"
            )

    write_report(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
