from __future__ import annotations

import csv
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# Configuration
# =========================================================

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("models/hand_landmarker.task")

CAMERA_INDEX = 0

# 너무 짧은 동작 방지
MINIMUM_FRAMES = 10

# 한 손 gesture를 대상으로 하지만,
# 화면에 두 손이 들어왔을 때 confidence가 높은 손을 선택하기 위해 2로 설정
MAX_HANDS_TO_DETECT = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# MediaPipe Hand Landmark 연결 관계
HAND_CONNECTIONS = (
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),

    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (0, 17),
)


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


# =========================================================
# MediaPipe
# =========================================================

def initialize_hand_detector():
    """
    MediaPipe Hand Landmarker를 VIDEO 모드로 초기화한다.

    VIDEO 모드는 각 프레임을 동기적으로 처리하므로
    데이터 수집 시 프레임과 landmark 결과를 일관되게 맞추기 쉽다.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nMediaPipe 모델 파일을 찾을 수 없습니다.\n"
            f"Expected: {MODEL_PATH.resolve()}\n"
            f"'hand_landmarker.task' 모델을 models 폴더에 넣어주세요."
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        # 최종적으로 하나만 선택하지만 후보는 최대 2개 검출
        num_hands=MAX_HANDS_TO_DETECT,

        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    return detector


def extract_landmarks(result):
    """
    MediaPipe 결과에서 가장 handedness confidence가 높은 손 하나를 선택한다.

    Returns
    -------
    landmarks : np.ndarray
        shape = (21, 3)

    world_landmarks : np.ndarray
        shape = (21, 3)

    detected : bool

    handedness : str
        "Left", "Right", 또는 ""

    handedness_confidence : float

    selected_landmarks
        화면 visualization을 위한 MediaPipe landmark 리스트
    """

    nan_landmarks = np.full((21, 3), np.nan, dtype=np.float32)

    # 손이 검출되지 않은 경우
    if not result.hand_landmarks:
        return (
            nan_landmarks.copy(),
            nan_landmarks.copy(),
            False,
            "",
            np.nan,
            None,
        )

    # -----------------------------------------------------
    # 여러 손이 들어온 경우 handedness confidence가
    # 가장 높은 손 하나 선택
    # -----------------------------------------------------

    def get_score(index: int) -> float:
        if (
            result.handedness
            and index < len(result.handedness)
            and result.handedness[index]
        ):
            return float(result.handedness[index][0].score)

        return 0.0

    best_index = max(
        range(len(result.hand_landmarks)),
        key=get_score,
    )

    selected_landmarks = result.hand_landmarks[best_index]

    # -----------------------------------------------------
    # Image normalized landmarks
    # -----------------------------------------------------

    landmarks = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in selected_landmarks
        ],
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # World landmarks
    # -----------------------------------------------------

    if (
        result.hand_world_landmarks
        and best_index < len(result.hand_world_landmarks)
    ):
        world_landmarks = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in result.hand_world_landmarks[best_index]
            ],
            dtype=np.float32,
        )
    else:
        world_landmarks = nan_landmarks.copy()

    # -----------------------------------------------------
    # Handedness
    # -----------------------------------------------------

    handedness = ""
    handedness_confidence = np.nan

    if (
        result.handedness
        and best_index < len(result.handedness)
        and result.handedness[best_index]
    ):
        category = result.handedness[best_index][0]

        handedness = category.category_name
        handedness_confidence = float(category.score)

    return (
        landmarks,
        world_landmarks,
        True,
        handedness,
        handedness_confidence,
        selected_landmarks,
    )


# =========================================================
# Dataset Management
# =========================================================

def get_next_sample_id(
    participant_id: str,
    label: str,
) -> int:
    """
    participant와 gesture의 기존 파일 번호를 확인하여
    다음 sample 번호를 반환한다.

    기존 파일을 절대 덮어쓰지 않는다.
    """

    gesture_dir = DATASET_DIR / label
    gesture_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        rf"^{re.escape(participant_id)}_"
        rf"{re.escape(label)}_(\d+)\.npz$"
    )

    ids = []

    for filepath in gesture_dir.glob("*.npz"):

        match = pattern.match(filepath.name)

        if match:
            ids.append(int(match.group(1)))

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

    return Counter(valid_values).most_common(1)[0][0]


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

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    index_path = DATASET_DIR / "index.csv"

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

    row = {
        "filepath": filepath.as_posix(),
        "participant_id": participant_id,
        "label": label,
        "sample_id": sample_id,
        "num_frames": num_frames,
        "duration": round(duration, 4),
        "detection_rate": round(detection_rate, 4),
        "handedness": handedness,
        "created_at": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
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
    하나의 gesture sequence를 NPZ 파일로 저장한다.

    landmark의 원본 variable-length sequence를 유지한다.
    """

    num_frames = len(landmarks_buffer)

    # -----------------------------------------------------
    # 너무 짧은 sequence는 저장하지 않음
    # -----------------------------------------------------

    if num_frames < MINIMUM_FRAMES:

        print()
        print(
            f"Sequence too short. Discarded. "
            f"({num_frames} frames)"
        )
        print()

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

    handedness_per_frame = np.asarray(
        handedness_buffer,
        dtype="U16",
    )

    handedness_confidence = np.asarray(
        handedness_confidence_buffer,
        dtype=np.float32,
    )

    # 구조 검증
    if landmarks.shape != (num_frames, 21, 3):
        raise ValueError(
            f"Invalid landmark shape: {landmarks.shape}"
        )

    # -----------------------------------------------------
    # Duration
    # -----------------------------------------------------

    if num_frames > 1:
        duration = float(
            timestamps[-1] - timestamps[0]
        )
    else:
        duration = 0.0

    # -----------------------------------------------------
    # Detection Rate
    # -----------------------------------------------------

    detection_rate = float(
        detected.mean() * 100.0
    )

    dominant_handedness = get_dominant_handedness(
        handedness_buffer
    )

    # -----------------------------------------------------
    # 저장 경로
    # -----------------------------------------------------

    gesture_dir = DATASET_DIR / label
    gesture_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{participant_id}_"
        f"{label}_"
        f"{sample_id:04d}.npz"
    )

    filepath = gesture_dir / filename

    # 절대 덮어쓰지 않음
    if filepath.exists():
        raise FileExistsError(
            f"File already exists: {filepath}"
        )

    # -----------------------------------------------------
    # NPZ 저장
    # -----------------------------------------------------

    np.savez_compressed(
        filepath,

        # 필수 데이터
        landmarks=landmarks,
        timestamps=timestamps,
        detected=detected,
        label=np.array(label),
        participant_id=np.array(participant_id),
        handedness=np.array(dominant_handedness),
        sample_id=np.array(
            sample_id,
            dtype=np.int32,
        ),

        # 향후 실험을 위해 추가 저장
        world_landmarks=world_landmarks,
        handedness_per_frame=handedness_per_frame,
        handedness_confidence=handedness_confidence,
        duration=np.array(
            duration,
            dtype=np.float32,
        ),
        detection_rate=np.array(
            detection_rate,
            dtype=np.float32,
        ),
    )

    # -----------------------------------------------------
    # index.csv 추가
    # -----------------------------------------------------

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

    print()
    print("=" * 55)
    print("Saved:")
    print(filepath)
    print()
    print(f"Landmark shape : {landmarks.shape}")
    print(f"Frames         : {num_frames}")
    print(f"Duration       : {duration:.2f} sec")
    print(f"Detection rate : {detection_rate:.1f}%")
    print(f"Handedness     : {dominant_handedness}")
    print("=" * 55)
    print()

    return True, filepath


# =========================================================
# Visualization
# =========================================================

def draw_hand_landmarks(
    frame: np.ndarray,
    hand_landmarks,
):
    """
    선택된 손의 MediaPipe landmark와 연결선을 OpenCV 화면에 표시한다.
    """

    if hand_landmarks is None:
        return

    height, width = frame.shape[:2]

    points = []

    for landmark in hand_landmarks:

        x = int(
            np.clip(
                landmark.x,
                0.0,
                1.0,
            )
            * (width - 1)
        )

        y = int(
            np.clip(
                landmark.y,
                0.0,
                1.0,
            )
            * (height - 1)
        )

        points.append((x, y))

    # 뼈대 연결선
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )

    # landmark point
    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (0, 255, 0),
            -1,
        )


def draw_status(
    frame: np.ndarray,
    participant_id: str,
    label: str,
    sample_id: int,
    recording: bool,
    frame_count: int,
    detection_rate: float,
):
    """
    현재 데이터 수집 상태를 OpenCV 화면에 표시한다.
    """

    status = (
        "RECORDING"
        if recording
        else "READY"
    )

    lines = [
        f"Participant : {participant_id}",
        f"Gesture     : {label}",
        f"Sample      : {sample_id:04d}",
        f"Status      : {status}",
        f"Frames      : {frame_count}",
        f"Detection   : {detection_rate:.1f}%",
    ]

    # 상단 정보 영역
    cv2.rectangle(
        frame,
        (10, 10),
        (390, 190),
        (0, 0, 0),
        -1,
    )

    y = 35

    for text in lines:

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 25

    # RECORDING일 때 빨간 테두리
    if recording:

        height, width = frame.shape[:2]

        cv2.rectangle(
            frame,
            (3, 3),
            (width - 4, height - 4),
            (0, 0, 255),
            5,
        )

        cv2.putText(
            frame,
            "REC",
            (width - 90, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 키 안내
    height = frame.shape[0]

    cv2.putText(
        frame,
        "SPACE: Start/Stop | R: Reset | Q: Quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# =========================================================
# Camera
# =========================================================

def open_camera(index: int = 0):
    """
    Windows에서 웹캠을 연다.
    DirectShow가 실패하면 기본 backend로 재시도한다.
    """

    cap = cv2.VideoCapture(
        index,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():

        cap.release()

        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Webcam {index}을 열 수 없습니다."
        )

    return cap


# =========================================================
# Main
# =========================================================

def main():
    """
    MediaPipe Hand Landmark 기반 gesture sequence 수집 프로그램.
    """

    print("=" * 60)
    print("Gesture Time-Series Dataset Collector")
    print("=" * 60)

    try:

        participant_id = sanitize_name(
            input("Participant ID: ")
        )

        label = sanitize_name(
            input("Gesture Label: ")
        )

    except ValueError as error:

        print(error)
        return

    # dataset 폴더 생성
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    next_sample_id = get_next_sample_id(
        participant_id,
        label,
    )

    print()
    print(f"Participant : {participant_id}")
    print(f"Gesture     : {label}")
    print(f"Next Sample : {next_sample_id:04d}")
    print()

    print("Controls")
    print("SPACE : Recording Start / Stop")
    print("R     : Reset current recording")
    print("Q     : Quit")
    print()

    cap = None
    detector = None

    try:

        detector = initialize_hand_detector()
        cap = open_camera(CAMERA_INDEX)

        # -------------------------------------------------
        # Recording State
        # -------------------------------------------------

        recording = False

        landmarks_buffer = []
        world_landmarks_buffer = []
        timestamps_buffer = []
        detected_buffer = []
        handedness_buffer = []
        handedness_confidence_buffer = []

        recording_start_time = None

        # MediaPipe VIDEO timestamp용
        application_start_time = time.perf_counter()
        previous_mp_timestamp_ms = -1

        while True:

            success, frame = cap.read()

            if not success:
                print(
                    "Webcam frame을 읽을 수 없습니다."
                )
                break

            current_time = time.perf_counter()

            # -------------------------------------------------
            # MediaPipe VIDEO mode에서는 timestamp가
            # 지속적으로 증가해야 한다.
            # -------------------------------------------------

            timestamp_ms = int(
                (
                    current_time
                    - application_start_time
                )
                * 1000
            )

            # 같은 millisecond가 나오는 상황 방지
            timestamp_ms = max(
                timestamp_ms,
                previous_mp_timestamp_ms + 1,
            )

            previous_mp_timestamp_ms = timestamp_ms

            # -------------------------------------------------
            # BGR -> RGB
            # -------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # -------------------------------------------------
            # Hand landmark inference
            # -------------------------------------------------

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            (
                landmarks,
                world_landmarks,
                detected,
                handedness,
                handedness_confidence,
                selected_landmarks,
            ) = extract_landmarks(result)

            # -------------------------------------------------
            # 화면에 landmark 표시
            # -------------------------------------------------

            draw_hand_landmarks(
                frame,
                selected_landmarks,
            )

            # -------------------------------------------------
            # RECORDING 상태에서만 시계열 데이터 저장
            # -------------------------------------------------

            if recording:

                relative_timestamp = (
                    current_time
                    - recording_start_time
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

            # -------------------------------------------------
            # 현재 detection rate
            # -------------------------------------------------

            if detected_buffer:

                detection_rate = (
                    np.mean(detected_buffer)
                    * 100.0
                )

            else:

                detection_rate = 0.0

            # -------------------------------------------------
            # UI
            # -------------------------------------------------

            draw_status(
                frame=frame,
                participant_id=participant_id,
                label=label,
                sample_id=next_sample_id,
                recording=recording,
                frame_count=len(
                    landmarks_buffer
                ),
                detection_rate=detection_rate,
            )

            cv2.imshow(
                "Gesture Dataset Collector",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            # =================================================
            # SPACE : Start / Stop
            # =================================================

            if key == 32:

                # ---------------------------------------------
                # READY -> RECORDING
                # ---------------------------------------------

                if not recording:

                    landmarks_buffer.clear()
                    world_landmarks_buffer.clear()
                    timestamps_buffer.clear()
                    detected_buffer.clear()
                    handedness_buffer.clear()
                    handedness_confidence_buffer.clear()

                    recording_start_time = (
                        time.perf_counter()
                    )

                    recording = True

                    print(
                        f"[REC] "
                        f"Sample {next_sample_id:04d} "
                        f"recording started."
                    )

                # ---------------------------------------------
                # RECORDING -> SAVE
                # ---------------------------------------------

                else:

                    recording = False

                    try:

                        saved, filepath = save_sequence(
                            participant_id=participant_id,
                            label=label,
                            sample_id=next_sample_id,
                            landmarks_buffer=landmarks_buffer,
                            world_landmarks_buffer=world_landmarks_buffer,
                            timestamps_buffer=timestamps_buffer,
                            detected_buffer=detected_buffer,
                            handedness_buffer=handedness_buffer,
                            handedness_confidence_buffer=(
                                handedness_confidence_buffer
                            ),
                        )

                        if saved:
                            next_sample_id += 1

                    except Exception as error:

                        print(
                            f"[ERROR] 저장 실패: {error}"
                        )

                    # 다음 sample을 위해 초기화
                    landmarks_buffer.clear()
                    world_landmarks_buffer.clear()
                    timestamps_buffer.clear()
                    detected_buffer.clear()
                    handedness_buffer.clear()
                    handedness_confidence_buffer.clear()

                    recording_start_time = None

            # =================================================
            # R : Reset
            # =================================================

            elif key in (
                ord("r"),
                ord("R"),
            ):

                recording = False

                landmarks_buffer.clear()
                world_landmarks_buffer.clear()
                timestamps_buffer.clear()
                detected_buffer.clear()
                handedness_buffer.clear()
                handedness_confidence_buffer.clear()

                recording_start_time = None

                print(
                    "[RESET] 현재 recording을 취소했습니다."
                )

            # =================================================
            # Q : Quit
            # =================================================

            elif key in (
                ord("q"),
                ord("Q"),
            ):

                if recording:
                    print(
                        "[WARNING] Recording 중 종료하여 "
                        "현재 sample은 저장하지 않습니다."
                    )

                break

    except FileNotFoundError as error:

        print(error)

    except RuntimeError as error:

        print(error)

    except KeyboardInterrupt:

        print(
            "\nKeyboardInterrupt: 프로그램을 종료합니다."
        )

    finally:

        if detector is not None:
            detector.close()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        print("Program terminated.")


if __name__ == "__main__":
    main()