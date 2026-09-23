import cv2
from pathlib import Path


# ========================================
# 설정
# ========================================

VIDEO_SUBJECT = "null"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "videos"
SAVE_DIR = PROJECT_ROOT / "images" / VIDEO_SUBJECT

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
INTERVAL_SECONDS = 0.5


# ========================================
# 다음 영상 번호 찾기
# ========================================

def get_next_video_path():
    index = 0

    while True:
        video_path = (
            VIDEO_DIR
            / f"{VIDEO_SUBJECT}_{index:04d}.mp4"
        )

        if not video_path.exists():
            return video_path

        index += 1


# ========================================
# 다음 이미지 번호 찾기
# ========================================

def get_next_image_index():
    image_files = SAVE_DIR.glob(
        f"{VIDEO_SUBJECT}_*.jpg"
    )

    indices = []

    for image_file in image_files:
        try:
            index = int(
                image_file.stem.split("_")[-1]
            )

            indices.append(index)

        except ValueError:
            continue

    if not indices:
        return 0

    return max(indices) + 1


# ========================================
# 현재 이미지 개수
# ========================================

def get_image_count():
    return len(
        list(
            SAVE_DIR.glob(
                f"{VIDEO_SUBJECT}_*.jpg"
            )
        )
    )


# ========================================
# 촬영 + 이미지 추출
# ========================================

def record_and_extract():
    video_path = get_next_video_path()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없습니다."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    # FPS를 정상적으로 가져오지 못한 경우
    if fps <= 0:
        fps = 30.0

    # 영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "영상 저장 파일을 생성할 수 없습니다."
        )

    # 0.5초마다 몇 프레임인지 계산
    frame_interval = max(
        1,
        int(fps * INTERVAL_SECONDS)
    )

    frame_count = 0

    # 이미지 파일명에 사용할 다음 번호
    save_index = get_next_image_index()

    # 실제 open_hand 폴더 안 이미지 개수
    total_image_count = get_image_count()

    # 이번 촬영에서 저장한 이미지 개수
    session_image_count = 0

    print("=" * 40)
    print(f"Subject: {VIDEO_SUBJECT}")
    print(f"현재 이미지: {total_image_count}장")
    print(f"저장 영상: {video_path}")
    print("Q를 누르면 촬영을 종료합니다.")
    print("=" * 40)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # --------------------------------
        # 원본 영상 저장
        # --------------------------------

        # 텍스트를 그리기 전에 저장하므로
        # 영상에는 안내 문구가 들어가지 않습니다.
        writer.write(frame)

        # --------------------------------
        # 0.5초마다 이미지 저장
        # --------------------------------

        if frame_count % frame_interval == 0:

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # 가운데 기준 최대 정사각형
            size = min(
                frame_width,
                frame_height
            )

            x1 = (
                frame_width - size
            ) // 2

            y1 = (
                frame_height - size
            ) // 2

            square = frame[
                y1:y1 + size,
                x1:x1 + size
            ]

            # 640 x 640 resize
            square = cv2.resize(
                square,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            file_path = (
                SAVE_DIR
                / f"{VIDEO_SUBJECT}_{save_index:04d}.jpg"
            )

            success = cv2.imwrite(
                str(file_path),
                square
            )

            if success:
                save_index += 1
                total_image_count += 1
                session_image_count += 1

        # --------------------------------
        # 화면 표시용 복사본
        # --------------------------------

        preview = frame.copy()

        cv2.putText(
            preview,
            f"Subject: {VIDEO_SUBJECT}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            preview,
            f"Images: {total_image_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            preview,
            f"Session: {session_image_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            preview,
            "Q: Quit",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Gesture Dataset Capture",
            preview
        )

        # --------------------------------
        # 종료
        # --------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frame_count += 1

    # ====================================
    # 정리
    # ====================================

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 40)
    print("촬영 종료")
    print(f"영상 저장: {video_path}")
    print(
        f"이번 촬영 이미지: "
        f"{session_image_count}장"
    )
    print(
        f"{VIDEO_SUBJECT} 전체 이미지: "
        f"{total_image_count}장"
    )
    print("=" * 40)


# ========================================
# 실행
# ========================================

if __name__ == "__main__":
    record_and_extract()