import cv2
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cap_vdieo(video_path, video_dir=ROOT / "videos", frames_dir=ROOT / "frames"):
    output_dir = Path(frames_dir) / video_path

    os.makedirs(output_dir, exist_ok=True)

    total_path = str(Path(video_dir) / video_path)
    cap = cv2.VideoCapture(total_path)

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        save_path = os.path.join(
            output_dir,
            f"frame_{frame_idx:06d}.jpg"
        )

        cv2.imwrite(save_path, frame)

        frame_idx += 1

    cap.release()

    print(f"총 {frame_idx}장의 이미지 저장 완료")


def main():
    parser = argparse.ArgumentParser(description="폴더 안 영상의 모든 프레임 추출")
    parser.add_argument("--input", type=Path, default=ROOT / "videos")
    parser.add_argument("--output", type=Path, default=ROOT / "frames")
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not source.is_dir():
        raise FileNotFoundError(f"영상 폴더가 없습니다: {source}")
    for video in sorted(source.iterdir()):
        if video.is_file() and video.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            cap_vdieo(video.name, source, output)


if __name__ == "__main__":
    main()
