import cv2
import os

video_path = "./video/WIN_20260904_15_52_39_Pro.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

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