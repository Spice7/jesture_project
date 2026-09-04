import cv2
import os


def cap_vdieo(video_path):  
    output_dir = f"frames/{video_path}"

    os.makedirs(output_dir, exist_ok=True)

    total_path = os.path.join('./video', video_path)
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
    

video_list = os.listdir('./video')
for video in video_list: cap_vdieo(video)