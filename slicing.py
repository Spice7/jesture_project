import cv2
import os


def cap_vdieo(video_path):  
    output_dir = f"frames/{video_path}"

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
    

video_list=['WIN_20260904_15_52_39_Pro.mp4'
            'WIN_20260904_15_53_32_Pro - 복사본.mp4'
            'WIN_20260904_15_53_32_Pro.mp4'
            'WIN_20260904_15_55_08_Pro - 복사본.mp4'
            'WIN_20260904_15_55_08_Pro.mp4'
            'WIN_20260904_18_49_13_Pro - 복사본.mp4'
            'WIN_20260904_18_49_13_Pro.mp4'
            ]
for video in video_list: cap_vdieo(video)