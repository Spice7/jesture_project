#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py
=======
Jester 데이터셋의 4번 영상(video_id=4)에 MediaPipe Hand Landmarker를
적용해보는 간단한 실행 스크립트.

check_jester_mediapipe.py 의 백엔드/전처리 유틸을 그대로 재사용한다.
"""

from pathlib import Path

import cv2

import check_jester_mediapipe as C

FRAMES_ROOT = Path("data/jester/20bn-jester-v1")
VIDEO_ID = "4"
OUT_DIR = Path("output") / f"video_{VIDEO_ID}_overlay"
SOURCE_FPS = 12.0  # Jester 원본 fps


def main():
    model_path = C.ensure_model(C.DEFAULT_MODEL_PATH)

    video_dir = FRAMES_ROOT / VIDEO_ID
    frames = C.read_frames(video_dir, "*.jpg", max_frames=0)
    if not frames:
        raise SystemExit(f"[오류] {video_dir} 에 프레임이 없습니다.")

    backend = C.make_backend(model_path, mode="video", det_conf=0.3,
                             presence_conf=0.3, track_conf=0.3, num_hands=2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fps_ms = int(round(1000 / SOURCE_FPS))

    n_detected = 0
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = C.preprocess(img, scale=2.0, method="none")
        res = backend.detect(img, timestamp_ms=i * fps_ms)
        if res.detected:
            n_detected += 1
            C._save_overlay(OUT_DIR / fp.name, img, res)

    backend.close()

    print(f"[영상 {VIDEO_ID}] 총 프레임 {len(frames)}개 중 "
          f"{n_detected}개 검출 (검출률 {n_detected / len(frames):.1%})")
    print(f"[결과] 오버레이 이미지 저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
