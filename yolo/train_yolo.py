from pathlib import Path

from ultralytics import YOLO

def main():
    project_root = Path(__file__).resolve().parent.parent

    data_yaml = (
        project_root
        / "datasets"
        / "static_gesture_v1"       # 정적 제스처
        / "data.yaml"
    )

    # COCO로 사전학습된 YOLOv8 Nano 모델
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,      # 입력 크기
        batch=-1,       # GPU VRAM에 맞게 batch 자동 설정       # 약 60% 사용
        device=0,       # 첫 번째 GPU
        patience=15,
        project=str(project_root / "runs" / "gesture"),
        name="yolov8n_baseline",
        plots=True,     # 평가 그래프 생성
    )

# Ultralytics 멀티프로세서 오류 방지
if __name__ == "__main__":
    main()