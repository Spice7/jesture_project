from pathlib import Path

from ultralytics import YOLO

# GESTURE_VERSION = "static_gesture_v2"
# MODEL_NAME = "yolov8n_v2_null"
GESTURE_VERSION = "static_gesture_v3"
MODEL_NAME = "yolov8n_v3_rotation"

def main():
    project_root = Path(__file__).resolve().parent.parent

    data_yaml = (
        project_root
        / "datasets"
        / GESTURE_VERSION
        / "data.yaml"
    )

    best_model = (
        project_root
        / "runs"
        / "gesture"
        / MODEL_NAME
        / "weights"
        / "best.pt"
    )

    model = YOLO(str(best_model))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        device=0,
        plots=True,
        project=str(project_root / "runs" / "gesture"),
        name="yolov8n_test",
    )

    print()
    print("===== Test Result =====")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print(f"mAP50        : {metrics.box.map50:.4f}")
    print(f"mAP50-95     : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()