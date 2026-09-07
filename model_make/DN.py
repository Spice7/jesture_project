from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="start-stop-cancle.v2i.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        patience=15,
        name="gesture_yolo_real",
    )