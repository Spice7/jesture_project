from ultralytics import YOLO
 
if __name__ == "__main__":
    # 학습된 best.pt 경로 확인 후 필요하면 수정하세요
    model = YOLO("runs/detect/gesture_yolo_real-4/weights/best.pt")
 
    # split="test" 로 지정하면 data.yaml 안의 test 경로 이미지로 평가합니다
    # (train/valid로 학습 중 이미 확인한 성능과 겹치지 않는, 완전히 안 본 데이터로 평가)
    metrics = model.val(
        data="start-stop-cancle.v2i.yolov8/data.yaml",
        split="test",
    )
 
    print("\n===== Test Result =====")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
 