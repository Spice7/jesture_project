from ultralytics import YOLO
 
# 학습 끝난 뒤 생성된 best.pt 경로로 수정하세요
# (runs/detect/gesture_yolo_real/weights/best.pt)
model = YOLO("last_48.pt")
 
# source=0 은 기본 웹캠을 의미합니다 (외장 캠이면 1, 2 등으로 바꿔보세요)
# show=True 를 주면 실시간으로 바운딩박스가 그려진 화면이 뜹니다
# 종료하려면 화면을 클릭한 상태에서 'q' 키를 누르세요
model.predict(
    source=0,
    show=True,
    conf=0.5,       # 이 확신도(50%) 이상일 때만 박스를 표시
    imgsz=640,
)