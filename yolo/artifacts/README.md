# yolo/artifacts/ — 학습 산출물 보존본 (앱이 읽지 않음)

이 폴더는 **원본 `kmj02` 브랜치(정적 포즈 YOLO 담당)에서 나온 학습 run 결과**를 그대로 보존한 기록물입니다.
실행·추론에는 전혀 쓰이지 않습니다. 런타임 게이트 모델은 `models/static/v3.pt`이며, 여기 있는 가중치는
초기 실험(`gesture_yolov8n`, start-stop-cancel 데이터셋) 산출물입니다.

> 왜 남겨두나: 최종 모델이 어떤 학습 과정에서 나왔는지 보여 주는 **재현성·발표 근거**입니다.
> 실행에 불필요하므로 지워도 앱은 동작하지만, 실험 이력으로 보존합니다.

## 어떤 학습의 결과인가

`yolo/yolo_param.yaml`(train name `gesture_yolov8n`, 데이터 `data/static/start_stop_cancel/`)로 돌린
YOLOv8n 학습의 출력입니다. 최종 가중치는 `models/static/checkpoints/last_100.pt`로 복사되어 있고,
여기에는 그 학습의 **전체 로그·그래프·가중치 원본**이 들어 있습니다.

## 구성

```
kmj02/runs/yolo/
├─ gesture_yolov8n/          # 학습 run
│  ├─ weights/best.pt, last.pt      # 학습된 가중치 (원본)
│  ├─ results.csv, results.png      # epoch별 손실·지표
│  ├─ loss_plots/                   # box / cls / dfl 손실 곡선
│  ├─ Box{P,R,F1,PR}_curve.png      # precision·recall·F1·PR 곡선
│  ├─ confusion_matrix*.png         # 혼동행렬 (원본·정규화)
│  ├─ labels.jpg                    # 라벨 분포
│  ├─ train_batch*.jpg              # 학습 입력 배치 미리보기
│  ├─ val_batch*_labels/pred.jpg    # 검증 정답 vs 예측
│  └─ args.yaml                     # 학습에 쓰인 인자 (경로는 학습 당시 절대경로)
└─ val/                       # 별도 검증(val) run 출력 (곡선·혼동행렬·예측 미리보기)
```

클래스는 `start` · `stop` · `cancel` 3종(정적 포즈 게이트)입니다.
