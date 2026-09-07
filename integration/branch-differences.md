# Branch differences before integration

Status is relative to origin/dev; A added, M modified, D deleted, R renamed.

## lkh

Merge base: 2adca814bc236a0f16ccb3500d9b9df192446282

```text
0	8
46cea52 학습 테스트
d6bb0fb 미검출 허용 시간 추가
2d73197 yolo 모델 학습 및 분석
7d103bc yolo 모델 사용 모듈화
a047b51 roboflow 데이터 학습
242f59b 시계열 데이터 추출
c389d25 gitignore 최신화
f321fe2 모션 데이터 생성중
A	gesture_model/__init__.py
A	gesture_model/gesture_detector.py
A	programs/collect_gesture.py
A	programs/collect_gesture_video.py
A	programs/extract_gesture_videos.py
A	programs/flip_image.py
A	programs/validate_gesture_dataset.py
A	util/slice.py
A	util/valid_npz.py
A	yolo/evaluate_yolo.py
A	yolo/test_yolo_webcam.py
A	yolo/train_yolo.py
```

## hwangsoon

Merge base: dbd70160b020c2916b66569db5bec580e0a95a06

```text
3	17
23cc75f 설정 문구: 확신도 문턱 → 인식 민감도(최소 확신도)
9161883 키 매핑 녹화 안정화: 창 전체 키 입력, 보조키 단독 저장 방지, 이중 저장 제거
494a08e 시연 UI 2차(customtkinter), YOLO 정적 포즈 게이트 연결, 키 매핑 UX 개편, 도움말·사용 설명서, GRU vs LSTM 비교 자료, README 실행 안내
0a1d4f0 시연용 서비스 UI 1차: 카메라 화면, 제스처별 키 매핑 편집·프리셋, 실행 기록, YOLO 게이트 연결 자리
a8d0467 창 전환 키 매핑(주먹 Ctrl+Alt+Tab, 스와이프 Tab, 스냅 Enter), 감지기 최소 길이 0.25초, 재생 시험 배속 옵션
fb1b9c0 4클래스 모델 성능 정리 문서와 5명 LOO 비교 결과 추가
a831692 팀 전체 데이터 반영: 스냅 기준 재보정, 스와이프 조기 판정·손 놓침 대기·짧은 스와이프 증강, 쿨다운 0.5
6b84b1a 동적 제스처 finger_snap 추가, 스냅 조기 판정, 명령 뒤 손 내리기 오인 대책
f3a99bd Windows 에서 torch/torchvision 을 CUDA 12.6 빌드로 받도록 변경
b0236cb 실시간 제스처 인식 루프, PyTorch GRU 이식, 4인 비교 실험 기록 추가
1861d58 why_not_jester: 폐기한 Jester 코드 복구 절차 → 요약으로 대체 (옛 브랜치 삭제)
e0c727b 학습 파이프라인을 JIN 베이스 위로 이식 (GRU 기본)
44fc036 지시사항 추가
81eb44f 지시사항 수정
abc0023 제스처 데이터 수집기 정리 및 수집 데이터 추가
53f10c3 mediapipe 코드
49d7cc6 테스트파일 추가
M	.gitignore
A	README.md
A	dataset/index.csv
A	dataset/swipe_left/user00_swipe_left_0001.npz
A	dataset/swipe_left/user00_swipe_left_0002.npz
A	dataset/swipe_left/user00_swipe_left_0003.npz
A	dataset/swipe_left/user00_swipe_left_0004.npz
A	dataset/swipe_left/user00_swipe_left_0005.npz
A	dataset/swipe_left/user00_swipe_left_0006.npz
A	dataset/swipe_left/user00_swipe_left_0007.npz
A	dataset/swipe_left/user00_swipe_left_0008.npz
A	dataset/swipe_left/user00_swipe_left_0009.npz
A	dataset/swipe_left/user00_swipe_left_0010.npz
A	dataset/swipe_left/user00_swipe_left_0011.npz
A	dataset/swipe_left/user00_swipe_left_0012.npz
A	dataset/swipe_left/user00_swipe_left_0013.npz
A	dataset/swipe_left/user00_swipe_left_0014.npz
A	dataset/swipe_left/user00_swipe_left_0015.npz
A	dataset/swipe_left/user00_swipe_left_0016.npz
A	dataset/swipe_left/user00_swipe_left_0017.npz
A	dataset/swipe_left/user00_swipe_left_0018.npz
A	dataset/swipe_left/user00_swipe_left_0019.npz
A	dataset/swipe_left/user00_swipe_left_0020.npz
A	dataset/swipe_left/user00_swipe_left_0021.npz
A	dataset/swipe_left/user00_swipe_left_0022.npz
A	dataset/swipe_left/user00_swipe_left_0023.npz
A	dataset/swipe_left/user00_swipe_left_0024.npz
A	dataset/swipe_left/user00_swipe_left_0025.npz
A	dataset/swipe_left/user00_swipe_left_0026.npz
A	dataset/swipe_left/user00_swipe_left_0027.npz
A	dataset/swipe_left/user00_swipe_left_0028.npz
A	dataset/swipe_left/user00_swipe_left_0029.npz
A	dataset/swipe_left/user00_swipe_left_0030.npz
A	dataset/swipe_left/user00_swipe_left_0031.npz
A	dataset/swipe_left/user00_swipe_left_0032.npz
A	dataset/swipe_left/user00_swipe_left_0033.npz
A	dataset/swipe_left/user00_swipe_left_0034.npz
A	dataset/swipe_left/user00_swipe_left_0035.npz
A	dataset/swipe_left/user00_swipe_left_0036.npz
A	dataset/swipe_left/user00_swipe_left_0037.npz
A	docs/experiment_report_0905.md
A	docs/gesture_app.md
A	docs/model_metrics_0907.md
A	docs/setup_windows.md
A	docs/team_conventions.md
A	docs/user_guide.md
A	docs/why_not_jester.md
A	gesture/__init__.py
A	gesture/actions.py
A	gesture/baseline.py
A	gesture/config.py
A	gesture/dataset.py
A	gesture/gate.py
A	gesture/landmarks.py
A	gesture/model.py
A	gesture/pipeline.py
A	gesture/preprocess.py
A	gesture/presets.py
A	gesture/sanity.py
A	gesture/segmenter.py
A	gesture/training.py
A	gesture_model/__init__.py
A	gesture_model/gesture_detector.py
A	gestures.json
D	main.py
A	models/hand_landmarker.task
A	programs/README.md
A	programs/check.py
A	programs/collect_gesture.py
M	pyproject.toml
A	reports/compare_models.csv
A	reports/compare_models_summary.md
A	reports/gru_vs_lstm_0907/fig1_accuracy_f1.png
A	reports/gru_vs_lstm_0907/fig2_per_class_recall.png
A	reports/gru_vs_lstm_0907/fig3_confusion_test_p006.png
A	reports/gru_vs_lstm_0907/fig3b_confusion_val_p004.png
A	reports/gru_vs_lstm_0907/fig4_params_latency.png
A	reports/gru_vs_lstm_0907/fig5_training_curves.png
A	reports/gru_vs_lstm_0907/fig_system_flow.png
A	reports/gru_vs_lstm_0907/numbers.json
A	reports/gru_vs_lstm_0907/raw/bench_infer.json
A	reports/gru_vs_lstm_0907/raw/jin_ckpt_scores.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/prepare_report.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed0/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed0/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed1/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed1/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed2/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed2/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed42/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed42/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_on_ours_scores.json
A	reports/gru_vs_lstm_0907/raw/ours_results.json
A	reports/gru_vs_lstm_0907/scripts/bench_infer.py
A	reports/gru_vs_lstm_0907/scripts/make_report.py
A	reports/gru_vs_lstm_0907/scripts/run_ours.py
A	reports/gru_vs_lstm_0907/scripts/score_jin_ckpt.py
A	reports/gru_vs_lstm_0907/summary.md
A	reports/gru_vs_lstm_0907/table_main.md
A	reports/loo/test_p002/compare_models.csv
A	reports/loo/test_p002/compare_models_summary.md
A	reports/loo/test_p003/compare_models.csv
A	reports/loo/test_p003/compare_models_summary.md
A	reports/loo/test_p004/compare_models.csv
A	reports/loo/test_p004/compare_models_summary.md
A	reports/loo/test_p006/compare_models.csv
A	reports/loo/test_p006/compare_models_summary.md
A	reports/loo_4cls/test_p001/compare_models.csv
A	reports/loo_4cls/test_p001/compare_models_summary.md
A	reports/loo_4cls/test_p002/compare_models.csv
A	reports/loo_4cls/test_p002/compare_models_summary.md
A	reports/loo_4cls/test_p003/compare_models.csv
A	reports/loo_4cls/test_p003/compare_models_summary.md
A	reports/loo_4cls/test_p004/compare_models.csv
A	reports/loo_4cls/test_p004/compare_models_summary.md
A	reports/loo_4cls/test_p006/compare_models.csv
A	reports/loo_4cls/test_p006/compare_models_summary.md
A	reports/loo_none/test_p002/compare_models.csv
A	reports/loo_none/test_p002/compare_models_summary.md
A	reports/loo_none/test_p003/compare_models.csv
A	reports/loo_none/test_p003/compare_models_summary.md
A	reports/loo_none/test_p004/compare_models.csv
A	reports/loo_none/test_p004/compare_models_summary.md
A	reports/loo_none/test_p006/compare_models.csv
A	reports/loo_none/test_p006/compare_models_summary.md
A	reports/loo_revfist/test_p002/compare_models.csv
A	reports/loo_revfist/test_p002/compare_models_summary.md
A	reports/loo_revfist/test_p003/compare_models.csv
A	reports/loo_revfist/test_p003/compare_models_summary.md
A	reports/loo_revfist/test_p004/compare_models.csv
A	reports/loo_revfist/test_p004/compare_models_summary.md
A	reports/loo_revfist/test_p006/compare_models.csv
A	reports/loo_revfist/test_p006/compare_models_summary.md
A	scripts/compare_models.py
A	scripts/dataset_summary.py
A	scripts/eval_model.py
A	scripts/gesture_app.py
A	scripts/label_stats.py
A	scripts/realtime_demo.py
A	scripts/replay_segments.py
A	scripts/train_model.py
M	uv.lock
```

## kmj02

Merge base: 1103baba06b21c78ff044d879176b0f1b56cc161

```text
1	15
ab45fd6 update readme
d77485f epoch100 + result
a3907c7 epoch 86
1006f3d epoch 78
e8fb3ab add ckp 62,71
2d88ea6 add ckpoint epoch58
bc380a6 add ckpoint(39,45,48)
02e8aa3 first train
dce4703 edit md file train mode
0759d03 add parm + change data
5953a85 add test mode
ca1c94c data EDA + yolo model setting
b74b94e slicing.py 수정
4bba653 slicing.py 여러영상 한번에 처리 가능하게 수정
42c5dfd video parsing
M	.gitignore
D	.python-version
A	EDA.ipynb
A	README.md
A	ckpoint/best_78.pt
A	ckpoint/last_100.pt
A	ckpoint/last_48.pt
A	ckpoint/last_58.pt
A	ckpoint/last_62.pt
A	ckpoint/last_71.pt
A	ckpoint/last_74.pt
A	ckpoint/last_86.pt
A	ckpoint/last_97.pt
A	ckpoint/last_98.pt
A	ckpoint/last_99.pt
A	data_preprocessing.py
A	dataset/README.roboflow.txt
A	dataset/data.yaml
A	dataset/data_EDA.yaml
A	model_yolo/yolo.py
A	model_yolo/yolo_param.yaml
M	pyproject.toml
A	realtime_detect.py
A	runs/yolo/gesture_yolov8n/BoxF1_curve.png
A	runs/yolo/gesture_yolov8n/BoxPR_curve.png
A	runs/yolo/gesture_yolov8n/BoxP_curve.png
A	runs/yolo/gesture_yolov8n/BoxR_curve.png
A	runs/yolo/gesture_yolov8n/args.yaml
A	runs/yolo/gesture_yolov8n/confusion_matrix.png
A	runs/yolo/gesture_yolov8n/confusion_matrix_normalized.png
A	runs/yolo/gesture_yolov8n/labels.jpg
A	runs/yolo/gesture_yolov8n/loss_plots/box_loss.png
A	runs/yolo/gesture_yolov8n/loss_plots/cls_loss.png
A	runs/yolo/gesture_yolov8n/loss_plots/dfl_loss.png
A	runs/yolo/gesture_yolov8n/results.csv
A	runs/yolo/gesture_yolov8n/results.png
A	runs/yolo/gesture_yolov8n/train_batch0.jpg
A	runs/yolo/gesture_yolov8n/train_batch1.jpg
A	runs/yolo/gesture_yolov8n/train_batch2.jpg
A	runs/yolo/gesture_yolov8n/train_batch21060.jpg
A	runs/yolo/gesture_yolov8n/train_batch21061.jpg
A	runs/yolo/gesture_yolov8n/train_batch21062.jpg
A	runs/yolo/gesture_yolov8n/val_batch0_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch0_pred.jpg
A	runs/yolo/gesture_yolov8n/val_batch1_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch1_pred.jpg
A	runs/yolo/gesture_yolov8n/val_batch2_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch2_pred.jpg
A	runs/yolo/gesture_yolov8n/weights/best.pt
A	runs/yolo/gesture_yolov8n/weights/last.pt
A	runs/yolo/val/BoxF1_curve.png
A	runs/yolo/val/BoxPR_curve.png
A	runs/yolo/val/BoxP_curve.png
A	runs/yolo/val/BoxR_curve.png
A	runs/yolo/val/confusion_matrix.png
A	runs/yolo/val/confusion_matrix_normalized.png
A	runs/yolo/val/val_batch0_labels.jpg
A	runs/yolo/val/val_batch0_pred.jpg
A	runs/yolo/val/val_batch1_labels.jpg
A	runs/yolo/val/val_batch1_pred.jpg
A	runs/yolo/val/val_batch2_labels.jpg
A	runs/yolo/val/val_batch2_pred.jpg
A	slicing.py
M	uv.lock
A	weights/yolo26n.pt
A	yolov8n.pt
```

## lkh vs hwangsoon

```text
M	.gitignore
A	README.md
A	dataset/index.csv
A	dataset/swipe_left/user00_swipe_left_0001.npz
A	dataset/swipe_left/user00_swipe_left_0002.npz
A	dataset/swipe_left/user00_swipe_left_0003.npz
A	dataset/swipe_left/user00_swipe_left_0004.npz
A	dataset/swipe_left/user00_swipe_left_0005.npz
A	dataset/swipe_left/user00_swipe_left_0006.npz
A	dataset/swipe_left/user00_swipe_left_0007.npz
A	dataset/swipe_left/user00_swipe_left_0008.npz
A	dataset/swipe_left/user00_swipe_left_0009.npz
A	dataset/swipe_left/user00_swipe_left_0010.npz
A	dataset/swipe_left/user00_swipe_left_0011.npz
A	dataset/swipe_left/user00_swipe_left_0012.npz
A	dataset/swipe_left/user00_swipe_left_0013.npz
A	dataset/swipe_left/user00_swipe_left_0014.npz
A	dataset/swipe_left/user00_swipe_left_0015.npz
A	dataset/swipe_left/user00_swipe_left_0016.npz
A	dataset/swipe_left/user00_swipe_left_0017.npz
A	dataset/swipe_left/user00_swipe_left_0018.npz
A	dataset/swipe_left/user00_swipe_left_0019.npz
A	dataset/swipe_left/user00_swipe_left_0020.npz
A	dataset/swipe_left/user00_swipe_left_0021.npz
A	dataset/swipe_left/user00_swipe_left_0022.npz
A	dataset/swipe_left/user00_swipe_left_0023.npz
A	dataset/swipe_left/user00_swipe_left_0024.npz
A	dataset/swipe_left/user00_swipe_left_0025.npz
A	dataset/swipe_left/user00_swipe_left_0026.npz
A	dataset/swipe_left/user00_swipe_left_0027.npz
A	dataset/swipe_left/user00_swipe_left_0028.npz
A	dataset/swipe_left/user00_swipe_left_0029.npz
A	dataset/swipe_left/user00_swipe_left_0030.npz
A	dataset/swipe_left/user00_swipe_left_0031.npz
A	dataset/swipe_left/user00_swipe_left_0032.npz
A	dataset/swipe_left/user00_swipe_left_0033.npz
A	dataset/swipe_left/user00_swipe_left_0034.npz
A	dataset/swipe_left/user00_swipe_left_0035.npz
A	dataset/swipe_left/user00_swipe_left_0036.npz
A	dataset/swipe_left/user00_swipe_left_0037.npz
A	docs/experiment_report_0905.md
A	docs/gesture_app.md
A	docs/model_metrics_0907.md
A	docs/setup_windows.md
A	docs/team_conventions.md
A	docs/user_guide.md
A	docs/why_not_jester.md
A	gesture/__init__.py
A	gesture/actions.py
A	gesture/baseline.py
A	gesture/config.py
A	gesture/dataset.py
A	gesture/gate.py
A	gesture/landmarks.py
A	gesture/model.py
A	gesture/pipeline.py
A	gesture/preprocess.py
A	gesture/presets.py
A	gesture/sanity.py
A	gesture/segmenter.py
A	gesture/training.py
A	gestures.json
D	main.py
A	models/hand_landmarker.task
A	programs/README.md
A	programs/check.py
M	programs/collect_gesture.py
D	programs/collect_gesture_video.py
D	programs/extract_gesture_videos.py
D	programs/flip_image.py
D	programs/validate_gesture_dataset.py
M	pyproject.toml
A	reports/compare_models.csv
A	reports/compare_models_summary.md
A	reports/gru_vs_lstm_0907/fig1_accuracy_f1.png
A	reports/gru_vs_lstm_0907/fig2_per_class_recall.png
A	reports/gru_vs_lstm_0907/fig3_confusion_test_p006.png
A	reports/gru_vs_lstm_0907/fig3b_confusion_val_p004.png
A	reports/gru_vs_lstm_0907/fig4_params_latency.png
A	reports/gru_vs_lstm_0907/fig5_training_curves.png
A	reports/gru_vs_lstm_0907/fig_system_flow.png
A	reports/gru_vs_lstm_0907/numbers.json
A	reports/gru_vs_lstm_0907/raw/bench_infer.json
A	reports/gru_vs_lstm_0907/raw/jin_ckpt_scores.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/prepare_report.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed0/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed0/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed1/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed1/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed2/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed2/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed42/history.csv
A	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed42/training_summary.json
A	reports/gru_vs_lstm_0907/raw/jin_on_ours_scores.json
A	reports/gru_vs_lstm_0907/raw/ours_results.json
A	reports/gru_vs_lstm_0907/scripts/bench_infer.py
A	reports/gru_vs_lstm_0907/scripts/make_report.py
A	reports/gru_vs_lstm_0907/scripts/run_ours.py
A	reports/gru_vs_lstm_0907/scripts/score_jin_ckpt.py
A	reports/gru_vs_lstm_0907/summary.md
A	reports/gru_vs_lstm_0907/table_main.md
A	reports/loo/test_p002/compare_models.csv
A	reports/loo/test_p002/compare_models_summary.md
A	reports/loo/test_p003/compare_models.csv
A	reports/loo/test_p003/compare_models_summary.md
A	reports/loo/test_p004/compare_models.csv
A	reports/loo/test_p004/compare_models_summary.md
A	reports/loo/test_p006/compare_models.csv
A	reports/loo/test_p006/compare_models_summary.md
A	reports/loo_4cls/test_p001/compare_models.csv
A	reports/loo_4cls/test_p001/compare_models_summary.md
A	reports/loo_4cls/test_p002/compare_models.csv
A	reports/loo_4cls/test_p002/compare_models_summary.md
A	reports/loo_4cls/test_p003/compare_models.csv
A	reports/loo_4cls/test_p003/compare_models_summary.md
A	reports/loo_4cls/test_p004/compare_models.csv
A	reports/loo_4cls/test_p004/compare_models_summary.md
A	reports/loo_4cls/test_p006/compare_models.csv
A	reports/loo_4cls/test_p006/compare_models_summary.md
A	reports/loo_none/test_p002/compare_models.csv
A	reports/loo_none/test_p002/compare_models_summary.md
A	reports/loo_none/test_p003/compare_models.csv
A	reports/loo_none/test_p003/compare_models_summary.md
A	reports/loo_none/test_p004/compare_models.csv
A	reports/loo_none/test_p004/compare_models_summary.md
A	reports/loo_none/test_p006/compare_models.csv
A	reports/loo_none/test_p006/compare_models_summary.md
A	reports/loo_revfist/test_p002/compare_models.csv
A	reports/loo_revfist/test_p002/compare_models_summary.md
A	reports/loo_revfist/test_p003/compare_models.csv
A	reports/loo_revfist/test_p003/compare_models_summary.md
A	reports/loo_revfist/test_p004/compare_models.csv
A	reports/loo_revfist/test_p004/compare_models_summary.md
A	reports/loo_revfist/test_p006/compare_models.csv
A	reports/loo_revfist/test_p006/compare_models_summary.md
A	scripts/compare_models.py
A	scripts/dataset_summary.py
A	scripts/eval_model.py
A	scripts/gesture_app.py
A	scripts/label_stats.py
A	scripts/realtime_demo.py
A	scripts/replay_segments.py
A	scripts/train_model.py
D	util/slice.py
D	util/valid_npz.py
M	uv.lock
D	yolo/evaluate_yolo.py
D	yolo/test_yolo_webcam.py
D	yolo/train_yolo.py
```

## lkh vs kmj02

```text
M	.gitignore
D	.python-version
A	EDA.ipynb
A	README.md
A	ckpoint/best_78.pt
A	ckpoint/last_100.pt
A	ckpoint/last_48.pt
A	ckpoint/last_58.pt
A	ckpoint/last_62.pt
A	ckpoint/last_71.pt
A	ckpoint/last_74.pt
A	ckpoint/last_86.pt
A	ckpoint/last_97.pt
A	ckpoint/last_98.pt
A	ckpoint/last_99.pt
A	data_preprocessing.py
A	dataset/README.roboflow.txt
A	dataset/data.yaml
A	dataset/data_EDA.yaml
D	gesture_model/__init__.py
D	gesture_model/gesture_detector.py
A	model_yolo/yolo.py
A	model_yolo/yolo_param.yaml
D	programs/collect_gesture.py
D	programs/collect_gesture_video.py
D	programs/extract_gesture_videos.py
D	programs/flip_image.py
D	programs/validate_gesture_dataset.py
M	pyproject.toml
A	realtime_detect.py
A	runs/yolo/gesture_yolov8n/BoxF1_curve.png
A	runs/yolo/gesture_yolov8n/BoxPR_curve.png
A	runs/yolo/gesture_yolov8n/BoxP_curve.png
A	runs/yolo/gesture_yolov8n/BoxR_curve.png
A	runs/yolo/gesture_yolov8n/args.yaml
A	runs/yolo/gesture_yolov8n/confusion_matrix.png
A	runs/yolo/gesture_yolov8n/confusion_matrix_normalized.png
A	runs/yolo/gesture_yolov8n/labels.jpg
A	runs/yolo/gesture_yolov8n/loss_plots/box_loss.png
A	runs/yolo/gesture_yolov8n/loss_plots/cls_loss.png
A	runs/yolo/gesture_yolov8n/loss_plots/dfl_loss.png
A	runs/yolo/gesture_yolov8n/results.csv
A	runs/yolo/gesture_yolov8n/results.png
A	runs/yolo/gesture_yolov8n/train_batch0.jpg
A	runs/yolo/gesture_yolov8n/train_batch1.jpg
A	runs/yolo/gesture_yolov8n/train_batch2.jpg
A	runs/yolo/gesture_yolov8n/train_batch21060.jpg
A	runs/yolo/gesture_yolov8n/train_batch21061.jpg
A	runs/yolo/gesture_yolov8n/train_batch21062.jpg
A	runs/yolo/gesture_yolov8n/val_batch0_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch0_pred.jpg
A	runs/yolo/gesture_yolov8n/val_batch1_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch1_pred.jpg
A	runs/yolo/gesture_yolov8n/val_batch2_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch2_pred.jpg
A	runs/yolo/gesture_yolov8n/weights/best.pt
A	runs/yolo/gesture_yolov8n/weights/last.pt
A	runs/yolo/val/BoxF1_curve.png
A	runs/yolo/val/BoxPR_curve.png
A	runs/yolo/val/BoxP_curve.png
A	runs/yolo/val/BoxR_curve.png
A	runs/yolo/val/confusion_matrix.png
A	runs/yolo/val/confusion_matrix_normalized.png
A	runs/yolo/val/val_batch0_labels.jpg
A	runs/yolo/val/val_batch0_pred.jpg
A	runs/yolo/val/val_batch1_labels.jpg
A	runs/yolo/val/val_batch1_pred.jpg
A	runs/yolo/val/val_batch2_labels.jpg
A	runs/yolo/val/val_batch2_pred.jpg
A	slicing.py
D	util/slice.py
D	util/valid_npz.py
M	uv.lock
A	weights/yolo26n.pt
D	yolo/evaluate_yolo.py
D	yolo/test_yolo_webcam.py
D	yolo/train_yolo.py
A	yolov8n.pt
```

## hwangsoon vs kmj02

```text
M	.gitignore
D	.python-version
A	EDA.ipynb
M	README.md
A	ckpoint/best_78.pt
A	ckpoint/last_100.pt
A	ckpoint/last_48.pt
A	ckpoint/last_58.pt
A	ckpoint/last_62.pt
A	ckpoint/last_71.pt
A	ckpoint/last_74.pt
A	ckpoint/last_86.pt
A	ckpoint/last_97.pt
A	ckpoint/last_98.pt
A	ckpoint/last_99.pt
A	data_preprocessing.py
A	dataset/README.roboflow.txt
A	dataset/data.yaml
A	dataset/data_EDA.yaml
D	dataset/index.csv
D	dataset/swipe_left/user00_swipe_left_0001.npz
D	dataset/swipe_left/user00_swipe_left_0002.npz
D	dataset/swipe_left/user00_swipe_left_0003.npz
D	dataset/swipe_left/user00_swipe_left_0004.npz
D	dataset/swipe_left/user00_swipe_left_0005.npz
D	dataset/swipe_left/user00_swipe_left_0006.npz
D	dataset/swipe_left/user00_swipe_left_0007.npz
D	dataset/swipe_left/user00_swipe_left_0008.npz
D	dataset/swipe_left/user00_swipe_left_0009.npz
D	dataset/swipe_left/user00_swipe_left_0010.npz
D	dataset/swipe_left/user00_swipe_left_0011.npz
D	dataset/swipe_left/user00_swipe_left_0012.npz
D	dataset/swipe_left/user00_swipe_left_0013.npz
D	dataset/swipe_left/user00_swipe_left_0014.npz
D	dataset/swipe_left/user00_swipe_left_0015.npz
D	dataset/swipe_left/user00_swipe_left_0016.npz
D	dataset/swipe_left/user00_swipe_left_0017.npz
D	dataset/swipe_left/user00_swipe_left_0018.npz
D	dataset/swipe_left/user00_swipe_left_0019.npz
D	dataset/swipe_left/user00_swipe_left_0020.npz
D	dataset/swipe_left/user00_swipe_left_0021.npz
D	dataset/swipe_left/user00_swipe_left_0022.npz
D	dataset/swipe_left/user00_swipe_left_0023.npz
D	dataset/swipe_left/user00_swipe_left_0024.npz
D	dataset/swipe_left/user00_swipe_left_0025.npz
D	dataset/swipe_left/user00_swipe_left_0026.npz
D	dataset/swipe_left/user00_swipe_left_0027.npz
D	dataset/swipe_left/user00_swipe_left_0028.npz
D	dataset/swipe_left/user00_swipe_left_0029.npz
D	dataset/swipe_left/user00_swipe_left_0030.npz
D	dataset/swipe_left/user00_swipe_left_0031.npz
D	dataset/swipe_left/user00_swipe_left_0032.npz
D	dataset/swipe_left/user00_swipe_left_0033.npz
D	dataset/swipe_left/user00_swipe_left_0034.npz
D	dataset/swipe_left/user00_swipe_left_0035.npz
D	dataset/swipe_left/user00_swipe_left_0036.npz
D	dataset/swipe_left/user00_swipe_left_0037.npz
D	docs/experiment_report_0905.md
D	docs/gesture_app.md
D	docs/model_metrics_0907.md
D	docs/setup_windows.md
D	docs/team_conventions.md
D	docs/user_guide.md
D	docs/why_not_jester.md
D	gesture/__init__.py
D	gesture/actions.py
D	gesture/baseline.py
D	gesture/config.py
D	gesture/dataset.py
D	gesture/gate.py
D	gesture/landmarks.py
D	gesture/model.py
D	gesture/pipeline.py
D	gesture/preprocess.py
D	gesture/presets.py
D	gesture/sanity.py
D	gesture/segmenter.py
D	gesture/training.py
D	gesture_model/__init__.py
D	gesture_model/gesture_detector.py
D	gestures.json
A	main.py
A	model_yolo/yolo.py
A	model_yolo/yolo_param.yaml
D	models/hand_landmarker.task
D	programs/README.md
D	programs/check.py
D	programs/collect_gesture.py
M	pyproject.toml
A	realtime_detect.py
D	reports/compare_models.csv
D	reports/compare_models_summary.md
D	reports/gru_vs_lstm_0907/fig1_accuracy_f1.png
D	reports/gru_vs_lstm_0907/fig2_per_class_recall.png
D	reports/gru_vs_lstm_0907/fig3_confusion_test_p006.png
D	reports/gru_vs_lstm_0907/fig3b_confusion_val_p004.png
D	reports/gru_vs_lstm_0907/fig4_params_latency.png
D	reports/gru_vs_lstm_0907/fig5_training_curves.png
D	reports/gru_vs_lstm_0907/fig_system_flow.png
D	reports/gru_vs_lstm_0907/numbers.json
D	reports/gru_vs_lstm_0907/raw/bench_infer.json
D	reports/gru_vs_lstm_0907/raw/jin_ckpt_scores.json
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/prepare_report.json
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed0/history.csv
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed0/training_summary.json
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed1/history.csv
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed1/training_summary.json
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed2/history.csv
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed2/training_summary.json
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed42/history.csv
D	reports/gru_vs_lstm_0907/raw/jin_lstm_on_our_data/seed42/training_summary.json
D	reports/gru_vs_lstm_0907/raw/jin_on_ours_scores.json
D	reports/gru_vs_lstm_0907/raw/ours_results.json
D	reports/gru_vs_lstm_0907/scripts/bench_infer.py
D	reports/gru_vs_lstm_0907/scripts/make_report.py
D	reports/gru_vs_lstm_0907/scripts/run_ours.py
D	reports/gru_vs_lstm_0907/scripts/score_jin_ckpt.py
D	reports/gru_vs_lstm_0907/summary.md
D	reports/gru_vs_lstm_0907/table_main.md
D	reports/loo/test_p002/compare_models.csv
D	reports/loo/test_p002/compare_models_summary.md
D	reports/loo/test_p003/compare_models.csv
D	reports/loo/test_p003/compare_models_summary.md
D	reports/loo/test_p004/compare_models.csv
D	reports/loo/test_p004/compare_models_summary.md
D	reports/loo/test_p006/compare_models.csv
D	reports/loo/test_p006/compare_models_summary.md
D	reports/loo_4cls/test_p001/compare_models.csv
D	reports/loo_4cls/test_p001/compare_models_summary.md
D	reports/loo_4cls/test_p002/compare_models.csv
D	reports/loo_4cls/test_p002/compare_models_summary.md
D	reports/loo_4cls/test_p003/compare_models.csv
D	reports/loo_4cls/test_p003/compare_models_summary.md
D	reports/loo_4cls/test_p004/compare_models.csv
D	reports/loo_4cls/test_p004/compare_models_summary.md
D	reports/loo_4cls/test_p006/compare_models.csv
D	reports/loo_4cls/test_p006/compare_models_summary.md
D	reports/loo_none/test_p002/compare_models.csv
D	reports/loo_none/test_p002/compare_models_summary.md
D	reports/loo_none/test_p003/compare_models.csv
D	reports/loo_none/test_p003/compare_models_summary.md
D	reports/loo_none/test_p004/compare_models.csv
D	reports/loo_none/test_p004/compare_models_summary.md
D	reports/loo_none/test_p006/compare_models.csv
D	reports/loo_none/test_p006/compare_models_summary.md
D	reports/loo_revfist/test_p002/compare_models.csv
D	reports/loo_revfist/test_p002/compare_models_summary.md
D	reports/loo_revfist/test_p003/compare_models.csv
D	reports/loo_revfist/test_p003/compare_models_summary.md
D	reports/loo_revfist/test_p004/compare_models.csv
D	reports/loo_revfist/test_p004/compare_models_summary.md
D	reports/loo_revfist/test_p006/compare_models.csv
D	reports/loo_revfist/test_p006/compare_models_summary.md
A	runs/yolo/gesture_yolov8n/BoxF1_curve.png
A	runs/yolo/gesture_yolov8n/BoxPR_curve.png
A	runs/yolo/gesture_yolov8n/BoxP_curve.png
A	runs/yolo/gesture_yolov8n/BoxR_curve.png
A	runs/yolo/gesture_yolov8n/args.yaml
A	runs/yolo/gesture_yolov8n/confusion_matrix.png
A	runs/yolo/gesture_yolov8n/confusion_matrix_normalized.png
A	runs/yolo/gesture_yolov8n/labels.jpg
A	runs/yolo/gesture_yolov8n/loss_plots/box_loss.png
A	runs/yolo/gesture_yolov8n/loss_plots/cls_loss.png
A	runs/yolo/gesture_yolov8n/loss_plots/dfl_loss.png
A	runs/yolo/gesture_yolov8n/results.csv
A	runs/yolo/gesture_yolov8n/results.png
A	runs/yolo/gesture_yolov8n/train_batch0.jpg
A	runs/yolo/gesture_yolov8n/train_batch1.jpg
A	runs/yolo/gesture_yolov8n/train_batch2.jpg
A	runs/yolo/gesture_yolov8n/train_batch21060.jpg
A	runs/yolo/gesture_yolov8n/train_batch21061.jpg
A	runs/yolo/gesture_yolov8n/train_batch21062.jpg
A	runs/yolo/gesture_yolov8n/val_batch0_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch0_pred.jpg
A	runs/yolo/gesture_yolov8n/val_batch1_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch1_pred.jpg
A	runs/yolo/gesture_yolov8n/val_batch2_labels.jpg
A	runs/yolo/gesture_yolov8n/val_batch2_pred.jpg
A	runs/yolo/gesture_yolov8n/weights/best.pt
A	runs/yolo/gesture_yolov8n/weights/last.pt
A	runs/yolo/val/BoxF1_curve.png
A	runs/yolo/val/BoxPR_curve.png
A	runs/yolo/val/BoxP_curve.png
A	runs/yolo/val/BoxR_curve.png
A	runs/yolo/val/confusion_matrix.png
A	runs/yolo/val/confusion_matrix_normalized.png
A	runs/yolo/val/val_batch0_labels.jpg
A	runs/yolo/val/val_batch0_pred.jpg
A	runs/yolo/val/val_batch1_labels.jpg
A	runs/yolo/val/val_batch1_pred.jpg
A	runs/yolo/val/val_batch2_labels.jpg
A	runs/yolo/val/val_batch2_pred.jpg
D	scripts/compare_models.py
D	scripts/dataset_summary.py
D	scripts/eval_model.py
D	scripts/gesture_app.py
D	scripts/label_stats.py
D	scripts/realtime_demo.py
D	scripts/replay_segments.py
D	scripts/train_model.py
A	slicing.py
M	uv.lock
A	weights/yolo26n.pt
A	yolov8n.pt
```
