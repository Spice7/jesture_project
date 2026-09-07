# 2026-09-06 LSTM 예비 학습 결과

p005 제출 전, 현재 자료로 데이터 처리부터 실제 LSTM 학습까지 한 번 실행한 결과입니다.
최종 실험이나 실시간 서비스 평가는 아닙니다.

## 원본 보존과 승인된 정리

원본 dataset의 NPZ 1,071개와 index.csv는 변경하지 않았습니다. 총 1,072개 파일의
SHA-256을 학습 후에도 비교해 동일함을 확인했습니다.

학습용 `input/` 복사본에만 다음을 적용했습니다.

- `user00_swipe_left_0001`~`0037` → `p001_swipe_left_0124`~`0160`.
  NPZ 내부 participant_id와 sample_id도 함께 변경했습니다. 좌표·시간·검출 배열과
  나머지 메타데이터는 원본 그대로이며 저장 후 다시 비교했습니다.
- `p004_make_fist_0033`은 0001과, 0034는 0002와 sample_id를 제외한 전체 내용이 같았습니다.
  학습용 복사본에서는 0001·0002만 남기고 0033·0034는 포함하지 않았습니다.

정리 후 1,069개 전부 최대 5초 전처리 기준을 통과했습니다. 추가 품질 제외, 중복, 충돌은 없습니다.
이 기준은 이번 실행에 `--max-duration 5`로 적용했으며 수집/전처리 코드 기본값을 바꾸지는 않았습니다.

[input_provenance.json](input_provenance.json)은 원본/복사본 경로, 해시, 변경·제외 사유를 기록합니다.
전처리 manifest의 원본 경로는 이번 학습용 input 폴더 기준이므로, 최초 수집 자료까지
추적할 때는 이 provenance 파일을 함께 확인해야 합니다.

원본 dataset에는 user00과 중복 파일이 여전히 남아 있습니다. 원본에 participant_map만
적용해 다시 실행하면 이전 번호 충돌이 다시 나타납니다. 최종 실험에서도 승인된 정리 내역을
반영한 새 복사본을 만들고, 현재 input에 원본 파일을 무작정 추가해 이중 포함하지 마세요.

## 참가자별 분리

| 용도 | 참가자 | swipe_left | make_fist | no_gesture | 합계 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | p001, p002, p003 | 304 | 271 | 165 | 740 |
| validation | p004 | 100 | 74 | 35 | 209 |
| test 보류 | p006 | 50 | 50 | 20 | 120 |

같은 참가자가 둘 이상의 split에 들어가지 않습니다. test 배열은 전처리 시 생성했지만
학습과 학습 후 모델 확인 과정에서는 로딩하거나 평가하지 않았습니다.

## 실행 설정과 결과

단방향 LSTM: input_size=66, hidden_size=64, num_layers=1, dropout=0.2, num_classes=3.
시간축 길이 32, CPU, seed=42, batch_size=32, AdamW(lr=0.001, weight_decay=0.0001),
CrossEntropyLoss, gradient clipping=1.0. 최대 100 epoch, patience=10, min_delta=0.0001.
설정 탐색이나 다른 배정으로 반복 학습하지 않고 한 번 실행했습니다.

- 정상 완료, 19 epoch에서 early stopping.
- validation loss 기준 최적 epoch: **9**.
- 최적 epoch train loss: 0.025216, train accuracy: 99.59% (업데이트 과정의 지표).
- 최적 epoch validation loss: 0.023936, validation accuracy: **99.52% (208/209)**.
- checkpoint를 CPU에 복원한 뒤 validation을 다시 계산해 저장된 지표와 일치함을 확인했습니다.
- 유일한 validation 오분류: `make_fist/p004_make_fist_0006.npz`, 정답 make_fist → 예측 no_gesture.
  오분류를 이유로 해당 샘플을 삭제하거나 라벨을 바꾸지 않았습니다.

클래스별 validation 정답 수는 swipe_left 100/100, make_fist 73/74, no_gesture 35/35입니다.
validation은 최적 모델을 고르는 데 사용한 자료이며, 이 수치는 독립적인 최종 test 정확도가 아닙니다.
또한 한 참가자의 잘라진 동작 시퀀스 결과이므로 연속 영상 오작동률, 새로운 사용자/환경에 대한
일반화 성능은 아직 알 수 없습니다. p005 도착 후 별도 최종 실험을 진행해야 합니다.

## 파일 안내

- `input/`: 승인된 정리를 반영한 NPZ 복사본 1,069개.
- `input_provenance.json`: 최초 원본까지 추적하는 변환 내역과 SHA-256.
- `prepared/`: 전처리 X/y, manifest, 설정, 점검 보고서.
- [lstm_run_001/best_model.pt](lstm_run_001/best_model.pt): 최적 9 epoch 가중치.
- [lstm_run_001/training_summary.json](lstm_run_001/training_summary.json): 학습 요약.
- [lstm_run_001/training_config.json](lstm_run_001/training_config.json): 실제 설정/버전.
- [lstm_run_001/history.csv](lstm_run_001/history.csv): epoch별 train/validation 기록.

## 실행한 명령

아래는 프로젝트 루트에서 실행한 기록입니다. 기존 결과를 덮어쓰지 않으므로
같은 출력 경로로 다시 실행하면 거부됩니다. build_inputs.py는 이번 승인 내용에 맞춘 일회성 도구입니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 .\artifacts\pilot_20260906\build_inputs.py
.\.venv\Scripts\python.exe -X utf8 .\training\prepare_dataset.py --input-dir .\artifacts\pilot_20260906\input --output-dir .\artifacts\pilot_20260906\prepared --max-duration 5 --train-subjects p001 p002 p003 --val-subjects p004 --test-subjects p006
.\.venv\Scripts\python.exe -X utf8 -u .\training\train.py --data-dir .\artifacts\pilot_20260906\prepared --output-dir .\artifacts\pilot_20260906\lstm_run_001 --device cpu
```
