# GRU vs LSTM 비교 — 2026-09-07

발표용 요약. 그림은 같은 폴더의 `fig1~fig5*.png`, 모든 수치의 원본은 `numbers.json`(집계)·`raw/`(시드별 원본), 재현 방법은 맨 끝.

## 1. 한 줄 결론

- **GRU 와 LSTM 은 셀(기억 방식)만 다르고 결과는 사실상 같다.** 같은 데이터·같은 학습 방법이면 차이는 시드(초기값) 흔들림 안이다.
- **점수 차이를 만드는 건 셀이 아니라 학습 레시피다.** 증강 + "명령이 아닌 동작" 합성(되감기·손 내리기)을 넣으면 처음 보는 사람(p006)에서 정확도 88 → 92~94 로 오르고, LSTM 원본 을 같은 데이터로 다시 학습하면 82 에 머문다. 차이는 전부 `no_gesture`(아무것도 아닌 동작을 걸러내는 능력)에서 난다.
- **룰 baseline(if-then 규칙, 학습 없음)이 이미 86~91** 이다. 모델은 그 위에 +1~6%p 를 얹는다. 숨기지 않는다.

## 2. LSTM 원본(JIN 브랜치) 평가 방식

| 항목 | LSTM 원본 (`training/`, `artifacts/final_*`) |
|---|---|
| 입력 특징 | 프레임당 66개: 손목 기준 상대 좌표 63 + 손목 누적 이동 xy 2 + 손 크기 1. 손목~중지 MCP 거리(시퀀스 중앙값)로 나눔 |
| 시퀀스 길이 | 32시점 (시간축 등간격 리샘플) |
| 품질 기준 | 검출률 ≥0.8, 연속 결측 ≤5, 20프레임 이상, **0.6~2.5초** (2.5초 넘는 클립 제외) |
| 라벨 | 4개: swipe_left=0, make_fist=1, no_gesture=2, finger_snap=3 |
| 분할 | **사람 단위**: train p001·p002·p003·p005 / **val p004** / test p006 (test 는 평가하지 않음, 배열만 만들어 둠) |
| 모델 | 단방향 LSTM hidden 64, **1~4층 비교**, dropout 0.2, 마지막 hidden → Linear |
| 학습 | AdamW lr 1e-3, weight decay 1e-4, batch 32, 최대 100 epoch, early stopping patience 10 (val loss), grad clip 1.0, seed 42, CUDA. **증강 없음** |
| 지표 | accuracy, macro precision/recall/F1, weighted F1, 클래스별 P/R/F1/support, confusion_matrix.csv, misclassified.csv, predictions.csv(확률), history.csv |
| 데이터 | 팀 드라이브 1,762개 (user00 없음, p005 make_fist 50 포함, p006 은 170개만) |

LSTM 원본의 최종 수치 (`artifacts/final_eval_*`, **val = p004, 298개**). test p006(170개) 열은 LSTM 원본 체크포인트를 원본 배열로 직접 채점한 것 (`scripts/score_jin_ckpt.py`, val 값이 원본 json 과 소수점까지 일치함을 확인).

| 층 수 | 파라미터 | val 정확도 | val macro F1 | val 재현율 스와이프/주먹/no_g/스냅 | test p006 정확도 | test macro F1 | test 재현율 |
|---:|---:|---:|---:|---|---:|---:|---|
| **1층** (원본 최종) | 34,052 | **96.6** | **94.6** | 100/100/**71**/100 | **95.3** | 91.9 | 100/100/**60**/100 |
| 2층 | 67,332 | 87.9 | 73.6 | 100/100/6/97 | 96.5 | 94.2 | 100/100/70/100 |
| 3층 | 100,612 | 87.2 | 70.4 | 98/98/0/100 | 87.6 | 70.4 | 100/100/0/98 |
| 4층 | 133,892 | 88.3 | 71.3 | 100/100/0/100 | 87.6 | 70.4 | 100/100/0/98 |

- 2~4층은 no_gesture 를 거의 못 걸러(0~6%) 층을 늘릴수록 나빠졌다 → LSTM 원본 실험도 1층을 골랐다.
- 1차 실험(`four_class_lstm_1layers_001`, p005 스냅 158개 추가 전, LSTM 원본 서비스의 기본 체크포인트)은 val 98.7 / F1 98.1, test 95.3 / 92.9 였다. 데이터가 늘었는데 val 이 내려간 것은 시드 1개짜리 흔들림 범위다(4절 참고).

## 3. 같은 시험지로 다시 비교 (핵심 표)

LSTM 원본 실험 방식 그대로: **사람 단위 분할 train p001·p002·p003·p005(+user00) / val p004 / test p006**, 이 저장소의 `dataset/`(1,912개) 사용, **시드 0·1·2 평균 ± 표준편차**. LSTM 원본 은 LSTM 원본 코드(`prepare_dataset.py` → `train.py` 1층)를 그대로 이 저장소 데이터에 돌린 것.

| 모델 | 구조 | 파라미터 | 학습 샘플 | val p004 정확도 | val macro F1 | **test p006 정확도** | **test macro F1** | test 재현율 스와이프/주먹/no_g/스냅 | CPU 추론 ms |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| **GRU (시연 레시피)** | GRU 2층 64 + FC32 | 51,940 | 8,476 | 93.6 ± 2.1 | 88.5 ± 4.1 | **92.3 ± 2.5** | **92.1 ± 2.3** | 99/99/**74**/95 | 0.92 |
| LSTM (시연 레시피) | LSTM 2층 64 + FC32 | 68,516 | 8,476 | 91.0 ± 1.9 | 80.2 ± 6.6 | 94.3 ± 1.6 | 93.7 ± 1.5 | 99/98/**86**/93 | 0.30 |
| LSTM 1층 (원본 코드, 같은 데이터) | LSTM 1층 64 + Linear | 34,052 | 1,250 | 92.5 ± 3.6 | 85.6 ± 9.1 | 82.4 ± 3.8 | 78.4 ± 5.4 | 99/100/**33**/100 | 0.15 |
| GRU (증강 없음) | GRU 2층 64 + FC32 | 51,940 | 1,264 | 89.9 ± 2.4 | 79.2 ± 6.4 | 88.4 ± 0.4 | 86.8 ± 0.5 | 100/99/58/97 | 0.92 |
| LSTM (증강 없음) | LSTM 2층 64 + FC32 | 68,516 | 1,264 | 92.4 ± 1.7 | 87.4 ± 3.5 | 89.2 ± 0.6 | 88.1 ± 0.7 | 100/100/60/97 | 0.30 |
| 룰 baseline (학습 없음) | if-then 규칙 | 0 | - | 91.3 | 88.3 | 86.6 | 85.3 | 99/100/54/92 | ~0 |

- val p004 = 311개 (100/76/35/100), test p006 = 337개 (100/100/87/50). LSTM 원본 코드는 2.5초 초과 클립을 빼서 val 298 / test 337.
- "시연 레시피" = 증강 ×3 + 되감기 no_gesture(주먹·스와이프 50%) + 손 내리기 합성 50% + 짧은 스와이프 50% (`models/gru_gesture.pt` 와 같은 조건).
- 학습 시간(GPU, RTX 4060): 시연 레시피 GRU 14초 / LSTM 16초, 증강 없음 3초. LSTM 원본 코드 1층은 수 초.
- CPU 추론(배치 1, 1스레드, 200회 중앙값, `scripts/bench_infer.py`): 셋 다 1ms 미만. 웹캠 한 프레임 33ms 에 비하면 전부 무시할 수준. (GRU 가 LSTM 보다 느린 것은 PyTorch CPU 커널 차이. 파라미터 수와 무관.)
- 시드별 원본: test 정확도 GRU 95.0/89.0/92.9, LSTM(시연 레시피) 95.5/92.0/95.3, LSTM 원본 78.6/81.0/87.5 (seed 42 는 86.1). `numbers.json` → `groups.*.test.per_seed`.

### 그림 읽는 법

- `fig1_accuracy_f1.png` 정확도·F1 막대. 오차막대 = 시드 3개 표준편차. 점선 = 룰 baseline.
- `fig2_per_class_recall.png` 클래스별 재현율. **스와이프·주먹·스냅은 누구나 93~100. 벌어지는 건 no_gesture 하나.**
- `fig3_confusion_test_p006.png` 혼동행렬(시드 3개 합). LSTM 원본 은 p006 의 no_gesture 261개 중 176개를 명령으로 오인(스와이프 48·주먹 75·스냅 53). GRU 는 67개.
- `fig3b_confusion_val_p004.png` val p004 도 같은 그림.
- `fig4_params_latency.png` 파라미터·추론 시간.
- `fig5_training_curves.png` 학습 곡선. 학습 정확도는 금방 100%, 검증은 88~96 에서 흔들림 = 사람이 바뀌면 생기는 일반화 문제이지 학습 부족이 아니다.
- `fig_system_flow.png` 전체 흐름 비교: LSTM 시스템(모델 매 프레임 판정 + 중복 방지) vs GRU 시스템(구간 감지 규칙 → GRU → 상식 검사 규칙 → 실행 판단 규칙, 4단 검문).

## 4. 차이 요약 (5줄)

1. **GRU vs LSTM(시연 레시피, 같은 레시피)**: val 93.6 vs 91.0, test 92.3 vs 94.3 → 서로 엎치락뒤치락, 표준편차(2~4) 안. **동률.**
2. **레시피 효과**: 증강 없음 → 시연 레시피에서 test 88.4 → 92.3(GRU), 89.2 → 94.3(LSTM). no_gesture 재현율 58 → 74 / 60 → 86. 셀 종류보다 훨씬 큰 차이.
3. **LSTM 원본 을 같은 데이터로 재학습**하면 val 은 92.5 로 GRU 와 같지만 **test 82.4**. 원인은 no_gesture 33% — 증강·합성 음성 샘플이 없어서 "손 내리기·되돌리기" 같은 동작을 명령으로 본다. 또 시드에 매우 민감(val 87.9~96.6, 표준편차 3.6~9.1).
4. **LSTM 원본 실험이 낸 96.6(val p004)은 GRU 93.6 보다 높다.** 단, 시드 1개(42)이고 이 저장소 데이터로 시드 3개를 돌리면 92.5 ± 3.6 → 같은 수준. p006 시험지도 LSTM 원본 것은 170개(no_gesture 20), GRU 것은 337개(no_gesture 87, 손 내리기 20 포함)라 LSTM 원본 test 95.3 과 GRU 92.3 은 **직접 비교 불가**. 직접 비교는 3절 표(같은 데이터)만.
5. **룰 baseline 이 val 91.3 / test 86.6.** 증강 없는 모델은 룰과 비슷하고, 시연 레시피 모델만 룰을 +6~8%p 앞선다. 룰이 잘 되는 이유는 문제가 단순하고 임계값을 사람이 이 데이터 보고 맞췄기 때문.

## 5. 왜 GRU + 룰을 골랐나 (발표 멘트)

- GRU 는 LSTM 의 게이트 3개를 2개로 줄인 셀이라 같은 층·유닛이면 파라미터가 24% 적다(51,940 vs 68,516). 이 문제에서는 정확도가 같으니 가벼운 쪽을 택했다. **"GRU 라서 더 정확하다"고 말하지 않는다.**
- 정확도를 올린 건 모델이 아니라 데이터 쪽 작업(증강, 되감기·내리기 합성)이고, 이건 GRU 든 LSTM 이든 똑같이 적용된다. LSTM 원본 에 같은 레시피를 넣으면 같은 수준이 될 것이다.
- 실시간에서는 모델 혼자 쓰지 않는다. **동작 구간 감지 규칙 → GRU → 상식 검사(손 모양·이동량) → 쿨다운** 4단 구조. 모델의 약점(no_gesture 74%)을 규칙이 받쳐서 웹캠 15분 실측 **188회 실행, 오작동 0** (`docs/model_metrics_0907.md` 4절). LSTM 원본 쪽은 모델 단독 점수만 있고 실시간 오작동 수치는 아직 없다.
- 남은 약점도 정직하게: 처음 보는 사람의 no_gesture 는 사람마다 "아무것도 아닌 동작"이 달라 40~85% 로 흔들린다. 시연 모델은 6명 전원을 학습에 넣어 이 문제를 피했다(같은 사람 검증셋 100%).

## 6. 주의점 (데이터·라벨 차이)

- **데이터가 다르다.** LSTM 원본 1,762개 vs 이 저장소 1,912개. 이 저장소: user00(담당자 스와이프 37) 포함, p005 make_fist 50 은 규약 위반으로 `dataset_excluded/` (LSTM 원본은 포함), p002 no_gesture 4개 제외(LSTM 원본 포함), **p006 337개 vs LSTM 원본 170개** (LSTM 원본 드라이브에 p006 1차분만 올라간 듯 — LSTM 원본에게 확인 필요).
- LSTM 원본 전처리는 2.5초 초과 클립 26개를 버린다(p002 12, p004 12, p003 2). GRU 전처리는 길이로 버리지 않는다(리샘플만). 그래서 3절 표에서 val 이 298 vs 311.
- LSTM 원본 `participant_map.json` 은 user00→p001 로 매핑하지만 이 저장소 데이터에서는 sample_id 가 p001 과 겹쳐 충돌(74건)이 나서, user00 을 별도 train 참가자로 넣었다.
- LSTM 원본 `evaluate.py` 는 입력 메타데이터 SHA-256 을 검사해 git 사본에서 실행이 막힌다 → 같은 계산을 `scripts/score_jin_ckpt.py` 로 직접 했고, val 수치가 원본 json 과 일치함을 확인했다.
- LSTM 원본 서비스(`service/settings.py`)의 기본 체크포인트는 `four_class_lstm_1layers_001`(1차 실험)이고, `final_1layers_001` 이 아니다.
- LSTM 원본 코드로 학습한 LSTM 의 best epoch 는 3~16 으로 시드마다 크게 달라, 시드 1개 결과로 층 수를 고른 LSTM 원본 표(2절)는 재현성이 약하다. 발표에서 "시드 3개 평균" 을 GRU 쪽 근거로 쓸 수 있다.

## 7. 못 한 것

- LSTM 원본 `best_model.pt` 를 GRU 특징에 바로 넣기: 특징 차원이 다르다(66 vs 63, 길이 32 vs 30) → 대신 LSTM 원본 코드로 이 저장소 데이터를 전처리해 재학습·채점했다(3절).
- LSTM 원본 에 GRU 시연 레시피(증강·합성 음성)를 넣어 재학습: LSTM 원본 전처리 코드는 npz 파일만 읽어 합성 샘플을 넣으려면 npz 로 저장하는 작업이 필요 → 시간상 생략. 대신 "시연 레시피 LSTM(2층)+레시피" 가 그 역할(94.3).
- LSTM 원본 p006 170개 부분집합으로 GRU 를 채점(LSTM 원본 test 95.3 과 직접 비교)은 하지 않았다. 원본 manifest 에 파일 목록이 있어 가능하다.
- 실시간(웹캠) 오작동 비교는 LSTM 원본 쪽 실측이 없어 못 했다.

## 8. 재현

모든 스크립트는 `scripts/` 에 있고, JIN 브랜치 파일은 `git show origin/JIN:<path>` 로 읽어 임시 폴더 `<S>/jin/` 에 복사해 썼다 (hwangsoon 트리 수정 없음). 파이썬은 `.venv/Scripts/python.exe`, `PYTHONUTF8=1`. 스크립트 안의 `SCR` 경로(임시 폴더)만 맞추면 그대로 돈다.

```
# 0) LSTM 원본 코드·결과 사본
git show origin/JIN:training/{prepare_dataset,train,evaluate,models,gesture_schema}.py > <S>/jin/training/...
git show origin/JIN:artifacts/{final_*,final_eval_*,four_class_data_00{1,2},four_class_lstm_1layers_001}/* > <S>/jin/artifacts/...

# 1) LSTM 원본 체크포인트를 원본 배열(X_val/X_test.npy)로 채점 → raw/jin_ckpt_scores.json
python scripts/score_jin_ckpt.py

# 2) LSTM 원본 전처리를 이 저장소 dataset/ 에 적용 (user00 은 별도 train 참가자) → raw/jin_lstm_on_our_data/prepare_report.json
python <S>/jin/training/prepare_dataset.py --input-dir dataset --output-dir <S>/jin/data_ours_001 --train-subjects p001 p002 p003 p005 user00 --val-subjects p004 --test-subjects p006

# 3) LSTM 원본 train.py 로 1층 LSTM 학습 (seed 42, 0, 1, 2) → raw/jin_lstm_on_our_data/seed*/
python <S>/jin/training/train.py --data-dir <S>/jin/data_ours_001 --output-dir <S>/jin/lstm_ours_1l_seed<N> --num-layers 1 --seed <N> --device cuda
python scripts/score_jin_ckpt.py lstm_ours_1l_seed0:data_ours_001 lstm_ours_1l_seed1:data_ours_001 lstm_ours_1l_seed2:data_ours_001 lstm_ours_1l_seed42:data_ours_001
#    → raw/jin_on_ours_scores.json

# 4) GRU / 시연 레시피 LSTM, 같은 분할, 시드 0·1·2, 레시피 2종(plain / recipe) + 룰 baseline → raw/ours_results.json
#    gesture.dataset.split_by_person(samples, ["p004"], ["p006"]); gesture.training.fit 기본값(Adam 1e-3, batch 32, 최대 80 epoch, patience 12, 최적 가중치 복원)
python scripts/run_ours.py raw/ours_results.json

# 5) CPU 추론 벤치(가중치 무관, 200회 중앙값) → raw/bench_infer.json
python scripts/bench_infer.py
# 6) 그림·numbers.json·table_main.md
python scripts/make_report.py reports/gru_vs_lstm_0907
```

- LSTM 원본 수치: `origin/JIN:artifacts/final_eval_{1,2,3,4}layers_001/evaluation_metrics.json` (val p004, seed 42), 학습 설정 `artifacts/final_*/training_config.json`, 데이터 보고 `artifacts/four_class_data_002/report.json`.
- GRU 기존 결과(다른 분할): `docs/model_metrics_0907.md`, `reports/loo_4cls/` (5명 leave-one-person-out, val p002 고정: GRU 89.0 / 룰 93.0). 이번 분할(val p004 고정)과 숫자가 다른 건 시험 대상 사람이 다르기 때문이다.