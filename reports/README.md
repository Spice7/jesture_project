# reports/ — 실험 리포트

동적 제스처(GRU) 모델을 만들며 돌린 비교·검증 실험의 결과입니다. 대부분
`scripts/compare_models.py` 출력이며, 각 폴더에 지표 표(`*.csv`)와 요약(`*_summary.md`)이 들어 있습니다.

## 대표 리포트

| 항목 | 내용 |
| --- | --- |
| [gru_vs_lstm_0907/](gru_vs_lstm_0907/) | **핵심 리포트** — GRU vs LSTM 정면 비교. 그래프(`fig*.png`), 요약([summary.md](gru_vs_lstm_0907/summary.md)), 표([table_main.md](gru_vs_lstm_0907/table_main.md)), 원자료(`raw/`), 재현 스크립트(`scripts/`) |
| [compare_models_summary.md](compare_models_summary.md) · `compare_models.csv` | 단일 비교 실행 요약·수치 |

## LOO 교차검증 스윕 (`loo*`)

**LOO = Leave-One-person-Out.** 참가자 한 명씩 test로 빼고 나머지로 학습해, 새 사용자에서의
일반화를 본다. 하위 `test_pXXX/`는 "pXXX를 test로 뺀" 실행이며 각각 GRU/LSTM 비교표를 담는다.
폴더는 **학습 negative(동작 없음) 전략**만 다르다.

| 폴더 | 전략 | 데이터 |
| --- | --- | --- |
| [loo/](loo/) · [loo_none/](loo_none/) | negative 증강 없음 (베이스라인) | 855클립 (일부 참가자) |
| [loo_revfist/](loo_revfist/) | 주먹 되감기를 no_gesture로 추가 | 855클립 |
| [loo_4cls/](loo_4cls/) | 4클래스 최종 설정, 되감기 no_gesture=both(0.5) | 1,808클립 (전체 참가자) |

> 최종 시연 모델의 성능 수치 정리는 [docs/model_metrics_0907.md](../docs/model_metrics_0907.md) 참고.
