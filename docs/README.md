# docs/ — 문서 안내

설치·실행은 **저장소 루트 `README.md`가 우선**입니다. 이 폴더는 사용 설명, 성능 수치,
실험·환경 기록을 모아 둔 곳입니다. 아래 표에서 필요한 문서를 바로 찾을 수 있습니다.

발표 슬라이드는 [`Jetsture_발표.pptx`](Jetsture_발표.pptx)입니다.

## 현행 이해용 (지금 저장소에 그대로 적용)

| 문서 | 내용 |
| --- | --- |
| [user_guide.md](user_guide.md) | 사용 설명서 — 웹캠 앞 손동작으로 키를 누르는 법, 게이트 켜기 |
| [gesture_app.md](gesture_app.md) | 시연 UI(`main.py`) 구조와 판정 로직 설명 |
| [model_metrics_0907.md](model_metrics_0907.md) | 최종 4클래스 GRU 시연 모델(`models/gru_gesture.pt`) 성능 수치 |
| [why_not_jester.md](why_not_jester.md) | 공개 데이터셋(Jester)을 쓰지 않은 이유 (발표 Q&A용) |

## 실험·환경 기록 (작성 시점 스냅샷)

| 문서 | 내용 |
| --- | --- |
| [experiment_report_0905.md](experiment_report_0905.md) | 2026-09-05 GRU 학습 실험 전 과정 기록 |
| [setup_windows.md](setup_windows.md) | Windows 시연 환경 세팅 (기기 2대 기준) |
| [team_conventions.md](team_conventions.md) | 수집 데이터 → 모델 학습 파이프라인 규약 |

## history/ — 원본 브랜치 작업 로그

통합 전 각자 브랜치에서 남긴 개인 작업 기록입니다. 그대로 보존합니다.

| 문서 | 내용 |
| --- | --- |
| [history/hwangsoon.md](history/hwangsoon.md) | 시계열(GRU) 담당 작업 로그·데이터 제출 절차 |
| [history/kmj02.md](history/kmj02.md) | 정적 YOLO 담당 작업 로그 (EDA·학습) |

---

### 경로 대응 (원본 문서 ↔ 현 저장소)

원본 다중브랜치 시절 문서 일부는 옛 폴더 이름을 씁니다. 현 구조와 이렇게 대응됩니다.

| 원본 문서 표기 | 현 저장소 경로 |
| --- | --- |
| `dataset/<label>/*.npz` (동적 수집 데이터) | `data/dynamic/<label>/*.npz` |
| 정적 YOLO 이미지·라벨 | `data/static/<version>/` (`start_stop_cancel`, `v3`) |
| 실험 리포트 | `reports/` |
| YOLO 학습 run 산출물 | `yolo/artifacts/` (→ [yolo/artifacts/README.md](../yolo/artifacts/README.md)) |
