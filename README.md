# Jesture — 손동작으로 Windows를 제어하는 제스처 인식 서비스

웹캠으로 오른손 동작을 인식해 Windows 단축키를 실행합니다. 정적 손모양(YOLO)으로 인식을 켜고 끄고, 동작 시퀀스(LSTM)로 명령을 판정하는 2단 구조입니다.

```
카메라 ─┬─ YOLO 손모양 게이트 (상시)  보자기 3초 → 인식 ON
        │                            주먹 3초   → 인식 OFF
        │                            총 모양 3초 → 창 표시/숨김
        └─ MediaPipe → LSTM (인식 ON일 때만)  →  단축키 전송
```

| 명령 제스처 | 동작 | 기본 단축키 |
| --- | --- | --- |
| 왼쪽으로 스와이프 | 편 오른손을 수행자 기준 왼쪽으로 이동 | `Left` |
| 손 오므리기 | 편 손 → 주먹 | `Space` |
| 핑거 스냅 | 손가락 튕기기 | 미지정 |

`no_gesture`는 명령이 아니라 "아무것도 실행하지 않는 상태"를 학습하기 위한 보조 라벨입니다.

## 빠르게 시작하기

목적에 따라 필요한 문서가 다릅니다.

| 하고 싶은 것 | 문서 |
| --- | --- |
| **프로그램만 써 보기** | 아래 [실행](#실행) |
| 실행 파일(exe) 만들어 배포하기 | [packaging/README.md](packaging/README.md) |
| GUI 조작·설정·손모양 게이트 상세 | [service/GUI_README.md](service/GUI_README.md) |
| 서비스 내부 구조와 판정 규칙 | [service/README.md](service/README.md) |
| 데이터 전처리·학습·평가 | [training/README.md](training/README.md) |
| 제스처 촬영 방법 | [programs/README.md](programs/README.md) |
| 수집한 데이터 제출 절차 | [programs/DATA_SUBMISSION.md](programs/DATA_SUBMISSION.md) |

## 환경 설치

Python 3.12와 Windows, 웹캠이 필요합니다.

```powershell
py -3.12 -m pip install uv
py -3.12 -m uv sync --extra gui
```

`--extra gui`를 빼면 GUI에 필요한 PySide6가 설치되지 않습니다. `pip install`로 따로 넣으면 다음 `uv sync` 때 사라지므로 이 방식을 쓰세요.

## 실행

```powershell
.\.venv\Scripts\python.exe -m service.gui --device auto
```

앱을 켜면 **카메라가 바로 열리고 손모양 게이트가 돕니다.** 제스처 인식 자체는 OFF로 시작하며, 보자기를 3초 유지하거나 `인식 시작` 버튼 또는 `Ctrl+Alt+G`로 켭니다.

**기본 단축키가 지정되지 않은 명령은 키를 보내지 않습니다.** `단축키 · 설정`에서 지정한 뒤, 저장하지 않아도 되는 창에서 먼저 시험하세요.

파이썬 없이 쓰려면 실행 파일을 만들 수 있습니다. 결과는 약 800MB 폴더이며 CPU만 사용합니다.

```powershell
.\.venv\Scripts\python.exe .\packaging\build_exe.py
.\dist\Jesture\Jesture.exe --self-test
```

## 저장소 구조

```text
programs/     제스처 데이터 수집기(collect_gesture.py)와 확인 도구
training/     전처리(prepare_dataset) · 모델(models) · 학습(train) · 평가(evaluate)
service/      실시간 서비스. CLI(main)와 Windows GUI(gui) + YOLO 게이트(yolo_gate)
packaging/    PyInstaller 빌드 스크립트와 스펙
dataset/      수집한 원본 NPZ (라벨별 폴더)
artifacts/    전처리 결과, 학습 run, 평가 보고서
models/       MediaPipe hand_landmarker.task
yolo/         정적 손모양 YOLO 가중치
```

## 현재 상태

**데이터** — 참가자 6명(`p001`~`p006`), 오른손 4클래스, NPZ 1,762개.

| 클래스 | 개수 |
| --- | ---: |
| finger_snap | 608 |
| swipe_left | 467 |
| make_fist | 447 |
| no_gesture | 240 |

**모델** — `artifacts/four_class_lstm_1layers_001`이 현재 기본 체크포인트입니다. 단방향 LSTM 1층(hidden 64), validation(p004, 298샘플) accuracy 98.66% / macro F1 0.981입니다. 층 수 비교 실험(1~4층)에서 1층이 가장 좋았습니다.

**손모양 게이트** — `yolo/v2/best.pt`, YOLOv8n 3클래스(`start`/`stop`/`cancel`)입니다.

**아직 안 한 것**

- **최종 test 평가(p006)를 실행하지 않았습니다.** 모델·설정 선택을 확정한 뒤 한 번만 수행해야 합니다.
- 위 98.66%는 모델을 고르는 데 이미 사용한 validation 수치입니다. 새 사용자·새 환경의 성능이 아닙니다.
- 연속 영상에서의 오작동률, 명령 중복 실행, 인식 지연 같은 **실시간 서비스 지표는 측정하지 않았습니다.** 잘라진 시퀀스 분류 성능만 확인한 상태입니다.
- p005의 finger_snap이 최근 크게 늘어 `artifacts/four_class_data_001`과 학습 결과는 현재 `dataset/`보다 오래되었습니다. 전처리부터 다시 실행해야 반영됩니다.

## 학습 다시 돌리기

전처리 → 학습 → validation 평가 순서입니다. **출력 폴더는 매번 새 이름을 써야 합니다.** 기존 폴더는 덮어쓰지 않고 거부합니다.

먼저 읽기 전용 점검으로 데이터 상태를 확인합니다.

```powershell
.\.venv\Scripts\python.exe -m training.prepare_dataset --input-dir .\dataset --output-dir .\artifacts\audit_002 --participant-map .\training\participant_map.json --train-subjects p001 p002 p003 p005 --val-subjects p004 --test-subjects p006 --audit-only
```

```powershell
.\.venv\Scripts\python.exe -m training.prepare_dataset --input-dir .\dataset --output-dir .\artifacts\four_class_data_002 --participant-map .\training\participant_map.json --train-subjects p001 p002 p003 p005 --val-subjects p004 --test-subjects p006
```

```powershell
.\.venv\Scripts\python.exe -m training.train --data-dir .\artifacts\four_class_data_002 --output-dir .\artifacts\four_class_lstm_1layers_002 --device cuda --num-layers 1
```

```powershell
.\.venv\Scripts\python.exe -m training.evaluate --data-dir .\artifacts\four_class_data_002 --run-dir .\artifacts\four_class_lstm_1layers_002 --output-dir .\artifacts\four_class_eval_1layers_002 --split val --device auto
```

같은 사람이 두 split에 들어가면 안 되고, 각 split에 네 클래스가 모두 있어야 합니다. 자세한 규칙과 옵션은 [training/README.md](training/README.md)에 있습니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s service -t . -p "test_*.py"
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_*.py"
```

현재 service 106개, training 75개입니다. 모두 임시 폴더의 합성 데이터와 모의 카메라·키보드를 사용하며 **실제 카메라나 사용자 키보드를 건드리지 않습니다.** 따라서 테스트 통과는 인터페이스와 계약이 맞는다는 뜻이지 실제 제스처 인식 성능을 보장하지 않습니다.

## 알아둘 제약

- **Windows 전용입니다.** 전역 단축키와 키 입력 전송이 Win32 API에 의존합니다.
- **오른손만 지원합니다.** 좌우 반전 증강과 왼손 모델은 지원하지 않으며, 좌표를 임의로 뒤집어 우회하지 마세요.
- 한 번에 손 하나만 봅니다. 양손 동시 등장이나 여러 사람은 지원하지 않습니다.
- 본 앱 창이 활성 상태면 단축키를 보내지 않습니다(의도된 안전장치). 그래서 총 모양으로 창을 띄울 때는 포커스를 가져오지 않습니다.
- `pyproject.toml`의 `tensorflow`는 코드 어디에서도 쓰지 않습니다. exe 빌드에서는 제외합니다.
