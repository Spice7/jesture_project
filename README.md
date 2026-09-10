# Jesture — 카메라 기반 AI 제스처 프로그램

YOLO 정적 제스처 인식, MediaPipe 손 랜드마크 추출, GRU/LSTM 동적 제스처 인식을 하나의 UI로 연결합니다. 인식한 제스처를 키 조합에 매핑하여 Windows의 창 선택, 바탕화면 보기, 미디어 제어 등에 사용할 수 있습니다. 기본 실행은 실제 키를 누르지 않는 연습 모드입니다.

## 주요 기능

- YOLO의 `start`·`stop` 포즈 유지로 동적 인식 게이트를 열고 닫으며, 열린 상태의 `cancel`은 매핑된 정적 단축키를 실행합니다.
- MediaPipe의 손 관절 21개를 추출하고 GRU(기본) 또는 LSTM으로 `swipe_left`, `make_fist`, `no_gesture`, `finger_snap`을 분류합니다. `no_gesture`는 기능을 실행하지 않습니다.
- UI에서 카메라 영상, YOLO 박스, FPS, 인식 기록을 확인하고 키 매핑·프리셋을 편집합니다.
- 포즈 유지 시간, 순간 미검출 허용, 확신도와 동작별 cooldown으로 반복 실행을 제어합니다.
- Windows 키 입력 API로 단축키와 볼륨·미디어 키를 실행합니다. 다른 OS에서는 연습 모드만 사용합니다.

## 프로젝트 구조

```text
main.py                 # 통합 CLI 진입점; 인자 없이 실행하면 app
gestures.json          # 제스처 매핑과 정적 게이트 설정
gesture/               # 랜드마크·동적 모델·추론 파이프라인·키 입력
gesture_model/         # YOLO 정적 제스처 검출
scripts/               # 통합 UI, 동적 학습·평가·재생
programs/              # 데이터 수집·전처리 도구
yolo/                  # YOLO 실행 코드와 학습 설정 YAML
models/                # 추론 모델과 metadata
data/gestures/         # 동적 학습용 NPZ
dataset/               # 기본 YOLO 학습 데이터 구조와 data.yaml
datasets/static_gesture_v3/ # 별도 v3 학습 데이터 배치 위치
tests/                 # 자동 통합 테스트
docs/, integration/    # 설명 및 과거 통합 기록
reports/, artifacts/   # 기존 실험 보고서와 학습 산출물
```

## 환경 설정

저장소 루트에서 실행합니다. Python **3.12**와 **uv**가 필요합니다. `.python-version`은 `3.12`, `pyproject.toml`의 허용 범위는 `>=3.12,<3.13`입니다. Windows와 웹캠을 기준으로 실행 확인했습니다.

```bash
uv sync --locked
```

`uv.lock`의 고정된 의존성을 사용합니다. Windows의 torch/torchvision은 프로젝트에 설정된 CUDA 12.6 인덱스를 사용하며, 정적 YOLO 장치는 기본 자동 선택입니다. 다른 OS에서의 전체 설치·GUI 동작은 검증하지 않았습니다.

## 필요한 모델 파일

현재 다음 파일은 **모두 Git에 추적되어 있어 정상 clone 시 포함됩니다**. 모델 파일이 누락된 복사본을 사용한다면 동일한 학습 결과의 실제 파일을 아래 위치에 배치하세요. 추론에는 원본 학습 데이터셋 전체가 필요하지 않습니다.

```text
models/
├─ hand_landmarker.task
├─ gru_gesture.pt
├─ gru_gesture.json
└─ static/
   └─ v3.pt
```

| 파일 | 역할 |
| --- | --- |
| `models/static/v3.pt` | YOLO 정적 제스처 모델. `start`, `stop`, `cancel` 클래스 이름 필요 |
| `models/gru_gesture.pt` | 기본 GRU 가중치와 모델 구조 정보 |
| `models/gru_gesture.json` | 위 GRU와 동일한 학습 결과의 metadata |
| `models/hand_landmarker.task` | MediaPipe 손 랜드마크 모델 |

YOLO 학습 결과 이름이 `best.pt`라면 사용할 학습 결과의 파일을 **`models/static/v3.pt`로 복사**합니다. 기본 앱은 `gestures.json`의 `gate.model`이 지정한 이 경로를 읽습니다. `models/static/checkpoints/`의 기존 모델과 초기 가중치는 보존된 별도 자산이며 기본 v3 모델과 동일하지 않습니다.

GRU의 `.pt`와 `.json`은 반드시 짝을 맞춰 사용합니다. `.pt`는 `state_dict`와 `arch`, `units`, `n_classes` 등의 구조 정보가 있는 PyTorch 체크포인트입니다. `.json`은 현재 학습 코드가 생성하는 JSON 객체로 다음 규약을 충족해야 합니다.

- `arch`: 기본 `gru`; 가중치의 아키텍처와 일치
- `labels`: **순서까지** `["swipe_left", "make_fist", "no_gesture", "finger_snap"]`; 가중치 클래스 수는 4
- `seq_len`: `30`, `feature_dim`: `63`, `mirror`: `false`, `use_z`: `true`

로더는 metadata의 좌표·시퀀스·라벨 규약과 가중치의 클래스 수·아키텍처를 검사합니다. LSTM을 사용하려면 동일 규약의 학습된 `.pt`와 같은 이름의 `.json`을 준비하고 `--model models/lstm_gesture.pt`로 지정합니다. 기본 제공 모델은 GRU입니다.

## 실행 방법

대표 실행 명령(연습 모드):

```bash
uv run --locked python main.py
```

카메라나 키 입력 없이 모델 준비 상태와 실제 로딩을 검사하려면:

```bash
uv run --locked python main.py check --load-models
```

실제 키 입력/OS 단축키 실행 모드는 Windows에서 다음과 같이 시작합니다. UI의 모드 스위치로도 전환할 수 있습니다.

```bash
uv run --locked python main.py --live
```

기본 매핑은 스와이프 → Space, 주먹 → Ctrl+Alt+Tab, 스냅 → Enter, cancel → Win+D입니다. UI에서 변경한 매핑은 설정 파일에 저장됩니다. 게이트의 기본 유지 시간은 2초, 순간 미검출 허용 시간은 0.3초입니다. 카메라 입력은 비반전으로 인식하며 UI의 좌우반전은 표시만 바꿉니다.

다른 카메라나 정적 YOLO의 CPU 실행을 지정하는 예:

```bash
uv run --locked python main.py --camera 1 --device cpu
```

`--device`는 정적 YOLO에 적용됩니다. `--gestures`로 다른 설정 JSON, `--static-model`로 다른 정적 모델을 지정할 수 있습니다. CLI의 상대 경로는 저장소 루트 기준이며, YOLO 설정 YAML 안의 상대 경로는 해당 YAML 디렉터리 기준입니다.

## 주요 CLI 명령

아래 명령은 모두 `uv run --locked python main.py` 뒤에 붙입니다. 세부 옵션은 `<명령> --help`로 확인하세요.

| 명령 | 용도 |
| --- | --- |
| `app` | 통합 UI 실행; 명령 생략과 동일 |
| `check --load-models` | 추론 파일 존재, 실제 모델 로딩과 metadata 검사 |
| `yolo --mode train` | 기본 YAML 설정으로 정적 모델 학습 |
| `collect` | 웹캠에서 동적 학습용 NPZ 수집 |
| `train-dynamic --arch gru` | GRU 학습; `--arch lstm`도 지원 |
| `eval-dynamic` | 동적 모델을 데이터셋에서 평가 |
| `compare --archs gru lstm` | GRU/LSTM을 학습하여 비교; 기본은 GRU만 비교 |
| `replay` | 저장된 NPZ를 스트림처럼 재생해 동적 판정 검사; 실제 키 입력 없음 |

```bash
uv run --locked python main.py --help
uv run --locked python main.py app --help
```

## 데이터셋 구조 및 학습

**기본 프로그램 실행(inference)에는 학습 이미지·라벨, NPZ 전체, 학습용 `data.yaml`이 필요하지 않습니다.** 아래 데이터는 수집·학습·평가 명령을 사용할 때 준비합니다.

### YOLO 학습/재학습

기본 `yolo/yolo_param.yaml`은 `dataset/data.yaml`을 사용합니다. 저장소의 실제 폴더 구조는 다음과 같으며, 이미지와 대응하는 YOLO `.txt` 라벨은 별도로 준비합니다.

```text
dataset/
├─ train/
│  ├─ images/
│  └─ labels/
├─ valid/
│  ├─ images/
│  └─ labels/
├─ test/
│  ├─ images/
│  └─ labels/
└─ data.yaml
```

현재 `data.yaml`은 `path: .`, `train: train/images`, `val: valid/images`, `test: test/images`, `names: ['cancel', 'start', 'stop']`이며 실제 폴더 경로와 일치합니다. 학습 코드가 원본 YAML 위치 기준으로 경로를 해석해 임시 설정을 전달하며 원본 YAML은 변경하지 않습니다. 폴더가 존재하는 것만으로 학습 데이터가 준비된 것은 아닙니다.

```bash
uv run --locked python main.py yolo --mode train
uv run --locked python main.py yolo --mode test
```

기본 test/predict 대상은 `models/static/checkpoints/last_100.pt`입니다. 새 모델 평가 시 `yolo/yolo_param.yaml`의 `paths.trained_model`을 해당 학습 결과로 지정하세요.

별도 `yolo/v3.yaml`은 **`datasets/static_gesture_v3/data.yaml`**을 사용합니다. 해당 디렉터리에도 위와 같은 train/valid/test 이미지·라벨 구조와 실제 클래스 정보가 맞는 `data.yaml`을 준비해야 합니다. 이 v3 데이터 YAML은 현재 Git에 포함되지 않습니다.

```bash
uv run --locked python main.py yolo --config yolo/v3.yaml --mode train
uv run --locked python main.py yolo --config yolo/v3.yaml --mode val
```

v3 설정의 `val`은 test split을 사용합니다. 학습 후 선택한 `best.pt`를 `models/static/v3.pt`에 복사하면 기본 앱에서 사용합니다.

### 동적 데이터 수집·학습

기본 데이터 위치는 `data/gestures/<label>/*.npz`입니다. 수집 기본값은 비반전·최대 2.5초입니다. 다른 좌표 규약으로 수집한 데이터를 혼합하지 마세요. Git의 user00 스와이프 샘플만으로는 전체 4클래스 학습에 충분하지 않습니다.

```bash
uv run --locked python main.py collect
uv run --locked python main.py train-dynamic --arch gru --out models/experiments/gru_gesture.pt
uv run --locked python main.py eval-dynamic --model models/experiments/gru_gesture.pt
```

학습은 가중치와 같은 이름의 `.json`을 함께 저장합니다. 위 예시는 제공된 시연 모델을 보존하도록 별도 출력 경로를 사용합니다. `--out`을 생략하면 `models/<arch>_gesture.pt`와 `.json`에 저장합니다. 다른 데이터 위치는 학습·평가의 `--dataset`으로 지정할 수 있습니다.

## 검증 상태와 직접 확인할 항목

- 사용자가 실제 PC의 `integration`에서 `uv run --locked python main.py`로 정상 실행을 확인했습니다.
- 2026-09-10 문서 정리 시 의존성 잠금 상태, 실제 GRU·MediaPipe·YOLO 로딩, 기존 자동 통합 테스트 11개, CLI 도움말/import, Python 파일 40개의 구문, YAML 경로와 충돌 마커 검사를 통과했습니다. 재현 명령은 아래와 같습니다.
- 자동 테스트는 경로 처리, 정적 모델의 빈 프레임 추론, 게이트 유지·미검출·cooldown·오류 처리, GRU/LSTM 순전파와 UI import를 다룹니다. 실제 웹캠 인식 정확도나 OS 키 실행 성공을 보장하는 검사는 아닙니다.
- 다양한 사용자·조명·카메라 환경의 인식률, 실제 `--live` 동작 전체, 전체 데이터셋 재학습 및 독립 평가 결과는 이번 자동 검증 범위 밖입니다. 실제 키 동작은 사용할 대상 프로그램에서 직접 확인하세요.

```bash
uv sync --locked
uv run --locked python main.py check --load-models
uv run --locked python -m unittest discover -s tests -v
uv run --locked python main.py --help
```

과거 구현 과정과 기능별 출처는 [통합 기록](integration/FINAL.md), [과거 문서](docs/history/)를 참고하세요. 과거 보고서의 브랜치·경로·검증 한계는 당시 상태를 설명하며, 현재 실행 방법은 이 README를 기준으로 합니다.
