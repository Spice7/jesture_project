# Jesture — 통합 제스처 프로그램

`integration` 브랜치는 정적 YOLO 게이트 + MediaPipe + GRU/LSTM + 키 매핑 UI를 하나의 프로그램으로 실행합니다. 원본 보존 지점은 `dev`의 `ae545b7`입니다.

## 설치와 실행

Python 3.12, Windows, 웹캠을 기준으로 합니다. 기본은 연습 모드이며 실제 키를 누르지 않습니다.

```powershell
uv sync --locked
uv run python main.py check
uv run python main.py
# 실제 키 입력을 원할 때만:
uv run python main.py --live
```

Windows의 torch/torchvision은 기존 CUDA 12.6 설정을 유지합니다. YOLO는 GPU가 없으면 CPU를 사용합니다. 키 입력은 Windows에서만 지원합니다. 다른 운영체제의 GUI/패키지 설치는 검증하지 않았습니다.

기존 `.venv`를 보존하려면 설치 전에 `$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-integration'`을 설정하세요. 이번 검증은 사용자 `.venv` 대신 TEMP의 별도 환경에서 수행했습니다.

다른 작업 디렉터리에서는 `uv run --project <저장소> python <저장소>/main.py`로 실행할 수 있습니다. 명령의 상대 파일 경로는 항상 저장소 루트 기준입니다. YOLO 파라미터 YAML 안의 상대 경로만 그 YAML 위치 기준입니다.

## 추론에 필요한 파일

| 경로 | 상태와 역할 |
| --- | --- |
| `models/hand_landmarker.task` | Git 포함. 손 관절 추출 |
| `models/gru_gesture.pt` | **외부 파일 필요**. 실제 학습된 동적 모델 |
| `models/gru_gesture.json` | **외부 파일 필요**. 위 가중치와 짝인 metadata |
| `models/static/v3.pt` | **외부 파일 필요**. 드라이브의 실제 lkh v3 정적 모델을 이 이름으로 복사 |
| `gestures.json` | Git 포함. 키 매핑·확신도·cooldown·정적 모델 경로·유지 시간 설정 |

이 세 외부 파일을 복사하면 기본 명령을 코드 수정 없이 사용할 수 있습니다. metadata는 저장된 모델과 일치해야 하며 `labels=[swipe_left, make_fist, no_gesture, finger_snap]`, `seq_len=30`, `feature_dim=63`, `mirror=false`, `use_z=true` 규약을 검사합니다. 현재 학습 코드가 생성하는 metadata 형식을 사용하세요. LSTM 가중치를 사용하는 경우 `--model models/lstm_gesture.pt`로 지정하며 같은 이름의 `.json`이 필요합니다.

**추론에는 train/valid/test 이미지·라벨, NPZ 학습 데이터, 학습용 data.yaml이 필요하지 않습니다.**

```powershell
uv run python main.py check --load-models
uv run python main.py --camera 1 --device cpu
```

`check`는 누락 경로를 나열하고 종료 코드 2를 반환합니다. `--load-models`를 붙이면 실제 모델 로드와 metadata 호환성까지 검사합니다. 카메라나 키 입력을 시작하지 않습니다.

### 제공된 정적 모델과 gate 모델

- `models/static/checkpoints/last_100.pt`: kmj02의 Git 포함 학습 모델. 명시적으로 사용하려면 `uv run python main.py --static-model models/static/checkpoints/last_100.pt`를 실행합니다. 동적 GRU와 metadata는 여전히 필요합니다.
- `models/static/checkpoints/`: 다른 epoch 체크포인트도 보존했습니다.
- `models/static/yolov8n.pt`, `models/static/yolo26n.pt`: Git 포함 초기 가중치. 완성된 제스처 게이트 모델을 대신하지 않습니다.
- `artifacts/kmj02/runs/`: 원래 학습 그래프·지표·best/last 가중치를 보존한 기록입니다. 기본 앱은 이 폴더를 읽지 않습니다.
- 기존 `yolo_gate.pt`는 hwangsoon 안내에서 lkh **v2** 모델을 복사해 붙인 이름이었습니다. 이제 게이트가 설정된 정적 모델을 직접 읽으므로 이 이름의 추가 복제는 필요 없습니다. v2, v3, kmj02 모델이 동일하다고 간주하지 않습니다. v2를 쓰려면 실제 파일을 별도 경로에 두고 `gestures.json`의 `gate.model` 또는 `--static-model`로 지정하세요.
- 정적 모델은 최소한 `start`, `stop`, `cancel` 클래스 이름을 포함해야 합니다. 추가 null/background 클래스는 허용하며 명령으로 실행하지 않습니다.

## 동작과 설정

손바닥 유지 → 게이트 열림, 주먹 유지 → 닫힘, cancel 포즈 → 설정된 정적 단축키. 열린 동안 동적 스와이프·주먹·스냅을 GRU로 분류해 매핑된 키를 실행합니다.

기존 유지 시간/한 번만 확정/순간 미검출 허용, 동적 smoothing·적응 임계값·짧은 손 놓침 대기·조기 판정·손 내림 오인 방지·동작별 cooldown을 유지합니다. 카메라 영상은 비반전으로 인식하며 UI 좌우반전은 표시만 바꿉니다. YOLO 바운딩 박스와 FPS, 인식 기록, 매핑 편집·프리셋은 같은 UI에 표시됩니다.

`gestures.json`의 `gate`에서 model/confidence/hold_seconds/miss_tolerance_seconds/imgsz/iou/max_det/device를 설정합니다. device=null은 자동 선택입니다. UI 변경은 설정 파일에 저장됩니다. 모델 누락은 시작 전에 안내하며 실행 중 게이트 오류는 인식을 닫습니다. 의도적으로 게이트 없이 쓰려면 `gate.enabled=false`로 설정합니다.

## 재학습에만 필요한 데이터

### kmj02 방식 YOLO 데이터

```text
dataset/
  train/images/   train/labels/
  valid/images/   valid/labels/
  test/images/    test/labels/
  data.yaml
```

이미지와 대응하는 실제 YOLO txt 라벨을 배치하세요. 빈 폴더는 `.gitkeep`으로 유지됩니다. `dataset/data.yaml`은 `path: .`, `train: train/images`, `val: valid/images`, `test: test/images`를 사용합니다. 학습 코드가 원본 YAML 위치 기준으로 경로를 정규화한 임시 설정을 Ultralytics에 전달하므로 CWD나 전역 datasets_dir에 영향받지 않습니다. 원본 YAML은 변경하지 않습니다.

```powershell
uv run python main.py yolo --mode train
uv run python main.py yolo --mode tune
uv run python main.py yolo --mode test
```

`yolo/yolo_param.yaml`에 기존 kmj02 학습 설정과 사용자 정의 평가 지표 처리를 유지했습니다. test/predict 기본 모델은 `models/static/checkpoints/last_100.pt`입니다. 새 학습 결과를 평가하려면 같은 YAML의 `paths.trained_model`을 실제 `runs/yolo/.../weights/best.pt`로 설정하세요. 코드 수정은 필요 없습니다.

### lkh v3 데이터와 설정

```text
datasets/static_gesture_v3/
  data.yaml                      # 실제 드라이브 데이터의 YAML을 복사
  train/images/   train/labels/
  valid/images/   valid/labels/
  test/images/    test/labels/
```

v3 YAML은 생성하지 않았습니다. 실제 YAML의 상대 경로를 이 구조에 맞춰 배치하세요(`path: .`, `train: train/images`, `val: valid/images`, `test: test/images`). 클래스 ID와 names는 실제 모델/라벨과 같아야 합니다.

```powershell
uv run python main.py yolo --config yolo/v3.yaml --mode train
uv run python main.py yolo --config yolo/v3.yaml --mode val
```

기존 lkh의 epochs=40, batch=-1, multi_scale=0.25, patience=10 등을 `yolo/v3.yaml`로 보존했습니다. val은 원래와 같이 test split을 사용합니다. 재학습한 best.pt를 `models/static/v3.pt`로 복사하면 앱에서 사용합니다. 추론만 하려면 이 데이터 폴더가 필요 없습니다.

### 동적 수집·학습과 보조 도구

`data/gestures/<label>/*.npz`가 동적 학습 데이터 위치입니다. 기존 사용자 `dataset/`의 NPZ·CSV는 옮기거나 덮어쓰지 않았습니다. 기존 데이터로 학습하려면 `--dataset dataset`을 명시하세요. Git의 user00 샘플 37개는 예제이며 4클래스 모델을 학습하기에 충분하지 않습니다.

```powershell
uv run python main.py collect
uv run python main.py collect --mirror --max-duration 5 --dataset data/legacy_mirrored
uv run python main.py collect-video
uv run python main.py extract
uv run python main.py validate
uv run python main.py train-dynamic --arch gru
uv run python main.py train-dynamic --arch lstm
uv run python main.py eval-dynamic
uv run python main.py compare
uv run python main.py replay
uv run python main.py frames --input videos --output frames
uv run python main.py record
uv run python main.py review
uv run python main.py flip
```

기본 수집기는 동적 모델과 맞는 비반전·최대 2.5초 설정입니다. lkh의 반전·5초 수집도 옵션으로 유지하지만 별도 폴더에 저장해야 하며 비반전 GRU 학습에 혼합하지 않습니다. 영상별 수집/일괄 추출/품질 검증의 기존 알고리즘은 유지했습니다.

`record`는 웹캠 녹화와 일정 간격 프레임 추출(`videos/`, `images/null/`)이고, `frames`는 폴더 내 영상의 모든 프레임을 추출합니다. `review`는 Gradio의 이미지 PASS/복구 가능한 trash 검토입니다. `flip`은 `two_fingers/`의 이미지를 `images/flipped/`로 저장합니다. EDA.ipynb는 저장소 루트에서 실행하며 같은 `dataset/data.yaml`을 사용합니다. 프레임·영상·학습 데이터는 직접 준비해야 합니다.

## 검증과 직접 확인

```powershell
uv run python -m unittest discover -s tests -v
uv run python main.py --help
```

테스트는 경로·누락 안내·실제 Ultralytics 데이터 해석·정적 모델 로드/메모리상 빈 프레임 추론·유지/미검출/재확정·게이트 오류·cooldown·GRU/LSTM 순전파·UI import를 확인합니다. 가짜 모델 파일이나 학습 데이터는 만들지 않습니다.

코드/초기화 수준 검증과 실제 모델로 가능한 smoke 검증을 수행했습니다. 외부 GRU/v3의 실물, 실제 웹캠·손 제스처·OS 키 입력·전체 학습·정확도는 직접 검증해야 합니다.

1. 외부 파일 배치 후 `main.py check --load-models` 실행.
2. 기본 연습 모드에서 카메라 연결, start/stop/cancel 및 동적 제스처 확인.
3. 짧은 미검출·반복 유지·연속 명령에서 중복 실행과 cooldown 확인.
4. 원할 때 실제 키 입력 스위치를 켜 매핑된 동작 확인.
5. 재학습 데이터 배치 후 학습·독립 test split 평가 수행.

[최종 통합 기록](integration/FINAL.md)에 기능별 출처·수정·삭제·검증 한계를 기록합니다. 기존 `integration/REPORT.md`, manifest, validation-results는 **dev 보존 당시의 역사 자료**이며 현재 경로에 대한 검증 결과가 아닙니다. `docs/history/` 및 과거 보고서의 브랜치/경로/성능 설명도 당시 기록입니다. 과거 JIN 비교 스크립트에는 별도 `external/jin` 소스/자료가 필요하며 `JESTURE_JIN_DIR`로 위치를 지정할 수 있습니다. 주 프로그램에는 필요하지 않습니다.
