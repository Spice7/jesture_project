# dev: 세 브랜치 구현 확인용

배포용 통합이나 구현 선택을 하지 않은 개발 브랜치입니다. 기존 알고리즘과 각 구현의 실행 구조를 그대로 보존합니다.

| 구현 | 실행 기준 디렉터리 | 주요 진입점 |
| --- | --- | --- |
| lkh | 저장소 루트 | `programs/collect_gesture.py`, `yolo/test_yolo_webcam.py` |
| hwangsoon | `implementations/hwangsoon` | `scripts/gesture_app.py`, `scripts/train_model.py` |
| kmj02 | `implementations/kmj02` | `realtime_detect.py`, `model_yolo/yolo.py`, `data_preprocessing.py` |

## 환경 설치

저장소 루트에서 Python 3.12와 `uv sync --locked`를 사용합니다. 세 구현은 루트의 단일 `pyproject.toml`과 `uv.lock`을 사용합니다. Windows torch/torchvision은 기존 hwangsoon의 CUDA 12.6 인덱스를 유지합니다. 기존 개인 환경을 보존하려면 별도 환경을 지정합니다.

```powershell
# 저장소 루트에서 실행
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-dev'
uv sync --locked
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts/python.exe'
```

이후 아래 명령의 `$python`은 위 환경의 절대 경로입니다. 기본 `.venv`를 설치했다면 `$python = (Resolve-Path .venv/Scripts/python.exe).Path`로 설정합니다.

## 각각 실행

```powershell
# lkh: 루트에서 실행
& $python programs/collect_gesture.py
& $python yolo/test_yolo_webcam.py

# hwangsoon: 기존 상대 경로와 import를 유지
Push-Location implementations/hwangsoon
& $python scripts/gesture_app.py --help
& $python scripts/gesture_app.py
Pop-Location

# kmj02
Push-Location implementations/kmj02
& $python realtime_detect.py --help
& $python realtime_detect.py
& $python model_yolo/yolo.py --help
# 데이터 준비 후: & $python model_yolo/yolo.py --mode train
# 이미지 정제 UI: & $python data_preprocessing.py
Pop-Location
```

같은 카메라를 쓰는 프로그램은 하나씩 실행합니다. hwangsoon UI는 원래의 연습 모드로 시작합니다.

## 모델과 데이터

- lkh: 루트 `models/hand_landmarker.task`, `datasets/static_gesture_v3/data.yaml`, `runs/gesture/yolov8n_v3_rotation/weights/best.pt`를 사용합니다. 기존 로컬 데이터와 모델을 이동하거나 교체하지 않았습니다. 새 checkout에는 추적되지 않은 모델·데이터를 별도로 준비해야 합니다. MediaPipe 모델 원본은 `implementations/hwangsoon/models/hand_landmarker.task`에도 있습니다.
- hwangsoon: `implementations/hwangsoon/models/`에 `gru_gesture.pt`, `gru_gesture.json`, `yolo_gate.pt`를 별도로 준비해야 합니다. 이 세 파일은 원래 브랜치에도 커밋되어 있지 않습니다. 다른 YOLO 가중치를 임의로 대신 연결하지 않았습니다. 커밋된 MediaPipe 모델과 NPZ 37개는 해당 구현 안에 보존했습니다.
- kmj02: 기본 실시간 가중치 `ckpoint/last_100.pt`와 학습 결과·모든 체크포인트를 해당 구현 안에 보존했습니다. 학습/평가 이미지·라벨은 별도 준비가 필요합니다.
- kmj02 원본 `dataset/data.yaml`은 `../train/images` 등을 지정하는 반면 README 및 `data_EDA.yaml`은 `dataset/train/images` 구조입니다. 특히 사용자 정의 평가 함수는 전자를 구현 루트의 `train/images`로 해석합니다. 실제 데이터 위치에 맞춘 별도 YAML과 `--config`를 사용하세요. 원본 설정은 수정하지 않았습니다.
- CWD 의존 도구: lkh의 `util/slice.py`, `util/valid_npz.py`, `programs/flip_image.py`는 루트에서 실행합니다. kmj02의 `slicing.py`는 구현 폴더의 `video/`가 필요하고, `EDA.ipynb`도 해당 폴더를 기준으로 실행합니다. 일부 도구는 import 시 파일 처리하므로 일괄 import하지 않습니다.
- hwangsoon의 과거 비교 보고서 생성 스크립트에는 작성자 PC의 절대 경로 및 JIN 외부 자료 참조가 남아 있습니다. 보고서 결과는 보존했지만 외부 자료 없이 재실행할 수는 없습니다.

## 원본 문서와 통합 기록

- [hwangsoon 원본 안내](implementations/hwangsoon/README.md)
- [kmj02 원본 안내](implementations/kmj02/README.md)
- [통합 보고서](integration/REPORT.md)
- [전체 브랜치 변경 파일·커밋 비교](integration/branch-differences.md)
- [원본 함수·클래스·import 목록](integration/source-inventory.md)
- [모든 보존 파일의 출처·SHA-256](integration/source-manifest.json)

원본 README의 clone/브랜치 변경 안내는 원래 작업 당시의 기록입니다. 이 dev에서는 위 경로와 공통 환경을 사용합니다.
