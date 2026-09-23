# 로컬 dev 통합 보고서

## 1. 작업 전 상태

- 시작 브랜치: `lkh`, HEAD `46cea52`, origin/lkh와 동일.
- 추적 파일의 staged/unstaged 수정 없음. `?? dataset/` 존재. 이 폴더의 238개 파일 SHA-256을 별도 임시 기록으로 비교하여 보존 확인.
- `.venv`, 모델, 영상, 학습 산출물 등 ignored 사용자 파일은 이동·교체하지 않음.
- 로컬 dev/hwangsoon/kmj02는 없었으며 origin의 해당 refs를 분석. `THIRD_BRANCH`는 사용자 목록의 `kmj02`로 해석.
- `git fetch origin` 수행 후 최신 refs 변화 없음. 상위 디렉터리와 저장소에서 적용할 AGENTS.md를 찾지 못함.

| 브랜치 | 최신 커밋 | dev와 merge-base | dev에만 / 해당 브랜치에만 커밋 |
| --- | --- | --- | --- |
| dev | `2adca814bc236a0f16ccb3500d9b9df192446282` | 동일 | 0 / 0 |
| lkh | `46cea52` | `2adca81` | 0 / 8 |
| hwangsoon | `23cc75f` | `dbd7016` | 3 / 17 |
| kmj02 | `ab45fd6` | `1103bab` | 1 / 15 |

전체 출처 SHA와 파일 상태는 source-manifest.json에 기록.

## 2. 브랜치 분석

### lkh

- 정적 YOLO start/stop/cancel 인식. `GestureDetector`, `GestureResult`: 유지 시간, 미검출 허용 시간, 한 번만 확정되는 상태 관리.
- 웹캠 수집, 영상 수집/추출, NPZ 품질 검증, 프레임 추출, 좌우 반전 도구, YOLO 학습·평가·웹캠 실행.
- dev 대비 추가 12개. 수정/삭제/이름 변경 없음.
- 추가 전체: `gesture_model/__init__.py`, `gesture_model/gesture_detector.py`, `programs/collect_gesture.py`, `programs/collect_gesture_video.py`, `programs/extract_gesture_videos.py`, `programs/flip_image.py`, `programs/validate_gesture_dataset.py`, `util/slice.py`, `util/valid_npz.py`, `yolo/evaluate_yolo.py`, `yolo/test_yolo_webcam.py`, `yolo/train_yolo.py`.
- 모델·데이터: `models/hand_landmarker.task`, `datasets/static_gesture_v3/data.yaml`, `runs/gesture/yolov8n_v3_rotation/weights/best.pt`; 학습 초기 가중치 `yolov8n.pt`는 CWD 기준.
- dependencies는 dev와 동일.

### hwangsoon

- MediaPipe 관절 → MotionSegmenter → GRU/LSTM `GestureRNN`/`GestureClassifier` → 규칙 검사 → `ActionMapper` → Windows 키 입력. `Pipeline`, `StaticGate`, `FakeGate`, `HandTracker` 포함.
- customtkinter `App` UI, 키 매핑·프리셋·녹화, 연습 모드/실제 입력, 동적 swipe_left/make_fist/no_gesture/finger_snap 학습·평가·재생 및 모델 비교 보고서.
- dev 대비 추가 138개, 수정 `.gitignore`, `pyproject.toml`, `uv.lock`, 삭제 빈 `main.py`. 이름 변경 없음.
- 추가 분류: `gesture/` 14, `scripts/` 8, `programs/` 3, `gesture_model/` 2, `docs/` 7, `reports/` 63, `dataset/` 38, 모델 1, README 및 gestures.json.
- MediaPipe 모델은 추적됨. GRU 가중치·메타 JSON·YOLO gate 가중치는 미추적 외부 준비물.
- customtkinter/scikit-learn/torchvision 추가, Windows torch/torchvision CUDA 12.6 전용 인덱스 설정.

### kmj02

- `cap_vdieo`: 영상 프레임 추출. `ReviewSession`/`build_app`: Gradio 이미지 검토·PASS 기록·trash 이동.
- `model_yolo/yolo.py`: YAML 설정을 읽어 train/tune/test/predict 실행. `load_config`, `build_common_args`, `run_train`, `run_tune`, `run_test`, `run_predict`, 사용자 정의 지표 및 ROC 관련 처리.
- `realtime_detect.py`: 체크포인트 기반 OpenCV 웹캠 감지. EDA 노트북, 체크포인트 11개, 학습/검증 산출물과 가중치 보존.
- dev 대비 추가 62개, 수정 `.gitignore`, `pyproject.toml`, `uv.lock`, 삭제 `.python-version`. 이름 변경 없음. 빈 main.py는 dev와 동일.
- 추가 분류: `runs/` 39, `ckpoint/` 11, `dataset/` 3, `model_yolo/` 2, EDA/README/전처리/실시간/slicing 각 1, `weights/yolo26n.pt`, `yolov8n.pt`.
- gradio/torchvision 추가, 모든 OS에 PyTorch CUDA 12.6 인덱스 설정.

개별 파일의 출처 SHA와 내용 해시는 [source-manifest.json](source-manifest.json)에 기록했다. (브랜치 간 파일 차이·소스 인벤토리·검증 로그 원문은 정리 과정에서 제거했으며, 필요하면 git 이력에서 확인할 수 있다.)

## 3. 브랜치 간 관계

- lkh와 hwangsoon의 `gesture_model` 두 파일은 Git blob과 내용이 동일. hwangsoon StaticGate는 lkh detector를 사용한다고 원문에도 명시. 각 실행 루트에서 원래 import가 되도록 양쪽 모두 그대로 보존.
- `programs/collect_gesture.py`는 실제 diff에서 최대 시간 5.0초 대 2.5초, lkh의 cv2.flip 호출 유무가 다름. 좌표 규약이 달라 선택·공통화하지 않음.
- lkh와 kmj02는 정적 제스처 YOLO 학습·검증·웹캠 구현이 겹치지만 독립 코드. util/slice.py와 slicing.py도 별도 프레임 추출 구현. 모두 보존.
- dataset/models/runs 같은 런타임 경로는 파일명뿐 아니라 CWD·__file__ 기반 경로 충돌도 가능. 구현별 루트로 분리하여 import와 자산 경로를 유지.
- main.py는 원래 빈 파일이므로 dev/lkh/kmj02의 루트 파일을 유지. hwangsoon의 삭제를 전파하지 않음. Python 3.12 조건은 모두 동일하므로 루트 .python-version 유지.
- 공통 pyproject는 모든 요구 패키지 합집합. torch 인덱스는 두 브랜치가 같은 cu126 URL을 사용하므로 hwangsoon의 Windows marker + explicit 설정 채택. kmj02의 모든 OS 전용 인덱스 설정은 전파하지 않음.
- 원본 README 두 개는 기능별 사용 설명과 상대 링크 보존을 위해 해당 구현 내부에 보존하고, 루트 README를 통합 실행 안내로 작성. 공통 설정을 무조건 복제하지 않음.

## 4. 통합 전략

1. origin/dev에서 로컬 dev 생성.
2. dev의 직접 후손인 lkh를 `git merge --ff-only lkh`로 반영. lkh 원래 루트 경로와 사용자 자산 유지.
3. origin/hwangsoon `23cc75f`, origin/kmj02 `ab45fd6`의 트리에서 공통 설정을 제외한 파일을 각각 implementations 아래로 바이트 그대로 복원.
4. 공통 의존성·ignore·실행 안내를 별도 판단하고 통합 커밋 작성.

일반 merge로 수집기·데이터·환경을 섞는 대신 사용자가 허용한 선택적 파일 복원을 사용했다. hwangsoon/kmj02 원본 커밋은 dev의 merge parent로 연결하지 않고 출처 manifest에 기록했다. 따라서 향후 두 브랜치를 무조건 다시 일반 merge하면 루트 충돌을 다시 만들 수 있다. 후속 반영도 동일 경로 매핑과 출처 비교를 사용해야 한다.

## 5. 최종 구조

```text
jesture_project/
  main.py                         # 기존 빈 진입점 보존
  gesture_model/                  # lkh 원본
  programs/                       # lkh 원본
  util/                           # lkh 원본
  yolo/                           # lkh 원본
  dataset/                        # 사용자 기존 untracked 데이터
  implementations/
    hwangsoon/
      gesture/  gesture_model/  programs/  scripts/
      models/  dataset/  reports/  docs/
      gestures.json  README.md
    kmj02/
      model_yolo/  ckpoint/  runs/  weights/  dataset/
      realtime_detect.py  data_preprocessing.py  slicing.py
      EDA.ipynb  yolov8n.pt  README.md
  integration/
    REPORT.md  FINAL.md  source-manifest.json
  README.md  pyproject.toml  uv.lock  .gitignore  .python-version
```

## 6. 기존 코드 수정 내역

- 기존 Python 소스 수정: **없음**. import, 함수명, 클래스, 알고리즘, 경로 문자열 모두 원본 그대로. 복원 212개 파일은 SHA-256으로 검증.
- `.gitignore`: dev의 기존 제외를 모두 유지하고 분리된 구현의 생성 데이터/개인 로그/모델 JSON 및 `.venv-dev` 규칙 추가. 추적해야 하는 원본 모델·NPZ·결과는 정확한 manifest 경로만 명시적으로 stage.
- `pyproject.toml`: 기존 조건을 유지하며 gradio/customtkinter/scikit-learn/torchvision 합집합, Windows cu126 소스 설정 반영.
- `uv.lock`: hwangsoon lock을 기준으로 gradio dependency graph 추가 후 `uv lock`으로 재생성. hwangsoon lock의 기존 패키지 버전 변경 0개.
- `.python-version`, `main.py`는 변경하지 않음. 기능 로직 변경 없음.

## 7. 새로 추가한 파일

- 새로운 wrapper/adapter/launcher `.py`: **0개**. 기존 실행 파일과 올바른 CWD만으로 연결 가능.
- 브랜치에서 가져온 200개 파일: hwangsoon 138개, kmj02 62개. 전체 경로·역할 분류는 2절, 개별 출처는 source-manifest.json. 재작성한 파일이 아님.
- 루트 README.md: 공통 환경 설치, 각 실행 위치, 외부 모델 준비 안내.
- integration/REPORT.md: 판단·변경·검증·한계 기록.
- integration/source-manifest.json: 보존한 212개 파일의 출처 SHA 및 내용 SHA-256. 누락/대체 여부 확인용.

## 8. 검증 결과

카메라·실제 키 입력·대규모 학습은 실행하지 않는다. 기존 .venv는 torch 2.9.0+cu128 등으로 선언된 환경과 달라 변경하지 않고 TEMP 아래 별도 Python 3.12 환경에 `uv sync --locked`로 125개 패키지를 설치하여 검증했다.

- `uv pip check --python <임시 환경>/Scripts/python.exe`: 125개 패키지 모두 호환.
- `uv lock`: 성공. 기존 hwangsoon 잠금 패키지 버전 변경 없음.
- `UV_PROJECT_ENVIRONMENT=<임시 환경>; uv sync --locked`: 성공. torch 2.14.0+cu126 / torchvision 0.29.0+cu126 등 실제 통합 lock 사용.
- 47개 Python 파일 compile 및 원본 212개 SHA-256 비교 수행.
- 원본 테스트 스위트·lint/type-check 설정은 없음. yolo/test_yolo_webcam.py는 자동 테스트가 아니라 웹캠 프로그램.
- 실제 import, --help 진입점, MediaPipe 모델 초기화, GRU/LSTM 순전파, 체크포인트 로드 및 CPU 가상 프레임 추론은 결과 JSON 참조.
- 최종 결과: lkh 주요 모듈 import와 MediaPipe 초기화 성공. hwangsoon 14개 gesture 모듈 import, GRU/LSTM 각각 (2, 4) 출력, MediaPipe 빈 프레임 처리, NPZ 37개 읽기, FakeGate 상태 전환, UI 모듈 import 성공. kmj02 체크포인트 11개 + 학습 best/last 2개 로드, CPU 추론, Gradio 앱 구성 성공. CLI --help 8개 모두 종료 코드 0.
- 초기 RNN 검증 스크립트에서 CPU 입력과 자동 선택된 CUDA 모델을 혼합해 1회 실패했다. 검증 코드에서 모델을 CPU로 옮겨 재실행 후 성공했으며 저장소 원본 코드는 수정하지 않았다. 실패와 재실행 출력 모두 JSON에 보존.
- lkh detector의 CPU smoke는 kmj02 체크포인트를 명시적 테스트 입력으로만 사용했다. lkh의 기본 모델 경로나 서비스 설정을 교체하지 않았다.

## 9. 남아 있는 문제

- lkh의 기본 static_gesture_v3 데이터 YAML과 yolov8n_v3_rotation/best.pt는 현재 로컬에도 없음. 해당 기본 학습·평가·웹캠 실행에는 원본 자산 필요.
- hwangsoon의 gru_gesture.pt, gru_gesture.json, yolo_gate.pt는 원본 커밋에 없음. 학습된 동적 모델의 정확도·전체 UI 인식 파이프라인은 해당 파일 준비 후 검증 필요.
- kmj02의 학습/평가 이미지·라벨 미포함. 원본 data.yaml의 ../ 경로와 README/EDA의 dataset 하위 구조 불일치 존재. 원본을 바꾸지 않고 README에 별도 YAML/config 사용 방법 안내.
- hwangsoon 과거 보고서 재생성 스크립트는 작성자 PC 절대 경로와 외부 JIN 자료에 의존. 결과물은 보존했으며 주 서비스 진입점과 분리.
- 일부 원본 도구는 import 시 파일 접근/출력을 수행하므로 일괄 import 대신 compile 및 부작용 없는 경로만 실행.
- 웹캠, GPU 동작, 실제 Windows 키 입력, 전체 재학습 및 실제 영상 정확도는 검증 범위 밖.

## 10. 최종 Git 상태

로컬 dev에만 통합하며 push하지 않는다. 원본 lkh 및 모든 origin refs를 변경·삭제하지 않는다. 사용자 dataset/은 의도적으로 untracked 상태로 유지한다. 최종 커밋 SHA·status·충돌 검사 결과는 완료 응답에도 기록한다.

실제 Git 충돌 마커는 0개. 원본 소스의 주석/출력 구분선에는 연속 등호가 포함되어 있으며 이는 merge conflict marker가 아니다. 원본 보존 원칙에 따라 장식용 구분선은 수정하지 않았다. 검증 중 Ultralytics가 만든 settings.json 2개는 임시 검증 폴더로 옮겨 작업 트리에 남기지 않았다.

최종 staging 검증: 원본과 Git blob 동일 212개, 예정된 경로만 stage 209개, 사용자 dataset 238개 내용 동일. `.git`·기존 `.venv`·Python 캐시를 제외한 작업 트리 전체 rg 충돌 마커 검색 결과 없음(exit 1 = 일치 없음).
