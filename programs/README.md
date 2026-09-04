# 제스처 시계열 데이터 수집 프로그램

이 폴더에는 MediaPipe Hand Landmarker로 오른손 제스처의 프레임별 랜드마크를 수집하고 저장된 NPZ 파일을 확인하는 프로그램이 있습니다.

## 파일 구성

- `collect_gesture.py`: 웹캠으로 제스처 시퀀스를 수집해 NPZ로 저장합니다.
- `check.py`: 저장된 NPZ의 기본 정보와 손목 이동 경로를 확인합니다.

프로그램은 프로젝트 루트의 다음 경로를 사용합니다.

```text
jesture_project/
├─ dataset/
├─ models/
│  └─ hand_landmarker.task
└─ programs/
   ├─ collect_gesture.py
   ├─ check.py
   └─ README.md
```

## 실행 환경

- Windows 11
- Python 3.12
- 웹캠
- `models/hand_landmarker.task` 모델 파일

프로젝트 루트에서 가상 환경과 의존성을 설치합니다.

### uv 사용

```powershell
python -m pip install uv
uv sync
```

### 일반 venv 사용

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## 참가자 ID 규칙

`participant_id`는 데이터를 수집한 팀원이 아니라 실제로 제스처를 수행한 사람을 식별합니다. 실명은 사용하지 않고 팀에서 중복되지 않는 익명 ID를 할당합니다.

예시:

```text
p001
p002
p003
p004
p005
p006
```

현재 6명의 팀원에게 다음과 같이 고유 ID를 하나씩 할당합니다.

```text
팀원 1: p001
팀원 2: p002
팀원 3: p003
팀원 4: p004
팀원 5: p005
팀원 6: p006
```

각 팀원은 자신에게 할당된 ID만 사용합니다. 추후 팀원이 아닌 다른 참가자의 데이터를 추가로 수집하면 `p007`부터 새로운 ID를 부여합니다.

ID에는 영문, 숫자, 하이픈, 밑줄만 사용할 수 있습니다.

## 제스처 라벨 규칙

팀원 모두 동일한 라벨을 입력해야 합니다. 대소문자와 구분 기호가 다르면 별도 클래스로 저장되므로 아래 표기를 그대로 사용합니다.

```text
swipe_left
finger_snap
make_fist
```

라벨에도 영문, 숫자, 하이픈, 밑줄만 사용할 수 있습니다.

## 데이터 수집 실행

프로젝트 루트에서 실행합니다.

### uv 사용

```powershell
uv run python .\programs\collect_gesture.py
```

### 가상 환경 사용

```powershell
.\.venv\Scripts\python.exe .\programs\collect_gesture.py
```

실행 후 참가자 ID와 제스처 라벨을 입력합니다.

```text
Participant ID: p001
Gesture Label: swipe_left
```

현재 수집기는 오른손 한 손 제스처를 대상으로 합니다. 화면에는 가능하면 오른손만 보이게 합니다. 왼손이 주로 검출된 시퀀스는 저장되지 않습니다.

웹캠이 열리지 않으면 `collect_gesture.py`의 `CAMERA_INDEX` 값을 `0`, `1`, `2` 순서로 변경해 확인합니다.

## 조작 방법

| 키 | 기능 |
| --- | --- |
| `Space` | 녹화 시작 또는 종료 후 저장 |
| `R` | 현재 녹화 취소 및 초기화 |
| `Q` | 프로그램 종료 |

녹화 중 종료하면 현재 시퀀스는 저장되지 않습니다.

## 저장 품질 기준

다음 조건을 모두 만족한 시퀀스만 저장됩니다.

- 최소 20프레임
- 손 검출률 80% 이상
- 연속 미검출 최대 5프레임
- 녹화 시간 0.6초 이상 2.5초 이하
- 주로 검출된 손이 오른손
- 버퍼 길이와 타임스탬프가 정상

손목 이동량이 작으면 경고를 표시하지만 데이터는 저장합니다. 짧은 제스처가 의도된 동작이면 그대로 사용하고, 녹화 실수라고 판단되면 다시 수집합니다.

## 저장 위치와 파일명

데이터는 라벨별 폴더에 저장됩니다.

```text
dataset/<label>/<participant_id>_<label>_<sample_id>.npz
```

예시:

```text
dataset/swipe_left/p001_swipe_left_0001.npz
dataset/swipe_left/p001_swipe_left_0002.npz
```

같은 참가자와 라벨의 기존 번호를 확인해 다음 번호를 사용하므로 동일한 컴퓨터에서는 기존 파일을 덮어쓰지 않습니다.

## NPZ 데이터 구성

`landmarks`와 `world_landmarks`에는 프레임마다 손 랜드마크 21개의 x, y, z 좌표가 저장됩니다.

```text
landmarks.shape = (프레임 수, 21, 3)
```

주요 필드:

| 필드 | 설명 |
| --- | --- |
| `landmarks` | 영상 기준 손 랜드마크 좌표 |
| `world_landmarks` | MediaPipe가 추정한 공간 좌표 |
| `timestamps` | 녹화 시작 후 프레임별 시간 |
| `detected` | 프레임별 손 검출 여부 |
| `label` | 제스처 라벨 |
| `participant_id` | 익명 참가자 ID |
| `sample_id` | 참가자와 라벨별 샘플 번호 |
| `handedness` | 시퀀스에서 주로 검출된 손 |
| `detection_rate` | 검출률 백분율 |
| `valid_frames` | 손이 검출된 프레임 수 |
| `longest_missing_run` | 최대 연속 미검출 프레임 수 |
| `wrist_dx`, `wrist_dy` | 손목의 시작점 대비 수평·수직 이동량 |
| `net_displacement` | 손목 시작점과 종료점 사이의 직선거리 |
| `path_length` | 손목이 이동한 전체 경로 길이 |
| `normalized_displacement` | 손 크기로 정규화한 직선 이동량 |
| `normalized_path_length` | 손 크기로 정규화한 경로 길이 |
| `low_motion_warning` | 이동량 경고 발생 여부 |
| `mirrored` | 수집 좌표의 좌우 반전 여부. 현재 `False` |
| `collection_schema_version` | 수집 데이터 스키마 버전 |

이 NPZ는 MediaPipe의 공식 파일 형식이 아니라 MediaPipe 결과를 NumPy 배열로 변환한 프로젝트 전용 원본 데이터 형식입니다. 추후 별도의 전처리 프로그램에서 결측값 처리, 좌표 정규화, 고정 길이 리샘플링을 수행한 뒤 LSTM 또는 GRU 학습에 사용합니다.

## 저장 데이터 확인

`check.py` 상단의 `filepath`를 확인할 NPZ 경로로 변경합니다.

```python
filepath = (
    PROJECT_ROOT
    / "dataset"
    / "swipe_left"
    / "p001_swipe_left_0001.npz"
)
```

프로젝트 루트에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe .\programs\check.py
```

터미널에는 NPZ 필드, 라벨, 참가자, 프레임 수, 좌표 형상과 검출률이 출력됩니다. 그래프의 초록색 점은 손목 시작점이고 빨간색 점은 종료점입니다.

## 팀 데이터 전달 및 병합

각 팀원은 자신이 수집한 `dataset/<label>/*.npz` 파일을 전달합니다. `dataset/index.csv`는 각 컴퓨터에서 별도로 생성되므로 다른 팀원의 파일로 덮어쓰지 않습니다.

병합 담당자는 다음 순서로 처리합니다.

1. 참가자 ID와 라벨이 팀 규칙에 맞는지 확인합니다.
2. 라벨별 폴더에 NPZ 파일을 복사합니다.
3. 같은 파일명이 존재하면 덮어쓰지 말고 참가자 ID를 확인합니다.
4. 모든 NPZ 병합 후 전체 파일을 기준으로 `index.csv`를 다시 생성합니다.

원본과 좌우 반전 또는 시간축 반전으로 만든 증강 데이터는 먼저 참가자 기준으로 학습·검증·테스트 그룹을 나눈 뒤 학습 데이터에만 추가합니다. 검증 및 테스트에는 실제 수집 데이터를 사용합니다.

## 주의사항

- 수집 중 카메라 거리와 조명을 지나치게 고정하지 말고 자연스러운 변화를 포함합니다.
- 제스처 시작 전과 종료 후 불필요하게 오래 정지하지 않습니다.
- 참가자 ID에 이름, 전화번호 등 개인정보를 넣지 않습니다.
- NPZ 파일과 `index.csv`를 임의로 편집하지 않습니다.
- 파일 전달 전 라벨, 참가자 ID와 샘플 개수를 확인합니다.
