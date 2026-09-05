# 제스처 학습 데이터 전처리

`prepare_dataset.py`는 팀원에게 받은 원본 NPZ를 검사하고 LSTM·GRU 공통 입력으로 변환합니다. Python 3.12와 NumPy만 사용하며, 카메라·MediaPipe·TensorFlow·PyTorch를 실행하지 않습니다. 모델 학습과 실시간 추론은 이 프로그램의 범위가 아닙니다.

구현용 요청은 [prepare_dataset_prompt.md](prepare_dataset_prompt.md)에 있습니다. 수집 방법은 [촬영 안내](../programs/README.md)를 확인하세요.

## 현재 데이터

2026-09-05 실제 실행에서 로컬 NPZ 37개를 확인했고 전부 검사를 통과했습니다. 모두 참가자 `user00`의 `swipe_left`이며, v1 12개와 v2 25개입니다. 프레임 수는 20~46, 녹화 시간은 약 0.64~1.50초, 유효 검출률은 약 91.3~100%입니다. 최대 연속 결측은 2프레임이고 중복·충돌은 없었습니다. 이는 해당 시점의 결과이며 코드에 개수를 고정하지 않습니다.

실제 점검 보고서는 [report.json](../artifacts/audit_20260905/report.json), 파일별 내역은 [manifest.csv](../artifacts/audit_20260905/manifest.csv)에 있습니다. 원본 NPZ 37개와 index.csv의 SHA-256 38개를 실행 전후 비교해 변경되지 않았음을 확인했습니다.

현재 데이터로는 파일 검사와 전처리 시험이 가능합니다. `make_fist`, `no_gesture`와 다른 참가자 데이터가 도착해야 3클래스 학습 및 참가자를 분리한 평가 데이터를 만들 수 있습니다.

## 데이터를 받는 방법

팀원은 드라이브에 참가자별로 자기 NPZ만 올립니다. 담당자는 파일명을 유지해 로컬 프로젝트의 다음 위치에 모읍니다.

```text
dataset/
├─ swipe_left/
│  ├─ p001_swipe_left_0001.npz
│  └─ p002_swipe_left_0001.npz
├─ make_fist/
│  └─ p001_make_fist_0001.npz
└─ no_gesture/
   └─ p001_no_gesture_0001.npz
```

- 같은 파일명이 있으면 덮어쓰지 말고 동일 파일의 재전달인지 확인합니다.
- index.csv는 각 팀원에게 받지 않아도 됩니다. 전처리는 NPZ 내부 정보를 읽고 자체 manifest를 만듭니다.
- 참가자 폴더가 추가된 `dataset/p001/swipe_left/…` 구조도 재귀 탐색합니다. 바로 위 폴더는 라벨명으로 유지합니다.
- `user00`이 나중에 `p001`로 등록한 사람과 같다면 담당자가 확인한 뒤 ID 별칭 매핑을 제공합니다. 파일명만 바꾸거나 다른 사람처럼 나누지 않습니다.

## 입력 NPZ

| 필드 | 역할 |
| --- | --- |
| landmarks | `(T,21,3)` 손목 및 손가락 좌표 |
| timestamps | `(T,)` 프레임별 실제 시간, 초 단위 |
| detected | `(T,)` 손 검출 여부 |
| label | swipe_left / make_fist / no_gesture |
| participant_id | 실제 수행자 식별자 |
| sample_id | 참가자와 라벨별 샘플 번호 |

v2의 거리·검출 요약값은 보조 정보입니다. v1에는 일부 필드가 없으므로 품질은 원본 배열에서 다시 계산합니다. world_landmarks는 보존된 원본 정보지만 이번 기본 특징에는 사용하지 않습니다.

입력은 `allow_pickle=False`로 열고, 배열 차원·dtype·필수 스칼라를 검사합니다. ID는 영문/숫자로 시작하고 이후 영문·숫자·밑줄·하이픈만 허용합니다. sample_id는 1 이상의 정수입니다. 파일명은 원본 메타데이터의 `{participant_id}_{label}_{sample_id:04d}.npz`와 정확히 일치해야 합니다. 별칭 매핑을 사용해도 원본 파일명은 바꾸지 않습니다.

버전 필드가 없으면 v1로 처리합니다. v1의 mirrored 누락은 수집 코드에 근거해 false로 가정하고 파일별 내역과 보고서에 기록합니다. v2의 mirrored 누락, mirrored=true, 알 수 없는 버전은 제외합니다. handedness는 있으면 Right여야 합니다. world_landmarks, handedness_per_frame, handedness_confidence는 선택 사항이지만 있으면 구조를 검사합니다.

합성 표시 `synthetic`, `is_synthetic`, `augmented`, `is_augmented`는 bool로 검사하여 true면 제외합니다. 문자열 `augmentation`, `augmentation_type`은 빈 값·none·original 외의 표시가 있으면 제외합니다. 메타데이터에 표시되지 않은 합성 여부를 자동으로 판별하는 기능은 아닙니다.

## 처리 과정

1. NPZ를 읽고 필수 필드, 배열 길이, 라벨·파일명 일치와 중복을 검사합니다.
2. 검출되었고 모든 좌표가 유한한 프레임만 유효한 것으로 판단합니다. 기본 품질 기준은 최소 20프레임, 유효 검출률 80% 이상, 연속 결측 최대 5프레임, 0.6~2.5초입니다.
3. 짧은 결측 구간은 실제 timestamps 기준 선형 보간으로 채웁니다. 양끝 결측은 가장 가까운 유효 좌표로 채웁니다. 전부 결측인 파일은 제외합니다.
4. 시퀀스를 실제 시간축의 등간격 32시점으로 변환합니다. 원래 20프레임이든 46프레임이든 출력 길이는 같습니다.
5. 손목 기준 상대 손 모양과 시작점 대비 손목 이동을 함께 사용해 프레임당 66개 특징을 만듭니다.
6. 실제 참가자 기준으로 train/validation/test를 나누고 NumPy 배열과 추적용 보고서를 저장합니다.

작은 손목 이동만으로 제외하지 않습니다. make_fist는 손가락 변화가 핵심이고, no_gesture는 정지가 정상입니다. swipe_left도 이동 크기가 다양할 수 있습니다. 좌우 반전·시간축 반전 증강은 이번 전처리 기본 과정에 포함하지 않습니다.

현재 영상은 좌우 반전하지 않으므로 수행자 기준 왼쪽 이동인 swipe_left의 저장 x 이동은 대개 양수입니다. dx 부호나 low_motion_warning을 제외 조건으로 사용하지 않습니다. timestamp는 유한하고 엄격히 증가해야 하며, 오류를 정렬로 숨기지 않습니다. detected=true라도 좌표 하나에 NaN/Inf가 있으면 해당 프레임 전체를 결측으로 셉니다.

## 66개 특징의 의미

크기 기준 `s`는 원래 유효 프레임에서 계산한 손목(0)~중지 MCP(9) xy 거리의 양수 중앙값입니다. 시퀀스마다 하나를 정해 고정적으로 사용합니다.

| 특징 위치 | 계산 | 목적 |
| --- | --- | --- |
| 0~62 | `(랜드마크 xyz - 현재 손목 xyz) / s` | 손가락 20개를 포함한 손 모양과 변화 보존 |
| 63~64 | `(현재 손목 xy - 첫 출력 프레임 손목 xy) / s` | 위치 차이를 줄이면서 스와이프 이동 보존 |
| 65 | 현재 손목~중지 MCP xy 거리 / s | 손 크기의 상대 변화 보존 |

이 상대좌표에서 손목 자신의 xyz는 0이지만, 손목의 이동 정보는 63~64번 특징에 별도로 들어갑니다. `world_landmarks`와 영상 좌표를 혼합하지 않습니다.

이 수식은 초기 실험을 위한 특징 설계입니다. 저장 x/y는 각각 영상 폭/높이 기준으로 정규화되어 있고 물리적 거리 단위가 아닙니다. 기존 파일에는 영상 크기가 없어 실제 종횡비 보정을 할 수 없습니다. 손 모양·방향에 따른 크기 기준의 변화도 있으므로 학습 결과로 설계를 검증해야 합니다.

32시점으로 맞추면 동작의 원래 소요 시간이 모델 입력에 직접 남지는 않습니다. 원본 duration은 manifest에 기록합니다. 이후 실시간 추론에서도 동일한 전처리 함수와 설정을 사용해야 합니다.

## 참가자 기준 분리

예를 들어 실제로 서로 다른 여섯 명의 자료가 모두 도착했다면 다음과 같이 지정할 수 있습니다.

| 용도 | 참가자 예시 |
| --- | --- |
| train | p001, p002, p003, p004 |
| validation | p005 |
| test | p006 |

이 배정은 예시입니다. 담당자가 참가자별 데이터 구성을 확인해 결정합니다. 각 split에 세 클래스가 모두 있어야 하며, 같은 사람이 두 split에 들어가면 안 됩니다. 현재 user00 한 명을 파일별로 무작위 분리해 이 조건을 우회하지 않습니다.

별칭 매핑이 필요하면 담당자가 확인한 관계만 JSON으로 제공합니다. 예를 들어 user00과 p001이 같은 사람인 경우에만 `{"user00": "p001"}`을 사용합니다. 매핑 후 파일 식별자 충돌도 검사합니다.

JSON의 키 중복, 잘못된 ID, 별칭의 연쇄·순환 매핑은 오류입니다. `a → b → c` 대신 `{"a":"c", "b":"c"}`처럼 최종 ID로 직접 연결하세요. split에 입력한 별칭도 canonical ID로 바꾼 뒤 중복과 참가자 누수를 검사합니다. 없는 ID, 채택된 자료가 없는 ID, 미배정 참가자, 빈 split, split 내 같은 ID 중복도 오류입니다.

## 중복과 충돌

파일 SHA-256 외에 landmarks·timestamps·detected 배열 해시와 canonical 참가자·라벨·sample_id를 함께 비교합니다. 배열 해시는 숫자를 float64로 통일하고 NaN 표현과 -0을 정규화하므로 NPZ 압축 방식만 다른 복사본도 찾습니다. 원래의 절대 timestamp는 해시에 포함합니다. world_landmarks 등 선택 메타데이터는 중복 판정의 기준 배열이 아닙니다.

- 같은 식별자와 같은 필수 배열: 상대경로 정렬상 첫 번째 유효 파일만 채택하고 나머지는 duplicate로 기록합니다.
- 같은 식별자에 다른 필수 배열: conflict입니다.
- 같은 필수 배열을 다른 식별자·라벨로 제출: conflict입니다. 확인된 ID 별칭으로 같은 식별자가 된 경우만 복사본으로 처리합니다.

충돌한 파일은 모두 conflict로 표시하고, 어느 쪽이 올바른지 추측하지 않습니다. 충돌이 하나라도 있으면 normal 학습 배열 출력을 중단하며 audit-only도 실패로 보고합니다. 원본 데이터를 수정하거나 삭제하지 않으므로 담당자가 보고서를 보고 별도로 해결해야 합니다.

## 실행 명령

아래 명령은 프로젝트 루트에서 실행합니다. `--input-dir`를 생략하면 현재 작업 디렉터리와 무관하게 스크립트가 속한 프로젝트의 dataset을 읽습니다. 명시적으로 입력한 상대경로와 output-dir은 현재 작업 디렉터리를 기준으로 해석합니다.

현재 데이터 점검용 예시입니다. 학습 배열은 생성하지 않습니다.

```powershell
.\.venv\Scripts\python.exe .\training\prepare_dataset.py --audit-only --output-dir .\artifacts\audit_001
```

여섯 명의 세 클래스 데이터가 모두 준비된 후 사용할 예시입니다.

```powershell
.\.venv\Scripts\python.exe .\training\prepare_dataset.py --input-dir .\dataset --output-dir .\artifacts\prepared_001 --seq-len 32 --train-subjects p001 p002 p003 p004 --val-subjects p005 --test-subjects p006
```

필요한 경우 `--participant-map .\participant_map.json`을 추가하고, 그 JSON 파일에는 확인된 별칭 관계만 입력합니다. 기존 폴더는 비어 있어도 덮어쓰지 않으므로 매 실행마다 새로운 output-dir을 사용합니다. 출력 폴더를 입력 데이터 폴더 또는 프로젝트 dataset 내부에 지정할 수 없습니다.

### 옵션과 기본값

| 옵션 | 기본값 / 허용 범위 |
| --- | --- |
| --input-dir | 스크립트 기준 프로젝트 dataset |
| --output-dir | 필수, 기존에 없는 데이터 폴더 밖 경로 |
| --audit-only | 생략 시 normal 모드 |
| --seq-len | 32 / 2 이상 정수 |
| --min-frames | 20 / 2 이상 정수 |
| --min-detection-rate | 0.8 / 0~1 비율 |
| --max-missing-run | 5 / 0 이상 정수 |
| --min-duration | 0.6초 / 0보다 큼 |
| --max-duration | 2.5초 / min-duration 이상 |
| --scale-epsilon | 0.000001 / 0보다 큼 |
| --participant-map | 선택, 원본 ID → canonical ID JSON |
| --train-subjects / --val-subjects / --test-subjects | 공백 구분 ID 목록, 자동 분리 없음 |

모든 실수 threshold는 NaN/Inf가 아닌 유한한 값이어야 합니다. 조건을 완화하면 더 많은 자료를 사용할 수 있지만, 보고서의 제외 사유를 확인한 뒤 변경하세요. 크기 s는 epsilon보다 커야 합니다.

## 출력 결과

```text
artifacts/prepared_001/
├─ X_train.npy
├─ y_train.npy
├─ X_val.npy
├─ y_val.npy
├─ X_test.npy
├─ y_test.npy
├─ label_map.json
├─ preprocessing_config.json
├─ manifest.csv
└─ report.json
```

- X는 float32 `(샘플 수,32,66)`, y는 int64 `(샘플 수,)`입니다. X의 i번째 샘플과 y의 i번째 정답이 대응합니다.
- 라벨 번호는 swipe_left=0, make_fist=1, no_gesture=2로 고정합니다.
- manifest는 발견한 파일당 한 행입니다. source_path는 입력 폴더 기준 상대경로이고, sha256은 파일 해시입니다. arrays_sha256과 identity_content_sha256은 중복 판정용입니다. 원본/매핑 ID, 라벨, sample_id, schema, 품질과 scale, status/reason, split과 해당 배열의 0부터 시작하는 output_index를 기록합니다. 입력 오류로 얻지 못한 값은 빈 칸입니다.
- status는 accepted(전처리 가능), excluded(검증/품질 제외), duplicate(중복), conflict(충돌)입니다. report에는 참가자·라벨별 input/accepted/excluded/duplicate/conflict 개수, 충돌 원인, 가정·경고와 split별 클래스 분포를 기록합니다. 각 상태 개수는 서로 배타적이며 합이 input입니다.
- preprocessing_config에는 보간·특징 수식·길이·품질 기준·참가자 배정을 저장합니다.

audit-only는 네 가지 메타데이터 파일만 생성합니다. 이때 accepted는 **전처리 가능**이라는 뜻이지 학습에 사용했다는 뜻은 아닙니다. split과 output_index는 비워 둡니다. 클래스 부족이나 참가자 분리 오류가 있는 normal 실행도 보고서만 남기고 배열 생성을 중단합니다. 정상 배열이 생성될 때만 split/output_index를 부여합니다.

보고서의 success와 training_arrays_written을 함께 확인하세요. audit-only는 split 요건이 부족해도 유효한 자료가 있고 치명적 충돌이 없으면 success=true이며, split_errors와 warnings에 부족한 조건을 남깁니다. 파일별 제외가 있어도 나머지 파일을 계속 검사합니다. 유효 자료가 전혀 없거나 충돌이 있으면 실패입니다.

종료 코드는 성공 0, 점검/분리 실패(보고서 저장) 1, 잘못된 CLI·경로·매핑 또는 I/O 오류 2입니다. CLI·경로 오류는 스캔 전에 거부하므로 보고서가 없을 수 있습니다. 결과는 임시 형제 폴더에서 완성한 뒤 한 번에 공개하며, 저장 I/O 오류가 발생하면 일부 배열을 최종 output-dir에 남기지 않습니다. 보고서조차 저장할 수 없는 I/O 오류는 콘솔에 표시합니다.

## 결과 확인 기준

담당자는 보고서에서 세 클래스가 각 split에 존재하는지, 참가자가 겹치지 않는지, 제외 사유가 예상 범위인지 확인합니다. 코드에서는 X/y의 샘플 수, manifest와의 대응, `(N,32,66)` 형상과 NaN/Inf 부재를 검사합니다.

이 출력은 LSTM과 GRU에 공통으로 사용할 전처리 데이터입니다. 학습·모델 성능 평가·연속 영상의 동작 구간 검출은 별도 단계입니다. 원본 NPZ와 수집용 index.csv는 유지합니다.

## 테스트와 함수 재사용

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_*.py" -v
```

테스트는 임시 폴더에서만 NPZ를 만듭니다. v1/v2, 불규칙 시간축, 결측/양끝 보간, 비정상 timestamp·dtype·스키마, 크기 0, 정지 동작 보존, 참가자 누수, 클래스 누락, 별칭, 중복·충돌, X/y와 manifest 행 대응, 원본 보존과 저장 오류를 검사합니다. 다중 참가자·3클래스 normal 모드 검증은 파일 입출력과 분리 로직에 대한 합성 테스트이며 실제 모델 성능을 의미하지 않습니다.

다른 Python 코드에서는 다음 순수 함수를 재사용할 수 있습니다. import만으로 파일이나 카메라를 열지 않습니다.

```python
from training.prepare_dataset import QualityConfig, preprocess_sequence

features, metadata = preprocess_sequence(
    landmarks, timestamps, detected,
    seq_len=32,
    config=QualityConfig(),
)
# features: float32 (32, 66), metadata: duration, 검출률, scale 등
```

이 함수는 잘라 둔 단일 시퀀스를 전처리합니다. 실시간 영상에서 동작 시작·끝을 찾는 기능은 포함하지 않습니다. 모델을 학습한 이후에는 학습 결과의 preprocessing_config와 같은 길이·품질·특징 설정을 사용해야 합니다.

## LSTM 분류 모델 (`models.py`)

`models.py`의 `LSTMClassifier`는 PyTorch 기반 단방향 LSTM 모델 정의만 제공합니다. 전처리된 특징을 받아 클래스별 점수(logits)를 반환하며, NPZ/NPY 로딩·전처리·학습·실시간 구간 감지는 하지 않습니다. import만으로 모델을 생성하거나 파일/카메라를 열지 않습니다. GRU는 구현하지 않았습니다.

기본 구조는 `입력 → LSTM → 최상위 층 최종 hidden state → Dropout → Linear → logits`입니다.

| 생성자 인자 | 기본값 | 의미 / 허용 범위 |
| --- | --- | --- |
| input_size | 66 | 프레임당 특징 개수 / 양의 int |
| hidden_size | 64 | LSTM 상태 벡터 크기 / 양의 int |
| num_layers | 1 | LSTM 층 수 / 양의 int |
| num_classes | 3 | 클래스 수 / 양의 int |
| dropout | 0.2 | 드롭아웃 비율 / 유한한 실수, 0 이상 1 미만 |

bool은 크기/비율 설정으로 허용하지 않습니다. 위 값은 초기 실험 설정이며 최적 설정이 아닙니다. 현재 데이터 계약에서는 input_size=66, num_classes=3을 사용하고 라벨 번호는 swipe_left=0, make_fist=1, no_gesture=2입니다. 다른 값으로 모델을 생성할 수는 있지만 실제 전처리 설정과 라벨 매핑에 맞춰야 합니다.

### 입력, 출력 및 상태

- 입력 X: float32 `(batch_size, seq_len, 66)`. 기본 전처리 길이는 32입니다.
- 출력: `(batch_size, 3)` logits. 배치 크기가 1이어도 배치 차원을 유지합니다.
- 정답 y: 모델 입력이 아니라 손실 함수에 전달하는 int64 `(batch_size,)`입니다.
- 모델은 양의 seq_len을 처리하지만 한 배치 내 모든 샘플 길이는 같아야 합니다. padding/packing은 없습니다. 길이 변경이 기술적으로 가능하다는 뜻이며, 실제 사용은 학습 당시 전처리 설정에 맞춥니다.
- no_gesture와 작은 이동량의 시퀀스도 정상 입력입니다. 모델은 움직임 필터나 좌표 정규화를 추가하지 않습니다.

LSTM의 전체 output은 최상위 층의 각 시점 출력이고, h_n은 각 층의 마지막 시점 상태입니다. 분류에는 최상위 층에서 시퀀스 전체를 읽은 `h_n[-1]`을 사용합니다. 매 forward 호출마다 초기 hidden/cell 상태는 0이며, 서로 다른 녹화나 배치 사이에 상태를 유지하지 않습니다.

LSTM 내부 dropout은 층 사이에 적용하므로 num_layers=1이면 0으로 설정합니다. 별도의 분류기 앞 Dropout은 단층에서도 적용합니다. `model.train()`에서는 활성화되고 `model.eval()`에서는 비활성화됩니다.

### 모델 생성과 forward 예시

프로젝트 루트에서 import하는 예시입니다. 랜덤 입력이므로 출력은 실제 제스처 예측 결과가 아닙니다.

```python
import torch
from training.models import LSTMClassifier

model_config = dict(
    input_size=66, hidden_size=64, num_layers=1,
    num_classes=3, dropout=0.2,
)
model = LSTMClassifier(**model_config)  # 기본 float32/CPU
x = torch.randn(4, 32, 66, dtype=torch.float32)
y = torch.tensor([0, 1, 2, 0], dtype=torch.int64)

model.train()
logits = model(x)  # (4, 3), softmax 적용 전 점수
loss = torch.nn.CrossEntropyLoss()(logits, y)
# 예시는 손실 계산까지만 수행합니다. 학습 루프나 optimizer 갱신은 없습니다.

model.eval()
with torch.no_grad():
    logits = model(x)
    probabilities = logits.softmax(dim=-1)  # 확률이 필요할 때 호출 측에서 계산
```

CrossEntropyLoss에는 softmax 결과가 아니라 logits를 전달합니다. 모델 내부에는 softmax, argmax, loss 계산이 없습니다. 입력은 부동소수점 Tensor여야 하며, 잘못된 차원/특징 수나 빈 배치·시퀀스는 오류입니다. 전처리에서 유한한 특징을 준비해야 합니다.

모델과 입력의 dtype/device는 호출 측에서 맞춥니다. 위 예시는 float32/CPU이며, 모델 내부에서 입력을 reshape하거나 dtype/device를 자동 변경하지 않습니다. GPU 사용이나 다른 dtype이 필요하면 호출 측에서 모델과 입력을 함께 이동/변환하고 실행 환경의 지원 여부를 확인해야 합니다.

### 향후 학습 코드와 체크포인트

추후 `train.py`는 전처리 결과·label_map/config 로딩, DataLoader, device 선택, loss/optimizer, 학습·검증 루프 및 체크포인트 저장을 담당합니다. 이번에는 구현하지 않았습니다. 모델의 파라미터 초기화는 PyTorch 기본 동작을 사용하며, 재현성을 위한 seed 설정은 호출 측/테스트에서 수행합니다.

체크포인트에는 `state_dict`와 생성에 사용한 `model_config`를 함께 보관해야 합니다. 특히 dropout 설정은 state_dict만으로 복원되지 않습니다. 예를 들어 `{"model_config": model_config, "state_dict": model.state_dict()}` 형태로 저장하고, 불러올 때 같은 설정으로 모델을 생성한 뒤 `load_state_dict()`를 호출합니다. 평가 시에는 `eval()`도 별도로 호출합니다. 실제 학습 결과에는 label_map과 preprocessing_config도 함께 보관해야 합니다.

### 모델 테스트

프로젝트 가상 환경에 PyTorch가 설치되어 있어야 합니다. 모델 테스트만 실행하려면:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_models.py" -v
```

전처리와 모델 테스트를 함께 실행하려면:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_*.py" -v
```

`test_models.py`는 CPU 합성 텐서로 출력 shape, 단층/다층, 가변 길이, 입력/설정 오류, dropout 구성, 최상위 hidden 선택, logits 반환, CrossEntropyLoss 역전파와 유한한 gradient, eval 재현성, 배치 상태 독립성, 정지 입력, dtype 계약, 임시 체크포인트 저장·복원을 검사합니다. 실제 데이터 학습과 제스처 분류 성능 평가는 아직 수행하지 않았습니다. 테스트 성공은 모델 구조와 인터페이스 검증 결과이지 실제 정확도를 의미하지 않습니다.
