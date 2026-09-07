# 제스처 학습 데이터 전처리

2026-09-06 실제 자료로 첫 예비 LSTM 학습을 완료했습니다. 승인된 번호 정리·중복 제외,
참가자 배정 및 결과는 [예비 학습 기록](../artifacts/pilot_20260906/README.md)을 확인하세요.
아래의 과거 데이터 개수와 “합성 테스트만 실행” 설명은 각 코드 구현 당시의 기록입니다.
이번 예비 실험에서도 최종 test 평가와 실시간 성능 평가는 하지 않았습니다.

`prepare_dataset.py`는 팀원에게 받은 원본 NPZ를 검사하고 LSTM·GRU 공통 입력으로 변환합니다. Python 3.12와 NumPy만 사용하며, 카메라·MediaPipe·TensorFlow·PyTorch를 실행하지 않습니다. 모델 학습과 실시간 추론은 이 프로그램의 범위가 아닙니다.

구현용 요청은 [prepare_dataset_prompt.md](prepare_dataset_prompt.md)에 있습니다. 수집 방법은 [촬영 안내](../programs/README.md)를 확인하세요.

## 현재 데이터

오른손 원본 데이터만 사용하며 좌우 반전 증강은 하지 않습니다. 공통 라벨 계약은 `gesture_schema.py`에 정의합니다.

| 번호 | 클래스 |
| --- | --- |
| 0 | swipe_left |
| 1 | make_fist |
| 2 | no_gesture |
| 3 | finger_snap |

새 전처리는 네 클래스를 모두 검사하며 각 split에 네 클래스가 있어야 합니다. 입력 특징과 수식은 그대로이므로 전처리 버전은 1.0.0을 유지하고, 모델 출력 의미는 저장된 label_map으로 구분합니다. LSTM 기본 출력은 4개이며 train.py는 실제 label_map 크기로 출력층을 구성합니다. 평가 보고서·confusion matrix·예측 확률 CSV에도 finger_snap을 포함합니다.

기존 오른손 3클래스 결과/체크포인트는 호환하지만 finger_snap을 추론할 수 없습니다. 새 모델은 네 클래스 데이터로 **새 출력 폴더에 새로 학습**하세요. 예전 swipe_right=3인 왼손 모델은 지원하지 않으며, 메타데이터 이름만 바꿔 재사용하면 안 됩니다. 실제 신규 학습·성능 평가는 이번 코드 수정에서 수행하지 않았습니다.

### 현재 원본 점검 및 실행 순서

2026-09-07 최신 확인: 사용자가 user00 자료를 삭제한 뒤 NPZ는 1,604개입니다. 네 클래스 모두 p001~p006에 존재하며 저장된 handedness는 모두 Right입니다.

번호만 다른 동일 참가자·라벨의 복사본은 자동 중복 제외합니다. 현재 기본 기준의 읽기 전용 점검 결과는 accepted 1,575개, duplicate 1개, 길이 제한 2.5초 초과 excluded 28개, conflict 0개입니다. p004_make_fist_0033은 0001의 중복으로 제외하고, 0002/0034는 둘 다 시간 기준 미달이라 품질 제외 상태를 유지합니다. 원본은 수정/삭제하지 않았습니다. 아래 참가자 배정으로 각 split에 네 클래스가 모두 남아 있습니다.

아래 참가자 배정은 기존 val=p004/test=p006을 유지하고 p005를 train에 추가한 **실행 예시**입니다. 결과 폴더가 이미 있으면 새 이름을 사용하세요. 먼저 audit-only 결과를 확인합니다.

```powershell
.\.venv\Scripts\python.exe -m training.prepare_dataset --input-dir .\dataset --output-dir .\artifacts\four_class_audit_001 --participant-map .\training\participant_map.json --train-subjects p001 p002 p003 p005 --val-subjects p004 --test-subjects p006 --audit-only
```

전처리 → 성공 여부 확인 → 학습 → validation 평가 순서로 실행합니다. CUDA를 사용하려면 해당 가상 환경에서 torch.cuda.is_available()이 True여야 합니다. 아래 명령은 자동 실행하지 않았습니다.

```powershell
.\.venv\Scripts\python.exe -m training.prepare_dataset --input-dir .\dataset --output-dir .\artifacts\four_class_data_001 --participant-map .\training\participant_map.json --train-subjects p001 p002 p003 p005 --val-subjects p004 --test-subjects p006
.\.venv\Scripts\python.exe -m training.train --data-dir .\artifacts\four_class_data_001 --output-dir .\artifacts\four_class_lstm_001 --device cuda --num-layers 2
.\.venv\Scripts\python.exe -m training.evaluate --data-dir .\artifacts\four_class_data_001 --run-dir .\artifacts\four_class_lstm_001 --output-dir .\artifacts\four_class_eval_val_001 --split val --device auto
```

최종 모델 선택 전에는 test를 튜닝에 사용하지 마세요. 새 체크포인트를 서비스에 지정하는 방법은 service/GUI_README.md에 있습니다.

### 초기 자료 점검 기록

2026-09-05 실제 실행에서 로컬 NPZ 37개를 확인했고 전부 검사를 통과했습니다. 모두 참가자 `user00`의 `swipe_left`이며, v1 12개와 v2 25개입니다. 프레임 수는 20~46, 녹화 시간은 약 0.64~1.50초, 유효 검출률은 약 91.3~100%입니다. 최대 연속 결측은 2프레임이고 중복·충돌은 없었습니다. 이는 해당 시점의 결과이며 코드에 개수를 고정하지 않습니다.

실제 점검 보고서는 [report.json](../artifacts/audit_20260905/report.json), 파일별 내역은 [manifest.csv](../artifacts/audit_20260905/manifest.csv)에 있습니다. 원본 NPZ 37개와 index.csv의 SHA-256 38개를 실행 전후 비교해 변경되지 않았음을 확인했습니다.

당시에는 자료가 부족해 파일 검사와 전처리 시험만 가능했습니다. 현재 네 클래스 자료 상태는 위 최신 점검 구역을 참고하세요.

## 데이터를 받는 방법

팀원은 드라이브에 참가자별로 자기 NPZ만 올립니다. 담당자는 파일명을 유지해 로컬 프로젝트의 다음 위치에 모읍니다.

```text
dataset/
├─ swipe_left/
│  ├─ p001_swipe_left_0001.npz
│  └─ p002_swipe_left_0001.npz
├─ make_fist/
│  └─ p001_make_fist_0001.npz
├─ finger_snap/
│  └─ p001_finger_snap_0001.npz
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

이 배정은 예시입니다. 담당자가 참가자별 데이터 구성을 확인해 결정합니다. 각 split에 네 클래스가 모두 있어야 하며, 같은 사람이 두 split에 들어가면 안 됩니다. 현재 user00 한 명을 파일별로 무작위 분리해 이 조건을 우회하지 않습니다.

별칭 매핑이 필요하면 담당자가 확인한 관계만 JSON으로 제공합니다. 예를 들어 user00과 p001이 같은 사람인 경우에만 `{"user00": "p001"}`을 사용합니다. 매핑 후 파일 식별자 충돌도 검사합니다.

JSON의 키 중복, 잘못된 ID, 별칭의 연쇄·순환 매핑은 오류입니다. `a → b → c` 대신 `{"a":"c", "b":"c"}`처럼 최종 ID로 직접 연결하세요. split에 입력한 별칭도 canonical ID로 바꾼 뒤 중복과 참가자 누수를 검사합니다. 없는 ID, 채택된 자료가 없는 ID, 미배정 참가자, 빈 split, split 내 같은 ID 중복도 오류입니다.

## 중복과 충돌

파일 SHA-256 외에 landmarks·timestamps·detected 배열 해시와 canonical 참가자·라벨·sample_id를 함께 비교합니다. 배열 해시는 숫자를 float64로 통일하고 NaN 표현과 -0을 정규화하므로 NPZ 압축 방식만 다른 복사본도 찾습니다. 원래의 절대 timestamp는 해시에 포함합니다. world_landmarks 등 선택 메타데이터는 중복 판정의 기준 배열이 아닙니다.

- 같은 대표 참가자·라벨과 같은 필수 배열: sample_id가 달라도 상대경로 정렬상 첫 번째 유효 파일만 채택하고 나머지는 duplicate로 기록합니다. reason의 duplicate_of에 채택한 파일 경로를 남기며 원본은 삭제하지 않습니다. 품질 미달 파일은 기존 excluded 사유를 유지합니다.
- 같은 식별자에 다른 필수 배열: conflict입니다.
- 같은 필수 배열을 다른 대표 참가자 또는 다른 라벨로 제출: conflict입니다. ID 별칭은 이 검사 전에 적용합니다. 참가자 누수나 잘못된 라벨을 중복 제거로 숨기지 않습니다.

충돌한 파일은 모두 conflict로 표시하고, 어느 쪽이 올바른지 추측하지 않습니다. 충돌이 하나라도 있으면 normal 학습 배열 출력을 중단하며 audit-only도 실패로 보고합니다. 원본 데이터를 수정하거나 삭제하지 않으므로 담당자가 보고서를 보고 별도로 해결해야 합니다.

## 실행 명령

아래 명령은 프로젝트 루트에서 실행합니다. `--input-dir`를 생략하면 현재 작업 디렉터리와 무관하게 스크립트가 속한 프로젝트의 dataset을 읽습니다. 명시적으로 입력한 상대경로와 output-dir은 현재 작업 디렉터리를 기준으로 해석합니다.

현재 데이터 점검용 예시입니다. 학습 배열은 생성하지 않습니다.

```powershell
.\.venv\Scripts\python.exe .\training\prepare_dataset.py --audit-only --output-dir .\artifacts\audit_001
```

여섯 명의 네 클래스 데이터가 모두 준비된 후 사용할 예시입니다.

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
- 라벨 번호는 swipe_left=0, make_fist=1, no_gesture=2, finger_snap=3으로 고정합니다.
- manifest는 발견한 파일당 한 행입니다. source_path는 입력 폴더 기준 상대경로이고, sha256은 파일 해시입니다. arrays_sha256과 identity_content_sha256은 중복 판정용입니다. 원본/매핑 ID, 라벨, sample_id, schema, 품질과 scale, status/reason, split과 해당 배열의 0부터 시작하는 output_index를 기록합니다. 입력 오류로 얻지 못한 값은 빈 칸입니다.
- status는 accepted(전처리 가능), excluded(검증/품질 제외), duplicate(중복), conflict(충돌)입니다. report에는 참가자·라벨별 input/accepted/excluded/duplicate/conflict 개수, 충돌 원인, 가정·경고와 split별 클래스 분포를 기록합니다. 각 상태 개수는 서로 배타적이며 합이 input입니다.
- preprocessing_config에는 보간·특징 수식·길이·품질 기준·참가자 배정을 저장합니다.

audit-only는 네 가지 메타데이터 파일만 생성합니다. 이때 accepted는 **전처리 가능**이라는 뜻이지 학습에 사용했다는 뜻은 아닙니다. split과 output_index는 비워 둡니다. 클래스 부족이나 참가자 분리 오류가 있는 normal 실행도 보고서만 남기고 배열 생성을 중단합니다. 정상 배열이 생성될 때만 split/output_index를 부여합니다.

보고서의 success와 training_arrays_written을 함께 확인하세요. audit-only는 split 요건이 부족해도 유효한 자료가 있고 치명적 충돌이 없으면 success=true이며, split_errors와 warnings에 부족한 조건을 남깁니다. 파일별 제외가 있어도 나머지 파일을 계속 검사합니다. 유효 자료가 전혀 없거나 충돌이 있으면 실패입니다.

종료 코드는 성공 0, 점검/분리 실패(보고서 저장) 1, 잘못된 CLI·경로·매핑 또는 I/O 오류 2입니다. CLI·경로 오류는 스캔 전에 거부하므로 보고서가 없을 수 있습니다. 결과는 임시 형제 폴더에서 완성한 뒤 한 번에 공개하며, 저장 I/O 오류가 발생하면 일부 배열을 최종 output-dir에 남기지 않습니다. 보고서조차 저장할 수 없는 I/O 오류는 콘솔에 표시합니다.

## 결과 확인 기준

담당자는 보고서에서 네 클래스가 각 split에 존재하는지, 참가자가 겹치지 않는지, 제외 사유가 예상 범위인지 확인합니다. 코드에서는 X/y의 샘플 수, manifest와의 대응, `(N,32,66)` 형상과 NaN/Inf 부재를 검사합니다.

이 출력은 LSTM과 GRU에 공통으로 사용할 전처리 데이터입니다. 학습·모델 성능 평가·연속 영상의 동작 구간 검출은 별도 단계입니다. 원본 NPZ와 수집용 index.csv는 유지합니다.

## 테스트와 함수 재사용

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_*.py" -v
```

테스트는 임시 폴더에서만 NPZ를 만듭니다. v1/v2, 불규칙 시간축, 결측/양끝 보간, 비정상 timestamp·dtype·스키마, 크기 0, 정지 동작 보존, 참가자 누수, 클래스 누락, 별칭, 중복·충돌, X/y와 manifest 행 대응, 원본 보존과 저장 오류를 검사합니다. 다중 참가자·4클래스 normal 모드 검증은 파일 입출력과 분리 로직에 대한 합성 테스트이며 실제 모델 성능을 의미하지 않습니다.

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
| num_classes | 4 | 클래스 수 / 양의 int |
| dropout | 0.2 | 드롭아웃 비율 / 유한한 실수, 0 이상 1 미만 |

bool은 크기/비율 설정으로 허용하지 않습니다. 위 값은 초기 실험 설정이며 최적 설정이 아닙니다. 현재 데이터 계약에서는 input_size=66, num_classes=4를 사용하고 라벨 번호는 swipe_left=0, make_fist=1, no_gesture=2, finger_snap=3입니다. 다른 값으로 모델을 생성할 수는 있지만 실제 전처리 설정과 라벨 매핑에 맞춰야 합니다.

### 입력, 출력 및 상태

- 입력 X: float32 `(batch_size, seq_len, 66)`. 기본 전처리 길이는 32입니다.
- 출력: `(batch_size, 4)` logits. 배치 크기가 1이어도 배치 차원을 유지합니다.
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
    num_classes=4, dropout=0.2,
)
model = LSTMClassifier(**model_config)  # 기본 float32/CPU
x = torch.randn(4, 32, 66, dtype=torch.float32)
y = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

model.train()
logits = model(x)  # (4, 4), softmax 적용 전 점수
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

`train.py`는 전처리 결과·label_map/config 로딩, DataLoader, device 선택, loss/optimizer, 학습·검증 루프 및 체크포인트 저장을 담당합니다. 실행법은 아래 학습 절을 참고하세요. 모델의 파라미터 초기화는 PyTorch 기본 동작을 사용하며, 재현성을 위한 seed 설정은 호출 측/테스트에서 수행합니다.

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

## LSTM 학습 (`train.py`)

실행 순서는 **전처리 완료 → train/validation 학습 및 모델 선택 → 추후 별도 test 평가**입니다. `train.py`는 기존 `LSTMClassifier`를 사용하며, GRU·실시간 추론·학습 재개 기능은 포함하지 않습니다. 원본 NPZ나 수집 시간 제한도 변경하지 않습니다.

이번 구현에서 실행한 것은 임시 합성 데이터의 CPU 테스트뿐입니다. 실제 수집 자료로 학습하거나 정확도를 평가하지 않았습니다. 위의 과거 점검 개수는 당시의 기록이며 현재 자료의 준비 상태를 보장하지 않습니다. 실제 학습 전 새 전처리 보고서를 확인하세요.

### 학습 시작

프로젝트 루트 PowerShell에서 다음과 같이 실행합니다. `prepared_001`은 예시 경로이며, 실제로 생성한 **정상 모드 전처리 결과 폴더**로 바꾸세요.

```powershell
.\.venv\Scripts\python.exe .\training\train.py --data-dir .\artifacts\prepared_001 --output-dir .\artifacts\lstm_run_001
```

CPU를 명시하려면 `--device cpu`를 추가합니다. 옵션만 확인하려면:

```powershell
.\.venv\Scripts\python.exe .\training\train.py --help
```

상대경로는 현재 작업 디렉터리 기준입니다. output-dir는 기존에 없는 새 폴더여야 하며, 입력 전처리 폴더나 프로젝트 dataset 내부는 거부합니다. 기본값은 초기 실험용이며 최적 설정이 아닙니다.

| 옵션 | 기본값 / 허용 범위 |
| --- | --- |
| --data-dir / --output-dir | 필수 |
| --hidden-size | 64 / 양의 정수 |
| --num-layers | 1 / 양의 정수 |
| --dropout | 0.2 / 0 이상 1 미만 |
| --batch-size | 32 / 양의 정수 |
| --epochs | 100 / 양의 정수, 최대 실행 횟수 |
| --learning-rate | 0.001 / 양수 |
| --weight-decay | 0.0001 / 0 이상 |
| --patience | 10 / 양의 정수 |
| --min-delta | 0.0001 / 0 이상 |
| --max-grad-norm | 1.0 / 양수 |
| --seed | 42 / 0~2**32-1 정수 |
| --device | auto / auto, cpu, cuda |
| --num-workers | 0 / 0 이상 정수, Windows 기본은 0 권장 |

모든 실수 옵션은 NaN/Inf가 아닌 유한한 값이어야 합니다. auto는 CUDA 사용 가능 시 CUDA를 선택하고 아니면 CPU를 사용합니다. 명시적으로 cuda를 요청했는데 사용할 수 없으면 오류입니다. 입력 차원 66과 클래스 수 3은 전처리 계약 검증 후 정합니다.

### 데이터 검사와 학습 동작

입력 폴더에는 X/y의 train/val/test 6개 NPY와 label_map.json, preprocessing_config.json, manifest.csv, report.json이 있어야 합니다. report가 normal/success이고 training_arrays_written=true여야 하며 audit-only/실패/충돌 결과는 거부합니다.

train/val은 allow_pickle=False로 열어 float32 `(N,seq_len,66)` / int64 `(N,)`, 유한한 값, 비어 있지 않음, label_map의 모든 클래스 존재를 검사합니다. manifest의 accepted 행 개수·라벨·연속된 output_index가 배열과 일치해야 합니다. config의 참가자 배정, manifest와 보고서의 클래스 분포/상태 개수도 확인합니다. canonical 참가자가 train/val/test 사이에 겹치면 학습하지 않습니다.

**test 배열은 로딩하지 않습니다.** 파일 존재와 manifest/config의 참가자 배정·클래스·인덱스만 확인하며 test 배열 자체의 길이·내용 검증도 별도 평가 단계로 남깁니다. validation은 매 epoch 모델 선택에 사용하지만 test는 최종 평가용입니다. test 결과로 학습 설정이나 최적 epoch를 반복 선택하면 독립적인 평가가 아니게 됩니다.

TensorDataset은 X/y를 한 쌍으로 묶고 DataLoader는 배치를 만듭니다. train만 shuffle하며 drop_last=False로 마지막 작은 배치도 사용합니다. train 모드에서는 logits → CrossEntropyLoss → backward → gradient clipping → AdamW 순으로 가중치를 갱신합니다. validation은 eval/no_grad로 계산하며 파라미터를 갱신하지 않습니다. 클래스 가중치·증강·scheduler·mixed precision은 없습니다.

loss는 각 배치 평균에 해당 배치 샘플 수를 곱해 누적한 뒤 전체 샘플 수로 나눕니다. accuracy는 전체 정답 수/샘플 수인 0~1 비율입니다. train 지표는 배치별 업데이트 과정에서 계산한 값이며, validation은 해당 epoch 학습이 끝난 모델로 계산한 값입니다. loss/gradient/업데이트 후 파라미터가 NaN/Inf이면 중단합니다.

### 최적 모델과 early stopping

두 기준은 별개입니다.

- **best_model.pt:** 이전 최저 val_loss보다 엄격히 작으면 저장합니다. min_delta보다 작은 개선도 저장하며 동률은 덮어쓰지 않습니다.
- **early stopping:** 첫 val_loss를 기준점(reference)으로 둡니다. 이후 `value < reference`이면서 `reference - value >= min_delta`이면 충분한 개선으로 인정해 reference를 갱신하고 대기 횟수를 0으로 만듭니다. 아니면 대기 횟수를 1 늘리고 patience에 도달하면 종료합니다.

작은 개선에서는 reference를 옮기지 않으므로 여러 epoch의 미세 개선이 누적되어 충분한 개선이 될 수 있습니다. min_delta 경계는 이상(>=)이며, min_delta=0이어도 동률은 개선이 아닙니다. 비교는 저장된 부동소수점 값으로 수행합니다. 예를 들어 min_delta=0.125, patience=2에서 `1 → 0.9375 → 0.875 → 0.875 → 0.8671875`는 3번째 값에서 대기를 초기화하고 5번째 값에서 종료합니다. 이때 최저값인 5번째 모델도 저장됩니다.

최적 가중치는 독립적인 CPU state_dict 복사본으로 즉시 저장합니다. 따라서 마지막 epoch가 최적이 아니면 이전 최적 모델을 유지합니다.

### 출력과 상태 확인

| 파일 | 내용 |
| --- | --- |
| best_model.pt | model_state_dict, model_config, best_epoch, 해당 epoch metrics, label_map, preprocessing_config |
| history.csv | epoch(1부터), train_loss/accuracy, val_loss/accuracy |
| training_config.json | 실제 옵션·모델 설정·device·seed·버전·전처리 설정·split별 샘플 개수·입력 메타데이터 SHA-256 |
| training_summary.json | status, epochs_completed, best_epoch/best_metrics, early_stopped, partial_run, error, test_evaluated |

config의 split_counts는 참가자 수가 아니라 **split별 accepted 샘플 수**입니다. 최적 epoch의 지표는 summary의 best_metrics를 확인하세요. 마지막 history 행이 항상 최적은 아닙니다. test_evaluated는 항상 false입니다.

상태는 실행 중 running, 정상 완료 success, 오류 failed, Ctrl+C 중단 interrupted입니다. 정상적인 early stopping은 success이며 early_stopped=true입니다. failed/interrupted는 partial_run=true이며 이전 체크포인트가 남아 있어도 완성된 실행으로 간주하지 않습니다. 완료 epoch 수는 학습과 validation을 끝내고 history까지 저장한 횟수입니다.

설정/경로/장치 오류는 종료 코드 2로 학습 전에 거부하며 출력 폴더가 없을 수 있습니다. 출력 폴더 생성 후 데이터 검증/학습 실패는 1, Ctrl+C는 130, 정상 완료는 0입니다. 데이터 검증 실패 시 summary만 있을 수 있고 첫 validation 전에 실패하면 best_model.pt가 없습니다.

JSON·CSV·체크포인트는 각각 임시 파일을 완성한 후 교체합니다. 여러 파일 전체를 하나의 트랜잭션으로 저장하는 것은 아니므로 I/O 오류 직후에는 파일 간 진행 epoch가 다를 수 있습니다. summary의 실패/부분 상태를 먼저 확인하세요. 저장 장치 오류로 summary도 기록할 수 없으면 콘솔에 알립니다. 강제 종료/전원 차단 시 running 상태가 남을 수 있으며 이것도 완료된 실행이 아닙니다.

학습 체크포인트는 모델 객체 전체가 아닌 텐서와 기본 Python 자료형으로 구성됩니다. 복원 시 `torch.load(path, map_location="cpu", weights_only=True)`로 읽고, model_config로 모델 생성 → model_state_dict 로딩 → eval() 순서로 사용합니다. 앞선 모델 단독 예시의 state_dict 키와 달리 **train.py의 키는 model_state_dict**입니다. optimizer 상태는 저장하지 않으며 학습 재개는 지원하지 않습니다.

### 재현성과 테스트

실행 시 Python random·NumPy·PyTorch seed와 DataLoader generator를 설정하고 worker에도 seed를 전달합니다. CUDA 사용 가능 환경에서는 CUDA seed 및 cuDNN의 deterministic=true, benchmark=false를 설정합니다. 하드웨어·device·라이브러리 버전·CUDA 연산 차이까지 완전한 동일 결과를 보장하지는 않습니다. 비교 실험은 seed뿐 아니라 저장된 설정과 환경 버전도 맞춰야 합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_train.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_*.py" -v
```

학습 테스트는 임시 합성 NPZ를 기존 prepare_dataset.py로 전처리한 결과만 사용합니다. 입력/manifest/참가자 검증, CPU 1~2 epoch 실행, 샘플 가중 평균, 파라미터 갱신과 validation 비갱신, best/early stopping, 안전한 저장·복원, 동일 seed 재현성, test 미로딩, 비정상 수치 및 중단 상태를 검사합니다. 실제 데이터 성능은 향후 실제 학습과 별도 test 평가를 통해 확인해야 합니다.

## 저장된 LSTM 평가 (`evaluate.py`)

`train.py`는 가중치를 학습하고 validation loss로 best epoch를 선택합니다. `evaluate.py`는 이미 저장된 `best_model.pt`를 한 번 불러와, 명시적으로 선택한 val 또는 test의 **잘라진 시퀀스 분류 성능**만 측정합니다. 학습, 재전처리, 데이터 재분리, 임계값 튜닝, 최적 epoch 재선택, 실시간 추론은 수행하지 않습니다.

이번 평가 도구 구현에서 실행한 검증은 임시 합성 데이터의 CPU 테스트뿐입니다. 실제 validation 및 p006 test 평가는 실행하지 않았습니다. 앞선 절의 구현 당시 기록과 별개로, 실제 실행 결과는 사용자가 생성하는 평가 출력에서 확인하세요.

### 직접 실행하는 방법

프로젝트 루트 PowerShell에서 validation을 확인하는 한 줄 명령입니다.

```powershell
.\.venv\Scripts\python.exe .\training\evaluate.py --run-dir .\artifacts\pilot_20260906\lstm_gpu_002 --data-dir .\artifacts\pilot_20260906\prepared --split val --output-dir .\artifacts\pilot_20260906\eval_val_001 --device cuda
```

2층 모델을 확인하려면 `--run-dir`를 `artifacts/pilot_20260906/lstm_gpu_2layers_001`로 바꾸고 output-dir도 새로운 이름으로 지정하세요. 모델은 체크포인트의 구조 설정으로 복원하므로 hidden-size나 num-layers 옵션은 없습니다.

다음 test 명령은 **모델/학습 설정 선택을 완료한 뒤에만** 실행하세요. 아래 run-dir는 예시이며 최종 선택한 학습 결과로 바꿉니다. 새 참가자 자료를 포함해 다시 전처리/학습했다면 run-dir와 data-dir 모두 해당 실행에 맞춰야 합니다.

```powershell
.\.venv\Scripts\python.exe .\training\evaluate.py --run-dir .\artifacts\pilot_20260906\lstm_gpu_002 --data-dir .\artifacts\pilot_20260906\prepared --split test --output-dir .\artifacts\pilot_20260906\eval_test_final_001 --device cuda
```

필수 옵션은 `--run-dir`, `--data-dir`, `--split`, `--output-dir`입니다. **split 기본값은 없으며** val/test 중 하나를 반드시 입력해야 합니다. 선택 옵션은 batch-size=32(양수), num-workers=0(0 이상), device=auto입니다. auto는 CUDA가 가능하면 GPU, 아니면 CPU를 선택합니다. 명시한 cuda를 사용할 수 없으면 오류이며 CPU 실행은 `--device cpu`로 지정합니다.

출력은 기존에 없는 새 폴더여야 합니다. 비어 있는 기존 폴더도 덮어쓰지 않으며 전처리 data-dir와 원본 dataset 내부 출력은 거부합니다. 입력 데이터와 학습 결과는 읽기만 합니다.

### 무엇을 검사하고 평가하나요?

- training_summary가 success이며 partial_run=false인 실행만 허용합니다. checkpoint의 model_config, best_epoch, metrics, label_map, preprocessing_config를 학습 설정/요약 및 전처리 설정과 교차 검사합니다. 안전한 `weights_only=True` 로딩과 `strict=True` 가중치 복원을 사용합니다.
- report가 성공한 normal 결과인지, manifest의 참가자 배정이 설정과 맞고 train/val/test 사이에 canonical ID 누수가 없는지 검사합니다. output_index는 0부터 연속이어야 하며 선택한 배열의 행/정답 라벨과 대응해야 합니다.
- **선택한 split의 X/y NPY만** `allow_pickle=False`로 읽고 해시를 계산합니다. 나머지 split의 메타데이터는 검사하지만 NPY 내용은 읽지 않습니다. X는 float32 `(N,seq_len,66)`, y는 int64 `(N,)`이며 유한값, 네 클래스 존재, 설정된 길이 등을 확인합니다. 정지 no_gesture나 작은 이동량은 배제하지 않습니다.
- eval()로 Dropout을 끄고 inference_mode()로 미분 그래프 생성을 막습니다. optimizer/backward 없이 가중치를 고정하고, shuffle=False/drop_last=False로 마지막 작은 배치까지 평가합니다. loss는 배치 평균의 단순 평균이 아닌 전체 샘플 수 기준 평균입니다.
- logits를 CrossEntropyLoss에 직접 넣고 argmax로 클래스를 선택합니다. softmax는 CSV에 확률을 기록할 때만 사용합니다. NaN/Inf 출력은 실패입니다.

학습 설정의 manifest/config/label_map/report SHA-256을 현재 파일 내용과 비교하므로, 경로가 이동했더라도 내용이 같으면 허용합니다. 반대로 메타데이터를 다시 생성해 바이트가 달라졌다면 검사를 통과하지 못할 수 있습니다. 이 검사를 피하려고 해시나 설정을 임의로 수정하지 마세요.

**검증 한계:** 현재 train.py는 NPY의 학습 당시 해시를 기록하지 않습니다. 메타데이터 해시 일치만으로 NPY가 학습 당시와 완전히 같다고 보증하지 않습니다. 평가 결과에 저장되는 선택 X/y SHA-256은 평가 시점 파일의 지문입니다. 입력 파일은 평가가 진행되는 동안 변경하지 마세요.

### 결과 파일과 해석

| 파일 | 확인할 내용 |
| --- | --- |
| evaluation_metrics.json | status/completed, split, sample_count, overall, per_class, 라벨 순서, 분모 0 정책, 오류, val/test 해석 안내 |
| confusion_matrix.csv | 행=실제 클래스, 열=예측 클래스. swipe_left → make_fist → no_gesture 순서의 정수 개수 |
| classification_report.txt | 클래스별 precision/recall/F1/support, accuracy, macro avg, weighted avg, loss를 읽기 쉬운 텍스트 표로 저장 |
| predictions.csv | 모든 샘플의 output_index, 경로, 참가자 ID, 실제/예측 라벨, 클래스별 확률, correct |
| misclassified.csv | predictions에서 correct=False인 행만 저장. 모두 정답이면 헤더만 있는 정상 파일 |
| evaluation_config.json | 실행 경로/옵션/device, 모델/전처리 설정, 체크포인트·선택 배열·메타데이터 해시, Python/NumPy/PyTorch 버전 |

accuracy는 전체 정답 비율입니다. precision은 해당 클래스로 예측한 것 중 정답 비율, recall은 해당 실제 클래스 중 맞힌 비율이며 F1은 둘의 조화 평균입니다. macro F1은 **label_map에 정의된 모든 클래스의 F1을 같은 비중으로 평균**하므로 샘플이 많은 클래스가 전체 accuracy를 높이는 경우에도 클래스별 약점을 확인하는 데 도움이 됩니다. weighted F1은 실제 클래스 샘플 수(support)를 가중치로 평균합니다. 분모가 0인 지표는 0으로 두며 macro에서 클래스를 빼지 않습니다. 점수와 확률은 모두 0~1 비율이고 loss는 0~1로 제한되지 않습니다.

혼동 행렬의 대각선은 정답 개수입니다. 예를 들어 swipe_left 행, make_fist 열은 실제 swipe_left를 make_fist로 잘못 판단한 개수입니다. 모든 칸의 합은 평가 샘플 수입니다.

틀린 샘플은 misclassified.csv의 source_path로 찾습니다. **source_path는 전처리의 input 폴더 기준 상대경로**이지 저장소 dataset 기준이라고 보장하지 않습니다. 이번 예비 학습은 `artifacts/pilot_20260906/input`의 정리된 복사본에서 전처리했으므로 최초 수집 파일까지 추적하려면 별도 `artifacts/pilot_20260906/input_provenance.json`을 참조하세요. participant_id도 전처리 입력 복사본에 저장된 ID이며, 최초 원본 ID라고 단정할 수 없습니다. canonical_participant_id는 전처리에서 참가자 분리에 사용한 대표 ID입니다.

`classification_report.txt`는 기존 평가 명령으로 자동 생성됩니다. 추가 패키지 없이 같은 평가 지표를 표로 표현하며, 점수는 소수점 4자리(loss는 6자리)로 표시합니다. weighted avg의 precision/recall/F1은 각각 support 가중 평균입니다. 기존 평가 폴더에 자동 추가되지는 않으며 다음 평가에서는 새 output-dir를 지정하세요.

### 완료 상태와 평가의 한계

반드시 evaluation_metrics.json의 **status=success, completed=true**를 먼저 확인하세요. 실행 중은 running, 오류는 failed, Ctrl+C는 interrupted이며 실패/중단의 overall/per_class/sample_count는 null입니다. JSON/CSV는 각각 임시 파일 후 교체하고 마지막에 완료 상태를 기록합니다. 파일 묶음 전체가 하나의 트랜잭션은 아니므로 저장 중 실패하면 일부 CSV가 남을 수 있습니다. 그런 CSV를 완성된 평가 결과로 사용하지 마세요. 상태 파일조차 저장할 수 없는 장치 오류는 콘솔에 알리며, 강제 종료 시 남는 running도 완료가 아닙니다.

정상 종료 코드는 0, 출력 생성 후 평가/검증 오류는 1, 사전 옵션/경로/장치 오류는 2, 중단은 130입니다. 사전 오류에서는 출력 폴더가 생기지 않을 수 있습니다.

validation은 이미 모델/epoch 선택에 사용한 자료입니다. validation 100%라도 최종 test나 새로운 사용자, 실시간 서비스 성능이 보장되지는 않습니다. test는 최종 선택 후 별도로 평가하며 **test 결과를 보며 반복 튜닝하거나 모델을 고르지 마세요.**

현재 지표는 잘라진 시퀀스 분류에만 해당합니다. 실시간 동작 구간 검출, 무동작 중 오작동률, 명령 중복 실행, 인식 지연은 측정하지 않습니다. 서비스 수준 검증은 별도의 연속 입력 테스트가 필요합니다.

### 합성 테스트 실행

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_evaluate.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test_*.py" -v
```

테스트는 TemporaryDirectory의 합성 NPZ를 전처리하고 작은 비학습 LSTM 체크포인트를 만들어 검증합니다. CPU 실행, 선택 split만 접근, 저장 구조 복원, 가중치 불변, 지표 수동 계산, 마지막 배치 손실, CSV 대응, 오류 입력과 중단/저장 실패 상태를 검사합니다. 합성 테스트 통과는 실제 제스처 분류 성능을 의미하지 않습니다.
