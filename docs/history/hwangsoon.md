# 시연 프로그램 사용 안내 (`hwangsoon` 브랜치, 2026-09-07)

손 제스처로 PC 를 제어하는 시연 프로그램입니다. 웹캠 → MediaPipe 손 관절 → 동작 구간 감지(규칙) → GRU 판정 → 상식 검사(규칙) → 키 입력. 정적 포즈(YOLO)로 인식을 켜고 끕니다(손바닥 = 시작, 주먹 = 끝, 세 번째 포즈 = 단축키 하나).

## 다른 팀원이 실행하려면

**1. 준비물**: Windows 10/11, Python 3.12, [uv](https://docs.astral.sh/uv/), 웹캠. NVIDIA GPU 는 있으면 YOLO 가 빨라지고 없어도 CPU 로 동작합니다.

**2. 코드 받기와 설치** (PowerShell, 처음 한 번. 약 3GB 내려받음)

```powershell
git clone -b hwangsoon https://github.com/Spice7/jesture_project.git
cd jesture_project
py -3.12 -m pip install uv      # uv 가 이미 있으면 생략
uv sync --locked
```

**3. git 에 없는 파일 3개를 팀 구글드라이브에서 받아 `models/` 폴더에 넣기** (용량 때문에 git 에서 제외)

| 파일 | 역할 | 드라이브 위치 |
|---|---|---|
| `models/gru_gesture.pt` | 동적 제스처 모델 (GRU, 4클래스) | 모델 폴더 |
| `models/gru_gesture.json` | 위 모델의 라벨·설정 | 모델 폴더 |
| `models/yolo_gate.pt` | 정적 포즈 모델 (YOLOv8n, start/stop/cancel) | `dataset_yolo/lkh/v2/best.pt` 를 이 이름으로 복사 |

`models/hand_landmarker.task` 는 git 에 들어 있어 따로 받을 필요가 없습니다. 데이터셋(`dataset/*.npz`)은 실행에 필요 없고 학습·재생 시험에만 씁니다.

**4. 실행**

```powershell
uv run python scripts/gesture_app.py          # 연습 모드(키 안 누름)로 시작. 사이드바 '도움말' 부터 보세요
uv run python scripts/gesture_app.py --live   # 처음부터 실제 키 입력
```

첫 화면에서 ① 사이드바 아래 "실제 키 입력" 스위치 → ② 손바닥을 카메라에 2초 → ③ 제스처. 키 매핑은 "키 매핑" 페이지에서 바꾸고 바로 저장됩니다(`gestures.json`).

**5. 더 읽을 것**

- 사용 설명서: [docs/user_guide.md](docs/user_guide.md) (제스처 하는 법, 문제 해결, FAQ)
- 프로그램 구조와 YOLO 게이트: [docs/gesture_app.md](docs/gesture_app.md)
- 모델 성능: [docs/model_metrics_0907.md](docs/model_metrics_0907.md), GRU vs LSTM 비교: [reports/gru_vs_lstm_0907/summary.md](reports/gru_vs_lstm_0907/summary.md)
- 학습·평가 절차: [docs/team_conventions.md](docs/team_conventions.md)

카메라는 한 프로그램만 쓸 수 있습니다. 수집기나 다른 데모가 켜져 있으면 화면이 깨집니다.

---

# 제스처 데이터 수집 및 제출 안내 (JIN 브랜치 원문)

팀원은 `JIN` 브랜치의 수집 프로그램을 받아 자신의 브랜치에서 데이터를 수집하고, 자기 NPZ 파일만 제출합니다. 촬영 자세와 라벨별 동작은 [제스처 수집 안내](programs/README.md)를 따릅니다.

## 시작 전 확인

- 담당자가 최신 수집 코드와 README를 `JIN`에 push한 뒤 시작합니다.
- 각 팀원은 실제 제스처 수행자 기준으로 `p001`~`p006` 중 고유 ID 하나를 배정받습니다.
- Git과 Python 3.12를 준비합니다. 현재 수집 환경은 Windows와 웹캠을 기준으로 합니다.
- 개인 브랜치를 push하려면 GitHub 저장소 쓰기 권한이 필요합니다. 권한이 없으면 저장소 담당자에게 요청합니다.

아래 명령은 **`p002`를 예시로 작성했습니다. 브랜치명과 파일명에 있는 `p002`를 모두 자신의 ID로 바꿔 사용하세요.**

## 1. JIN 브랜치 받기

저장소를 처음 받는 경우 PowerShell에서 실행합니다.

```powershell
git clone -b JIN https://github.com/Spice7/jesture_project.git
cd jesture_project
```

이미 저장소가 있다면 `git status`로 작업 상태를 확인합니다. 미완료 변경이 없는 경우 다음 명령으로 `JIN`을 업데이트합니다. 변경이 남아 있다면 먼저 해당 작업을 정리하고 진행하세요.

```powershell
git switch JIN
git pull --ff-only origin JIN
```

이후 명령은 프로젝트 루트인 `jesture_project`에서 실행합니다.

## 2. 개인 수집 브랜치 만들기

```powershell
git switch -c data/p002
git branch --show-current
```

출력이 `data/p002`인지 확인합니다. 이미 같은 브랜치를 만들어 수집 중이라면 새로 만들지 말고 기존 브랜치에서 계속합니다.

## 3. 환경 설치와 수집

프로젝트에는 의존성 설정과 `uv.lock`이 포함되어 있습니다. Python 3.12가 설치된 상태에서 다음 명령을 실행합니다.

```powershell
py -3.12 -m pip install uv
py -3.12 -m uv sync --locked
.\.venv\Scripts\python.exe .\programs\collect_gesture.py
```

참가자 ID에는 자신의 ID를 입력하고, 라벨은 아래 세 가지 중 이번에 수집할 항목을 입력합니다.

| 라벨 | 구분 | 인당 초기 목표 |
| --- | --- | ---: |
| `swipe_left` | 오른손을 수행자 기준 왼쪽으로 이동하는 명령 | 50개 |
| `make_fist` | 편 오른손을 주먹으로 쥐는 명령 | 50개 |
| `no_gesture` | 명령하지 않는 상태의 보조 데이터 | 20개 |
| 합계 | | 120개 |

시험 수집 데이터도 목표 개수에 포함합니다. 먼저 각 명령 10개씩을 수집해 담당자에게 확인받고 나머지를 채웁니다. 개수는 녹화 시도 횟수가 아닌 정상 저장된 NPZ 수를 기준으로 합니다.

구체적인 촬영 순서, 키 조작, 이동량 경고의 의미는 [programs/README.md](programs/README.md)를 확인하세요.

## 4. 자기 NPZ만 커밋하기

수집 완료 후 변경 파일을 확인합니다.

```powershell
git status --short
```

아래 명령으로 자신의 ID에 해당하는 NPZ만 스테이징합니다. `git add -A`는 사용하지 않습니다.

```powershell
git add -- "dataset/swipe_left/p002_*.npz"
git add -- "dataset/make_fist/p002_*.npz"
git add -- "dataset/no_gesture/p002_*.npz"
git --no-pager diff --cached --stat
```

스테이징 목록에 자기 NPZ만 있는지 확인합니다. 수집 중 수정된 `dataset/index.csv`, 개인적으로 수정한 코드, 다른 사람의 파일은 제출 커밋에 포함하지 않습니다. 이전에 다른 파일을 스테이징했다면 이 명령이 자동으로 제외해 주지는 않으므로 목록을 반드시 확인하세요.

Git 작성자 정보가 설정되어 있지 않으면 현재 저장소에서 한 번 설정합니다. 예시 문자열 대신 본인의 이름과 GitHub에 등록된 이메일 또는 계정의 noreply 이메일을 입력합니다.

```powershell
git config user.name "본인 이름 또는 GitHub 사용자명"
git config user.email "본인의 커밋 이메일"
```

```powershell
git commit -m "p002 제스처 데이터 추가"
git push -u origin data/p002
```

`dataset/index.csv`가 커밋되지 않은 변경으로 남는 것은 예상된 상태입니다. 자기 NPZ 커밋과 push는 그대로 진행할 수 있습니다.

## 5. JIN으로 Pull Request 만들기

GitHub 저장소에서 Pull Request를 생성할 때 다음 브랜치를 선택합니다.

- **base:** `JIN`
- **compare:** `data/p002` 등 자신의 수집 브랜치

PR 제목은 `p002 제스처 데이터 추가`처럼 작성하고, 본문에 참가자 ID, 라벨별 실제 저장 개수, 촬영 중 특이사항을 적습니다. **대상 브랜치를 `main`으로 선택하거나 다른 사람의 브랜치에 push하지 않습니다.**

저장소 담당자가 파일과 라벨을 확인한 후 `JIN`에 병합합니다. 추가 수집이나 수정이 필요하면 같은 개인 브랜치에서 자기 NPZ만 커밋하고 다시 push하면 PR에 반영됩니다.

## 담당자의 병합 순서

1. PR에 해당 참가자의 NPZ만 포함됐는지 확인합니다.
2. 파일 내부 라벨·참가자 ID·품질과 파일명 중복을 검사합니다.
3. 확인한 PR을 `JIN`에 병합합니다.
4. 전체 NPZ가 모이면 파일들을 기준으로 `dataset/index.csv`를 재생성합니다. 각 팀원의 CSV를 덮어쓰는 방식으로 합치지 않습니다.
5. 수집 결과를 검토한 뒤 프로젝트 통합 시점에 `JIN`에서 `main`으로 PR을 만듭니다.

현재 인덱스를 재생성하는 전용 프로그램은 포함되어 있지 않으며, 병합 담당자가 전체 데이터 취합 단계에서 마련합니다.
