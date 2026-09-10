> 현재 integration의 모델 배치와 설치는 루트 README.md를 우선합니다. 아래 성능/실험 설명은 기존 기록을 포함합니다.

# Windows 세팅 (시연 환경)

기기가 두 대다.

| 기기 | 용도 | 필요한 것 |
|---|---|---|
| **집 노트북** (웹캠) | 데이터 녹화, 시연 리허설 | 카메라 권한, 수집기 |
| **원격 데스크톱** (GPU, 크롬 원격 데스크톱으로 접속) | 학습·비교, 코드 작업 | 저장소 clone, uv sync |

두 기기 모두 이 저장소를 받고 `uv sync` 하면 된다. 코드는 git 으로 오간다.
**원격 데스크톱에서는 카메라를 쓸 수 없다.** 원격 접속 중인 화면의 카메라는 그 기기 앞에 사람이 없다. 녹화는 노트북에서만.

## GPU 에 대한 솔직한 안내

- TensorFlow 2.21 은 **Windows 에서 GPU 를 지원하지 않는다** (2.10 이후 중단). GRU 는 CPU 로 돌아간다.
- 이 규모(샘플 수백~천, 30×63 입력, 2층 64유닛)는 **CPU 로 한 번 학습에 1~3분**이다. GPU 가 없어도 아무 문제 없다.
- GPU 는 YOLO(정적 포즈) 팀의 학습에 의미가 있다. PyTorch 는 Windows GPU 를 지원한다.
- TF GPU 를 억지로 세팅(WSL2 등)하는 데 시간을 쓰지 말 것. 하루짜리 프로젝트다.

## 1. 도구 설치 (기기마다 한 번)

PowerShell 에서:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
```

설치 후 PowerShell 을 닫고 다시 연다. `py -3.12 --version` 이 찍히면 된다.

## 2. 저장소 받기 + 의존성

```powershell
cd $HOME\Desktop
git clone -b hwangsoon https://github.com/Spice7/jesture_project.git
cd jesture_project
py -3.12 -m pip install uv
py -3.12 -m uv sync --locked
```

`uv sync` 는 처음에 약 3GB 를 받는다 (5~10분). 두 번째부터는 몇 초. **pip 로 따로 설치하지 않는다.**

git 사용자 설정이 처음이면:

```powershell
git config --global user.name "HwangSoon96"
git config --global user.email "te44382@gmail.com"
```

이후 명령은 `uv run python ...` 또는 `.\.venv\Scripts\python.exe ...` 둘 다 같다. JIN README 는 후자를 쓴다.

## 3. 노트북: 녹화

녹화는 팀 규약대로 **별도 수집 브랜치**에서 한다. 이 브랜치(hwangsoon)에 데이터를 직접 커밋하지 않는다.

```powershell
git switch -c data/p00X origin/JIN         # p00X = 담당자에게 받은 내 ID
.\.venv\Scripts\python.exe .\programs\collect_gesture.py
```

카메라 권한: Windows 설정 → 개인 정보 및 보안 → 카메라 → "앱 허용" 과 **"데스크톱 앱 허용"** 둘 다 켠다.

촬영 규칙·키 조작은 `programs/README.md`. 요점만:
- 오른손만 화면에. `Space` 시작/저장, `R` 취소, `Q` 종료.
- swipe_left 는 **내 왼쪽으로**. 화면에는 오른쪽으로 가는 것처럼 보이는데 그게 맞다.
- make_fist 는 손목 고정, 편 손 → 주먹. "이동량 경고" 는 정상.
- 먼저 10개씩 → 담당자 확인 → swipe_left 50, make_fist 50, no_gesture 20.
- `Saved:` 가 떠야 저장된 것. `Discarded` 면 다시.

올리기 (자기 파일만, `git add -A` 금지):

```powershell
git add -- "dataset/swipe_left/p00X_*.npz" "dataset/make_fist/p00X_*.npz" "dataset/no_gesture/p00X_*.npz"
git commit -m "p00X 제스처 데이터 추가"
git push -u origin data/p00X
```

GitHub 에서 PR: base `JIN`, compare `data/p00X`.

녹화가 끝나면 `git switch hwangsoon` 으로 돌아온다.

## 4. 원격 데스크톱: 학습

```powershell
git switch hwangsoon
git fetch origin
git merge origin/JIN                       # 팀원 데이터 가져오기
uv run python scripts/dataset_summary.py   # 현황 확인
uv run python scripts/compare_models.py --seeds 3 --test-persons p00A --val-persons p00B
```

결과 표는 `reports/compare_models_summary.md`. 커밋해서 올린다.

## 5. 자주 나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `py : 용어가 인식되지 않습니다` | Python 설치 후 PowerShell 재시작 |
| `uv sync` 중 `Access is denied` | 백신이 .venv 를 잡는 경우. 프로젝트 폴더를 예외 등록 |
| TensorFlow import 시 DLL 오류 | [Visual C++ 재배포 패키지](https://aka.ms/vs/17/release/vc_redist.x64.exe) 설치 후 재부팅 |
| 카메라 열림 실패 | 권한 두 개 확인. Zoom/Teams 등 카메라 쓰는 앱 종료. 원격 데스크톱에서는 원래 안 됨 |
| 녹화 창이 안 보임 | 작업 표시줄에서 창 찾기. 뒤로 숨는 경우가 있다 |
| 한글이 `?` 로 보임 | 구형 cmd. Windows Terminal 사용. 동작에는 영향 없음 |
| `경고(형식): ... 키 없음` | 수집기 저장 형식이 바뀐 것. JIN 담당자에게 확인 후 `gesture/dataset.py` 로더 수정 |
| 경로에 한글/공백 | `C:\Users\<이름>\Desktop\jesture_project` 처럼 영문 경로 권장 |

## 6. 시연 전 체크 (노트북)

- `uv sync` 완료, `models/gru_gesture.keras` 존재 (원격 데스크톱에서 학습한 걸 git 으로 받거나 직접 복사)
- 발표장 조명에서 수집기로 손이 잡히는지 당일 아침 확인
- 카메라 쓰는 다른 앱 없음, 전원 연결, 절전 해제
