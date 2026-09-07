# Jesture GUI 실행 파일 만들기

Windows용 `Jesture.exe`를 만듭니다. 학습·데이터 수집 코드는 담지 않으며, GUI 서비스(`service.gui`)만 묶습니다. 팀원이 파이썬 없이 제스처 제어를 써 볼 수 있게 하는 것이 목적입니다.

## 빌드

프로젝트 루트에서 한 번 실행하면 됩니다.

```powershell
.\.venv\Scripts\python.exe .\packaging\build_exe.py
```

결과는 `dist/Jesture/Jesture.exe`입니다. **폴더 전체가 하나의 프로그램**이므로 exe만 따로 옮기면 실행되지 않습니다. 압축해서 전달하세요.

| 옵션 | 용도 |
| --- | --- |
| `--recreate` | 빌드 환경을 지우고 처음부터 다시 만듭니다. |
| `--skip-install` | 이미 만든 빌드 환경을 그대로 쓰고 설치를 건너뜁니다. 스펙만 고쳤을 때 빠릅니다. |
| `--keep-build` | PyInstaller 중간 산출물(`build/`)을 남깁니다. 누락 모듈을 추적할 때 씁니다. |

## 왜 빌드 환경을 따로 만드나요?

개발용 `.venv`에는 CUDA torch가 들어 있어 그대로 묶으면 결과물이 4.5GB를 넘습니다. 스크립트는 `.venv-build`에 **CPU torch만** 설치하고 그 환경에서 PyInstaller를 실행합니다. 개발/학습용 `.venv`는 건드리지 않습니다.

CPU로 내려도 실사용에는 무리가 없습니다. 측정값 기준으로 YOLO는 640에서 약 23ms, 416에서 약 15ms이고 MediaPipe는 원래 CPU, LSTM 추론은 1.2ms입니다. 1.5초 창에 최소 20프레임(약 13.3 FPS, 프레임당 75ms)이면 되므로 여유가 있습니다. GPU가 필요하면 저장소에서 `python -m service.gui`로 실행하세요.

`tensorflow`는 `pyproject.toml`에 남아 있지만 코드 어디에서도 쓰지 않습니다(1.5GB). matplotlib·pandas·ipykernel과 함께 빌드에서 제외합니다.

## 번들에 들어가는 자산

`service/resources.py`가 저장소 실행과 번들 실행에서 같은 상대 경로를 돌려줍니다. 스펙은 아래 세 파일을 그 경로 그대로 넣습니다.

| 파일 | 번들 위치 | 용도 |
| --- | --- | --- |
| `models/hand_landmarker.task` | `models/` | MediaPipe 손 랜드마크 |
| `yolo/v2/best.pt` | `yolo/v2/` | 정적 손모양 게이트 |
| `artifacts/four_class_lstm_1layers_001/best_model.pt` | `artifacts/four_class_lstm_1layers_001/` | 기본 LSTM 체크포인트 |

배치가 어긋나면 실행 시점에야 드러나므로 `service/test_resources.py`가 스펙과 코드의 기대 경로가 같은지 검사합니다. 자산 경로를 바꾸면 이 테스트가 먼저 실패합니다.

사용자가 설정에서 다른 체크포인트를 지정하면 그 경로를 우선합니다. 설정 파일은 번들이 아니라 `%LOCALAPPDATA%\JestureService\settings.json`에 저장되므로 프로그램을 다시 받아도 유지됩니다.

## 빌드 후 자체 점검

카메라나 창 없이 번들 구성만 확인합니다. **빌드할 때마다 이걸 먼저 돌리세요.**

```powershell
.\dist\Jesture\Jesture.exe --self-test
```

자산 3개, 의존성, 그리고 LSTM·YOLO·MediaPipe의 **실제 추론 1회**까지 확인합니다. 창 없는 빌드라 결과는 아래 파일에 남습니다. 종료 코드 0이면 정상입니다.

```text
%LOCALAPPDATA%\JestureService\selftest.log
```

로딩만 확인하면 부족합니다. 예를 들어 `torchvision`의 확장 모듈(`_C_stable.pyd`)이 빠져도 import는 통과하고 YOLO의 NMS 단계에서야 `operator torchvision::nms does not exist`로 죽습니다. 그래서 점검이 추론까지 지나갑니다.

## 크기를 줄인 방법

| 제외 대상 | 절약 | 근거 |
| --- | ---: | --- |
| CUDA torch → CPU torch | 약 3.6 GB | 측정상 CPU로도 프레임 예산에 여유가 있습니다. |
| `tensorflow` | 1.5 GB | 코드 어디에서도 참조하지 않습니다. |
| `polars` | 179 MB | ultralytics가 결과를 데이터프레임으로 바꿀 때만 씁니다. 이 서비스는 `boxes`만 직접 읽습니다. |
| PySide6 QtWebEngine·Quick·Charts 등 | 수백 MB | 쓰지 않는 Qt 모듈입니다. |

**표준 라이브러리와 torch 내부 모듈은 제외하지 마세요.** 예를 들어 torch는 `unittest`를 import하므로, 안 쓸 것 같다고 빼면 실행 시점에 `ModuleNotFoundError`로 죽습니다. 제외를 늘렸으면 반드시 `--self-test`로 확인하세요.

## 받은 사람이 확인할 것

1. 압축을 풀고 `Jesture.exe`를 실행합니다. 처음 실행은 로딩이 몇 초 걸립니다. 잘 안 되면 `Jesture.exe --self-test`를 먼저 돌려 보세요.
2. 카메라 권한을 허용합니다. 앱은 시작과 동시에 카메라를 열고 손모양 게이트를 켭니다.
3. **기본 단축키가 지정되지 않은 명령은 키를 보내지 않습니다.** `단축키 · 설정`에서 지정한 뒤, 저장하지 않아도 되는 창에서 먼저 시험하세요.
4. 조작 방법은 [../service/GUI_README.md](../service/GUI_README.md)를 참고하세요.

## 시작 실패를 확인하는 방법

창 없는 빌드라 오류가 화면에 바로 보이지 않습니다. `packaging/entry_gui.py`가 시작 실패를 잡아 대화상자로 알리고 아래 파일에 남깁니다.

```text
%LOCALAPPDATA%\JestureService\startup.log
```

`ModuleNotFoundError`가 보이면 그 모듈을 스펙의 `hiddenimports`에 추가하고 다시 빌드하세요. PyInstaller가 동적 import를 자동으로 찾지 못하는 경우입니다. 자산을 찾지 못한다는 오류면 `datas` 배치를 확인하세요.

## 아직 하지 않은 것

- 코드 서명이 없습니다. SmartScreen 경고가 뜰 수 있으며, 받는 사람이 `추가 정보 → 실행`을 눌러야 합니다.
- 설치 관리자(MSI/Inno Setup)나 자동 업데이트는 만들지 않았습니다. 폴더를 압축해 전달하는 방식입니다.
- 단일 파일(onefile) 빌드는 하지 않습니다. 1GB에 가까운 의존성을 실행할 때마다 임시 폴더에 푸는 방식이라 시작이 매우 느려집니다.
- 실제 카메라·단축키 동작은 빌드 후 사람이 확인해야 합니다. 자동 테스트는 경로 계약과 스펙 구성만 검사합니다.
