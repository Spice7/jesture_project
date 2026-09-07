# 시연용 서비스 UI — `scripts/gesture_app.py`

웹캠 화면, 제스처 → 키 매핑 편집, 설정, 실행 기록을 한 창에 모은 시연 프로그램. 2차(09-07 오후)부터 customtkinter
(순수 파이썬, `uv sync --locked` 로 같이 설치됨)로 다크 테마 사이드바 구성. 판정 로직은 `scripts/realtime_demo.py` 와
같다 (`gesture/pipeline.py` 가 그 루프를 화면 없이 묶은 것).

참고한 앱들과 가져온 것
- **Logi Options+ / PowerToys**: 왼쪽 사이드바로 페이지 이동, 설정은 카드 단위, 연결 상태·모드는 항상 보이는 자리에.
- **Elgato Stream Deck**: 동작 하나 = 카드 하나, 키는 칩(Ctrl + Alt + Tab)으로 크게, 프로필(프리셋)을 한 번에 교체.
- **Project Gameface**: 카메라 화면을 크게 두고, 판정 문턱은 슬라이더, 무엇이 인식됐는지 즉시 큰 글씨로.

## 실행

```powershell
uv run python scripts/gesture_app.py            # 연습 모드(키 안 누름)로 시작. 시연 전 확인용
uv run python scripts/gesture_app.py --live     # 처음부터 실제 키 입력
uv run python scripts/gesture_app.py --camera 1 # 카메라 번호
```

카메라를 다른 프로그램(수집기, realtime_demo)이 쓰고 있으면 화면이 깨지거나 1 fps 가 된다. 하나만 켠다.

## 화면

```
┌ 사이드바 ─────┬─ 홈 ───────────────────────────────────────────────────────────┐
│ ✋ Jesture     │ ┌ 카메라 640×480 ─────────────┐  ┌ 지금 인식 ────────────────┐ │
│ ● 홈          │ │ [동작 중 0.6s]      [30 fps] │  │ 주먹 쥐기   (큰 글씨, 색) │ │
│ ⌨ 키 매핑     │ │  손 관절 표시                │  │ ▮▮▮▮▮▮▮▮▮ 확신도 100%     │ │
│ ⚙ 설정        │ │  실행되면 초록 테두리          │  │ 실행: Ctrl+Alt+Tab        │ │
│ ≡ 기록        │ └─────────────────────────────┘  ├ 모델 확률 ─────────────────┤ │
│               │  움직임 ▮▮▮▯▯ (시작/멈춤 기준)     │ 라벨별 막대 4개            │ │
│ ┌ 상태 ─────┐ │ ┌ ⇦ 왼쪽 스와이프 ┐ ┌ ✊ 주먹 쥐기 ─────┐ ┌ ✦ 손가락 튕기기 ┐     │
│ │ 카메라 0   │ │ │ [Tab]            │ │ [Ctrl]+[Alt]+[Tab]│ │ [Enter]          │     │
│ │ 게이트     │ │ └─────────────────┘ └──────────────────┘ └─────────────────┘     │
│ │ ○ 실제 키  │ │   (인식되면 해당 카드 테두리가 1.2초 빛남, 오른쪽 위에 토스트 알림)   │
│ └───────────┘ │                                                                  │
└───────────────┴──────────────────────────────────────────────────────────────────┘
```

- **연습 ↔ 실행**: 사이드바 아래 "실제 키 입력" 스위치. 켜면 빨간 토스트로 한 번 알린다(시연 중 모달 창은 방해라 뺌).
- **키 매핑 페이지**: 프리셋 5개(창 전환 / 발표 / 음악·영상 / 브라우저 / 바탕화면)를 세그먼트 버튼으로 고르고 `적용`.
  제스처 카드마다 `키 바꾸기` → 보조키 체크 + 칸을 클릭하고 키 하나 누르기, 또는 재생/음량/브라우저 키를 목록에서 선택.
  `사용` 스위치를 끄면 그 제스처는 동작 없음(다시 켜면 직전 키 복원). 바꾸는 즉시 `gestures.json` 저장 + 인식 루프 반영.
- **설정 페이지**: 확신도 문턱 슬라이더(0.5~1.0, 0.05 단위, 놓으면 저장), 거울처럼 보기, 테마(어둡게/밝게), 카메라 번호,
  게이트 수동 테스트, 인식 상태 초기화, 기록 폴더 열기.
- **기록 페이지**: 판정 한 줄씩(● = 실행됨). 전체 기록은 `reports/realtime/session_*.log`.

## 키 이름

`gestures.json` 의 `keys` 는 `gesture/actions.py` 의 `VK` 표 이름을 쓴다. 보조키 `ctrl alt shift win`, 글자 `a`~`z`,
숫자, `f1`~`f12`, `tab enter esc space backspace delete insert left up right down pageup pagedown home end`,
특수 `media_play_pause media_next media_prev media_stop volume_up volume_down volume_mute browser_back
browser_forward browser_refresh printscreen`. 방향키·Ins/Del 류는 확장 키 플래그를 붙여 보낸다.

## YOLO 게이트 (09-07 오후 연결 완료)

팀(lkh 브랜치) 의 `gesture_model/gesture_detector.py` 를 그대로 가져와 `gesture/gate.py` 로 감쌌다. 가중치는
`models/yolo_gate.pt` (= 드라이브 `dataset_yolo/lkh/v2/best.pt`, yolov8n, 클래스 cancel/start/stop). git 에 안 올라가므로
새 기기에서는 복사해야 한다. GPU 에서 프레임당 6ms, CPU 면 25ms 라 CPU 는 3프레임에 한 번만 돌린다.

포즈는 전부 **3초 유지가 트리거** (팀 코드 hold_seconds, 설정 페이지에서 1~5초 조절):

| 포즈 | 클래스 | 하는 일 |
|---|---|---|
| 손바닥 | start | 게이트 열림 = 동적 제스처 실행 시작 |
| 주먹 | stop | 게이트 닫힘 = 실행 끝. 취소 없음 |
| 세 번째 포즈 | cancel | 정적 명령 하나. 키 매핑 페이지의 "세 번째 포즈" 에 매핑된 키 (기본 Win+D). 게이트 열려 있을 때만 |

- 앱을 켜면 게이트는 **닫힘**으로 시작한다. 카메라 아래에 "손바닥을 3초 보여 주면 시작" 안내, 포즈가 보이면 보라색 칩에
  유지 시간(1.2/3s)이 찬다. 홈 오른쪽 위 게이트 카드와 사이드바에도 같은 상태.
- 닫혀 있는 동안 동적 제스처는 판정·기록만 하고 키는 누르지 않는다 (`GATE closed -> ... ignored`).
- **주먹 명령 vs 주먹 3초(닫기) 충돌**: 게이트가 켜져 있으면 make_fist 실행을 0.8초 미룬다. 그때까지 손이 계속 주먹이면
  "닫으려는 주먹" 으로 보고 버리고(`held fist`), 손을 내렸거나 폈으면 실행한다. → 주먹 명령은 쥐고 0.3초 멈춘 뒤 바로 내린다.
- `gestures.json` 의 `"gate"`: `enabled`(끄면 항상 열림, 재시작 후 적용), `model`, `confidence`(0.7), `hold_seconds`(3.0).
- 모델 파일이 없거나 ultralytics 오류면 게이트 없이(항상 열림) 돌고, 설정 페이지에 이유가 표시된다.
- 카메라 없이 논리만 시험: `gesture.gate.FakeGate` 를 `Pipeline(gate=FakeGate())` 로 넣고 `next_pose` 를 바꿔 준다.

## 파일

| 파일 | 역할 |
|---|---|
| `scripts/gesture_app.py` | 화면(customtkinter). 카메라·인식은 워커 스레드, 화면은 30ms 마다 갱신. 1차 Tkinter 판은 커밋 0a1d4f0 |
| `gesture/pipeline.py` | 프레임 → 관절 → 구간 → GRU → 상식검사 → 게이트 → 실행. `Event`/`FrameState` 로 결과 전달 |
| `gesture/actions.py` | 키 표(`VK`), 표시 이름(`key_display`), 매핑 저장/수정(`ActionMapper.set_action/save/reload`) |
| `gesture/gate.py` | YOLO 게이트 (팀 GestureDetector 래퍼) + FakeGate |
| `gesture_model/` | 팀(lkh) YOLO 포즈 검출 모듈, 수정 없음 |
| `gesture/presets.py` | 프리셋과 제스처 한글 이름, STATIC_LABELS |
| `gestures.json` | 사용자가 바꾼 매핑이 저장되는 곳 |
