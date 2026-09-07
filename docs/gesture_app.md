# 시연용 서비스 UI — `scripts/gesture_app.py`

웹캠 화면, 제스처 → 키 매핑 편집, 실행 기록을 한 창에 모은 시연 프로그램. Tkinter 로 만들어 추가 설치가 없다.
판정 로직은 `scripts/realtime_demo.py` 와 같다 (`gesture/pipeline.py` 가 그 루프를 화면 없이 묶은 것).

## 실행

```powershell
uv run python scripts/gesture_app.py            # 연습 모드(키 안 누름)로 시작. 시연 전 확인용
uv run python scripts/gesture_app.py --live     # 처음부터 실제 키 입력
uv run python scripts/gesture_app.py --camera 1 # 카메라 번호
```

카메라를 다른 프로그램(수집기, realtime_demo)이 쓰고 있으면 화면이 깨지거나 1 fps 가 된다. 하나만 켠다.

## 화면

```
┌ 손 제스처 PC 제어      카메라 0 연결됨   게이트: 미연결   [연습 모드 ▶ 실행 켜기] ┐
│ ┌──────────── 카메라 ────────────┐  ┌ 제스처 → 키 │ 설정 │ 기록 ┐               │
│ │  손 관절 표시                   │  │ 왼쪽 스와이프   Tab         [변경] [끄기] │ │
│ │  실행되면 초록 테두리 0.5초       │  │ 주먹 쥐기       Ctrl+Alt+Tab [변경] [끄기] │ │
│ └────────────────────────────────┘  │ 손가락 튕기기   Enter       [변경] [끄기] │ │
│ 대기/동작 중/쿨다운  ▮▮▮ 에너지   fps │ 프리셋: [창 전환 (기본) ▾] [적용]          │ │
│ 마지막 판정  (예: 주먹 쥐기 100%)     │ 저장됨 14:20:03                           │ │
│ 무슨 일이 있었는지 한 줄              └──────────────────────────────────────────┘ │
│ 라벨별 확률 막대                                                                   │
└ 팁: 주먹은 쥔 뒤 0.3초 멈추고 내리기 · 스와이프는 자기 왼쪽으로 · 명령 사이 0.5초 ┘
```

- **연습 ↔ 실행**: 오른쪽 위 버튼. 실행으로 바꿀 때 한 번 확인창이 뜬다. 실행 중이면 버튼이 빨갛다.
- **키 바꾸기**: `변경` → 보조키(Ctrl/Alt/Shift/Win) 체크 + 칸을 클릭하고 키 하나 누르기. 재생/음량/브라우저 키처럼
  키보드로 못 누르는 건 목록에서 골라 `이 키 쓰기`. `지우기` 는 그 제스처를 동작 없음으로.
  바꾸면 즉시 `gestures.json` 에 저장되고 인식 루프에도 바로 반영된다.
- **프리셋**: 창 전환(기본) / 발표(PowerPoint) / 음악·영상 / 브라우저 / 바탕화면. `gesture/presets.py` 에 정의.
- **설정**: 확신도 문턱(기본 0.8), 화면 좌우반전(판정엔 영향 없음), 카메라 번호, 게이트 수동 테스트, 상태 초기화.
- **기록**: 최근 판정 한 줄씩. 전체 기록은 `reports/realtime/session_*.log` (설정 탭에 경로 표시).

## 키 이름

`gestures.json` 의 `keys` 는 `gesture/actions.py` 의 `VK` 표 이름을 쓴다. 보조키 `ctrl alt shift win`, 글자 `a`~`z`,
숫자, `f1`~`f12`, `tab enter esc space backspace delete insert left up right down pageup pagedown home end`,
특수 `media_play_pause media_next media_prev media_stop volume_up volume_down volume_mute browser_back
browser_forward browser_refresh printscreen`. 방향키·Ins/Del 류는 확장 키 플래그를 붙여 보낸다.

## YOLO 게이트 연결 자리

정적 포즈(YOLO)로 ON/OFF 하는 게이트는 아직 연결 전이다. 연결은 한 줄이다:

```python
app.pipeline.set_gate(True)    # 손바닥 포즈 3초 → 열림
app.pipeline.set_gate(False)   # 주먹 포즈 3초 → 닫힘
```

- 닫혀 있으면 판정은 기록만 하고 키를 누르지 않는다 (`GATE closed -> ... ignored`).
- 한 번이라도 `set_gate` 가 불리면 상단 표시가 "게이트: 열림/닫힘 (YOLO)" 로 바뀌고 수동 테스트는 꺼진다.
- YOLO 추론을 어디서 돌릴지는 두 가지: ① 같은 프레임을 `App._run` 루프 안에서 YOLO 에도 넣기(권장, 카메라 하나),
  ② 별도 스레드/프로세스에서 돌리고 결과만 `set_gate` 로 넘기기.
- 주먹 명령(make_fist, 0.3초)과 게이트 닫기(주먹 3초 유지)는 시간대가 달라 겹치지 않는다. 다만 게이트를 닫으려고
  천천히 쥐면 조기 판정으로 make_fist 가 한 번 실행될 수 있다. 문제가 되면 "닫힘 전환 시 직전 0.5초 명령 취소" 를 붙인다.

## 파일

| 파일 | 역할 |
|---|---|
| `scripts/gesture_app.py` | 화면(Tkinter). 카메라·인식은 워커 스레드, 화면은 30ms 마다 갱신 |
| `gesture/pipeline.py` | 프레임 → 관절 → 구간 → GRU → 상식검사 → 게이트 → 실행. `Event`/`FrameState` 로 결과 전달 |
| `gesture/actions.py` | 키 표(`VK`), 표시 이름(`key_display`), 매핑 저장/수정(`ActionMapper.set_action/save/reload`) |
| `gesture/presets.py` | 프리셋과 제스처 한글 이름 |
| `gestures.json` | 사용자가 바꾼 매핑이 저장되는 곳 |
