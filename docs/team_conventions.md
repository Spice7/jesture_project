# 학습 파이프라인 안내 (hwangsoon 브랜치)

이 브랜치는 **JIN 브랜치(팀 데이터 창고 + 수집기)를 그대로 물려받고, 그 위에 학습·평가 코드를 얹은 것**이다.
데이터 수집 방법은 이 저장소의 `README.md` 와 `programs/README.md`(JIN 담당자 작성)를 따른다.
이 문서는 수집된 `dataset/` 을 어떻게 모델로 만드는지만 다룬다.

## 1. 폴더 구조

```
programs/collect_gesture.py   수집기 (JIN, 건드리지 않는다)
dataset/<label>/<pid>_<label>_NNNN.npz   수집 데이터 (JIN 에서 merge 로 가져온다)
gesture/                      학습 패키지
  config.py     라벨·시퀀스 길이·품질 기준 (수집기 규약과 동일하게 맞춰둠)
  dataset.py    npz 로딩 + 형식 검사, 사람 단위 분할, 배열 생성
  preprocess.py 결측 보간 → 30프레임 리샘플 → 정규화 → 63차원, 증강
  model.py      GRU(기본) / LSTM 정의, GestureClassifier 추론 래퍼
  training.py   학습·평가 공용
  baseline.py   룰 분류기 (비교용)
  landmarks.py  MediaPipe 래퍼 (실시간 추론 루프용, 수집에는 안 씀)
scripts/
  dataset_summary.py   데이터 현황·형식 오류·품질 미달
  train_model.py       GRU 학습 → models/gru_gesture.keras
  compare_models.py    시드 여러 개로 GRU (+LSTM) vs baseline 표 → reports/
docs/
  setup_windows.md     Windows 노트북 세팅 (녹화용·GPU용 2대)
  why_not_jester.md    공개 데이터셋을 안 쓴 이유
```

## 2. 데이터 가져오기

팀원 데이터는 JIN 에 모인다. 우리 브랜치로 가져오려면:

```powershell
git fetch origin
git merge origin/JIN
```

거의 항상 그냥 합쳐진다. 충돌이 나면 `pyproject.toml`/`uv.lock` 정도이고, 우리 쪽은 `scikit-learn` 한 줄 추가가 전부다.
merge 뒤 `uv sync` 한 번.

**JIN 에 데이터 말고 다른 코드(담당자의 모델 작업 등)까지 쌓이기 시작하면** merge 대신 데이터 폴더만 꺼내온다.
그러면 그쪽 코드는 우리 브랜치에 섞이지 않는다.

```powershell
git fetch origin
git checkout origin/JIN -- dataset          # dataset/ 폴더만 JIN 최신으로
git commit -m "JIN 데이터 동기화"
```

```powershell
uv run python scripts/dataset_summary.py    # 누가 몇 개 냈는지, 형식 오류·품질 미달이 있는지
```

**형식 오류가 뜨면 수집기가 바뀐 것이다.** `gesture/dataset.py` 의 `load_sample` 이 npz 키·단위·라벨 일치를 검사한다.
JIN 담당자에게 무엇이 바뀌었는지 확인하고 로더를 맞춘다.

## 3. 규약 (수집기와 동일)

| 항목 | 값 |
|---|---|
| 라벨 (출력 인덱스 순) | `0 swipe_left`, `1 make_fist`, `2 no_gesture`, `3 finger_snap` (09-07 추가) |
| 좌표 | 카메라 원본, 좌우 반전 없음. swipe_left = 수행자 왼쪽 = 이미지 x 증가 |
| 손 | 오른손 |
| 참가자 | `p001`~`p006` (`user00` 은 담당자 시험 데이터) |
| npz 시간 | 초 단위 → 로더가 ms 로 변환 |
| 품질 | 검출률 ≥ 0.8, 연속 미검출 ≤ 5 (수집기가 이미 거름) |
| 모델 입력 | 30프레임 × 63차원 (21관절 × xyz) |

## 3-1. finger_snap 수집 안내 (09-07 추가)

엄지와 중지를 튕기는 동작. 수집기는 실행 뒤 Participant ID 와 Gesture Label 을 물어본다(옵션 없음). 라벨은 자유 문자열이라 그대로 쓴다.

```powershell
uv run python programs/collect_gesture.py
Participant ID: p00X
Gesture Label: finger_snap
```
SPACE 시작 → 동작 → SPACE 종료. 수집기 기준: 0.6~2.5초, 20프레임 이상, 검출률 0.8 이상, 연속 미검출 5 이하.

- **시연 때 할 자연스러운 자세로** (보통 손등이나 손 옆면이 카메라 쪽). 손목은 제자리. 09-07 p006 10개는 이 자세로 찍었고 검출·구간 감지 모두 정상.
- 한 클립 = 엄지·중지를 붙인 자세로 0.3초 정지 → 튕김 → 튕긴 자세로 0.3초 정지. 정지가 없으면 구간 감지기가 시작/끝을 못 잡는다.
- 튕기는 순간은 0.1초라 프레임 2~3장에만 담긴다. 모델은 "붙은 손 모양 → 튕긴 손 모양" 변화를 배우므로 앞뒤 정지가 중요하다.
- no_gesture 에 **"엄지·중지 붙인 채 가만히 있기"** 와 **"손가락 꼼지락"** 을 꼭 넣는다. 안 그러면 붙인 자세만으로 스냅으로 오인한다.
- 1인당 30개 이상. 양은 make_fist 와 비슷하게.

코드에서 finger_snap 에 걸린 곳: `config.LABELS`(맨 뒤), `sanity.py`(끝에서 엄지끝-중지끝 ≥1.2 & 증가량 ≥0.6 & 검지 펴짐 ≥1.0. p006 10개로 보정, 스냅 10/10 통과·no_gesture 누출 6/157·주먹 0), `baseline.py`(같은 룰),
`dataset.REVERSE_SETS["all"]`(스냅 되감기도 no_gesture), `gestures.json`(Play/Pause), `realtime_demo` 색.
"시작에 붙어 있음" 조건은 뺐다: 손등 방향에서는 엄지가 가려져 시작 핀치가 0.08~1.28 로 흔들린다. 끝에서 떨어짐과 검지 펴짐은 안정적.
다른 사람 데이터가 오면 `scripts/label_stats.py --labels finger_snap` 으로 통과율이 유지되는지 확인하고 필요하면 `sanity.SNAP_*`/`baseline.SNAP_*` 재보정.
그 다음 `train_model.py --test-persons --val-persons --reverse-neg both --out models/gru_gesture_4cls.pt` 로 4클래스 재학습
→ `replay_segments.py --model models/gru_gesture_4cls.pt` 로 확인. **기존 3클래스 모델은 스냅을 전부 no_gesture(0.98~1.00) 로 본다**(오작동은 없고 그냥 무시됨).

09-07 4클래스 재학습 결과 (p006 스냅 50 + no_gesture 보강 20, 전원 학습·랜덤 val 15%):
- `--reverse-neg all`(스냅 되감기 포함) 은 재생에서 주먹이 56/60 으로 떨어짐(4개 → no_gesture). `both`(주먹·스와이프 되감기만) 는 주먹 60/60 → **both 채택**.
- 채택 모델 재생(전원, 라벨당 15): swipe 74/75, fist 60/60, snap 15/15(실행 14, 1개는 상식검사), 오작동 4 = 전부 p002 라벨 착오 클립(0005/0017/0019). p006 만(라벨당 25): 75/75, 오작동 0.
- 사람 단위 검증(test p003, val p002, 시드 3): GRU 95.3±2.8, no_gesture 재현율 80.6±11.3 → 3클래스 때(92.4±3.6 / 69.1±14.8)와 같거나 나음. 스냅은 p006 뿐이라 처음 보는 사람 성능은 아직 못 잼.
- 상식검사 기준은 50개로 재보정(끝 거리 1.2→0.8). 스냅 48/50 통과, no_gesture 누출 12/177, 붙인 채 정지·꼼지락 20개 중 1개.
- 시연 모델 교체는 `models/gru_gesture_4cls.pt` → `models/gru_gesture.pt` 로 복사(.json 포함). 교체 전 `realtime_demo.py --model models/gru_gesture_4cls.pt` 로 웹캠 확인.
- **주먹 조기 판정 강화**: 네 손가락 평균만 보면 스냅 뒤 손(검지만 펴짐)도 주먹으로 잡혀 스냅 구간을 주먹 규칙이 먼저 닫았다 → 손가락별 최대 펴짐도 시작 대비 0.6 이하여야 주먹(`sanity.FIST_ALL_CLOSED`. 주먹 p95 0.52, 스냅 끝 검지 p5 0.90). 주먹 313/317 실행 유지.
- **스냅 조기 판정** (`segmenter.early_snap`, `sanity.snap_released_now`): 웹캠 실측에서 스냅이 주먹보다 0.2~0.3초 느렸다(주먹은 조기 판정이 있고 스냅은 튕긴 뒤 멈춤 0.27초를 기다림).
  엄지·중지가 붙어 있다가 떨어진 상태가 4프레임 이어지면 바로 닫는다. 50개 실측: 3f 는 46/50, **4f 49/50(조기 판정 없을 때와 동일), 판정 시점 −293ms**, 붙인 채 정지 20개 실행 0. 재생 시험 스냅 15/15·25/25 유지.
- **명령 뒤 손 내리기 오인 (09-07 웹캠 2차 실측)**: 스냅·주먹 실행 뒤 손을 내리는 구간을 모델이 같은 명령으로 1.00 확신 → 상식검사가 아슬아슬하게 차단(0.58 vs 기준 0.6). 재생 시험 `--lower`(명령 클립 뒤에 내리기 0.5초 합성)로 재현: 내리기 구간이 명령으로 판정된 게 75건(전부 차단).
  대책 = **학습 데이터에 내리기 부정 예시**: ① `dataset.lowering_negatives` 합성(끝 자세 유지 → ±40° 아래로 3~5 손바닥, 손가락 0~30% 풀림, `train_model.py --lower-neg 0.5`), ② 실제 녹화 no_gesture 20개(p006 0071~0090: 튕긴 손 모양/주먹 모양으로 시작해 내리기).
  검증: 실제 내리기 20개(학습 전)에 대해 옛 모델 0/20 no_gesture(17건 상식검사 차단) → 합성 학습 모델 **20/20 no_gesture**. `--lower` 재생: 명령 판정 75 → 2~7건. 최종 모델 = 합성+실제, 시드 42 (시드 1·2 는 스냅 12~14/15 로 흔들림. 스냅은 한 사람 50개뿐이라 시드 민감).
  상식검사·쿨다운은 2차 방어로 유지.
- 웹캠 실측(10:49~10:51, 새 모델): 스냅 15건 전부 0.82~1.00, 주먹 12건, 오작동 0. **"화면 밖에서 들어오며 바로 주먹"은 인식 안 됨**(4건 전부 상식검사가 차단, 실행은 안 됨): 들어오는 손이 옆으로 기울어 반쯤 접힌 채 이동해 편 손 구간이 없음. 데이터 규약(편 손으로 멈춘 뒤 쥐기)과 다른 상황 → 시연 습관으로: 화면 안에서 편 채 잠깐 멈춘 뒤 쥔다.

## 4. 전처리 (`gesture/preprocess.py`)

1. NaN 프레임을 시간축 선형 보간
2. 타임스탬프 기준 30프레임으로 리샘플 (녹화 길이가 달라도 같은 길이)
3. 시퀀스 전체의 손목 평균을 원점으로, 평균 손바닥 크기로 나눔.
   **프레임별로 손목 기준 정규화하면 안 된다** (이동 정보가 사라져 스와이프를 못 배운다)
4. 증강(학습 시만): 스케일·평행이동·시간 왜곡·노이즈. **좌우반전은 안 한다** (swipe_right 가 없다)

## 5. 평가 규약

- **사람 단위 분할.** test 에 들어간 사람의 샘플은 학습에 한 개도 넣지 않는다. 그래야 "처음 보는 사람도 되나"를 잰다.
- 4명 이상: 1명 test + 1명 val. 2~3명: 1명 test. 혼자면 랜덤 분할 (점수 과대평가).
- **val 로 모델을 고르고, test 는 마지막에 한 번만 본다.** test 보고 고르면 그 숫자를 발표에 못 쓴다.
- 항상 baseline(룰) 과 나란히 보고한다.

```powershell
uv run python scripts/train_model.py --test-persons p003 --val-persons p002          # GRU 한 번 학습 + 리포트
uv run python scripts/compare_models.py --seeds 3 --test-persons p003 --val-persons p002   # 시드 3개 평균 표
uv run python scripts/compare_models.py --archs gru lstm --seeds 3 ...                  # 팀원 LSTM 과 나란히
```

## 6. 모델 비교 규칙

- 학습은 초기값이 랜덤이라 한 번 돌릴 때마다 2~3%p 씩 흔들린다. **한 번씩 돌려 비교하지 않는다.**
- `compare_models.py` 가 같은 분할에 시드 N개를 돌려 평균 ± 표준편차를 낸다. 차이가 표준편차 안이면 동률.
- 결과는 `reports/compare_models_summary.md`(표) 와 `reports/compare_models.csv`(원본). 발표 자료에 그대로 쓴다.
- baseline 이 스와이프는 잘 잡지만 no_gesture 거부와 make_fist 에서 흔들린다는 걸 숫자로 보여주는 게 목표.

## 7. 실시간 추론 (다음 단계)

`GestureClassifier(arch="gru")` 가 `models/gru_gesture.keras` 를 읽는다. 입력은 전처리와 같은 (30, 63).
카메라 → MediaPipe(`gesture/landmarks.py`) → 구간 감지 → 리샘플·정규화 → 분류 → gestures.json 매핑 → 실행.

## 8. Git

- 이 브랜치에서 작업하고 main 으로 PR. **JIN 이 main 에 먼저 들어간 뒤** 우리가 따라간다.
- `gesture/config.py` 를 바꾸면 모델을 다시 학습한다.
- 수집 데이터는 이 브랜치에 직접 커밋하지 않는다. `data/p00X` 브랜치 → JIN PR → merge 로 가져온다.
