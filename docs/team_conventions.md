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
| 라벨 (출력 인덱스 순) | `0 swipe_left`, `1 make_fist`, `2 no_gesture` |
| 좌표 | 카메라 원본, 좌우 반전 없음. swipe_left = 수행자 왼쪽 = 이미지 x 증가 |
| 손 | 오른손 |
| 참가자 | `p001`~`p006` (`user00` 은 담당자 시험 데이터) |
| npz 시간 | 초 단위 → 로더가 ms 로 변환 |
| 품질 | 검출률 ≥ 0.8, 연속 미검출 ≤ 5 (수집기가 이미 거름) |
| 모델 입력 | 30프레임 × 63차원 (21관절 × xyz) |

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
