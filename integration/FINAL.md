# integration 최종 통합 기록

기준: dev `ae545b7` (개별 브랜치 merge). 시작 시 dev와 origin/dev가 같았고 추적 파일 수정은 없었다. 사용자 dataset/ 238개 파일은 untracked 상태였으며 SHA-256으로 보존을 확인했다. 새 integration에서만 작업했다. main/dev/lkh 및 origin의 모든 브랜치 ref는 유지하며 push하지 않는다.

## A. 통합한 기능과 기준

| 기능 | 기준과 고유 기능 반영 | 최종 위치 |
| --- | --- | --- |
| 단일 기본 실행 | hwangsoon UI + 공통 Pipeline. 빈 main.py를 유일한 사용자 진입점으로 사용 | main.py, scripts/gesture_app.py |
| 정적 인식 | lkh GestureDetector의 confidence, 유지 시간, 순간 미검출, 1회 확정/재시작. kmj02의 장치 자동 선택·IoU·max_det·박스 표시 연결 | gesture_model/, gesture/gate.py |
| 동적 인식 | hwangsoon MediaPipe→구간 분리→GRU/LSTM→규칙 검사. smoothing, 적응 잡음 임계값, 조기 주먹/스냅/스와이프, 손 놓침 대기와 보간 유지 | gesture/ |
| 게이트·키 입력 | start/stop/cancel, 주먹 게이트 의도 구분, 동작 시작 기준 cooldown, 프리셋/매핑 녹화, 연습 모드, 기록 유지 | gesture/pipeline.py, actions.py, scripts/gesture_app.py |
| YOLO 학습/튜닝/평가/추론 | kmj02 설정 기반 코드와 custom 지표/ROC 출력. lkh 학습/평가 설정은 별도 YAML로 같은 엔진에서 실행 | yolo/yolo.py, yolo_param.yaml, v3.yaml |
| 동적 수집 | 두 collect_gesture.py의 실제 차이(반전/녹화 길이)를 CLI 옵션으로 흡수. 기본은 모델 규약에 맞는 비반전/2.5초 | programs/collect_gesture.py |
| 영상 수집·추출·검증 | lkh의 영상별 수집/일괄 추출/NPZ 검증은 고유 기능이므로 유지 | programs/ |
| 이미지 처리 | lkh 웹캠 녹화·간격 추출·반전, kmj02 전체 프레임 추출·Gradio PASS/trash 검토·EDA 모두 유지 | util/, programs/, EDA.ipynb |
| 학습 비교 자료 | GRU/LSTM 학습·평가·재생·비교와 기존 결과/샘플/체크포인트 보존 | scripts/, reports/, artifacts/, data/gestures/ |

정적 gate 모델 yolo_gate.pt는 원문에서 lkh v2/best.pt의 복사 이름이었다. 별도 독립 모델 파일을 강제하지 않고 설정된 정적 모델을 직접 읽는다. v3/kmj02와 v2의 가중치가 동일하다고 판단하지 않았다. 기본 v3와 명시적 다른 모델 선택은 서로 다른 실제 학습 모델이며 정확도는 각각 평가해야 한다.

## B. 제거한 중복

복사본이 남아 있는 상태에서 기능 이전·회귀 검사·모델 smoke를 수행한 후 제거했다.

- implementations/hwangsoon/gesture_model/ → 루트 gesture_model/ 하나. 두 원본은 바이트 동일.
- implementations/hwangsoon/programs/collect_gesture.py → 루트 programs/collect_gesture.py 옵션. 비반전 2.5초와 반전 5초를 모두 지원한다.
- implementations/kmj02/realtime_detect.py, yolo/test_yolo_webcam.py → 통합 UI의 동일 카메라·정적 detector·박스/FPS·게이트·종료 흐름. 독립 웹캠 루프 제거.
- yolo/train_yolo.py, yolo/evaluate_yolo.py → yolo/yolo.py + yolo/v3.yaml. 40 epoch/batch auto/회전 multi_scale/patience 및 test split 평가 보존.
- hwangsoon scripts/realtime_demo.py → 공통 Pipeline과 UI의 연습 전환·반전 표시·초기화·확률/에너지/기록. 별도 동적 분류/키 입력 루프 제거.
- kmj02 dataset/data_EDA.yaml → 정정한 dataset/data.yaml 하나. EDA와 YOLO가 같은 경로/클래스 설정 사용.
- 나머지 implementations 파일은 역할별 루트 디렉터리로 이전했으며 해당 보존 디렉터리는 남기지 않았다. 보고서·모델·체크포인트는 삭제하지 않았다.
- collect_gesture_video.py와 extract_gesture_videos.py, util/slice.py와 programs/slicing.py는 각각 대화형/일괄 처리, 간격/전 프레임 추출 차이가 있어 그대로 유지했다. 알고리즘 공통화는 하지 않았다.

## C. 수정한 기존 소스

아래는 이전만 한 파일을 제외한 모든 수정 Python 파일이다. hwangsoon/kmj02 파일의 이전 경로는 각각 implementations/<branch>/ 아래였다.

| 최종 파일 | 수정 이유·내용 | 기능 로직 변화 |
| --- | --- | --- |
| main.py | 기존 빈 파일에 단일 UI/도구 명령, CWD 고정, check, 누락 파일 안내 추가 | 진입/연결 로직 추가 |
| gesture_model/gesture_detector.py | 모델 존재/클래스 검사, 자동 CPU/GPU, IoU/max_det 전달, UI용 결과 보관 | 시간 상태 알고리즘 유지 |
| gesture/config.py | 동적 데이터 분리, 모델 경로·project_path·추론 준비 검사 | 경로/초기화 |
| gesture/actions.py | 설정 project-relative, 공통 정적 모델·추론 옵션 기본값 | cooldown/키 조합 로직 유지 |
| gesture/gate.py | 공통 정적 모델과 미검출/IoU/max_det/device 연결 | 기존 gate 상태 유지 |
| gesture/landmarks.py | 없는 모델의 자동 다운로드/정의되지 않은 URL 참조 대신 명확한 오류 | 초기화만 |
| gesture/model.py | 모델/metadata 존재 확인, 라벨 순서·mirror/use_z·seq_len/feature_dim·모델 arch 검사 | 호환성 검사 추가, RNN 계산 유지 |
| gesture/dataset.py | mirrored NPZ를 비반전 모델 학습에 혼합하지 않도록 오류 처리 | 입력 검증 추가 |
| gesture/pipeline.py | 공통 정적 모델 옵션 연결, 비Windows 연습 모드 유지, 게이트 오류 시 닫기·pending 해제·수동 재개 방지, 초기 gate 상태 동기화 | 오류 시 기존 자동 열림을 인식 중지로 변경. 분류·규칙·cooldown 유지 |
| scripts/gesture_app.py | 공통 모델/device CLI, 사전 파일 검사, YOLO 결과 박스 표시, 오류 상태 UI, 시스템 폰트 위치 환경변수 | 연결/표시만 |
| scripts/train_model.py | 변경된 동적 데이터 경로에 맞는 안내 | 오류 문구만 |
| programs/collect_gesture.py | mirror/max-duration/camera/dataset 옵션, 비반전 기본, 정확한 mirrored metadata, 혼합 방지 | 기본 반전/시간은 hwangsoon 규약 선택, lkh 방식은 옵션 유지 |
| programs/collect_gesture_video.py | data/gestures 출력 | 경로만 |
| programs/extract_gesture_videos.py | data/gestures 출력 | 경로만 |
| programs/validate_gesture_dataset.py | data/gestures 검사/보고서 | 경로만 |
| programs/check.py | 이전된 예제 NPZ 경로 | 경로만 |
| programs/flip_image.py | 프로젝트 기준 입출력, main guard | 반전 알고리즘 유지 |
| programs/slicing.py | videos/frames 경로와 CLI, 영상 확장자 필터, import 부작용 제거 | 프레임 추출 유지 |
| programs/data_preprocessing.py | 입력한 상대 폴더를 프로젝트 기준 해석 | PASS/trash 동작 유지 |
| util/slice.py | 프로젝트 기준 videos/images 출력, main guard | 녹화/간격 추출 유지 |
| util/valid_npz.py | 특정 참가자 경로를 CLI 입력으로 교체, context manager | 진단 출력 일반화 |
| yolo/yolo.py | predict에서 학습 데이터 의존 제거, YAML 경로 정규화/임시 전달, 빈 데이터 안내, val/device 지원 | 학습·custom 지표 알고리즘 유지 |
| reports/gru_vs_lstm_0907/scripts/run_ours.py | 작성자 절대 경로 제거, raw 출력 위치 | 경로만 |
| reports/gru_vs_lstm_0907/scripts/bench_infer.py | 프로젝트 기준 + external/jin 설정, 누락 안내, raw 출력 | 경로/입력 검사 |
| reports/gru_vs_lstm_0907/scripts/score_jin_ckpt.py | external/jin 설정/누락 안내, raw 출력 | 경로/입력 검사 |
| reports/gru_vs_lstm_0907/scripts/make_report.py | 실제 보존 raw JSON 위치 참조 | 경로만 |

기타 수정: gestures.json 공통 모델/옵션, dataset/data.yaml split 상대 경로, yolo/yolo_param.yaml 새 모델/출력 경로 및 자동 device, EDA.ipynb 단일 YAML 참조, README 및 현재 UI 안내 문서 3개/programs README, .gitignore 실행 산출물 규칙. pyproject.toml과 uv.lock은 dev와 동일하며 패키지 버전을 변경하지 않았다.

## D. 새 코드와 파일

- 신규 Python 파일은 tests/test_integration.py **1개**. 경로/누락/학습-추론 분리/정적 debounce/게이트 실패/cooldown/실제 모델 경계를 검증하는 회귀 검사다.
- main.py는 기존 빈 파일을 구현했다. 별도 launcher/wrapper Python 파일은 생성하지 않았다.
- yolo/v3.yaml은 기존 lkh 학습 설정을 중복 코드 없이 보존하기 위한 설정.
- 빈 데이터 디렉터리와 외부 v3 모델 위치에는 .gitkeep만 추가. 가짜 모델, metadata, 이미지, 라벨 파일은 만들지 않았다.
- 이번 최종 기록은 이 문서 하나다. 기존 integration 자료는 dev 당시 기록으로 그대로 보존한다.

## E. 모델 파일

기본 앱의 외부 필수 파일:

1. models/static/v3.pt — 실제 드라이브의 lkh v3 weight.
2. models/gru_gesture.pt — 실제 학습된 GRU weight.
3. models/gru_gesture.json — 해당 weight의 실제 metadata.

Git 제공 모델: models/hand_landmarker.task, models/static/checkpoints/의 11개 checkpoint, models/static/yolov8n.pt, models/static/yolo26n.pt, artifacts/kmj02/runs/yolo/gesture_yolov8n/weights/의 best/last. 이전된 모델 파일은 원본 SHA-256 동일.

별도 yolo_gate.pt 복제는 불필요. pretrained yolov8n/yolo26n은 제스처 분류용 최종 weight가 아니다. 기본 앱은 학습 기록 artifacts를 필요로 하지 않는다. 실제 사용 모델 변경은 --static-model 또는 gestures.json에서 가능하며, GRU/LSTM 변경은 --model과 짝인 metadata로 지정한다.

## F. 데이터셋

- 추론용 데이터셋: **없음**. 모델/metadata/실행 설정과 카메라 입력만 필요.
- 정적 재학습: dataset/{train,valid,test}/{images,labels}, dataset/data.yaml. YAML의 train/val/test가 각 images 디렉터리와 일치한다. labels는 Ultralytics의 images→labels 규약으로 대응하며 회귀 검사로 확인했다.
- v3 재학습: datasets/static_gesture_v3/data.yaml 및 train/valid/test의 images/labels. YAML 포함 실제 자료는 사용자가 배치한다.
- 동적 재학습: data/gestures/<label>/*.npz. 예제 user00 37개만 Git 제공. 실제 4클래스/참가자별 데이터를 준비해야 한다.
- 기존 사용자 dataset/의 NPZ와 CSV 238개는 그대로 유지. 신규 YOLO config/빈 디렉터리만 추가했고 사용자 파일은 stage하지 않았다.
- 코드가 YAML 위치에서 path/split을 해석하고 런타임 임시 YAML에 절대 경로로 전달한다. 이는 이식 가능한 원본 설정을 유지하기 위한 실행 중 정규화이며, PC 경로를 저장소 설정에 기록하지 않는다.

## G. 절대 경로

- run_ours.py와 bench_infer.py의 작성자 C:\Users\Admin 경로를 __file__ 기반 루트로 변경.
- 외부 JIN 비교 소스는 external/jin(또는 JESTURE_JIN_DIR)로 명시. 없는 경우 주 앱과 무관한 외부 자료 누락 안내.
- UI 폰트 C:/Windows/Fonts는 WINDIR 기반으로 변경.
- 실행 코드의 Path 호출에 작성자 PC 절대 경로가 남지 않았음을 AST로 확인.
- EDA 저장 출력, 과거 보고서·README·dev 검증 JSON의 과거 경로는 역사 자료로 보존. 사용자에게 입력 예시로 보이는 경로와 런타임에 출력되는 실제 절대 경로는 하드코딩 의존성이 아니다.

## H. 검증

사용자 .venv를 변경하지 않고 TEMP/gesture-integration-20260907/venv의 Python 3.12 환경을 사용했다.

| 실제 명령/검사 | 결과 |
| --- | --- |
| UV_PROJECT_ENVIRONMENT=<검증 환경>; uv sync --locked | 150개 해석, 설치 환경 125개 일치 |
| uv pip check --python <검증 환경>/Scripts/python.exe | 125개 모두 호환 |
| Python compile(source, path, exec) | 최종 40개 Python 문법 통과 |
| python -m unittest discover -s tests -v | 11개 회귀 검사 통과 |
| python <절대 경로>/main.py check (외부 CWD) | 기대한 종료 2, 실제 외부 파일 3개 누락 안내. 학습 데이터 검사 없음 |
| python <절대 경로>/main.py (외부 CWD) | 기대한 종료 2, 누락 파일 목록. 카메라 시작 없음 |
| main.py app/yolo/collect/frames/train-dynamic/eval-dynamic/compare/replay/review/record/collect-video/extract/validate/flip --help | 14개 모두 종료 0 |
| 실제 HandTracker 초기화/메모리상 빈 프레임 처리 | 통과 |
| 기존 NPZ 37개 로드 및 실제 샘플 전처리 | 통과 |
| 기존 kmj02 last_100.pt + StaticGate CPU 추론 | 통과, 빈 프레임 처리 및 plot 출력 확인 |
| main.py yolo --mode predict --config <임시 설정> --source <기존 train_batch0.jpg> (외부 CWD) | 학습 data 키를 제거한 설정으로 실제 추론 종료 0, 1개 이미지 처리. 검출 0개이며 정확도 검증을 의미하지 않음 |
| 숨긴 App 위젯 생성/종료 (_start_worker만 차단) | 홈/매핑/설정/기록/도움말 생성 확인. 카메라·키 입력 없음 |
| 이전된 binary/샘플/기록 asset hash 비교 | 원본 동일 |
| 사용자 dataset 파일 SHA-256 | 기존 238개 모두 동일 |

첫 회귀 실행에서 테스트용 gate 객체의 hold_seconds가 숫자가 아니어서 실패했으며 테스트 설정을 수정한 뒤 재실행했다. 제품 알고리즘 오류로 처리하거나 성공으로 숨기지 않았다. Gradio 앱 구성 중 외부 라이브러리의 event-loop ResourceWarning이 있었으나 검사 종료 코드는 0이다.

외부 GRU 파일은 만들거나 대체하지 않았다. GRU/LSTM 순전파 검사는 메모리에서 초기화한 네트워크의 인터페이스/shape 검사이며 실제 학습 모델의 인식률을 검증한 것이 아니다. 모델 파일 없이 동적 경계 회귀 검사는 명시적 테스트 객체를 사용했다. 기존에 별도 테스트/lint/type-check 설정은 없었다.

## I. 직접 검증 필요

코드/초기화 수준 검증 완료, 실제 하드웨어 검증 필요.

- 실제 GRU/v3 파일과 metadata로 check --load-models 실행.
- 웹캠 해상도/번호/권한, 실제 손 인식과 start/stop/cancel 확인.
- 스와이프/주먹/스냅, 잠깐 손 놓침, 연속 명령 cooldown, 게이트 오류 후 중지 확인.
- 연습 모드 확인 후 실제 Windows 키 입력을 직접 켜서 검증.
- 실제 데이터 전체 학습, 독립 test split 평가, 모델 버전별 정확도 측정.

## J. Git와 역사 자료

작업 브랜치는 integration이며 main에 merge하거나 remote에 push하지 않는다. dev와 개인 브랜치 refs는 시작 값 그대로 유지한다. 기존 integration/REPORT.md, source-manifest.json, validation-results.json은 **dev의 보존 통합 당시 상태**를 가리키며 최종 경로에는 적용되지 않는다. 삭제하지 않고 이 문서와 루트 README에서 구분했다.

작업 트리 전체( .git/기존 .venv/cache 제외)의 실제 conflict marker 검색 결과 0개. 원본 주석의 장식용 연속 등호는 merge marker가 아니므로 스타일 정리 목적으로 변경하지 않는다. 사용자 기존 untracked dataset/index.csv와 dataset/validation_report.csv는 커밋에 포함하지 않는다. 최종 커밋/status는 완료 응답에도 기록한다.
