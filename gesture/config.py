"""학습 파이프라인 공통 설정. 데이터 규약은 JIN 브랜치 수집기(programs/collect_gesture.py)를 따른다.

여기 값을 바꾸면 이미 학습한 모델과 호환되지 않으므로, 바꾸면 다시 학습한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"                # 수집기 저장 위치: dataset/<label>/<pid>_<label>_NNNN.npz
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"   # JIN 에 커밋되어 있음

# ── 라벨 (수집기와 동일. 순서 = 모델 출력 인덱스. 추가는 맨 뒤에만) ─────────
LABELS = ["swipe_left", "make_fist", "no_gesture", "finger_snap"]   # 09-07 finger_snap 추가(엄지+중지 튕기기)
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

# ── 좌표 규약 ─────────────────────────────────────────────────────────
# 수집기는 영상을 좌우 반전하지 않는다 (수행자 시점 라벨, 카메라 원본 좌표).
# swipe_left = 수행자의 왼쪽 = 화면(이미지 x)에서는 오른쪽으로 이동 → x 증가.
MIRROR = False
N_LANDMARKS = 21
USE_Z = True                                   # False 면 (x, y) 만 → 42차원
FEATURE_DIM = N_LANDMARKS * (3 if USE_Z else 2)

# ── 시퀀스 규약 ───────────────────────────────────────────────────────
SEQ_LEN = 30                                   # 모델 입력 길이 (시간 기준 리샘플)
# 수집기가 이미 검출률 ≥0.8, 연속 미검출 ≤5 로 걸러 저장한다. 여기서는 같은 기준으로 재확인만 한다.
MIN_DETECTION_RATIO = 0.8
MAX_GAP_FRAMES = 5

# ── 참가자 ────────────────────────────────────────────────────────────
PARTICIPANT_PATTERN = r"^(p\d{3}|user\d{2})$"  # p001~p006 (user00 은 담당자 시험 데이터)

# ── 학습 ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
