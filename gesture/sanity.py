"""모델 판정을 실행하기 전의 '상식 검사'.

모델은 학습에 없던 동작(예: 주먹을 쥔 채 손을 내리기)을 엉뚱한 명령으로 답할 수 있다.
각 명령이 물리적으로 반드시 갖는 성질만 확인해서, 그게 없으면 실행하지 않는다.
진짜 명령은 이 기준을 크게 넘으므로(스와이프 x이동 ≥1.7 손바닥, 편 손 펴짐 ≥1.6) 정상 동작에는 영향이 없다.

09-06 3차 실측: 주먹 쥔 뒤 손을 내리는 동작이 swipe_left 0.93~1.00 으로 실행됨 → 이 검사 추가.
"""
from __future__ import annotations

import numpy as np

WRIST, MIDDLE_MCP = 0, 9
TIPS = [8, 12, 16, 20]
THUMB_TIP, MIDDLE_TIP = 4, 12

# 기준 (단위: 손바닥 크기 = 손목~중지 뿌리 거리)
SWIPE_MIN_DX = 0.6        # 스와이프: 손목 순이동(x+) 최소. 실제 짧은 스와이프 최소 0.87
SWIPE_MIN_OPEN = 1.1      # 스와이프: 구간 평균 손가락 펴짐 최소. 편 손 1.6~2.4, 기울어진 편 손 1.2~1.5, 주먹 0.6~0.9
                          # (4차 실측: 1.3 이면 기울어진 진짜 스와이프(1.22) 가 막힘 → 1.1)
FIST_MIN_DROP = 0.4       # 주먹: 손가락 펴짐이 (앞부분 최대 → 구간 최소) 이만큼은 줄어야 함. 실제 주먹 0.6~0.7.
                          # 0.25 였을 때 "손가락 꼼지락"(0.26~0.37) no_gesture 가 새어 나감 (09-06 22시 재생 시험) → 0.4
FIST_MAX_MOVE = 3.0       # 주먹: 손목 순이동 최대. 녹화 데이터는 0.0~0.4 이지만 실제 사용에선 손을 올리며 쥐는 게 자연스러움.
                          # (09-06 22시 실측: 위로 1.6~2.5 이동하며 쥔 진짜 주먹 12개가 1.5 에 막힘 → 3.0.
                          #  "쥔 주먹 내리기" 오작동은 FIST_MIN_DROP 조건이 잡으므로 이 조건은 극단적 이동만 거른다.)
# finger_snap (09-07 추가, 엄지+중지 튕기기). 물리적 성질: 튕긴 뒤 엄지끝과 중지끝이 크게 떨어지고, 검지는 펴진 채다.
# 09-07 p006 50개로 보정 (손등이 카메라를 향하는 자연스러운 자세, 거리·각도 다양). "시작에 붙어 있음"은 엄지가 가려져
# 좌표가 흔들리므로(시작 핀치 0.01~1.28) 조건에서 뺐다. 대신 구간 전체 최소 핀치 → 끝 최대 핀치의 증가량을 본다.
# 스냅 50개: pinch_end p5 0.91, release p5 0.78, idx_end p5 1.27. 처음 10개(가까운 거리)로 잡은 1.2 는 멀리서 찍은 22개를 막았다.
# 현재 기준: 스냅 48/50 통과, no_gesture 177개 중 12개 누출(붙인 채 정지·꼼지락 20개 중 1개), 주먹 2, 스와이프 39.
SNAP_MIN_PINCH_END = 0.8    # 구간 뒤 1/3 에서 엄지끝-중지끝 최대 거리 (손바닥 단위). 주먹 p95 0.66, 붙인 채 정지 p50 0.31
SNAP_MIN_RELEASE = 0.6      # (뒤 1/3 최대 핀치) - (구간 전체 최소 핀치). 스냅 p5 0.78, no_gesture p95 0.78, 붙인 채 정지 p95 0.71
SNAP_MIN_INDEX_OPEN = 1.0   # 뒤 1/3 검지 펴짐 평균. 주먹은 0.66~1.0 → 주먹 누출 차단
SNAP_MAX_MOVE = 3.0         # 손목 순이동 최대 (주먹과 동일)
FIST_ALL_CLOSED = 0.6       # 주먹 조기 판정: 네 손가락 각각의 펴짐이 시작 대비 이 비율 이하 (주먹 p95 0.52, 스냅 끝 검지 p5 0.90)
SNAP_HELD_PINCH = 0.6       # 조기 판정용: 구간 안에서 이만큼은 붙어 있었어야 함 (스냅 50개 pinch_min p95 0.90, 붙인 채 정지 p50 0.08)


def _basic(lm: np.ndarray):
    det = ~np.isnan(lm[:, 0, 0])
    x = lm[det]
    if len(x) < 4:
        return None
    palm = float(np.median(np.linalg.norm(x[:, MIDDLE_MCP, :2] - x[:, WRIST, :2], axis=1)))
    if palm < 1e-4:
        return None
    ext = np.linalg.norm(x[:, TIPS, :2] - x[:, WRIST:WRIST + 1, :2], axis=2).mean(axis=1) / palm
    k = max(2, len(x) // 6)
    third = max(2, len(x) // 3)
    dx = float((x[-k:, WRIST, 0].mean() - x[:k, WRIST, 0].mean()) / palm)
    dy = float((x[-k:, WRIST, 1].mean() - x[:k, WRIST, 1].mean()) / palm)
    pinch = np.linalg.norm(x[:, THUMB_TIP, :2] - x[:, MIDDLE_TIP, :2], axis=1) / palm   # 엄지끝-중지끝 거리
    idx_ext = np.linalg.norm(x[:, 8, :2] - x[:, WRIST, :2], axis=1) / palm               # 검지 펴짐
    return dict(palm=palm, ext_mean=float(ext.mean()), ext_start=float(ext[:k].mean()),
                ext_end=float(ext[-k:].mean()),
                ext_open=float(ext[:third].max()),     # 구간 앞 1/3 의 최대 펴짐 (빨리 쥐어도 첫 프레임의 편 손이 잡힘)
                ext_min=float(ext.min()),              # 구간 중 최소 펴짐 (접힌 순간)
                pinch_start=float(pinch[:third].min()),   # 앞 1/3 에서 가장 붙었을 때 (참고용, 손등 방향에선 불안정)
                pinch_min=float(pinch.min()),             # 구간 전체에서 가장 붙었을 때
                pinch_end=float(pinch[-third:].max()),    # 뒤 1/3 에서 가장 떨어졌을 때
                idx_end=float(idx_ext[-third:].mean()),   # 뒤 1/3 검지 펴짐
                dx=dx, dy=dy)


def check(label: str, lm: np.ndarray) -> tuple[bool, str]:
    """(통과 여부, 이유). lm: (T,21,3) NaN 포함 가능(원본 좌표)."""
    b = _basic(lm)
    if b is None:
        return False, "too few frames"
    if label == "swipe_left":
        if b["dx"] < SWIPE_MIN_DX:
            return False, f"swipe but dx {b['dx']:+.2f} < {SWIPE_MIN_DX}"
        if b["ext_mean"] < SWIPE_MIN_OPEN:
            return False, f"swipe but hand not open (ext {b['ext_mean']:.2f} < {SWIPE_MIN_OPEN})"
        return True, f"dx {b['dx']:+.2f} ext {b['ext_mean']:.2f}"
    if label == "make_fist":
        # "앞부분 최대 펴짐 → 구간 중 최소 펴짐" 으로 접힘을 잰다.
        # 09-06 22시: "시작 5프레임 평균 vs 끝" 방식은 빨리 쥔 진짜 주먹(시작 몇 프레임에 이미 반쯤 접힘)을 막았다.
        # 펴는 동작(시작이 이미 주먹)은 앞부분 최대가 낮아 여전히 막힌다.
        drop = 1 - b["ext_min"] / max(b["ext_open"], 1e-6)
        if drop < FIST_MIN_DROP:
            return False, f"fist but fingers did not close (drop {drop:.2f} < {FIST_MIN_DROP}, open {b['ext_open']:.2f} min {b['ext_min']:.2f})"
        if abs(b["dx"]) > FIST_MAX_MOVE or abs(b["dy"]) > FIST_MAX_MOVE:
            return False, f"fist but wrist moved (dx {b['dx']:+.2f} dy {b['dy']:+.2f})"
        return True, f"drop {drop:.2f} move {abs(b['dx']):.2f}/{abs(b['dy']):.2f}"
    if label == "finger_snap":
        release = b["pinch_end"] - b["pinch_min"]
        if b["pinch_end"] < SNAP_MIN_PINCH_END:
            return False, f"snap but thumb-middle not apart at end (pinch_end {b['pinch_end']:.2f} < {SNAP_MIN_PINCH_END})"
        if release < SNAP_MIN_RELEASE:
            return False, f"snap but no release (release {release:.2f} < {SNAP_MIN_RELEASE}, pinch {b['pinch_min']:.2f}->{b['pinch_end']:.2f})"
        if b["idx_end"] < SNAP_MIN_INDEX_OPEN:
            return False, f"snap but index folded (idx {b['idx_end']:.2f} < {SNAP_MIN_INDEX_OPEN})"
        if abs(b["dx"]) > SNAP_MAX_MOVE or abs(b["dy"]) > SNAP_MAX_MOVE:
            return False, f"snap but wrist moved (dx {b['dx']:+.2f} dy {b['dy']:+.2f})"
        return True, f"pinch {b['pinch_min']:.2f}->{b['pinch_end']:.2f} idx {b['idx_end']:.2f} move {abs(b['dx']):.2f}/{abs(b['dy']):.2f}"
    return True, ""


def snap_released_now(lm_window: np.ndarray) -> bool:
    """스냅 조기 판정용: 구간 안에서 엄지끝-중지끝이 붙어 있었고(최소 ≤ SNAP_HELD_PINCH), 지금은 떨어져 있으며
    (≥ SNAP_MIN_PINCH_END, 최소 대비 ≥ SNAP_MIN_RELEASE), 검지가 펴져 있으면 '지금 튕긴 직후'.
    09-07: 스냅은 주먹과 달리 조기 판정이 없어 튕긴 뒤 멈춤 0.27초를 더 기다렸다(구간 0.66~0.82s vs 주먹 0.46s)."""
    det = ~np.isnan(lm_window[:, 0, 0])
    if det.sum() < 6 or not det[-1]:
        return False
    x = lm_window[det]
    palm = float(np.median(np.linalg.norm(x[:, MIDDLE_MCP, :2] - x[:, WRIST, :2], axis=1)))
    if palm < 1e-4:
        return False
    pinch = np.linalg.norm(x[:, THUMB_TIP, :2] - x[:, MIDDLE_TIP, :2], axis=1) / palm
    held = float(pinch[:-1].min())
    now = float(pinch[-1])
    idx_now = float(np.linalg.norm(x[-1, 8, :2] - x[-1, WRIST, :2]) / palm)
    return (held <= SNAP_HELD_PINCH and now >= SNAP_MIN_PINCH_END
            and now - held >= SNAP_MIN_RELEASE and idx_now >= SNAP_MIN_INDEX_OPEN)


def fist_closed_now(lm_window: np.ndarray, close_ratio: float = 0.5) -> bool:
    """조기 판정용: 구간 앞 1/3 의 최대 펴짐 대비 최근 프레임의 펴짐이 close_ratio 이하면 '지금 주먹이 쥐어져 있다'.
    lm_window: 현재 구간의 (T,21,3) (NaN 가능). 최근 프레임이 미검출이면 False."""
    det = ~np.isnan(lm_window[:, 0, 0])
    if det.sum() < 6 or not det[-1]:
        return False
    x = lm_window[det]
    palm = float(np.median(np.linalg.norm(x[:, MIDDLE_MCP, :2] - x[:, WRIST, :2], axis=1)))
    if palm < 1e-4:
        return False
    ext_each = np.linalg.norm(x[:, TIPS, :2] - x[:, WRIST:WRIST + 1, :2], axis=2) / palm   # (T,4) 손가락별
    ext = ext_each.mean(axis=1)
    third = max(2, len(x) // 3)
    open_ref = float(ext[:third].max())
    # 09-07: 네 손가락 '평균'만 보면 스냅 뒤 손(검지만 펴짐, 평균비 0.44~0.84)도 주먹으로 잡혀 스냅 구간을 주먹 규칙이
    # 먼저 닫았다(그 결과 스냅 상식검사가 아슬아슬하게 막힘). 주먹은 네 손가락 전부 접히므로(최대비 p95 0.52) 최대도 본다.
    # 스냅 끝 자세의 최대비(검지)는 p5 0.90 → FIST_ALL_CLOSED 0.6 으로 깔끔히 갈린다.
    return (open_ref >= 1.2 and float(ext[-1]) <= close_ratio * open_ref
            and float(ext_each[-1].max()) <= FIST_ALL_CLOSED * open_ref)
