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

# 기준 (단위: 손바닥 크기 = 손목~중지 뿌리 거리)
SWIPE_MIN_DX = 0.6        # 스와이프: 손목 순이동(x+) 최소. 실제 짧은 스와이프 최소 0.87
SWIPE_MIN_OPEN = 1.1      # 스와이프: 구간 평균 손가락 펴짐 최소. 편 손 1.6~2.4, 기울어진 편 손 1.2~1.5, 주먹 0.6~0.9
                          # (4차 실측: 1.3 이면 기울어진 진짜 스와이프(1.22) 가 막힘 → 1.1)
FIST_MIN_DROP = 0.4       # 주먹: 손가락 펴짐이 (앞부분 최대 → 구간 최소) 이만큼은 줄어야 함. 실제 주먹 0.6~0.7.
                          # 0.25 였을 때 "손가락 꼼지락"(0.26~0.37) no_gesture 가 새어 나감 (09-06 22시 재생 시험) → 0.4
FIST_MAX_MOVE = 3.0       # 주먹: 손목 순이동 최대. 녹화 데이터는 0.0~0.4 이지만 실제 사용에선 손을 올리며 쥐는 게 자연스러움.
                          # (09-06 22시 실측: 위로 1.6~2.5 이동하며 쥔 진짜 주먹 12개가 1.5 에 막힘 → 3.0.
                          #  "쥔 주먹 내리기" 오작동은 FIST_MIN_DROP 조건이 잡으므로 이 조건은 극단적 이동만 거른다.)


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
    return dict(palm=palm, ext_mean=float(ext.mean()), ext_start=float(ext[:k].mean()),
                ext_end=float(ext[-k:].mean()),
                ext_open=float(ext[:third].max()),     # 구간 앞 1/3 의 최대 펴짐 (빨리 쥐어도 첫 프레임의 편 손이 잡힘)
                ext_min=float(ext.min()),              # 구간 중 최소 펴짐 (접힌 순간)
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
    return True, ""


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
    ext = np.linalg.norm(x[:, TIPS, :2] - x[:, WRIST:WRIST + 1, :2], axis=2).mean(axis=1) / palm
    third = max(2, len(x) // 3)
    open_ref = float(ext[:third].max())
    return open_ref >= 1.2 and float(ext[-1]) <= close_ratio * open_ref
