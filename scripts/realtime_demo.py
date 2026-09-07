"""실시간 제스처 인식 데모: 웹캠 → 손 21관절 → 구간 감지 → GRU → gestures.json 기능 실행.

  uv run python scripts/realtime_demo.py              # 연습 모드(키 안 누름) 로 시작
  uv run python scripts/realtime_demo.py --live       # 처음부터 실제 키 입력
  uv run python scripts/realtime_demo.py --camera 1

창에서:  Q 종료 | A 연습↔실제 전환 | R 상태 초기화 | M 화면 좌우반전 표시(판정엔 영향 없음)

영상은 수집기와 같이 좌우 반전 없이 처리한다(config.MIRROR=False). 화면 표시만 M 으로 뒤집을 수 있다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gesture import config, preprocess, sanity  # noqa: E402
from gesture.actions import ActionMapper  # noqa: E402
from gesture.landmarks import HandTracker, draw_landmarks  # noqa: E402
from gesture.model import GestureClassifier  # noqa: E402
from gesture.segmenter import MotionSegmenter  # noqa: E402

COLORS = {"swipe_left": (0, 200, 255), "make_fist": (255, 120, 0), "no_gesture": (160, 160, 160),
          "finger_snap": (80, 220, 80)}

# 실시간 품질 기준 (학습 데이터 기준 0.8 / 5 보다 느슨). 09-06 2차 실측: 빠른 스와이프에서 추적이 7~8프레임
# 끊기는 일이 매번 있어 38개 구간이 버려졌다. 0.25초 구멍은 직선 보간으로 충분히 메워지고, 확신도 문턱이 뒤를 막는다.
REALTIME_MIN_DET = 0.5     # 4차 실측: 가장자리에서 시작한 짧은 동작이 54~59% 로 6개 버려짐 → 0.5
REALTIME_MAX_GAP = 12
# 손 추적 신뢰도 (기본 0.5). 흐릿한 프레임에서도 추적을 이어가도록 낮춤.
TRACK_CONF = 0.3
PRESENCE_CONF = 0.4


def gap_runs(det: np.ndarray) -> list[tuple[int, int]]:
    """미검출 구간 [(시작, 끝)] (프레임 인덱스, 끝 포함)"""
    runs, start = [], None
    for i, ok in enumerate(det):
        if not ok and start is None:
            start = i
        if ok and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(det) - 1))
    return runs


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"카메라 {index} 를 열 수 없습니다. 권한/다른 앱 확인.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def classify_segment(seg, clf):
    """Segment → (label, conf, probs) 또는 (None, reason)

    09-06: 오른손 필터 제거. MediaPipe 는 손등이 보이면 오른손을 Left 로 판정하는 일이 많아
    스와이프 끝에서 손이 돌아가면 구간 전체가 버려졌다. 판정은 모델이 하므로 필터는 정보로만 남긴다."""
    feats = preprocess.sample_to_features(seg.landmarks, seg.timestamps_ms,
                                          min_ratio=REALTIME_MIN_DET, max_gap=REALTIME_MAX_GAP)
    det = ~np.isnan(seg.landmarks[:, 0, 0])
    runs = gap_runs(det)
    gap_txt = ("gaps " + ",".join(f"{a}-{b}" for a, b in runs) + f" of {len(det)}f") if runs else "no gaps"
    if feats is None:
        gap = preprocess.longest_gap(det)
        return None, (f"quality: det {seg.detection_ratio:.0%} (min {REALTIME_MIN_DET:.0%}), gap {gap} (max {REALTIME_MAX_GAP}), "
                      f"{seg.duration_sec:.2f}s, {gap_txt}")
    label, conf, probs = clf.predict(feats)
    return (label, conf, probs), gap_txt


class SessionLog:
    """판정·버림·실행을 한 줄씩 파일과 콘솔에 남긴다. reports/realtime/session_YYYYMMDD_HHMMSS.log"""

    def __init__(self, root: Path = config.REPORTS_DIR / "realtime"):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / time.strftime("session_%Y%m%d_%H%M%S.log")
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, line: str):
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {line}")
        self._f.write(f"[{stamp}] {line}\n")
        self._f.flush()

    def close(self):
        self._f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="시작부터 실제 키 입력")
    ap.add_argument("--model", default=None, help="기본 models/gru_gesture.pt")
    ap.add_argument("--gestures", default=None, help="기본 gestures.json")
    args = ap.parse_args()

    print("모델 로딩...")
    clf = GestureClassifier(model_file=args.model) if args.model else GestureClassifier()
    mapper = ActionMapper(args.gestures) if args.gestures else ActionMapper()
    mapper.dry_run = not args.live
    seg = MotionSegmenter()
    log = SessionLog()
    log.write(f"모델 {clf.arch} | 라벨 {clf.labels} | 매핑: " + ", ".join(f"{k}->{mapper.describe(k)}" for k in clf.labels))
    log.write(f"segmenter on=max({seg.on_thresh},{seg.on_over_floor}*floor) off=max({seg.off_thresh},{seg.off_over_floor}*floor) "
              f"on_frames={seg.on_frames} off_frames={seg.off_frames} "
              f"| quality det>={REALTIME_MIN_DET} gap<={REALTIME_MAX_GAP} | track_conf={TRACK_CONF} presence={PRESENCE_CONF} "
              f"| min_conf={mapper.min_confidence} cooldown={mapper.cooldown_sec}s | mode={'LIVE' if not mapper.dry_run else 'DRY'}")
    print(f"기록 파일: {log.path}")
    print("Q 종료 | A 연습<->실제 | R 초기화 | M 표시 반전")

    cap = open_camera(args.camera)
    mirror_view = False
    t0 = time.perf_counter()
    fps, n_frames, fps_t = 0.0, 0, t0
    last_result = "waiting..."
    last_probs = None
    last_msg = ""
    last_time = 0.0
    flash_until = 0.0

    with HandTracker(track_conf=TRACK_CONF, presence_conf=PRESENCE_CONF) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임 읽기 실패")
                break
            now = time.perf_counter()
            ts_ms = (now - t0) * 1000.0
            lm, hand = tracker.process(frame, int(ts_ms))

            segment = seg.push(lm, ts_ms, hand)
            if seg.last_drop:
                log.write(f"DROP  {seg.last_drop}")
                seg.last_drop = None
            if segment is not None:
                res, why = classify_segment(segment, clf)
                if res is None:
                    last_result = f"skipped: {why.split(',')[0]}"
                    last_probs = None
                    log.write(f"SKIP  {why} | palm={segment.palm:.3f} peak={segment.peak_energy:.3f} hand={segment.handedness or '-'}")
                else:
                    label, conf, probs = res
                    last_probs = probs
                    last_result = f"{label}  {conf:.2f}  ({segment.duration_sec:.2f}s)"
                    seg_start = t0 + segment.timestamps_ms[0] / 1000.0      # 구간 시작 시각 (cooldown 판단용)
                    ok, reason = sanity.check(label, segment.landmarks)
                    if label != "no_gesture" and not ok:
                        last_msg = f"GUARD {reason} -> ignored"
                    else:
                        last_msg = mapper.handle(label, conf, now, start=seg_start)
                    pr = " ".join(f"{l[:5]}={p:.2f}" for l, p in zip(clf.labels, probs))
                    log.write(f"{label:10s} {conf:.2f} {segment.duration_sec:.2f}s{' EARLY' if seg.last_early else ''} det={segment.detection_ratio:.0%} "
                              f"palm={segment.palm:.3f} peak={segment.peak_energy:.3f} floor={seg.noise_floor:.3f} "
                              f"hand={segment.handedness or '-'} [{pr}] {why} | {reason} -> {last_msg}")
                    if label != "no_gesture" and conf >= mapper.min_confidence:
                        flash_until = now + 0.6
                last_time = now

            # ── 화면 ──
            n_frames += 1
            if now - fps_t >= 1.0:
                fps, n_frames, fps_t = n_frames / (now - fps_t), 0, now
            vis = frame.copy()
            if lm is not None:
                draw_landmarks(vis, lm)
            h, w = vis.shape[:2]
            if mirror_view:
                vis = cv2.flip(vis, 1)

            # 상단 패널
            cv2.rectangle(vis, (0, 0), (w, 118), (0, 0, 0), -1)
            mode = "LIVE (keys ON)" if not mapper.dry_run else "DRY RUN (no keys)"
            mode_color = (0, 0, 255) if not mapper.dry_run else (0, 255, 0)
            cv2.putText(vis, f"{mode}   fps {fps:4.1f}   hand: {hand or '-'}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
            st = seg.state
            st_color = (0, 0, 255) if st == "ACTIVE" else (200, 200, 200)
            cv2.putText(vis, f"state: {st}  {seg.active_sec:.2f}s" if st == "ACTIVE" else f"state: {st}",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, st_color, 1, cv2.LINE_AA)
            # 에너지 막대
            e = min(seg.last_energy, 0.3) / 0.3
            cv2.rectangle(vis, (10, 56), (10 + int(e * 300), 66), (0, 200, 255), -1)
            xon = 10 + int(min(seg.on_level, 0.3) / 0.3 * 300)      # 시작 기준 (흰 선)
            xoff = 10 + int(min(seg.off_level, 0.3) / 0.3 * 300)    # 멈춤 기준 (노란 선)
            cv2.line(vis, (xon, 54), (xon, 68), (255, 255, 255), 1)
            cv2.line(vis, (xoff, 54), (xoff, 68), (0, 255, 255), 1)
            cv2.putText(vis, f"energy {seg.last_energy:.3f}  floor {seg.noise_floor:.3f}", (320, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            # 마지막 판정
            col = (255, 255, 255)
            for k, c in COLORS.items():
                if last_result.startswith(k):
                    col = c
            cv2.putText(vis, f"last: {last_result}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            cv2.putText(vis, last_msg[:70], (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
            # 확률 막대 (우측)
            if last_probs is not None:
                for i, (lab, p) in enumerate(zip(clf.labels, last_probs)):
                    y = 130 + i * 22
                    cv2.putText(vis, lab, (w - 230, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(vis, (w - 120, y), (w - 120 + int(p * 110), y + 14), COLORS.get(lab, (200, 200, 200)), -1)
            if mapper.in_cooldown(now):
                cv2.putText(vis, "cooldown", (w - 120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if now < flash_until:
                cv2.rectangle(vis, (2, 2), (w - 3, h - 3), (0, 255, 0), 6)
            cv2.putText(vis, "Q quit | A dry/live | R reset | M mirror view", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("gesture realtime", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("a"), ord("A")):
                mapper.dry_run = not mapper.dry_run
                log.write("MODE  " + ("LIVE (실제 키 입력)" if not mapper.dry_run else "DRY RUN (연습)"))
            if key in (ord("r"), ord("R")):
                seg.reset(); last_result = "reset"; last_probs = None; last_msg = ""
                log.write("RESET")
            if key in (ord("m"), ord("M")):
                mirror_view = not mirror_view

    cap.release()
    cv2.destroyAllWindows()
    log.write(f"END  실행 {len(mapper.log)}건")
    log.close()
    print(f"\n기록 파일: {log.path}")


if __name__ == "__main__":
    main()
