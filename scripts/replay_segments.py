"""카메라 없이 실시간 루프를 검증: 녹화 npz 를 이어 붙여 가짜 스트림을 만들고
구간 감지기 → 전처리 → GRU → 매핑까지 그대로 돌려 본다.

  uv run python scripts/replay_segments.py                 # 참가자별 몇 개씩 샘플링
  uv run python scripts/replay_segments.py --per-label 30 --persons p006

각 클립 사이에 "정지" 프레임(마지막 자세 유지 + 미세 잡음)을 끼워 실제 흐름과 비슷하게 만든다.
확인하는 것:
  1. 명령 클립(swipe/fist)이 구간으로 잘 잘리는가 (감지율)
  2. 잘린 구간의 판정이 맞는가 (정확도)
  3. no_gesture 클립에서 기능이 실행되지 않는가 (오작동 수)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from gesture import config, dataset, preprocess, sanity  # noqa: E402
from gesture.actions import ActionMapper  # noqa: E402
from gesture.model import GestureClassifier  # noqa: E402
from gesture.segmenter import MotionSegmenter  # noqa: E402


def idle_frames(last_lm: np.ndarray, n: int, rng, jitter=0.0003):
    """정지 프레임: 마지막 자세 + 아주 작은 떨림.
    실제 정지 손의 에너지(관절 평균 이동/손바닥)는 중간 0.007, 90%ile 0.02 수준. 손바닥 크기가 정규화 좌표로
    0.1 안팎이므로 좌표 잡음 0.0003 이면 에너지 ≈ 0.004 가 되어 실제와 비슷하다."""
    for _ in range(n):
        yield last_lm + rng.normal(0, jitter, size=last_lm.shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=15, help="참가자×라벨당 클립 수")
    ap.add_argument("--persons", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--leave-frame", type=float, default=0.0,
                    help="명령 클립 뒤쪽 이 비율의 프레임을 미검출(NaN)로 바꿔 '손이 화면 밖으로 나간' 상황을 흉내냄. 예 0.3")
    ap.add_argument("--hole", type=int, default=0,
                    help="명령 클립의 가장 빠른 지점에 이 프레임 수만큼 미검출 구멍을 냄 (빠른 동작에서 추적 끊김 흉내). 예 7")
    ap.add_argument("--min-det", type=float, default=0.5, help="실시간 품질 기준: 검출률 (realtime_demo 와 동일)")
    ap.add_argument("--max-gap", type=int, default=12, help="실시간 품질 기준: 연속 미검출 (realtime_demo 와 동일)")
    ap.add_argument("--fist-lower", action="store_true",
                    help="make_fist 클립 뒤에 '주먹 쥔 채 손 내리기' 0.5초를 합성해 붙임 (3차 실측의 오작동 재현)")
    ap.add_argument("--no-guard", action="store_true", help="상식 검사(sanity) 끄기 (효과 비교용)")
    ap.add_argument("--noise-rel", type=float, default=0.0,
                    help="모든 프레임에 손바닥 크기 × 이 값의 좌표 잡음을 더함 (먼 거리 손의 떨림 흉내). 예 0.02")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    samples = dataset.load_dataset(config.DATASET_DIR)
    if args.persons:
        samples = [s for s in samples if s.person in args.persons]
    by = defaultdict(list)
    for s in samples:
        by[(s.person, s.label)].append(s)
    picked = []
    for k, v in sorted(by.items()):
        idx = rng.permutation(len(v))[: args.per_label]
        picked += [v[i] for i in idx]
    rng.shuffle(picked)
    print(f"클립 {len(picked)}개로 스트림 구성 (참가자 {sorted({s.person for s in picked})})")

    clf = GestureClassifier()
    mapper = ActionMapper(dry_run=True)
    seg = MotionSegmenter(fps_hint=args.fps)
    dt = 1000.0 / args.fps

    t = 0.0
    detected = Counter(); correct = Counter(); total = Counter(); fired = Counter(); skipped = Counter(); guarded = Counter(); multi = 0
    misses = []; guard_false = []; ignored = Counter(); misfire_detail = []

    drops = Counter()

    def feed(lm, hand="Right"):
        nonlocal t
        if lm is not None and args.noise_rel > 0:
            palm = float(np.linalg.norm(lm[9, :2] - lm[0, :2]))
            lm = lm + rng.normal(0, args.noise_rel * palm, size=lm.shape).astype(np.float32)
        out = seg.push(lm, t, hand)
        if seg.last_drop:
            drops[seg.last_drop.split(" (")[0]] += 1
            seg.last_drop = None
        t += dt
        return out

    # 시작 전 정지 1초
    first = picked[0].landmarks
    first = first[~np.isnan(first[:, 0, 0])][0]
    for f in idle_frames(first, int(args.fps), rng):
        feed(f)

    for s in picked:
        total[s.label] += 1
        lm_seq = s.landmarks
        # 클립 시작 자세로 0.7초 정지: 손이 "다음 시작 위치에 이미 놓여 있는" 상태. (이걸 안 하면 이전 클립 자세에서
        # 순간이동한 프레임이 pre-roll 에 섞여 손목 이동량이 엉뚱하게 계산된다 — 실제 카메라에는 없는 현상)
        first_pose = lm_seq[~np.isnan(lm_seq[:, 0, 0])][0]
        for f in idle_frames(first_pose, int(args.fps * 0.7), rng):
            feed(f)
        if args.leave_frame > 0 and s.label != "no_gesture":
            lm_seq = lm_seq.copy()
            cut = int(len(lm_seq) * (1 - args.leave_frame))
            lm_seq[cut:] = np.nan          # 뒤쪽은 손이 화면 밖 → 미검출
        if args.hole > 0 and s.label != "no_gesture":
            lm_seq = lm_seq.copy()
            det = ~np.isnan(lm_seq[:, 0, 0])
            if det.sum() > args.hole + 4:
                w = lm_seq[:, 0, :2]
                speed = np.nan_to_num(np.linalg.norm(np.diff(w, axis=0), axis=1))
                c = int(np.argmax(speed)) + 1              # 가장 빠른 프레임
                a = max(1, c - args.hole // 2)
                lm_seq[a:a + args.hole] = np.nan           # 그 주변을 미검출로
        hits = []
        for i in range(len(lm_seq)):
            lm = lm_seq[i]
            lm = None if np.isnan(lm[0, 0]) else lm
            out = feed(lm)
            if out is not None:
                hits.append(out)
        last = lm_seq[~np.isnan(lm_seq[:, 0, 0])][-1]
        if args.fist_lower and s.label == "make_fist":
            # 주먹 쥔 뒤 0.3초 멈춤 → 0.5초 동안 아래로(+y) 내리며 살짝 옆으로(+x) → 정지
            for f in idle_frames(last, int(args.fps * 0.3), rng):
                out = feed(f)
                if out is not None:
                    hits.append(out)
            palm = float(np.linalg.norm(last[9, :2] - last[0, :2]))
            cur = last.copy()
            for _ in range(int(args.fps * 0.5)):
                cur = cur.copy()
                cur[:, 1] += palm * 0.25          # 프레임당 손바닥 1/4 만큼 아래로 → 0.5초에 약 4 손바닥
                cur[:, 0] += palm * 0.04          # 약간 옆으로
                out = feed(cur + rng.normal(0, 0.0003, cur.shape).astype(np.float32))
                if out is not None:
                    hits.append(out)
            last = cur
        # 클립 뒤 정지 1.2초 (여기서 구간이 마감됨). 손이 화면 밖으로 나간 흉내면 잠시 미검출 유지 후 복귀
        if args.leave_frame > 0 and s.label != "no_gesture":
            for _ in range(int(args.fps * 0.5)):
                out = feed(None)
                if out is not None:
                    hits.append(out)
        for f in idle_frames(last, int(args.fps * 1.2), rng):
            out = feed(f)
            if out is not None:
                hits.append(out)
        if len(hits) > 1:
            multi += 1
        if hits:
            detected[s.label] += 1
        clip_correct = False
        for h in hits:
            feats = preprocess.sample_to_features(h.landmarks, h.timestamps_ms,
                                                  min_ratio=args.min_det, max_gap=args.max_gap)
            if feats is None:
                skipped[s.label] += 1
                continue
            label, conf, _ = clf.predict(feats)
            if label == s.label and s.label != "no_gesture" and not clip_correct:
                correct[s.label] += 1; clip_correct = True
            ok, reason = (True, "") if args.no_guard else sanity.check(label, h.landmarks)
            if label != "no_gesture" and not ok:
                guarded[(s.label, label)] += 1
                if label == s.label:
                    guard_false.append((s.path.name, label, reason, h.duration_sec))
                msg = "GUARD"
            else:
                # now = 구간이 끝난 시각 (실시간에서 판정이 일어나는 시점). 하네스 시계 t 는 이미 뒤 정지까지 지나 있으므로 쓰지 않음
                msg = mapper.handle(label, conf, now=h.timestamps_ms[-1] / 1000.0, start=h.timestamps_ms[0] / 1000.0)
            if msg.startswith("[DRY]"):
                fired[(s.label, label)] += 1
                if label != s.label:
                    misfire_detail.append((s.path.name, s.label, label, conf, reason, h.duration_sec,
                                           getattr(seg, "last_early", False)))
            elif "ignored" in msg:
                ignored[msg.split("->")[-1].strip()] += 1
            if label != s.label:
                misses.append((s.path.name, s.label, label, conf, h.duration_sec))
        # (다음 클립 앞의 0.7초 정지와 합쳐 클립 간격 약 1.9초 = 사람이 연속으로 명령하는 현실적인 간격)

    print("\n=== 구간 감지율 (클립이 구간으로 잘렸나) ===")
    for lab in config.LABELS:
        print(f"  {lab:12s} {detected[lab]:3d}/{total[lab]:3d}")
    print("  (no_gesture 는 정지 클립이면 안 잘리는 게 정상. 움직이는 no_gesture 는 잘려도 판정이 no_gesture 면 OK)")
    print(f"  한 클립에서 구간이 2개 이상 잡힌 경우: {multi}")
    print(f"  품질 기준(det>={args.min_det}, gap<={args.max_gap})으로 버려진 구간: {dict(skipped) or 0}")
    print(f"  감지기가 버린 구간(DROP): {dict(drops) or 0}")

    print("\n=== 판정 정확도 (감지된 명령 클립 기준) ===")
    for lab in ("swipe_left", "make_fist"):
        d = detected[lab]
        print(f"  {lab:12s} {correct[lab]:3d}/{d:3d}" + (f"  ({correct[lab]/d:.0%})" if d else ""))

    print("\n=== 기능 실행 (연습 모드) : 실제라벨 -> 실행된 기능 ===")
    for (true, pred), n in sorted(fired.items()):
        mark = "" if true == pred else "   <-- 오작동" if true == "no_gesture" or pred != true else ""
        print(f"  {true:12s} -> {pred:12s} x{n}{mark}")
    bad = sum(n for (true, pred), n in fired.items() if true != pred)
    print(f"  오작동(실제 라벨과 다른 기능 실행) 합계: {bad}")
    for name, t, p, c, reason, dur, early in misfire_detail:
        print(f"    {name}: {t} -> {p} ({c:.2f}, {dur:.2f}s{', EARLY' if early else ''}) | guard: {reason}")
    if guarded:
        print("  상식 검사(sanity)로 막은 실행: " + ", ".join(f"{t}->{p} x{n}" for (t, p), n in sorted(guarded.items())))
    if ignored:
        print("  무시(확신도/cooldown): " + ", ".join(f"{k} x{n}" for k, n in ignored.items()))
    if guard_false:
        print(f"\n=== 상식 검사가 '정답 판정'을 막은 경우 ({len(guard_false)}개) — 기준이 너무 엄격한지 확인 ===")
        for name, lab, reason, dur in guard_false[:15]:
            print(f"  {name}: {lab} | {reason} ({dur:.2f}s)")

    if misses:
        print(f"\n=== 오판 목록 ({len(misses)}개, 최대 15개 표시) ===")
        for name, true, pred, conf, dur in misses[:15]:
            print(f"  {name}: {true} -> {pred} ({conf:.2f}, {dur:.2f}s)")


if __name__ == "__main__":
    main()
