"""라벨별 상식검사(sanity) 측정값 분포. 새 제스처의 기준값을 정하거나 기존 기준이 진짜 동작을 막는지 볼 때 쓴다.

  uv run python scripts/label_stats.py                      # dataset/ 전체
  uv run python scripts/label_stats.py --labels finger_snap no_gesture
  uv run python scripts/label_stats.py --dataset C:\\...\\dataset_p001 --persons p001

출력: 라벨별로 dx, dy(손목 순이동), ext_open→ext_min(손가락 펴짐 최대→최소, 주먹 기준),
      pinch_start→pinch_end(엄지끝-중지끝 거리, 스냅 기준) 의 5/50/95 백분위.
      마지막에 현재 sanity 기준으로 각 라벨 클립이 '자기 라벨 검사'를 몇 개 통과하는지 센다.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from gesture import config, dataset, sanity  # noqa: E402

FIELDS = ["dx", "dy", "ext_open", "ext_min", "pinch_min", "pinch_end", "idx_end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASET_DIR))
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--persons", nargs="*", default=None)
    args = ap.parse_args()

    samples = dataset.load_dataset(Path(args.dataset))
    if args.persons:
        samples = [s for s in samples if s.person in set(args.persons)]
    labels = args.labels or config.LABELS
    by = defaultdict(list)
    for s in samples:
        if s.label in labels:
            b = sanity._basic(s.landmarks)
            if b:
                by[s.label].append(b)

    print(f"{'label':12s} {'n':>4s} " + " ".join(f"{f:>26s}" for f in FIELDS))
    print(" " * 18 + " ".join(f"{'p5 / p50 / p95':>26s}" for _ in FIELDS))
    for lab in labels:
        rows = by.get(lab, [])
        if not rows:
            print(f"{lab:12s} {0:4d}  (클립 없음)")
            continue
        cells = []
        for f in FIELDS:
            v = np.array([r[f] for r in rows])
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            cells.append(f"{p5:7.2f} /{p50:7.2f} /{p95:7.2f}")
        print(f"{lab:12s} {len(rows):4d} " + " ".join(f"{c:>26s}" for c in cells))

    print("\n=== 현재 sanity 기준으로 '자기 라벨' 검사 통과율 (명령 라벨만) ===")
    for lab in labels:
        if lab == "no_gesture":
            continue
        clips = [s for s in samples if s.label == lab]
        if not clips:
            continue
        ok = sum(sanity.check(lab, s.landmarks)[0] for s in clips)
        fails = defaultdict(int)
        for s in clips:
            passed, why = sanity.check(lab, s.landmarks)
            if not passed:
                fails[why.split("(")[0].strip()] += 1
        print(f"  {lab:12s} {ok:4d}/{len(clips):4d}" + (f"   막힌 이유: {dict(fails)}" if fails else ""))
    print("\n=== no_gesture 클립이 각 명령 검사를 '잘못 통과'하는 비율 (낮을수록 좋음) ===")
    ng = [s for s in samples if s.label == "no_gesture"]
    for lab in labels:
        if lab == "no_gesture" or not ng:
            continue
        leak = sum(sanity.check(lab, s.landmarks)[0] for s in ng)
        print(f"  {lab:12s} {leak:4d}/{len(ng):4d}")


if __name__ == "__main__":
    main()
