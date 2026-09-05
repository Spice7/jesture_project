"""dataset/ 현황: 참가자×라벨 개수, 형식 오류, 품질 기준 미달 샘플.

  uv run python scripts/dataset_summary.py
  uv run python scripts/dataset_summary.py --dataset dataset other/dataset
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gesture import config, dataset, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=[str(config.DATASET_DIR)])
    args = ap.parse_args()
    samples = []
    for r in args.dataset:
        samples += dataset.load_dataset(Path(r))
    if not samples:
        print("샘플 없음")
        return
    print(dataset.summarize(samples))
    bad = [s for s in samples if preprocess.interpolate_missing(s.landmarks) is None]
    print(f"\n총 {len(samples)}개, 품질 미달 {len(bad)}개 "
          f"(검출비율<{config.MIN_DETECTION_RATIO} 또는 연속결측>{config.MAX_GAP_FRAMES})")
    for s in bad[:20]:
        m = preprocess.detection_mask(s.landmarks)
        print(f"  {s.path.name}: det={m.mean():.0%} gap={preprocess.longest_gap(m)} frames={len(m)}")
    import numpy as np
    durs = [s.timestamps_ms[-1] / 1000 for s in samples]
    frames = [len(s.landmarks) for s in samples]
    print(f"길이(초): 평균 {np.mean(durs):.2f}, 최소 {np.min(durs):.2f}, 최대 {np.max(durs):.2f}   "
          f"프레임: 평균 {np.mean(frames):.0f}, 최소 {np.min(frames)}, 최대 {np.max(frames)}")
    hands = {}
    for s in samples:
        hands[s.meta.get("handedness", "?")] = hands.get(s.meta.get("handedness", "?"), 0) + 1
    print(f"손: {hands}")


if __name__ == "__main__":
    main()
