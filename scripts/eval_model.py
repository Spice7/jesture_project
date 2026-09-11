"""저장된 모델을 임의 데이터 폴더·참가자로 채점한다 (학습 없음). 룰 기반 baseline 도 같은 시험지로.

  uv run python scripts/eval_model.py --dataset "C:/.../dataset_p005" --persons p005
  uv run python scripts/eval_model.py --model models/gru_gesture.pt --dataset data/dynamic --persons p002 p003

용도: 최종 모델(학습에 넣지 않은 사람으로) 정직한 "처음 보는 사람" 점수. 새 팀원 데이터가 오면 학습에 넣기 *전에* 먼저 채점.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from gesture import baseline, config, dataset, training  # noqa: E402
from gesture.model import GestureClassifier, model_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="기본 models/gru_gesture.pt")
    ap.add_argument("--dataset", nargs="+", default=[str(config.DATASET_DIR)])
    ap.add_argument("--persons", nargs="*", default=None, help="이 참가자만 (기본 전체)")
    ap.add_argument("--list-errors", type=int, default=20, help="오답 파일 최대 표시 수")
    args = ap.parse_args()

    samples = []
    for r in args.dataset:
        samples += dataset.load_dataset(Path(r))
    if args.persons:
        samples = [s for s in samples if s.person in args.persons]
    if not samples:
        sys.exit("샘플이 없습니다.")
    print(dataset.summarize(samples))

    X, y, dropped = dataset.build_arrays(samples)
    kept = [s for s in samples if s not in dropped]
    if dropped:
        print(f"품질 기준 미달로 제외 {len(dropped)}개")
    mf = Path(args.model) if args.model else model_path()
    clf = GestureClassifier(model_file=mf)
    print(f"모델: {mf}  (학습 참가자: 메타 참고)")

    probs = clf.predict_proba(X)
    pred = probs.argmax(1)
    training.report(f"{clf.arch.upper()} / {'+'.join(args.persons or ['all'])}", y, pred)
    training.report(f"baseline(rule) / {'+'.join(args.persons or ['all'])}", y, baseline.predict_features(X))

    conf = probs.max(1)
    print(f"\n확신도: 중간 {np.median(conf):.2f}, 0.8 미만 {int((conf < 0.8).sum())}개 / {len(conf)}")
    wrong = [(s, config.LABELS[t], config.LABELS[p], c) for s, t, p, c in zip(kept, y, pred, conf) if t != p]
    if wrong:
        print(f"\n오답 {len(wrong)}개 (최대 {args.list_errors}개 표시):")
        by = Counter((t, p) for _, t, p, _ in wrong)
        for (t, p), n in by.most_common():
            print(f"  {t} -> {p}: {n}")
        for s, t, p, c in wrong[: args.list_errors]:
            print(f"    {s.path.name}: {t} -> {p} ({c:.2f})")


if __name__ == "__main__":
    main()
