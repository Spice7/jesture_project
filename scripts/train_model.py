"""dataset/ 의 수집 데이터로 GRU(기본) 또는 LSTM 학습 + baseline 비교 + 평가 리포트.
저장: models/<arch>_gesture.pt + .json (PyTorch)

  uv run python scripts/train_model.py                                   # GRU, 자동 분할
  uv run python scripts/train_model.py --test-persons p003 --val-persons p002
  uv run python scripts/train_model.py --arch lstm                       # 비교용 LSTM
  uv run python scripts/train_model.py --dataset dataset other/dataset   # 여러 폴더 합치기

여러 시드로 공정 비교하려면 scripts/compare_models.py 를 쓴다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gesture import baseline, config, dataset, training  # noqa: E402
from gesture.model import ARCHS, DEFAULT_ARCH, model_path, save_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default=DEFAULT_ARCH)
    ap.add_argument("--dataset", nargs="+", default=[str(config.DATASET_DIR)])
    ap.add_argument("--test-persons", nargs="*", default=None)
    ap.add_argument("--val-persons", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--augment", type=int, default=3, help="샘플당 증강 복제 수")
    ap.add_argument("--reverse-neg", choices=list(dataset.REVERSE_SETS), default="none",
                    help="명령 클립을 되감아 no_gesture 학습 샘플로 추가: fist(펴기) | both(펴기+스와이프 복귀)")
    ap.add_argument("--reverse-frac", type=float, default=0.5, help="되감을 클립 비율")
    ap.add_argument("--lower-neg", type=float, default=0.0,
                    help="주먹·스냅 클립의 끝 자세로 '손 내리기'를 합성해 no_gesture 로 추가하는 비율 (예 0.5). 09-07 내리기 오인 대책")
    ap.add_argument("--init-weights", default=None, help="기존 가중치(.pt)로 초기화")
    ap.add_argument("--out", default=None, help="저장 경로 (기본 models/<arch>_gesture.pt)")
    args = ap.parse_args()

    samples = []
    for r in args.dataset:
        samples += dataset.load_dataset(Path(r))
    if not samples:
        sys.exit("dataset/ 에 npz 가 없습니다. programs/collect_gesture.py 로 수집하거나 JIN 을 merge 하세요.")
    print(dataset.summarize(samples))
    labels_present = {s.label for s in samples}
    if len(labels_present) < 2:
        sys.exit(f"라벨이 {sorted(labels_present)} 하나뿐이라 학습할 수 없습니다. 다른 라벨 데이터가 필요합니다.")

    train, val, test = dataset.split_by_person(samples, args.val_persons, args.test_persons, seed=args.seed)
    print(f"\nsplit: train={len(train)} val={len(val)} test={len(test)}  "
          f"(val: {sorted({s.person for s in val})}, test: {sorted({s.person for s in test})})")

    if args.reverse_neg != "none":
        rev = dataset.reversed_negatives(train, dataset.REVERSE_SETS[args.reverse_neg], args.reverse_frac, seed=args.seed)
        train = train + rev
        print(f"되감기 no_gesture 추가: {len(rev)}개 ({args.reverse_neg}, 비율 {args.reverse_frac}) → train={len(train)}")
    if args.lower_neg > 0:
        low = dataset.lowering_negatives(train, fraction=args.lower_neg, seed=args.seed)
        train = train + low
        print(f"내리기 no_gesture 추가: {len(low)}개 (비율 {args.lower_neg}) → train={len(train)}")
    Xtr, ytr, drop_tr = dataset.build_arrays(train, augment_times=args.augment, seed=args.seed)
    Xva, yva, drop_va = dataset.build_arrays(val)
    Xte, yte, drop_te = dataset.build_arrays(test)
    dropped = drop_tr + drop_va + drop_te
    if dropped:
        print(f"품질 기준 미달로 제외된 샘플 {len(dropped)}개 (예: {dropped[0].name})")
    print(f"arrays: Xtr={Xtr.shape} Xva={Xva.shape} Xte={Xte.shape}")
    if len(Xtr) == 0 or len(Xva) == 0:
        sys.exit("학습/검증 데이터가 부족합니다.")

    model, hist = training.fit(args.arch, Xtr, ytr, Xva, yva, seed=args.seed, epochs=args.epochs,
                               batch=args.batch, init_weights=args.init_weights, verbose=2)
    model.summary(line_length=80)
    print(f"학습 {len(hist.history['loss'])} epoch, {hist.train_sec:.0f}초")

    for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        if len(X) == 0:
            continue
        training.report(f"{args.arch.upper()} / {name}", y, training.predict(model, X))
        training.report(f"baseline(rule) / {name}", y, baseline.predict_features(X))

    out = Path(args.out) if args.out else model_path(args.arch)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    save_meta(out.with_suffix(".json"), arch=args.arch, seed=args.seed, n_train=int(len(Xtr)), reverse_neg=args.reverse_neg, lower_neg=args.lower_neg,
              persons=sorted({s.person for s in samples}),
              val_persons=sorted({s.person for s in val}), test_persons=sorted({s.person for s in test}))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
