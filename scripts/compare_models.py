"""GRU (필요하면 LSTM 도) vs baseline(룰) 공정 비교.

같은 데이터 분할로 시드별로 여러 번 학습해 평균±표준편차를 표로 뽑는다.
모델 선택은 val 기준으로 하고, test 는 보고용으로만 본다.

  uv run python scripts/compare_models.py                                  # GRU 시드 3개
  uv run python scripts/compare_models.py --seeds 5 --test-persons p003 --val-persons p002
  uv run python scripts/compare_models.py --archs gru lstm                 # 팀원 LSTM 과 나란히 비교

결과: reports/compare_models.csv (한 줄 = 모델×시드), reports/compare_models_summary.md (발표용 표)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from gesture import config, dataset, training  # noqa: E402
from gesture.model import ARCHS, DEFAULT_ARCH  # noqa: E402

COLS = ["arch", "seed", "params", "epochs", "train_sec", "infer_ms",
        "val_acc", "val_macro_f1", "val_ng_recall",
        "test_acc", "test_macro_f1", "test_ng_recall", "test_ng_precision"]


def run_one(arch, seed, Xtr, ytr, Xva, yva, Xte, yte, epochs, batch) -> dict:
    model, hist = training.fit(arch, Xtr, ytr, Xva, yva, seed=seed, epochs=epochs, batch=batch)
    row = dict(arch=arch, seed=seed, params=model.count_params(), epochs=len(hist.history["loss"]),
               train_sec=hist.train_sec, infer_ms=training.infer_ms_per_sample(model, Xva))
    row.update({f"val_{k}": v for k, v in training.metrics(yva, training.predict(model, Xva)).items()})
    if len(Xte):
        row.update({f"test_{k}": v for k, v in training.metrics(yte, training.predict(model, Xte)).items()})
    return row


def fmt(vals, pct=True):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "-"
    m, s = np.mean(vals), np.std(vals)
    return f"{m*100:5.1f} ± {s*100:4.1f}" if pct else f"{m:6.1f} ± {s:4.1f}"


def summarize(rows: list[dict], base_val: dict, base_test: dict) -> str:
    archs = sorted({r["arch"] for r in rows}, key=ARCHS.index)
    lines = ["| 모델 | 파라미터 | val acc | val F1 | val no_gesture 재현율 | test acc | test F1 | "
             "test no_gesture 재현율 | 추론 ms | 학습 초 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in archs:
        rs = [r for r in rows if r["arch"] == a]
        g = lambda k: [r.get(k) for r in rs]
        lines.append(f"| {a.upper()} (n={len(rs)}) | {rs[0]['params']:,} | {fmt(g('val_acc'))} | "
                     f"{fmt(g('val_macro_f1'))} | {fmt(g('val_ng_recall'))} | {fmt(g('test_acc'))} | "
                     f"{fmt(g('test_macro_f1'))} | {fmt(g('test_ng_recall'))} | "
                     f"{fmt(g('infer_ms'), pct=False)} | {fmt(g('train_sec'), pct=False)} |")
    b = lambda d, k: f"{d[k]*100:5.1f}" if d else "-"
    lines.append(f"| baseline(룰) | 0 | {b(base_val,'acc')} | {b(base_val,'macro_f1')} | {b(base_val,'ng_recall')} | "
                 f"{b(base_test,'acc')} | {b(base_test,'macro_f1')} | {b(base_test,'ng_recall')} | ~0 | 0 |")

    means = {a: np.mean([r["val_acc"] for r in rows if r["arch"] == a]) for a in archs}
    best = max(means, key=means.get)
    verdict = f"\n**val 정확도 기준: {best.upper()} 평균 {means[best]*100:.1f}%**"
    if base_val:
        gap = (means[best] - base_val["acc"]) * 100
        verdict += f" (baseline 대비 {gap:+.1f}%p)"
    others = [a for a in archs if a != best]
    if others:
        gap = (means[best] - means[others[0]]) * 100
        stds = [np.std([r["val_acc"] for r in rows if r["arch"] == a]) * 100 for a in archs]
        if gap < max(stds + [1.0]):
            lighter = min(archs, key=lambda a: [r["params"] for r in rows if r["arch"] == a][0])
            verdict += (f". {others[0].upper()} 과 차이({gap:.1f}%p)가 시드 편차 안이라 사실상 동률 → "
                        f"더 가벼운 {lighter.upper()} 권장")
        else:
            verdict += f". {others[0].upper()} 대비 +{gap:.1f}%p (시드 편차보다 큼)"
    return "\n".join(lines) + "\n" + verdict + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", choices=ARCHS, default=[DEFAULT_ARCH])
    ap.add_argument("--seeds", type=int, default=3, help="시드 개수 (0,1,2,...)")
    ap.add_argument("--seed-list", nargs="*", type=int, default=None)
    ap.add_argument("--dataset", nargs="+", default=[str(config.DATASET_DIR)])
    ap.add_argument("--test-persons", nargs="*", default=None)
    ap.add_argument("--val-persons", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--augment", type=int, default=3)
    ap.add_argument("--out-dir", default=str(config.REPORTS_DIR))
    args = ap.parse_args()
    seeds = args.seed_list if args.seed_list else list(range(args.seeds))

    samples = []
    for r in args.dataset:
        samples += dataset.load_dataset(Path(r))
    if not samples:
        sys.exit("dataset/ 에 npz 가 없습니다.")
    print(dataset.summarize(samples))
    if len({s.label for s in samples}) < 2:
        sys.exit("라벨이 하나뿐이라 학습할 수 없습니다.")

    train, val, test = dataset.split_by_person(samples, args.val_persons, args.test_persons)
    print(f"\nsplit: train={len(train)} val={len(val)} test={len(test)}  "
          f"(val: {sorted({s.person for s in val})}, test: {sorted({s.person for s in test})})")
    Xtr, ytr, _ = dataset.build_arrays(train, augment_times=args.augment)
    Xva, yva, _ = dataset.build_arrays(val)
    Xte, yte, _ = dataset.build_arrays(test)
    print(f"arrays: Xtr={Xtr.shape} Xva={Xva.shape} Xte={Xte.shape}")
    if len(Xtr) == 0 or len(Xva) == 0:
        sys.exit("학습/검증 데이터가 부족합니다.")

    total = len(args.archs) * len(seeds)
    rows, k = [], 0
    for arch in args.archs:
        for seed in seeds:
            k += 1
            print(f"\n[{k}/{total}] {arch.upper()} seed={seed} 학습 중...", flush=True)
            row = run_one(arch, seed, Xtr, ytr, Xva, yva, Xte, yte, args.epochs, args.batch)
            rows.append(row)
            print(f"  → {row['epochs']} epoch, {row['train_sec']:.0f}초, val acc {row['val_acc']*100:.1f}%"
                  + (f", test acc {row['test_acc']*100:.1f}%" if "test_acc" in row else ""))

    base_val = training.baseline_metrics(Xva, yva)
    base_test = training.baseline_metrics(Xte, yte)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "compare_models.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})

    summary = summarize(rows, base_val, base_test)
    header = (f"# 모델 비교\n\n샘플 {len(samples)}개, 참가자 {sorted({s.person for s in samples})}, "
              f"val={sorted({s.person for s in val})}, test={sorted({s.person for s in test})}, "
              f"시드 {seeds}, 증강 x{args.augment}\n\n")
    md_path = out_dir / "compare_models_summary.md"
    md_path.write_text(header + summary, encoding="utf-8")
    print("\n" + header + summary)
    print(f"저장: {csv_path}\n      {md_path}")


if __name__ == "__main__":
    main()
