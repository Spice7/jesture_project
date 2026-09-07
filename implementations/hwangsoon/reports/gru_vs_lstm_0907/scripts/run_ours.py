"""우리 GRU(+우리 구현 LSTM 대조군)를 팀원과 같은 사람 단위 분할로 채점한다.

분할(팀원 four_class_data_002 와 동일): train p001 p002 p003 p005 (+user00) / val p004 / test p006
두 가지 학습 레시피:
  plain  : 증강 없음 (팀원 LSTM 학습 조건과 같음)
  recipe : 시연 모델 레시피 (증강 x3, 되감기 both 0.5, 내리기 0.5, 짧은 스와이프 0.5)
시드 0,1,2. 지표: accuracy, macro F1, 클래스별 recall, 혼동행렬, 파라미터, CPU 추론 ms(배치 1), 학습 초, epoch, 학습 곡선.
룰 baseline 도 같은 val/test 로.
"""
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Users\Admin\Desktop\jesture_project")
sys.path.insert(0, str(ROOT))
from gesture import baseline, config, dataset, training  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "ours_results.json"
SEEDS = [0, 1, 2]
ARCHS = ["gru", "lstm"]
LABELS = config.LABELS


def full_metrics(y, pred):
    n = len(LABELS)
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    return dict(accuracy=float(tp.sum() / len(y)), macro_f1=float(f1.mean()),
                macro_precision=float(prec.mean()), macro_recall=float(rec.mean()),
                per_class={l: dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]),
                                   support=int(cm[i].sum())) for i, l in enumerate(LABELS)},
                confusion=cm.tolist(), n=int(len(y)))


@torch.no_grad()
def infer_ms_cpu(model, X, n=50):
    m = copy.deepcopy(model).cpu().eval()
    xs = X[:n]
    for x in xs[:3]:
        m.predict_proba(x[None])
    t0 = time.perf_counter()
    for x in xs:
        m.predict_proba(x[None])
    return (time.perf_counter() - t0) / len(xs) * 1000


def main():
    torch.set_num_threads(1)
    samples = dataset.load_dataset(config.DATASET_DIR)
    print(dataset.summarize(samples))
    train, val, test = dataset.split_by_person(samples, ["p004"], ["p006"])
    print(f"split clips: train={len(train)} val={len(val)} test={len(test)} train persons={sorted({s.person for s in train})}")
    Xva, yva, dva = dataset.build_arrays(val)
    Xte, yte, dte = dataset.build_arrays(test)
    print(f"val {Xva.shape} dropped {len(dva)} / test {Xte.shape} dropped {len(dte)}")
    results = dict(split=dict(train_persons=sorted({s.person for s in train}), val=["p004"], test=["p006"],
                              n_train_clips=len(train), n_val=int(len(yva)), n_test=int(len(yte)),
                              val_dist={l: int((yva == i).sum()) for i, l in enumerate(LABELS)},
                              test_dist={l: int((yte == i).sum()) for i, l in enumerate(LABELS)}),
                   baseline=dict(val=full_metrics(yva, baseline.predict_features(Xva)),
                                 test=full_metrics(yte, baseline.predict_features(Xte))),
                   runs=[])
    print("baseline val", results["baseline"]["val"]["accuracy"], "test", results["baseline"]["test"]["accuracy"])

    recipes = {
        "plain": dict(augment=0, reverse=None, lower=0.0, short=0.0),
        "recipe": dict(augment=3, reverse="both", lower=0.5, short=0.5),
    }
    for rname, rc in recipes.items():
        tr = list(train)
        if rc["reverse"]:
            tr += dataset.reversed_negatives(train, dataset.REVERSE_SETS[rc["reverse"]], 0.5)
        if rc["lower"] > 0:
            tr += dataset.lowering_negatives(tr, fraction=rc["lower"])
        if rc["short"] > 0:
            tr += dataset.shortened_swipes(tr, fraction=rc["short"])
        Xtr, ytr, _ = dataset.build_arrays(tr, augment_times=rc["augment"])
        print(f"\n## {rname}: Xtr={Xtr.shape}")
        for arch in ARCHS:
            for seed in SEEDS:
                model, hist = training.fit(arch, Xtr, ytr, Xva, yva, seed=seed)
                pv = training.predict(model, Xva)
                pt = training.predict(model, Xte)
                row = dict(recipe=rname, arch=arch, seed=seed, params=model.count_params(),
                           n_train=int(len(ytr)), epochs=len(hist.history["loss"]), train_sec=hist.train_sec,
                           infer_ms_cpu=infer_ms_cpu(model, Xva),
                           val=full_metrics(yva, pv), test=full_metrics(yte, pt),
                           history=hist.history if seed == 0 else None)
                results["runs"].append(row)
                print(f"{rname:6s} {arch:4s} seed{seed} ep={row['epochs']:2d} {row['train_sec']:5.1f}s "
                      f"val acc={row['val']['accuracy']*100:5.1f} F1={row['val']['macro_f1']*100:5.1f} | "
                      f"test acc={row['test']['accuracy']*100:5.1f} F1={row['test']['macro_f1']*100:5.1f} "
                      f"infer={row['infer_ms_cpu']:.2f}ms", flush=True)
                OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
