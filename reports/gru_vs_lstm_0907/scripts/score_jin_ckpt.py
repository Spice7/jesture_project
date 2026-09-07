"""팀원(JIN) LSTM 체크포인트를 팀원 전처리 배열(X_val/X_test .npy)로 채점한다.
팀원 evaluate.py 와 같은 지표(accuracy, macro F1, 클래스별 P/R/F1, 혼동행렬) + CPU 추론 ms + 파라미터 수.
evaluate.py 는 입력 메타데이터 SHA-256 검사 때문에 git show 사본에서 실패하여 같은 계산을 직접 한다.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR / "jin" / "training"))
from models import LSTMClassifier  # noqa: E402

LABELS = ["swipe_left", "make_fist", "no_gesture", "finger_snap"]


def metrics(y, pred):
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


def load(run_dir):
    ck = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    m = LSTMClassifier(**ck["model_config"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(m, X):
    out = []
    for i in range(0, len(X), 256):
        out.append(m(torch.from_numpy(X[i:i + 256])).argmax(1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def infer_ms(m, X, n=50):
    xs = torch.from_numpy(X[:n])
    for x in xs[:3]:
        m(x[None])
    t0 = time.perf_counter()
    for x in xs:
        m(x[None])
    return (time.perf_counter() - t0) / len(xs) * 1000


def score(run_name, data_name, splits=("val", "test")):
    run_dir, data_dir = SCR / "jin" / "artifacts" / run_name, SCR / "jin" / "artifacts" / data_name
    if not (run_dir / "best_model.pt").exists():
        run_dir = SCR / "jin" / run_name
    if not data_dir.exists():
        data_dir = SCR / "jin" / data_name
    m, ck = load(run_dir)
    res = dict(run=run_name, data=data_name, model_config=ck["model_config"], best_epoch=ck.get("best_epoch"),
               params=sum(p.numel() for p in m.parameters()))
    for sp in splits:
        X = np.load(data_dir / f"X_{sp}.npy", allow_pickle=False)
        y = np.load(data_dir / f"y_{sp}.npy", allow_pickle=False)
        res[sp] = metrics(y, predict(m, X))
        res[sp]["infer_ms_cpu"] = infer_ms(m, X)
    return res


if __name__ == "__main__":
    torch.set_num_threads(1)
    jobs = [a.split(":") for a in sys.argv[1:]] or [
        ("final_1layers_001", "four_class_data_002"), ("final_2layers_001", "four_class_data_002"),
        ("final_3layers_001", "four_class_data_002"), ("final_4layers_001", "four_class_data_002"),
        ("four_class_lstm_1layers_001", "four_class_data_001"),
    ]
    out = [score(r, d) for r, d in jobs]
    for r in out:
        for sp in ("val", "test"):
            if sp in r:
                pc = r[sp]["per_class"]
                print(f"{r['run']:32s} {sp:4s} n={r[sp]['n']:4d} acc={r[sp]['accuracy']*100:5.1f} "
                      f"F1={r[sp]['macro_f1']*100:5.1f} recall=" +
                      " ".join(f"{l[:5]}={pc[l]['recall']*100:5.1f}" for l in LABELS) +
                      f" infer={r[sp]['infer_ms_cpu']:.2f}ms params={r['params']}")
    (SCR / "jin_ckpt_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
