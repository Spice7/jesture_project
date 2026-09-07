"""CPU 배치1 추론 시간 재측정 (가중치 무관): 우리 GRU/LSTM(2층64+FC) vs 팀원 LSTM 1층. 200회 중앙값, 1스레드."""
import sys, time, json, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
JIN = ROOT / os.environ.get("JESTURE_JIN_DIR", "external/jin")
if not (JIN / "training" / "models.py").is_file():
    raise SystemExit(f"비교용 외부 JIN 소스가 필요합니다: {JIN / 'training' / 'models.py'} (주 프로그램에는 불필요)")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JIN / "training"))
from gesture.model import GestureRNN
from models import LSTMClassifier
torch.set_num_threads(1)
def bench(m, x, n=200):
    m.eval()
    with torch.no_grad():
        for _ in range(20): m(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); m(x); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.mean(ts))
out = {}
for name, m, x in (("gru_ours", GestureRNN("gru"), torch.randn(1, 30, 63)), ("lstm_ours", GestureRNN("lstm"), torch.randn(1, 30, 63)),
                   ("lstm_jin_1layer", LSTMClassifier(66, 64, 1, 4, 0.2), torch.randn(1, 32, 66)),
                   ("lstm_jin_2layer", LSTMClassifier(66, 64, 2, 4, 0.2), torch.randn(1, 32, 66))):
    med, mean = bench(m.cpu(), x)
    out[name] = dict(params=sum(p.numel() for p in m.parameters()), median_ms=med, mean_ms=mean)
    print(f"{name:16s} params={out[name]['params']:>7,d} median={med:.3f}ms mean={mean:.3f}ms")
(Path(__file__).resolve().parent.parent / "raw" / "bench_infer.json").write_text(json.dumps(out, indent=1))
