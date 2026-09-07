"""ours_results.json + jin_ckpt_scores.json + jin_on_ours_scores.json → PNG 5장 + numbers.json + 표(markdown 조각)."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SCR = Path(__file__).resolve().parent
RAW = SCR.parent / "raw"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else SCR / "report_out"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["swipe_left", "make_fist", "no_gesture", "finger_snap"]
KO = {"swipe_left": "스와이프", "make_fist": "주먹", "no_gesture": "no_gesture", "finger_snap": "스냅"}
C_GRU, C_LSTM_OURS, C_LSTM_JIN, C_RULE = "#2A6F97", "#89B4D6", "#E07A2F", "#7A7A7A"

ours = json.loads((RAW / "ours_results.json").read_text(encoding="utf-8"))
jin_own = {r["run"]: r for r in json.loads((RAW / "jin_ckpt_scores.json").read_text(encoding="utf-8"))}
jin_ours = {r["run"]: r for r in json.loads((RAW / "jin_on_ours_scores.json").read_text(encoding="utf-8"))}


def agg(rows, split, key):
    v = np.array([r[split][key] for r in rows]) * 100
    return float(v.mean()), float(v.std())


def agg_rec(rows, split, label):
    v = np.array([r[split]["per_class"][label]["recall"] for r in rows]) * 100
    return float(v.mean()), float(v.std())


def cm_sum(rows, split):
    return np.sum([np.array(r[split]["confusion"]) for r in rows], axis=0)


groups = {}
for rec in ("plain", "recipe"):
    for arch in ("gru", "lstm"):
        rows = [r for r in ours["runs"] if r["recipe"] == rec and r["arch"] == arch]
        if rows:
            groups[f"{arch}_{rec}"] = rows
jin_seeds = [jin_ours[f"lstm_ours_1l_seed{s}"] for s in (0, 1, 2) if f"lstm_ours_1l_seed{s}" in jin_ours]
groups["jin_lstm1_ours_data"] = jin_seeds

NAMES = {"gru_recipe": "GRU (시연 레시피)", "gru_plain": "GRU (증강 없음)",
         "lstm_recipe": "LSTM (시연 레시피)", "lstm_plain": "LSTM (증강 없음)",
         "jin_lstm1_ours_data": "LSTM 1층 (원본 코드, 같은 데이터)"}
SHORT = {"gru_recipe": "GRU\n시연 레시피", "gru_plain": "GRU\n증강 없음",
         "lstm_recipe": "LSTM\n시연 레시피", "lstm_plain": "LSTM\n증강 없음",
         "jin_lstm1_ours_data": "LSTM 1층\n원본 코드\n같은 데이터"}
COLORS = {"gru_recipe": C_GRU, "gru_plain": "#5B93B8", "lstm_recipe": C_LSTM_OURS, "lstm_plain": "#B9D3E6",
          "jin_lstm1_ours_data": C_LSTM_JIN}

numbers = dict(split=ours["split"], baseline=ours["baseline"], groups={}, jin_own={}, jin_on_ours_per_seed={})
for g, rows in groups.items():
    d = dict(name=NAMES[g], n_seeds=len(rows), params=rows[0]["params"],
             infer_ms_cpu=float(np.mean([r.get("infer_ms_cpu", r["val"].get("infer_ms_cpu")) for r in rows])))
    if "train_sec" in rows[0]:
        d["train_sec"] = float(np.mean([r["train_sec"] for r in rows]))
        d["epochs"] = float(np.mean([r["epochs"] for r in rows]))
        d["n_train"] = rows[0]["n_train"]
    for sp in ("val", "test"):
        d[sp] = dict(accuracy=agg(rows, sp, "accuracy"), macro_f1=agg(rows, sp, "macro_f1"),
                     recall={l: agg_rec(rows, sp, l) for l in LABELS}, confusion_sum=cm_sum(rows, sp).tolist(),
                     per_seed=[dict(seed=r.get("seed", r.get("run")), accuracy=r[sp]["accuracy"] * 100,
                                    macro_f1=r[sp]["macro_f1"] * 100) for r in rows])
    numbers["groups"][g] = d
for k, r in jin_own.items():
    numbers["jin_own"][k] = {sp: dict(n=r[sp]["n"], accuracy=r[sp]["accuracy"] * 100, macro_f1=r[sp]["macro_f1"] * 100,
                                      recall={l: r[sp]["per_class"][l]["recall"] * 100 for l in LABELS},
                                      confusion=r[sp]["confusion"], infer_ms_cpu=r[sp]["infer_ms_cpu"])
                             for sp in ("val", "test")} | dict(params=r["params"], model_config=r["model_config"],
                                                               best_epoch=r["best_epoch"])
for k, r in jin_ours.items():
    numbers["jin_on_ours_per_seed"][k] = {sp: dict(accuracy=r[sp]["accuracy"] * 100, macro_f1=r[sp]["macro_f1"] * 100,
                                                   recall={l: r[sp]["per_class"][l]["recall"] * 100 for l in LABELS})
                                          for sp in ("val", "test")} | dict(best_epoch=r["best_epoch"])
(OUT / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding="utf-8")

# ── ① 정확도 / macro F1 막대 (val=p004, test=p006), 오차막대 = 시드 표준편차 ──
main_keys = ["gru_recipe", "lstm_recipe", "jin_lstm1_ours_data", "gru_plain", "lstm_plain"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, sp, title in zip(axes, ("val", "test"), ("검증 p004 (LSTM 원본 실험의 검증셋)", "시험 p006 (아무도 학습에 안 넣음)")):
    x = np.arange(len(main_keys)) * 1.0
    w = 0.36
    for j, (key, lab) in enumerate((("accuracy", "정확도"), ("macro_f1", "macro F1"))):
        m = [numbers["groups"][g][sp][key][0] for g in main_keys]
        s = [numbers["groups"][g][sp][key][1] for g in main_keys]
        bars = ax.bar(x + (j - 0.5) * w, m, w, yerr=s, capsize=3, label=lab,
                      color=[COLORS[g] for g in main_keys], alpha=1.0 if j == 0 else 0.55,
                      edgecolor="white")
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    b = ours["baseline"][sp]
    ax.axhline(b["accuracy"] * 100, color=C_RULE, ls="--", lw=1.2)
    ax.text(len(main_keys) - 0.5, b["accuracy"] * 100 + 0.6, f"룰 baseline 정확도 {b['accuracy']*100:.1f}",
            ha="right", va="bottom", fontsize=8, color=C_RULE)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[g] for g in main_keys], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(50, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("%")
axes[0].legend(loc="lower left", fontsize=8, title="진한색=정확도, 연한색=macro F1", title_fontsize=8)
fig.suptitle("GRU vs LSTM — 같은 데이터, 같은 사람 단위 분할, 시드 3개 평균 ± 표준편차", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig1_accuracy_f1.png", dpi=150)
plt.close(fig)

# ── ② 클래스별 recall 묶음 막대 ──
keys2 = ["gru_recipe", "lstm_recipe", "jin_lstm1_ours_data"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, sp, title in zip(axes, ("val", "test"), ("검증 p004", "시험 p006")):
    x = np.arange(len(LABELS))
    w = 0.22
    for j, g in enumerate(keys2):
        m = [numbers["groups"][g][sp]["recall"][l][0] for l in LABELS]
        s = [numbers["groups"][g][sp]["recall"][l][1] for l in LABELS]
        bars = ax.bar(x + (j - 1) * w, m, w, yerr=s, capsize=2, color=COLORS[g], label=NAMES[g], edgecolor="white")
        for b_, v in zip(bars, m):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    rb = [ours["baseline"][sp]["per_class"][l]["recall"] * 100 for l in LABELS]
    ax.scatter(x + 2 * w, rb, marker="_", s=300, color=C_RULE, label="룰 baseline", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([KO[l] for l in LABELS])
    ax.set_title(title)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("재현율 (%)  = 그 동작을 맞게 알아본 비율")
fig.legend(*axes[1].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=4, frameon=False)
fig.suptitle("클래스별 재현율 — 차이는 거의 전부 no_gesture 에서 난다", fontsize=12)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig2_per_class_recall.png", dpi=150)
plt.close(fig)

# ── ③ 혼동행렬: GRU(시연 레시피) vs LSTM(LSTM 원본 코드, 같은 데이터), test p006, 시드 3개 합 ──
def draw_cm(ax, cm, title, cmap):
    cm = np.array(cm)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([KO[l] for l in LABELS], fontsize=9)
    ax.set_yticklabels([KO[l] for l in LABELS], fontsize=9)
    ax.set_xlabel("모델이 답한 것")
    ax.set_ylabel("실제 동작")
    ax.set_title(title, fontsize=10)

for sp, fname in (("test", "fig3_confusion_test_p006.png"), ("val", "fig3b_confusion_val_p004.png")):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    draw_cm(axes[0], numbers["groups"]["gru_recipe"][sp]["confusion_sum"],
            f"GRU — {sp} {'p006' if sp=='test' else 'p004'}, 시드 3개 합\n"
            f"정확도 {numbers['groups']['gru_recipe'][sp]['accuracy'][0]:.1f}%", "Blues")
    draw_cm(axes[1], numbers["groups"]["jin_lstm1_ours_data"][sp]["confusion_sum"],
            f"LSTM 1층 (원본 코드, 같은 데이터) — 시드 3개 합\n"
            f"정확도 {numbers['groups']['jin_lstm1_ours_data'][sp]['accuracy'][0]:.1f}%", "Oranges")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)

# ── ④ 파라미터 수 / CPU 추론 시간 (bench_infer.py 200회 중앙값) ──
keys4 = ["gru_recipe", "lstm_recipe", "jin_lstm1_ours_data"]
bench = json.loads((RAW / "bench_infer.json").read_text(encoding="utf-8"))
BENCH_KEY = {"gru_recipe": "gru_ours", "gru_plain": "gru_ours", "lstm_recipe": "lstm_ours", "lstm_plain": "lstm_ours",
             "jin_lstm1_ours_data": "lstm_jin_1layer"}
for g in numbers["groups"]:
    numbers["groups"][g]["infer_ms_cpu_median200"] = bench[BENCH_KEY[g]]["median_ms"]
numbers["bench_infer"] = bench
(OUT / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding="utf-8")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
p = [numbers["groups"][g]["params"] for g in keys4]
t = [numbers["groups"][g]["infer_ms_cpu_median200"] for g in keys4]
for ax, vals, ttl, fmt in zip(axes, (p, t), ("파라미터 수 (모델 크기)", "추론 시간, CPU, 클립 1개 (ms)"),
                              ("{:,}", "{:.2f} ms")):
    bars = ax.bar(range(len(keys4)), vals, color=[COLORS[g] for g in keys4], edgecolor="white")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(keys4)))
    ax.set_xticklabels([SHORT[g] for g in keys4], fontsize=8)
    ax.set_title(ttl)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18)
fig.suptitle("셋 다 실시간에 충분히 가볍다 (웹캠 한 프레임 33ms 대비)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig4_params_latency.png", dpi=150)
plt.close(fig)

# ── ⑤ 학습 곡선: GRU/LSTM (시드 0, 시연 레시피) + LSTM 원본(같은 데이터 seed 42) + LSTM 원본 final_1layers ──
def read_hist_csv(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return ([float(r["train_loss"]) for r in rows], [float(r["val_loss"]) for r in rows],
            [float(r["train_accuracy"]) * 100 for r in rows], [float(r["val_accuracy"]) * 100 for r in rows])

curves = []
for g, lab in (("gru_recipe", "GRU (시드0)"), ("lstm_recipe", "LSTM (시드0)")):
    r0 = [r for r in groups[g] if r.get("history")][0]
    h = r0["history"]
    curves.append((lab, h["loss"], h["val_loss"], [a * 100 for a in h["accuracy"]], [a * 100 for a in h["val_accuracy"]], COLORS[g]))
jp = SCR / "jin" / "lstm_ours_1l_seed42" / "history.csv"
if jp.exists():
    curves.append(("LSTM 1층 (원본 코드, 같은 데이터, seed42)", *read_hist_csv(jp), C_LSTM_JIN))
jp2 = SCR / "jin" / "artifacts" / "final_1layers_001" / "history.csv"
if jp2.exists():
    curves.append(("LSTM 1층 (원본 final_1layers_001)", *read_hist_csv(jp2), "#B35A1E"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for lab, tl, vl, ta, va, col in curves:
    ep = np.arange(1, len(tl) + 1)
    axes[0].plot(ep, vl, color=col, label=lab)
    axes[0].plot(ep, tl, color=col, ls=":", alpha=0.7)
    axes[1].plot(ep, va, color=col, label=lab)
    axes[1].plot(ep, ta, color=col, ls=":", alpha=0.7)
axes[0].set_title("손실 (실선 = 검증 p004, 점선 = 학습)")
axes[1].set_title("정확도 % (실선 = 검증 p004, 점선 = 학습)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylim(0, 1.6)
axes[1].set_ylim(50, 101)
axes[1].legend(fontsize=8, loc="lower right")
fig.suptitle("학습 곡선 — GRU/LSTM(시연 레시피)은 최적 가중치 복원 + patience 12, LSTM 원본은 patience 10", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig5_training_curves.png", dpi=150)
plt.close(fig)


# ── markdown 표 조각 ──
def fmt(ms):
    return f"{ms[0]:.1f} ± {ms[1]:.1f}"


lines = ["| 모델 | 구조 | 파라미터 | 학습 데이터 | val p004 정확도 | val macro F1 | test p006 정확도 | test macro F1 | "
         "test 재현율 스와이프/주먹/no_g/스냅 | CPU 추론 ms |", "|---|---|---:|---|---:|---:|---:|---:|---|---:|"]
STRUCT = {"gru_recipe": "GRU 2층 64 + FC32", "gru_plain": "GRU 2층 64 + FC32", "lstm_recipe": "LSTM 2층 64 + FC32",
          "lstm_plain": "LSTM 2층 64 + FC32", "jin_lstm1_ours_data": "LSTM 1층 64 + Linear"}
for g in main_keys:
    d = numbers["groups"][g]
    rec = "/".join(f"{d['test']['recall'][l][0]:.0f}" for l in LABELS)
    lines.append(f"| {d['name']} | {STRUCT[g]} | {d['params']:,} | {d.get('n_train', '')} | {fmt(d['val']['accuracy'])} | "
                 f"{fmt(d['val']['macro_f1'])} | {fmt(d['test']['accuracy'])} | {fmt(d['test']['macro_f1'])} | {rec} | "
                 f"{d['infer_ms_cpu_median200']:.2f} |")
b = ours["baseline"]
rec = "/".join(f"{b['test']['per_class'][l]['recall']*100:.0f}" for l in LABELS)
lines.append(f"| 룰 baseline (학습 없음) | if-then 규칙 | 0 | - | {b['val']['accuracy']*100:.1f} | {b['val']['macro_f1']*100:.1f} | "
             f"{b['test']['accuracy']*100:.1f} | {b['test']['macro_f1']*100:.1f} | {rec} | ~0 |")
(OUT / "table_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("\nsaved to", OUT)
