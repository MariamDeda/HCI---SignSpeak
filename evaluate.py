"""Evaluation helpers.

  python evaluate.py acc --data data_lowlight --name lowlight
      accuracy / per-class report / confusion matrix of the trained model on a folder
      of recorded sequences (use a separate folder per condition or per signer).

  python evaluate.py latency --log logs/predictions.csv [--condition lab]
      per-frame pipeline latency (frame available -> prediction) and
      stable-run -> committed-text latency.

  python evaluate.py stab --log logs/predictions.csv --script hello,thank_you,goodbye
      replays a scripted session with different stabilization settings and counts
      insertions (false commits), deletions (missed signs) and substitutions.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config


# ---------------------------------------------------------------- classification
def save_confusion_png(cm, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_classification(y_true, y_pred, labels, name):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    config.RESULTS_DIR.mkdir(exist_ok=True)
    idx = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=idx)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, labels=idx, target_names=labels, zero_division=0)
    pairs = sorted(((cm[i, j], labels[i], labels[j]) for i in idx for j in idx if i != j and cm[i, j]),
                   reverse=True)
    text = (f"=== {name} ===\nn = {len(y_true)}  accuracy = {acc:.3f}  macro-F1 = {macro_f1:.3f}\n\n{rep}\n"
            "Most frequent confusions (count, true -> predicted):\n" +
            "\n".join(f"  {c}  {a} -> {b}" for c, a, b in pairs[:8]) + "\n")
    print(text)
    (config.RESULTS_DIR / f"{name}_report.txt").write_text(text)
    with open(config.RESULTS_DIR / f"{name}_confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, row in enumerate(cm):
            w.writerow([labels[i]] + list(map(int, row)))
    save_confusion_png(cm, labels, config.RESULTS_DIR / f"{name}_confusion.png")
    return acc


def cmd_acc(args):
    from tensorflow import keras
    from dataio import load_sequences
    model = keras.models.load_model(config.MODEL_PATH)
    X, y, _ = load_sequences(Path(args.data))
    pred = model.predict(X, verbose=0).argmax(1)
    report_classification(y, pred, config.SIGNS, args.name)


# ---------------------------------------------------------------- logs
def read_log(path, condition=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if condition and r["condition"] != condition:
                continue
            rows.append(r)
    return rows


def stats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "n=0"
    return (f"n={a.size}  mean={a.mean():.1f}  median={np.median(a):.1f}  "
            f"p95={np.percentile(a, 95):.1f}  max={a.max():.1f}")


def cmd_latency(args):
    rows = read_log(args.log, args.condition)
    pipe = [(float(r["t_pred"]) - float(r["t_capture"])) * 1000 for r in rows]
    commit = [(float(r["t_pred"]) - float(r["cand_start"])) * 1000
              for r in rows if r["committed"] and r["cand_start"]]
    print("Pipeline latency, frame available -> prediction (ms):", stats(pipe))
    print("Stable-run start -> committed text (ms):             ", stats(commit))
    print("(UI render time and camera/driver delay are not included - state this in the report.)")


def edit_counts(expected, got):
    """Substitutions, deletions (missed), insertions (false commits) via DP alignment."""
    n, m = len(expected), len(got)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (expected[i - 1] != got[j - 1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (expected[i - 1] != got[j - 1]):
            S += expected[i - 1] != got[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D, i = D + 1, i - 1
        else:
            I, j = I + 1, j - 1
    return S, D, I


def cmd_stab(args):
    from stabilizer import Stabilizer, NO_HAND
    rows = read_log(args.log, args.condition)
    expected = [s.strip() for s in args.script.split(",")]
    out = []
    print(f"{'N':>3} {'thr':>5} {'commits':>7} {'S':>3} {'D':>3} {'I':>3} {'mean latency ms':>16}")
    for n in args.n:
        st = Stabilizer(conf_thr=args.conf, n_stable=n)
        got, lat = [], []
        for r in rows:
            label = r["label"] if r["label"] else NO_HAND
            c = st.update(label, float(r["conf"]), float(r["t_pred"]), float(r["t_capture"]))
            if c:
                got.append(c["label"])
                lat.append((c["t"] - c["cand_start"]) * 1000)
        S, D, I = edit_counts(expected, got)
        ml = float(np.mean(lat)) if lat else float("nan")
        print(f"{n:>3} {args.conf:>5.2f} {len(got):>7} {S:>3} {D:>3} {I:>3} {ml:>16.1f}")
        out.append([n, args.conf, len(got), S, D, I, round(ml, 1)])
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "stabilization_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_stable", "conf_thr", "commits", "substitutions", "deletions",
                    "insertions", "mean_latency_ms"])
        w.writerows(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acc"); a.add_argument("--data", required=True); a.add_argument("--name", required=True)
    l = sub.add_parser("latency"); l.add_argument("--log", required=True); l.add_argument("--condition")
    s = sub.add_parser("stab")
    s.add_argument("--log", required=True); s.add_argument("--condition")
    s.add_argument("--script", required=True, help="comma-separated expected signs, e.g. hello,thank_you")
    s.add_argument("--n", type=int, nargs="+", default=[1, 3, 5, 8, 12])
    s.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    args = ap.parse_args()
    {"acc": cmd_acc, "latency": cmd_latency, "stab": cmd_stab}[args.cmd](args)


if __name__ == "__main__":
    main()
