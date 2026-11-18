"""STEP 2 - turn islr_raw.npz (from islr_extract.py) into SignSpeak training sequences.

  python islr_convert.py --raw islr_raw.npz                 # auto-calibrates against YOUR recordings
  python islr_convert.py --raw islr_raw.npz --aspect 0.75 --flip no     # set it by hand

Writes data_islr/<sign>/islr<signer>_<clip>.npy  (30 x 65, same format as collect_data.py).

What it does for each clip
  1. fills short tracking gaps and cuts the clip to the part where the hand is visible;
  2. converts landmarks to the same 65 features as features.py;
  3. stretches the sign to the 30-frame window (and makes a second copy where the sign fills only
     60-90% of the window with empty edges, like a live window);
  4. fixes two camera differences, which it picks by comparing with YOUR recorded data:
       - picture shape (the dataset is portrait phone video, your webcam is 4:3)
       - mirror image (your app mirrors the webcam).
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

import config

L, D = config.SEQ_LEN, config.FEATURE_DIM
OWN_ASPECT = 640 / 480          # width / height of the webcam picture the app uses
ASPECTS = [0.5625, 0.75, 1.0, 4 / 3]


def to_features(pts, aspect=0.75):
    """pts[T,21,3] (NaN = no hand) -> features[T,65]; zeros where no hand.
    Same maths as features.landmarks_to_features, plus a width correction for the picture shape."""
    T = len(pts)
    out = np.zeros((T, D), np.float32)
    ok = ~np.isnan(pts).any(axis=(1, 2))
    if not ok.any():
        return out
    p = pts[ok].copy()
    wrist_raw = p[:, 0, :2].copy()
    p[..., 0] *= aspect / OWN_ASPECT                       # make dx, dy comparable to a 4:3 picture
    wrist = p[:, 0:1, :]
    scale = np.maximum(np.linalg.norm(p[:, 9, :2] - p[:, 0, :2], axis=1), 1e-6)
    rel = (p - wrist) / scale[:, None, None]
    out[ok] = np.concatenate([rel.reshape(len(p), -1), wrist_raw], axis=1)
    return out


def fill_gaps(pts, max_gap=4):
    """Linear interpolation across tracking gaps of at most max_gap frames."""
    ok = ~np.isnan(pts).any(axis=(1, 2))
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return pts
    out = pts.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = pts[a] * (1 - w)[:, None, None] + pts[b] * w[:, None, None]
    return out


def resample(feats, n):
    """feats[T,65] with zero rows for missing frames -> [n,65]; frames near a gap stay zero."""
    T = len(feats)
    present = np.abs(feats).sum(1) > 0
    if present.sum() < 2:
        return np.zeros((n, D), np.float32)
    src = np.linspace(0, T - 1, n)
    xs = np.flatnonzero(present)
    out = np.stack([np.interp(src, xs, feats[xs, j]) for j in range(D)], axis=1).astype(np.float32)
    nearest = np.clip(np.rint(src).astype(int), 0, T - 1)
    out[~present[nearest]] = 0
    return out


def make_sequences(pts, aspect, flip, rng, padded=True):
    """One raw clip -> list of (suffix, [30,65]) sequences, or [] if the clip is unusable."""
    if flip:
        pts = pts.copy()
        pts[..., 0] = 1.0 - pts[..., 0]
    pts = fill_gaps(pts)
    ok = ~np.isnan(pts).any(axis=(1, 2))
    idx = np.flatnonzero(ok)
    if len(idx) < 6:
        return []
    a, b = idx[0], idx[-1]
    span = pts[a:b + 1]
    if ok[a:b + 1].mean() < 0.7:
        return []
    feats = to_features(span, aspect)
    out = [("", resample(feats, L))]
    if padded:
        n = int(rng.integers(int(0.6 * L), int(0.9 * L) + 1))
        seq = np.zeros((L, D), np.float32)
        start = int(rng.integers(0, L - n + 1))
        seq[start:start + n] = resample(feats, n)
        out.append(("p", seq))
    return out


# ------------------------------------------------------------------ calibration against your own data
def own_mean_pose(sign):
    files = sorted((config.DATA_DIR / sign).glob("*.npy"))
    rows = []
    for f in files:
        a = np.load(f)
        if a.shape == (L, D):
            rows.append(a[np.abs(a).sum(1) > 0, :63])
    return np.concatenate(rows).mean(0) if rows else None


def calibrate(clips_by_sign, rng, per_sign=60):
    """Try every (aspect, flip); keep the pair whose average hand shape is closest to your recordings."""
    common = {s: own_mean_pose(s) for s in clips_by_sign}
    common = {s: m for s, m in common.items() if m is not None and len(clips_by_sign[s])}
    if not common:
        print("No recordings of the same signs in", config.DATA_DIR, "- using defaults (aspect 0.75, no mirror).")
        return 0.75, False, None
    table = {}
    for aspect, flip in itertools.product(ASPECTS, (False, True)):
        total = 0.0
        for s, mine in common.items():
            clips = clips_by_sign[s][:per_sign]
            rows = []
            for c in clips:
                for _, seq in make_sequences(c, aspect, flip, rng, padded=False):
                    rows.append(seq[np.abs(seq).sum(1) > 0, :63])
            if rows:
                total += float(np.linalg.norm(np.concatenate(rows).mean(0) - mine) / np.sqrt(63))
        table[(aspect, flip)] = total / len(common)
    best = min(table, key=table.get)
    return best[0], best[1], (table, list(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="islr_raw.npz")
    ap.add_argument("--out", default="data_islr")
    ap.add_argument("--aspect", type=float, default=None, help="width/height of the Kaggle video, e.g. 0.75")
    ap.add_argument("--flip", choices=["yes", "no"], default=None)
    ap.add_argument("--no-padded", action="store_true", help="skip the 'sign fills only part of the window' copies")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.raw, allow_pickle=False)
    lens, labels, pids, seqs = z["lens"], z["labels"], z["pids"], z["seqs"]
    pts = z["pts"]                                       # read the big array ONCE (every z["pts"] re-reads the file)
    edges = np.concatenate([[0], np.cumsum(lens)])
    clips = [pts[edges[i]:edges[i + 1]] for i in range(len(lens))]
    rng = np.random.default_rng(args.seed)

    by_sign = {}
    for c, lab in zip(clips, labels):
        by_sign.setdefault(str(lab), []).append(c)
    known = {s: v for s, v in by_sign.items() if s in config.SIGNS and s != config.UNKNOWN}

    if args.aspect is not None and args.flip is not None:
        aspect, flip, info = args.aspect, args.flip == "yes", None
    else:
        aspect, flip, info = calibrate(known, rng)
        if args.aspect is not None:
            aspect = args.aspect
        if args.flip is not None:
            flip = args.flip == "yes"
    if info:
        table, signs = info
        print("Calibration against your recordings of:", ", ".join(signs))
        print("  (smaller = closer to your webcam data)")
        for (a, f), v in sorted(table.items(), key=lambda kv: kv[1]):
            print(f"  aspect {a:.4f}  mirror {'yes' if f else 'no ':3}  distance {v:.4f}")
    print(f"Using aspect {aspect:.4f}, mirror {'yes' if flip else 'no'}")

    out_dir = Path(args.out)
    for s in config.SIGNS:
        (out_dir / s).mkdir(parents=True, exist_ok=True)
    kept, dropped, skipped_sign = {}, 0, set()
    for c, lab, pid, sq in zip(clips, labels, pids, seqs):
        lab = str(lab)
        if lab not in config.SIGNS:
            skipped_sign.add(lab)
            continue
        made = make_sequences(c, aspect, flip, rng, padded=not args.no_padded)
        if not made:
            dropped += 1
            continue
        for suffix, seq in made:
            np.save(out_dir / lab / f"islr{pid}_{sq}{suffix}.npy", seq)
        kept[lab] = kept.get(lab, 0) + 1
    print("Clips written per sign:", kept, f"| dropped (hand not visible enough): {dropped}")
    if skipped_sign:
        print("Not in config.SIGNS, so skipped:", ", ".join(sorted(skipped_sign)))
    print("Next: python train.py --test-person YOURNAME --extra-data", args.out)


if __name__ == "__main__":
    main()
