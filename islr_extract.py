"""STEP 1 - pull the clips we need out of the Google ISLR (Kaggle "asl-signs") landmark dataset.

Run it where the Kaggle files are (best: a free Kaggle Notebook, so you never download 40 GB):

  python islr_extract.py --root /kaggle/input/asl-signs --out islr_raw.npz
  python islr_extract.py --root D:/asl-signs --out islr_raw.npz            (if you downloaded it)
  python islr_extract.py --root D:/asl-signs --list-signs                  (just show matches)

Per clip it keeps only the signing hand (the one tracked in more frames). A left hand is mirrored
to look like a right hand, so every clip is "one hand". The result is a small file (about 10-20 MB)
that STEP 2 (islr_convert.py) turns into SignSpeak training sequences.

Needs:  pip install pandas pyarrow numpy
"""
import argparse
import re
from pathlib import Path

import numpy as np

DEFAULT_SIGNS = ["hello", "goodbye", "thank_you", "i_love_you", "yes", "no", "please", "sorry", "help"]
# Our sign name -> other spellings used by the Kaggle dataset
SYNONYMS = {"goodbye": ["bye", "goodbye"], "i_love_you": ["iloveyou", "loveyou"],
            "thank_you": ["thankyou", "thanks"]}


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def our_signs():
    try:
        import config
        return [s for s in config.SIGNS if s != getattr(config, "UNKNOWN", "unknown")]
    except Exception:
        return DEFAULT_SIGNS


def match_signs(targets, kaggle_signs):
    """{kaggle sign: our sign} for every target we can find; also returns the targets not found."""
    by_norm = {norm(k): k for k in kaggle_signs}
    found, missing = {}, []
    for t in targets:
        hit = None
        for cand in [norm(t)] + [norm(c) for c in SYNONYMS.get(t, [])]:
            if cand in by_norm:
                hit = by_norm[cand]
                break
        if hit:
            found[hit] = t
        else:
            missing.append(t)
    return found, missing


def read_hands(path):
    """One parquet file -> (left[T,21,3], right[T,21,3]) with NaN where a hand is missing."""
    import pandas as pd
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = np.sort(df["frame"].unique())
    out = {}
    for name in ("left_hand", "right_hand"):
        arr = np.full((len(frames), 21, 3), np.nan, np.float32)
        sub = df[df["type"] == name]
        if len(sub):
            fi = np.searchsorted(frames, sub["frame"].to_numpy())
            arr[fi, sub["landmark_index"].to_numpy()] = sub[["x", "y", "z"]].to_numpy(np.float32)
        out[name] = arr
    return out["left_hand"], out["right_hand"]


def valid_frames(arr):
    return ~np.isnan(arr).any(axis=(1, 2))


def signing_hand(left, right, min_frames=6):
    """Pick the hand tracked in more frames; mirror a left hand so it looks like a right hand."""
    nl, nr = valid_frames(left).sum(), valid_frames(right).sum()
    if max(nl, nr) < min_frames:
        return None
    if nr >= nl:
        return right
    mirrored = left.copy()
    mirrored[..., 0] = 1.0 - mirrored[..., 0]
    return mirrored


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with train.csv and train_landmark_files/")
    ap.add_argument("--out", default="islr_raw.npz")
    ap.add_argument("--signs", default=None, help="comma list of OUR sign names (default: from config.py)")
    ap.add_argument("--max-per-sign", type=int, default=200)
    ap.add_argument("--unknown", type=int, default=600, help="clips of OTHER signs, used as the 'unknown' class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-signs", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    meta = pd.read_csv(root / "train.csv")
    kaggle_signs = sorted(meta["sign"].unique())
    targets = [s.strip() for s in args.signs.split(",")] if args.signs else our_signs()
    found, missing = match_signs(targets, kaggle_signs)
    print(f"Kaggle has {len(kaggle_signs)} signs and {len(meta)} clips from {meta['participant_id'].nunique()} signers.")
    for k, o in found.items():
        print(f"  match: kaggle '{k}'  ->  ours '{o}'   ({(meta['sign'] == k).sum()} clips)")
    if missing:
        print("  NOT in the dataset (keep recording these yourself):", ", ".join(missing))
    if args.list_signs:
        print("\nAll Kaggle signs:", ", ".join(kaggle_signs))
        return
    if not found:
        raise SystemExit("None of your signs were found. Run with --list-signs and use --signs.")

    rng = np.random.default_rng(args.seed)
    jobs = []                                           # (row, our label)
    for k, o in found.items():
        rows = meta[meta["sign"] == k]
        take = rows.sample(min(len(rows), args.max_per_sign), random_state=args.seed)
        jobs += [(r, o) for r in take.itertuples()]
    others = meta[~meta["sign"].isin(found)]
    if args.unknown and len(others):
        take = others.sample(min(len(others), args.unknown), random_state=args.seed)
        jobs += [(r, "unknown") for r in take.itertuples()]
    rng.shuffle(jobs)

    clips, labels, pids, seqs, src_signs = [], [], [], [], []
    skipped = 0
    for n, (r, label) in enumerate(jobs, 1):
        try:
            left, right = read_hands(root / r.path)
        except Exception as e:                           # missing / unreadable file
            skipped += 1
            continue
        hand = signing_hand(left, right)
        if hand is None:
            skipped += 1
            continue
        clips.append(hand); labels.append(label); pids.append(int(r.participant_id))
        seqs.append(int(r.sequence_id)); src_signs.append(r.sign)
        if n % 100 == 0:
            print(f"  {n}/{len(jobs)} clips read")
    lens = np.array([len(c) for c in clips])
    np.savez_compressed(args.out, pts=np.concatenate(clips).astype(np.float32), lens=lens,
                        labels=np.array(labels), pids=np.array(pids), seqs=np.array(seqs),
                        kaggle_sign=np.array(src_signs))
    print(f"Saved {len(clips)} clips to {args.out} (skipped {skipped}: no usable hand or unreadable).")
    from collections import Counter
    print("Per class:", dict(Counter(labels)))


if __name__ == "__main__":
    main()
