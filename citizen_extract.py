"""STEP 1 for the ASL Citizen dataset (videos of 52 signers) - pull out the signs we need as hand landmarks.

  python citizen_extract.py --root D:/asl-citizen --list                 # show what matches (no processing)
  python citizen_extract.py --root D:/asl-citizen --find water           # search glosses containing a word
  python citizen_extract.py --root D:/asl-citizen                        # extract (about 20-60 min on a laptop)

It writes TWO files in the same format as islr_raw.npz, so the existing islr_convert.py can process them:
  citizen_raw_trainval.npz   clips from the dataset's train + validation signers  -> for TRAINING
  citizen_raw_test.npz       clips from the dataset's TEST signers                -> never trained on, used to
                             measure accuracy on people the model has never seen

Then:
  python islr_convert.py --raw citizen_raw_trainval.npz --out data_citizen --aspect 1.3333
  python islr_convert.py --raw citizen_raw_test.npz --out data_citizen_test --aspect 1.3333 --flip <yes|no>
      (use the same --flip value that the first command printed as "Using aspect ..., mirror ...")
  python train.py --test-person YOURNAME --extra-data data_islr data_citizen
  python evaluate.py acc --data data_citizen_test --name citizen_test

The file layout of the Kaggle copy is detected automatically: any train/val/test .csv under --root with
columns for participant, video file and gloss, and the videos anywhere below --root.
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from islr_extract import signing_hand

DEFAULT_SIGNS = ["hello", "goodbye", "thank_you", "i_love_you", "yes", "no", "please", "sorry", "help",
                 "mom", "dad", "water", "food", "drink", "hungry", "happy", "sad", "bad", "where"]
# our sign -> glosses (letters only, lower case) accepted from the dataset
SYNONYMS = {
    "hello": ["hello", "hi"], "goodbye": ["goodbye", "bye"], "thank_you": ["thankyou", "thanks"],
    "i_love_you": ["iloveyou", "loveyou"], "mom": ["mom", "mother", "mommy", "mama"],
    "dad": ["dad", "father", "daddy", "papa"], "food": ["food", "eat"], "sorry": ["sorry"],
}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
OWN_ASPECT = 4 / 3
MAX_FRAMES = 300


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def read_tables(root):
    """-> list of (split name, participant, video file name, gloss)."""
    rows = []
    for csv_path in sorted(root.rglob("*.csv")):
        low = csv_path.stem.lower()
        split = "test" if "test" in low else "val" if ("val" in low or "dev" in low) else "train" if "train" in low else None
        if split is None:
            continue
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = {norm(c): c for c in (reader.fieldnames or [])}
            pc = next((cols[k] for k in cols if "participant" in k or k in ("signer", "user", "person")), None)
            vc = next((cols[k] for k in cols if "video" in k or k in ("file", "filename", "path")), None)
            gc = next((cols[k] for k in cols if "gloss" in k or k in ("sign", "label", "word")), None)
            if not (vc and gc):
                print(f"  skipping {csv_path.name}: could not find video/gloss columns in {reader.fieldnames}")
                continue
            n = 0
            for r in reader:
                rows.append((split, (r.get(pc) or "").strip() if pc else "", (r[vc] or "").strip(), (r[gc] or "").strip()))
                n += 1
            print(f"  read {csv_path.name}: {n} rows  [{split}]  columns -> participant: {pc}, video: {vc}, gloss: {gc}")
    return rows


def index_videos(root):
    idx = {}
    for dp, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in VIDEO_EXT:
                idx[fn] = os.path.join(dp, fn)
                idx.setdefault(Path(fn).stem, os.path.join(dp, fn))
    return idx


def pid_number(pid):
    """'P12' -> 100012 (offset keeps these signers apart from the ISLR signers)."""
    m = re.search(r"\d+", str(pid))
    return 100000 + (int(m.group()) if m else 0)


def parse_map(items):
    """--map mom=MOM,MOTHER  ->  {'mom': ['mom','mother']}"""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = [norm(x) for x in v.split(",") if x.strip()]
    return out


def match_glosses(targets, glosses, extra):
    by_norm = {}
    for g in glosses:
        by_norm.setdefault(norm(g), []).append(g)
    found, missing = {}, []                              # dataset gloss -> our sign
    for t in targets:
        cands = extra.get(t) or ([norm(t)] + SYNONYMS.get(t, []))
        hit = False
        for c in cands:
            for g in by_norm.get(c, []):
                found[g] = t
                hit = True
        if not hit:
            missing.append(t)
    return found, missing


def process_video(job):
    """(path, stride) -> hand[T,21,3] (x already rescaled as if the picture were 4:3), or None."""
    path, stride = job
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    left, right, n = [], [], 0
    try:
        while n < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if (n - 1) % stride:
                continue
            h, w = frame.shape[:2]
            fix = (w / h) / OWN_ASPECT                     # turn any picture shape into "as if 4:3"
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            L = np.full((21, 3), np.nan, np.float32)
            R = np.full((21, 3), np.nan, np.float32)
            if res.multi_hand_landmarks:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)
                    pts[:, 0] = 0.5 + (pts[:, 0] - 0.5) * fix
                    (L if hd.classification[0].label == "Left" else R)[:] = pts
            left.append(L)
            right.append(R)
    finally:
        cap.release()
        hands.close()
    if not left:
        return None
    return signing_hand(np.stack(left), np.stack(right))


def run_jobs(jobs, workers, stride):
    work = [(j["path"], stride) for j in jobs]
    if workers <= 1:
        for i, w in enumerate(work, 1):
            yield process_video(w)
            if i % 25 == 0:
                print(f"  {i}/{len(work)} videos", flush=True)
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap(process_video, work, chunksize=4), 1):
                yield r
                if i % 25 == 0:
                    print(f"  {i}/{len(work)} videos", flush=True)


def save(jobs, results, out_path):
    clips, labels, pids, seqs, glosses = [], [], [], [], []
    for j, hand in zip(jobs, results):
        if hand is None:
            continue
        clips.append(hand)
        labels.append(j["label"]); pids.append(j["pid"]); seqs.append(j["seq"]); glosses.append(j["gloss"])
    if not clips:
        print(f"Nothing usable for {out_path}")
        return
    np.savez_compressed(out_path, pts=np.concatenate(clips).astype(np.float32),
                        lens=np.array([len(c) for c in clips]), labels=np.array(labels),
                        pids=np.array(pids), seqs=np.array(seqs), kaggle_sign=np.array(glosses))
    print(f"Saved {len(clips)} clips -> {out_path}  per class: {dict(Counter(labels))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with the ASL Citizen csv files and videos")
    ap.add_argument("--out", default="citizen_raw", help="output prefix")
    ap.add_argument("--signs", default=None, help="comma list of OUR sign names (default: your config.py)")
    ap.add_argument("--map", nargs="*", default=[], help="override matching, e.g. --map mom=MOM,MOTHER water=WATER")
    ap.add_argument("--max-per-sign", type=int, default=60)
    ap.add_argument("--unknown", type=int, default=300, help="clips of OTHER glosses for the 'unknown' class")
    ap.add_argument("--stride", type=int, default=2, help="use every k-th frame (2 = twice as fast)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="only show what matches, then stop")
    ap.add_argument("--find", default=None, help="print dataset glosses containing this text, then stop")
    args = ap.parse_args()

    root = Path(args.root)
    print("Reading the csv files ...")
    table = read_tables(root)
    if not table:
        sys.exit("No train/val/test csv with video + gloss columns found under --root.")
    glosses = sorted({g for _, _, _, g in table})
    print(f"{len(table)} rows, {len(glosses)} different glosses, "
          f"{len({p for _, p, _, _ in table})} signers.")
    if args.find:
        key = norm(args.find)
        hits = [g for g in glosses if key in norm(g)]
        print(f"Glosses containing '{args.find}':", ", ".join(hits) if hits else "(none)")
        return

    if args.signs:
        targets = [s.strip() for s in args.signs.split(",") if s.strip()]
    else:
        try:
            import config
            targets = [s for s in config.SIGNS if s != getattr(config, "UNKNOWN", "unknown")]
        except Exception:
            targets = DEFAULT_SIGNS
    found, missing = match_glosses(targets, glosses, parse_map(args.map))
    counts = Counter((t, s) for s, _, _, g in table for gg, t in [(g, found.get(g))] if t)
    for g, t in sorted(found.items(), key=lambda kv: kv[1]):
        print(f"  match: dataset '{g}' -> ours '{t}'   "
              + ", ".join(f"{sp}:{sum(1 for s, _, _, gg in table if gg == g and s == sp)}" for sp in ("train", "val", "test")))
    if missing:
        print("  NOT found in the dataset:", ", ".join(missing),
              "(check spelling with --find WORD, then pass --map sign=GLOSS)")
    if args.list or not found:
        return

    videos = index_videos(root)
    print(f"Found {len(videos)} video files under {root}.")
    rng = np.random.default_rng(args.seed)

    def build(splits):
        jobs, per = [], {}
        rows = [r for r in table if r[0] in splits]
        rng.shuffle(rows)
        for split, pid, vf, g in rows:
            t = found.get(g)
            if t is None or per.get(t, 0) >= args.max_per_sign:
                continue
            p = videos.get(vf) or videos.get(Path(vf).stem)
            if not p:
                continue
            per[t] = per.get(t, 0) + 1
            jobs.append({"path": p, "label": t, "gloss": g, "pid": pid_number(pid), "seq": len(jobs)})
        if args.unknown:
            others = [r for r in rows if r[3] not in found]
            for split, pid, vf, g in others[:args.unknown * 3]:
                p = videos.get(vf) or videos.get(Path(vf).stem)
                if p and sum(1 for j in jobs if j["label"] == "unknown") < args.unknown:
                    jobs.append({"path": p, "label": "unknown", "gloss": g, "pid": pid_number(pid), "seq": len(jobs)})
        return jobs

    for name, splits in (("trainval", {"train", "val"}), ("test", {"test"})):
        jobs = build(splits)
        if not jobs:
            print(f"No videos for {name}; skipping.")
            continue
        print(f"Processing {len(jobs)} videos for {name} (this is the slow part) ...")
        results = list(run_jobs(jobs, args.workers, args.stride))
        save(jobs, results, f"{args.out}_{name}.npz")
    print("Next: python islr_convert.py --raw citizen_raw_trainval.npz --out data_citizen --aspect 1.3333")


if __name__ == "__main__":
    main()
