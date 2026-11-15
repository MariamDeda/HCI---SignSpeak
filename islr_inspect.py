"""Look inside a downloaded Kaggle folder and say whether it can be used for training.

  python islr_inspect.py D:\\path\\to\\folder
"""
import sys
from collections import Counter
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python islr_inspect.py FOLDER")
    root = Path(sys.argv[1])
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    files = [p for p in root.rglob("*") if p.is_file()]
    ext = Counter(p.suffix.lower() or "(none)" for p in files)
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} files, {size_mb:,.0f} MB. By type: {dict(ext)}")
    print("Top-level items:", ", ".join(sorted(p.name for p in root.iterdir()))[:300])

    import pandas as pd
    csvs = sorted(root.rglob("*.csv"))
    labelled = None
    for c in csvs[:6]:
        df = pd.read_csv(c, nrows=5)
        n = sum(1 for _ in open(c, encoding="utf-8", errors="ignore")) - 1
        print(f"\n{c.relative_to(root)}: {n} rows, columns = {list(df.columns)}")
        if "sign" in df.columns and "path" in df.columns:
            labelled = c
    pq = next(iter(sorted(root.rglob("*.parquet"))), None)
    if pq is not None:
        d = pd.read_parquet(pq)
        print(f"\nExample landmark file {pq.relative_to(root)}: {len(d)} rows, columns = {list(d.columns)}")
        if "type" in d.columns:
            print("  landmark types:", dict(Counter(d["type"])))
    print()
    if labelled is not None and pq is not None:
        print(f"RESULT: usable. {labelled.name} has the labels ('sign') and the landmark files exist.")
        print("Next:  python islr_extract.py --root", root, "--out islr_raw.npz")
    else:
        print("RESULT: NOT usable for training. There is no CSV that has both 'path' and 'sign' columns")
        print("(labels), or no landmark .parquet files. A test or 'rerun' set normally has no labels.")
        print("Use the competition training data instead (see ISLR_GUIDE.md, step 1).")


if __name__ == "__main__":
    main()
