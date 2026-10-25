"""Load recorded sequences. File name format: <person>_<timestamp>.npy"""
import numpy as np
import config


def load_sequences(data_dir=None):
    data_dir = data_dir or config.DATA_DIR
    X, y, persons = [], [], []
    for idx, sign in enumerate(config.SIGNS):
        for f in sorted((data_dir / sign).glob("*.npy")):
            arr = np.load(f)
            if arr.shape != (config.SEQ_LEN, config.FEATURE_DIM):
                print(f"skip {f} (shape {arr.shape})")
                continue
            X.append(arr)
            y.append(idx)
            persons.append(f.stem.split("_")[0])
    if not X:
        raise SystemExit(f"No data found in {data_dir}. Run collect_data.py first.")
    return np.stack(X).astype(np.float32), np.array(y), np.array(persons)
