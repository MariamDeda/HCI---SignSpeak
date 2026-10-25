"""Train the LSTM sign classifier.

  python train.py                       # random stratified 80/20 split by sequence
  python train.py --test-person mariam  # leave one signer out (recommended: no leakage)
"""
import argparse
import json

import numpy as np

import config
from dataio import load_sequences
from evaluate import report_classification


def augment(X, rng, copies=3):
    """Noise + scale jitter on hand coordinates; missing-hand (all-zero) frames stay zero."""
    out = [X]
    for _ in range(copies):
        A = X.copy()
        present = (np.abs(A).sum(-1, keepdims=True) > 0)
        A[..., :63] *= rng.uniform(0.9, 1.1, size=(len(A), 1, 1)).astype(np.float32)
        A += rng.normal(0, 0.01, A.shape).astype(np.float32)
        out.append(A * present)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--test-person", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from tensorflow import keras
    from tensorflow.keras import layers

    X, y, persons = load_sequences(Path(args.data_dir))
    print("Sequences per class:", {s: int((y == i).sum()) for i, s in enumerate(config.SIGNS)})
    print("Signers:", sorted(set(persons)))

    if args.test_person:
        test = persons == args.test_person.replace("_", "-")
        if not test.any() or test.all():
            raise SystemExit("--test-person must match some, but not all, of the data")
        Xtr, ytr, Xte, yte = X[~test], y[~test], X[test], y[test]
        split = f"leave-one-signer-out ({args.test_person})"
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        split = "random stratified 80/20 (same signers in train and test - optimistic)"
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=1)

    rng = np.random.default_rng(0)
    Xtr_aug = augment(Xtr, rng)
    ytr_aug = np.tile(ytr, 4)

    inp = keras.Input((config.SEQ_LEN, config.FEATURE_DIM))
    h = layers.LSTM(64, return_sequences=True)(inp)
    h = layers.Dropout(0.3)(h)
    h = layers.LSTM(32)(h)
    h = layers.Dense(32, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(len(config.SIGNS), activation="softmax")(h)
    model = keras.Model(inp, out)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr_aug, ytr_aug, validation_data=(Xva, yva), epochs=args.epochs, batch_size=16,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=2)

    pred = model.predict(Xte, verbose=0).argmax(1)
    acc = report_classification(yte, pred, config.SIGNS, "test")

    config.MODEL_DIR.mkdir(exist_ok=True)
    model.save(config.MODEL_PATH)
    config.META_PATH.write_text(json.dumps({
        "signs": config.SIGNS, "seq_len": config.SEQ_LEN, "feature_dim": config.FEATURE_DIM,
        "split": split, "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "test_accuracy": float(acc)}, indent=2))
    print("Saved", config.MODEL_PATH)


if __name__ == "__main__":
    main()
