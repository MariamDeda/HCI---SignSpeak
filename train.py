"""Train the LSTM sign classifier.

  python train.py                       # random stratified 80/20 split by sequence
  python train.py --test-person mariam  # leave one signer out (recommended: no leakage)
  python train.py --test-person mariam --extra-data data_islr
                                        # also learn from the Google ISLR clips. For signs you did NOT record
                                        # yourself, ~20% of the ISLR signers are held out (never trained on) so
                                        # the report still measures accuracy on people the model has not seen.
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
    ap.add_argument("--extra-data", nargs="*", default=[], help="more data folders (e.g. data_islr)")
    ap.add_argument("--islr-holdout", type=float, default=0.2,
                    help="share of extra-data signers kept out of training to test signs you have no recordings of")
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
    n_own = len(Xtr)
    own_classes = np.unique(y)                           # classes you recorded yourself
    hrng = np.random.default_rng(5)
    for folder in args.extra_data:
        Xe, ye, pe = load_sequences(Path(folder))
        print(f"Extra data {folder}: {len(Xe)} sequences from {len(set(pe))} signers")
        new_cls = ~np.isin(ye, own_classes)              # signs with no recording of yours
        if new_cls.any() and args.islr_holdout > 0:
            people = np.array(sorted(set(pe[new_cls])))
            hrng.shuffle(people)
            n_te = max(1, int(round(len(people) * args.islr_holdout)))
            n_va = max(1, n_te // 2)
            te_p, va_p = set(people[:n_te]), set(people[n_te:n_te + n_va])
            m_te = new_cls & np.isin(pe, list(te_p))
            m_va = new_cls & np.isin(pe, list(va_p))
            Xte, yte = np.concatenate([Xte, Xe[m_te]]), np.concatenate([yte, ye[m_te]])
            Xva, yva = np.concatenate([Xva, Xe[m_va]]), np.concatenate([yva, ye[m_va]])
            keep = ~(m_te | m_va)
            print(f"  held out {int(m_te.sum())} test + {int(m_va.sum())} validation clips from unseen signers "
                  f"for signs without your own recordings: "
                  f"{', '.join(config.SIGNS[c] for c in sorted(set(ye[new_cls])))}")
            Xe, ye = Xe[keep], ye[keep]
        Xtr, ytr = np.concatenate([Xtr, Xe]), np.concatenate([ytr, ye])

    rng = np.random.default_rng(0)
    Xtr_aug = augment(Xtr, rng)
    ytr_aug = np.tile(ytr, 4)
    # classes with fewer sequences (for example a sign that is not in the extra data) count more
    counts = np.bincount(ytr_aug, minlength=len(config.SIGNS))
    class_weight = {i: float(len(ytr_aug) / (np.count_nonzero(counts) * c)) for i, c in enumerate(counts) if c}
    print("Training sequences per class:", {s: int(counts[i]) for i, s in enumerate(config.SIGNS)}, f"(own: {n_own})")

    inp = keras.Input((config.SEQ_LEN, config.FEATURE_DIM))
    h = layers.LSTM(64, return_sequences=True)(inp)
    h = layers.Dropout(0.3)(h)
    h = layers.LSTM(32)(h)
    h = layers.Dense(32, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(len(config.SIGNS), activation="softmax")(h)
    model = keras.Model(inp, out)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr_aug, ytr_aug, validation_data=(Xva, yva), epochs=args.epochs, batch_size=16, class_weight=class_weight,
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
