"""Record landmark sequences for training / testing.

Usage:  python collect_data.py --person zuhaa
        python collect_data.py --person mariam --data-dir data_lowlight   (another condition)

Keys:  1..N = choose sign   SPACE = record (2 s countdown, then 30 frames)
       u = undo last        q = quit
Tips:  for 'unknown' record idle hands, random gestures and hand-in-and-out of frame.
       Vary distance, angle, lighting. Sign with the same hand every time.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from features import make_hands, extract_features, draw_landmarks, open_camera, grab_frame


def put(img, text, y, scale=0.8, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def count_existing(data_dir):
    return {s: len(list((data_dir / s).glob("*.npy"))) for s in config.SIGNS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="signer id (no underscores)")
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()
    person = args.person.replace("_", "-")
    data_dir = Path(args.data_dir)
    for s in config.SIGNS:
        (data_dir / s).mkdir(parents=True, exist_ok=True)

    cap, hands = open_camera(), make_hands()
    counts = count_existing(data_dir)
    sel, state, t0, seq, hand_frames = 0, "idle", 0.0, [], 0
    last_saved, msg = None, ""

    while True:
        frame = grab_frame(cap)
        if frame is None:
            print("Camera not available")
            break
        feats, detected, res = extract_features(frame, hands)
        draw_landmarks(frame, res)
        now = time.time()

        if state == "countdown":
            left = 2.0 - (now - t0)
            if left <= 0:
                state, seq, hand_frames = "recording", [], 0
            else:
                put(frame, f"GET READY: {left:.1f}", 90, 1.2, (0, 255, 255))
        elif state == "recording":
            seq.append(feats)
            hand_frames += int(detected)
            put(frame, f"RECORDING {len(seq)}/{config.SEQ_LEN}", 90, 1.0, (0, 0, 255))
            if len(seq) == config.SEQ_LEN:
                sign = config.SIGNS[sel]
                if sign != config.UNKNOWN and hand_frames < config.SEQ_LEN * config.MIN_HAND_FRACTION:
                    msg = "Hand lost too often - discarded"
                else:
                    path = data_dir / sign / f"{person}_{datetime.now():%Y%m%d%H%M%S%f}.npy"
                    np.save(path, np.stack(seq).astype(np.float32))
                    counts[sign] += 1
                    last_saved, msg = path, f"Saved {sign}"
                state = "idle"

        put(frame, f"Sign: {config.SIGNS[sel]}  ({counts[config.SIGNS[sel]]} saved)", 30)
        put(frame, "hand: " + ("YES" if detected else "NO"), frame.shape[0] - 40, 0.7,
            (0, 255, 0) if detected else (0, 0, 255))
        put(frame, msg, frame.shape[0] - 10, 0.7, (255, 255, 0))
        put(frame, " ".join(f"[{i+1}]{s}:{counts[s]}" for i, s in enumerate(config.SIGNS)),
            frame.shape[0] - 70, 0.5)
        cv2.imshow("SignSpeak - collect", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if state == "idle":
            if ord("1") <= k < ord("1") + len(config.SIGNS):
                sel, msg = k - ord("1"), ""
            elif k == ord(" "):
                state, t0, msg = "countdown", now, ""
            elif k == ord("u") and last_saved is not None and last_saved.exists():
                last_saved.unlink()
                counts[last_saved.parent.name] -= 1
                msg, last_saved = "Undid last", None

    cap.release()
    cv2.destroyAllWindows()
    print("Final counts:", counts)


if __name__ == "__main__":
    main()
