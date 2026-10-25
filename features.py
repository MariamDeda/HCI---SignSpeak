"""MediaPipe landmark extraction + normalization (shared by collection and app)."""
import cv2
import numpy as np
import mediapipe as mp

import config

_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils


def make_hands():
    return _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def landmarks_to_features(hand_landmarks) -> np.ndarray:
    """21 landmarks -> 65 features.
    63 = coordinates relative to the wrist, divided by wrist->middle-finger-base
    distance (removes position and hand-size/camera-distance effects).
    +2 = wrist x,y in the frame (keeps WHERE the sign happens, e.g. forehead vs chest).
    """
    pts = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark], dtype=np.float32)
    wrist = pts[0].copy()
    scale = max(float(np.linalg.norm(pts[9, :2] - pts[0, :2])), 1e-6)
    rel = (pts - wrist) / scale
    return np.concatenate([rel.reshape(-1), wrist[:2]]).astype(np.float32)


def extract_features(frame_bgr, hands):
    """Returns (features[65], hand_detected, mediapipe_results). Zeros if no hand."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        return landmarks_to_features(results.multi_hand_landmarks[0]), True, results
    return np.zeros(config.FEATURE_DIM, dtype=np.float32), False, results


def draw_landmarks(frame_bgr, results):
    if results is not None and results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            _mp_draw.draw_landmarks(frame_bgr, hl, _mp_hands.HAND_CONNECTIONS)


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    return cap


def grab_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1) if config.MIRROR else frame
