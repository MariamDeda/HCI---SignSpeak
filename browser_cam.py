"""Live camera in the BROWSER (works on your laptop and on the deployed streamlit.app).

The picture comes from the visitor's own webcam through WebRTC (streamlit-webrtc). Each frame is
processed on the server exactly like before (MediaPipe hand tracking -> LSTM -> stabilizer), drawn
with the hand landmarks and sent back to the page, so the visitor sees a live, smooth video.

Install once:  python -m pip install streamlit-webrtc
To use the old server-side camera instead, put  CAMERA_MODE = "server"  in config.py.
"""
import threading
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import config
from features import draw_landmarks, extract_features, make_hands
from stabilizer import NO_HAND, Stabilizer

ASPECT = config.FRAME_W / config.FRAME_H            # training data is 4:3, so frames are cropped to 4:3


def crop_to_aspect(img):
    h, w = img.shape[:2]
    if w / h > ASPECT + 0.01:                          # 16:9 webcam -> cut the sides
        new_w = int(h * ASPECT)
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    if img.shape[1] > config.FRAME_W:                  # keep processing fast
        img = cv2.resize(img, (config.FRAME_W, int(img.shape[0] * config.FRAME_W / img.shape[1])))
    return img


class SignProcessor(VideoProcessorBase):
    """Runs in a background thread: one call of recv() per camera frame."""

    def __init__(self, model, signs, logger=None, conf_thr=config.CONF_THRESHOLD,
                 n_stable=config.STABLE_PREDICTIONS, condition="lab"):
        self.model, self.signs, self.logger, self.condition = model, signs, logger, condition
        self.hands = make_hands()
        self.window = deque(maxlen=config.SEQ_LEN)
        self.stab = Stabilizer(conf_thr=conf_thr, n_stable=n_stable)
        self.conf_thr, self.n_stable = conf_thr, n_stable
        self.n_frames, self.label, self.conf, self.detected = 0, NO_HAND, 0.0, False
        self.lock = threading.Lock()
        self.commits = []
        self.state, self.hint = "warming", None

    def set_params(self, conf_thr, n_stable, condition=None):
        with self.lock:
            self.conf_thr = self.stab.conf_thr = conf_thr
            self.n_stable = self.stab.n_stable = n_stable
            if condition:
                self.condition = condition

    def snapshot(self):
        """Called by the page: what happened since last time."""
        with self.lock:
            out = {"commits": self.commits, "state": self.state, "hint": self.hint,
                   "label": self.label, "conf": self.conf, "thr": self.conf_thr}
            self.commits = []
            return out

    def _state(self):
        if not self.detected:
            return "nohand"
        if self.model is None:
            return "nomodel"
        if self.n_frames < config.SEQ_LEN:
            return "warming"
        if self.label in (config.UNKNOWN, NO_HAND):
            return "nosign" if self.conf >= self.conf_thr else "low"
        if self.conf < self.conf_thr:
            return "low"
        if not (self.stab.last_label == self.label and not self.stab.released):
            return "holding"
        return "ok"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if config.MIRROR:
            img = cv2.flip(img, 1)
        img = crop_to_aspect(img)
        try:
            t_cap = time.perf_counter()
            feats, self.detected, res = extract_features(img, self.hands)
            draw_landmarks(img, res)
            self.window.append(feats)
            self.n_frames += 1
            if (self.model is not None and len(self.window) == config.SEQ_LEN
                    and self.n_frames % config.PREDICT_EVERY == 0):
                arr = np.stack(self.window)
                hand_frac = float((np.abs(arr).sum(1) > 0).mean())
                if hand_frac < config.MIN_HAND_FRACTION:
                    self.label, self.conf = NO_HAND, 0.0
                else:
                    probs = self.model(arr[None], training=False).numpy()[0]
                    if len(probs) != len(self.signs):
                        with self.lock:
                            self.state = "mismatch"
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                    k = int(probs.argmax())
                    self.label, self.conf = self.signs[k], float(probs[k])
                t_pred = time.perf_counter()
                commit = self.stab.update(self.label, self.conf, t_pred, t_cap)
                if self.logger is not None:
                    self.logger.log(self.condition, t_cap, t_pred, self.label, self.conf, commit)
                if commit:
                    with self.lock:
                        self.commits.append(commit["label"])
            with self.lock:
                self.state = self._state()
                self.hint = (f"Same sign steady for {self.stab.cand_count} of {self.n_stable} readings. "
                             "Keep holding it." if self.state == "holding" else None)
        except Exception:                               # never let one bad frame stop the video
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def _twilio_servers():
    """Optional: free Twilio TURN. Put TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in the Streamlit secrets."""
    try:
        sid, tok = st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"]
        from twilio.rest import Client
        return Client(sid, tok).tokens.create().ice_servers
    except Exception:
        return []


def _rtc_configuration():
    """Public STUN is enough on most networks. If the deployed app cannot connect, add a TURN server
    in the Streamlit secrets:  [turn]  urls = ["turn:..."]  username = "..."  credential = "..."."""
    servers = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
    try:
        turn = st.secrets.get("turn")
        if turn:
            servers.append({"urls": list(turn["urls"]), "username": turn["username"],
                            "credential": turn["credential"]})
    except Exception:
        pass
    servers += _twilio_servers()
    return {"iceServers": servers}


TEXTS = {"start": "Start camera", "stop": "Stop camera", "select_device": "Choose camera",
         "media_api_not_available": "This browser cannot use the camera. Try Chrome or Edge.",
         "device_ask_permission": "Click Start camera and allow camera access when the browser asks.",
         "device_not_available": "No camera was found. Plug one in and reload.",
         "device_access_denied": "Camera access was blocked. Click the lock icon in the address bar, allow the camera, then reload."}


def camera(model, signs, logger=None, key="signer_cam"):
    """Draws the live video box and returns the streamer context."""
    return webrtc_streamer(
        key=key, mode=WebRtcMode.SENDRECV, rtc_configuration=_rtc_configuration(),
        media_stream_constraints={"video": {"width": {"ideal": config.FRAME_W}, "height": {"ideal": config.FRAME_H}},
                                  "audio": False},
        video_processor_factory=lambda: SignProcessor(model, signs, logger),
        async_processing=True, translations=TEXTS,
        video_html_attrs={"autoPlay": True, "playsInline": True, "controls": False, "muted": True,
                          "style": {"width": "100%", "borderRadius": "12px"}})
