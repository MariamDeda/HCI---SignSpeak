"""Role 1 - the person who signs (deaf / mute).

Camera -> MediaPipe hand landmarks -> LSTM -> stabilizer -> BIG text for the people watching,
optionally spoken aloud. The recognition logic is unchanged from Phase 1.
"""
import csv
import html
import time
from collections import deque
from datetime import datetime

import numpy as np
import streamlit as st

import config
import nav
import signs_ref
import theme
from features import draw_landmarks, extract_features, grab_frame, make_hands, open_camera
from stabilizer import NO_HAND, Stabilizer

ss = st.session_state

# text size choice -> (current word size, sentence size)
SIZES = {
    "Large":  ("clamp(2.2rem, 4.8vw, 3.8rem)", "clamp(1.15rem, 2.3vw, 1.6rem)"),
    "Larger": ("clamp(2.8rem, 6vw, 5rem)",     "clamp(1.3rem, 2.8vw, 2rem)"),
    "Huge":   ("clamp(4rem, 10vw, 9rem)",      "clamp(2rem, 4.8vw, 3.6rem)"),
}

# state -> (colour class, icon, title, hint)
STATES = {
    "off":     ("idle", "⏸", "Camera is off", "Turn on the camera above to start signing."),
    "warming": ("info", "⏳", "Getting ready…", "Keep your hand in view for a moment."),
    "nohand":  ("warn", "✋", "I can't see your hand", "Move your hand into the camera view."),
    "low":     ("info", "🤔", "Not sure yet", "Hold the sign steady, or try it again."),
    "nosign":  ("idle", "👋", "Waiting for a sign", "Your hand isn't making one of the known signs."),
    "holding": ("info", "⏱", "Hold steady…", "Keep the sign still until it is accepted."),
    "ok":      ("ok",   "✅", "Got it!", "Added to the screen."),
    "nomodel": ("warn", "🧠", "Hand tracking works – no model yet",
                "Train a model to recognise signs, or try the demo buttons in Settings."),
    "mismatch": ("err", "⚠️", "Model and sign list don't match",
                 "The trained model has a different number of signs than config.py. "
                 "Run  python restore_signs.py  or retrain."),
    "camfail": ("err",  "📷", "Camera not available",
                "Close other apps using the camera, then switch the camera off and on again."),
}


# ------------------------------------------------------------------ cached resources
@st.cache_resource
def get_hands():
    return make_hands()


@st.cache_resource
def get_model():
    """Returns (model or None, error text or None)."""
    if not (config.MODEL_PATH.exists() and config.META_PATH.exists()):
        return None, None
    try:
        from tensorflow import keras
        return keras.models.load_model(config.MODEL_PATH), None
    except Exception as e:  # pragma: no cover - depends on the user's TensorFlow install
        return None, f"The model file could not be loaded ({type(e).__name__}). Try training it again."


class PredictionLogger:
    def __init__(self):
        config.LOG_DIR.mkdir(exist_ok=True)
        self.path = config.LOG_DIR / f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["condition", "t_capture", "t_pred", "label", "conf", "committed", "cand_start"])
        self.n = 0

    def log(self, condition, t_cap, t_pred, label, conf, commit):
        self.w.writerow([condition, f"{t_cap:.6f}", f"{t_pred:.6f}", label, f"{conf:.4f}",
                         commit["label"] if commit else "",
                         f"{commit['cand_start']:.6f}" if commit else ""])
        self.n += 1
        if self.n % 30 == 0:
            self.f.flush()


@st.cache_resource
def get_logger():
    return PredictionLogger()


# ------------------------------------------------------------------ camera (released when not in use)
def release_camera():
    cap = ss.pop("_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def get_cap():
    cap = ss.get("_cap")
    if cap is None or not cap.isOpened():
        release_camera()
        cap = open_camera()
        ss["_cap"] = cap
    return cap


_STRETCH_OK = None


def show_frame(placeholder, frame):
    """Full-width image that works on both older and newer Streamlit."""
    global _STRETCH_OK
    if _STRETCH_OK is None:
        try:
            placeholder.image(frame, channels="BGR", width="stretch")
            _STRETCH_OK = True
            return
        except Exception:
            _STRETCH_OK = False
    if _STRETCH_OK:
        placeholder.image(frame, channels="BGR", width="stretch")
    else:
        placeholder.image(frame, channels="BGR", use_container_width=True)


# ------------------------------------------------------------------ transcript
def add_signed(text, sim=False):
    ss.signer_log.append({"text": text, "sim": sim, "t": datetime.now().strftime("%H:%M")})


def clear_all():
    ss.signer_log = []


def repeat_last():
    if ss.signer_log:
        ss.pending_say = ss.signer_log[-1]["text"]


def simulate(sign):
    text = signs_ref.pretty(sign)
    add_signed(text, sim=True)
    if ss.get("speak"):
        ss.pending_say = text


def display_html(size):
    now_sz, sent_sz = SIZES.get(size, SIZES["Larger"])
    log = ss.signer_log
    if not log:
        body = ('<div class="now empty"><span class="wave">👋</span>Turn on the camera and start signing.<br>'
                'Your words will appear here, big enough for everyone to read.</div>')
        return f'<div class="display" style="--now-size:{now_sz};--sent-size:{sent_sz}">{body}</div>'
    last = log[-1]
    now = (f'<div class="now"><b>{html.escape(last["text"].capitalize())}</b></div>')
    older = log[-8:-1]
    parts = []
    if len(log) > 8:
        parts.append('<span class="old">…</span>')
    parts += [f'<span class="old">{html.escape(e["text"])}</span>' for e in older]
    demo = "<small>demo</small>" if last["sim"] else ""
    parts.append(f'<span><mark>{html.escape(last["text"])}</mark>{demo}</span>')
    return (f'<div class="display" style="--now-size:{now_sz};--sent-size:{sent_sz}">'
            f'<div class="k">Now saying</div>{now}'
            f'<div class="sentence">{"".join(parts)}</div></div>')


def get_state(model, detected, label, conf, thr, n_frames, stab):
    if not detected:
        return "nohand"
    if model is None:
        return "nomodel"
    if n_frames < config.SEQ_LEN:
        return "warming"
    if label in (config.UNKNOWN, NO_HAND):
        return "nosign" if conf >= thr else "low"
    if conf < thr:
        return "low"
    if not (stab.last_label == label and not stab.released):
        return "holding"          # confident, but the stabilizer has not accepted it yet
    return "ok"


# ------------------------------------------------------------------ page
def render():
    model, model_err = get_model()
    signs = signs_ref.all_signs()         # the model's own labels (falls back to config.SIGNS)
    running = bool(ss.get("running"))
    if not running:
        release_camera()                      # camera light off whenever the switch is off
    ss.setdefault("signer_log", [])

    pill = ('<span class="pill live"><span class="dot"></span>Camera on</span>' if running
            else '<span class="pill off"><span class="dot"></span>Camera off</span>')
    nav.topbar("Signer", "Turn on the camera and sign. Your words appear on the screen once recognized.", pill)

    with st.container(key="card_toolbar"):
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2.6, 1.1, 1.3], vertical_alignment="center")
        c1.toggle("🎥 Camera", key="running", help="Turn the camera on to start recognising your signs.")
        c2.toggle("🔊 Speak aloud", key="speak", help="Say each recognised sign out loud.")
        with c3:
            st.segmented_control("Text size", list(SIZES), default="Larger", key="size",
                                 help="How big the words look on screen.")
        c4.button("🗑 Clear", on_click=clear_all, help="Remove all words from the screen.")
        c5.button("🔁 Repeat", on_click=repeat_last, help="Say the last sign out loud again.")

    if model is None:
        msg = model_err or ("No trained model yet. Hand tracking works, but signs are not recognised until "
                            "you run collect_data.py and train.py. You can try the demo buttons in Settings.")
        st.markdown(theme.banner("warn", "🧠", "No sign model available", html.escape(msg)),
                    unsafe_allow_html=True)

    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(key="card_camera"):
            st.markdown('<div class="sec">Your camera</div>', unsafe_allow_html=True)
            video_ph = st.empty()
            status_ph = st.empty()
            conf_ph = st.empty()
    with right:
        with st.container(key="card_display"):
            disp_ph = st.empty()

    with st.expander("⚙️ Settings and demo mode"):
        st.slider("Confidence needed", 0.50, 0.99, config.CONF_THRESHOLD, 0.01, key="conf_thr",
                  help="Higher = fewer mistakes, but you may need to repeat signs more often.")
        st.slider("Steadiness needed", 1, 15, config.STABLE_PREDICTIONS, key="n_stable",
                  help="How many matching readings in a row before a sign is accepted.")
        st.text_input("Experiment label (saved in the logs)", value="lab", key="condition")
        st.markdown("**Try it without a camera** (simulated, not real recognition)")
        cols = st.columns(max(len(signs_ref.vocabulary()), 1))
        for col, s in zip(cols, signs_ref.vocabulary()):
            col.button(signs_ref.pretty(s), key=f"sim_{s}", on_click=simulate, args=(s,))

    speaker_ph = st.empty()               # invisible: hosts the browser voice
    size = ss.get("size") or "Larger"

    # feedback when the voice switch is turned on, and for Repeat / demo buttons
    if ss.get("speak") and not ss.get("_prev_speak"):
        ss.pending_say = "Voice is on"
    ss["_prev_speak"] = bool(ss.get("speak"))
    if ss.get("pending_say"):
        theme.say(speaker_ph, ss.pop("pending_say"))

    disp_ph.markdown(display_html(size), unsafe_allow_html=True)

    def show_state(state, label, conf, thr, hint=None):
        cls, ic, title, default_hint = STATES[state]
        status_ph.markdown(theme.banner(cls, ic, title, hint or default_hint), unsafe_allow_html=True)
        if state in ("off", "camfail", "mismatch"):
            conf_ph.empty()
        elif state == "nosign":
            conf_ph.progress(min(max(conf, 0.0), 1.0), text=f"{conf:.0%} sure this is not a sign")
        else:
            conf_ph.progress(min(max(conf, 0.0), 1.0), text=f"Confidence {conf:.0%} · needs {thr:.0%}")

    if not running:
        video_ph.markdown(
            '<div style="aspect-ratio:4/3;border-radius:18px;background:linear-gradient(135deg,#EEF4F8,#E4EEF3);'
            'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.4rem;'
            'color:#4C5B68;text-align:center;padding:1rem;border:2px dashed #C6D6DF">'
            '<div style="font-size:3rem">📷</div><b style="font-size:1.1rem;color:#23323F">Camera is off</b>'
            '<span>Switch on “🎥 Camera” above to begin.</span></div>', unsafe_allow_html=True)
        show_state("off", NO_HAND, 0.0, ss.conf_thr)
        return

    # ------------------------------------------------------------------ live loop (recognition unchanged)
    hands, logger = get_hands(), get_logger()
    cap = get_cap()
    stab = Stabilizer(conf_thr=ss.conf_thr, n_stable=ss.n_stable)
    window = deque(maxlen=config.SEQ_LEN)
    n_frames, label, conf, detected = 0, NO_HAND, 0.0, False
    show_state("warming", NO_HAND, 0.0, ss.conf_thr)

    while ss.running:
        frame = grab_frame(cap)
        if frame is None:
            show_state("camfail", NO_HAND, 0.0, ss.conf_thr)
            release_camera()
            break
        t_cap = time.perf_counter()
        feats, detected, res = extract_features(frame, hands)
        draw_landmarks(frame, res)
        window.append(feats)
        n_frames += 1

        if model is not None and len(window) == config.SEQ_LEN and n_frames % config.PREDICT_EVERY == 0:
            arr = np.stack(window)
            hand_frac = float((np.abs(arr).sum(1) > 0).mean())
            if hand_frac < config.MIN_HAND_FRACTION:
                label, conf = NO_HAND, 0.0
            else:
                probs = model(arr[None], training=False).numpy()[0]
                if len(probs) != len(signs):
                    show_state("mismatch", NO_HAND, 0.0, ss.conf_thr)
                    release_camera()
                    break
                k = int(probs.argmax())
                label, conf = signs[k], float(probs[k])
            t_pred = time.perf_counter()
            commit = stab.update(label, conf, t_pred, t_cap)
            logger.log(ss.condition, t_cap, t_pred, label, conf, commit)
            if commit:
                text = signs_ref.pretty(commit["label"])
                add_signed(text)
                disp_ph.markdown(display_html(size), unsafe_allow_html=True)
                if ss.get("speak"):
                    theme.say(speaker_ph, text)

        show_frame(video_ph, frame)
        if n_frames % 3 == 0:
            state = get_state(model, detected, label, conf, ss.conf_thr, n_frames, stab)
            hint = (f"Same sign steady for {stab.cand_count} of {ss.n_stable} readings. Keep holding it."
                    if state == "holding" else None)
            show_state(state, label, conf, ss.conf_thr, hint)
