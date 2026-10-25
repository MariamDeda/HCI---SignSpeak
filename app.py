"""SignSpeak Phase 1 prototype (Streamlit).   Run:  streamlit run app.py

REAL in this prototype : webcam, MediaPipe skeleton overlay, LSTM sign recognition,
                         confidence + low-confidence state, stabilization, logging.
PARTLY REAL            : hearing-person reply (typed, or Vosk 'Listen' if installed),
                         optional text-to-speech (pyttsx3).
MOCK                   : 'Simulated sign' buttons (Wizard-of-Oz) - clearly labelled.
"""
import csv
import html
import time
from collections import deque
from datetime import datetime

import numpy as np
import streamlit as st

import config
import speech
from features import draw_landmarks, extract_features, grab_frame, make_hands, open_camera
from stabilizer import NO_HAND, Stabilizer

st.set_page_config(page_title="SignSpeak", page_icon="🤟", layout="wide")

# ------------------------------------------------------------------ visual design
# Palette: teal = brand/actions, indigo = the signer, amber = the hearing person.
# Status colours are always paired with an icon + words (never colour alone).
st.markdown("""
<style>
:root{
  --ink:#1B2430; --muted:#5B6675; --line:#A17767; --card:#DB8AEB;
  --brand:#0F766E; --brand-soft:#D7F0EC;
  --signer:#4338CA; --signer-soft:#EAEBFF;
  --hearing:#B4AB09; --hearing-soft:#FCFFD9;
  --ok:#166534; --ok-soft:#DCF5E4;
  --warn:#920E47; --warn-soft:#FECFE3;
  --info:#1E4E8C; --info-soft:#E3EEFB;
  --idle:#475467; --idle-soft:#EAEEF4;
  --err:#B42318; --err-soft:#FDE4E1;
  /* card backgrounds */
  --panel:#FFFFFF;           /* Camera, Conversation, Detected-sign box, top toolbar */
  --panel-border:#E3E8EF;    /* outline of those cards */
  --side-item:#FFFFFF;       /* sidebar: How to use, Recognition settings, Demo mode */
}
html, body, [class*="css"]{font-family:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;}
.block-container{padding-top:3.6rem; padding-bottom:2rem; max-width:1280px;}
footer, #MainMenu{visibility:hidden;}

/* card backgrounds (edit --panel / --side-item above) */
.st-key-toolbar_card, .st-key-camera_card, .st-key-sign_box, .st-key-conversation_card{
  background:var(--panel) !important; border-color:var(--panel-border) !important;}
[data-testid="stSidebar"] [data-testid="stExpander"] details{background:var(--side-item); border-radius:.6rem;}

/* header */
.ss-title{font-size:2rem; font-weight:800; letter-spacing:-.02em; color:var(--ink); line-height:1.1;}
.ss-title span{color:var(--brand);}
.ss-sub{color:var(--muted); font-size:1rem; margin-top:.25rem;}
.pill{display:inline-flex; align-items:center; gap:.45rem; padding:.35rem .9rem; border-radius:999px;
      font-weight:700; font-size:.95rem; float:right; margin-top:.4rem;}
.pill.live{background:var(--ok-soft); color:var(--ok);}
.pill.off{background:var(--idle-soft); color:var(--idle);}
.dot{width:.6rem; height:.6rem; border-radius:50%; background:currentColor; display:inline-block;}
.pill.live .dot{animation:pulse 1.4s infinite;}
@keyframes pulse{0%{opacity:1}50%{opacity:.3}100%{opacity:1}}

/* section titles */
.sec{font-size:.8rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase;
     color:var(--muted); margin:0 0 .6rem 0;}

/* camera placeholder + status banner */
.cam-off{aspect-ratio:4/3; border-radius:14px; background:linear-gradient(135deg,#EEF2F7,#E3E9F1);
         display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.4rem;
         color:var(--idle); text-align:center; padding:1rem; border:2px dashed #C9D2DE;}
.cam-off .big-ic{font-size:3rem;}
.cam-off b{font-size:1.15rem; color:var(--ink);}
.banner{display:flex; align-items:center; gap:.8rem; padding:.8rem 1rem; border-radius:12px; margin-top:.8rem;}
.banner .ic{font-size:1.6rem; line-height:1;}
.banner .t{font-weight:800; font-size:1.1rem; line-height:1.2;}
.banner .h{font-size:.95rem; opacity:.95;}
.banner.ok{background:var(--ok-soft); color:var(--ok);}
.banner.warn{background:var(--warn-soft); color:var(--warn);}
.banner.info{background:var(--info-soft); color:var(--info);}
.banner.idle{background:var(--idle-soft); color:var(--idle);}
.banner.err{background:var(--err-soft); color:var(--err);}

/* big detected-sign card */
.sign-card{border-radius:14px; padding:1.1rem 1.2rem 1.3rem; margin-bottom:.5rem; text-align:center; border:2px solid transparent;}
.sign-card .k{font-size:.8rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; opacity:.85;}
.sign-card .v{font-size:3rem; font-weight:800; line-height:1.15; margin:.15rem 0 .1rem; letter-spacing:-.01em;
              word-break:break-word;}
.sign-card.ok{background:var(--ok-soft); color:var(--ok); border-color:#9BDDB1;}
.sign-card.warn{background:var(--warn-soft); color:var(--warn);}
.sign-card.info{background:var(--info-soft); color:var(--info);}
.sign-card.idle{background:var(--idle-soft); color:var(--idle);}
.sign-card.err{background:var(--err-soft); color:var(--err);}

/* conversation bubbles */
.chat{display:flex; flex-direction:column; gap:.55rem; min-height:7rem; margin-bottom:1rem;}
.row{display:flex;}
.row.hearing{justify-content:flex-end;}
.bubble{max-width:88%; padding:.55rem .9rem; border-radius:16px; font-size:1.25rem; line-height:1.35; color:var(--ink);}
.bubble .who{display:block; font-size:.72rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; margin-bottom:.1rem;}
.bubble .time{font-size:.72rem; color:var(--muted); font-weight:500; margin-left:.4rem; letter-spacing:0; text-transform:none;}
.bubble.signer{background:var(--signer-soft); border-bottom-left-radius:4px;}
.bubble.signer .who{color:var(--signer);}
.bubble.hearing{background:var(--hearing-soft); border-bottom-right-radius:4px;}
.bubble.hearing .who{color:var(--hearing);}
.tag-sim{font-size:.68rem; font-weight:800; background:#fff; border:1px solid var(--line); color:var(--muted);
         border-radius:6px; padding:0 .35rem; margin-left:.4rem; vertical-align:middle;}
.empty{color:var(--muted); text-align:center; padding:1.2rem .5rem; font-size:1rem;
       border:2px dashed var(--line); border-radius:12px;}
.legend{font-size:.85rem; color:var(--muted); margin-bottom:.5rem;}
.legend i{display:inline-block; width:.7rem; height:.7rem; border-radius:3px; margin:0 .25rem 0 .6rem; vertical-align:-1px;}

/* guide + steps */
.step{display:flex; gap:.7rem; align-items:flex-start; margin:.35rem 0;}
.step .n{flex:0 0 1.7rem; height:1.7rem; border-radius:50%; background:var(--brand); color:#fff;
         display:flex; align-items:center; justify-content:center; font-weight:800; font-size:.9rem;}
.step .tx{font-size:1rem; color:var(--ink);}
.step .tx b{color:var(--brand);}

/* controls */
.stButton>button, .stFormSubmitButton>button{border-radius:10px; font-weight:700; min-height:2.6rem;}
.stFormSubmitButton>button[kind="primary"], .stButton>button[kind="primary"]{background:var(--brand); border-color:var(--brand);}
.stButton>button:focus-visible, .stFormSubmitButton>button:focus-visible, input:focus-visible{outline:3px solid #7DD3C8 !important; outline-offset:2px;}
[data-testid="stSidebar"]{background:#FFFFFF; border-right:1px solid var(--line);}
[data-testid="stMetricValue"]{font-weight:800;}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("conversation", [])      # [(speaker, text, "HH:MM")]


# ------------------------------------------------------------------ cached resources
@st.cache_resource
def get_camera():
    return open_camera()


@st.cache_resource
def get_hands():
    return make_hands()


@st.cache_resource
def get_model():
    if not (config.MODEL_PATH.exists() and config.META_PATH.exists()):
        return None
    from tensorflow import keras
    return keras.models.load_model(config.MODEL_PATH)


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


# ------------------------------------------------------------------ helpers
SIM_TAG = " [simulated]"


def pretty(label):
    return config.DISPLAY.get(label, label.replace("_", " "))


def add_message(who, text):
    ss.conversation.append((who, text, datetime.now().strftime("%H:%M")))


def clear_all():
    ss.conversation = []


def simulate(sign):
    add_message("signer", pretty(sign) + SIM_TAG)


def repeat_last():
    for who, text, _ in reversed(ss.conversation):
        if who == "signer":
            speech.speak(text.replace(SIM_TAG, ""))
            break


_STRETCH_OK = None


def show_frame(placeholder, frame):
    """Full-width image that works on both old (use_container_width) and new (width) Streamlit."""
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


# state -> (colour class, icon, title, hint)
STATES = {
    "off":     ("idle", "⏸", "Waiting to start", "Nothing is being recorded or recognised."),
    "warming": ("info", "⏳", "Getting ready…", "Keep your hand in view for a moment."),
    "nohand":  ("warn", "✋", "No hand detected", "Move your hand into the camera view."),
    "low":     ("info", "🤔", "Not sure yet", "Hold the sign steady, or repeat it."),
    "nosign":  ("idle", "👋", "No sign detected", "Your hand isn't making one of the known signs."),
    "ok":      ("ok",   "✅", "Sign recognized", "Added to the conversation."),
    "nomodel": ("warn", "🧠", "Hand tracking works – no model yet",
                "Record data and train a model to recognise signs (see the guide)."),
    "camfail": ("err",  "📷", "Camera not available",
                "Close other apps using the camera (Zoom, Teams…), then switch this off and on again."),
}


def get_state(model, detected, label, conf, thr, n_frames):
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
    return "ok"


def banner_html(state):
    cls, ic, title, hint = STATES[state]
    return (f'<div class="banner {cls}" role="status" aria-live="polite"><span class="ic">{ic}</span>'
            f'<div><div class="t">{title}</div><div class="h">{hint}</div></div></div>')


def sign_card_html(state, label):
    cls = STATES[state][0]
    value = pretty(label) if state == "ok" else "–"
    return (f'<div class="sign-card {cls}"><div class="k">Detected sign</div>'
            f'<div class="v">{html.escape(value)}</div></div>')


# ------------------------------------------------------------------ sidebar (settings & help)
with st.sidebar:
    st.markdown('<div class="ss-title" style="font-size:1.4rem">Sign<span>Speak</span></div>',
                unsafe_allow_html=True)
    st.caption("Settings & help")

    with st.expander("📖 How to use", expanded=False):
        st.markdown(
            '<div class="step"><div class="n">1</div><div class="tx">Switch on <b>Camera &amp; recognition</b>.</div></div>'
            '<div class="step"><div class="n">2</div><div class="tx">Show one sign clearly and <b>hold it steady</b> for about a second.</div></div>'
            '<div class="step"><div class="n">3</div><div class="tx">Take your hand out of view between signs.</div></div>'
            '<div class="step"><div class="n">4</div><div class="tx">The other person <b>types or speaks</b> a reply on the right.</div></div>',
            unsafe_allow_html=True)
        st.caption("Signs it knows: " + ", ".join(pretty(s) for s in config.SIGNS if s != config.UNKNOWN))

    with st.expander("⚙️ Recognition settings", expanded=False):
        st.slider("Confidence needed", 0.50, 0.99, config.CONF_THRESHOLD, 0.01, key="conf_thr",
                  help="Higher = fewer mistakes, but you may need to repeat signs more often.")
        st.slider("Steadiness needed", 1, 15, config.STABLE_PREDICTIONS, key="n_stable",
                  help="How many matching readings in a row before a sign is accepted. "
                       "Higher = more reliable but a little slower.")
        st.text_input("Experiment label (for logs)", value="lab", key="condition",
                      help="Saved with every prediction, e.g. 'lab' or 'low light'.")

    with st.expander("🧪 Demo mode (simulated)", expanded=False):
        st.caption("MOCK – these buttons add a sign to the chat without using the camera. "
                   "They are marked “simulated”.")
        for s in config.SIGNS:
            if s != config.UNKNOWN:
                st.button(pretty(s), key=f"sim_{s}", on_click=simulate, args=(s,))

# ------------------------------------------------------------------ header + toolbar
model = get_model()
running = bool(ss.get("running"))

h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown('<div class="ss-title">Sign<span>Speak</span></div>'
                '<div class="ss-sub">Sign language to text, live – so everyone in the conversation is understood.</div>',
                unsafe_allow_html=True)
with h_right:
    pill = ('<span class="pill live"><span class="dot"></span>Live</span>' if running
            else '<span class="pill off"><span class="dot"></span>Camera off</span>')
    st.markdown(pill, unsafe_allow_html=True)

with st.container(border=True, key="toolbar_card"):
    t1, t2, t3, t4 = st.columns([2.2, 2.2, 1, 1.2], vertical_alignment="center")
    t1.toggle("🎥 Camera & recognition", key="running",
              help="Turn the camera on to start recognising signs.")
    t2.toggle("🔊 Read signs aloud", key="speak",
              help="Speak each recognised sign out loud (needs pyttsx3).")
    t3.button("🗑 Clear", on_click=clear_all, help="Remove all messages from the conversation.")
    t4.button("🔁 Repeat last", on_click=repeat_last, help="Speak the last recognised sign again.")

if model is None:
    st.info("**No trained model yet.** The camera and hand tracking work, but signs aren't recognised "
            "until you run `python collect_data.py` and then `python train.py`. "
            "You can still try the chat, or use *Demo mode* in the sidebar.", icon="ℹ️")

# ------------------------------------------------------------------ main layout
left, right = st.columns([3, 2], gap="large")
with left:
    with st.container(border=True, key="camera_card"):
        st.markdown('<div class="sec">Camera</div>', unsafe_allow_html=True)
        video_ph = st.empty()
        status_ph = st.empty()

with right:
    with st.container(border=True, key="sign_box"):
        sign_ph = st.empty()
        conf_ph = st.empty()
    with st.container(border=True, key="conversation_card"):
        st.markdown('<div class="sec">Conversation</div>', unsafe_allow_html=True)
        st.markdown('<div class="legend"><i style="background:var(--signer)"></i>Signer'
                    '<i style="background:var(--hearing)"></i>Hearing person</div>', unsafe_allow_html=True)
        conv_ph = st.empty()
        with st.form("reply_form", clear_on_submit=True, border=False):
            f1, f2 = st.columns([3, 1.3], vertical_alignment="bottom")
            reply = f1.text_input("Reply as the hearing person", key="reply_box",
                                  placeholder="Type a reply and press Enter…")
            sent = f2.form_submit_button("Send", type="primary")
        if st.button("🎤 Speak a reply (5 s)", help="Records 5 seconds from your microphone and types what it hears."):
            try:
                with st.spinner("Listening… speak now"):
                    heard = speech.listen()
                add_message("hearing", heard if heard else "(nothing heard)")
            except RuntimeError as e:
                st.warning(str(e))
        if sent and reply.strip():
            add_message("hearing", reply.strip())


def render_conversation():
    rows = []
    for who, text, ts in ss.conversation[-6:]:
        is_sim = text.endswith(SIM_TAG)
        body = html.escape(text.replace(SIM_TAG, ""))
        badge = '<span class="tag-sim">simulated</span>' if is_sim else ""
        name = "Signer" if who == "signer" else "Hearing"
        cls = "signer" if who == "signer" else "hearing"
        rows.append(f'<div class="row {cls}"><div class="bubble {cls}">'
                    f'<span class="who">{name}<span class="time">{ts}</span></span>{body}{badge}</div></div>')
    if rows:
        conv_ph.markdown('<div class="chat">' + "".join(rows) + '</div>', unsafe_allow_html=True)
    else:
        conv_ph.markdown('<div class="empty">💬 No messages yet.<br>Signs you make appear here, '
                         'and the other person can type a reply below.</div>', unsafe_allow_html=True)


def show_state(state, label, conf, thr):
    status_ph.markdown(banner_html(state), unsafe_allow_html=True)
    sign_ph.markdown(sign_card_html(state, label), unsafe_allow_html=True)
    if state in ("off", "camfail"):
        conf_ph.empty()
    else:
        if state == "nosign":
            txt = f"{conf:.0%} sure this is NOT one of the signs"
        else:
            txt = f"Confidence {conf:.0%}  ·  needs {thr:.0%} to be accepted"
        conf_ph.progress(min(max(conf, 0.0), 1.0), text=txt)


render_conversation()
if not ss.get("running"):
    video_ph.markdown('<div class="cam-off"><div class="big-ic">📷</div><b>Camera is off</b>'
                      '<span>Switch on “Camera &amp; recognition” above to start.</span></div>',
                      unsafe_allow_html=True)
    show_state("off", NO_HAND, 0.0, ss.conf_thr)

# ------------------------------------------------------------------ live loop
if ss.get("running"):
    cap, hands, logger = get_camera(), get_hands(), get_logger()
    stab = Stabilizer(conf_thr=ss.conf_thr, n_stable=ss.n_stable)
    window = deque(maxlen=config.SEQ_LEN)
    n_frames, label, conf, detected = 0, NO_HAND, 0.0, False
    show_state("warming", NO_HAND, 0.0, ss.conf_thr)

    while ss.running:
        frame = grab_frame(cap)
        if frame is None:
            show_state("camfail", NO_HAND, 0.0, ss.conf_thr)
            break
        t_cap = time.perf_counter()          # frame available in software
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
                k = int(probs.argmax())
                label, conf = config.SIGNS[k], float(probs[k])
            t_pred = time.perf_counter()
            commit = stab.update(label, conf, t_pred, t_cap)
            logger.log(ss.condition, t_cap, t_pred, label, conf, commit)
            if commit:
                text = pretty(commit["label"])
                add_message("signer", text)
                render_conversation()
                if ss.get("speak"):
                    speech.speak(text)

        show_frame(video_ph, frame)
        if n_frames % 3 == 0:
            show_state(get_state(model, detected, label, conf, ss.conf_thr, n_frames),
                       label, conf, ss.conf_thr)