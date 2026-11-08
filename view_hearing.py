"""Role 2 - the person who does not know ASL.

They type, tap a word, or speak; the message is mapped to signs and shown big, slowly and on
repeat for the signer. All playback happens in the browser, so the page does not flicker.
"""
import hashlib
import html

import streamlit as st

import nav
import signs_ref
import speech
import theme
import voice

ss = st.session_state


def _fmt(keys):
    return " → ".join(signs_ref.pretty(k) for k in keys)


def process_text(text, source="typed"):
    """Map a message to signs and store the result + a plain-language feedback message."""
    text = (text or "").strip()
    ss.h_nonce = ss.get("h_nonce", 0) + 1
    if not text:
        ss.h_signs, ss.h_unknown, ss.h_text = [], [], ""
        ss.h_feedback = ("warn", "✍️", "Nothing to show yet", "Type a message, speak, or tap a word.")
        return
    keys, unknown = signs_ref.map_text(text)
    ss.h_signs, ss.h_unknown, ss.h_text, ss.h_source = keys, unknown, text, source
    vocab = ", ".join(signs_ref.pretty(s) for s in signs_ref.vocabulary())
    if keys:
        n = len(keys)
        extra = ""
        if unknown:
            extra = " No sign yet for: " + html.escape(", ".join(unknown)) + "."
        ss.h_feedback = ("ok", "✅", f"Showing {n} sign{'s' if n != 1 else ''}: {html.escape(_fmt(keys))}",
                         "The signer can see it below." + extra)
        hist = [t for t in ss.get("h_history", []) if t.lower() != text.lower()]
        ss.h_history = ([text] + hist)[:5]
    else:
        ss.h_feedback = ("warn", "🤷", "I don't have a sign for that yet",
                         f"For now I know: {html.escape(vocab)}. Try one of those words.")


def quick(sign):
    ss.h_input = signs_ref.pretty(sign)          # show the tapped word in the box
    process_text(signs_ref.pretty(sign), source="tapped")


def replay(text):
    ss.h_input = text
    process_text(text, source="recent")


def _offline_recording():
    """Vosk fallback (works without internet, needs the Vosk model once)."""
    st.caption("Use this if the microphone button above does not work in your browser. "
               "It needs the 'vosk' package and the Vosk model in the models folder.")
    if not st.toggle("Turn on offline recording", key="h_offline"):
        return
    audio = st.audio_input("Press the mic, say your message, then press stop",
                           sample_rate=16000, key="h_audio")
    if audio is None:
        return
    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if digest == ss.get("h_audio_digest"):              # only handle each recording once
        return
    ss.h_audio_digest = digest
    try:
        with st.spinner("Understanding what you said…"):
            heard = speech.transcribe_wav(data)
    except RuntimeError as e:
        ss.h_feedback = ("err", "🎤", "Offline voice isn't set up", html.escape(str(e)))
    except Exception:
        ss.h_feedback = ("err", "🎤", "I couldn't read that recording",
                         "Please try again, or type your message instead.")
    else:
        if heard:
            ss.h_input = heard
            process_text(heard, "voice")
        else:
            ss.h_feedback = ("warn", "🎤", "I didn't catch that",
                             "Speak a little closer to the microphone and try again.")


def render():
    ss.setdefault("h_signs", [])
    ss.setdefault("h_feedback", None)
    ss.setdefault("h_nonce", 0)
    ss.setdefault("h_history", [])

    nav.topbar("Speaker",
               "Say or type your message. The signer will see it as signs, big and clear.")

    left, right = st.columns([5, 7], gap="large")

    # ---------------------------------------------------------------- step 1: say or type
    with left:
        with st.container(key="card_say"):
            st.markdown('<div class="sec"><span class="stepno">1</span>Say or type your message</div>',
                        unsafe_allow_html=True)
            with st.form("say_form", border=False):
                typed = st.text_input("Your message", key="h_input", label_visibility="collapsed",
                                      placeholder="Type here, e.g. “Hello, thank you!”")
                go = st.form_submit_button("Show sign  →", type="primary")
            if go:
                process_text(typed, "typed")

            if st.toggle("🎤 Use the microphone", key="h_mic",
                         help="Turn this on, tap the button and speak."):
                theme.embed(voice.mic_html(), height=170)
                with st.expander("Voice not working? Try offline recording"):
                    _offline_recording()

            st.markdown('<div class="sec" style="margin-top:.4rem">Or tap a word</div>', unsafe_allow_html=True)
            with st.container(key="quick_row"):
                vocab = signs_ref.vocabulary()
                per_row = 2
                for i in range(0, len(vocab), per_row):
                    cols = st.columns(per_row)
                    for col, s in zip(cols, vocab[i:i + per_row]):
                        col.button(f'{signs_ref.info(s)["emoji"]} {signs_ref.pretty(s)}', key=f"quick_{s}",
                                   on_click=quick, args=(s,))

        with st.container(key="card_recent"):
            st.markdown('<div class="sec">Recent messages</div>', unsafe_allow_html=True)
            if ss.h_history:
                for i, text in enumerate(ss.h_history):
                    label = text if len(text) <= 36 else text[:35] + "…"
                    st.button(f"↻  {label}", key=f"recent_{i}", on_click=replay, args=(text,),
                              help="Show this message again")
            else:
                st.markdown('<div class="page-sub">Messages you send appear here, so you can show them '
                            'again with one tap.</div>', unsafe_allow_html=True)
            st.markdown('<div class="tip">💡 Short sentences work best. Words that have no sign yet are '
                        'skipped, and you will see which ones.</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------- step 2: the sign
    with right:
        with st.container(key="card_stage"):
            st.markdown('<div class="sec"><span class="stepno">2</span>The signer sees this</div>',
                        unsafe_allow_html=True)
            fb = ss.get("h_feedback")
            if fb:
                st.markdown(theme.banner(*fb), unsafe_allow_html=True)
            if ss.h_signs:
                theme.embed(signs_ref.player_html(ss.h_signs, ss.h_nonce), height=570)
            else:
                st.markdown('<div class="empty-stage"><span class="wave">👋</span>'
                            'Your message will appear here as a sign.<br>'
                            'Type something, speak, or tap a word on the left.</div>',
                            unsafe_allow_html=True)
