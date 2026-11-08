"""SignSpeak Phase 1 prototype (Streamlit).   Run:  python -m streamlit run app.py

Pages (all in this one app; the choice is kept in the URL as ?page=...):
  home     - "Who is using SignSpeak?"  (two roles)
  signer   - view_signer.py   deaf / mute person: camera -> big text (+ optional voice)
  hearing  - view_hearing.py  person who doesn't know ASL: type or speak -> big sign

REAL   : webcam, MediaPipe hand tracking, LSTM sign recognition, stabilization, logging.
PARTLY : voice input (Vosk), browser text-to-speech.
MOCK   : demo buttons in Settings ("simulated", clearly labelled).
"""
import streamlit as st

st.set_page_config(page_title="SignSpeak", page_icon="🤟", layout="wide", initial_sidebar_state="collapsed")

import nav          # noqa: E402
import theme        # noqa: E402

theme.inject()
ss = st.session_state

# first load: restore the page from the URL (?page=signer) so a browser refresh stays put
if "page" not in ss:
    wanted = st.query_params.get("page", "home")
    ss.page = wanted if wanted in nav.PAGES else "home"

if ss.page == "signer":
    import view_signer
    view_signer.render()
else:
    # leaving the signer page (or never being on it): make sure the camera is released
    cap = ss.pop("_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    if ss.page == "hearing":
        import view_hearing
        view_hearing.render()
    else:
        import view_home
        view_home.render()
