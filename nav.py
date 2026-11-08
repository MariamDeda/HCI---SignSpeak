"""Tiny page router helpers (the app keeps the current page in session_state and the URL)."""
import streamlit as st

import theme

PAGES = ("home", "signer", "hearing")


def go(page):
    """Button callback: switch page. Also updates ?page=... so a browser refresh stays on the page."""
    st.session_state.page = page
    if page == "home":
        st.query_params.clear()
    else:
        st.query_params["page"] = page


def topbar(title, subtitle, pill_html=""):
    """Back button + page title (+ optional status pill) shown at the top of the role pages."""
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([1.6, 6, 1.9], vertical_alignment="center", gap="medium")
        c1.button("← Change role", on_click=go, args=("home",), key="back_home",
                  help="Go back and pick a different role")
        c2.markdown(f'<div class="topbar-title"><div class="page-title">{title}</div>'
                    f'<div class="page-sub">{subtitle}</div></div>', unsafe_allow_html=True)
        if pill_html:
            c3.markdown(f'<div class="pillrow">{pill_html}</div>', unsafe_allow_html=True)
