"""Landing page: who is using SignSpeak right now?"""
import streamlit as st

import nav
import signs_ref


def render():
    st.markdown(
        '<div class="hero"><div class="logo">Sign<span>Speak</span></div>'
        '<div class="tag">A bridge between sign language and spoken words.</div></div>'
        '<div class="ask">Who is using SignSpeak right now?</div>',
        unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(key="card_role_signer"):
            st.markdown(
                '<div class="role-ic sky">🤟</div>'
                '<div class="role-t">Signer</div>'
                '<div class="role-d">You want people to understand your ASL.</div>'
            )
            st.button("Start signing  →", key="go_signer", type="primary",
                      on_click=nav.go, args=("signer",))
    with right:
        with st.container(key="card_role_hearing"):
            st.markdown(
                '<div class="role-ic mint">💬</div>'
                '<div class="role-t">Speaker</div>'
                '<div class="role-d">You want to talk with someone who signs.</div>'
            )
            st.button("Start talking  →", key="go_hearing", type="primary",
                      on_click=nav.go, args=("hearing",))

    chips = "".join(f'<span class="chip">{signs_ref.info(s)["emoji"]} {signs_ref.pretty(s)}</span>'
                    for s in signs_ref.vocabulary())
    st.markdown(
        '<div class="foot">You can switch roles at any time with the <b>← Change role</b> button.'
        '<br>Prototype knows these signs currently</div>'
        f'<div class="chips">{chips}</div>',
        unsafe_allow_html=True)
