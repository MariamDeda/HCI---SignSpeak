"""Look & feel for SignSpeak: light blue / mint green / sunny yellow / white / grey.

To recolour the app, edit the :root{...} block in CSS below (every colour is a variable).
Status colours are always paired with an icon and words, never colour alone.
"""
import json
import time

import streamlit as st
import streamlit.components.v1 as components

import config

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  /* text */
  --ink:#1B2A38;  --muted:#4F6377;  --line:#C6DCE8;
  /* surfaces */
  --card:#FFFFFF;  --bg:#E4F2FA;
  /* brand: teal-blue (buttons, links, focus) */
  --brand:#0B7F96;  --brand-dark:#08687B;  --brand-soft:#CBEAF2;
  /* roles: signer = blue, hearing person = green, highlights = yellow */
  --sky:#BFE4F8;   --sky-ink:#14577F;   --sky-strong:#46B1E8;
  --mint:#BDEBD3;  --mint-ink:#17694B;  --mint-strong:#35C590;
  --sun:#FFE27A;   --sun-ink:#6B4F00;   --sun-strong:#F5B800;
  /* status */
  --ok:#BDEBD3;    --ok-ink:#145C40;
  --info:#BFE4F8;  --info-ink:#14577F;
  --warn:#FFE27A;  --warn-ink:#6B4F00;
  --idle:#DAE5ED;  --idle-ink:#3E4F5E;
  --err:#FBCFC8;   --err-ink:#9A2F20;
}
html, body, [class*="css"], .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp button, .stApp input, .stApp textarea,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"]{
  font-family:"IBM Plex Sans","Segoe UI",system-ui,-apple-system,Arial,sans-serif !important;}
.stApp{
  background:
    radial-gradient(rgba(11,127,150,.10) 1.6px, transparent 1.7px) 0 0 / 28px 28px,
    radial-gradient(1000px 640px at 4% -8%, rgba(86,184,238,.70), transparent 64%),
    radial-gradient(900px 620px at 100% 0%, rgba(90,215,160,.60), transparent 64%),
    radial-gradient(900px 640px at 60% 115%, rgba(255,214,70,.55), transparent 64%),
    radial-gradient(700px 500px at -5% 105%, rgba(86,184,238,.40), transparent 64%),
    var(--bg);
  background-attachment:fixed;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"], footer, #MainMenu{display:none !important;}
.block-container{max-width:1120px; padding-top:1.8rem; padding-bottom:4rem; animation:rise .5s ease both;}
/* even vertical rhythm between the big blocks of a page */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]{gap:1.7rem;}
@keyframes rise{from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none}}
@media (prefers-reduced-motion:reduce){*{animation:none !important; transition:none !important;}}

/* ---------- cards: any container whose key starts with card_ ---------- */
div[class*="st-key-card_"]{
  --accent:var(--sky-strong);
  background:var(--card); border:1px solid var(--line); border-radius:20px;
  padding:1.3rem 1.3rem 1.2rem; gap:.9rem;
  box-shadow:inset 0 5px 0 0 var(--accent), 0 10px 26px rgba(20,60,90,.13);}
.st-key-card_toolbar, .st-key-card_recent{--accent:var(--sun-strong);}
.st-key-card_camera, .st-key-card_say{--accent:var(--sky-strong);}
.st-key-card_display, .st-key-card_stage, .st-key-card_role_hearing{--accent:var(--mint-strong);}
.st-key-card_display{min-height:26rem; justify-content:center;}
div[class*="st-key-card_role_"]{
  padding:1.8rem 1.6rem 1.5rem; transition:transform .25s ease, box-shadow .25s ease;}
div[class*="st-key-card_role_"]:hover{transform:translateY(-4px); box-shadow:inset 0 5px 0 0 var(--accent), 0 16px 34px rgba(20,60,90,.20);}
.st-key-card_role_signer{background:linear-gradient(160deg,#FFFFFF 35%,var(--sky) 135%);}
.st-key-card_role_hearing{background:linear-gradient(160deg,#FFFFFF 35%,var(--mint) 135%);}
.st-key-go_signer, .st-key-go_hearing, .st-key-go_signer button, .st-key-go_hearing button{width:100% !important;}
.st-key-go_signer button, .st-key-go_hearing button{min-height:2.8rem;}
.st-key-go_signer button p, .st-key-go_hearing button p{font-size:1rem; font-weight:600;}

/* ---------- typography ---------- */
.hero{text-align:center; margin:.8rem 0 .4rem;}
.hero .logo{font-size:2.6rem; font-weight:700; letter-spacing:-.02em; color:var(--ink); line-height:1;}
.hero .logo span{color:var(--brand);}
.hero .tag{font-size:1.1rem; color:var(--muted); margin-top:.7rem;}
.ask{text-align:center; font-size:1.3rem; font-weight:700; margin:.6rem 0 .4rem; color:var(--ink);}
.role-ic{width:4.6rem; height:4.6rem; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:2.5rem; line-height:1; margin-bottom:1rem; box-shadow:0 8px 20px rgba(20,60,90,.14);}
.role-ic.sky{background:var(--sky);} .role-ic.mint{background:var(--mint);}
.role-t{font-size:1.4rem; font-weight:700; letter-spacing:-.01em; margin:0 0 .45rem;}
.role-d{font-size:1rem; color:var(--muted); line-height:1.5; margin:0 0 .9rem; min-height:2.6rem;}
.role-list{margin:0 0 1.5rem; padding:0; list-style:none;}
.role-list li{padding:.4rem 0 .4rem 2.1rem; position:relative; font-size:1.05rem;}
.role-list li:before{content:"✓"; position:absolute; left:0; top:.4rem; width:1.5rem; height:1.5rem; border-radius:50%;
  background:var(--brand); color:#fff; font-weight:700; font-size:.85rem; display:flex; align-items:center; justify-content:center;}
.foot{text-align:center; color:var(--muted); font-size:1rem; margin-top:.6rem; line-height:1.6;}
.chips{display:flex; flex-wrap:wrap; gap:.6rem; justify-content:center; margin-top:.8rem;}
.chip{background:#fff; border:1.5px solid var(--line); border-radius:999px; padding:.35rem 1rem; font-weight:600; font-size:.95rem; color:var(--ink);
  box-shadow:0 4px 10px rgba(20,60,90,.08);}

.topbar-title{display:flex; flex-direction:column; justify-content:center; min-height:3rem;}
.page-title{font-size:1.65rem; font-weight:700; letter-spacing:-.02em; line-height:1.1; margin:0; color:var(--ink);}
.page-sub{color:var(--muted); font-size:.98rem; margin-top:.3rem; line-height:1.4;}
.sec{font-size:.82rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--brand-dark); margin:0 0 .9rem;}
.stepno{display:inline-flex; width:1.8rem; height:1.8rem; border-radius:50%; background:var(--brand); color:#fff;
  align-items:center; justify-content:center; font-size:.95rem; font-weight:700; margin-right:.6rem; letter-spacing:0;}
.tip{background:var(--sun); color:var(--sun-ink); border-radius:14px; padding:.7rem 1rem; font-size:.97rem; font-weight:600; margin-top:.4rem;}

/* ---------- pills + banners ---------- */
.pillrow{display:flex; justify-content:flex-end; align-items:center; min-height:3rem;}
.pill{display:inline-flex; align-items:center; gap:.5rem; padding:.5rem 1.1rem; border-radius:999px; font-weight:700; font-size:.98rem;}
.pill.live{background:var(--mint-strong); color:#06382A;}
.pill.off{background:var(--idle); color:var(--idle-ink);}
.dot{width:.65rem; height:.65rem; border-radius:50%; background:currentColor; display:inline-block;}
.pill.live .dot{animation:pulse 1.4s infinite;}
@keyframes pulse{0%{opacity:1}50%{opacity:.3}100%{opacity:1}}
.banner{display:flex; align-items:center; gap:.9rem; padding:.95rem 1.2rem; border-radius:18px; margin:0 0 .6rem; animation:rise .3s ease both;}
.banner .ic{font-size:1.8rem; line-height:1;}
.banner .t{font-weight:700; font-size:1.12rem; line-height:1.25;}
.banner .h{font-size:.98rem; opacity:.95; line-height:1.4;}
.banner.ok{background:var(--ok); color:var(--ok-ink);}
.banner.info{background:var(--info); color:var(--info-ink);}
.banner.warn{background:var(--warn); color:var(--warn-ink);}
.banner.idle{background:var(--idle); color:var(--idle-ink);}
.banner.err{background:var(--err); color:var(--err-ink);}

/* ---------- signer: big translation display ---------- */
.display{text-align:center; padding:.4rem .2rem .6rem;}
.display .k{font-size:.82rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--brand-dark);}
.display .now{font-size:var(--now-size,clamp(2.8rem,6vw,5rem)); font-weight:700; letter-spacing:-.03em; line-height:1.05;
  margin:.35rem 0 .6rem; color:var(--ink); overflow-wrap:anywhere; min-height:1.1em;}
.display .now b{display:inline-block; animation:pop .45s cubic-bezier(.2,1.4,.4,1) both;}
@keyframes pop{from{opacity:0; transform:scale(.88)} to{opacity:1; transform:none}}
.display .now.empty{font-size:clamp(1.15rem,2.2vw,1.5rem); font-weight:500; color:var(--muted); letter-spacing:0; line-height:1.45;}
.display .now.empty .wave{display:block; font-size:4.6rem; margin-bottom:.5rem; animation:float 2.6s ease-in-out infinite;}
@keyframes float{0%,100%{transform:translateY(0) rotate(-4deg)}50%{transform:translateY(-8px) rotate(4deg)}}
.sentence{font-size:var(--sent-size,clamp(1.5rem,3.2vw,2.4rem)); font-weight:600; line-height:1.5; color:var(--ink);
  border-top:3px dashed var(--line); padding-top:1rem; margin-top:.6rem;}
.sentence span{margin:0 .15em; white-space:nowrap;}
.sentence .old{color:#6E8294;}
.sentence mark{background:var(--sun); color:var(--ink); border-radius:.35em; padding:.02em .3em;}
.sentence small{font-size:.38em; font-weight:600; color:var(--muted); background:#fff; border:1px solid var(--line); border-radius:6px; padding:0 .4em; margin-left:.3em; vertical-align:middle;}

/* ---------- hearing ---------- */
.heard{background:var(--brand-soft); color:var(--ink); border-radius:14px; padding:.7rem 1rem; font-size:1.05rem;}
.heard b{color:var(--brand-dark);}
.empty-stage{text-align:center; padding:2.6rem 1rem; border:3px dashed #A9CBDC; border-radius:22px; color:var(--muted); font-size:1.02rem; line-height:1.6;
  background:linear-gradient(135deg,#E7F4FB,#E6F8EF);}
.empty-stage .wave{font-size:3.6rem; display:block; margin-bottom:.6rem; animation:float 2.6s ease-in-out infinite;}

/* ---------- controls ---------- */
.stButton > button, .stFormSubmitButton > button{
  border-radius:12px; font-weight:600; min-height:2.5rem; border:2px solid #A9CBDC; background:#fff; color:var(--ink);
  transition:transform .15s ease, box-shadow .15s ease, background .15s ease;}
.stButton > button:hover, .stFormSubmitButton > button:hover{transform:translateY(-1px); box-shadow:0 6px 16px rgba(20,60,90,.16); border-color:var(--brand); color:var(--ink);}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--brand) !important; border-color:var(--brand) !important; color:#fff !important; box-shadow:0 6px 16px rgba(11,127,150,.30);}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover{background:var(--brand-dark) !important; color:#fff !important;}
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible, input:focus-visible{outline:3px solid #8CD3E4 !important; outline-offset:2px;}
.stTextInput input{border-radius:14px; font-size:1.02rem; padding:.6rem .9rem; background:#F2FAFE; border:2px solid #A9CBDC;}
.stTextInput input:focus{border-color:var(--brand); background:#fff;}
[data-testid="InputInstructions"]{display:none;}
[data-testid="stForm"] [data-testid="stElementContainer"], div:has(> [data-testid="stFormSubmitButton"]),
[data-testid="stFormSubmitButton"], [data-testid="stFormSubmitButton"] button{width:100% !important;}
[data-testid="stFormSubmitButton"] button{min-height:2.7rem;}
div[data-testid="stExpander"] details{border-radius:16px; border:1.5px solid var(--line); background:rgba(255,255,255,.85);}
.stToggle label p, .stCheckbox label p{font-weight:600;}
div[class*="st-key-quick"] .stButton > button{border-radius:999px; min-height:2.5rem; padding:.2rem 1.1rem; background:#fff; width:100%;}
div[class*="st-key-card_recent"] .stButton > button{width:100%; justify-content:flex-start; text-align:left; background:#FFFBEA; border-color:#EFD36A;}
div.st-key-sv_loader{display:none}
"""


def inject():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def embed(html, height):
    """Show an HTML page in an iframe on both older and newer Streamlit versions."""
    try:
        if hasattr(st, "iframe"):
            st.iframe(html, height=max(int(height), 1))
            return
    except Exception:
        pass
    components.html(html, height=height)


def speak_js(text):
    """Browser text-to-speech (works locally and when deployed). Unique each call so it always fires."""
    return ("<script>(function(){try{var w=window.parent||window;"
            "var s=w.speechSynthesis||window.speechSynthesis;"
            "var U=w.SpeechSynthesisUtterance||window.SpeechSynthesisUtterance;"
            "s.cancel();var u=new U(%s);u.rate=0.95;s.speak(u);}catch(e){}})();</script><!--%s-->"
            % (json.dumps(text), time.time()))


def say(placeholder, text):
    """Speak `text` out loud. `placeholder` is an st.empty() used to host the invisible speaker."""
    if getattr(config, "SPEAK_VIA", "browser") == "server":
        import speech
        speech.speak(text)
        return
    with placeholder.container():
        embed(speak_js(text), 1)


def banner(cls, icon, title, hint):
    return (f'<div class="banner {cls}" role="status" aria-live="polite"><span class="ic">{icon}</span>'
            f'<div><div class="t">{title}</div><div class="h">{hint}</div></div></div>')
