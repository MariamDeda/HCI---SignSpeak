"""Survey mode: only active when the app is opened with ?survey=1 (the "Start" button of the intro page).

After 60 seconds a floating "I'm done exploring" button appears on every page of the app and
takes the participant to the survey (GitHub Pages). The timer lives in the browser (sessionStorage),
so it keeps counting when the participant switches pages, and a rerun of Streamlit never resets it.

To change the link or the waiting time, edit the two settings below.
"""
import json

import streamlit as st

import theme

SURVEY_URL = "https://mariamdeda.github.io/HCI---SignSpeak/survey.html"
WAIT_SECONDS = 60

_JS = r"""<!doctype html><html><body><script>
(function(){
  var W; try { W = window.parent; void W.document; } catch(e){ return; }
  var cfg = __CFG__;
  function main(cfg){
    if (window.__ssSurvey) return; window.__ssSurvey = 1;
    var store; try { store = window.sessionStorage; store.setItem('_t','1'); } catch(e){ store = {_d:{}, getItem:function(k){return this._d[k]||null}, setItem:function(k,v){this._d[k]=String(v)}}; }
    // each press of "Start" on the intro page brings a fresh ?sv=<time> id, so every visit counts its own minute
    var sv = new URLSearchParams(location.search).get('sv') || '';
    var start = +store.getItem('ss_start') || 0;
    if (sv && store.getItem('ss_sv') !== sv){ store.setItem('ss_sv', sv); store.setItem('ss_roles', ''); start = 0; }
    if (start && Date.now() - start > 3 * 3600 * 1000) start = 0;   // stale value from an old visit
    if (!start){ start = Date.now(); store.setItem('ss_start', start); }
    var css = document.createElement('style');
    css.textContent = '#ss-sv{position:fixed;right:22px;bottom:22px;z-index:2147483000;font-family:"Open Sans","Segoe UI",system-ui,sans-serif;max-width:calc(100vw - 32px)}'
      + '#ss-sv a,#ss-sv .w{display:flex;align-items:center;gap:10px;border-radius:999px;font-weight:700;font-size:1rem;text-decoration:none;padding:13px 20px;box-shadow:0 10px 28px rgba(27,42,56,.28)}'
      + '#ss-sv .w{background:#fff;color:#1B2A38;border:2px solid #CFE0EA;font-weight:600}'
      + '#ss-sv .w b{color:#0B7F96;font-variant-numeric:tabular-nums}'
      + '#ss-sv a{background:linear-gradient(135deg,#35C590,#0B7F96);color:#fff;font-size:1.08rem;padding:15px 24px;animation:ssPop .5s cubic-bezier(.2,1.4,.4,1),ssRing 1.8s 0.5s infinite}'
      + '#ss-sv a:hover{filter:brightness(1.08);transform:translateY(-1px)}'
      + '#ss-sv a:focus-visible{outline:4px solid #F5B800;outline-offset:3px}'
      + '@keyframes ssPop{from{transform:scale(.6);opacity:0}to{transform:scale(1);opacity:1}}'
      + '@keyframes ssRing{0%{box-shadow:0 10px 28px rgba(27,42,56,.28),0 0 0 0 rgba(53,197,144,.6)}100%{box-shadow:0 10px 28px rgba(27,42,56,.28),0 0 0 18px rgba(53,197,144,0)}}'
      + '@media (max-width:600px){#ss-sv{right:16px;left:16px;bottom:16px}#ss-sv a,#ss-sv .w{justify-content:center}}'
      + '@media (prefers-reduced-motion:reduce){#ss-sv a{animation:none!important}}';
    document.head.appendChild(css);
    var box = document.createElement('div'); box.id = 'ss-sv'; box.setAttribute('role','status'); box.setAttribute('aria-live','polite');
    document.body.appendChild(box);
    var ready = false;
    function roles(){
      var p = new URLSearchParams(location.search).get('page');
      var seen = (store.getItem('ss_roles') || '').split(',').filter(Boolean);
      if ((p === 'signer' || p === 'hearing') && seen.indexOf(p) < 0){ seen.push(p); store.setItem('ss_roles', seen.join(',')); }
      return seen.join(',');
    }
    function fmt(s){ s = Math.max(0, Math.ceil(s)); return Math.floor(s/60) + ':' + ('0' + s%60).slice(-2); }
    function tick(){
      var el = (Date.now() - start) / 1000, seen = roles();
      if (el < cfg.wait){
        box.innerHTML = '<div class="w">⏱ Explore freely · survey unlocks in <b>' + fmt(cfg.wait - el) + '</b></div>';
      } else {
        var href = cfg.url + '?src=app&t=' + Math.round(el) + '&v=' + encodeURIComponent(seen);
        if (!ready){
          ready = true;
          box.innerHTML = '<a target="_blank" rel="noopener"></a>';
          box.firstChild.textContent = "Take the Survey →";
          box.firstChild.addEventListener('click', function(ev){
            ev.preventDefault();
            var u = box.firstChild.href, w = null;
            try { w = window.open(u, '_blank'); } catch(e){}
            if (!w){ window.location.href = u; }   // popups/top navigation blocked by the host frame: open in this frame
          });
        }
        box.firstChild.href = href;
      }
    }
    tick(); window.setInterval(tick, 1000);
  }
  if (W.__ssSurvey) return;
  var s = W.document.createElement('script');
  s.textContent = '(' + main.toString() + ')(' + JSON.stringify(cfg) + ')';
  W.document.head.appendChild(s);
})();
</script></body></html>"""


def active():
    """True for the whole session once the app was opened with ?survey=1."""
    if "survey" not in st.session_state:
        st.session_state.survey = st.query_params.get("survey", "") in ("1", "true", "yes")
    return st.session_state.survey


def render():
    if not active():
        return
    with st.container(key="sv_loader"):
        theme.embed(_JS.replace("__CFG__", json.dumps({"url": SURVEY_URL, "wait": WAIT_SECONDS})), height=1)
