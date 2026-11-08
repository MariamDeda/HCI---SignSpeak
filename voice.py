"""Browser microphone for the hearing page - no model download needed.

Uses the browser's built-in speech recognition (Chrome and Edge; needs internet). The recognised
words are put into the page's message box and "Show sign" is pressed for the user, so everything
after that (mapping, feedback, player) is the normal typed-message path.

If the browser has no speech recognition, the page says so and typing still works. An offline
alternative (Vosk) is available in view_hearing.py under "Offline recording".
"""

_MIC = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--ink:#1B2A38;--muted:#52667A;--line:#CFE0EA;--brand:#0B7F96;--brand-d:#08687B;--sun:#FFE27A;--err:#FBCFC8;--err-ink:#9A2F20}
*{box-sizing:border-box}
html,body{margin:0;background:transparent;color:var(--ink);font-family:"Nunito","Segoe UI Variable","Segoe UI",system-ui,sans-serif}
.box{display:flex;flex-direction:column;gap:12px;align-items:stretch;padding:2px}
button{font:inherit;font-weight:900;font-size:1.15rem;border:0;border-radius:18px;padding:16px 20px;cursor:pointer;
  color:#fff;background:var(--brand);box-shadow:0 6px 16px rgba(11,127,150,.30);transition:transform .15s,background .15s}
button:hover{transform:translateY(-1px);background:var(--brand-d)}
button:focus-visible{outline:3px solid #8CD3E4;outline-offset:2px}
button.on{background:var(--sun);color:#5A4300;box-shadow:0 0 0 0 rgba(245,184,0,.6);animation:ring 1.4s infinite}
@keyframes ring{0%{box-shadow:0 0 0 0 rgba(245,184,0,.55)}100%{box-shadow:0 0 0 18px rgba(245,184,0,0)}}
.status{font-size:1rem;color:var(--muted);min-height:1.4em}
.status.err{background:var(--err);color:var(--err-ink);border-radius:12px;padding:8px 12px;font-weight:700}
.heard{background:#CDEBF3;border-radius:14px;padding:10px 14px;font-size:1.1rem}
.heard.interim{opacity:.7}
.heard b{color:var(--brand)}
.hide{display:none}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="box" data-run="__NONCE__">
  <button id="mic" type="button">🎤 Tap and speak</button>
  <div id="status" class="status" aria-live="polite">Press the button, then say your message.</div>
  <div id="heard" class="heard hide"></div>
</div>
<script>
const P = (function(){ try { void window.parent.document; return window.parent; } catch(e){ return window; } })();
const SR = P.SpeechRecognition || P.webkitSpeechRecognition || window.SpeechRecognition || window.webkitSpeechRecognition;
const $ = id => document.getElementById(id);
const mic = $('mic'), statusEl = $('status'), heardEl = $('heard');
let rec = null, listening = false, got = false;

function setStatus(t, err){ statusEl.textContent = t; statusEl.className = 'status' + (err ? ' err' : ''); }
function showHeard(t, interim){
  heardEl.classList.remove('hide'); heardEl.classList.toggle('interim', !!interim);
  heardEl.innerHTML = ''; const lab = document.createElement('span'); lab.textContent = 'I heard: ';
  const b = document.createElement('b'); b.textContent = '“' + t + '”'; heardEl.append(lab, b);
}
function sendToApp(text){
  try {
    const doc = P.document;
    const input = doc.querySelector('input[aria-label="Your message"]');
    const btn = Array.from(doc.querySelectorAll('button')).find(b => /show sign/i.test(b.innerText));
    if (!input || !btn) return false;
    const setter = Object.getOwnPropertyDescriptor(P.HTMLInputElement.prototype, 'value').set;
    setter.call(input, text);
    input.dispatchEvent(new P.Event('input', {bubbles: true}));
    setTimeout(() => btn.click(), 250);
    return true;
  } catch(e){ return false; }
}
window.__sendTextForTest = sendToApp;

function stopUi(){ listening = false; mic.classList.remove('on'); mic.textContent = '🎤 Tap and speak'; }
const ERR = {
  'not-allowed': 'The microphone is blocked. Click the lock icon in the address bar, allow the microphone, then try again.',
  'service-not-allowed': 'The microphone is blocked. Click the lock icon in the address bar, allow the microphone, then try again.',
  'no-speech': "I didn't hear anything. Tap the button and speak a little louder.",
  'audio-capture': "I can't find a microphone. Plug one in or check your sound settings.",
  'network': 'Voice typing needs an internet connection. You can still type your message.',
  'aborted': ''
};
if (!SR){
  mic.disabled = true; mic.style.opacity = .5; mic.style.cursor = 'default';
  setStatus('This browser has no built-in voice typing. Please use Chrome or Edge, or type your message.', true);
} else {
  mic.onclick = () => {
    if (listening){ try { rec.stop(); } catch(e){} return; }
    try {
      rec = new SR(); rec.lang = 'en-US'; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
    } catch(e){ setStatus('Voice typing could not start. Please type your message.', true); return; }
    got = false;
    rec.onstart = () => { listening = true; mic.classList.add('on'); mic.textContent = '⏹ Listening… tap to stop'; setStatus('Listening. Say your message now.'); };
    rec.onresult = ev => {
      let txt = '', fin = false;
      for (let i = ev.resultIndex; i < ev.results.length; i++){ txt += ev.results[i][0].transcript; if (ev.results[i].isFinal) fin = true; }
      txt = txt.trim(); if (!txt) return;
      showHeard(txt, !fin);
      if (fin){ got = true;
        if (sendToApp(txt)) setStatus('Sent. Look at the signer panel.');
        else setStatus('I heard you, but could not send it. Please copy it into the box and press Show sign.', true); }
    };
    rec.onerror = ev => { const m = ERR[ev.error]; if (m === undefined) setStatus('Voice typing stopped (' + ev.error + '). Please type instead.', true); else if (m) setStatus(m, true); };
    rec.onend = () => { stopUi(); if (!got && !statusEl.classList.contains('err')) setStatus("I didn't catch that. Tap the button and try again."); };
    try { rec.start(); } catch(e){ setStatus('Voice typing could not start. Please try again.', true); }
  };
}
</script></body></html>"""


def mic_html(nonce=0):
    return _MIC.replace("__NONCE__", str(nonce))
