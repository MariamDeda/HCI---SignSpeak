"""Sign reference data, text -> sign mapping, and the animated sign player.

* SIGN_INFO     : emoji, spoken synonyms and a plain-language "how to sign it" per sign.
                  Signs that are in config.SIGNS but not listed here still work (generic text).
* map_text()    : turns a typed / spoken sentence into an ordered list of sign keys.
* player_html() : a self-contained HTML page (no reruns, no server round trips) that plays the
                  sequence big and slow enough to follow. If recorded landmark data exists in
                  data/<sign>/*.npy it animates a real hand skeleton from it; otherwise it shows
                  the big emoji + description.

Check every description against an ASL reference (e.g. Handspeak, ASL University) before
using the app with real participants.
"""
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st

import config

SIGN_INFO = {
    "hello": {
        "emoji": "👋",
        "words": ["hello", "hi", "hey", "hiya", "howdy", "greetings", "good morning",
                  "good afternoon", "good evening"],
        "how": "Raise a flat hand to your forehead, then move it outward and away, like a relaxed salute.",
    },
    "goodbye": {
        "emoji": "🖐️",
        "words": ["goodbye", "good bye", "bye", "bye bye", "see you", "see you later",
                  "farewell", "take care"],
        "how": "Hold up an open hand and wave it, folding the fingers down and up toward yourself.",
    },
    "thank_you": {
        "emoji": "🙏",
        "words": ["thank you", "thanks", "thank", "thankyou", "thanks a lot", "many thanks"],
        "how": "Touch your fingertips to your chin, then move your flat hand forward and down toward the other person.",
    },
    "i_love_you": {
        "emoji": "🤟",
        "words": ["i love you", "love you", "i luv you", "love"],
        "how": "Hold up your hand with thumb, index finger and little finger out, and the middle and ring fingers folded down.",
    },
    "yes": {
        "emoji": "👍",
        "words": ["yes", "yeah", "yep", "yup", "sure", "ok", "okay"],
        "how": "Make a fist and nod it up and down, like a head nodding.",
    },
    "no": {
        "emoji": "🙅",
        "words": ["no", "nope", "nah"],
        "how": "Tap your index and middle fingers against your thumb, like a beak closing.",
    },
    "please": {
        "emoji": "🙏",
        "words": ["please"],
        "how": "Rub a flat hand in a circle on your chest.",
    },
    "sorry": {
        "emoji": "😔",
        "words": ["sorry", "apologies", "i am sorry", "i'm sorry"],
        "how": "Make a fist and rub it in a circle on your chest.",
    },
    "help": {
        "emoji": "🆘",
        "words": ["help", "help me"],
        "how": "Rest a fist with the thumb up on your other flat hand, then lift both hands together.",
    },
}

# Filler words that never need a sign and should not trigger "no sign for ..." notes.
STOPWORDS = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "and", "of", "it",
             "this", "that", "i", "you", "me", "my", "your", "we", "us", "can", "could", "will",
             "would", "do", "does", "just", "so", "very", "um", "uh"}


def pretty(key):
    return config.DISPLAY.get(key, key.replace("_", " "))


def all_signs():
    """The sign list the trained model really uses (models/meta.json), else config.SIGNS.

    Reading it from the model's own meta file means the app keeps working even if config.py's
    list is out of date, which is what caused 'IndexError: list index out of range'."""
    try:
        meta = json.loads(Path(config.META_PATH).read_text())
        signs = meta.get("signs")
        if isinstance(signs, list) and signs:
            return [str(x) for x in signs]
    except Exception:
        pass
    return list(config.SIGNS)


def vocabulary():
    """Signs the app can show (the 'unknown' class is not a sign)."""
    return [s for s in all_signs() if s != config.UNKNOWN]


def info(key):
    base = SIGN_INFO.get(key, {})
    return {
        "emoji": base.get("emoji", "🤟"),
        "words": base.get("words", []) + [key.replace("_", " "), pretty(key).lower()],
        "how": base.get("how", "Perform this sign the same way it was recorded for the app."),
    }


def _phrase_table():
    table = {}
    for key in vocabulary():
        for w in info(key)["words"]:
            table.setdefault(w.lower(), key)
    return table


def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def map_text(text):
    """Returns (sign_keys_in_order, unknown_words). Longest phrase wins ('thank you' before 'you')."""
    table = _phrase_table()
    max_len = max((len(p.split()) for p in table), default=1)
    toks = _tokens(text)
    matches, unknown, i = [], [], 0
    while i < len(toks):
        for n in range(min(max_len, len(toks) - i), 0, -1):
            phrase = " ".join(toks[i:i + n])
            if phrase in table:
                matches.append(table[phrase])
                i += n
                break
        else:
            w = toks[i]
            if w not in STOPWORDS and w not in unknown:
                unknown.append(w)
            i += 1
    return matches, unknown


# ------------------------------------------------------------------ recorded hand animation
HAND_SCALE_GUESS = 0.14   # wrist->middle-finger-base distance as a share of the frame (not stored)


def _to_points(arr):
    """(30, 65) feature window -> (30, 21, 2) points in 'frame units', or None if mostly empty."""
    if arr.shape != (config.SEQ_LEN, config.FEATURE_DIM):
        return None
    present = np.abs(arr).sum(1) > 0
    if present.mean() < 0.8:
        return None
    rel = arr[:, :63].reshape(config.SEQ_LEN, 21, 3)[:, :, :2]
    wrist = arr[:, 63:65]
    pts = wrist[:, None, :] + rel * HAND_SCALE_GUESS
    pts = np.stack([pts[..., 0] * 4.0, pts[..., 1] * 3.0], axis=-1)   # keep the 4:3 camera aspect
    idx = np.where(present)[0]
    for t in range(config.SEQ_LEN):                                    # fill no-hand frames
        if not present[t]:
            pts[t] = pts[idx[np.abs(idx - t).argmin()]]
    return pts


@st.cache_data(show_spinner=False)
def _frames_for(sign, data_dir, n_files):
    files = sorted((Path(data_dir) / sign).glob("*.npy"))[:60]
    seqs = []
    for f in files:
        try:
            pts = _to_points(np.load(f))
        except Exception:
            continue
        if pts is not None:
            seqs.append(pts)
    if not seqs:
        return None
    stack = np.stack(seqs)
    best = stack[((stack - stack.mean(0)) ** 2).sum((1, 2, 3)).argmin()]   # most typical recording
    lo, hi = best.reshape(-1, 2).min(0), best.reshape(-1, 2).max(0)
    side = max(float((hi - lo).max()) * 1.25, 1e-3)
    centre = (lo + hi) / 2
    norm = (best - centre) / side + 0.5
    return [np.round(f.reshape(-1), 3).tolist() for f in norm]


def frames_for(sign):
    d = Path(config.DATA_DIR) / sign
    n = len(list(d.glob("*.npy"))) if d.exists() else 0
    return _frames_for(sign, str(config.DATA_DIR), n) if n else None


# ------------------------------------------------------------------ the player (HTML, runs in the browser)
_PLAYER = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--ink:#23323F;--muted:#5F7080;--line:#DCE8EF;--brand:#1F7A8C;--brand-d:#186577;--sun:#FFF3C4;--sun-s:#F2B705}
*{box-sizing:border-box}
html,body{margin:0;background:transparent;color:var(--ink);font-family:"Nunito","Segoe UI Variable","Segoe UI",system-ui,sans-serif}
.stage{background:linear-gradient(135deg,#F2F9FE 0%,#F1FAF5 60%,#FFFBE8 100%);border:1px solid var(--line);
  border-radius:24px;padding:18px 22px 16px;box-shadow:0 10px 32px rgba(35,50,63,.07)}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;min-height:22px}
#count{font-weight:800;font-size:.82rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.dots{display:flex;gap:9px;flex-wrap:wrap}
.dot{width:13px;height:13px;border-radius:50%;background:#C9DDE6;border:0;padding:0;cursor:pointer;transition:transform .25s,background .25s}
.dot.on{background:var(--brand);transform:scale(1.4)}
.main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;align-items:center}
.viz{position:relative;width:100%;aspect-ratio:1/1;max-height:340px;margin:auto;background:#fff;border-radius:20px;
  border:1px solid var(--line);overflow:hidden;display:flex;align-items:center;justify-content:center}
canvas{width:100%;height:100%;display:block}
.bigemoji{font-size:clamp(5.5rem,17vw,9.5rem);line-height:1;animation:float 2.6s ease-in-out infinite}
@keyframes float{0%,100%{transform:translateY(0) rotate(-3deg)}50%{transform:translateY(-10px) rotate(3deg)}}
.hide{display:none!important}
.info{min-width:0}
.emoji{font-size:2.6rem;line-height:1}
.label{font-size:clamp(2.6rem,7.5vw,4.6rem);font-weight:900;letter-spacing:-.02em;line-height:1.04;margin:.12em 0 .28em;overflow-wrap:anywhere}
.how{font-size:1.18rem;line-height:1.45;background:var(--sun);border-radius:16px;padding:13px 16px}
.how b{display:block;font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:#7A5C00;margin-bottom:4px}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
button.c{font:inherit;font-weight:800;font-size:.98rem;padding:10px 18px;border-radius:14px;cursor:pointer;
  border:1.5px solid var(--line);background:#fff;color:var(--ink);transition:transform .15s,box-shadow .15s,background .15s}
button.c:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 12px rgba(35,50,63,.10)}
button.c:focus-visible,.dot:focus-visible{outline:3px solid #8CD3E4;outline-offset:2px}
button.c.primary{background:var(--brand);border-color:var(--brand);color:#fff}
button.c.primary:hover:not(:disabled){background:var(--brand-d)}
button.c[aria-pressed="true"]{background:#E3F4EC;border-color:#7CC8A5}
button.c:disabled{opacity:.45;cursor:default}
.cap{font-size:.82rem;color:var(--muted);margin-top:12px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.swap{animation:fade .45s ease}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@media (max-width:600px){
  .main{grid-template-columns:1fr;gap:12px}.viz{max-height:210px;aspect-ratio:2/1}
  .emoji{display:none}.label{font-size:2.3rem;margin:.05em 0 .15em}.how{font-size:1rem;padding:10px 12px}
  .controls{margin-top:10px;gap:8px}button.c{padding:8px 12px;font-size:.9rem}.stage{padding:14px 16px 12px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="stage" data-run="__NONCE__">
  <div class="top"><span id="count"></span><span class="dots" id="dots"></span></div>
  <div class="main">
    <div class="viz"><canvas id="cv" width="720" height="720"></canvas><div id="big" class="bigemoji hide"></div></div>
    <div class="info" id="info">
      <div class="emoji" id="emoji"></div>
      <div class="label" id="label"></div>
      <div class="how"><b>How to sign it</b><span id="how"></span></div>
    </div>
  </div>
  <div class="controls">
    <button class="c" id="prev">⏮ Back</button>
    <button class="c primary" id="play">⏸ Pause</button>
    <button class="c" id="next">Next ⏭</button>
    <button class="c" id="slow" aria-pressed="false">🐢 Slow</button>
  </div>
  <div class="cap" id="cap"></div>
  <div class="sr" id="live" aria-live="polite"></div>
</div>
<script>
const SEQ = __DATA__;
const n = SEQ.length;
const $ = id => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
              [9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const TIPS = [4,8,12,16,20];
let idx = 0, playing = true, slow = false, elapsed = 0, last = performance.now();

function buildDots(){
  const d = $('dots'); d.innerHTML = '';
  if (n < 2) return;
  SEQ.forEach((s,i) => {
    const b = document.createElement('button');
    b.className = 'dot'; b.setAttribute('aria-label', 'Show sign ' + (i+1) + ': ' + s.label);
    b.onclick = () => show(i); d.appendChild(b);
  });
}
function show(i){
  idx = Math.max(0, Math.min(n-1, i)); elapsed = 0;
  const s = SEQ[idx];
  $('count').textContent = n > 1 ? 'Sign ' + (idx+1) + ' of ' + n : 'Sign';
  $('emoji').textContent = s.emoji; $('label').textContent = s.label; $('how').textContent = s.how;
  $('big').textContent = s.emoji;
  const has = !!s.frames;
  cv.classList.toggle('hide', !has); $('big').classList.toggle('hide', has);
  $('cap').textContent = has
    ? 'Animated from recorded hand landmarks, shown as in a mirror: copy it as if looking at yourself.'
    : 'Picture is a stand-in. Follow the description to make the sign.';
  document.querySelectorAll('.dot').forEach((d,k) => d.classList.toggle('on', k === idx));
  $('prev').disabled = idx === 0; $('next').disabled = idx === n-1;
  const info = $('info'); info.classList.remove('swap'); void info.offsetWidth; info.classList.add('swap');
  $('live').textContent = s.label + '. How to sign it: ' + s.how;
}
function lerpFrame(frames, p){
  const f = p * (frames.length - 1), i = Math.floor(f), a = f - i;
  const A = frames[i], B = frames[Math.min(i+1, frames.length-1)];
  return A.map((v,k) => v + (B[k]-v) * a);
}
function draw(frames, p){
  const W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,W,H);
  // path of the wrist so far (shows the movement)
  ctx.save(); ctx.setLineDash([2,16]); ctx.lineCap = 'round'; ctx.lineWidth = 8; ctx.strokeStyle = '#BFE3D4';
  ctx.beginPath();
  const upto = Math.floor(p * (frames.length-1));
  for (let i = 0; i <= upto; i++){ const x = frames[i][0]*W, y = frames[i][1]*H; i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); }
  ctx.stroke(); ctx.restore();
  const f = lerpFrame(frames, p);
  const P = k => [f[2*k]*W, f[2*k+1]*H];
  ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 12; ctx.strokeStyle = '#1F7A8C';
  CONN.forEach(([a,b]) => { const A = P(a), B = P(b); ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke(); });
  for (let k = 0; k < 21; k++){
    const [x,y] = P(k), tip = TIPS.includes(k);
    ctx.beginPath(); ctx.arc(x, y, tip ? 14 : 10, 0, 6.2832);
    ctx.fillStyle = tip ? '#F2B705' : '#fff'; ctx.fill();
    ctx.lineWidth = 5; ctx.strokeStyle = '#1F7A8C'; ctx.stroke();
  }
}
function tick(now){
  const dt = Math.max(0, Math.min(now - last, 100)); last = now;
  if (playing) elapsed += dt;
  const s = SEQ[idx];
  if (s.frames){
    const move = slow ? 3400 : 1800, hold = slow ? 900 : 650, cycle = move + hold;
    draw(s.frames, Math.max(0, Math.min(1, (elapsed % cycle) / move)));
    if (playing && idx < n-1 && elapsed >= cycle * 2) show(idx+1);
  } else if (playing && idx < n-1 && elapsed >= 4200) show(idx+1);
  requestAnimationFrame(tick);
}
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
$('prev').onclick = () => show(idx-1);
$('next').onclick = () => show(idx+1);
$('slow').onclick = () => { slow = !slow; $('slow').setAttribute('aria-pressed', slow); elapsed = 0; };
buildDots(); show(0); requestAnimationFrame(tick);
</script></body></html>"""


def player_html(keys, nonce=0):
    seq = []
    for k in keys:
        i = info(k)
        seq.append({"key": k, "label": pretty(k).capitalize(), "emoji": i["emoji"],
                    "how": i["how"], "frames": frames_for(k)})
    data = json.dumps(seq).replace("</", "<\\/")
    return _PLAYER.replace("__DATA__", data).replace("__NONCE__", str(nonce))
