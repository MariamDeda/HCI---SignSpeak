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
    "mom": {
        "emoji": "👩",
        "words": ["mom", "mum", "mommy", "mummy", "mama", "mother", "mothers"],
        "how": "Spread your fingers wide and tap your thumb on your chin, twice.",
    },
    "dad": {
        "emoji": "👨",
        "words": ["dad", "daddy", "papa", "father", "fathers"],
        "how": "Spread your fingers wide and tap your thumb on your forehead, twice.",
    },
    "water": {
        "emoji": "💧",
        "words": ["water"],
        "how": "Make a W: index, middle and ring fingers up, thumb touching the little finger. Tap your index finger on your chin, twice.",
    },
    "food": {
        "emoji": "🍎",
        "words": ["food", "eat", "eating", "meal", "lunch", "dinner", "breakfast", "snack"],
        "how": "Bunch your fingertips together with your thumb, then tap them on your lips, twice.",
    },
    "drink": {
        "emoji": "🥤",
        "words": ["drink", "drinking", "drinks"],
        "how": "Shape your hand as if holding a cup, then bring it to your mouth and tilt it as if drinking.",
    },
    "hungry": {
        "emoji": "🍽️",
        "words": ["hungry", "starving", "hunger"],
        "how": "Make a C with your hand and place it at the top of your chest, then slide it down to your stomach.",
    },
    "happy": {
        "emoji": "😊",
        "words": ["happy", "glad", "joy", "joyful", "cheerful"],
        "how": "Brush both flat hands up your chest in a circle, twice.",
    },
    "sad": {
        "emoji": "😢",
        "words": ["sad", "unhappy", "upset"],
        "how": "Hold both open hands in front of your face, then drag them slowly down, with a sad face.",
    },
    "bad": {
        "emoji": "👎",
        "words": ["bad", "terrible", "awful"],
        "how": "Touch your fingertips to your lips, then turn the hand over and move it down and away.",
    },
    "where": {
        "emoji": "❓",
        "words": ["where", "wheres", "where's"],
        "how": "Hold up your index finger and shake it from side to side, with raised eyebrows.",
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


def speaker_vocabulary():
    """Signs the Speaker side can SHOW. The animation does not need the trained model, so every sign with a
    written description and animation is available, plus anything the model knows."""
    out = list(vocabulary())
    for k in SIGN_INFO:
        if k not in out:
            out.append(k)
    return out


def info(key):
    base = SIGN_INFO.get(key, {})
    return {
        "emoji": base.get("emoji", "🤟"),
        "words": base.get("words", []) + [key.replace("_", " "), pretty(key).lower()],
        "how": base.get("how", "Perform this sign the same way it was recorded for the app."),
    }


def _phrase_table():
    table = {}
    for key in speaker_vocabulary():
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
# An animated cartoon signer (head, torso, both arms, hands with individual fingers) performs each sign
# from hand-written keyframes. No data files needed. Signs without a script show a big emoji instead.
_PLAYER = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600;700&display=swap');
:root{--ink:#23323F;--muted:#5F7080;--line:#DCE8EF;--brand:#1F7A8C;--brand-d:#186577;--sun:#FFF3C4}
*{box-sizing:border-box}
html,body{margin:0;background:transparent;color:var(--ink);font-family:"Open Sans","Segoe UI Variable","Segoe UI",system-ui,sans-serif}
.stage{background:linear-gradient(135deg,#F2F9FE 0%,#F1FAF5 60%,#FFFBE8 100%);border:1px solid var(--line);
  border-radius:12px;padding:14px 18px 12px;box-shadow:0 4px 14px rgba(35,50,63,.06)}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;min-height:22px}
#count{font-weight:600;font-size:.82rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.dots{display:flex;gap:9px;flex-wrap:wrap}
.dot{width:13px;height:13px;border-radius:50%;background:#C9DDE6;border:0;padding:0;cursor:pointer;transition:transform .25s,background .25s}
.dot.on{background:var(--brand);transform:scale(1.4)}
.main{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:24px;align-items:center}
.viz{position:relative;width:100%;aspect-ratio:10/11;max-height:430px;margin:auto;background:#fff;border-radius:12px;
  border:1px solid var(--line);overflow:hidden;display:flex;align-items:center;justify-content:center}
canvas{width:100%;height:100%;display:block}
.v3{position:absolute;inset:0;background:linear-gradient(#F4FAFD,#E5F1F6)}
.bigemoji{font-size:clamp(5.5rem,17vw,9.5rem);line-height:1;animation:float 2.6s ease-in-out infinite}
@keyframes float{0%,100%{transform:translateY(0) rotate(-3deg)}50%{transform:translateY(-10px) rotate(3deg)}}
.hide{display:none!important}
.info{min-width:0}
.label{font-size:clamp(2rem,5vw,3rem);font-weight:700;letter-spacing:-.02em;line-height:1.04;margin:0 0 .3em;overflow-wrap:anywhere}
.how{font-size:1rem;line-height:1.45;background:var(--sun);border-radius:8px;padding:13px 16px}
.how b{display:block;font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:#7A5C00;margin-bottom:6px}
.steps{margin:0;padding-left:1.25rem}
.steps li{margin:.25rem 0;padding:.1rem .45rem;border-radius:6px;transition:background .25s}
.steps li::marker{font-weight:700;color:#7A5C00}
.steps li.on{background:#FFE48A;font-weight:600}
.phase{position:absolute;left:10px;top:10px;font-size:.74rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  padding:4px 10px;border-radius:999px;background:#E1F1F6;border:1px solid #8CD3E4;color:#14566a}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
button.c{font:inherit;font-weight:600;font-size:.9rem;padding:8px 15px;border-radius:8px;cursor:pointer;
  border:1.5px solid var(--line);background:#fff;color:var(--ink);transition:transform .15s,box-shadow .15s,background .15s}
button.c:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 12px rgba(35,50,63,.10)}
button.c:focus-visible,.dot:focus-visible{outline:3px solid #8CD3E4;outline-offset:2px}
button.c.primary{background:var(--brand);border-color:var(--brand);color:#fff}
button.c.primary:hover:not(:disabled){background:var(--brand-d)}
button.c:disabled{opacity:.45;cursor:default}
.scrub{display:flex;align-items:center;gap:10px;margin-top:12px}
.scrub input{flex:1;accent-color:var(--brand);height:6px;cursor:pointer}
.scrub span{font-size:.75rem;color:var(--muted);min-width:3.2rem;text-align:right}
.cap{font-size:.82rem;color:var(--muted);margin-top:10px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.swap{animation:fade .45s ease}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@media (max-width:600px){
  .main{grid-template-columns:1fr;gap:12px}.viz{max-height:300px;aspect-ratio:10/9}
  .label{font-size:2.2rem}.how{font-size:1rem;padding:10px 12px}
  .controls{margin-top:10px;gap:8px}button.c{padding:8px 12px;font-size:.9rem}.stage{padding:14px 16px 12px}}
@media (prefers-reduced-motion:reduce){.bigemoji,.swap{animation:none!important}}
</style></head><body>
<div class="stage" data-run="__NONCE__">
  <div class="top"><span id="count"></span><span class="dots" id="dots"></span></div>
  <div class="main">
    <div class="viz"><div id="v3" class="v3 hide"></div><canvas id="cv" width="800" height="880"></canvas><div id="big" class="bigemoji hide"></div>
      <div class="phase hide" id="phase"></div></div>
    <div class="info" id="info">
      <div class="label" id="label"></div>
      <div class="how"><b>How to sign it</b><ol class="steps" id="how"></ol></div>
    </div>
  </div>
  <div class="controls">
    <button class="c" id="prev">⏮ Back</button>
    <button class="c primary" id="play">⏸ Pause</button>
    <button class="c" id="next">Next ⏭</button>
    <button class="c" id="slow" title="Change the speed">🐢 Speed: Slow</button>
    <button class="c" id="replay">↺ Replay</button>
  </div>
  <div class="scrub" id="scrubrow"><input type="range" id="scrub" min="0" max="1000" value="0" aria-label="Move through the sign"><span id="pct"></span></div>
  <div class="cap" id="cap"></div>
  <div class="sr" id="live" aria-live="polite"></div>
</div>
<script>
const SEQ = __DATA__;
const n = SEQ.length;
const $ = id => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
const D = Math.PI/180;

/* ---------------------------------------------------------------- the signer */
const SKIN='#EBB992', SKIN_D='#B98462', SHIRT='#1F7A8C', SHIRT_D='#175F6E', HAIR='#4A2E22';
const SH = {R:[146,200], L:[254,200]};          // shoulders (R = the signer's right hand = viewer's left)
const LEN1 = 92, LEN2 = 86;                      // upper arm, forearm

// hand shapes: c = curl of index, middle, ring, little (0 straight .. 1 folded); th = thumb angle (deg,
// + out to the side, - across the palm); sp = finger spread (deg)
const SHAPES = {
  relax:{c:[.35,.35,.4,.45], th:25, sp:3},
  flat:{c:[0,0,0,0], th:28, sp:2},
  open5:{c:[0,0,0,0], th:55, sp:16},
  fist:{c:[1,1,1,1], th:-55, sp:0},
  thumbup:{c:[1,1,1,1], th:6, sp:0, tl:1.35},
  ily:{c:[0,1,1,0], th:75, sp:14},
  two:{c:[0,0,1,1], th:45, sp:6},
  W:{c:[0,0,0,1], th:-62, sp:15},
  C:{c:[.42,.42,.42,.42], th:48, sp:4, tl:1.1},
  bunch:{c:[.5,.5,.5,.5], th:-28, sp:0, tl:1.1},
  point:{c:[0,1,1,1], th:-50, sp:0}
};
function h(x,y,r,shape,o){
  const S = SHAPES[shape], out = x < 200 ? -1 : 1, e = ik(out < 0 ? SH.R : SH.L, [x,y], out).E;
  const q = Object.assign({x:x,y:y,r:r,pz:0,pinch:0,s:1,w:1,fl:0,tap:0,ts:1,tl:S.tl||1,c:S.c.slice(),th:S.th,sp:S.sp,ex:e[0],ey:e[1]}, o||{});
  ['x','y','ex','ey','w','r'].forEach(k => { if (q[k+'3'] === undefined) q[k+'3'] = q[k]; });   // optional separate coordinates for the 3D signer
  if (q.nl === undefined) q.nl = (out < 0 ? q.ts < 0 : q.ts > 0) ? 1 : 0;      // back of the hand showing?
  return q;
}
const REST = {R:h(132,338,6,'relax'), L:h(268,338,-6,'relax',{ts:-1})};
function F(t,o){ return Object.assign({t:t,R:REST.R,L:REST.L,hx:0,hy:0,st:0,cam:{x:200,y:220,z:1}}, o||{}); }
function circle(shape,cx,cy,rad,rot,o,t0,dt,loops,st){
  const out=[], N=8;
  for(let k=0;k<=N*loops;k++){
    const a=(k/N)*2*Math.PI - Math.PI/2;
    out.push(F(t0+k*dt,{R:h(cx+rad*Math.cos(a),cy+rad*Math.sin(a),rot,shape,o),st:st}));
  }
  return out;
}
function ends(frames,T){ frames.push(F(T,{st:frames[frames.length-1].st})); return frames; }

const SCRIPTS = {
  hello:{mood:'smile', frames:(()=>{
    const a=h(104,122,52,'flat',{ex:96,ey:215}), b=h(66,132,8,'flat',{ex:60,ey:222});
    return ends([F(0),F(1.1,{R:a,st:0}),F(1.9,{R:a,st:0}),F(3.0,{R:b,st:1}),F(3.6,{R:b,st:1})],4.7);})()},
  goodbye:{mood:'smile', frames:(()=>{
    const o=h(122,176,0,'open5',{ex:104,ey:285}), f=h(122,180,-4,'open5',{c:[.85,.85,.85,.85],th:40,ex:104,ey:285});
    return ends([F(0),F(1.1,{R:o}),F(1.7,{R:o}),F(2.05,{R:f}),F(2.4,{R:o}),F(2.75,{R:f}),F(3.1,{R:o}),F(3.45,{R:f}),F(3.8,{R:o}),F(4.2,{R:o})],5.1);})()},
  thank_you:{mood:'smile', frames:(()=>{
    // flat hand, palm towards the face, fingertips at the chin ... then out and down towards the other person
    const a=h(164,240,18,'flat',{ts:-1,ex:132,ey:330}), b=h(150,300,55,'flat',{ts:-1,s:1.4,ex:122,ey:356});
    return ends([F(0),F(1.1,{R:a,st:0,hy:2}),F(2.0,{R:a,st:0,hy:2}),F(3.2,{R:b,st:1}),F(4.0,{R:b,st:1})],5.0);})()},
  i_love_you:{mood:'smile', frames:(()=>{
    const r=a=>h(120,192,a,'ily',{ex:100,ey:292});
    return ends([F(0),F(1.1,{R:r(0)}),F(1.6,{R:r(0)}),F(2.0,{R:r(-10)}),F(2.4,{R:r(10)}),F(2.8,{R:r(-10)}),F(3.2,{R:r(10)}),F(3.6,{R:r(0)}),F(4.1,{R:r(0)})],5.0);})()},
  yes:{mood:'smile', frames:(()=>{
    // fist (thumb folded over the fingers) nods forward and back at the wrist, like a head nodding
    const up=h(152,238,0,'fist',{ex:120,ey:326}), dn=h(152,252,16,'fist',{ex:120,ey:334,pz:0.9});
    return ends([F(0),F(1.0,{R:up}),F(1.4,{R:up}),F(1.8,{R:dn,hy:5}),F(2.2,{R:up}),F(2.6,{R:dn,hy:5}),F(3.0,{R:up}),F(3.4,{R:dn,hy:5}),F(3.8,{R:up}),F(4.2,{R:up})],5.0);})()},
  no:{mood:'neutral', frames:(()=>{
    // front view: index + middle finger straight up, thumb out to the side; then the two fingers fold down and tap the thumb
    const o=h(128,236,18,'two',{c:[0,0,1,1],th:62,ex:104,ey:330}),
          c=h(128,236,18,'two',{c:[.4,.4,1,1],th:-15,tap:1,pinch:1,ex:104,ey:330});
    return ends([F(0),F(1.0,{R:o}),F(1.9,{R:o}),F(2.15,{R:c,hx:-3}),F(2.4,{R:c,hx:-3}),F(2.8,{R:o,hx:3}),F(3.05,{R:c,hx:-3}),F(3.3,{R:c,hx:-3}),F(3.7,{R:o,hx:3}),F(3.95,{R:c,hx:-3}),F(4.2,{R:c,hx:-3}),F(4.7,{R:o,hx:0}),F(5.2,{R:o})],6.0);})()},
  mom:{mood:'smile', frames:(()=>{
    const P=(x,y,x3,y3)=>h(x,y,-15,'open5',{ts:1,ex:128,ey:335,x3:x3,y3:y3,w3:.35}), T=P(167,190,191,202), A=P(158,205,182,217);
    return ends([F(0),F(1.1,{R:A}),F(1.5,{R:T}),F(1.85,{R:A}),F(2.2,{R:T}),F(2.55,{R:A}),F(2.9,{R:T}),F(3.5,{R:T})],4.5);})()},
  dad:{mood:'smile', frames:(()=>{
    const P=(x,y,x3,y3)=>h(x,y,-15,'open5',{ts:1,ex:128,ey:300,x3:x3,y3:y3,w3:.35,r3:45,ex3:80,ey3:130}), T=P(158,120,160,99), A=P(148,136,150,115);
    return ends([F(0),F(1.2,{R:A}),F(1.6,{R:T}),F(1.95,{R:A}),F(2.3,{R:T}),F(2.65,{R:A}),F(3.0,{R:T}),F(3.6,{R:T})],4.6);})()},
  water:{mood:'neutral', frames:(()=>{
    const P=(x,y,x3,y3)=>h(x,y,25,'W',{ts:1,ex:118,ey:330,x3:x3,y3:y3}), T=P(142,221,159,203), A=P(132,234,149,216);
    return ends([F(0),F(1.2,{R:A}),F(1.6,{R:T}),F(1.95,{R:A}),F(2.3,{R:T}),F(2.65,{R:A}),F(3.0,{R:T}),F(3.6,{R:T})],4.6);})()},
  food:{mood:'smile', frames:(()=>{
    const P=(x,y,x3,y3)=>h(x,y,25,'bunch',{ts:1,nl:1,pinch:1,c:[.38,.38,.38,.38],ex:128,ey:335,x3:x3,y3:y3}), T=P(180,182,187,199), A=P(168,198,175,215);
    return ends([F(0),F(1.2,{R:A}),F(1.6,{R:T}),F(1.95,{R:A}),F(2.3,{R:T}),F(2.65,{R:A}),F(3.0,{R:T}),F(3.6,{R:T})],4.6);})()},
  drink:{mood:'neutral', frames:(()=>{
    const a=h(170,206,0,'C',{ts:1,ex:128,ey:335,s:1.0}), b=h(182,188,28,'C',{ts:1,ex:130,ey:335});
    return ends([F(0),F(1.2,{R:a,st:0}),F(1.9,{R:a,st:0}),F(2.8,{R:b,st:1}),F(3.6,{R:b,st:1,hy:-2}),F(3.9,{R:a,st:1})],5.0);})()},
  hungry:{mood:'neutral', frames:(()=>{
    const a=h(166,200,80,'C',{ts:-1,ex:128,ey:330}), b=h(166,268,80,'C',{ts:-1,ex:128,ey:340});
    return ends([F(0),F(1.2,{R:a,st:0}),F(1.9,{R:a,st:0}),F(3.1,{R:b,st:1}),F(3.7,{R:b,st:1,hy:2})],5.0);})()},
  happy:{mood:'smile', frames:(()=>{
    const R=(x,y)=>h(x,y,75,'flat',{ts:-1,ex:110,ey:335}), L=(x,y)=>h(400-x,y,-75,'flat',{ts:1,ex:290,ey:335});
    const A=[154,256], B=[154,214], C=[138,232], fr=[F(0),F(1.0,{R:R(...A),L:L(...A),st:0})];
    let t=1.3;
    for(let k=0;k<3;k++){ fr.push(F(t,{R:R(...A),L:L(...A)})); fr.push(F(t+.5,{R:R(...B),L:L(...B)})); fr.push(F(t+.8,{R:R(...C),L:L(...C)})); t+=1.05; }
    fr.push(F(t,{R:R(...A),L:L(...A)})); return ends(fr,t+.9);})()},
  sad:{mood:'sad', frames:(()=>{
    const R=(y)=>h(163,y,12,'open5',{ts:-1,ex:120,ey:330}), L=(y)=>h(237,y,-12,'open5',{ts:1,ex:280,ey:330});
    return ends([F(0),F(1.2,{R:R(205),L:L(205),st:0}),F(2.0,{R:R(205),L:L(205),st:0}),F(3.6,{R:R(278),L:L(278),st:1,hy:5}),F(4.3,{R:R(278),L:L(278),st:1,hy:5})],5.4);})()},
  bad:{mood:'sad', frames:(()=>{
    const a=h(180,182,20,'flat',{ts:-1,ex:130,ey:330}), b=h(140,290,95,'flat',{ts:1,ex:105,ey:350});
    return ends([F(0),F(1.2,{R:a,st:0}),F(1.9,{R:a,st:0}),F(3.1,{R:b,st:1}),F(3.7,{R:b,st:1})],4.8);})()},
  where:{mood:'ask', frames:(()=>{
    const r=a=>h(132,236,a,'point',{ts:1,ex:108,ey:335});
    return ends([F(0),F(1.0,{R:r(0)}),F(1.4,{R:r(-16)}),F(1.8,{R:r(16)}),F(2.2,{R:r(-16)}),F(2.6,{R:r(16)}),F(3.0,{R:r(-16)}),F(3.4,{R:r(16)}),F(3.9,{R:r(0)})],4.8);})()},
  please:{mood:'smile', frames:(()=>{
    const s=circle('flat',150,230,15,72,{ts:-1,ex:110,ey:322},1.9,.24,2,0);
    return ends([F(0),F(1.0,{R:h(150,215,72,'flat',{ts:-1,ex:110,ey:322})}),F(1.6,{R:s[0].R})].concat(s.slice(1)).concat([F(4.2,{R:s[s.length-1].R})]),5.2);})()},
  sorry:{mood:'sad', frames:(()=>{
    const s=circle('fist',176,234,13,0,{ts:-1,ex:130,ey:322},1.9,.26,2,0);
    return ends([F(0),F(1.0,{R:h(176,220,0,'fist',{ts:-1,ex:130,ey:322})}),F(1.6,{R:s[0].R})].concat(s.slice(1)).concat([F(4.4,{R:s[s.length-1].R})]),5.4);})()},
  help:{mood:'neutral', frames:(()=>{
    const L0=h(272,282,-84,'flat',{ts:-1,th:8,ex:300,ey:366}), R0=h(238,270,0,'thumbup',{ex:186,ey:340});
    const L1=h(272,236,-84,'flat',{ts:-1,th:8,ex:300,ey:322}), R1=h(238,224,0,'thumbup',{ex:186,ey:296});
    return ends([F(0),F(1.4,{R:R0,L:L0,st:0}),F(2.2,{R:R0,L:L0,st:0}),F(3.3,{R:R1,L:L1,st:1}),F(4.0,{R:R1,L:L1,st:1})],5.2);})()}
};

// camera: zoom in on the action while a sign is performed, zoom back out at rest
const ARROWS = {   // a guide arrow drawn while the hand travels: [t from, t to, start, control, end]
  thank_you:[1.9,3.4,[206,150],[262,150],[268,228]],
  hungry:[1.9,3.1,[212,205],[222,238],[212,272]],
  bad:[1.9,3.1,[196,150],[150,170],[118,262]],
  hello:[1.9,3.2,[150,62],[110,40],[70,92]]
};
const CAMS = {hello:[135,140,1.7], goodbye:[150,185,1.6], thank_you:[160,215,1.5], i_love_you:[150,190,1.7],
              yes:[170,220,1.7], mom:[165,170,1.6], dad:[155,115,1.5], water:[158,185,1.6], food:[180,175,1.7], drink:[178,190,1.6], hungry:[175,230,1.5], happy:[200,225,1.45], sad:[200,190,1.35], bad:[168,215,1.45], where:[130,205,1.7], no:[132,190,1.75], please:[170,215,1.7], sorry:[180,215,1.7], help:[230,260,1.55]};
Object.keys(CAMS).forEach(k => { const c = CAMS[k], fr = SCRIPTS[k].frames;
  for (let i = 1; i < fr.length-1; i++) fr[i].cam = {x:c[0], y:c[1], z:c[2]}; });

/* ---------------------------------------------------------------- interpolation */
function smooth(u){ return u*u*(3-2*u); }
function mix(a,b,u){
  if (typeof a === 'number') return a + (b-a)*u;
  if (Array.isArray(a)) return a.map((v,i)=>mix(v,b[i],u));
  const o = {}; for (const k in a) o[k] = mix(a[k], b[k], u); return o;
}
function sample(frames,t){
  let i = 0; while (i < frames.length-2 && t >= frames[i+1].t) i++;
  const A = frames[i], B = frames[i+1];
  const u = Math.max(0, Math.min(1, (t-A.t)/(B.t-A.t)));
  const e = smooth(u);
  const out = {cam:mix(A.cam,B.cam,e), R:mix(A.R,B.R,e), L:mix(A.L,B.L,e), hx:mix(A.hx,B.hx,e), hy:mix(A.hy,B.hy,e), st:(u<.5?A.st:B.st)};
  return out;
}

/* ---------------------------------------------------------------- drawing */
function limb(pts, w, col, out){
  ctx.lineCap='round'; ctx.lineJoin='round';
  [[w+3.5,out],[w,col]].forEach(([lw,c])=>{
    ctx.strokeStyle=c; ctx.lineWidth=lw; ctx.beginPath();
    pts.forEach((p,i)=> i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.stroke();
  });
}
function rrect(x,y,w,hh,r){
  ctx.beginPath(); ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+hh-r); ctx.quadraticCurveTo(x+w,y+hh,x+w-r,y+hh); ctx.lineTo(x+r,y+hh);
  ctx.quadraticCurveTo(x,y+hh,x,y+hh-r); ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
}
const SKIN_L='#F6D2B6', NAIL='#F3DDD0';
function tube(pts, r0, r1){                       // tapered, rounded limb with its own outline
  const segs=[]; let tot=0;
  for(let i=1;i<pts.length;i++){ const l=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]); segs.push(l); tot+=l; }
  [[1.9,SKIN_D],[0,SKIN]].forEach(([g,col])=>{
    ctx.fillStyle=col; let acc=0;
    for(let i=1;i<pts.length;i++){
      const A=pts[i-1], B=pts[i], ra=r0+(r1-r0)*(acc/(tot||1))+g, rb=r0+(r1-r0)*((acc+segs[i-1])/(tot||1))+g;
      acc+=segs[i-1];
      const dx=B[0]-A[0], dy=B[1]-A[1], L=Math.hypot(dx,dy)||1, nx=-dy/L, ny=dx/L;
      ctx.beginPath(); ctx.moveTo(A[0]+nx*ra,A[1]+ny*ra); ctx.lineTo(B[0]+nx*rb,B[1]+ny*rb);
      ctx.lineTo(B[0]-nx*rb,B[1]-ny*rb); ctx.lineTo(A[0]-nx*ra,A[1]-ny*ra); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.arc(B[0],B[1],rb,0,7); ctx.fill();
      if(i===1){ ctx.beginPath(); ctx.arc(A[0],A[1],ra,0,7); ctx.fill(); }
    }
  });
}
function finger(i,p,PH){
  const base=[16.5,5.8,-5.2,-15.5], dy0=[0,-1.5,0,3.5], len=[37,42,38,29], rad=[5.7,5.9,5.5,4.8],
        frac=[.45,.30,.25], bend=[85,105,75];
  const c=p.c[i], ang=(1.5-i)*p.sp/3*D - ((i<2)?(p.fl||0)*D:0);
  let x=base[i]*p.w, y=-PH+4+dy0[i], cum=0; const pts=[[x,y]];
  for(let k=0;k<3;k++){ cum+=c*bend[k]; const sl=len[i]*frac[k]*Math.cos(cum*D); x+=Math.sin(ang)*sl; y-=Math.cos(ang)*sl; pts.push([x,y]); }
  tube(pts, rad[i], rad[i]*.74);
  const tip=pts[3], pr=pts[2];
  if (c<.5){
    const tx=tip[0]-pr[0], ty=tip[1]-pr[1], L=Math.hypot(tx,ty)||1, ux=tx/L, uy=ty/L, nx=-uy, ny=ux;
    if (p.nl){                                     // fingernail (back of the hand is showing)
      const nw=rad[i]*.78, nlen=9;
      ctx.fillStyle=NAIL; ctx.strokeStyle='rgba(150,100,80,.55)'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(tip[0]-ux*nlen+nx*nw, tip[1]-uy*nlen+ny*nw);
      ctx.lineTo(tip[0]-ux*1+nx*nw*.9, tip[1]-uy*1+ny*nw*.9);
      ctx.quadraticCurveTo(tip[0]+ux*2.5, tip[1]+uy*2.5, tip[0]-ux*1-nx*nw*.9, tip[1]-uy*1-ny*nw*.9);
      ctx.lineTo(tip[0]-ux*nlen-nx*nw, tip[1]-uy*nlen-ny*nw); ctx.closePath(); ctx.fill(); ctx.stroke();
    } else {                                       // palm side: finger joint creases
      ctx.strokeStyle='rgba(150,95,70,.5)'; ctx.lineWidth=1;
      [1,2].forEach(k=>{ const q=pts[k]; ctx.beginPath(); ctx.moveTo(q[0]-nx*rad[i]*.7,q[1]-ny*rad[i]*.7); ctx.lineTo(q[0]+nx*rad[i]*.7,q[1]+ny*rad[i]*.7); ctx.stroke(); });
    }
  }
}
function palmPath(PW,PH,w){
  const hw=PW/2*w;
  ctx.beginPath(); ctx.moveTo(-hw*.72,6);
  ctx.bezierCurveTo(-hw*1.05,-PH*.25,-hw*1.12,-PH*.7,-hw,-PH+2);
  ctx.quadraticCurveTo(0,-PH-4,hw,-PH+3);                                   // knuckle line
  ctx.bezierCurveTo(hw*1.25,-PH*.65,hw*1.35,-PH*.28,hw*.72,6);               // thumb-side bulge
  ctx.closePath();
}
function hand(p){
  ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.r*D); ctx.scale(p.ts>=0?1:-1,1);   // thumb is always +x here
  const HS=1.15*p.s; ctx.scale(HS,HS);
  const PW=44, PH=46;
  for(let i=0;i<4;i++) if (p.c[i] <= .5) finger(i,p,PH);               // open fingers sit behind the palm
  const g=ctx.createRadialGradient(0,-PH*.45,4,0,-PH*.4,PW*.9); g.addColorStop(0,SKIN_L); g.addColorStop(1,SKIN);
  palmPath(PW,PH,p.w); ctx.fillStyle=g; ctx.fill(); ctx.strokeStyle=SKIN_D; ctx.lineWidth=2; ctx.stroke();
  // palm creases (palm side) or knuckle marks (back of the hand)
  ctx.strokeStyle='rgba(150,95,70,.45)'; ctx.lineWidth=1.2; ctx.lineCap='round';
  if (!p.nl){
    ctx.beginPath(); ctx.moveTo(-14,-PH*.45); ctx.quadraticCurveTo(0,-PH*.32,14,-PH*.52); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-14,-PH*.7); ctx.quadraticCurveTo(0,-PH*.58,16,-PH*.78); ctx.stroke();
  } else {
    [16.5,5.8,-5.2,-15.5].forEach((bx,k)=>{ ctx.beginPath(); ctx.arc(bx*p.w,-PH+7+[0,-1,0,3][k],3.4,Math.PI*1.1,Math.PI*1.9); ctx.stroke(); });
  }
  for(let i=0;i<4;i++) if (p.c[i] > .5) finger(i,p,PH);                // folded fingers lie on the palm
  // thumb: two bones, thicker at the base
  const a1=p.th*.55*D, a2=p.th*D+10*D, L1=19*p.tl, L2=17*p.tl, bx=PW/2*.62*p.w, by=-9;
  const m=[bx+Math.sin(a1)*L1, by-Math.cos(a1)*L1], e=[m[0]+Math.sin(a2)*L2, m[1]-Math.cos(a2)*L2];
  tube([[bx,by],m,e],7.4,5.3);
  if (p.nl && p.th>0){ const L=Math.hypot(e[0]-m[0],e[1]-m[1])||1, ux=(e[0]-m[0])/L, uy=(e[1]-m[1])/L;
    ctx.fillStyle=NAIL; ctx.strokeStyle='rgba(150,100,80,.55)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.ellipse(e[0]-ux*3,e[1]-uy*3,3.4,4.6,Math.atan2(uy,ux)+Math.PI/2,0,7); ctx.fill(); ctx.stroke(); }
  if (p.tap > .25){                                // little 'tap' marks where the fingertips meet the thumb
    ctx.save(); ctx.globalAlpha=Math.min(1,(p.tap-.25)*2); ctx.strokeStyle='#F2B705'; ctx.lineWidth=3.2; ctx.lineCap='round';
    [-60,-15,30].forEach(a=>{ const r=(a-90)*D; ctx.beginPath(); ctx.moveTo(4+Math.cos(r)*16,-PH-6+Math.sin(r)*16); ctx.lineTo(4+Math.cos(r)*27,-PH-6+Math.sin(r)*27); ctx.stroke(); });
    ctx.restore();
  }
  ctx.restore();
}
function ik(S,W,out){
  let dx=W[0]-S[0], dy=W[1]-S[1], d=Math.hypot(dx,dy);
  const mx=LEN1+LEN2-1, mn=Math.abs(LEN1-LEN2)+2;
  if (d>mx){ dx*=mx/d; dy*=mx/d; d=mx; } else if (d<mn){ if(d<.01){dx=0;dy=mn;} else {dx*=mn/d;dy*=mn/d;} d=mn; }
  const a=(d*d+LEN1*LEN1-LEN2*LEN2)/(2*d), hh=Math.sqrt(Math.max(0,LEN1*LEN1-a*a)), ux=dx/d, uy=dy/d;
  const e1=[S[0]+ux*a-uy*hh, S[1]+uy*a+ux*hh], e2=[S[0]+ux*a+uy*hh, S[1]+uy*a-ux*hh];
  const sc=e=>out*e[0]+.5*e[1];
  return {E: sc(e1)>=sc(e2)?e1:e2, W:[S[0]+dx,S[1]+dy]};
}
function arm(S,p,out){
  // the elbow sits one forearm-length from the wrist, towards the elbow hint (ex, ey);
  // the upper arm may look short, as it does when an arm points towards the camera
  const W=[p.x,p.y]; let dx=p.ex-W[0], dy=p.ey-W[1]; const d=Math.hypot(dx,dy)||1;
  const E=[W[0]+dx/d*LEN2, W[1]+dy/d*LEN2];
  limb([S,E],27,SHIRT,SHIRT_D);                   // sleeve
  limb([E,W],17,SKIN,SKIN_D);                     // forearm
  hand(p);
}
function face(cx,cy,mood,blink){
  // long hair behind the head and shoulders
  ctx.fillStyle=HAIR; ctx.strokeStyle='#2B1A13'; ctx.lineWidth=1.6;
  ctx.beginPath(); ctx.moveTo(cx-44,cy-4); ctx.bezierCurveTo(cx-52,cy-62,cx+52,cy-62,cx+44,cy-4);
  ctx.bezierCurveTo(cx+50,cy+30,cx+52,cy+58,cx+46,cy+82); ctx.quadraticCurveTo(cx,cy+92,cx-46,cy+82);
  ctx.bezierCurveTo(cx-52,cy+58,cx-50,cy+30,cx-44,cy-4); ctx.closePath(); ctx.fill(); ctx.stroke();
  // neck (slim)
  ctx.fillStyle=SKIN; ctx.strokeStyle=SKIN_D; ctx.lineWidth=1.8;
  rrect(188,128,24,52,8); ctx.fill(); ctx.stroke();
  // face
  ctx.fillStyle=SKIN; ctx.beginPath();
  ctx.moveTo(cx-34,cy-8); ctx.bezierCurveTo(cx-35,cy+22,cx-24,cy+42,cx,cy+44);
  ctx.bezierCurveTo(cx+24,cy+42,cx+35,cy+22,cx+34,cy-8); ctx.bezierCurveTo(cx+34,cy-36,cx-34,cy-36,cx-34,cy-8);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  // fringe and side locks in front of the ears
  ctx.fillStyle=HAIR; ctx.strokeStyle='#2B1A13'; ctx.lineWidth=1.4;
  ctx.beginPath(); ctx.moveTo(cx-37,cy+26); ctx.bezierCurveTo(cx-44,cy-6,cx-38,cy-34,cx-4,cy-38);
  ctx.bezierCurveTo(cx+30,cy-38,cx+42,cy-8,cx+37,cy+26);
  ctx.bezierCurveTo(cx+36,cy+8,cx+34,cy-6,cx+30,cy-16);                       // right side lock
  ctx.bezierCurveTo(cx+16,cy-14,cx-2,cy-22,cx-14,cy-30);                     // swept fringe
  ctx.bezierCurveTo(cx-26,cy-18,cx-31,cy-2,cx-33,cy+6);
  ctx.bezierCurveTo(cx-34,cy+14,cx-35,cy+20,cx-37,cy+26); ctx.closePath(); ctx.fill(); ctx.stroke();
  // cheeks
  ctx.fillStyle='rgba(235,110,110,.20)'; [-1,1].forEach(s=>{ ctx.beginPath(); ctx.arc(cx+s*20,cy+17,7.5,0,7); ctx.fill(); });
  // eyes with lashes
  ctx.fillStyle='#2B2B2B'; [-1,1].forEach(s=>{ ctx.beginPath(); ctx.ellipse(cx+s*13,cy+3,4.4,blink?.7:5.4,0,0,7); ctx.fill(); });
  ctx.strokeStyle='#2B1A13'; ctx.lineWidth=2; ctx.lineCap='round';
  [-1,1].forEach(s=>{ ctx.beginPath(); ctx.moveTo(cx+s*8.5,cy-1.5); ctx.quadraticCurveTo(cx+s*13,cy-4.5,cx+s*18,cy-1); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx+s*17.5,cy-.5); ctx.lineTo(cx+s*21,cy-3.5); ctx.stroke(); });
  // eyebrows (thin, arched)
  ctx.strokeStyle=HAIR; ctx.lineWidth=2.2;
  [-1,1].forEach(s=>{ ctx.beginPath();
    if (mood==='sad'){ ctx.moveTo(cx+s*21,cy-10); ctx.lineTo(cx+s*7,cy-16); }
    else { var u=(mood==='ask')?5:0; ctx.moveTo(cx+s*21,cy-10-u); ctx.quadraticCurveTo(cx+s*13,cy-17-u,cx+s*6,cy-12-u); }
    ctx.stroke(); });
  ctx.strokeStyle=SKIN_D; ctx.lineWidth=1.8; ctx.beginPath(); ctx.moveTo(cx,cy+8); ctx.quadraticCurveTo(cx+3.5,cy+15,cx-1,cy+16); ctx.stroke();
  // lips
  ctx.strokeStyle='#B84A5E'; ctx.fillStyle='#D2687B'; ctx.lineWidth=2.4; ctx.lineJoin='round'; ctx.beginPath();
  if (mood==='smile'){ ctx.moveTo(cx-12,cy+25); ctx.quadraticCurveTo(cx,cy+36,cx+12,cy+25); ctx.quadraticCurveTo(cx,cy+28,cx-12,cy+25); }
  else if (mood==='sad'){ ctx.moveTo(cx-10,cy+31); ctx.quadraticCurveTo(cx,cy+23,cx+10,cy+31); ctx.quadraticCurveTo(cx,cy+28,cx-10,cy+31); }
  else { ctx.moveTo(cx-9,cy+28); ctx.quadraticCurveTo(cx,cy+31,cx+9,cy+28); ctx.quadraticCurveTo(cx,cy+26,cx-9,cy+28); }
  ctx.fill(); ctx.stroke();
}
function drawArrow(pose, a){
  const t=pose.t; if (!a || t<a[0]-.2 || t>a[1]+.6) return;
  const al=Math.max(0,Math.min(1,Math.min((t-(a[0]-.2))/.5,(a[1]+.6-t)/.4)));
  ctx.save(); ctx.globalAlpha=.9*al; ctx.strokeStyle='#F2B705'; ctx.fillStyle='#F2B705'; ctx.lineWidth=5; ctx.lineCap='round'; ctx.setLineDash([2,11]);
  const [A,C,B]=[a[2],a[3],a[4]]; ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.quadraticCurveTo(C[0],C[1],B[0],B[1]); ctx.stroke(); ctx.setLineDash([]);
  const ang=Math.atan2(B[1]-C[1],B[0]-C[0]); ctx.translate(B[0],B[1]); ctx.rotate(ang);
  ctx.beginPath(); ctx.moveTo(8,0); ctx.lineTo(-8,-9); ctx.lineTo(-8,9); ctx.closePath(); ctx.fill(); ctx.restore();
}
function drawSigner(pose, mood, arrow){
  ctx.setTransform(1,0,0,1,0,0);
  const g=ctx.createLinearGradient(0,0,0,880); g.addColorStop(0,'#F4FAFD'); g.addColorStop(1,'#E5F1F6');
  ctx.fillStyle=g; ctx.fillRect(0,0,800,880);
  const z=pose.cam.z; ctx.setTransform(2*z,0,0,2*z,2*(200-z*pose.cam.x),2*(220-z*pose.cam.y));
  // torso
  ctx.fillStyle=SHIRT; ctx.strokeStyle=SHIRT_D; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(130,208); ctx.quadraticCurveTo(130,182,162,178); ctx.lineTo(238,178);
  ctx.quadraticCurveTo(270,182,270,208); ctx.lineTo(278,445); ctx.lineTo(122,445); ctx.closePath(); ctx.fill(); ctx.stroke();
  const blink = (performance.now() % 3300) < 130;
  face(200+pose.hx, 100+pose.hy, mood, blink);
  ctx.fillStyle=SKIN; ctx.strokeStyle=SHIRT_D; ctx.lineWidth=3;               // rounded neckline
  ctx.beginPath(); ctx.moveTo(180,170); ctx.lineTo(176,177); ctx.quadraticCurveTo(200,220,224,177); ctx.lineTo(220,170); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.moveTo(176,177); ctx.quadraticCurveTo(200,220,224,177); ctx.stroke();
  arm(SH.L, pose.L, 1);                           // left arm first, right arm on top of it
  arm(SH.R, pose.R, -1);
  drawArrow(pose, arrow);
}

/* ---------------------------------------------------------------- playback */
const SPEEDS = [['Normal',1],['Slow',0.6],['Very slow',0.35]];
let idx = 0, playing = true, si = 1, spd = SPEEDS[1][1], elapsed = 0, last = performance.now(), dragging = false;

function buildDots(){
  const d = $('dots'); d.innerHTML = '';
  if (n < 2) return;
  SEQ.forEach((s,i) => {
    const b = document.createElement('button');
    b.className = 'dot'; b.setAttribute('aria-label', 'Show sign ' + (i+1) + ': ' + s.label);
    b.onclick = () => show(i); d.appendChild(b);
  });
}
function stepsOf(t){
  return t.split(/,?\s+then\s+|;\s+|\.\s+(?=[A-Z])/).map(x => x.trim().replace(/\.$/,'')).filter(Boolean)
          .map(x => x.charAt(0).toUpperCase() + x.slice(1) + '.');
}
let stepEls = [];
function show(i){
  idx = Math.max(0, Math.min(n-1, i)); elapsed = 0;
  const s = SEQ[idx], sc = SCRIPTS[s.key];
  $('count').textContent = n > 1 ? 'Sign ' + (idx+1) + ' of ' + n : 'Sign';
  $('label').textContent = s.label;
  $('how').innerHTML = ''; stepEls = [];
  stepsOf(s.how).forEach(t => { const li = document.createElement('li'); li.textContent = t; $('how').appendChild(li); stepEls.push(li); });
  $('big').textContent = s.emoji;
  const has = !!sc;
  cv.classList.toggle('hide', !has || use3d); $('v3').classList.toggle('hide', !has || !use3d); $('big').classList.toggle('hide', has);
  $('scrubrow').classList.toggle('hide', !has); $('replay').classList.toggle('hide', !has);
  $('phase').classList.add('hide');
  $('cap').textContent = has ? '' : 'No animation for this sign yet. Follow the written steps.';
  $('cap').style.display = has ? 'none' : '';
  document.querySelectorAll('.dot').forEach((d,k) => d.classList.toggle('on', k === idx));
  $('prev').disabled = idx === 0; $('next').disabled = idx === n-1;
  const info = $('info'); info.classList.remove('swap'); void info.offsetWidth; info.classList.add('swap');
  $('live').textContent = s.label + '. How to sign it: ' + s.how;
}
function render(t, sc){
  const pose = sample(sc.frames, t); pose.t = t;
  if (use3d) Signer3D.apply(pose, sc.mood); else drawSigner(pose, sc.mood, ARROWS[SEQ[idx].key]);
  const T = sc.frames[sc.frames.length-1].t;
  $('scrub').value = Math.round(t/T*1000); $('pct').textContent = Math.round(t/T*100) + '%';
  const k = Math.min(pose.st, stepEls.length-1);
  stepEls.forEach((li,j) => li.classList.toggle('on', stepEls.length > 1 && j === k));
  const ph = $('phase');
  if (stepEls.length > 1){ ph.classList.remove('hide'); ph.textContent = 'Step ' + (k+1) + ' of ' + stepEls.length; }
  else ph.classList.add('hide');
}
function tick(now){
  const dt = Math.max(0, Math.min(now - last, 100)); last = now;
  const s = SEQ[idx], sc = SCRIPTS[s.key];
  if (playing && !dragging) elapsed += dt * spd;
  if (sc){
    const T = sc.frames[sc.frames.length-1].t;
    if (!dragging) render((elapsed/1000) % T, sc);
    if (playing && !dragging && idx < n-1 && elapsed/1000 >= T*3) show(idx+1);
  } else if (playing && idx < n-1 && elapsed >= 5500) show(idx+1);
  requestAnimationFrame(tick);
}
$('scrub').oninput = e => {
  dragging = true; const sc = SCRIPTS[SEQ[idx].key];
  if (sc){ const T = sc.frames[sc.frames.length-1].t; render(e.target.value/1000*T, sc); }
};
$('scrub').onchange = () => { dragging = false; elapsed = 0; };
$('replay').onclick = () => { elapsed = 0; playing = true; $('play').textContent = '⏸ Pause'; };
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
$('prev').onclick = () => show(idx-1);
$('next').onclick = () => show(idx+1);
$('slow').onclick = () => { si = (si+1) % SPEEDS.length; spd = SPEEDS[si][1]; $('slow').textContent = '🐢 Speed: ' + SPEEDS[si][0]; };
let use3d = false;
const BASES = ['https://mariamdeda.github.io/HCI---SignSpeak/', (location.origin && location.origin !== 'null' ? location.origin : '') + '/app/static/'];
function loadScript(u){ return new Promise((ok,no)=>{ const e=document.createElement('script'); e.src=u; e.onload=ok; e.onerror=no; document.head.appendChild(e); }); }
async function load3d(){
  try { const c=document.createElement('canvas'); if(!(c.getContext('webgl2')||c.getContext('webgl'))) return; } catch(e){ return; }
  for (const base of BASES){
    try {
      if (!window.Signer3D) await loadScript(base + 'signer3d_v9.js');
      await Signer3D.init($('v3'), base + 'signer.vrm?v=4');
      use3d = true; show(idx); const bd=document.createElement('div'); bd.textContent='3D v9'; bd.style.cssText='position:absolute;left:8px;bottom:6px;font:600 10px sans-serif;color:#5F7080;opacity:.7'; $('v3').appendChild(bd); return;
    } catch(e) { console.warn('3D avatar failed from', base, e); }
  }
}
buildDots(); show(0); requestAnimationFrame(tick); load3d();
</script></body></html>"""


def player_html(keys, nonce=0):
    seq = []
    for k in keys:
        i = info(k)
        seq.append({"key": k, "label": pretty(k).capitalize(), "emoji": i["emoji"], "how": i["how"]})
    data = json.dumps(seq).replace("</", "<\\/")
    return _PLAYER.replace("__DATA__", data).replace("__NONCE__", str(nonce))
