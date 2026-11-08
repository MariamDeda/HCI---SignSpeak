# SignSpeak - Phase 1 prototype

Webcam -> MediaPipe hand landmarks -> LSTM sign classifier -> stabilization -> text,
with a two-sided conversation panel (typed or Vosk speech reply) and optional TTS.
Vocabulary (edit `config.SIGNS`): hello, goodbye, thank_you, i_love_you, unknown (ASL).

## Setup (Python 3.11)
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
Keep `mediapipe==0.10.14`: newer releases removed the `mp.solutions` API this code uses.
If pip complains about `ml-dtypes`, run `pip install "ml-dtypes~=0.3.1"`.
Optional speech input: download `vosk-model-small-en-us-0.15` from alphacephei.com/vosk/models
and unzip it into `models/`.

## Workflow
1. **Collect** (each signer, same hand every time; aim for 25-40 per sign):
       python collect_data.py --person zuhaa
   Keys: 1-5 pick sign, SPACE record, u undo, q quit. For `unknown` record idle hands,
   random gestures and the hand entering/leaving the frame.
   Extra conditions (e.g. low light) go in their own folder:
       python collect_data.py --person zuhaa --data-dir data_lowlight
2. **Train** (leave one signer out if you have 2+ signers):
       python train.py --test-person mariam
   Writes `models/signspeak.keras`, `results/test_report.txt`, `results/test_confusion.png`.
3. **Run the prototype:**
       streamlit run app.py
   Predictions are logged to `logs/predictions_*.csv`.
4. **Experiments:**
       python evaluate.py acc --data data_lowlight --name lowlight
       python evaluate.py latency --log logs/predictions_XXXX.csv
       python evaluate.py stab --log logs/predictions_XXXX.csv --script hello,thank_you,goodbye
   For `stab`, perform exactly the scripted signs in order during one app session
   (hand out of frame between signs) and put the same list in --script.

## Notes / known limits
- Mirror mode and the signing hand must be the same in collection and inference.
- Landmarks only are saved - no video - which keeps participant data anonymous.
- Latency excludes camera-driver delay and UI render time; say so in the report.
- Dynamic sign timing depends on camera FPS; collect and run on the same machine/camera.
- Streamlit reruns the script on every widget click, which restarts the camera loop;
  this is normal for the prototype.
- Targets (92% accuracy, <500 ms) come from the proposal; report what you measure.

## The app (two roles)
`python -m streamlit run app.py` opens a home page: **"I use sign language"** or **"I don't know sign language"**.
- **Signer page** (`view_signer.py`): turn the camera on and sign; recognised signs appear in big text
  (size switch: Large / Larger / Huge) and can be spoken aloud (browser voice).
- **Hearing page** (`view_hearing.py`): type, tap a word, or use the microphone; the message is mapped to
  signs (`signs_ref.py`) and shown big, slow and on repeat. If `data/<sign>/*.npy` recordings exist, the
  sign is animated from them; otherwise a large picture and a written "how to sign it" are shown.
- Add signs by editing `config.SIGNS`; add spoken synonyms and a description in `signs_ref.SIGN_INFO`.
- Colours and fonts: the `:root{...}` block at the top of `theme.py`.
- Voice input (microphone on the hearing page) needs `vosk` and the Vosk model in `models/`.
