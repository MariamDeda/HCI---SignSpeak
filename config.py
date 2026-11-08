"""Central configuration for the SignSpeak Phase 1 prototype.
Edit values here instead of changing code."""
from pathlib import Path

ROOT = Path(__file__).parent

# ---- Vocabulary (ASL). Verify each sign against a reference (e.g. Handspeak,
# ASL University) and perform it the SAME way every time. 'unknown' = idle hand,
# random movement, anything that is not one of the signs.
SIGNS = ["hello", "goodbye", "thank_you", "i_love_you", "yes", "no", "unknown"]
UNKNOWN = "unknown"
DISPLAY = {"thank_you": "thank you", "i_love_you": "I love you"}

# ---- Sequence / features
SEQ_LEN = 30            # frames per window (about 1 s at 30 FPS)
FEATURE_DIM = 65        # 21 landmarks x 3 (wrist-relative, scaled) + wrist x,y
MIN_HAND_FRACTION = 0.5 # a window needs a hand in at least this share of frames

# ---- Camera
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
MIRROR = True           # keep identical for collection AND inference

# ---- Decision layer (stabilization)
CONF_THRESHOLD = 0.80
STABLE_PREDICTIONS = 5  # consecutive confident, identical predictions to commit
COOLDOWN_S = 1.5
RELEASE_PREDICTIONS = 10  # idle predictions needed before the same sign can repeat
PREDICT_EVERY = 1       # run the model every k frames

# ---- Paths
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "signspeak.keras"
META_PATH = MODEL_DIR / "meta.json"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"

# ---- Speech (optional)
VOSK_MODEL_DIR = MODEL_DIR / "vosk-model-small-en-us-0.15"
LISTEN_SECONDS = 5

# ---- Voice output (the "Speak aloud" switch on the signer page)
# "browser": the browser speaks (works on your laptop AND when the app is deployed).
# "server" : the computer running the app speaks through pyttsx3 (only for local use).
SPEAK_VIA = "browser"
