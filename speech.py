"""Optional speech helpers. The app works without them (typed reply / no TTS)."""
import json
import threading

import config

_vosk_model = None


def listen(seconds=config.LISTEN_SECONDS) -> str:
    """Record from the microphone for `seconds` and transcribe with Vosk (offline)."""
    global _vosk_model
    try:
        import sounddevice as sd
        import vosk
    except ImportError as e:
        raise RuntimeError("Install sounddevice and vosk to use speech input") from e
    if not config.VOSK_MODEL_DIR.exists():
        raise RuntimeError(f"Vosk model not found at {config.VOSK_MODEL_DIR}. "
                           "Download vosk-model-small-en-us-0.15 from alphacephei.com/vosk/models")
    if _vosk_model is None:
        vosk.SetLogLevel(-1)
        _vosk_model = vosk.Model(str(config.VOSK_MODEL_DIR))
    rate = 16000
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    rec = vosk.KaldiRecognizer(_vosk_model, rate)
    rec.AcceptWaveform(audio.tobytes())
    return json.loads(rec.FinalResult()).get("text", "").strip()


def speak(text: str):
    """Non-blocking text-to-speech (pyttsx3). Silently does nothing if unavailable."""
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def transcribe_wav(wav_bytes: bytes) -> str:
    """Transcribe a WAV recording (e.g. from st.audio_input) offline with Vosk."""
    global _vosk_model
    import io
    import wave

    import numpy as np
    try:
        import vosk
    except ImportError as e:
        raise RuntimeError("Voice input needs the 'vosk' package. Run: python -m pip install vosk") from e
    if not config.VOSK_MODEL_DIR.exists():
        raise RuntimeError(f"Voice model not found. Download 'vosk-model-small-en-us-0.15' from "
                           f"alphacephei.com/vosk/models and unzip it into: {config.VOSK_MODEL_DIR.parent}")
    if _vosk_model is None:
        vosk.SetLogLevel(-1)
        _vosk_model = vosk.Model(str(config.VOSK_MODEL_DIR))
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise RuntimeError("Unsupported audio format (expected 16-bit).")
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if rate != 16000:
        n_out = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio).astype(np.int16)
    rec = vosk.KaldiRecognizer(_vosk_model, 16000)
    rec.AcceptWaveform(audio.tobytes())
    return json.loads(rec.FinalResult()).get("text", "").strip()
