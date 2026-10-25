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
