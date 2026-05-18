import os
import shutil
import subprocess
import sounddevice as sd
import numpy as np

try:
    from kittentts import KittenTTS
except ImportError:
    KittenTTS = None
    print("Warning: 'kittentts' package not installed; KittenTTS-based audio will be disabled.")

m = None
if KittenTTS is not None:
    try:
        m = KittenTTS("KittenML/kitten-tts-nano-0.2")
    except Exception as e:
        # This typically happens when 'espeak'/phonemizer setup is not right
        m = None
        print(f"Warning: KittenTTS could not be initialized ({e}); falling back to system TTS.")

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

def _speak_with_system_tts(text: str) -> bool:
    """
    Try to speak using a system TTS tool (espeak on Linux/macOS via Homebrew,
    or 'say' on macOS). Returns True if something was invoked, False otherwise.
    """
    # Prefer espeak if available
    espeak_path = shutil.which("espeak")
    if espeak_path:
        try:
            subprocess.run([espeak_path, text], check=True)
            return True
        except Exception as e:
            print(f"[TTS fallback] espeak failed: {e}")

    # macOS built-in TTS
    say_path = shutil.which("say")
    if say_path:
        try:
            subprocess.run([say_path, text], check=True)
            return True
        except Exception as e:
            print(f"[TTS fallback] say failed: {e}")

    return False


def speak(text, voice="expr-voice-2-m", sr=24000):
    # First try KittenTTS if initialized
    if m is not None:
        try:
            audio = m.generate(text, voice=voice)
            y = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            y = np.squeeze(y)
            y = np.clip(y, -1.0, 1.0).astype(np.float32)   # make PortAudio happy
            sd.play(y, sr)
            sd.wait()
            return
        except Exception as e:
            print(f"[KittenTTS] playback failed ({e}); falling back to system TTS.")

    # If KittenTTS is unavailable or fails, try system TTS tools
    if _speak_with_system_tts(text):
        return

    # Last resort: just print so the program keeps working
    print(f"[TTS disabled] {text}")

if __name__ == "__main__":
    # Quick manual check when running this file directly.
    speak("Name the item you are looking for.")
    speak("Hello.")
    speak("Oreo.")
