import numpy as np
import sounddevice as sd
import librosa
import sys, os

SR = 44100
FILES = {"left":"left.wav","right":"right.wav","up":"up.wav","down":"down.wav"}

def load_mono(path, sr=SR):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def pan_stereo(y, pan=0.0):
    pan = float(np.clip(pan, -1.0, 1.0))      # -1=L, +1=R
    lg = (1.0 - pan) * 0.5
    rg = (1.0 + pan) * 0.5
    stereo = np.column_stack([y*lg, y*rg])
    return np.clip(stereo, -1.0, 1.0).astype(np.float32)

def play_direction(direction, pan=0.0, amp=0.4):
    direction = direction.lower()
    path = FILES.get(direction)
    if not path:
        raise ValueError(f"Unknown direction: {direction}")
    y = load_mono(path)
    y *= float(amp)                            # volume
    stereo = pan_stereo(y, pan)
    sd.play(stereo, SR)
    sd.wait()
    print(f"Played {direction} | pan={pan:.2f} | amp={amp}")

if __name__ == "__main__":
    # Usage: python play_direction.py up  or  python play_direction.py left -0.8
    direction = sys.argv[1] if len(sys.argv) > 1 else "up"
    pan = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    play_direction(direction, pan=pan, amp=0.4)