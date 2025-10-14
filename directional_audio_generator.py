import numpy as np
import soundfile as sf
import subprocess
import os

def generate_directional_audio_files(samplerate=44100):
    """
    Generate directional audio files (left.wav, right.wav, up.wav, down.wav)
    using mimic3 TTS or fallback synthesized audio.
    """
    directions = ["left", "right", "up", "down"]
    
    for direction in directions:
        filename = f"{direction}.wav"
        if not os.path.exists(filename):
            print(f"Generating {filename}...")
            try:
                # Try mimic3 TTS first
                generate_with_mimic3(direction, filename)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"mimic3 failed for {direction}: {e}")
                # Fallback to synthesized audio
                generate_synthesized_audio(direction, filename, samplerate)
        else:
            print(f"{filename} already exists")

def generate_with_mimic3(direction, filename):
    """Generate directional audio using mimic3 TTS."""
    mimic3_cmd = [
        "mimic3",
        "--voice", "en_US/ljspeech_low",
        "--stdout"
    ]
    
    with open(filename, "wb") as audio_file:
        subprocess.run(mimic3_cmd, input=direction.encode(), stdout=audio_file, check=True)
    print(f"Generated {filename} with mimic3")

def generate_synthesized_audio(direction, filename, samplerate=44100):
    """Create synthesized beep patterns for each direction."""
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(samplerate * duration))
    
    # Different frequency patterns for each direction
    if direction == "left":
        # Low frequency, left-panned
        frequency = 200
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    elif direction == "right":
        # High frequency, right-panned  
        frequency = 800
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    elif direction == "up":
        # Rising tone
        frequency = 400 + 200 * t
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    elif direction == "down":
        # Falling tone
        frequency = 600 - 200 * t
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    else:
        # Default beep
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Save as mono WAV
    sf.write(filename, audio, samplerate)
    print(f"Created synthesized {filename}")

if __name__ == "__main__":
    # Generate all directional audio files
    generate_directional_audio_files()
