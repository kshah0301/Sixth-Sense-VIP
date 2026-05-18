# demo_audio_thread.py

import time
import numpy as np
import sounddevice as sd

from instrumental_beeping import AudioThread  # uses your InstrumentSampler inside

def choose_output_device():
    print("=== Available audio devices ===")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        print(f"{i:>2}: {d['name']} | in={d['max_input_channels']} out={d['max_output_channels']}")
    print("================================")

    # Choose a device index with out > 0
    idx = int(input("Select output device index (one with out>0): ").strip() or "0")
    sd.default.device = idx
    info = sd.query_devices(idx)
    print(f"Using device {idx}: {info['name']} (out={info['max_output_channels']})")
    return idx

def main():
    choose_output_device()

    # Start AudioThread with generous timeout so it doesn't go idle too quickly
    audio = AudioThread(
        samplerate=44100,
        beep_duration=0.15,
        min_interval=0.1,
        max_interval=1.0,
        min_distance=0,
        max_distance=150,
        min_amplitude=0.2,
        max_amplitude=1.0,
        update_timeout=5.0,   # plenty of time for this demo
    )

    # Reasonable starting amplitude; your update_params can overwrite this
    audio.overall_amplitude = 0.5

    audio.start()
    print("[demo] AudioThread started. Simulating fingertip movement...")

    try:
        # Fixed target in the middle of a fake 640x480 frame
        target_center = (320.0, 240.0)

        # Sweep the fingertip left/right around the target for a while
        t0 = time.time()
        for step in range(40):
            t = time.time() - t0

            # Make the finger move in a horizontal sine wave around the target
            x = target_center[0] + 150 * np.sin(t)   # oscillate left/right
            y = target_center[1]                     # same vertical position
            finger_pos = (x, y)

            # Euclidean distance in pixels
            distance = float(np.linalg.norm(np.array(finger_pos) - np.array(target_center)))

            print(f"[demo] step={step:02d} finger={tuple(map(int, finger_pos))} "
                  f"target={tuple(map(int, target_center))} dist={distance:.1f}")

            # This calls your update_params(finger_pos, target_center, distance)
            audio.update_params(finger_pos=finger_pos,
                                target_center=target_center,
                                distance=distance)

            time.sleep(0.3)  # update more often than update_timeout

        print("[demo] Done simulating movement. Waiting a bit before shutdown...")
        time.sleep(2.0)

    finally:
        print("[demo] Stopping AudioThread...")
        audio.stop()
        audio.join()
        print("[demo] AudioThread stopped.")

if __name__ == "__main__":
    main()
"""
import sounddevice as sd
from instrumental_beeping import InstrumentSampler

print("=== Devices ===")
for i, d in enumerate(sd.query_devices()):
    print(f"{i:>2}: {d['name']} | in={d['max_input_channels']} out={d['max_output_channels']}")
print("===============")

# IMPORTANT: use the index that you know works (the one where you heard sound)


sampler = InstrumentSampler(samplerate=44100)
sampler.play_directional_sample("left", amplitude=0.5, pan=-1.0)
sampler.play_directional_sample("right", amplitude=0.5, pan=+1.0)
"""