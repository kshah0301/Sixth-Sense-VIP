import time
import numpy as np

from instrumental_beeping import AudioThread
from directional_audio_generator import check_directional_audio_files


def main():
    # Check for directional wavs
    check_directional_audio_files()

    # Start audio thread
    audio = AudioThread(
        beep_duration=0.15,
        min_interval=0.1,
        max_interval=1.0,
        min_distance=0,
        max_distance=200,
        min_amplitude=0.2,
        max_amplitude=1.0,
        update_timeout=0.5,
    )
    audio.start()

    # Fixed target center; simulate finger moving around it
    target_center = (320, 240)

    try:
        # Cycle through directions for a short demo
        sequence = [
            (200, 240),  # left of target -> "right" cue
            (440, 240),  # right of target -> "left" cue
            (320, 120),  # above target -> "down" cue
            (320, 360),  # below target -> "up" cue
        ]

        for _ in range(3):  # repeat a few times
            for finger_pos in sequence:
                # Distance controls cue rate; smaller = faster
                dx = target_center[0] - finger_pos[0]
                dy = target_center[1] - finger_pos[1]
                distance = float(np.hypot(dx, dy))

                audio.update_params(finger_pos, target_center, distance)
                time.sleep(0.6)

        # Also show a smooth orbit around the target
        radius = 120
        for t in np.linspace(0, 2 * np.pi, 60):
            finger_pos = (
                int(target_center[0] + radius * np.cos(t)),
                int(target_center[1] + radius * np.sin(t)),
            )
            audio.update_params(finger_pos, target_center, radius)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        audio.join()


if __name__ == "__main__":
    main()


