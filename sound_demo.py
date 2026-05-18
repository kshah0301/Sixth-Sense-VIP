 audio = AudioThread(
        beep_duration=0.15, min_interval=0.1, max_interval=1.0,
        min_distance=0, max_distance=200,
        min_amplitude=0.2, max_amplitude=1.0, update_timeout=0.5
    )
    audio.start()

