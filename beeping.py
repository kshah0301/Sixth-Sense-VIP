import time
import numpy as np
import threading
import tempfile
import os
import subprocess
import sounddevice as sd
import soundfile as sf
from midiutil import MIDIFile

class FluidSynthBeep(threading.Thread):
    def __init__(
        self,
        soundfont_path,
        instrument=26,
        samplerate=48000,
        duration=5,
        release=2
    ):
        super().__init__()
        self.soundfont_path = soundfont_path
        self.instrument = instrument
        self.samplerate = samplerate
        self.duration = duration  # note hold time in seconds
        self.release = release    # release tail in seconds
        self.running = False
        self.current_distance = None
        self.direction = True
        self.lock = threading.Lock()

        # Pre-render mono note buffers including reverb tail and envelope
        self.note_buffers = {}
        self._prepare_note_samples([81, 79, 76, 72])  # notes A5, G5, E5, C5

        # Active voices for callback mixing
        self.active_voices = []  # list of {'data': stereo_array, 'pos': int}

        # Audio stream with callback for mixing
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            blocksize=1024,
            latency=0.05,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()

    def _apply_fade(self, mono, total_frames):
        # Linear fade-out over release frames
        release_frames = int(self.release * self.samplerate)
        envelope = np.ones(total_frames, dtype=np.float32)
        if release_frames > 0:
            fade = np.linspace(1.0, 0.0, release_frames, dtype=np.float32)
            envelope[-release_frames:] = fade
        return mono * envelope

    def _prepare_note_samples(self, notes):
        record_len = self.duration + self.release
        total_frames = int(self.samplerate * record_len)
        for note in notes:
            # Temporary files
            mid_fd, mid_path = tempfile.mkstemp(suffix='.mid')
            wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(mid_fd); os.close(wav_fd)

            # Create MIDI with CC91 reverb and note
            mf = MIDIFile(1)
            mf.addTrackName(0, 0, 'Track')
            mf.addTempo(0, 0, 120)
            # Reverb level
            mf.addControllerEvent(0, 0, 0, 91, int(0.4 * 127))
            mf.addProgramChange(0, 0, 0, self.instrument)
            mf.addNote(0, 0, note, 0, self.duration, 100)
            # Write MIDI
            with open(mid_path, 'wb') as f:
                mf.writeFile(f)

            # Render WAV
            subprocess.run([
                'fluidsynth', '-ni', self.soundfont_path,
                mid_path, '-F', wav_path,
                '-r', str(self.samplerate)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load mono data
            mono, _ = sf.read(wav_path)
            os.remove(mid_path); os.remove(wav_path)
            if mono.ndim > 1:
                mono = mono[:, 0]

            # Trim or pad to total_frames
            if mono.shape[0] < total_frames:
                mono = np.pad(mono, (0, total_frames - mono.shape[0]), mode='constant')
            else:
                mono = mono[:total_frames]

            # Apply fade for natural decay
            mono = self._apply_fade(mono, total_frames)
            self.note_buffers[note] = mono.astype(np.float32)

    def select_note_and_interval(self, distance):
        # clamp
        d = max(0.0, float(distance))

        # choose note by coarse range
        if d < 100:     note = 81  # A5
        elif d < 250:   note = 79  # G5
        elif d < 400:   note = 76  # E5
        else:           note = 72  # C5

        # continuous interval mapping (faster when closer)
        #  d in [0, 1000+]  -> interval in [0.05, 1.7]
        d_norm = min(d, 1000.0) / 1000.0
        # exponential-ish easing for better feel
        interval = 0.05 + (1.7 - 0.05) * (d_norm ** 1.7)
        return note, float(interval)


    def update_distance(self, distance, direction):
        with self.lock:
            self.current_distance = distance
            self.direction = direction

    def _audio_callback(self, outdata, frames, time_info, status):
        # pre-zero output
        out = outdata
        out[:] = 0.0

        remove = []
        with self.lock:
            # simple cap to avoid unbounded growth
            if len(self.active_voices) > 8:
                self.active_voices = self.active_voices[-8:]

            for i, voice in enumerate(list(self.active_voices)):
                data = voice['data']
                pos  = voice['pos']
                end  = pos + frames
                chunk = data[pos:end]

                if chunk.shape[0] < frames:
                    # write what we have, then mark for removal
                    if chunk.shape[0] > 0:
                        out[:chunk.shape[0], :] += chunk
                    remove.append(voice)
                else:
                    out[:] += chunk

                voice['pos'] = end

            for v in remove:
                try: self.active_voices.remove(v)
                except ValueError: pass

        # soft limiter: y = x / max(1, |x| peak)
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out[:] = out / peak


    def run(self):
        try:
            self.running = True
            max_dist = 150
            while self.running:
                with self.lock:
                    d = self.current_distance; dir_right = self.direction
                if d is None:
                    time.sleep(0.02); continue
                pan = 0.0 if d <= 0 else (d/max_dist if dir_right else -d/max_dist)
                pan = np.clip(pan, -1.0, 1.0)
                note, interval = self.select_note_and_interval(d)
                mono = self.note_buffers.get(note)
                if mono is not None:
                    left = mono * (1-pan)/2.0
                    right= mono * (1+pan)/2.0
                    stereo = np.column_stack((left, right))
                    with self.lock:
                        self.active_voices.append({'data': stereo, 'pos': 0})
                time.sleep(interval)
        finally:
            try: self.stream.stop()
            except Exception: pass
            try: self.stream.close()
            except Exception: pass
           
    def stop(self):
        self.running = False

if __name__ == '__main__':
    soundfont = 'GeneralUser_GS_v2.0.2--doc_r4/GeneralUser-GS/GeneralUser-GS.sf2'
    beep = FluidSynthBeep(soundfont_path=soundfont)
    beep.start()
    try:
        prev=None; direction=True; start=time.time()
        while True:
            t=time.time()-start
            d=70+80*np.sin(0.4*t)
            if prev is not None and prev>5 and d<5:
                direction=not direction
            prev=d; beep.update_distance(d, direction)
            time.sleep(0.02)
    except KeyboardInterrupt:
        beep.stop(); beep.join()