import os
from datetime import datetime
import threading

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import whisper

class WhisperTranscriber(threading.Thread):
    def __init__(self, duration=10, sample_rate=16000, model_name="base", **kwargs):
        """
        Parameters:
          - duration: Recording duration in seconds.
          - sample_rate: Sampling rate for the recording.
          - model_name: The Whisper model variant to load ("base", "small", etc.).
        """
        super().__init__(**kwargs)
        self.duration = duration
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.transcription = None

    def record_audio(self):
        print(f"Recording for {self.duration} seconds...")
        audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")
        return audio

    def save_wav_to_directory(self, audio, directory="recordings"):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"whisper_recording_{timestamp}.wav"
        file_path = os.path.join(directory, filename)
        wav.write(file_path, self.sample_rate, audio)
        print(f"Audio saved to: {file_path}")
        return file_path

    def transcribe_audio(self, audio_path):
        print(f"Loading Whisper model '{self.model_name}'...")
        model = whisper.load_model(self.model_name)
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        return result["text"]

    def run(self):
        audio = self.record_audio()
        #audio_path = self.save_wav_to_directory(audio)
        self.transcription = self.transcribe_audio(audio)
        print("Transcription complete.")
        print("\nTranscription:")
        print(self.transcription)

if __name__ == "__main__":
    # Example of using the thread directly.
    transcriber = WhisperTranscriber(duration=5, sample_rate=16000, model_name="base")
    transcriber.start()
    transcriber.join()  # Wait for the thread to finish.
