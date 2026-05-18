import os
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import whisper

# default
duration = 5  
sample_rate = 16000 # sampling rate  

def record_audio(duration, sample_rate):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return audio

def save_wav_to_directory(audio, sample_rate, directory="recordings"):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    file_path = os.path.join(directory, filename)
    wav.write(file_path, sample_rate, audio)
    print(f"Audio saved to: {file_path}")
    return file_path

def transcribe_with_whisper(audio_path):
    model = whisper.load_model("base")  
    result = model.transcribe(audio_path)
    return result['text']

if __name__ == "__main__":
    print("Now Recording")
    audio = record_audio(duration, sample_rate)
    print("Recording finished.")
    #audio_path = save_wav_to_directory(audio, sample_rate)
    transcription = transcribe_with_whisper(audio)
    print("Transcription complete.")
    print("\nTranscription:")
    print(transcription)
