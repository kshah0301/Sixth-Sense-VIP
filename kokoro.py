import kokoro
import os


def text_to_speech_offline(text, voice="en_US/ljspeech_low", output_file="output.wav"):
    """
    Generate speech using kokoro TTS and save to file.
    """
    try:
        # Use kokoro to generate speech
        kokoro.speak(text, output_file=output_file)
        print(f"Generated speech saved to {output_file}")

        # Play the generated audio file
        os.system(f"afplay {output_file}")  # For macOS
    except Exception as e:
        print(f"An error occurred with kokoro TTS: {e}")
