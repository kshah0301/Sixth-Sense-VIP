import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import threading
import librosa
import os
import parler_tts
import mimic

# ================================================
# InstrumentSampler class: encapsulate loading and playing samples.
# ================================================
class InstrumentSampler:
    #def __init__(self, piano_samples=None, samplerate=44100):
    def __init__(self, directional_samples=None, samplerate=44100):
        """
        Parameters:
          - piano_samples: a dictionary mapping key names to file paths.
            Default keys: "A5", "G5", "E5", "C5".
          - samplerate: desired sample rate in Hz.
        if piano_samples is None:
            # Define default sample files.
            piano_samples = {
                "A5": "A5.wav",   # ~880 Hz sample
                "G5": "G5.wav",   # ~784 Hz sample
                "E5": "E5.wav",   # ~659.26 Hz sample
                "C5": "C5.wav"    # ~523.25 Hz sample
            }
        """

        if directional_samples is None:
            # Define default directional sample files.
            directional_samples = {
                "left": "left.wav",
                "right": "right.wav", 
                "up": "up.wav",
                "down": "down.wav"
            }
        #self.piano_samples = piano_samples
        self.directional_samples = directional_samples
        self.samplerate = samplerate
        self.preloaded_samples = {}
        self.preload_samples()

    def preload_samples(self):
        """Load each piano sample from disk into memory."""
        """
        for key, filename in self.piano_samples.items():
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found. Skipping.")
                continue
            y, sr = librosa.load(filename, sr=self.samplerate)
            self.preloaded_samples[key] = y
        """
        for key, filename in self.directional_samples.items():
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found. Please run directional_audio_generator.py first.")
                continue
            y, sr = librosa.load(filename, sr=self.samplerate)
            self.preloaded_samples[key] = y
        print("Samples preloaded:", list(self.preloaded_samples.keys()))

    def get_sample_for_direction(self, direction, duration = 0.15):
        if direction not in self.preloaded_samples:
            raise ValueError(f"Sample for direction {direction} not loaded.")
        sample = self.preloaded_samples[key]
        desired_length = int(self.samplerate * duration)
        if len(sample) < desired_length:
            sample = np.pad(sample, (0, desired_length - len(sample)), mode='constant')
        else:
            sample = sample[:desired_length]
        return sample
    """
    def get_sample_for_key(self, key, duration=0.15):
     
        Retrieve the preloaded sample for the given key.
        Trim or pad the sample so that its length is exactly `duration` seconds.
  
        if key not in self.preloaded_samples:
            raise ValueError(f"Sample for key {key} not loaded.")
        sample = self.preloaded_samples[key]
        desired_length = int(self.samplerate * duration)
        if len(sample) < desired_length:
            sample = np.pad(sample, (0, desired_length - len(sample)), mode='constant')
        else:
            sample = sample[:desired_length]
        return sample
    """
    #def play_instrument_sample(self, key, amplitude=1.0, pan=0.0,samplerate=44100, duration=0.1):
    def play_directional_sample(self, direction, amplitude=1.0, pan=0.0, samplerate=44100, duration=0.1):
        sample = self.get_sample_for_direction(direction, duration=duration)
        """
        Play the preloaded instrument sample for the specified key.
          - amplitude: Volume multiplier.
          - pan: Stereo panning value (-1: full left, 0: center, 1: full right).
        Returns the stereo sample as a NumPy array.
        """
        #sample = self.get_sample_for_key(key, duration=duration)
        # Compute stereo gains using linear panning:
        left_gain = amplitude * (1 - pan) / 2.0
        right_gain = amplitude * (1 + pan) / 2.0
        left_channel = sample * left_gain
        right_channel = sample * right_gain
        stereo_sample = np.column_stack((left_channel, right_channel))
        
        sd.play(stereo_sample, self.samplerate)
        sd.wait()
        return stereo_sample

# ================================================
# AudioThread Class: uses InstrumentSampler for playback.
# ================================================
class AudioThread(threading.Thread):
    def __init__(self, samplerate=44100, beep_duration=0.1,
                 min_interval=0.1, max_interval=1.0,
                 min_distance=0, max_distance=150,
                 min_amplitude=0.2, max_amplitude=1.0, update_timeout=0.5):
        """
        Parameters:
          - samplerate: Playback sample rate.
          - beep_duration: Duration (in seconds) for each instrument playback.
          - min_interval, max_interval: Silence interval bounds (in seconds) between plays.
          - min_distance, max_distance: Input distance range.
              When distance == 0, amplitude = max_amplitude and pan = 0.
              When distance == max_distance, amplitude = min_amplitude and pan = +1 (if direction True) or -1 (if direction False).
          - min_amplitude, max_amplitude: Amplitude (linear gain) range.
          - update_timeout: Time (seconds) without an update before ceasing playback.
        """
        super().__init__()
        self.samplerate = samplerate
        self.beep_duration = beep_duration
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.update_timeout = update_timeout

        # Fix amplitude to constant 13 (or you can change as needed).
        self.overall_amplitude = 13  
        self.pan = 0.0
        self.current_distance = None
        self.last_update = None

        self.lock = threading.Lock()
        self.exit_flag = False

        self.audio_buffer = []

        # Instantiate an InstrumentSampler.
        self.instrument_sampler = InstrumentSampler(samplerate=self.samplerate)

    #def select_key_and_interval(self, distance):
    def select_direction_and_interval(self, finger_pos, target_center, distance):
        """
        根据距离选择琴键和静音间隔：
        - 如果 distance < 0：
              返回 ("A5", 0.05)
        - 如果 0 <= distance < 30：
              用线性插值计算间隔，从 [0.05, 0.4]，返回 ("G5", interval)
        - 如果 30 <= distance < 90：
              如果 distance < 50，则 interval = 0.5；
              如果 distance < 70，则 interval = 0.65；
              否则 interval = 0.8；
              返回 ("E5", interval)
        - 如果 90 <= distance < 150：
              如果 distance < 110，则 interval = 0.9；
              如果 distance < 120，则 interval = 1.1；
              如果 distance < 135，则 interval = 1.3；
              否则 interval = 1.5；
              返回 ("C5", interval)
        - 如果 distance >= 150：
              返回 ("C5", 1.7)
        """
        
        if finger_pos is None or target_center is None:
            return "up", 1.0  # Default direction and slow interval
        
        # Calculate direction vector from finger to target
        dx = target_center[0] - finger_pos[0]  # positive = target is to the right
        dy = target_center[1] - finger_pos[1]  # positive = target is below
        
        # Determine primary direction based on larger offset
        if abs(dx) > abs(dy):
            # Horizontal movement needed
            if dx > 0:
                direction = "right"
            else:
                direction = "left"
        else:
            # Vertical movement needed
            if dy > 0:
                direction = "down"
            else:
                direction = "up"
        # Calculate interval based on distance (closer = faster feedback)
        if distance < 20:
            interval = 0.1  # Very fast when very close
        elif distance < 50:
            interval = 0.2  # Fast when close
        elif distance < 100:
            interval = 0.4  # Medium speed
        elif distance < 150:
            interval = 0.6  # Slower when far
        else:
            interval = 1.0  # Slow when very far
        return direction, interval
    #def play_instrument_for_key(self, key):
        """使用当前音量和立体声平移，播放指定琴键的采样。"""
    def play_directional_audio(self, direction):
        with self.lock:
            amp = self.overall_amplitude  # 固定值（13）
            pan = self.pan
            stereo_wave = self.instrument_sampler.play_directional_sample(direction, amplitude=amp, pan=pan,
                                                                    samplerate=self.samplerate, duration=self.beep_duration)                                                            
        self.audio_buffer.append(stereo_wave)

    #def update_params(self, distance, direction):
    def update_params(self, finger_pos, target_center, distance):
        """
        更新参数：
          - distance: float，属于 [min_distance, max_distance]。0 表示最大音量。
          - direction: 布尔值；True 表示向右平移，False 表示向左平移。
        立体声平移按照距离线性缩放（距离为 0 时为 0，距离为 max_distance 时为 ±1）。
        Amplitude 固定为 13。
        """
        new_amplitude = 13  # Fixed value
        # Calculate panning based on horizontal offset
        if finger_pos is not None and target_center is not None:
            dx = target_center[0] - finger_pos[0]
            effective_pan = np.clip(dx / 100.0, -1.0, 1.0)  # Scale horizontal offset
        else:
            effective_pan = 0.0

        with self.lock:
            self.current_distance = distance
            self.last_update = time.time()
            self.overall_amplitude = new_amplitude
            self.pan = effective_pan

    def run(self):
        while not self.exit_flag:
            with self.lock:
                if self.last_update is None or (time.time() - self.last_update) > self.update_timeout:
                    self.current_distance = None
                    self.finger_pos = None
                    self.target_center = None
            with self.lock:
                distance = self.current_distance
                finger_pos = self.finger_pos
                target_center = self.target_center
            if distance is None:
                time.sleep(self.max_interval)
                continue
            
            # 根据距离选取琴键和静音间隔。
            #key, interval = self.select_key_and_interval(distance)
            #print(f"Distance: {distance:.2f} -> Key: {key}, Silence Interval: {interval:.2f} s, Pan: {self.pan:.2f}")
            direction, interval = self.select_direction_and_interval(finger_pos, target_center, distance)
            print(f"Distance: {distance:.2f} -> Direction: {direction}, Interval: {interval:.2f} s, Pan: {self.pan:.2f}")
            
            #self.play_instrument_for_key(key)
            # 插入静音段，防止声音重叠。
            self.play_directional_audio(direction)
            silence_samples = int(self.samplerate * interval)
            silence_stereo = np.column_stack((np.zeros(silence_samples, dtype=np.float32),
                                               np.zeros(silence_samples, dtype=np.float32)))
            self.audio_buffer.append(silence_stereo)
            time.sleep(interval)
        sd.sleep(50)
            
    def stop(self):
        self.exit_flag = True

# ================================================
# Main simulation
# ================================================
if __name__ == "__main__":
    # 创建音频线程实例
    audio_thread = AudioThread(beep_duration=0.15, min_interval=0.1, max_interval=1.0,
                               min_distance=0, max_distance=150,
                               min_amplitude=0.2, max_amplitude=1.0, update_timeout=0.5)
    audio_thread.start()

    start_time = time.time()
    flip_threshold = 5.0
    direction = True  # 初始方向：向右
    prev_distance = None
    try:
        while True:
            elapsed = time.time() - start_time
            # 模拟距离在 0 到 150 之间震荡
            distance = 70 + 80 * np.sin(0.4 * elapsed)
            if prev_distance is None or (prev_distance > flip_threshold and distance < flip_threshold):
                direction = not direction
                print(f"Flipping direction to: {'Right' if direction else 'Left'} (Distance: {distance:.2f})")
            prev_distance = distance
            audio_thread.update_params(distance, direction)
            print(f"Updating: Distance = {distance:.2f}, Direction = {'Right' if direction else 'Left'}")
            sd.sleep(20)  # 更新间隔 20ms
    except KeyboardInterrupt:
        print("Stopping audio...")
    finally:
        audio_thread.stop()
        audio_thread.join()
        # 可选：保存录制的音频到 WAV 文件。
        if audio_thread.audio_buffer:
            full_audio = np.concatenate(audio_thread.audio_buffer, axis=0)
            sf.write("recorded_instrument_loop.wav", full_audio, audio_thread.samplerate)
            print("Recorded audio saved to 'recorded_instrument_loop.wav'")
