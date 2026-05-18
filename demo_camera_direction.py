import cv2 as cv
import numpy as np
import mediapipe as mp
import time

from instrumental_beeping import AudioThread
from directional_audio_generator import check_directional_audio_files


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def main():
    # Ensure directional audio files exist (speech or synthesized)
    check_directional_audio_files()

    cap = cv.VideoCapture(0)
    output_size = (640, 480)

    # Start audio thread
    audio = AudioThread(
        beep_duration=0.25, min_interval=0.1, max_interval=1.0,
        min_distance=0, max_distance=200,
        min_amplitude=0.2, max_amplitude=1.0, update_timeout=5.0
    )
    audio.start()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_time = time.time()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv.resize(frame, output_size)
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)

            finger_tip = None
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                pts = calc_landmark_list(frame, lm.landmark)
                finger_tip = pts[8]  # index fingertip

            # Draw center target and fingertip for visualization
            cv.circle(frame, center, 6, (0, 255, 255), -1)
            if finger_tip is not None:
                cv.circle(frame, tuple(finger_tip), 6, (0, 0, 255), -1)
                dx = center[0] - finger_tip[0]
                dy = center[1] - finger_tip[1]
                distance = float(np.hypot(dx, dy))
                audio.update_params(tuple(finger_tip), center, distance)
                cv.line(frame, tuple(finger_tip), center, (0, 0, 255), 2)
            else:
                # No update -> thread will idle
                pass

            # FPS
            now = time.time()
            fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now
            cv.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame, "Move your index finger; audio guides toward center", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv.imshow("Camera Directional Beeps", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
        audio.stop()
        audio.join()


if __name__ == "__main__":
    main()

"""
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import sounddevice as sd

from instrumental_beeping import AudioThread
from directional_audio_generator import check_directional_audio_files


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def choose_output_device():
    print("=== Audio devices ===")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        print(f"{i:>2}: {d['name']} | in={d['max_input_channels']} out={d['max_output_channels']}")
    print("=================================")

    # Use the SAME index that worked in your AudioThread demo
    idx_str = input("Select output device index (the one that worked before): ").strip()
    if not idx_str:
        idx = sd.default.device or 0
    else:
        idx = int(idx_str)
    sd.default.device = idx
    info = sd.query_devices(idx)
    print(f"[main] Using device {idx}: {info['name']} (out={info['max_output_channels']})")
    return idx


def main():
    # Ensure directional audio files exist (speech or synthesized)
    check_directional_audio_files()

    # 1) Make sure audio uses a real output device
    choose_output_device()

    cap = cv.VideoCapture(0)
    output_size = (640, 480)

    # 2) Start audio thread with same kind of settings as your working demo
    audio = AudioThread(
        beep_duration=0.25,        # a bit longer so you can clearly hear it
        min_interval=0.1,
        max_interval=1.0,
        min_distance=0,
        max_distance=200,
        min_amplitude=0.2,
        max_amplitude=1.0,
        update_timeout=5.0,        # more generous timeout so it doesn’t idle instantly
    )
    audio.overall_amplitude = 0.5  # sane volume
    audio.start()
    print("[main] AudioThread started")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_time = time.time()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv.resize(frame, output_size)
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)

            finger_tip = None
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                pts = calc_landmark_list(frame, lm.landmark)
                finger_tip = pts[8]  # index fingertip

            # Draw center target and fingertip for visualization
            cv.circle(frame, center, 6, (0, 255, 255), -1)
            if finger_tip is not None:
                cv.circle(frame, tuple(finger_tip), 6, (0, 0, 255), -1)
                dx = center[0] - finger_tip[0]
                dy = center[1] - finger_tip[1]
                distance = float(np.hypot(dx, dy))

                # 3) Debug log so we know updates are happening
                print(f"[main] tip={finger_tip}, center={center}, dist={distance:.1f}")
                audio.update_params(tuple(finger_tip), center, distance)

                cv.line(frame, tuple(finger_tip), center, (0, 0, 255), 2)
            else:
                print("[main] no hand detected")
                # no update → AudioThread will reuse last values until timeout

            # FPS
            now = time.time()
            fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now
            cv.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame, "Move your index finger; audio guides toward center",
                       (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv.imshow("Camera Directional Beeps", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
        print("[main] Stopping AudioThread…")
        audio.stop()
        audio.join()
        print("[main] AudioThread stopped")


if __name__ == "__main__":
    main()

"""