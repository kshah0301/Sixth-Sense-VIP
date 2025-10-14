import cv2 as cv
import time
import numpy as np
import torch
from ultralytics import YOLOWorld
from paddleocr import PaddleOCR
from rapidfuzz import fuzz
import hashlib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import mediapipe as mp
import threading
import sounddevice as sd
import soundfile as sf
from instrumental_beeping import AudioThread
from test_stt import WhisperTranscriber
from mimic import text_to_speech_offline
from directional_audio_generator import generate_directional_audio_files

class GestureYoloOcr:
    def __init__(self):
        # Generate directional audio files if they don't exist
        generate_directional_audio_files()
        
        # Mediapipe gesture setup
        self.latest_gesture_result = None
        model_path = "hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.update_gesture_result
        )
        print("CUDA available:", torch.cuda.is_available())

        # Display option
        self.display_green_boxes = False

        text_to_speech_offline("Name the item you are looking for?")
        self.transcriber = WhisperTranscriber(duration=2, sample_rate=16000, model_name="base")
        self.transcriber.start()
        self.transcriber.join()
        transcription_result = self.transcriber.transcription
        #self.desired_product_keywords = transcription_result.split() 
        self.desired_product_keywords =  ['RiceKrispies']
        text_to_speech_offline("Searching for " + ", ".join(self.desired_product_keywords))
        print(f"Searching for {self.desired_product_keywords}")

        # Color mapping
        self.keyword_colors = {kw: self.get_color_from_keyword(kw) for kw in self.desired_product_keywords}

        # YOLOWorld model
        self.model = YOLOWorld("yolov8s-world.pt").to("mps")
        self.model.set_classes(["items","bodypart","human"])

        # PaddleOCR
        self.ocr = PaddleOCR(show_log=False, use_gpu=False, use_tensorrt=False, lang='en')

        # Video capture
        self.cap = cv.VideoCapture(0)
        self.output_size = (640, 480)
        self.prev_time = time.time()

        # Audio thread
        self.audio_thread = AudioThread(
            beep_duration=0.15, min_interval=0.1, max_interval=1.0,
            min_distance=0, max_distance=200,
            min_amplitude=0.2, max_amplitude=1.0
        )
        self.audio_thread.start()

        self.running = False

        # Instrument sampler
    """
        from instrumental_beeping import InstrumentSampler
        self.sampler = InstrumentSampler()
        print("Loaded:", self.sampler.preloaded_samples.keys())
        self.sampler.play_directional_sample()
    """

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for landmark in landmarks:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def update_gesture_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_gesture_result = result
        print(f"[callback] got {len(result.hand_landmarks)} hand(s) at {timestamp_ms}ms")

    def get_color_from_keyword(self, keyword):
        digest = hashlib.md5(keyword.encode('utf-8')).hexdigest()
        r = int(digest[0:2], 16)
        g = int(digest[2:4], 16)
        b = int(digest[4:6], 16)
        return (b, g, r)

    def dice_coefficient_str(self, a, b):
        def bigrams(s):
            s = s.lower()
            return {s[i:i+2] for i in range(len(s) - 1)}
        a_bigrams = bigrams(a)
        b_bigrams = bigrams(b)
        if not a_bigrams or not b_bigrams:
            return 0.0
        overlap = len(a_bigrams & b_bigrams)
        return 2 * overlap / (len(a_bigrams) + len(b_bigrams))

    def get_best_matching_keyword(self, text, desired_product_keywords, threshold=0.7):
        best_keyword = None
        best_score = 0
        for keyword in desired_product_keywords:
            score = self.dice_coefficient_str(keyword, text)
            print(f"Comparing OCR text '{text}' with keyword '{keyword}' yields score: {score:.4f}")
            if score > best_score and score >= threshold:
                best_score = score
                best_keyword = keyword
        print(f"Best match for OCR text '{text}' is '{best_keyword}' with score: {best_score:.4f}")
        return best_keyword, best_score

    def center_inside(self, ocr_box, yolo_box):
        ocr_box = np.array(ocr_box)
        center = np.mean(ocr_box, axis=0)
        x_center, y_center = center
        x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = yolo_box
        return (x_center >= x_min_yolo) and (x_center <= x_max_yolo) and (y_center >= y_min_yolo) and (y_center <= y_max_yolo)
    
    def fetch_off_candidates(query_text: str, limit: int = 15):
    # OFF search API: return products with front image urls
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_simple": 1, "action": "process", "json": 1,
            "page_size": limit, "search_terms": query_text
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        products = r.json().get("products", [])
        # Prefer front image
        candidates = []
        for p in products:
            img = p.get("image_front_url") or p.get("image_url")
            name = p.get("product_name") or p.get("brands") or "unknown"
            if img:
                candidates.append((img, name))
        return candidates

    def download_image_to_tensor(url: str, preprocess):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            image = Image.open(io.BytesIO(r.content)).convert("RGB")
            return preprocess(image)
        except Exception:
            return None

    def clip_image_similarity(crop_bgr: np.ndarray, candidates, model, preprocess, device):
        # Convert crop to PIL and preprocess
        crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crop_tensor = preprocess(crop_pil).unsqueeze(0).to(device)

        # Encode query crop
        with torch.no_grad():
            q = model.encode_image(crop_tensor)
            q = q / q.norm(dim=-1, keepdim=True)

    def run(self):
        self._running = True
        with mp.tasks.vision.HandLandmarker.create_from_options(self.options) as recognizer:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv.resize(frame, self.output_size)
                print(frame.shape)
                current_time = time.time()
                fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
                self.prev_time = current_time

                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                with mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as debug_hands:
                    debug_res = debug_hands.process(frame_rgb)
                    if debug_res.multi_hand_landmarks:
                        lm = debug_res.multi_hand_landmarks[0]
                        pts = self.calc_landmark_list(frame, lm.landmark)
                        print("DEBUG: old-solutions fingertip pixel coords =", pts[8])
                    else:
                        print("DEBUG: old-solutions saw no hand")

                start_yolo = time.time()
                results = list(self.model.predict(frame, conf=0.0001, iou=0.3, half=False, stream=True, agnostic_nms=True))
                end_yolo = time.time()
                yolo_time = end_yolo - start_yolo

                yolo_boxes_all = results[0].boxes.xyxy.cpu().numpy()
                yolo_cls = results[0].boxes.cls.cpu().numpy()
                class_names = results[0].names
                store_product_boxes = [yolo_boxes_all[idx] for idx, cls_idx in enumerate(yolo_cls) if class_names[int(cls_idx)] == "items"]
                store_product_boxes = np.array(store_product_boxes)

                

                start_ocr = time.time()
                ocr_result = self.ocr.ocr(frame, cls=False, det=True)
                end_ocr = time.time()
                ocr_time = end_ocr - start_ocr

                ocr_detections = []
                if ocr_result and ocr_result[0]:
                    for detection in ocr_result[0]:
                        ocr_box = detection[0]
                        ocr_text, ocr_score = detection[1]
                        ocr_detections.append((ocr_box, ocr_text, ocr_score))

                for i, (ocr_box, ocr_text, _) in enumerate(ocr_detections):
                    ocr_center = np.mean(np.array(ocr_box), axis=0)
                    print(f"[DEBUG] OCR #{i} text='{ocr_text}' center={ocr_center.astype(int)}")
                for j, box in enumerate(store_product_boxes):
                    print(f"[DEBUG] YOLO #{j} box={box.astype(int).tolist()}")

                matched_ocr = []
                for box, text, score in ocr_detections:
                    best_keyword, sim_score = self.get_best_matching_keyword(text, self.desired_product_keywords)
                    if best_keyword is not None:
                        matched_ocr.append((box, text, score, best_keyword))

                matched_yolo = {}
                for ocr_box, ocr_text, ocr_score, keyword in matched_ocr:
                    center = np.mean(np.array(ocr_box), axis=0)
                    candidate_indices = [idx for idx, yolo_box in enumerate(store_product_boxes)
                                         if (center[0] >= yolo_box[0] and center[0] <= yolo_box[2] and
                                             center[1] >= yolo_box[1] and center[1] <= yolo_box[3])]
                    if candidate_indices:
                        best_idx = min(candidate_indices, key=lambda idx: (store_product_boxes[idx][2] - store_product_boxes[idx][0]) * (store_product_boxes[idx][3] - store_product_boxes[idx][1]))
                        matched_yolo[best_idx] = keyword
                        print(f"Matched YOLO box {best_idx} with OCR text '{ocr_text}' and keyword '{keyword}'")

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                recognizer.detect_async(mp_image, timestamp_ms)

                index_finger_tip = None
                
                if self.latest_gesture_result is None:
                    print("[loop] latest_gesture_result is None")
                else:
                    hand_landmarks_list = self.latest_gesture_result.hand_landmarks
                    print(f"[loop] received {len(hand_landmarks_list)} hand(s)")
                    if not hand_landmarks_list:
                        print("  → no hand landmarks in list")
                    else:
                        raw0 = hand_landmarks_list[0][0]
                        print(f"  raw first landmark normalized coords: x={raw0.x:.3f}, y={raw0.y:.3f}, z={raw0.z:.3f}")
                        for i, hand_landmarks in enumerate(hand_landmarks_list):
                            print(f"  hand[{i}] has {len(hand_landmarks)} landmarks")
                            landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                            print(f"    pixel coords of landmark 0 (wrist): {landmark_list[0]}")
                            print(f"    pixel coords of landmark 8 (index tip): {landmark_list[8]}")
                            index_finger_tip = landmark_list[8]
                        for i, hand_landmarks in enumerate(hand_landmarks_list):
                            proto = landmark_pb2.NormalizedLandmarkList()
                            proto.landmark.extend([
                                landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z)
                                for l in hand_landmarks
                            ])
                            solutions.drawing_utils.draw_landmarks(
                                frame,
                                proto,
                                solutions.hands.HAND_CONNECTIONS,
                                solutions.drawing_styles.get_default_hand_landmarks_style(),
                                solutions.drawing_styles.get_default_hand_connections_style(),
                            )
                print(f"DEBUG: index_finger_tip = {index_finger_tip}, matched_yolo keys = {list(matched_yolo.keys())}")
                if index_finger_tip is not None and len(matched_yolo) > 0:
                    print("tip found")
                    distances = []
                    centers = []
                    for idx in matched_yolo:
                        yolo_box = store_product_boxes[idx]
                        center = [
                            (yolo_box[0] + yolo_box[2]) / 2,
                            (yolo_box[1] + yolo_box[3]) / 2
                        ]
                        centers.append(center)
                        distance = np.linalg.norm(np.array(index_finger_tip) - np.array(center))
                        distances.append(distance)
                    if distances:
                        min_distance = min(distances)
                        best_idx = distances.index(min_distance)
                        best_center = centers[best_idx]
                        best_yolo_box = store_product_boxes[best_idx]
                        bbox_width = best_yolo_box[2] - best_yolo_box[0]
                        half_width = bbox_width / 2.0
                        effective_distance = min_distance - half_width
                        direction = index_finger_tip[0] < best_center[0]
                        self.audio_thread.update_params(index_finger_tip, best_center, effective_distance)
                        print(f"Closest distance (pixels): {min_distance:.2f}")
                        tip = tuple(map(int, index_finger_tip))
                        center_pt = tuple(map(int, best_center))
                        cv.line(frame, tip, center_pt, (0, 0, 255), 2)
                        cv.putText(frame, f"Effective Distance: {effective_distance:.2f} pixels", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                start_post = time.time()
                for idx, yolo_box in enumerate(store_product_boxes):
                    if idx in matched_yolo:
                        box_color = self.keyword_colors.get(matched_yolo[idx], (0, 0, 255))
                        yolo_box_int = yolo_box.astype(int)
                        cv.rectangle(frame, (yolo_box_int[0], yolo_box_int[1]), (yolo_box_int[2], yolo_box_int[3]), box_color, 2)
                    else:
                        if self.display_green_boxes:
                            yolo_box_int = yolo_box.astype(int)
                            cv.rectangle(frame, (yolo_box_int[0], yolo_box_int[1]), (yolo_box_int[2], yolo_box_int[3]), (0, 255, 0), 2)
                for ocr_box, ocr_text, ocr_score in ocr_detections:
                    pts = np.array(ocr_box, dtype=np.int32)
                    cv.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
                    x, y = pts[0]
                    cv.putText(frame, ocr_text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                end_post = time.time()
                post_time = end_post - start_post
                info_text = (f"FPS: {fps:.1f} | YOLO: {yolo_time*1000:.1f}ms, "
                             f"OCR: {ocr_time*1000:.1f}ms, Post: {post_time*1000:.1f}ms")
                cv.putText(frame, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.imshow("Test", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                
        self.cap.release()
        cv.destroyAllWindows()
        self.audio_thread.stop()
        self.audio_thread.join()
        

if __name__ == "__main__":
    detector = GestureYoloOcr()
    detector.run()