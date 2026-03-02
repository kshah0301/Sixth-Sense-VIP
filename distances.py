#!/usr/bin/env python3
# fingertip_item_distance_debug_v5.py
#
# PURPOSE:
#   Debug + extract pixel distances between:
#   - Fingertip point (MediaPipe primary, YOLO fallback with hand-ROI gating)
#   - Item point (YOLO item box center)
#
# KEY PATCHES vs v4:
#   - YOLO fingertip fallback now uses HAND-ROI gating (from MediaPipe hand bbox if available,
#     otherwise from a YOLO hand/finger anchor box).
#   - Rejects border-touching fingertip candidates (common false positives).
#   - Ranks fingertip candidates relative to hand ROI (not just global top_y).
#   - Keeps separate item and fingertip annotations clearly visible.
#
# EXAMPLE:
#   python fingertip_item_distance_debug_v5.py \
#       --images_dir /path/to/images \
#       --out_csv fingertip_item_distances_v5.csv \
#       --save_annotated \
#       --annotated_dir debug_out_v5 \
#       --use_preprocess \
#       --draw_landmark_indices \
#       --debug_all_finger_yolo_boxes \
#       --debug_all_item_yolo_boxes \
#       --finger_yolo_classes fingertip "finger tip" fingernail "index finger" finger hand palm \
#       --item_yolo_classes "gum pack" gum "product box" "item" "sanitizer bottle" "bottle"
#
# NOTES:
#   - Distance = fingertip point -> item box center (baseline).
#   - Tightening item prompts (as you said) will help item selection a lot.

import csv
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2 as cv
import numpy as np
import mediapipe as mp
import torch
from ultralytics import YOLOWorld


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def list_image_files(images_dir: Path) -> List[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files, key=lambda p: p.name.lower())


def clamp_xy(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return x, y


def preprocess_for_mediapipe(frame_bgr: np.ndarray) -> np.ndarray:
    # mild preprocessing for flash-lit images
    img = cv.GaussianBlur(frame_bgr, (3, 3), 0)
    img = cv.convertScaleAbs(img, alpha=1.05, beta=5)
    return img


def box_area(box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = box_xyxy.tolist()
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box_xyxy: np.ndarray, w: int, h: int) -> Tuple[int, int]:
    x1, y1, x2, y2 = box_xyxy.tolist()
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    return clamp_xy(cx, cy, w, h)


def box_top_center(box_xyxy: np.ndarray, w: int, h: int) -> Tuple[int, int]:
    x1, y1, x2, y2 = box_xyxy.tolist()
    tx = int(round((x1 + x2) / 2.0))
    ty = int(round(y1))
    return clamp_xy(tx, ty, w, h)


def expand_box_xyxy(box_xyxy: np.ndarray, w: int, h: int, pad_frac: float = 0.08) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy.astype(float)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_frac
    py = bh * pad_frac
    out = np.array([x1 - px, y1 - py, x2 + px, y2 + py], dtype=np.float32)
    out[0] = max(0, min(out[0], w - 1))
    out[1] = max(0, min(out[1], h - 1))
    out[2] = max(0, min(out[2], w - 1))
    out[3] = max(0, min(out[3], h - 1))
    return out


def point_in_box(pt: Tuple[int, int], box_xyxy: np.ndarray, margin: int = 0) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box_xyxy.astype(float)
    return (x >= x1 - margin and x <= x2 + margin and y >= y1 - margin and y <= y2 + margin)


def distance_point_to_box(pt: Tuple[int, int], box_xyxy: np.ndarray) -> float:
    """0 if inside, otherwise Euclidean distance to nearest edge."""
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = box_xyxy.astype(float)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def box_touches_border(box_xyxy: np.ndarray, w: int, h: int, margin: int = 6) -> bool:
    x1, y1, x2, y2 = box_xyxy.astype(float)
    return (x1 <= margin or y1 <= margin or x2 >= (w - 1 - margin) or y2 >= (h - 1 - margin))


def compute_landmark_bbox(hand_landmarks, w: int, h: int, pad: int = 18) -> np.ndarray:
    xs = []
    ys = []
    for lm in hand_landmarks.landmark:
        px = int(lm.x * w)
        py = int(lm.y * h)
        px, py = clamp_xy(px, py, w, h)
        xs.append(px)
        ys.append(py)
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xs) + pad)
    y2 = min(h - 1, max(ys) + pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def class_priority_finger(name: str) -> int:
    """Lower is better."""
    n = str(name).lower().strip()
    if "fingertip" in n or "finger tip" in n or "nail tip" in n:
        return 0
    if "fingernail" in n or "nail" in n:
        return 1
    if "index finger" in n:
        return 2
    if "finger" in n:
        return 3
    if "hand" in n or "palm" in n:
        return 5
    return 4


def is_handish_class(name: str) -> bool:
    n = str(name).lower().strip()
    return any(tok in n for tok in ["hand", "palm", "finger"])


def draw_point_crosshair(vis: np.ndarray, pt: Tuple[int, int], color: Tuple[int, int, int],
                         label: Optional[str] = None, label_offset=(16, -14)):
    h, w = vis.shape[:2]
    x, y = int(pt[0]), int(pt[1])

    # Crosshair
    L = 16
    cv.line(vis, (x - L, y), (x + L, y), (0, 0, 0), 4)
    cv.line(vis, (x, y - L), (x, y + L), (0, 0, 0), 4)
    cv.line(vis, (x - L, y), (x + L, y), color, 2)
    cv.line(vis, (x, y - L), (x, y + L), color, 2)

    # Concentric point
    cv.circle(vis, (x, y), 12, (0, 0, 0), -1)
    cv.circle(vis, (x, y), 8, color, -1)
    cv.circle(vis, (x, y), 14, color, 2)

    if label:
        text_x = min(max(10, x + label_offset[0]), max(10, w - 650))
        text_y = max(25, y + label_offset[1])

        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv.rectangle(vis, (text_x - 3, text_y - th - 6), (text_x + tw + 5, text_y + 4), (0, 0, 0), -1)
        cv.putText(vis, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# ------------------------------------------------------------
# Core detector
# ------------------------------------------------------------

class FingertipItemDistanceExtractor:
    def __init__(
        self,
        yolo_model_path: str = "yolov8s-world.pt",
        device: str = "auto",

        finger_yolo_classes: Optional[List[str]] = None,
        finger_yolo_conf: float = 0.001,
        finger_yolo_iou: float = 0.3,
        finger_point_mode: str = "top_center",
        finger_max_area_ratio: float = 0.20,

        item_yolo_classes: Optional[List[str]] = None,
        item_yolo_conf: float = 0.001,
        item_yolo_iou: float = 0.3,
        item_max_area_ratio: float = 0.40,

        debug_all_finger_yolo_boxes: bool = False,
        debug_all_item_yolo_boxes: bool = False,
        resize_to: Optional[Tuple[int, int]] = None,
    ):
        # Device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"[INFO] Torch device: {self.device}")

        # Defaults
        if finger_yolo_classes is None:
            finger_yolo_classes = ["fingertip", "finger tip", "fingernail", "nail", "index finger", "finger", "hand", "palm"]
        if item_yolo_classes is None:
            item_yolo_classes = ["item", "product", "product box", "box", "package"]

        self.finger_yolo_classes = finger_yolo_classes
        self.finger_yolo_conf = finger_yolo_conf
        self.finger_yolo_iou = finger_yolo_iou
        self.finger_point_mode = finger_point_mode
        self.finger_max_area_ratio = finger_max_area_ratio

        self.item_yolo_classes = item_yolo_classes
        self.item_yolo_conf = item_yolo_conf
        self.item_yolo_iou = item_yolo_iou
        self.item_max_area_ratio = item_max_area_ratio

        self.debug_all_finger_yolo_boxes = debug_all_finger_yolo_boxes
        self.debug_all_item_yolo_boxes = debug_all_item_yolo_boxes
        self.resize_to = resize_to

        # Separate YOLOWorld models
        self.yolo_finger = YOLOWorld(yolo_model_path).to(self.device)
        self.yolo_finger.set_classes(self.finger_yolo_classes)
        print(f"[INFO] Finger YOLOWorld classes: {self.finger_yolo_classes}")

        self.yolo_item = YOLOWorld(yolo_model_path).to(self.device)
        self.yolo_item.set_classes(self.item_yolo_classes)
        print(f"[INFO] Item YOLOWorld classes: {self.item_yolo_classes}")

        # MediaPipe refs
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

    # -------------------- MediaPipe fingertip --------------------

    def detect_fingertip_mediapipe(self, frame_bgr: np.ndarray, hands_ctx, use_preprocess: bool):
        mp_input = preprocess_for_mediapipe(frame_bgr) if use_preprocess else frame_bgr
        rgb = cv.cvtColor(mp_input, cv.COLOR_BGR2RGB)
        result = hands_ctx.process(rgb)

        if not result.multi_hand_landmarks:
            return None, result, None

        hand_landmarks = result.multi_hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        lm8 = hand_landmarks.landmark[8]  # index fingertip
        x = int(lm8.x * w)
        y = int(lm8.y * h)
        x, y = clamp_xy(x, y, w, h)

        hand_bbox = compute_landmark_bbox(hand_landmarks, w, h, pad=20)
        return (x, y), result, hand_bbox

    # -------------------- YOLO helper --------------------

    def _run_yolo(self, model: YOLOWorld, frame_bgr: np.ndarray, conf: float, iou: float):
        results = list(
            model.predict(
                frame_bgr,
                conf=conf,
                iou=iou,
                half=False,
                stream=True,
                agnostic_nms=True,
                verbose=False,
            )
        )
        if not results:
            return np.empty((0, 4), dtype=np.float32), np.array([]), []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.array([]), []

        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else np.ones(len(boxes))
        cls_ids = r0.boxes.cls.cpu().numpy()
        names = r0.names
        cls_names = [names[int(c)] for c in cls_ids]
        return boxes, confs, cls_names

    # -------------------- Fingertip fallback (PATCHED) --------------------

    def detect_fingertip_yolo_fallback(self, frame_bgr: np.ndarray, hand_roi_hint: Optional[np.ndarray] = None):
        """
        Returns:
          tip_xy, best_box, best_conf, all_boxes, all_confs, all_cls_names, best_cls, debug_info
        """
        boxes, confs, cls_names = self._run_yolo(
            self.yolo_finger, frame_bgr, self.finger_yolo_conf, self.finger_yolo_iou
        )
        h, w = frame_bgr.shape[:2]
        img_area = float(w * h)

        debug_info: Dict[str, Any] = {
            "hand_roi_hint": hand_roi_hint.copy() if hand_roi_hint is not None else None,
            "hand_roi_used": None,
            "candidate_indices_after_basic": [],
            "candidate_indices_after_roi": [],
            "anchor_idx": None,
        }

        if len(boxes) == 0:
            return None, None, None, boxes, confs, cls_names, None, debug_info

        def area(i): return float(box_area(boxes[i]))
        def area_ratio(i): return area(i) / max(img_area, 1.0)

        # Basic filter: remove very large and border-touching boxes
        basic = []
        for i in range(len(boxes)):
            if area_ratio(i) > self.finger_max_area_ratio:
                continue
            if box_touches_border(boxes[i], w, h, margin=8):
                # Finger false positives often appear on frame borders
                continue
            basic.append(i)

        if len(basic) == 0:
            # fallback if overfiltered
            basic = list(range(len(boxes)))

        debug_info["candidate_indices_after_basic"] = basic.copy()

        # Build a hand ROI if not provided:
        # choose the largest handish box as anchor (hand/palm/finger)
        hand_roi = None
        if hand_roi_hint is not None:
            hand_roi = expand_box_xyxy(hand_roi_hint, w, h, pad_frac=0.08)
        else:
            handish = [i for i in basic if is_handish_class(cls_names[i])]
            if handish:
                # Largest handish box tends to be actual hand / finger shaft region
                anchor_idx = sorted(handish, key=lambda i: (-area(i), -float(confs[i])))[0]
                debug_info["anchor_idx"] = int(anchor_idx)
                hand_roi = expand_box_xyxy(boxes[anchor_idx], w, h, pad_frac=0.15)

        debug_info["hand_roi_used"] = hand_roi.copy() if hand_roi is not None else None

        # ROI-gated candidates
        candidates = []
        for i in basic:
            cand_pt = box_top_center(boxes[i], w, h) if self.finger_point_mode == "top_center" else box_center(boxes[i], w, h)

            # Prefer fingertip-ish classes; hand/palm boxes only as last resort
            cls_pr = class_priority_finger(cls_names[i])

            # If we have a hand ROI, require candidate point near or inside it
            if hand_roi is not None:
                d_to_roi = distance_point_to_box(cand_pt, hand_roi)
                if d_to_roi > 60:  # too far from hand region
                    continue
            candidates.append(i)

        if len(candidates) == 0:
            # relax: use basic candidates but still avoid absurd "hand/palm" if fingertip boxes exist
            candidates = basic.copy()

        debug_info["candidate_indices_after_roi"] = candidates.copy()

        # If fingertip-ish candidates exist, down-prioritize generic hand/palm
        fingertipish = [i for i in candidates if class_priority_finger(cls_names[i]) <= 3]
        if fingertipish:
            candidates = fingertipish

        # Score candidates relative to hand ROI if available
        scored = []
        for i in candidates:
            box = boxes[i]
            conf = float(confs[i])
            cls_name = str(cls_names[i])
            cls_pr = class_priority_finger(cls_name)
            cand_pt = box_top_center(box, w, h) if self.finger_point_mode == "top_center" else box_center(box, w, h)
            a = area(i)

            # geometric terms
            if hand_roi is not None:
                hr_x1, hr_y1, hr_x2, hr_y2 = hand_roi.astype(float)
                hr_cx = (hr_x1 + hr_x2) / 2.0
                # Prefer candidate near top of hand ROI and not too far horizontally from ROI center
                y_above_penalty = max(0.0, float(cand_pt[1]) - (hr_y1 + 0.45 * (hr_y2 - hr_y1)))
                x_center_penalty = abs(float(cand_pt[0]) - hr_cx)
                roi_dist = distance_point_to_box(cand_pt, hand_roi)
            else:
                y_above_penalty = float(cand_pt[1])  # fallback: top-most in image
                x_center_penalty = 0.0
                roi_dist = 0.0

            # Prefer smaller boxes but not microscopic noise
            tiny_penalty = 0.0
            if a < 40:
                tiny_penalty = 500.0

            score = (
                cls_pr * 1000.0 +
                roi_dist * 20.0 +
                y_above_penalty * 1.2 +
                x_center_penalty * 0.5 +
                a * 0.08 +
                tiny_penalty -
                conf * 120.0
            )
            scored.append((score, i))

        scored.sort(key=lambda t: t[0])
        best_idx = scored[0][1]

        best_box = boxes[best_idx]
        best_conf = float(confs[best_idx])
        best_cls = str(cls_names[best_idx])

        if self.finger_point_mode == "center":
            tip_xy = box_center(best_box, w, h)
        else:
            tip_xy = box_top_center(best_box, w, h)

        return tip_xy, best_box, best_conf, boxes, confs, cls_names, best_cls, debug_info

    # -------------------- Item detection --------------------

    def detect_item(self, frame_bgr: np.ndarray):
        """
        Detect item box and return center point.
        Ranking heuristic:
          - avoid border-touching huge boxes
          - prefer mid-size boxes with decent confidence
          - slight preference for upper boxes if object is often above hand
        """
        boxes, confs, cls_names = self._run_yolo(
            self.yolo_item, frame_bgr, self.item_yolo_conf, self.item_yolo_iou
        )
        if len(boxes) == 0:
            return None, None, None, boxes, confs, cls_names, None

        h, w = frame_bgr.shape[:2]
        img_area = float(w * h)

        def area(i): return float(box_area(boxes[i]))
        def area_ratio(i): return area(i) / max(img_area, 1.0)
        def top_y(i): return float(boxes[i][1])

        candidates = []
        for i in range(len(boxes)):
            if area_ratio(i) > self.item_max_area_ratio:
                continue
            # Skip giant border boxes frequently produced by open-vocab prompts
            if box_touches_border(boxes[i], w, h, margin=4) and area_ratio(i) > 0.03:
                continue
            candidates.append(i)

        if len(candidates) == 0:
            candidates = list(range(len(boxes)))

        # Better ranking than pure top_y
        def score_item(i):
            a = area(i)
            conf = float(confs[i])
            ar = area_ratio(i)
            # prefer neither tiny nor huge boxes
            size_penalty = 0.0
            if ar < 0.0003:
                size_penalty += 1500.0
            if ar > 0.08:
                size_penalty += 800.0
            return (
                size_penalty +
                top_y(i) * 0.25 +   # weak prior only
                a * 0.02 -
                conf * 150.0
            )

        ranked = sorted(candidates, key=score_item)
        best_idx = ranked[0]
        best_box = boxes[best_idx]
        best_conf = float(confs[best_idx])
        best_cls = str(cls_names[best_idx])

        item_center = box_center(best_box, w, h)
        return item_center, best_box, best_conf, boxes, confs, cls_names, best_cls

    # -------------------- Process single image --------------------

    def process_image(
        self,
        image_path: Path,
        hands_ctx,
        use_preprocess: bool,
        save_annotated: bool,
        annotated_dir: Optional[Path],
        draw_landmark_indices: bool,
    ) -> dict:
        frame = cv.imread(str(image_path))
        if frame is None:
            print(f"[ERROR] Failed to read image: {image_path}")
            return {"image_name": image_path.name, "status": "read_fail"}

        orig_h, orig_w = frame.shape[:2]
        if self.resize_to is not None:
            frame = cv.resize(frame, self.resize_to)
        h, w = frame.shape[:2]

        # --- Fingertip (MediaPipe primary) ---
        mp_tip_xy, mp_result, mp_hand_bbox = self.detect_fingertip_mediapipe(frame, hands_ctx, use_preprocess)
        mp_hand_detected = int(mp_result.multi_hand_landmarks is not None and len(mp_result.multi_hand_landmarks) > 0)

        tip_xy = None
        tip_source = "none"
        tip_cls = ""
        tip_conf = ""

        finger_best_box = None
        finger_best_conf = None
        finger_best_cls = None
        finger_all_boxes, finger_all_confs, finger_all_cls_names = np.empty((0, 4)), np.array([]), []
        finger_debug = None

        if mp_tip_xy is not None:
            tip_xy = mp_tip_xy
            tip_source = "mediapipe"

            # Still run finger YOLO if debug requested, but use MP hand bbox as hint for diagnostics
            if self.debug_all_finger_yolo_boxes:
                (
                    _tmp_pt, _tmp_box, _tmp_conf,
                    finger_all_boxes, finger_all_confs, finger_all_cls_names,
                    _tmp_cls, finger_debug
                ) = self.detect_fingertip_yolo_fallback(frame, hand_roi_hint=mp_hand_bbox)

            print(f"[TIP] {image_path.name}: MediaPipe tip={tip_xy}")
        else:
            (
                tip_xy, finger_best_box, finger_best_conf,
                finger_all_boxes, finger_all_confs, finger_all_cls_names,
                finger_best_cls, finger_debug
            ) = self.detect_fingertip_yolo_fallback(frame, hand_roi_hint=mp_hand_bbox)

            if tip_xy is not None:
                tip_source = "yolo_fallback"
                tip_cls = finger_best_cls if finger_best_cls is not None else ""
                tip_conf = finger_best_conf
                print(f"[TIP] {image_path.name}: YOLO fallback tip={tip_xy} cls={tip_cls} conf={float(finger_best_conf):.3f}")
            else:
                print(f"[TIP] {image_path.name}: no fingertip found")

        # --- Item detection ---
        (
            item_center, item_box, item_conf, item_all_boxes, item_all_confs, item_all_cls_names, item_cls
        ) = self.detect_item(frame)

        if item_center is not None:
            print(f"[ITEM] {image_path.name}: item_center={item_center} cls={item_cls} conf={float(item_conf):.3f}")
        else:
            print(f"[ITEM] {image_path.name}: no item found")

        # --- Distance ---
        dist_px = None
        if tip_xy is not None and item_center is not None:
            dist_px = float(np.linalg.norm(np.array(tip_xy, dtype=np.float32) - np.array(item_center, dtype=np.float32)))

        # --- Save annotated image ---
        if save_annotated and annotated_dir is not None:
            vis = frame.copy()

            # debug marker
            cv.circle(vis, (20, 20), 8, (255, 0, 255), -1)
            cv.putText(vis, "DEBUG_SAVE", (35, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Draw MediaPipe landmarks + MP hand bbox
            if mp_result is not None and mp_result.multi_hand_landmarks:
                for hand_landmarks in mp_result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        vis,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style(),
                    )

                    if draw_landmark_indices:
                        for li, lm in enumerate(hand_landmarks.landmark):
                            px = int(lm.x * w)
                            py = int(lm.y * h)
                            px, py = clamp_xy(px, py, w, h)
                            color = (0, 0, 255) if li == 8 else (255, 255, 255)
                            r = 6 if li == 8 else 3
                            cv.circle(vis, (px, py), r, color, -1)
                            cv.putText(vis, str(li), (px + 4, py - 4),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            if mp_hand_bbox is not None:
                x1, y1, x2, y2 = mp_hand_bbox.astype(int)
                cv.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv.putText(vis, "MP_HAND_BBOX", (x1, max(20, y1 - 6)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)

            # Finger fallback hand ROI used (if any)
            if finger_debug is not None and finger_debug.get("hand_roi_used") is not None:
                rx1, ry1, rx2, ry2 = finger_debug["hand_roi_used"].astype(int)
                cv.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
                cv.putText(vis, "FALLBACK_HAND_ROI", (rx1, min(h - 10, ry2 + 18)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 2)

            # --- Finger YOLO debug boxes (cyan) + candidate points (magenta)
            if self.debug_all_finger_yolo_boxes and len(finger_all_boxes) > 0:
                cand_after_roi = set(finger_debug.get("candidate_indices_after_roi", [])) if finger_debug else set()
                for i, box in enumerate(finger_all_boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    color = (255, 255, 0) if i in cand_after_roi else (120, 120, 120)
                    thick = 1 if i in cand_after_roi else 1
                    cv.rectangle(vis, (x1, y1), (x2, y2), color, thick)

                    cand = box_top_center(box, w, h) if self.finger_point_mode == "top_center" else box_center(box, w, h)
                    cv.circle(vis, cand, 4, (255, 0, 255), -1)

                    label = f"F{i}:{finger_all_cls_names[i]} {float(finger_all_confs[i]):.2f}"
                    cv.putText(vis, label, (x1, max(18, y1 - 4)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # --- Selected finger fallback box (RED) if used
            if finger_best_box is not None:
                x1, y1, x2, y2 = finger_best_box.astype(int)
                cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), 5)
                cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"F_SELECTED {tip_cls} {float(finger_best_conf):.2f}" if finger_best_conf is not None else "F_SELECTED"
                (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ly = max(25, y1 - 8)
                cv.rectangle(vis, (x1 - 2, ly - th - 6), (x1 + tw + 4, ly + 4), (0, 0, 0), -1)
                cv.putText(vis, label, (x1, ly), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            # --- Item YOLO debug boxes (light green)
            if self.debug_all_item_yolo_boxes and len(item_all_boxes) > 0:
                for i, box in enumerate(item_all_boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    cv.rectangle(vis, (x1, y1), (x2, y2), (120, 255, 120), 1)
                    cpt = box_center(box, w, h)
                    cv.circle(vis, cpt, 3, (255, 0, 255), -1)
                    label = f"I{i}:{item_all_cls_names[i]} {float(item_all_confs[i]):.2f}"
                    cv.putText(vis, label, (x1, min(h - 5, y2 + 12)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.35, (120, 255, 120), 1)

            # --- Selected item box (GREEN) + center point (BLUE)
            item_x1 = item_y1 = item_x2 = item_y2 = None
            if item_box is not None:
                item_x1, item_y1, item_x2, item_y2 = map(int, item_box)

                cv.rectangle(vis, (item_x1, item_y1), (item_x2, item_y2), (0, 0, 0), 5)
                cv.rectangle(vis, (item_x1, item_y1), (item_x2, item_y2), (0, 255, 0), 3)

                if item_center is not None:
                    icx, icy = int(item_center[0]), int(item_center[1])

                    # BLUE item center marker
                    cv.circle(vis, (icx, icy), 10, (0, 0, 0), -1)
                    cv.circle(vis, (icx, icy), 6, (255, 0, 0), -1)
                    cv.circle(vis, (icx, icy), 12, (255, 0, 0), 2)

                    item_text = f"item pt=({icx},{icy})"
                    if item_cls:
                        item_text = f"{item_cls} {item_text}"
                    if item_conf is not None:
                        item_text += f" conf={float(item_conf):.2f}"

                    tx = max(10, item_x1)
                    ty = max(25, item_y1 - 10)
                    (tw, th), _ = cv.getTextSize(item_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv.rectangle(vis, (tx - 3, ty - th - 6), (tx + tw + 5, ty + 4), (0, 0, 0), -1)
                    cv.putText(vis, item_text, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- Fingertip point annotation (draw LAST so it stays visible)
            if tip_xy is not None:
                tx, ty = int(tip_xy[0]), int(tip_xy[1])

                if tip_source == "mediapipe":
                    f_color = (0, 0, 255)       # red
                elif tip_source == "yolo_fallback":
                    f_color = (0, 255, 255)     # yellow
                else:
                    f_color = (255, 255, 255)

                label = f"{tip_source} pt=({tx},{ty})"
                if tip_source == "yolo_fallback" and finger_best_conf is not None:
                    label += f" cls={tip_cls} conf={float(finger_best_conf):.2f}"

                draw_point_crosshair(vis, (tx, ty), f_color, label=label)

            # --- Distance line + label
            if tip_xy is not None and item_center is not None and dist_px is not None:
                tx, ty = int(tip_xy[0]), int(tip_xy[1])
                icx, icy = int(item_center[0]), int(item_center[1])

                cv.line(vis, (tx, ty), (icx, icy), (255, 255, 255), 4)
                cv.line(vis, (tx, ty), (icx, icy), (255, 0, 255), 2)

                mx = int((tx + icx) / 2)
                my = int((ty + icy) / 2)
                dtext = f"dist={dist_px:.2f}px"

                (tw, th), _ = cv.getTextSize(dtext, cv.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv.rectangle(vis, (mx - 5, my - th - 8), (mx + tw + 6, my + 5), (0, 0, 0), -1)
                cv.putText(vis, dtext, (mx, my), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

            # Status line
            status = f"mp_hand={mp_hand_detected} | tip={tip_source} | item={'yes' if item_box is not None else 'no'}"
            cv.putText(vis, status, (10, h - 15), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            out_path = annotated_dir / image_path.name
            ok = cv.imwrite(str(out_path), vis)
            print(f"[SAVE] {out_path} -> {'OK' if ok else 'FAILED'}")

        # Build CSV row
        item_x1 = item_y1 = item_x2 = item_y2 = ""
        if item_box is not None:
            item_x1, item_y1, item_x2, item_y2 = map(int, item_box)

        row = {
            "image_name": image_path.name,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "proc_width": w,
            "proc_height": h,

            "mp_hand_detected": mp_hand_detected,
            "mp_tip_found": int(mp_tip_xy is not None),
            "mp_tip_x": mp_tip_xy[0] if mp_tip_xy is not None else "",
            "mp_tip_y": mp_tip_xy[1] if mp_tip_xy is not None else "",

            "tip_found": int(tip_xy is not None),
            "tip_source": tip_source,
            "tip_x": tip_xy[0] if tip_xy is not None else "",
            "tip_y": tip_xy[1] if tip_xy is not None else "",
            "tip_point_xy": f"({int(tip_xy[0])},{int(tip_xy[1])})" if tip_xy is not None else "",
            "tip_conf": tip_conf if tip_conf != "" else "",
            "tip_cls": tip_cls,

            "item_found": int(item_box is not None),
            "item_x1": item_x1,
            "item_y1": item_y1,
            "item_x2": item_x2,
            "item_y2": item_y2,
            "item_center_x": item_center[0] if item_center is not None else "",
            "item_center_y": item_center[1] if item_center is not None else "",
            "item_center_xy": f"({int(item_center[0])},{int(item_center[1])})" if item_center is not None else "",
            "item_conf": item_conf if item_conf is not None else "",
            "item_cls": item_cls if item_cls is not None else "",

            "distance_px": f"{dist_px:.4f}" if dist_px is not None else "",

            "finger_yolo_num_boxes": len(finger_all_boxes) if hasattr(finger_all_boxes, "__len__") else 0,
            "item_yolo_num_boxes": len(item_all_boxes) if hasattr(item_all_boxes, "__len__") else 0,

            "status": "ok",
        }
        return row


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fingertip + item annotation + Euclidean distance debug extractor (MediaPipe + YOLOWorld, patched hand-ROI fallback).")

    parser.add_argument("--images_dir", required=True, help="Path to image folder")
    parser.add_argument("--out_csv", default="fingertip_item_distances_v5.csv", help="Output CSV path")

    parser.add_argument("--save_annotated", action="store_true", help="Save annotated images")
    parser.add_argument("--annotated_dir", default="debug_out_v5", help="Annotated output dir")

    parser.add_argument("--draw_landmark_indices", action="store_true", help="Draw MediaPipe landmark indices")
    parser.add_argument("--use_preprocess", action="store_true", help="Use preprocessing before MediaPipe")

    parser.add_argument("--min_det_conf", type=float, default=0.3, help="MediaPipe min_detection_confidence")
    parser.add_argument("--min_track_conf", type=float, default=0.3, help="MediaPipe min_tracking_confidence")

    parser.add_argument("--yolo_model", default="yolov8s-world.pt", help="YOLOWorld model path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Torch device")

    # Finger fallback YOLO config
    parser.add_argument("--finger_yolo_conf", type=float, default=0.001, help="Finger fallback YOLO conf")
    parser.add_argument("--finger_yolo_iou", type=float, default=0.3, help="Finger fallback YOLO IoU")
    parser.add_argument("--finger_point_mode", default="top_center", choices=["top_center", "center"],
                        help="Convert selected finger fallback box to point")
    parser.add_argument("--finger_max_area_ratio", type=float, default=0.20,
                        help="Filter huge finger boxes above this image-area ratio")
    parser.add_argument("--finger_yolo_classes", nargs="+", default=None,
                        help='Custom finger YOLO prompts, e.g. --finger_yolo_classes fingertip "finger tip" fingernail "index finger" finger hand')

    # Item YOLO config
    parser.add_argument("--item_yolo_conf", type=float, default=0.001, help="Item YOLO conf")
    parser.add_argument("--item_yolo_iou", type=float, default=0.3, help="Item YOLO IoU")
    parser.add_argument("--item_max_area_ratio", type=float, default=0.40,
                        help="Filter huge item boxes above this image-area ratio")
    parser.add_argument("--item_yolo_classes", nargs="+", default=None,
                        help='Custom item YOLO prompts, e.g. --item_yolo_classes "gum pack" gum "product box" "item"')

    # Debug overlays
    parser.add_argument("--debug_all_finger_yolo_boxes", action="store_true", help="Draw all finger YOLO boxes + candidate points")
    parser.add_argument("--debug_all_item_yolo_boxes", action="store_true", help="Draw all item YOLO boxes")

    # Optional resize
    parser.add_argument("--resize_w", type=int, default=0, help="Resize width (0 = original)")
    parser.add_argument("--resize_h", type=int, default=0, help="Resize height (0 = original)")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    resize_to = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)
        print(f"[INFO] Resize enabled: {resize_to}")
    else:
        print("[INFO] Using original resolution")

    annotated_dir = None
    if args.save_annotated:
        annotated_dir = Path(args.annotated_dir)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Annotated output dir: {annotated_dir.resolve()}")

    image_paths = list_image_files(images_dir)
    print(f"[INFO] Found {len(image_paths)} image(s) in {images_dir}")
    if len(image_paths) == 0:
        print("[WARN] No images found.")
        return

    extractor = FingertipItemDistanceExtractor(
        yolo_model_path=args.yolo_model,
        device=args.device,

        finger_yolo_classes=args.finger_yolo_classes,
        finger_yolo_conf=args.finger_yolo_conf,
        finger_yolo_iou=args.finger_yolo_iou,
        finger_point_mode=args.finger_point_mode,
        finger_max_area_ratio=args.finger_max_area_ratio,

        item_yolo_classes=args.item_yolo_classes,
        item_yolo_conf=args.item_yolo_conf,
        item_yolo_iou=args.item_yolo_iou,
        item_max_area_ratio=args.item_max_area_ratio,

        debug_all_finger_yolo_boxes=args.debug_all_finger_yolo_boxes,
        debug_all_item_yolo_boxes=args.debug_all_item_yolo_boxes,
        resize_to=resize_to,
    )

    rows = []

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
    ) as hands_ctx:

        total = len(image_paths)
        for idx, img_path in enumerate(image_paths, start=1):
            print(f"\n[{idx}/{total}] Processing {img_path.name}")
            row = extractor.process_image(
                image_path=img_path,
                hands_ctx=hands_ctx,
                use_preprocess=args.use_preprocess,
                save_annotated=args.save_annotated,
                annotated_dir=annotated_dir,
                draw_landmark_indices=args.draw_landmark_indices,
            )
            rows.append(row)

    out_csv_path = Path(args.out_csv)
    fieldnames = [
        "image_name",
        "orig_width", "orig_height",
        "proc_width", "proc_height",

        "mp_hand_detected",
        "mp_tip_found", "mp_tip_x", "mp_tip_y",

        "tip_found", "tip_source", "tip_x", "tip_y", "tip_point_xy", "tip_conf", "tip_cls",

        "item_found",
        "item_x1", "item_y1", "item_x2", "item_y2",
        "item_center_x", "item_center_y", "item_center_xy",
        "item_conf", "item_cls",

        "distance_px",

        "finger_yolo_num_boxes",
        "item_yolo_num_boxes",
        "status",
    ]

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    total_images = len(rows)
    tip_found = sum(int(r["tip_found"]) for r in rows if str(r.get("tip_found", "")).strip() != "")
    item_found = sum(int(r["item_found"]) for r in rows if str(r.get("item_found", "")).strip() != "")
    dist_found = sum(1 for r in rows if str(r.get("distance_px", "")).strip() != "")
    mp_count = sum(1 for r in rows if r.get("tip_source") == "mediapipe")
    yolo_fb_count = sum(1 for r in rows if r.get("tip_source") == "yolo_fallback")

    print("\n========== SUMMARY ==========")
    print(f"Total images: {total_images}")
    print(f"Fingertip found: {tip_found}/{total_images}")
    print(f"  - MediaPipe: {mp_count}")
    print(f"  - YOLO fallback: {yolo_fb_count}")
    print(f"Item found: {item_found}/{total_images}")
    print(f"Distance computed: {dist_found}/{total_images}")
    print(f"CSV saved to: {out_csv_path.resolve()}")
    if args.save_annotated and annotated_dir is not None:
        print(f"Annotated images saved to: {annotated_dir.resolve()}")


if __name__ == "__main__":
    main()