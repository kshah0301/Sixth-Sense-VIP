#!/usr/bin/env python3
# fingertip_item_distance_debug_v5.py
#
# PURPOSE:
#   Debug + extract pixel distances between:
#   - Fingertip point (MediaPipe primary, YOLO fallback with item-overlap rejection + skin check)
#   - Item point (YOLO item box center)
#
# OUTPUTS:
#   - CSV with fingertip/item annotations + distance
#   - Annotated images with:
#       * fingertip point + source
#       * item box + center point
#       * distance line + distance text
#
# KEY FIX vs v4:
#   - Detect item first
#   - Finger YOLO fallback rejects boxes overlapping selected item
#   - Skin-ratio heuristic helps avoid selecting the gum pack / number "5" region as fingertip
#   - Ranking no longer blindly prefers "top-most" box


import csv
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

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
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
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


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def intersection_over_candidate(candidate: np.ndarray, other: np.ndarray) -> float:
    """
    Intersection area / candidate box area (useful for rejecting finger candidates inside item box)
    """
    cx1, cy1, cx2, cy2 = [float(v) for v in candidate]
    ox1, oy1, ox2, oy2 = [float(v) for v in other]

    ix1 = max(cx1, ox1)
    iy1 = max(cy1, oy1)
    ix2 = min(cx2, ox2)
    iy2 = min(cy2, oy2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return inter / c_area if c_area > 0 else 0.0


def class_priority_finger(name: str) -> int:
    """
    Lower is better.
    """
    n = str(name).lower().strip()

    preferred = ["fingertip", "finger tip", "fingernail", "nail tip"]
    indexish = ["index finger"]
    generic_finger = ["finger"]
    handish = ["hand", "palm"]

    if any(tok in n for tok in preferred):
        return 0
    if any(tok in n for tok in indexish):
        return 1
    if any(tok in n for tok in generic_finger):
        return 2
    if any(tok in n for tok in handish):
        return 4
    return 3


def draw_point_crosshair(
    vis: np.ndarray,
    pt: Tuple[int, int],
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    label_offset=(16, -14)
):
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
        text_x = min(max(10, x + label_offset[0]), max(10, w - 700))
        text_y = max(25, y + label_offset[1])

        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv.rectangle(vis, (text_x - 3, text_y - th - 6), (text_x + tw + 5, text_y + 4), (0, 0, 0), -1)
        cv.putText(vis, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def skin_ratio_bgr(frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> float:
    """
    Very simple skin heuristic robust enough for flash-lit hand images:
    combine YCrCb + HSV thresholds.
    Returns fraction of pixels classified as skin in the box.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = clamp_xy(x1, y1, w, h)
    x2, y2 = clamp_xy(x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    ycrcb = cv.cvtColor(crop, cv.COLOR_BGR2YCrCb)

    # Broad thresholds (flash/indoor)
    lower_hsv = np.array([0, 20, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb skin range
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv.bitwise_and(mask_hsv, mask_ycrcb)
    skin_frac = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)
    return skin_frac


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
        finger_max_area_ratio: float = 0.15,
        finger_item_overlap_reject_ratio: float = 0.25,   # NEW
        finger_min_skin_ratio: float = 0.03,              # NEW

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
            finger_yolo_classes = ["fingertip", "finger tip", "fingernail", "index finger", "finger", "hand"]
        if item_yolo_classes is None:
            item_yolo_classes = ["gum pack", "gum", "product box", "box", "package", "item", "product"]

        self.finger_yolo_classes = finger_yolo_classes
        self.finger_yolo_conf = finger_yolo_conf
        self.finger_yolo_iou = finger_yolo_iou
        self.finger_point_mode = finger_point_mode
        self.finger_max_area_ratio = finger_max_area_ratio
        self.finger_item_overlap_reject_ratio = finger_item_overlap_reject_ratio
        self.finger_min_skin_ratio = finger_min_skin_ratio

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
            return None, result

        hand_landmarks = result.multi_hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        lm8 = hand_landmarks.landmark[8]  # index fingertip
        x = int(round(lm8.x * w))
        y = int(round(lm8.y * h))
        x, y = clamp_xy(x, y, w, h)
        return (x, y), result

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

    # -------------------- Item detection --------------------

    def detect_item(self, frame_bgr: np.ndarray):
        """
        Detect item box and return center point.
        Ranking heuristic:
          - prefer boxes in upper half (often item is above hand)
          - prefer compact medium boxes
          - confidence
        """
        boxes, confs, cls_names = self._run_yolo(
            self.yolo_item, frame_bgr, self.item_yolo_conf, self.item_yolo_iou
        )
        if len(boxes) == 0:
            return None, None, None, boxes, confs, cls_names, None

        h, w = frame_bgr.shape[:2]
        img_area = float(w * h)

        def top_y(i): return float(boxes[i][1])
        def area(i): return float(box_area(boxes[i]))
        def area_ratio(i): return area(i) / max(img_area, 1.0)
        def too_big(i): return area_ratio(i) > self.item_max_area_ratio

        candidates = list(range(len(boxes)))
        filtered = [i for i in candidates if not too_big(i)]
        if len(filtered) == 0:
            filtered = candidates

        # Slightly prefer upper half; keep compactness
        ranked = sorted(
            filtered,
            key=lambda i: (
                top_y(i),
                area(i),
                -float(confs[i]),
            )
        )

        best_idx = ranked[0]
        best_box = boxes[best_idx]
        best_conf = float(confs[best_idx])
        best_cls = str(cls_names[best_idx])

        item_center = box_center(best_box, w, h)
        return item_center, best_box, best_conf, boxes, confs, cls_names, best_cls

    # -------------------- Fingertip fallback --------------------

    def detect_fingertip_yolo_fallback(self, frame_bgr: np.ndarray, selected_item_box: Optional[np.ndarray] = None):
        """
        YOLO fallback with:
          - item-overlap rejection
          - skin-ratio scoring
          - finger class priority
          - compact boxes preferred
        """
        boxes, confs, cls_names = self._run_yolo(
            self.yolo_finger, frame_bgr, self.finger_yolo_conf, self.finger_yolo_iou
        )
        if len(boxes) == 0:
            return None, None, None, boxes, confs, cls_names, None

        h, w = frame_bgr.shape[:2]
        img_area = float(w * h)

        # Precompute metrics
        metrics = []
        for i in range(len(boxes)):
            box = boxes[i]
            area = float(box_area(box))
            ar = area / max(img_area, 1.0)
            sx = skin_ratio_bgr(frame_bgr, box)
            inter_item = 0.0
            if selected_item_box is not None:
                inter_item = intersection_over_candidate(box, selected_item_box)

            x1, y1, x2, y2 = [float(v) for v in box]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            aspect = bh / bw  # fingertip/finger boxes often tall-ish

            metrics.append({
                "i": i,
                "area": area,
                "area_ratio": ar,
                "skin_ratio": sx,
                "item_overlap_ratio": inter_item,
                "top_y": y1,
                "aspect": aspect,
                "cls": str(cls_names[i]),
                "conf": float(confs[i]),
            })

        # Candidate filtering
        candidates = []
        for m in metrics:
            i = m["i"]

            # Reject huge boxes (often entire table / large false regions)
            if m["area_ratio"] > self.finger_max_area_ratio:
                continue

            # Reject candidates substantially inside selected item box
            if selected_item_box is not None and m["item_overlap_ratio"] >= self.finger_item_overlap_reject_ratio:
                continue

            # Soft skin threshold: keep if skin-like OR explicit fingertip-ish class
            cp = class_priority_finger(m["cls"])
            if (m["skin_ratio"] < self.finger_min_skin_ratio) and (cp > 1):
                # generic weak class + no skin => likely false positive on item/logo
                continue

            candidates.append(i)

        # If over-filtered, fall back to boxes that at least don't overlap item too much
        if len(candidates) == 0:
            for m in metrics:
                if selected_item_box is not None and m["item_overlap_ratio"] >= 0.50:
                    continue
                candidates.append(m["i"])

        if len(candidates) == 0:
            # absolute fallback: all
            candidates = list(range(len(boxes)))

        # Ranking
        # Better candidate = lower tuple
        # - fingertip/fingernail classes first
        # - more skin (negative)
        # - smaller area
        # - higher confidence (negative)
        # - top_y as weak final tie-breaker
        ranked = sorted(
            candidates,
            key=lambda i: (
                class_priority_finger(cls_names[i]),
                -skin_ratio_bgr(frame_bgr, boxes[i]),
                float(box_area(boxes[i])),
                -float(confs[i]),
                float(boxes[i][1]),
            )
        )

        best_idx = ranked[0]
        best_box = boxes[best_idx]
        best_conf = float(confs[best_idx])
        best_cls = str(cls_names[best_idx])

        if self.finger_point_mode == "center":
            tip_xy = box_center(best_box, w, h)
        else:
            tip_xy = box_top_center(best_box, w, h)

        return tip_xy, best_box, best_conf, boxes, confs, cls_names, best_cls

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

        # --- ITEM FIRST (important for rejecting finger false positives on item/logo) ---
        (
            item_center, item_box, item_conf, item_all_boxes, item_all_confs, item_all_cls_names, item_cls
        ) = self.detect_item(frame)

        if item_center is not None:
            print(f"[ITEM] {image_path.name}: item_center={item_center} cls={item_cls} conf={float(item_conf):.3f}")
        else:
            print(f"[ITEM] {image_path.name}: no item found")

        # --- Fingertip (MediaPipe primary) ---
        mp_tip_xy, mp_result = self.detect_fingertip_mediapipe(frame, hands_ctx, use_preprocess)
        mp_hand_detected = int(mp_result.multi_hand_landmarks is not None and len(mp_result.multi_hand_landmarks) > 0)

        tip_xy = None
        tip_source = "none"
        tip_cls = ""
        tip_conf = ""

        finger_best_box = None
        finger_best_conf = None
        finger_best_cls = None
        finger_all_boxes, finger_all_confs, finger_all_cls_names = np.empty((0, 4)), np.array([]), []

        if mp_tip_xy is not None:
            tip_xy = mp_tip_xy
            tip_source = "mediapipe"

            # Optional debug draw of all finger YOLO boxes even if MP succeeds
            if self.debug_all_finger_yolo_boxes:
                (
                    _tmp_pt, _tmp_box, _tmp_conf,
                    finger_all_boxes, finger_all_confs, finger_all_cls_names,
                    _tmp_cls
                ) = self.detect_fingertip_yolo_fallback(frame, selected_item_box=item_box)

            print(f"[TIP] {image_path.name}: MediaPipe tip={tip_xy}")
        else:
            (
                tip_xy, finger_best_box, finger_best_conf,
                finger_all_boxes, finger_all_confs, finger_all_cls_names,
                finger_best_cls
            ) = self.detect_fingertip_yolo_fallback(frame, selected_item_box=item_box)

            if tip_xy is not None:
                tip_source = "yolo_fallback"
                tip_cls = finger_best_cls if finger_best_cls is not None else ""
                tip_conf = finger_best_conf
                print(f"[TIP] {image_path.name}: YOLO fallback tip={tip_xy} cls={tip_cls} conf={float(finger_best_conf):.3f}")
            else:
                print(f"[TIP] {image_path.name}: no fingertip found")

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

            # Draw MediaPipe landmarks
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
                            px = int(round(lm.x * w))
                            py = int(round(lm.y * h))
                            px, py = clamp_xy(px, py, w, h)
                            color = (0, 0, 255) if li == 8 else (255, 255, 255)
                            r = 6 if li == 8 else 3
                            cv.circle(vis, (px, py), r, color, -1)
                            cv.putText(vis, str(li), (px + 4, py - 4),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # --- Finger YOLO debug boxes (cyan) + candidate points (magenta)
            if self.debug_all_finger_yolo_boxes and len(finger_all_boxes) > 0:
                for i, box in enumerate(finger_all_boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    cv.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cand = box_top_center(box, w, h) if self.finger_point_mode == "top_center" else box_center(box, w, h)
                    cv.circle(vis, cand, 4, (255, 0, 255), -1)

                    # show skin ratio and item-overlap debug
                    sk = skin_ratio_bgr(frame, box)
                    ov = intersection_over_candidate(box, item_box) if item_box is not None else 0.0
                    label = f"F{i}:{finger_all_cls_names[i]} c={float(finger_all_confs[i]):.2f} sk={sk:.2f} ov={ov:.2f}"
                    cv.putText(vis, label, (x1, max(18, y1 - 4)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1)

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

            # --- Fingertip point annotation
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
    parser = argparse.ArgumentParser(description="Fingertip + item annotation + Euclidean distance debug extractor (MediaPipe + YOLOWorld).")

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
    parser.add_argument("--finger_max_area_ratio", type=float, default=0.15,
                        help="Filter huge finger boxes above this image-area ratio")
    parser.add_argument("--finger_item_overlap_reject_ratio", type=float, default=0.15,
                        help="Reject finger candidates whose intersection/candidate_area with selected item exceeds this")
    parser.add_argument("--finger_min_skin_ratio", type=float, default=0.05,
                        help="Minimum skin ratio for weak finger classes (helps reject item/logo false positives)")
    parser.add_argument("--finger_yolo_classes", nargs="+", default=None,
                        help='Custom finger YOLO prompts, e.g. --finger_yolo_classes fingertip "finger tip" fingernail "index finger" hand')

    # Item YOLO config
    parser.add_argument("--item_yolo_conf", type=float, default=0.001, help="Item YOLO conf")
    parser.add_argument("--item_yolo_iou", type=float, default=0.3, help="Item YOLO IoU")
    parser.add_argument("--item_max_area_ratio", type=float, default=0.40,
                        help="Filter huge item boxes above this image-area ratio")
    parser.add_argument("--item_yolo_classes", nargs="+", default=None,
                        help='Custom item YOLO prompts, e.g. --item_yolo_classes "gum pack" "gum" "product box" "item"')

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
        finger_item_overlap_reject_ratio=args.finger_item_overlap_reject_ratio,
        finger_min_skin_ratio=args.finger_min_skin_ratio,

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