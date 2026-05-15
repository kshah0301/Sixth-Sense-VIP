#!/usr/bin/env python3
# ocr_product_reader.py
#
# OCR-only: read product text from an image (or webcam), draw boxes, and print a best guess.
#
# deps:
#   pip install opencv-python paddleocr
#
# optional (recommended on mac):
#   pip install paddlepaddle  # if you don't already have it

import argparse
import re
import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR


# ------------------------- OCR helpers -------------------------

def run_ocr(ocr: PaddleOCR, img_bgr: np.ndarray):
    """
    Returns detections:
      [{'pts': [(x,y)*4], 'text': str, 'conf': float, 'w': float, 'h': float, 'area': float, 'cx': float, 'cy': float}]
    """
    result = ocr.ocr(img_bgr, cls=True, det=True)
    dets = []
    if not result or not result[0]:
        return dets

    for det in result[0]:
        pts = det[0]
        text, conf = det[1]
        text = (text or "").strip()
        if not text:
            continue

        pts_i = [(int(p[0]), int(p[1])) for p in pts]
        xs = [p[0] for p in pts_i]
        ys = [p[1] for p in pts_i]
        w = float(max(xs) - min(xs))
        h = float(max(ys) - min(ys))
        area = max(w * h, 1.0)
        cx = float(sum(xs) / 4.0)
        cy = float(sum(ys) / 4.0)

        dets.append({
            "pts": pts_i,
            "text": text,
            "conf": float(conf),
            "w": w,
            "h": h,
            "area": area,
            "cx": cx,
            "cy": cy,
        })
    return dets


def draw_dets(img_bgr: np.ndarray, dets, min_conf: float = 0.55):
    for d in dets:
        if d["conf"] < min_conf:
            continue
        pts = np.array(d["pts"], dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(img_bgr, [pts], True, (255, 0, 0), 2)
        x0, y0 = min(p[0] for p in d["pts"]), min(p[1] for p in d["pts"])
        label = f"{d['text']} ({d['conf']:.2f})"
        cv.putText(
            img_bgr,
            label,
            (x0, max(0, y0 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,          # ↑ font scale (try 0.9–1.2)
            (255, 0, 0),
            3             # ↑ thickness
        )



# ------------------------- Product text scoring -------------------------

_STOP = {
    # very common non-product phrases
    "nutrition", "facts", "ingredients", "ingredient", "serving", "servings", "calories",
    "best", "before", "use", "by", "keep", "refrigerated", "net", "wt", "weight",
    "distributed", "manufactured", "www", "com", "ltd", "inc", "llc",
    "barcode", "scan", "qr"
}

def _norm_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str):
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]

def score_line(line: str, conf: float, area: float, img_w: int, img_h: int, cx: float, cy: float):
    """
    Heuristic “product name likelihood” score.
    Prefers:
      - higher confidence
      - larger text region
      - centered-ish text (front label tends to be central)
      - alphabetic content
    Penalizes:
      - tiny text
      - mostly numeric
      - boilerplate terms (nutrition facts, etc.)
    """
    t = _norm_text(line)
    toks = _tokenize(t)

    if not toks:
        return -1e9

    # Center bias (front label often near mid)
    nx = abs(cx - img_w / 2) / max(img_w, 1)
    ny = abs(cy - img_h / 2) / max(img_h, 1)
    center_bonus = 1.0 - min(1.0, 0.7 * nx + 0.7 * ny)

    # Text composition
    letters = sum(c.isalpha() for c in t)
    digits = sum(c.isdigit() for c in t)
    letter_ratio = letters / max(letters + digits, 1)

    # Stopword penalty
    stop_hits = sum(1 for tok in toks if tok in _STOP)
    stop_pen = 0.15 * stop_hits

    # Short/long penalty
    length = len(t)
    if length < 3:
        return -1e9
    len_bonus = min(1.0, length / 20.0)

    # Area normalization (relative)
    rel_area = area / max(img_w * img_h, 1)
    area_bonus = min(1.0, rel_area * 30.0)  # scale up small text regions

    # Combine
    score = 0.0
    score += 2.2 * conf
    score += 1.6 * area_bonus
    score += 1.0 * center_bonus
    score += 0.8 * letter_ratio
    score += 0.6 * len_bonus
    score -= stop_pen

    # Penalize mostly numeric lines
    if letter_ratio < 0.35:
        score -= 0.7

    return score


def best_product_text(dets, img_shape):
    if not dets:
        return None, []

    h, w = img_shape[:2]
    scored = []
    for d in dets:
        s = score_line(d["text"], d["conf"], d["area"], w, h, d["cx"], d["cy"])
        scored.append((s, d))

    scored.sort(key=lambda x: x[0], reverse=True)

    # best line
    best = scored[0][1]["text"] if scored else None

    # also return top few candidates for debugging
    topk = [(round(s, 3), d["text"], round(d["conf"], 2)) for s, d in scored[:8]]
    return best, topk


# ------------------------- Optional auto-crop to “main label region” -------------------------

def auto_crop_label_region(img_bgr: np.ndarray):
    """
    Cheap heuristic crop:
      - find strongest edges
      - take largest connected-ish contour box
    This often helps by removing shelf clutter.
    """
    h, w = img_bgr.shape[:2]
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 60, 160)

    # Close gaps
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, (0, 0, w, h), False

    # pick large-ish contour near center
    best = None
    best_score = -1
    for c in contours:
        x, y, cw, ch = cv.boundingRect(c)
        area = cw * ch
        if area < 0.08 * w * h:
            continue  # too small
        cx = x + cw / 2
        cy = y + ch / 2
        center_bias = 1.0 - (abs(cx - w/2) / w + abs(cy - h/2) / h)
        s = area * (0.5 + max(0.0, center_bias))
        if s > best_score:
            best_score = s
            best = (x, y, cw, ch)

    if best is None:
        return img_bgr, (0, 0, w, h), False

    x, y, cw, ch = best
    pad = int(0.06 * min(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + cw + pad)
    y1 = min(h, y + ch + pad)

    crop = img_bgr[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1 - x0, y1 - y0), True


# ------------------------- CLI modes -------------------------

def run_image(path: str, save_path: str | None, min_conf: float, use_autocrop: bool):
    img = cv.imread(path)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")

    # PaddleOCR: cls=True helps rotated text; adjust if too slow
    ocr = PaddleOCR(show_log=False, use_gpu=False, use_tensorrt=False, lang="en")

    if use_autocrop:
        crop, (x, y, cw, ch), ok = auto_crop_label_region(img)
        dets = run_ocr(ocr, crop)
        best, topk = best_product_text(dets, crop.shape)

        # draw on crop then paste back
        draw_dets(crop, dets, min_conf=min_conf)
        out = img.copy()
        out[y:y+ch, x:x+cw] = crop
        cv.rectangle(out, (x, y), (x+cw, y+ch), (0, 255, 255), 2)

    else:
        dets = run_ocr(ocr, img)
        best, topk = best_product_text(dets, img.shape)
        out = img.copy()
        draw_dets(out, dets, min_conf=min_conf)

    print("\n[product text guess]")
    print(best if best else "(no text detected)")

    print("\n[top candidates]")
    for s, t, c in topk:
        print(f"  score={s:>5} conf={c:>4}  {t}")

    if save_path:
        cv.imwrite(save_path, out)
        print("\n[saved]", save_path)
    else:
        cv.imshow("ocr_product_reader", out)
        cv.waitKey(0)
        cv.destroyAllWindows()


def run_webcam(cam_index: int, every_n: int, min_conf: float, use_autocrop: bool):
    cap = cv.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {cam_index}")

    ocr = PaddleOCR(show_log=False, use_gpu=False, use_tensorrt=False, lang="en")
    frame_i = 0
    last_best = None
    last_topk = []
    last_dets = []
    last_crop_box = None

    print("\nPress 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        if frame_i % max(1, every_n) == 0:
            if use_autocrop:
                crop, box, _ = auto_crop_label_region(frame)
                dets = run_ocr(ocr, crop)
                best, topk = best_product_text(dets, crop.shape)
                last_crop_box = box
                last_dets = dets
                last_best = best
                last_topk = topk
            else:
                dets = run_ocr(ocr, frame)
                best, topk = best_product_text(dets, frame.shape)
                last_crop_box = None
                last_dets = dets
                last_best = best
                last_topk = topk

            print("\n--- frame", frame_i, "---")
            print("[guess]", last_best if last_best else "(none)")

        disp = frame.copy()

        if use_autocrop and last_crop_box is not None:
            x, y, w, h = last_crop_box
            # draw OCR on cropped overlay area by re-running draw on that region
            crop_view = disp[y:y+h, x:x+w]
            draw_dets(crop_view, last_dets, min_conf=min_conf)
            cv.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 255), 2)
        else:
            draw_dets(disp, last_dets, min_conf=min_conf)

        if last_best:
            cv.putText(disp, f"Guess: {last_best[:60]}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv.imshow("ocr_product_reader", disp)
        if (cv.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="OCR-only product text reader (bounding boxes + best guess).")
    ap.add_argument("--image", default="", help="Path to an image to OCR")
    ap.add_argument("--save", default="", help="Save annotated output image")
    ap.add_argument("--cam", type=int, default=-1, help="Webcam index (>=0)")
    ap.add_argument("--every", type=int, default=10, help="In webcam mode, OCR every N frames")
    ap.add_argument("--min_conf", type=float, default=0.55, help="Min confidence to draw boxes")
    ap.add_argument("--autocrop", action="store_true", help="Auto-crop likely label region before OCR")
    args = ap.parse_args()

    if args.image:
        run_image(args.image, args.save or None, args.min_conf, args.autocrop)
    elif args.cam >= 0:
        run_webcam(args.cam, args.every, args.min_conf, args.autocrop)
    else:
        raise SystemExit("Provide --image PATH or --cam N")

if __name__ == "__main__":
    main()
