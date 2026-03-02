#!/usr/bin/env python3
# evaluate_fingertip_item_predictions.py
#
# PURPOSE:
#   Compare prediction CSV (from your debug pipeline) against GT CVAT XML annotations.
#
# GT expected:
#   CVAT XML with labels:
#     - "finger point"
#     - "object"
#
# Prediction CSV expected (from your current pipeline):
#   Required columns:
#     - image_name
#     - tip_x, tip_y
#     - item_center_x, item_center_y
#   Optional columns:
#     - item_x1, item_y1, item_x2, item_y2
#     - distance_px
#
# METRICS:
#   - fingertip point error (px)
#   - object center error (px)
#   - GT distance vs predicted distance error (px and %)
#
# USAGE:
#   python evaluate_fingertip_item_predictions.py \
#       --gt_xml /path/to/annotations.xml \
#       --pred_csv /path/to/fingertip_item_distances_v5.csv \
#       --out_csv /path/to/eval_per_image.csv
#
# NOTES:
#   - If GT object has multiple points, uses centroid of GT object points.
#   - If predicted distance_px missing, recomputes from predicted tip/item center.
#   - If an image is missing GT or prediction fields, it is skipped for that metric.

import argparse
import csv
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ------------------------------------------------------------
# Basic geometry helpers
# ------------------------------------------------------------

def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def centroid(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def rmse(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals) / len(vals))


# ------------------------------------------------------------
# GT loading from CVAT XML
# ------------------------------------------------------------

def parse_points_attr(points_attr: str) -> List[Tuple[float, float]]:
    """
    Example:
      "2186.34,621.54;2190.91,616.97"
    """
    out = []
    if not points_attr:
        return out
    for part in points_attr.split(";"):
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) != 2:
            continue
        try:
            x = float(xy[0])
            y = float(xy[1])
            out.append((x, y))
        except ValueError:
            continue
    return out


def load_gt_from_cvat_xml(xml_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "IMG_4912.JPG": {
            "finger_gt": (x, y) or None,
            "object_gt_points": [(x,y), ...],
            "object_gt_center": (x, y) or None,
            "gt_distance_px": float or None,
            "width": int,
            "height": int,
        },
        ...
      }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt = {}

    for image_el in root.findall("image"):
        image_name = image_el.attrib.get("name", "").strip()
        width = int(image_el.attrib.get("width", "0"))
        height = int(image_el.attrib.get("height", "0"))

        finger_pt = None
        object_pts: List[Tuple[float, float]] = []

        for points_el in image_el.findall("points"):
            label = points_el.attrib.get("label", "").strip().lower()
            pts_attr = points_el.attrib.get("points", "")
            pts = parse_points_attr(pts_attr)

            if label == "finger point":
                # Use first point if multiple were somehow given
                if pts:
                    finger_pt = pts[0]

            elif label == "object":
                # Use all object points (could be one point or multiple)
                object_pts.extend(pts)

        object_center = centroid(object_pts)

        gt_distance = None
        if finger_pt is not None and object_center is not None:
            gt_distance = euclidean(finger_pt, object_center)

        gt[image_name] = {
            "finger_gt": finger_pt,
            "object_gt_points": object_pts,
            "object_gt_center": object_center,
            "gt_distance_px": gt_distance,
            "width": width,
            "height": height,
        }

    return gt


# ------------------------------------------------------------
# Prediction CSV loading
# ------------------------------------------------------------

def load_predictions(pred_csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Expected columns from your pipeline:
      image_name
      tip_x, tip_y
      item_center_x, item_center_y
      optional: item_x1, item_y1, item_x2, item_y2, distance_px
    """
    preds = {}

    with open(pred_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = str(row.get("image_name", "")).strip()
            if not image_name:
                continue

            tip_x = safe_float(row.get("tip_x"))
            tip_y = safe_float(row.get("tip_y"))
            item_cx = safe_float(row.get("item_center_x"))
            item_cy = safe_float(row.get("item_center_y"))

            item_x1 = safe_float(row.get("item_x1"))
            item_y1 = safe_float(row.get("item_y1"))
            item_x2 = safe_float(row.get("item_x2"))
            item_y2 = safe_float(row.get("item_y2"))

            pred_tip = (tip_x, tip_y) if tip_x is not None and tip_y is not None else None
            pred_item_center = (item_cx, item_cy) if item_cx is not None and item_cy is not None else None

            pred_distance = safe_float(row.get("distance_px"))
            if pred_distance is None and pred_tip is not None and pred_item_center is not None:
                pred_distance = euclidean(pred_tip, pred_item_center)

            pred_item_box = None
            if None not in (item_x1, item_y1, item_x2, item_y2):
                pred_item_box = (item_x1, item_y1, item_x2, item_y2)

            preds[image_name] = {
                "pred_tip": pred_tip,
                "pred_item_center": pred_item_center,
                "pred_item_box": pred_item_box,
                "pred_distance_px": pred_distance,
                "raw_row": row,
            }

    return preds


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate(gt_map: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Per-image evaluation rows.
    """
    rows = []

    all_image_names = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    for image_name in all_image_names:
        gt = gt_map.get(image_name)
        pred = pred_map.get(image_name)

        row: Dict[str, Any] = {
            "image_name": image_name,
            "has_gt": int(gt is not None),
            "has_pred": int(pred is not None),
        }

        if gt is None:
            row["status"] = "missing_gt"
            rows.append(row)
            continue

        if pred is None:
            row["status"] = "missing_prediction"
            rows.append(row)
            continue

        finger_gt = gt.get("finger_gt")
        object_gt_center = gt.get("object_gt_center")
        gt_distance_px = gt.get("gt_distance_px")

        pred_tip = pred.get("pred_tip")
        pred_item_center = pred.get("pred_item_center")
        pred_item_box = pred.get("pred_item_box")
        pred_distance_px = pred.get("pred_distance_px")

        # Copy GT
        row["finger_gt_x"] = finger_gt[0] if finger_gt is not None else ""
        row["finger_gt_y"] = finger_gt[1] if finger_gt is not None else ""

        row["object_gt_center_x"] = object_gt_center[0] if object_gt_center is not None else ""
        row["object_gt_center_y"] = object_gt_center[1] if object_gt_center is not None else ""
        row["gt_distance_px"] = gt_distance_px if gt_distance_px is not None else ""

        # Copy prediction
        row["pred_tip_x"] = pred_tip[0] if pred_tip is not None else ""
        row["pred_tip_y"] = pred_tip[1] if pred_tip is not None else ""

        row["pred_item_center_x"] = pred_item_center[0] if pred_item_center is not None else ""
        row["pred_item_center_y"] = pred_item_center[1] if pred_item_center is not None else ""
        row["pred_distance_px"] = pred_distance_px if pred_distance_px is not None else ""

        if pred_item_box is not None:
            row["pred_item_x1"] = pred_item_box[0]
            row["pred_item_y1"] = pred_item_box[1]
            row["pred_item_x2"] = pred_item_box[2]
            row["pred_item_y2"] = pred_item_box[3]
        else:
            row["pred_item_x1"] = row["pred_item_y1"] = row["pred_item_x2"] = row["pred_item_y2"] = ""

        # Metrics
        finger_err = None
        object_err = None
        dist_abs_err = None
        dist_rel_err_pct = None

        if finger_gt is not None and pred_tip is not None:
            finger_err = euclidean(finger_gt, pred_tip)

        if object_gt_center is not None and pred_item_center is not None:
            object_err = euclidean(object_gt_center, pred_item_center)

        if gt_distance_px is not None and pred_distance_px is not None:
            dist_abs_err = abs(pred_distance_px - gt_distance_px)
            if gt_distance_px > 1e-9:
                dist_rel_err_pct = (dist_abs_err / gt_distance_px) * 100.0

        row["finger_error_px"] = finger_err if finger_err is not None else ""
        row["object_center_error_px"] = object_err if object_err is not None else ""
        row["distance_abs_error_px"] = dist_abs_err if dist_abs_err is not None else ""
        row["distance_rel_error_pct"] = dist_rel_err_pct if dist_rel_err_pct is not None else ""

        # Object box diagnostic (optional)
        if pred_item_box is not None and object_gt_center is not None:
            x1, y1, x2, y2 = pred_item_box
            inside = int(x1 <= object_gt_center[0] <= x2 and y1 <= object_gt_center[1] <= y2)
            row["gt_object_center_inside_pred_box"] = inside
        else:
            row["gt_object_center_inside_pred_box"] = ""

        row["status"] = "ok"
        rows.append(row)

    return rows


# ------------------------------------------------------------
# Summary printing
# ------------------------------------------------------------

def summarize(eval_rows: List[Dict[str, Any]]) -> None:
    finger_errs = []
    object_errs = []
    dist_errs = []
    dist_rel_errs = []

    missing_gt = 0
    missing_pred = 0

    for r in eval_rows:
        status = r.get("status")
        if status == "missing_gt":
            missing_gt += 1
            continue
        if status == "missing_prediction":
            missing_pred += 1
            continue

        fe = safe_float(r.get("finger_error_px"))
        oe = safe_float(r.get("object_center_error_px"))
        de = safe_float(r.get("distance_abs_error_px"))
        dre = safe_float(r.get("distance_rel_error_pct"))

        if fe is not None:
            finger_errs.append(fe)
        if oe is not None:
            object_errs.append(oe)
        if de is not None:
            dist_errs.append(de)
        if dre is not None:
            dist_rel_errs.append(dre)

    def print_metric_block(name: str, vals: List[float]) -> None:
        if not vals:
            print(f"{name}: no valid samples")
            return
        print(
            f"{name}: "
            f"n={len(vals)} | "
            f"mean={sum(vals)/len(vals):.3f} | "
            f"median={statistics.median(vals):.3f} | "
            f"rmse={rmse(vals):.3f} | "
            f"min={min(vals):.3f} | "
            f"max={max(vals):.3f}"
        )

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total rows: {len(eval_rows)}")
    print(f"Missing GT: {missing_gt}")
    print(f"Missing prediction: {missing_pred}")
    print_metric_block("Finger error (px)", finger_errs)
    print_metric_block("Object center error (px)", object_errs)
    print_metric_block("Distance abs error (px)", dist_errs)
    print_metric_block("Distance rel error (%)", dist_rel_errs)


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

def save_eval_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        return

    # Collect all keys in stable order
    preferred = [
        "image_name", "status", "has_gt", "has_pred",
        "finger_gt_x", "finger_gt_y",
        "object_gt_center_x", "object_gt_center_y", "gt_distance_px",
        "pred_tip_x", "pred_tip_y",
        "pred_item_center_x", "pred_item_center_y", "pred_distance_px",
        "pred_item_x1", "pred_item_y1", "pred_item_x2", "pred_item_y2",
        "finger_error_px", "object_center_error_px",
        "distance_abs_error_px", "distance_rel_error_pct",
        "gt_object_center_inside_pred_box",
    ]

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    fieldnames = preferred + [k for k in sorted(all_keys) if k not in preferred]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fingertip/object/distance predictions against GT CVAT XML."
    )
    parser.add_argument("--gt_xml", required=True, help="Path to GT CVAT XML annotations file")
    parser.add_argument("--pred_csv", required=True, help="Path to prediction CSV from your pipeline")
    parser.add_argument("--out_csv", default="evaluation_per_image.csv", help="Path to save per-image evaluation CSV")

    args = parser.parse_args()

    gt_xml = Path(args.gt_xml)
    pred_csv = Path(args.pred_csv)
    out_csv = Path(args.out_csv)

    if not gt_xml.exists():
        raise FileNotFoundError(f"GT XML not found: {gt_xml}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    print(f"[INFO] Loading GT from: {gt_xml}")
    gt_map = load_gt_from_cvat_xml(gt_xml)
    print(f"[INFO] Loaded GT for {len(gt_map)} images")

    print(f"[INFO] Loading predictions from: {pred_csv}")
    pred_map = load_predictions(pred_csv)
    print(f"[INFO] Loaded predictions for {len(pred_map)} images")

    eval_rows = evaluate(gt_map, pred_map)
    save_eval_csv(eval_rows, out_csv)

    summarize(eval_rows)
    print(f"\nPer-image evaluation saved to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()