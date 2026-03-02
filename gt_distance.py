import argparse
import csv
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional


def parse_points_attr(points_str: str) -> List[Tuple[float, float]]:
    """
    CVAT points format examples:
      "1573.77,1499.26"
      "2186.34,621.54;2190.91,616.97"
    """
    pts = []
    points_str = (points_str or "").strip()
    if not points_str:
        return pts

    for chunk in points_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_str, y_str = chunk.split(",")
        pts.append((float(x_str), float(y_str)))
    return pts


def centroid(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not pts:
        return None
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    return (sx / len(pts), sy / len(pts))


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to CVAT annotations.xml")
    ap.add_argument("--out_csv", default="distances.csv", help="Output CSV path")
    ap.add_argument("--finger_label", default="finger point", help="Label name for fingertip point")
    ap.add_argument("--object_label", default="object", help="Label name for object point(s)")
    args = ap.parse_args()

    tree = ET.parse(args.xml)
    root = tree.getroot()

    rows = []
    skipped = 0

    for img in root.findall("image"):
        img_id = img.get("id")
        img_name = img.get("name")
        w = float(img.get("width", "0"))
        h = float(img.get("height", "0"))
        diag = math.sqrt(w * w + h * h) if (w > 0 and h > 0) else None

        finger_pts = []
        object_pts = []

        for p in img.findall("points"):
            label = p.get("label", "").strip()
            pts = parse_points_attr(p.get("points", ""))
            if label == args.finger_label:
                finger_pts.extend(pts)
            elif label == args.object_label:
                object_pts.extend(pts)

        # Fingertip: if multiple accidentally exist, we use the centroid too (robust).
        finger_xy = centroid(finger_pts)
        object_xy = centroid(object_pts)

        if finger_xy is None or object_xy is None:
            skipped += 1
            continue

        dist_px = euclidean(finger_xy, object_xy)
        dist_norm = (dist_px / diag) if diag and diag > 0 else None

        rows.append({
            "image_id": img_id,
            "image_name": img_name,
            "width": int(w) if w else "",
            "height": int(h) if h else "",
            "finger_x": finger_xy[0],
            "finger_y": finger_xy[1],
            "object_x": object_xy[0],
            "object_y": object_xy[1],
            "dist_px": dist_px,
            "dist_norm": dist_norm if dist_norm is not None else "",
        })

    with open(args.out_csv, "w", newline="") as f:
        fieldnames = [
            "image_id", "image_name", "width", "height",
            "finger_x", "finger_y", "object_x", "object_y",
            "dist_px", "dist_norm"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}. Skipped {skipped} images (missing finger/object).")


if __name__ == "__main__":
    main()