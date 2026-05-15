#!/usr/bin/env python3
# embed_dino.py
#
# YOLOWorld + DINOv3-small + HSV histogram fusion
# for retail / grocery shelf product matching.
#
# DINOv3 (Meta, Aug 2025) key differences from DINOv2:
#   - Patch size 16px (not 14px)
#   - 4 register tokens sit between CLS and patch tokens:
#     [CLS][reg0][reg1][reg2][reg3][patch_0...patch_P]
#     Patch tokens start at index 5 (was 1 in DINOv2)
#   - Register tokens absorb background attention → cleaner spatial mask
#   - Trained on 1.7B images via 7B-param teacher → richer features
#   - Requires transformers >= 4.56.0
#   - Gated on HuggingFace — accept license at:
#     https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
#
# Changes vs. original embed_world_aisle.py (CLIP version):
#   1. DINOv3-small replaces open_clip (448px, multi-layer CLS,
#      attention-weighted patch pooling with register token skip)
#   2. Pure DINOv3 cosine similarity (register tokens handle color masking)
#   3. TARGET mode: pure sim ranking (no YOLO-conf / center bias sabotage)
#   4. TARGET mode: draws ALL accepted boxes, not just one winner
#   5. Aspect-ratio filter: rejects near-square blobs (badges, logos, circles)
#
# Install:
#   pip install torch torchvision ultralytics opencv-python numpy
#   pip install --upgrade "transformers>=4.56.0"
#   pip install rembg onnxruntime          # background removal for reference images
#
# Example (target mode — find all Orbit Spearmint packs):
#   python embed_dino.py \
#     --images_dir images2 \
#     --gallery_dir reference2 \
#     --det_model yolov8s-worldv2.pt \
#     --prompts "item,product,package,box,square" \
#     --mode target \
#     --target_label spearmint \
#     --det_conf 0.001 \
#     --min_box_area_px 50 \
#     --max_det 150 \
#     --sim_threshold 0.62 \
#     --out_csv orbit_spearmint_search.csv \
#     --save_annotated --annotated_dir aisle_embed_debug2 \
#     --save_crops    --crop_dir aisle_best_crops2 \
#     --save_raw_debug --raw_debug_dir aisle_raw_debug2 \
#     --save_all_candidate_crops --all_candidate_crop_dir aisle_all_candidate_crops

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError as e:
    raise ImportError("pip install transformers") from e

try:
    from ultralytics import YOLOWorld
except ImportError:
    from ultralytics import YOLO as YOLOWorld


# ─────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────
DINO_MODEL_ID       = "facebook/dinov3-vits16-pretrain-lvd1689m"  # ViT-S/16, 384-dim
EMBED_DIM           = 384
MULTILAYER_DEPTH    = 4      # average CLS from last N transformer layers
DINO_INPUT_SIZE     = 448    # 448/16 = 28x28 = 784 patches
NUM_REGISTER_TOKENS = 4      # DINOv3: [CLS][reg0..reg3][patch_0..patch_P]
                             # patch tokens start at index 5, NOT 1

# Spatial color histogram (thirds-based)
HIST_BINS   = 16     # HSV bins per channel per third
HIST_WEIGHT = 0.25   # blend: final = (1-HIST_WEIGHT)*dino_cos + HIST_WEIGHT*thirds_hist
REF_BG_FILL = (40, 40, 40)   # dark grey to replace white background in reference images
                             # matches typical shelf/dark-background store environment


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def list_image_files(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name.lower(),
    )


def ensure_dir(path: Optional[Path]) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def clamp_xy(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))


def compute_box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box.tolist()
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_box_center(box: np.ndarray, w: int, h: int) -> Tuple[int, int]:
    x1, y1, x2, y2 = box.tolist()
    return clamp_xy(int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)), w, h)


def box_aspect_ratio(box: np.ndarray) -> float:
    """width / height of a xyxy box. >1 = landscape, <1 = portrait, ~1 = square."""
    x1, y1, x2, y2 = box.tolist()
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    return bw / bh


def pad_to_square(img: np.ndarray, fill: int = 0) -> np.ndarray:
    h, w   = img.shape[:2]
    side   = max(h, w)
    canvas = (
        np.full((side, side, img.shape[2]), fill, dtype=img.dtype)
        if img.ndim == 3 else
        np.full((side, side), fill, dtype=img.dtype)
    )
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def parse_prompts(s: str) -> List[str]:
    prompts = [p.strip() for p in s.split(",") if p.strip()]
    if not prompts:
        raise ValueError("No valid prompts provided.")
    return prompts


def infer_label(path: Path) -> str:
    """spearmint__1.jpg  →  spearmint  |  spearmint.jpg  →  spearmint"""
    stem = path.stem
    return stem.split("__")[0] if "__" in stem else stem






# ─────────────────────────────────────────────────────────────
# Reference image preprocessing  (domain gap fix)
# ─────────────────────────────────────────────────────────────

try:
    from rembg import remove as rembg_remove
    from rembg import new_session as rembg_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# rembg model options (best → fastest):
#   birefnet-general      — best quality, handles product packs well (recommended)
#   birefnet-general-lite — faster birefnet, slightly lower quality
#   isnet-general-use     — good fallback, faster than birefnet
#   u2net                 — original default, tends to over-remove on products
REMBG_MODEL = "birefnet-general"

def remove_reference_background(
    img_bgr: np.ndarray,
    bg_fill: Tuple[int, int, int] = REF_BG_FILL,
    save_debug_path: Optional[str] = None,
    _session=None,
    # ── Sensitivity controls ──────────────────────────────────────────────
    alpha_threshold:  int   = 0,
    # alpha_threshold (0-255):
    #   Pixels with alpha < this value are treated as fully background.
    #   Default 0 = use raw rembg alpha exactly as-is (most conservative,
    #   least product removal). Raise to 25-50 if edges look too soft/hazy.
    #   Lower = keeps more product pixels (less aggressive removal).
    #   Higher = more aggressive — may eat into product edges.
    erode_px:         int   = 0,
    # erode_px (0-5):
    #   After thresholding, shrink the foreground mask inward by this many
    #   pixels. Cleans up "halo" fringing where background color bleeds into
    #   the mask edge. Set to 1-2 if you see a white fringe around the product.
    #   0 = no erosion (safest — preserves maximum product area).
    dilate_px:        int   = 3,
    # dilate_px (0-10):
    #   After erode, expand the foreground mask outward by this many pixels.
    #   This recovers product pixels that rembg under-classified as background
    #   (thin bag edges, text at the border, crinkled bag areas near edges).
    #   dilate_px > erode_px = net expansion = LESS aggressive removal.
    #   dilate_px < erode_px = net shrink    = MORE aggressive removal.
    #   Default dilate=3, erode=0: expands by 3px — recovers lost edges.
    blur_mask_px:     int   = 3,
    # blur_mask_px (0-21, must be odd):
    #   Gaussian blur on the final alpha mask before compositing.
    #   Creates a soft feathered transition instead of a hard binary edge.
    #   0 = hard edge (sharp cutout — good for flat product photos).
    #   3-7 = soft blend — better for crinkled bags with irregular edges.
    #   Higher = more blending into background (looks natural but loses detail).
) -> np.ndarray:
    """
    Remove the background from a reference image using rembg (U2Net model)
    and composite the product onto a solid dark background.

    Sensitivity summary — what to change when:

      Too much removed (product edges missing, thin parts gone):
        → Lower alpha_threshold (try 0)
        → Increase dilate_px (try 5-8)
        → Decrease erode_px (try 0)
        → These all = LESS aggressive = more product kept

      Too little removed (background halo or fringe remains):
        → Raise alpha_threshold (try 25-50)
        → Increase erode_px (try 1-3)
        → Decrease dilate_px (try 0-1)
        → These all = MORE aggressive = more background removed

      Edges look jagged / harsh cutout:
        → Increase blur_mask_px (try 5-9)

      Edges look blurry / washed out:
        → Decrease blur_mask_px (try 0-1)

    Default settings (alpha_threshold=0, erode=0, dilate=3, blur=3) are
    tuned to be conservative — keep as much product as possible and only
    expand the mask slightly to recover rembg under-classification.
    """
    if not REMBG_AVAILABLE:
        print("[WARN] rembg not installed — returning image unchanged.")
        print("       Install with: pip install rembg onnxruntime")
        return img_bgr

    # ── Step 1: rembg inference → raw RGBA ───────────────────────────────
    rgb_pil  = Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    rgba_pil = rembg_remove(rgb_pil, session=_session) if _session else rembg_remove(rgb_pil)

    rgba  = np.array(rgba_pil)                          # (H, W, 4) uint8
    rgb   = rgba[:, :, :3]                              # (H, W, 3) uint8
    alpha = rgba[:, :, 3]                               # (H, W)    uint8  0-255

    # ── Step 2: Alpha threshold ───────────────────────────────────────────
    # Binarise the raw soft alpha into a hard mask.
    # alpha_threshold=0 keeps rembg's soft alpha as-is (skip binarise).
    # Any value >0 makes it binary: foreground or background, nothing in between.
    if alpha_threshold > 0:
        mask = (alpha >= alpha_threshold).astype(np.uint8) * 255   # (H, W) binary
    else:
        mask = alpha.copy()                                         # keep soft alpha

    # ── Step 3: Morphological erode then dilate ───────────────────────────
    # Erode first: removes background fringe/halo at product edges.
    # Dilate second: recovers under-classified product pixels near edges.
    # Net effect = dilate_px - erode_px pixels of expansion/contraction.
    if erode_px > 0:
        k    = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        mask = cv.erode(mask, k, iterations=1)

    if dilate_px > 0:
        k    = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        mask = cv.dilate(mask, k, iterations=1)

    # ── Step 4: Gaussian blur for soft feathered edges ────────────────────
    # Converts hard binary mask back into a soft alpha for natural blending.
    if blur_mask_px > 0:
        ksize = blur_mask_px if blur_mask_px % 2 == 1 else blur_mask_px + 1
        mask  = cv.GaussianBlur(mask, (ksize, ksize), 0)

    # ── Step 5: Composite onto background fill ────────────────────────────
    # output = (mask/255) * product + (1 - mask/255) * bg_fill
    alpha_f = mask.astype(np.float32)[:, :, None] / 255.0          # (H, W, 1) [0,1]
    rgb_f   = rgb.astype(np.float32)
    bg_rgb  = np.array(bg_fill[::-1], dtype=np.float32)            # BGR→RGB
    composited = (alpha_f * rgb_f + (1.0 - alpha_f) * bg_rgb).clip(0, 255).astype(np.uint8)

    result = cv.cvtColor(composited, cv.COLOR_RGB2BGR)

    if save_debug_path:
        cv.imwrite(save_debug_path, result)

    return result

# ─────────────────────────────────────────────────────────────
# Spatial color histogram — thirds
# ─────────────────────────────────────────────────────────────

def _hsv_hist_single(region_bgr: np.ndarray, bins: int) -> np.ndarray:
    """
    Per-channel L1-normalised HSV histogram for one image region.
    Returns shape (bins * 3,) — concatenated H, S, V channels.

    NOTE on normalization:
      Each channel is independently L1-normalised → each channel sums to 1.0.
      The FULL returned vector therefore sums to 3.0 (one per channel), not 1.0.
      This is intentional — thirds_histogram_similarity accounts for this by
      dividing by (num_thirds × num_channels) = 9 to keep scores in [0, 1].
      Do NOT normalise the full concatenated vector to 1.0 — that would make
      low-saturation regions (near-grey pixels with S≈0) dominate the histogram
      by inflating the V channel's relative contribution.
    """
    hsv    = cv.cvtColor(region_bgr, cv.COLOR_BGR2HSV)
    ranges = [(0, 180), (0, 256), (0, 256)]   # OpenCV H ∈ [0,179]
    hists  = []
    for ch, (lo, hi) in enumerate(ranges):
        h = cv.calcHist([hsv], [ch], None, [bins], [lo, hi]).flatten()
        s = h.sum()
        hists.append(h / s if s > 0 else h)
    return np.concatenate(hists).astype(np.float32)


def extract_thirds_histogram(img_bgr: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """
    Split the image into thirds along its LONG axis, compute an HSV histogram
    for each third, and concatenate them.

    Why thirds instead of a global histogram:
      A global histogram sees "35% green" for both Orbit Spearmint and Dentyne ICE.
      Thirds capture *where* the color lives spatially:
        Orbit Spearmint (portrait pack):
          top-third    → green (header band)
          middle-third → white + blue logo
          bottom-third → green (footer band)
        Dentyne ICE (landscape pack):
          top-third    → green + white text
          middle-third → dark blue logo
          bottom-third → green + silver text
      Those are very different histograms even though both packs are "green."

    Split axis:
      height > width  → portrait  → split horizontally (top / mid / bot)
      width >= height → landscape → split vertically   (left / mid / right)

    Returns shape (3 * bins * 3,) — 3 thirds × 3 channels × bins.
    """
    h, w = img_bgr.shape[:2]

    if h >= w:
        # Portrait: horizontal thirds
        t = h // 3
        regions = [
            img_bgr[0:t,       :],   # top
            img_bgr[t:2*t,     :],   # middle
            img_bgr[2*t:,      :],   # bottom
        ]
    else:
        # Landscape: vertical thirds
        t = w // 3
        regions = [
            img_bgr[:,   0:t  ],     # left
            img_bgr[:,   t:2*t],     # middle
            img_bgr[:,   2*t: ],     # right
        ]

    hists = []
    for region in regions:
        if region.size == 0:
            hists.append(np.zeros(bins * 3, dtype=np.float32))
        else:
            hists.append(_hsv_hist_single(region, bins))

    return np.concatenate(hists)   # (3 * bins * 3,)


def thirds_histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Per-third histogram intersection, averaged across thirds AND channels.
    Result is always in [0, 1].

    THE NORMALIZATION BUG (fixed here):
      Each third contains 3 HSV channels, each L1-normalised to 1.0.
      So each third's vector sums to 3.0, not 1.0.
      The raw intersection of one third can therefore be up to 3.0.
      Dividing by 3 (number of thirds) alone gives max = 3.0 — WRONG.
      We must also divide by 3 (number of channels) per third.
      Total denominator: 3 thirds × 3 channels = 9.
      This gives intersection ∈ [0, 1] correctly.

    Example for Orbit Spearmint vs Dentyne ICE:
      third 0 (top):    green vs green     → raw ≈ 2.4 / 9 term = 0.27
      third 1 (middle): white/logo vs navy → raw ≈ 0.6 / 9 term = 0.07
      third 2 (bottom): green vs green+sil → raw ≈ 1.6 / 9 term = 0.18
      total → 0.52   (correctly low despite both being "green packs")

    vs Orbit Spearmint vs Orbit Spearmint reference:
      all thirds match well → total → 0.85+
    """
    NUM_THIRDS   = 3
    NUM_CHANNELS = 3   # H, S, V
    chunk = len(h1) // NUM_THIRDS
    score = 0.0
    for i in range(NUM_THIRDS):
        a = h1[i*chunk:(i+1)*chunk]
        b = h2[i*chunk:(i+1)*chunk]
        score += float(np.minimum(a, b).sum())
    # Divide by thirds × channels so result ∈ [0, 1]
    return score / float(NUM_THIRDS * NUM_CHANNELS)


# ─────────────────────────────────────────────────────────────
# DINOv3 embedder
# ─────────────────────────────────────────────────────────────

class DINOv2Embedder:
    """
    DINOv3-small embedder. Identical public interface to the old DINOv2 version.

    KEY DINOv3 STRUCTURAL CHANGES:

    1. PATCH SIZE 16 (was 14)
       448px / 16 = 28x28 = 784 patches (vs 1024 DINOv2 patches at same res).

    2. REGISTER TOKENS: 4 extra tokens inserted after CLS, before patches.
       DINOv3 sequence: [CLS][reg0][reg1][reg2][reg3][patch_0 ... patch_783]
       Old [:, 1:] slice included registers as fake patches -- now use [:, 5:].

    HOW REGISTER TOKENS IMPROVE THE COLOR MASK:
       In DINOv2 the background (shelf, price tags, empty space) had no
       dedicated token to attend to, so it polluted the CLS->patch attention
       map with diffuse low-value weights across the whole image.
       DINOv3 register tokens act as attention sinks: background patches
       dump their attention onto registers, leaving the CLS->patch weights
       concentrated on the discriminative product region (logo, color face).
       The attention-weighted pooling therefore produces a much cleaner
       descriptor -- essentially a learned foreground mask over color+texture.

    3. MULTI-LAYER CLS: average last MULTILAYER_DEPTH layer CLS tokens.
    4. ATTENTION-WEIGHTED PATCH POOL: register-aware slicing at index ps=5.
    """

    def __init__(self, model_id: str = DINO_MODEL_ID, device: str = "cpu"):
        self.device      = device
        self.n_reg       = NUM_REGISTER_TOKENS
        self.patch_start = 1 + self.n_reg   # = 5 for DINOv3
        print(f"[DINOv3] Loading {model_id} on {device} at {DINO_INPUT_SIZE}px ...")

        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            size={"height": DINO_INPUT_SIZE, "width": DINO_INPUT_SIZE},
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            output_hidden_states=True,
            output_attentions=True,
        ).to(device).eval()

        patch_size = getattr(self.model.config, "patch_size", 16)
        n_layers   = self.model.config.num_hidden_layers
        n_patches  = (DINO_INPUT_SIZE // patch_size) ** 2
        print(f"[DINOv3] Ready -- patch={patch_size}px, patches={n_patches}, "
              f"layers={n_layers}, cls_depth={MULTILAYER_DEPTH}, "
              f"registers={self.n_reg}, patch_start={self.patch_start}")

    @torch.inference_mode()
    def embed_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """Returns L2-normed (N, EMBED_DIM) tensor."""
        inputs  = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        ps = self.patch_start   # 5: skip [CLS, reg0, reg1, reg2, reg3]

        # ── Multi-layer CLS (color+texture in mid layers, semantics in final) ─
        all_hidden = outputs.hidden_states                   # tuple len = L+1
        layer_cls  = torch.stack(
            [h[:, 0, :] for h in all_hidden[-MULTILAYER_DEPTH:]],
            dim=0,
        ).mean(dim=0)                                        # (N, D)

        # ── Attention-weighted patch pooling (the "color mask") ────────────
        # attentions[-1]: (N, heads, seq_len, seq_len)
        # seq_len = 1 CLS + 4 registers + 784 patches = 789
        #
        # cls_attn[:,  :, 0, ps:] = CLS->patch weights, registers excluded.
        # Because registers absorb background, these weights are peaked over
        # the product face -- they form an implicit spatial color mask.
        last_attn    = outputs.attentions[-1]                # (N, H, S, S)
        cls_attn     = last_attn[:, :, 0, ps:]              # (N, H, P)  patch cols only
        attn_w       = cls_attn.mean(dim=1)                 # (N, P)  avg over heads
        attn_w       = attn_w / (attn_w.sum(dim=1, keepdim=True) + 1e-8)

        patch_tokens = outputs.last_hidden_state[:, ps:, :] # (N, P, D)  skip registers
        attn_pooled  = (attn_w.unsqueeze(-1) * patch_tokens).sum(1)  # (N, D)

        # ── Fuse: normalise each component independently ───────────────────
        cls_n  = F.normalize(layer_cls,   dim=-1)
        attn_n = F.normalize(attn_pooled, dim=-1)
        fused  = 0.5 * cls_n + 0.5 * attn_n
        return F.normalize(fused, dim=-1).cpu()

    def embed_bgr(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Single BGR image -> (1, D) CPU tensor."""
        pil = Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
        return self.embed_batch([pil])


# ─────────────────────────────────────────────────────────────
# Debug visualisation helper
# ─────────────────────────────────────────────────────────────

def draw_raw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    cls_ids: np.ndarray,
    names: Dict[int, str],
    min_area: float,
    max_ratio: float,
    min_ar: float,
    max_ar: float,
) -> np.ndarray:
    vis      = frame.copy()
    h, w     = frame.shape[:2]
    img_area = float(h * w)

    for i in range(len(boxes)):
        box      = boxes[i]
        conf     = float(confs[i])
        cls_name = names.get(int(cls_ids[i]), str(cls_ids[i]))
        x1, y1, x2, y2 = box.astype(int).tolist()
        area     = compute_box_area(box)
        ratio    = area / max(img_area, 1.0)
        ar       = box_aspect_ratio(box)

        color  = (0, 255, 255)
        reason = None
        if area < min_area:
            reason, color = "too_small",   (100, 100, 100)
        elif ratio > max_ratio:
            reason, color = "too_large",   (0, 0, 255)
        elif ar < min_ar or ar > max_ar:
            reason, color = f"bad_ar={ar:.2f}", (0, 165, 255)

        cv.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv.putText(vis, f"{i}:{cls_name} {conf:.2f}",
                   (x1, max(20, y1 - 24)), cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv.putText(vis, f"ar={ar:.2f} a={area:.0f}",
                   (x1, max(34, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        if reason:
            cv.putText(vis, reason,
                       (x1, min(h - 6, y2 + 14)), cv.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return vis


# ─────────────────────────────────────────────────────────────
# Main matcher class
# ─────────────────────────────────────────────────────────────

class AisleEmbeddingMatcher:

    def __init__(
        self,
        det_model_path:     str   = "yolov8s-worldv2.pt",
        prompts:            Optional[List[str]] = None,
        device:             str   = "auto",
        det_conf:           float = 0.02,
        det_iou:            float = 0.45,
        max_det:            int   = 150,
        min_box_area_px:    int   = 80,
        min_crop_dim_px:    int   = 40,   # min width AND height of crop in pixels
                                           # below this DINOv3 upscales too aggressively
        max_box_area_ratio: float = 0.18,
        # Aspect-ratio gate: keeps packs (landscape/portrait), rejects badges (square)
        min_aspect_ratio:   float = 0.30,   # narrowest portrait pack allowed
        max_aspect_ratio:   float = 4.00,   # widest landscape pack allowed
        crop_pad:           int   = 6,
        background_fill:    int   = 0,
        det_conf_weight:    float = 0.10,   # used in multiclass mode only
        center_bias_weight: float = 0.01,   # used in multiclass mode only
        resize_to:          Optional[Tuple[int, int]] = None,
        dino_model_id:      str   = DINO_MODEL_ID,
        hist_weight:        float = HIST_WEIGHT,   # thirds histogram contribution
        hist_bins:          int   = HIST_BINS,     # HSV bins per channel per third
        ref_preprocess:     bool  = True,          # normalize reference image backgrounds
        ref_bg_fill:        Tuple[int,int,int] = REF_BG_FILL,  # replacement bg color
        ref_alpha_threshold: int  = 0,             # 0=soft, 25-50=harder edge
        ref_erode_px:       int   = 0,             # shrink mask (0=off)
        ref_dilate_px:      int   = 3,             # expand mask — recovers lost edges
        ref_blur_mask_px:   int   = 3,             # feather edges (0=hard cutout)
        ref_rembg_model:    str   = REMBG_MODEL,   # background removal model
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"[INFO] Device: {self.device}")

        self.det_conf           = det_conf
        self.det_iou            = det_iou
        self.max_det            = max_det
        self.min_box_area_px    = min_box_area_px
        self.min_crop_dim_px    = min_crop_dim_px
        self.max_box_area_ratio = max_box_area_ratio
        self.min_aspect_ratio   = min_aspect_ratio
        self.max_aspect_ratio   = max_aspect_ratio
        self.crop_pad           = crop_pad
        self.background_fill    = background_fill
        self.det_conf_weight    = det_conf_weight
        self.center_bias_weight = center_bias_weight
        self.resize_to          = resize_to
        self.hist_weight        = hist_weight
        self.hist_bins          = hist_bins
        self.ref_preprocess      = ref_preprocess
        self.ref_bg_fill         = ref_bg_fill
        self.ref_alpha_threshold = ref_alpha_threshold
        self.ref_erode_px        = ref_erode_px
        self.ref_dilate_px       = ref_dilate_px
        self.ref_blur_mask_px    = ref_blur_mask_px
        self.ref_rembg_model     = ref_rembg_model
        # Pre-load rembg session once — model loaded here, reused per gallery image
        self._rembg_session = None
        if ref_preprocess and REMBG_AVAILABLE:
            print(f"[GALLERY] Loading rembg session: {ref_rembg_model} ...")
            self._rembg_session = rembg_session(ref_rembg_model)
            print("[GALLERY] rembg ready.")
        elif ref_preprocess and not REMBG_AVAILABLE:
            print("[WARN] ref_preprocess=True but rembg not installed.")
            print("       pip install rembg onnxruntime")

        # ── YOLOWorld ────────────────────────────────────────────────────
        self.det_model = YOLOWorld(det_model_path)
        self.prompts   = prompts or ["package", "product", "box", "container", "bottle", "can"]
        self.det_model.set_classes(self.prompts)
        print(f"[YOLO] {det_model_path}  prompts: {self.prompts}")

        # ── DINOv2 ───────────────────────────────────────────────────────
        self.embedder = DINOv2Embedder(model_id=dino_model_id, device=self.device)

        # Gallery state
        self.gallery_paths:      List[Path]             = []
        self.gallery_labels:     List[str]              = []
        self.gallery_embeddings: Optional[torch.Tensor] = None  # (G, D)
        self.gallery_hists:      Optional[np.ndarray]   = None  # (G, 3*bins*3)
        self.label_to_indices:   Dict[str, List[int]]   = {}

    # ── Gallery ───────────────────────────────────────────────

    def build_gallery(self, gallery_dir: Path) -> None:
        paths = list_image_files(gallery_dir)
        if not paths:
            raise FileNotFoundError(f"No gallery images in {gallery_dir}")

        embs, hists, labels = [], [], []
        label_to_idx: Dict[str, List[int]] = {}

        print(f"[GALLERY] Building from {len(paths)} image(s) …")
        for i, p in enumerate(paths, 1):
            img = cv.imread(str(p))
            if img is None:
                print(f"[GALLERY] WARN: cannot read {p}, skipping")
                continue
            label = infer_label(p)
            print(f"[GALLERY] [{i}/{len(paths)}] {p.name} → {label}")

            # Remove background using rembg (U2Net) and composite onto dark fill.
            # Closes the domain gap: Open Food Facts studio photos have white
            # backgrounds; shelf query crops have dark shelf backgrounds.
            # After removal the reference shows only the product face,
            # which matches what DINOv3 sees in shelf crops.
            if self.ref_preprocess:
                debug_path = str(p.parent / f"_debug_preproc_{p.name}")
                img = remove_reference_background(
                    img,
                    bg_fill=self.ref_bg_fill,
                    save_debug_path=debug_path,
                    _session=self._rembg_session,
                    alpha_threshold=self.ref_alpha_threshold,
                    erode_px=self.ref_erode_px,
                    dilate_px=self.ref_dilate_px,
                    blur_mask_px=self.ref_blur_mask_px,
                )
                print(f"[GALLERY]   bg removed → {debug_path}")

            emb  = self.embedder.embed_bgr(img)                        # (1, D) CPU
            hist = extract_thirds_histogram(img, self.hist_bins)       # (3*bins*3,)

            embs.append(emb)
            hists.append(hist)
            labels.append(label)
            label_to_idx.setdefault(label, []).append(len(labels) - 1)

        self.gallery_paths      = paths
        self.gallery_labels     = labels
        self.gallery_embeddings = torch.cat(embs, dim=0)               # (G, D)
        self.gallery_hists      = np.stack(hists)                      # (G, 3*bins*3)
        self.label_to_indices   = label_to_idx
        print(
            f"[GALLERY] Done — {len(labels)} entries, "
            f"{len(label_to_idx)} labels"
        )

    # ── Detection ─────────────────────────────────────────────

    def _detect(self, frame: np.ndarray):
        r = self.det_model.predict(
            source=frame, conf=self.det_conf, iou=self.det_iou,
            max_det=self.max_det, verbose=False,
        )
        if not r or r[0].boxes is None or len(r[0].boxes) == 0:
            return None
        b = r[0].boxes
        return (
            b.xyxy.detach().cpu().numpy(),
            b.conf.detach().cpu().numpy(),
            b.cls.detach().cpu().numpy().astype(int),
            r[0].names,
        )

    # ── Cropping + aspect-ratio gate ──────────────────────────

    def _crop(self, frame: np.ndarray, box: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w     = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int).tolist()
        x1 = max(0, x1 - self.crop_pad)
        y1 = max(0, y1 - self.crop_pad)
        x2 = min(w - 1, x2 + self.crop_pad)
        y2 = min(h - 1, y2 + self.crop_pad)

        if x2 <= x1 or y2 <= y1:
            return None

        # ── Minimum crop dimension gate ──────────────────────────────────────
        # Gum packs and small products can produce crops as small as 30-50px.
        # DINOv3 upscales these to 448px, creating interpolation artifacts
        # that look nothing like the actual product — embeddings become noisy.
        bw = x2 - x1
        bh = y2 - y1
        if bw < self.min_crop_dim_px or bh < self.min_crop_dim_px:
            return None

        # ── Aspect-ratio gate ────────────────────────────────────────────
        # Rejects near-square objects (badges, logos, price circles, coins)
        # that share color/shape with rectangular product packs.
        ar = bw / max(bh, 1.0)
        if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
            return None

        crop = frame[y1:y2 + 1, x1:x2 + 1].copy()
        if crop.size == 0:
            return None

        return {
            "crop_bgr":            crop,
            "isolated_square_bgr": pad_to_square(crop, self.background_fill),
            "crop_xyxy":           (x1, y1, x2, y2),
            "aspect_ratio":        ar,
        }

    # ── Per-crop similarity ────────────────────────────────────

    def _sims(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """
        Fused similarity: DINOv3 cosine + spatial thirds histogram.

        DINOv3 cosine:  captures identity, shape, texture, learned color structure.
        Thirds histogram: captures WHERE specific colors live spatially in the pack.

        The two signals are complementary:
          - DINOv3 alone can mistake Dentyne ICE for Orbit Spearmint (both green,
            similar shape, similar logo placement).
          - Thirds histogram alone can miss packs under different lighting.
          - Together: DINOv3 gates on product identity, thirds gates on color layout.

        final_score = (1 - hist_weight) * dino_cosine
                    +      hist_weight  * thirds_intersection
        """
        if self.gallery_embeddings is None or self.gallery_hists is None:
            raise RuntimeError("Gallery not built.")

        # DINOv3 cosine similarities — (G,)
        q_emb      = self.embedder.embed_bgr(crop_bgr)               # (1, D) L2-normed
        dino_sims  = (q_emb @ self.gallery_embeddings.T).squeeze(0)  # (G,)

        # Spatial thirds histogram similarities — (G,)
        q_hist     = extract_thirds_histogram(crop_bgr, self.hist_bins)
        hist_sims  = torch.tensor(
            np.array([thirds_histogram_similarity(q_hist, self.gallery_hists[i])
                      for i in range(len(self.gallery_hists))],
                     dtype=np.float32)
        )                                                             # (G,)

        return (1.0 - self.hist_weight) * dino_sims + self.hist_weight * hist_sims

    def _match_multiclass(self, crop: np.ndarray, top_k: int) -> List[Dict]:
        sims       = self._sims(crop)
        label_best: Dict[str, Tuple[float, int]] = {}
        for idx, sc in enumerate(sims.tolist()):
            lbl = self.gallery_labels[idx]
            if lbl not in label_best or sc > label_best[lbl][0]:
                label_best[lbl] = (sc, idx)
        ranked = sorted(label_best.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        return [{"label": l, "score": float(s), "path": str(self.gallery_paths[i]),
                 "gallery_index": i} for l, (s, i) in ranked]

    def _match_target(self, crop: np.ndarray, target_label: str, top_k: int) -> List[Dict]:
        if target_label not in self.label_to_indices:
            raise ValueError(f"Label '{target_label}' not in gallery: "
                             f"{sorted(self.label_to_indices)}")
        sims    = self._sims(crop)
        scores  = sorted(
            [(float(sims[i].item()), i) for i in self.label_to_indices[target_label]],
            reverse=True,
        )
        return [{"label": target_label, "score": s,
                 "path": str(self.gallery_paths[i]), "gallery_index": i}
                for s, i in scores[:top_k]]

    # ── Candidate ranking ─────────────────────────────────────

    def rank_candidates(
        self,
        frame:        np.ndarray,
        boxes:        np.ndarray,
        confs:        np.ndarray,
        cls_ids:      np.ndarray,
        names:        Dict[int, str],
        top_k:        int,
        mode:         str,
        target_label: Optional[str],
    ) -> List[Dict[str, Any]]:
        h, w       = frame.shape[:2]
        img_area   = float(h * w)
        img_center = (w / 2.0, h / 2.0)
        candidates = []
        print(f"[RAW DETS] {len(boxes)}")

        for i in range(len(boxes)):
            box      = boxes[i]
            conf     = float(confs[i])
            cls_name = names.get(int(cls_ids[i]), str(cls_ids[i]))
            area     = compute_box_area(box)
            ar       = box_aspect_ratio(box)
            ratio    = area / max(img_area, 1.0)

            if area < self.min_box_area_px:
                print(f"  skip {i}: too_small ({area:.0f}px)"); continue
            if ratio > self.max_box_area_ratio:
                print(f"  skip {i}: too_large (ratio={ratio:.4f})"); continue
            if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
                print(f"  skip {i}: bad_aspect ({ar:.2f})"); continue

            crop_info = self._crop(frame, box)
            if crop_info is None:
                print(f"  skip {i}: crop failed"); continue

            matches = (
                self._match_multiclass(crop_info["isolated_square_bgr"], top_k)
                if mode == "multiclass" else
                self._match_target(crop_info["isolated_square_bgr"], target_label, top_k)
            )
            if not matches:
                print(f"  skip {i}: no match"); continue

            top1 = matches[0]
            top2_score   = matches[1]["score"] if len(matches) > 1 else -1.0
            score_margin = top1["score"] - top2_score if top2_score >= 0 else top1["score"]

            # TARGET: rank purely by embedding similarity.
            # Removing det_conf_weight / center_bias_weight prevents a YOLO-confident
            # wrong product from outranking a correct off-center pack.
            # MULTICLASS: keep the weighted formula to find the single best answer.
            if mode == "target":
                final_score = top1["score"]
            else:
                cx, cy    = compute_box_center(box, w, h)
                dist      = ((cx - img_center[0])**2 + (cy - img_center[1])**2)**0.5
                norm_dist = dist / max((w**2 + h**2)**0.5, 1.0)
                final_score = (top1["score"]
                               + self.det_conf_weight * conf
                               - self.center_bias_weight * norm_dist)

            cx, cy = compute_box_center(box, w, h)
            print(f"  keep {i}: label={top1['label']} sim={top1['score']:.4f} "
                  f"ar={ar:.2f} margin={score_margin:.4f} final={final_score:.4f}")

            candidates.append({
                "det_idx":      i,
                "box":          box,
                "conf":         conf,
                "cls_name":     cls_name,
                "box_area":     area,
                "aspect_ratio": ar,
                "center_x":     cx,
                "center_y":     cy,
                "top1_label":   top1["label"],
                "top1_score":   top1["score"],
                "top2_score":   top2_score,
                "score_margin": score_margin,
                "topk":         matches,
                "final_score":  final_score,
                "crop_info":    crop_info,
            })

        candidates.sort(key=lambda c: c["final_score"], reverse=True)
        return candidates

    # ── Process single image ──────────────────────────────────

    def process_image(
        self,
        image_path:              Path,
        top_k:                   int   = 3,
        sim_threshold:           float = 0.62,
        margin_threshold:        float = 0.05,
        mode:                    str   = "multiclass",
        target_label:            Optional[str]  = None,
        save_annotated:          bool  = False,
        annotated_dir:           Optional[Path] = None,
        save_raw_debug:          bool  = False,
        raw_debug_dir:           Optional[Path] = None,
        save_crops:              bool  = False,
        crop_dir:                Optional[Path] = None,
        save_all_candidate_crops: bool = False,
        all_candidate_crop_dir:  Optional[Path] = None,
    ) -> Dict[str, Any]:

        frame = cv.imread(str(image_path))
        if frame is None:
            print(f"[ERR] Cannot read {image_path}")
            return {"image_name": image_path.name, "status": "read_fail"}

        orig_h, orig_w = frame.shape[:2]
        if self.resize_to:
            frame = cv.resize(frame, self.resize_to)
        proc_h, proc_w = frame.shape[:2]

        det = self._detect(frame)
        if det is None:
            print(f"[DET] {image_path.name}: no detections")
            return {"image_name": image_path.name,
                    "orig_width": orig_w, "orig_height": orig_h,
                    "proc_width": proc_w, "proc_height": proc_h,
                    "status": "no_detection"}

        boxes, confs, cls_ids, names = det

        if save_raw_debug and raw_debug_dir:
            raw_vis = draw_raw_detections(
                frame, boxes, confs, cls_ids, names,
                self.min_box_area_px, self.max_box_area_ratio,
                self.min_aspect_ratio, self.max_aspect_ratio,
            )
            cv.imwrite(str(raw_debug_dir / image_path.name), raw_vis)

        candidates = self.rank_candidates(
            frame, boxes, confs, cls_ids, names,
            top_k=top_k, mode=mode, target_label=target_label,
        )

        if not candidates:
            print(f"[DET] {image_path.name}: no valid candidates")
            return {"image_name": image_path.name,
                    "orig_width": orig_w, "orig_height": orig_h,
                    "proc_width": proc_w, "proc_height": proc_h,
                    "status": "no_valid_candidate"}

        if save_all_candidate_crops and all_candidate_crop_dir:
            for idx, c in enumerate(candidates):
                cv.imwrite(
                    str(all_candidate_crop_dir /
                        f"{image_path.stem}_cand{idx:03d}_{c['top1_label']}.png"),
                    c["crop_info"]["isolated_square_bgr"],
                )

        best = candidates[0]

        # ── TARGET: accept ALL boxes above threshold ───────────────────
        # Draw every matching instance, not just one winner.
        if mode == "target":
            accepted_cands = [c for c in candidates if c["top1_score"] >= sim_threshold]
            accepted       = bool(accepted_cands)
            accepted_label = target_label if accepted else "unknown"
            accepted_score = accepted_cands[0]["top1_score"] if accepted else ""
            accepted_count = len(accepted_cands)
            if accepted:
                for ac in accepted_cands:
                    print(f"[HIT]  {image_path.name} box={ac['box'].astype(int).tolist()} "
                          f"sim={ac['top1_score']:.4f} ar={ac['aspect_ratio']:.2f}")
            else:
                print(f"[MISS] {image_path.name}: nothing reached threshold={sim_threshold}")

        # ── MULTICLASS: single best winner ─────────────────────────────
        else:
            accepted = (best["top1_score"] >= sim_threshold
                        and best["score_margin"] >= margin_threshold)
            accepted_label = best["top1_label"] if accepted else "unknown"
            accepted_score = best["top1_score"] if accepted else ""
            accepted_count = 1 if accepted else 0
            accepted_cands = [best] if accepted else []
            print(f"[MATCH] {image_path.name}: "
                  f"top1={best['top1_label']} sim={best['top1_score']:.4f} "
                  f"margin={best['score_margin']:.4f} accepted={accepted_label}")

        # ── Save best crop ──────────────────────────────────────────────
        if save_crops and crop_dir and accepted_cands:
            cv.imwrite(str(crop_dir / f"{image_path.stem}_best.png"),
                       accepted_cands[0]["crop_info"]["isolated_square_bgr"])

        # ── Annotated output ────────────────────────────────────────────
        if save_annotated and annotated_dir:
            vis = frame.copy()

            # All candidates in dim yellow
            for c in candidates:
                x1, y1, x2, y2 = map(int, c["box"])
                cv.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 200), 1)

            if mode == "target":
                # Every accepted box — bright green with sim score
                for ac in accepted_cands:
                    x1, y1, x2, y2 = map(int, ac["box"])
                    cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(vis,
                               f"{ac['top1_label']} {ac['top1_score']:.2f}",
                               (x1, max(16, y1 - 5)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                header = (f"found: {accepted_count}x {accepted_label}"
                          if accepted else f"not found: {target_label}")
                hdr_color = (0, 255, 0) if accepted else (0, 0, 255)
                cv.putText(vis, header, (10, 28),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, hdr_color, 2)
            else:
                if accepted_cands:
                    x1, y1, x2, y2 = map(int, best["box"])
                    cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(vis, f"{best['top1_label']} {best['top1_score']:.3f}",
                               (x1, max(20, y1 - 8)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.putText(vis, f"accepted: {accepted_label}", (10, 28),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv.putText(vis, f"mode={mode} cands={len(candidates)}",
                       (10, proc_h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv.imwrite(str(annotated_dir / image_path.name), vis)

        return {
            "image_name":      image_path.name,
            "orig_width":      orig_w,  "orig_height":  orig_h,
            "proc_width":      proc_w,  "proc_height":  proc_h,
            "num_candidates":  len(candidates),
            "num_accepted":    accepted_count,
            "det_cls":         best["cls_name"],
            "det_conf":        best["conf"],
            "det_box_x1":      int(best["box"][0]),
            "det_box_y1":      int(best["box"][1]),
            "det_box_x2":      int(best["box"][2]),
            "det_box_y2":      int(best["box"][3]),
            "det_center_x":    int(best["center_x"]),
            "det_center_y":    int(best["center_y"]),
            "det_box_area_px": float(best["box_area"]),
            "top1_label":      best["top1_label"],
            "top1_score":      best["top1_score"],
            "top2_score":      best["top2_score"],
            "score_margin":    best["score_margin"],
            "match_label":     accepted_label,
            "match_score":     accepted_score,
            "final_score":     best["final_score"],
            "topk_json":       json.dumps(best["topk"]),
            "status":          "ok" if accepted else "unknown",
        }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="YOLOWorld + DINOv2-small retail shelf matcher"
    )

    # I/O
    ap.add_argument("--images_dir",  required=True)
    ap.add_argument("--gallery_dir", required=True)
    ap.add_argument("--out_csv",     default="aisle_dinov2_matches.csv")

    # Detection
    ap.add_argument("--det_model",           default="yolov8s-worldv2.pt")
    ap.add_argument("--prompts",             default="package,product,box,container,bottle,can")
    ap.add_argument("--mode",                default="multiclass", choices=["multiclass","target"])
    ap.add_argument("--target_label",        default="")
    ap.add_argument("--device",              default="auto", choices=["auto","cpu","mps","cuda"])
    ap.add_argument("--det_conf",            type=float, default=0.02)
    ap.add_argument("--det_iou",             type=float, default=0.45)
    ap.add_argument("--max_det",             type=int,   default=150)
    ap.add_argument("--min_box_area_px",     type=int,   default=80)
    ap.add_argument("--min_crop_dim_px",     type=int,   default=40,
                    help="Min crop width AND height in px — below this DINOv3 upscales too much")
    ap.add_argument("--max_box_area_ratio",  type=float, default=0.18)
    ap.add_argument("--min_aspect_ratio",    type=float, default=0.30,
                    help="Min box width/height ratio (rejects near-square blobs like badges)")
    ap.add_argument("--max_aspect_ratio",    type=float, default=4.00,
                    help="Max box width/height ratio (rejects very wide strips)")
    ap.add_argument("--crop_pad",            type=int,   default=6)
    ap.add_argument("--background_fill",     type=int,   default=0)
    ap.add_argument("--det_conf_weight",     type=float, default=0.10)
    ap.add_argument("--center_bias_weight",  type=float, default=0.01)
    ap.add_argument("--top_k",               type=int,   default=3)
    ap.add_argument("--sim_threshold",       type=float, default=0.62)
    ap.add_argument("--margin_threshold",    type=float, default=0.05)

    # DINOv3
    ap.add_argument("--dino_model",   default=DINO_MODEL_ID)
    ap.add_argument("--hist_weight",      type=float, default=HIST_WEIGHT,
                    help="Thirds histogram contribution to similarity (0=pure DINOv3)")
    ap.add_argument("--hist_bins",        type=int,   default=HIST_BINS,
                    help="HSV bins per channel per third")
    ap.add_argument("--no_ref_preprocess",   action="store_true",
                    help="Disable reference image background normalization")
    ap.add_argument("--ref_bg_fill",         type=int, nargs=3,
                    default=list(REF_BG_FILL), metavar=("B","G","R"),
                    help="BGR background fill color for reference preprocessing")
    ap.add_argument("--ref_alpha_threshold", type=int, default=0,
                    help="Alpha threshold 0-255 (0=soft, 25-50=harder edge)")
    ap.add_argument("--ref_erode_px",        type=int, default=0,
                    help="Erode mask px — more = more aggressive bg removal")
    ap.add_argument("--ref_dilate_px",       type=int, default=3,
                    help="Dilate mask px — more = less aggressive, recovers edges")
    ap.add_argument("--ref_blur_mask_px",    type=int, default=3,
                    help="Blur mask px for soft edges (0=hard cutout, odd number)")
    ap.add_argument("--ref_rembg_model",     type=str, default=REMBG_MODEL,
                    help="rembg model: birefnet-general (best), birefnet-general-lite, "
                         "isnet-general-use, u2net")

    # Debug / output
    ap.add_argument("--save_raw_debug",           action="store_true")
    ap.add_argument("--raw_debug_dir",            default="aisle_raw_debug")
    ap.add_argument("--save_annotated",           action="store_true")
    ap.add_argument("--annotated_dir",            default="aisle_dinov2_debug")
    ap.add_argument("--save_crops",               action="store_true")
    ap.add_argument("--crop_dir",                 default="aisle_best_crops")
    ap.add_argument("--save_all_candidate_crops", action="store_true")
    ap.add_argument("--all_candidate_crop_dir",   default="aisle_all_candidate_crops")
    ap.add_argument("--resize_w", type=int, default=0)
    ap.add_argument("--resize_h", type=int, default=0)

    args = ap.parse_args()

    if args.mode == "target" and not args.target_label.strip():
        raise ValueError("--target_label required with --mode target")

    images_dir  = Path(args.images_dir)
    gallery_dir = Path(args.gallery_dir)
    if not images_dir.exists():  raise FileNotFoundError(images_dir)
    if not gallery_dir.exists(): raise FileNotFoundError(gallery_dir)

    resize_to      = (args.resize_w, args.resize_h) if args.resize_w > 0 else None
    raw_debug_dir  = Path(args.raw_debug_dir)          if args.save_raw_debug          else None
    annotated_dir  = Path(args.annotated_dir)          if args.save_annotated          else None
    crop_dir       = Path(args.crop_dir)               if args.save_crops              else None
    all_cand_dir   = Path(args.all_candidate_crop_dir) if args.save_all_candidate_crops else None

    for d in [raw_debug_dir, annotated_dir, crop_dir, all_cand_dir]:
        ensure_dir(d)

    image_paths = list_image_files(images_dir)
    if not image_paths:
        print(f"[WARN] No images in {images_dir}"); return

    print(f"[INFO] {len(image_paths)} query image(s)")

    matcher = AisleEmbeddingMatcher(
        det_model_path     = args.det_model,
        prompts            = parse_prompts(args.prompts),
        device             = args.device,
        det_conf           = args.det_conf,
        det_iou            = args.det_iou,
        max_det            = args.max_det,
        min_box_area_px    = args.min_box_area_px,
        min_crop_dim_px    = args.min_crop_dim_px,
        max_box_area_ratio = args.max_box_area_ratio,
        min_aspect_ratio   = args.min_aspect_ratio,
        max_aspect_ratio   = args.max_aspect_ratio,
        crop_pad           = args.crop_pad,
        background_fill    = args.background_fill,
        det_conf_weight    = args.det_conf_weight,
        center_bias_weight = args.center_bias_weight,
        resize_to          = resize_to,
        dino_model_id      = args.dino_model,
        hist_weight        = args.hist_weight,
        hist_bins          = args.hist_bins,
        ref_preprocess      = not args.no_ref_preprocess,
        ref_bg_fill         = tuple(args.ref_bg_fill),
        ref_alpha_threshold = args.ref_alpha_threshold,
        ref_erode_px        = args.ref_erode_px,
        ref_dilate_px       = args.ref_dilate_px,
        ref_blur_mask_px    = args.ref_blur_mask_px,
        ref_rembg_model     = args.ref_rembg_model,
    )
    matcher.build_gallery(gallery_dir)

    rows = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")
        row = matcher.process_image(
            image_path               = img_path,
            top_k                    = args.top_k,
            sim_threshold            = args.sim_threshold,
            margin_threshold         = args.margin_threshold,
            mode                     = args.mode,
            target_label             = args.target_label.strip() or None,
            save_annotated           = args.save_annotated,
            annotated_dir            = annotated_dir,
            save_raw_debug           = args.save_raw_debug,
            raw_debug_dir            = raw_debug_dir,
            save_crops               = args.save_crops,
            crop_dir                 = crop_dir,
            save_all_candidate_crops = args.save_all_candidate_crops,
            all_candidate_crop_dir   = all_cand_dir,
        )
        rows.append(row)

    fieldnames = [
        "image_name", "orig_width", "orig_height", "proc_width", "proc_height",
        "num_candidates", "num_accepted",
        "det_cls", "det_conf",
        "det_box_x1", "det_box_y1", "det_box_x2", "det_box_y2",
        "det_center_x", "det_center_y", "det_box_area_px",
        "top1_label", "top1_score", "top2_score", "score_margin",
        "match_label", "match_score", "final_score",
        "topk_json", "status",
    ]
    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    ok    = sum(1 for r in rows if r.get("status") == "ok")
    unk   = sum(1 for r in rows if r.get("status") == "unknown")
    nodet = sum(1 for r in rows if r.get("status") == "no_detection")
    nocan = sum(1 for r in rows if r.get("status") == "no_valid_candidate")

    print("\n========== SUMMARY ==========")
    print(f"Total             : {len(rows)}")
    print(f"Accepted          : {ok}/{len(rows)}")
    print(f"Unknown           : {unk}/{len(rows)}")
    print(f"No detections     : {nodet}/{len(rows)}")
    print(f"No valid candidate: {nocan}/{len(rows)}")
    print(f"CSV               : {out_csv.resolve()}")


if __name__ == "__main__":
    main()