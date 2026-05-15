# Sixth Sense

Assistive computer vision system for visually impaired users to independently navigate and shop in grocery stores.

---

## Overview

Grocery stores assume you can see. For a visually impaired shopper there is no reliable way to identify a specific product on a shelf, distinguish variants like low-sodium or lactose-free, or locate an item without asking someone for help. Sixth Sense is built to close that gap.

The system lets a user point at a shelf and receive immediate audio feedback about what they are pointing at and how to reach the correct item. A user can state what they want to cook, have the system generate a full ingredient list, verify each item against a product catalog, and get walked through an entire shopping run from start to finish.

---

## Repository Structure

```
Sixth-Sense-VIP/
├── app.py              # Full end-to-end workflow with gesture detection
├── embed_dino.py       # DINOv3 embedding pipeline
├── match_item.py       # OpenFoodFacts product matcher and filtering pipeline
└── README.md
```

### `embed_dino.py`
The core embedding model. Takes an image crop from YOLOWorld and produces a high-dimensional feature vector using DINOv3. The pipeline runs at 448x448 resolution, fuses CLS tokens across the last several transformer layers to capture both color/texture and semantic information, and uses attention-weighted patch pooling to concentrate the embedding on the product itself and suppress background noise. Reference images from OpenFoodFacts are preprocessed with rembg to remove studio backgrounds before embedding, closing the domain gap with real shelf crops. Final embeddings are L2-normalized and stored as a reference gallery.

### `match_item.py`
The product identification and verification pipeline. Takes a user query (brand, product name, quantity) and filters against the OpenFoodFacts database sequentially — brand filtering, product name filtering, then quantity filtering. If multiple candidates remain, the top three are surfaced for user confirmation. Once confirmed, a reference image is pulled from OpenFoodFacts and passed to `embed_dino.py` to build the gallery entry. At inference, the live shelf crop embedding is compared against the gallery using cosine similarity combined with an HSV color histogram score as a weighted fraction of the final match result.

### `app.py`
The full system workflow. Integrates MediaPipe gesture detection to track the user's pointing direction, YOLOWorld for real-time object detection and bounding box cropping, the embedding and matching pipeline from `embed_dino.py` and `match_item.py`, and Kitten TTS for verbal directional audio guidance. This is the entry point for running the system end to end.

---

## How It Works

**Phase 1 — Pre-shopping**

The user states a meal request via voice. An LLM generates a structured ingredient list. Each ingredient is passed through the filtering pipeline in `match_item.py` against OpenFoodFacts. Once a product is confirmed, a reference image is pulled, preprocessed with rembg, embedded via DINOv3, and stored in the reference gallery.

**Phase 2 — In-store detection**

The user runs `app.py` and points at shelf items. YOLOWorld crops the pointed-at item, `embed_dino.py` embeds the crop, and `match_item.py` compares it against the reference gallery. Kitten TTS delivers directional cues — move left, move right, move up, move down — until the user reaches the correct product. This repeats for each ingredient.

---

## Model Details

**Embedding — DINOv3**

| Parameter | Value |
|---|---|
| Input resolution | 448x448 |
| CLS fusion | Last N transformer layers averaged |
| Patch pooling | Attention-weighted, register-token-aware |
| Final output | L2-normalized embedding vector |

DINOv3's register tokens absorb background attention, keeping embeddings focused on the product without needing explicit color features at the embedding stage.

**Matching**

Final match score is a weighted combination of DINOv3 cosine similarity and HSV color histogram similarity. The histogram acts as a secondary discriminator for visually similar product variants.

**Model Evolution**

| Model | Why we moved on |
|---|---|
| CLIP | Poor fine-grained discrimination between similar product variants |
| DINOv2 | Better spatial embeddings; added HSV fusion to compensate for color discrimination |
| DINOv3 | Register tokens handle background natively; HSV retained only as match score fraction |

**Audio Guidance**

Started with Mimic TTS — clean pronunciation but too heavy for real-time use. Switched to Kitten TTS (under 25MB). Verbal directional cues replaced the original beep-based system which was not intuitive enough for navigation.

---

## Results (Filtering Pipeline)

| Metric | Result |
|---|---|
| Overall accuracy | 103 / 120 correct matches (85.8%) |
| Beverages | 90% |
| Frozen | 80% |
| Produce | 60% |
| All other categories | 95% |
| Exact brand match | 85% (90 / 106 branded queries) |

---

## Setup

```bash
git clone https://github.com/kshah0301/Sixth-Sense-VIP
cd Sixth-Sense-VIP
git checkout embedding
pip install -r requirements.txt
```

**Build the reference gallery**

```bash
python match_item.py --brand "Orbit" --product "Spearmint Gum" --quantity "14 pieces"
```

This filters against OpenFoodFacts, confirms a candidate, pulls the reference image, removes the background, embeds it with DINOv3, and saves it to the gallery.

**Run the full system**

```bash
python app.py
```

---

## Dependencies

```
mediapipe
ultralytics          # YOLOWorld
torch
transformers         # DINOv3
rembg                # U2Net background removal
opencv-python
numpy
Pillow
scipy
requests             # OpenFoodFacts API
```

---

## What Works

- DINOv3 embedding pipeline with attention-weighted pooling and background removal
- OpenFoodFacts filtering pipeline at 85.8% accuracy across 120 test queries
- Real-time gesture detection and object detection via MediaPipe + YOLOWorld
- Verbal directional guidance via Kitten TTS

## What Still Needs Work

- Full end-to-end integration of image similarity matching with the filtering pipeline
- Validation of gallery matching on real shelf photos at scale
- LLM recipe generation and voice input pipeline
- FPS still choppy — guidance refresh rate needs smoothing
- English only across TTS and filtering

---

## Next Steps

- Complete end-to-end integration: voice input → LLM → filtering → gallery match → in-store guidance
- Build and validate a larger OpenFoodFacts embedding catalog against real shelf photos
- Improve FPS and guidance responsiveness
- Explore multilingual TTS support

---

## Contributors

Karan Shah, Evelynn Mak — NYU Tandon School of Engineering
