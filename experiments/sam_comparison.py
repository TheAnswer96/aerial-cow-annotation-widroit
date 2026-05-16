import os
import json
import csv
import re
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

# =============================================
# GLOBAL CONFIGURATION VARIABLES
# Edit these if your folder names differ.
# =============================================
COCO_ANNOTATIONS_DIR = os.path.join("1k", "annotations")
SAM_DIR = "sam"                          # folder containing the YOLO .txt segmentation files
IMAGES_DIR = os.path.join("1k", "images")  # only needed if COCO is missing width/height (rare)
RESULTS_DIR = "sam_comparison"

# =============================================
# HELPER: Extract matching key from both JSON and YOLO filenames
# =============================================
def extract_match_key(filename: str) -> str | None:
    """
    Extracts the canonical key <date>_<number>_<drone> from both naming conventions.
    JSON example:  20251020_3837_drone6_jpg.rf.ac8333ab7ca71aeac6893027e7bc1c26.jpg
    YOLO example:  2025-10-08_20251008_3833_drone1_58036301.txt
    """
    # Remove extension and any .rf. suffix
    name = os.path.splitext(filename)[0]
    # Look for the pattern: 8digits _ digits _ drone digits
    match = re.search(r'(\d{8})_(\d+)_drone(\d+)', name)
    if match:
        return f"{match.group(1)}_{match.group(2)}_drone{match.group(3)}"
    return None


# =============================================
# RLE decoder (copied from previous module – supports original COCO)
# =============================================
def _decode_rle(rle_dict: dict, height: int, width: int) -> np.ndarray:
    counts = rle_dict.get("counts")
    if not isinstance(counts, list):
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    val = 0
    for count in counts:
        if pos + count > len(mask):
            break
        mask[pos:pos + count] = val
        pos += count
        val = 1 - val
    return mask.reshape((height, width))


# =============================================
# Create binary mask (object = 1, background = 0)
# =============================================
def create_coco_mask(coco_annotations_for_image: list, height: int, width: int) -> np.ndarray:
    """Build ground-truth mask from COCO annotations (polygons + RLE supported)."""
    mask_pil = Image.new("L", (width, height), color=255)  # white = bg
    draw = ImageDraw.Draw(mask_pil)

    for ann in coco_annotations_for_image:
        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg:  # polygon(s)
            for polygon in seg:
                if len(polygon) < 6 or len(polygon) % 2 != 0:
                    continue
                points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                draw.polygon(points, fill=0)  # black = object
        elif isinstance(seg, dict) and "counts" in seg:  # RLE
            rle_mask = _decode_rle(seg, height, width)
            rle_pil = Image.fromarray((rle_mask * 255).astype(np.uint8))
            mask_pil.paste(rle_pil, (0, 0), mask=rle_pil)

    mask_np = np.array(mask_pil) == 0          # True where object
    return mask_np.astype(np.uint8)            # 1 = object, 0 = bg


def create_yolo_mask(txt_path: str, width: int, height: int) -> np.ndarray:
    """Build SAM prediction mask from YOLO segmentation .txt (polygon)."""
    mask_pil = Image.new("L", (width, height), color=255)  # white = bg
    draw = ImageDraw.Draw(mask_pil)

    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 3:
                    continue
                # parts[0] = class (should be 0), rest = normalized x y x y ...
                points = []
                for i in range(1, len(parts), 2):
                    x = int(parts[i] * width)
                    y = int(parts[i + 1] * height)
                    points.append((x, y))
                if len(points) >= 3:
                    draw.polygon(points, fill=0)  # black = object
    except Exception:
        pass  # if txt is empty or malformed, return empty mask

    mask_np = np.array(mask_pil) == 0
    return mask_np.astype(np.uint8)


# =============================================
# Metrics (pixel-wise)
# =============================================
def compute_segmentation_metrics(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> dict:
    """gt and pred are binary (1=object, 0=bg)"""
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# =============================================
# MAIN MODULE
# =============================================
def run_sam_comparison() -> None:
    """
    Matches the 1K COCO JSON annotations with SAM2 YOLO masks,
    computes DICE, IoU, Precision, Recall, F1 for each image,
    and saves everything to sam_comparison/sam_metrics.csv
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Find the single COCO JSON in 1k/annotations/
    json_files = [f for f in os.listdir(COCO_ANNOTATIONS_DIR) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No COCO JSON found in {COCO_ANNOTATIONS_DIR}")
    coco_path = os.path.join(COCO_ANNOTATIONS_DIR, json_files[0])
    print(f"Loading COCO annotations: {coco_path}")

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup: match_key → (image_info, list_of_annotations)
    coco_by_key = {}
    for img in coco["images"]:
        key = extract_match_key(img["file_name"])
        if key:
            img_id = img["id"]
            anns = [a for a in coco.get("annotations", []) if a["image_id"] == img_id]
            coco_by_key[key] = (img, anns)

    print(f"Found {len(coco_by_key)} images with valid match keys in the 1K COCO JSON.")

    # 2. Build SAM YOLO lookup: match_key → txt_path
    sam_by_key = {}
    for fname in os.listdir(SAM_DIR):
        if not fname.lower().endswith(".txt"):
            continue
        key = extract_match_key(fname)
        if key:
            sam_by_key[key] = os.path.join(SAM_DIR, fname)

    print(f"Found {len(sam_by_key)} SAM YOLO annotation files.")

    # 3. Compute metrics for matched pairs
    results = []
    matched_count = 0

    for key, (img_info, coco_anns) in coco_by_key.items():
        if key not in sam_by_key:
            continue  # no SAM annotation for this image

        txt_path = sam_by_key[key]
        matched_count += 1

        width = img_info["width"]
        height = img_info["height"]
        filename = img_info["file_name"]   # original JSON filename for reporting

        # Ground truth mask (COCO)
        gt_mask = create_coco_mask(coco_anns, height, width)

        # SAM prediction mask (YOLO)
        pred_mask = create_yolo_mask(txt_path, width, height)

        # Metrics
        metrics = compute_segmentation_metrics(gt_mask, pred_mask)

        results.append({
            "filename": filename,
            "match_key": key,
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

        if matched_count % 100 == 0:
            print(f"  Processed {matched_count} matched images...")

    # 4. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "sam_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "match_key", "iou", "dice",
                                               "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(results)

    # 5. Summary
    if results:
        avg = {k: np.mean([r[k] for r in results]) for k in ["iou", "dice", "precision", "recall", "f1"]}
        std = {k: np.std([r[k] for r in results]) for k in ["iou", "dice", "precision", "recall", "f1"]}

        print("\n" + "="*90)
        print("SAM2 vs COCO COMPARISON FINISHED!")
        print(f"Matched {matched_count} images out of {len(coco_by_key)} in the 1K subset")
        print(f"Results saved → {csv_path}")
        print("\nAverage metrics (± std):")
        print(f"  IoU       : {avg['iou']:.4f} ± {std['iou']:.4f}")
        print(f"  Dice      : {avg['dice']:.4f} ± {std['dice']:.4f}")
        print(f"  Precision : {avg['precision']:.4f} ± {std['precision']:.4f}")
        print(f"  Recall    : {avg['recall']:.4f} ± {std['recall']:.4f}")
        print(f"  F1-score  : {avg['f1']:.4f} ± {std['f1']:.4f}")
        print("="*90)
    else:
        print("No matches found. Check the folder paths or filename patterns.")


# =============================================
# How to use this module
# =============================================
# In your main.py:
#    from sam_comparison import run_sam_comparison
#    run_sam_comparison()
#
# The module will create the folder "sam_comparison/" and the file sam_metrics.csv