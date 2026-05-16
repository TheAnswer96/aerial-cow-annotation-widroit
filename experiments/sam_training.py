import os
import json
import csv
import re
import shutil
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

# =============================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================
COCO_ANNOTATIONS_DIR = os.path.join("1k", "annotations")
SAM_DIR = "sam"                          # YOLO .txt files from SAM2
IMAGES_DIR = os.path.join("1k", "images")
RESULTS_DIR = "sam_tp"                   # new folder with TP-only dataset

TP_IOU_THRESHOLD = 0.75                  # SAM prediction is considered "true positive" if IoU >= this

# =============================================
# Matching helper (same as previous sam_comparison.py)
# =============================================
def extract_match_key(filename: str) -> str | None:
    name = os.path.splitext(filename)[0]
    match = re.search(r'(\d{8})_(\d+)_drone(\d+)', name)
    if match:
        return f"{match.group(1)}_{match.group(2)}_drone{match.group(3)}"
    return None


# =============================================
# RLE decoder (for original COCO)
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
# Build binary masks
# =============================================
def create_coco_mask(coco_anns: list, height: int, width: int) -> np.ndarray:
    mask_pil = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(mask_pil)
    for ann in coco_anns:
        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg:
            for polygon in seg:
                if len(polygon) < 6 or len(polygon) % 2 != 0:
                    continue
                points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                draw.polygon(points, fill=0)
        elif isinstance(seg, dict) and "counts" in seg:
            rle_mask = _decode_rle(seg, height, width)
            rle_pil = Image.fromarray((rle_mask * 255).astype(np.uint8))
            mask_pil.paste(rle_pil, (0, 0), mask=rle_pil)
    return np.array(mask_pil) == 0   # 1 = object


def yolo_to_polygon(txt_path: str, width: int, height: int) -> list[float]:
    """Convert SAM YOLO normalized polygon to absolute COCO-style flat list."""
    polygon = []
    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:  # at least class + 2 points
                    continue
                for i in range(1, len(parts), 2):
                    x = parts[i] * width
                    y = parts[i + 1] * height
                    polygon.extend([x, y])
    except Exception:
        pass
    return polygon


def create_sam_mask_and_polygon(txt_path: str, width: int, height: int):
    """Returns both binary mask (for PNG) and polygon list (for COCO JSON)."""
    polygon = yolo_to_polygon(txt_path, width, height)
    if not polygon:
        return np.zeros((height, width), dtype=np.uint8), []

    mask_pil = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(mask_pil)
    points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
    if len(points) >= 3:
        draw.polygon(points, fill=0)
    mask_np = np.array(mask_pil) == 0
    return mask_np.astype(np.uint8), polygon


# =============================================
# Metrics (same as before)
# =============================================
def compute_segmentation_metrics(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> dict:
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    return {"iou": float(iou), "dice": float(dice), "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}


# =============================================
# MAIN FUNCTIONS
# =============================================
def generate_sam_tp_dataset(iou_threshold: float = TP_IOU_THRESHOLD) -> None:
    """
    1. Matches SAM2 YOLO annotations with the 1K COCO JSON.
    2. Identifies True Positives (SAM IoU >= threshold).
    3. Creates:
       - sam_tp/images/      (symlinks to original images - saves disk)
       - sam_tp/masks/       (PNG masks generated from SAM polygons)
       - sam_tp/annotations/sam_tp_coco.json   (COCO format using SAM polygons)
       - sam_tp/non_tp_images.txt
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "annotations"), exist_ok=True)

    # Load original COCO
    json_files = [f for f in os.listdir(COCO_ANNOTATIONS_DIR) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No COCO JSON in {COCO_ANNOTATIONS_DIR}")
    coco_path = os.path.join(COCO_ANNOTATIONS_DIR, json_files[0])
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    # COCO lookup: match_key → (image_info, annotations)
    coco_by_key = {}
    for img in coco["images"]:
        key = extract_match_key(img["file_name"])
        if key:
            img_id = img["id"]
            anns = [a for a in coco.get("annotations", []) if a["image_id"] == img_id]
            coco_by_key[key] = (img, anns)


    # SAM YOLO lookup
    sam_by_key = {}
    for fname in os.listdir(SAM_DIR):
        if fname.lower().endswith(".txt"):
            reduced_name = fname.split("_")
            reduced_key = reduced_name[1] + "_" + reduced_name[2] + "_" + reduced_name[3]
            key = extract_match_key(reduced_key)
            if key:
                sam_by_key[key] = os.path.join(SAM_DIR, fname)

    print(f"Found {len(coco_by_key)} images in COCO | {len(sam_by_key)} SAM YOLO files")

    tp_images = []
    tp_annotations = []
    non_tp_filenames = []
    ann_id_counter = 1
    img_id_counter = 1

    matched_tp = 0
    for key, (img_info, coco_anns) in coco_by_key.items():
        if key not in sam_by_key:
            non_tp_filenames.append(img_info["file_name"])
            continue

        txt_path = sam_by_key[key]
        width, height = img_info["width"], img_info["height"]
        filename = img_info["file_name"]

        gt_mask = create_coco_mask(coco_anns, height, width)
        sam_mask, sam_polygon = create_sam_mask_and_polygon(txt_path, width, height)

        if len(sam_polygon) == 0:
            non_tp_filenames.append(filename)
            continue

        metrics = compute_segmentation_metrics(gt_mask, sam_mask)

        if metrics["iou"] >= iou_threshold:
            # === TRUE POSITIVE ===
            matched_tp += 1

            # 1. Create symlink / copy for image
            src_img = os.path.join(IMAGES_DIR, filename)
            dst_img = os.path.join(RESULTS_DIR, "images", filename)
            if not os.path.exists(dst_img):
                try:
                    os.symlink(os.path.abspath(src_img), dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)

            # 2. Save SAM mask as PNG (same name as image)
            mask_path = os.path.join(RESULTS_DIR, "masks", os.path.splitext(filename)[0] + ".png")
            Image.fromarray((sam_mask * 255).astype(np.uint8)).save(mask_path)

            # 3. COCO image entry
            new_img = img_info.copy()
            new_img["id"] = img_id_counter
            tp_images.append(new_img)

            # 4. COCO annotation entry (using SAM polygon)
            if sam_polygon:
                xs = sam_polygon[0::2]
                ys = sam_polygon[1::2]
                minx, miny = min(xs), min(ys)
                w = max(xs) - minx
                h = max(ys) - miny
                area = int(np.sum(sam_mask))  # exact pixel area

                new_ann = {
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": 1,   # assume single class "cow"
                    "segmentation": [sam_polygon],
                    "area": area,
                    "bbox": [minx, miny, w, h],
                    "iscrowd": 0
                }
                tp_annotations.append(new_ann)

            ann_id_counter += 1
            img_id_counter += 1
        else:
            non_tp_filenames.append(filename)

    # Build final COCO JSON
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": tp_images,
        "annotations": tp_annotations,
        "categories": [{"id": 1, "name": "cow", "supercategory": "animal"}]
    }

    json_path = os.path.join(RESULTS_DIR, "annotations", "sam_tp_coco.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)

    # Save non-TP list
    with open(os.path.join(RESULTS_DIR, "non_tp_images.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(non_tp_filenames))

    print("\n" + "="*90)
    print("SAM TRUE-POSITIVE DATASET GENERATED!")
    print(f"TP images selected (IoU >= {iou_threshold}): {matched_tp}")
    print(f"Non-TP images (for evaluation): {len(non_tp_filenames)}")
    print(f"→ Images   : {os.path.join(RESULTS_DIR, 'images')}/")
    print(f"→ Masks    : {os.path.join(RESULTS_DIR, 'masks')}/")
    print(f"→ JSON     : {json_path}")
    print(f"→ Non-TP list: {os.path.join(RESULTS_DIR, 'non_tp_images.txt')}")
    print("="*90)


def retrain_previous_models_on_tp(iou_threshold: float = TP_IOU_THRESHOLD) -> None:
    """
    1. Generates the SAM TP dataset.
    2. Temporarily overrides the global DATA_... paths in all previous model modules.
    3. Calls the full experiment runners (efficient.csv, micro, nano, pico).
    4. Restores original paths after each run.
    """
    generate_sam_tp_dataset(iou_threshold)

    new_images_dir = os.path.join(RESULTS_DIR, "images")
    new_masks_dir = os.path.join(RESULTS_DIR, "masks")

    model_configs = [
        ("unet800k", "run_efficient_unet_experiments"),
        ("unet200k", "run_micro_cow_unet_experiments"),
        ("unet80k", "run_nano_cow_unet_experiments"),
        ("unet14k", "run_pico_cow_unet_experiments"),
    ]

    for module_name, func_name in model_configs:
        try:
            module = __import__(module_name)
            # Backup original globals
            orig_images = getattr(module, "DATA_IMAGES_DIR", None)
            orig_masks = getattr(module, "DATA_MASKS_DIR", None)

            # Override
            module.DATA_IMAGES_DIR = new_images_dir
            module.DATA_MASKS_DIR = new_masks_dir

            print(f"\n{'='*80}\nTraining {module_name} on SAM True-Positive subset...\n{'='*80}")

            # Call the experiment runner
            runner = getattr(module, func_name)
            runner()

            # Restore originals
            if orig_images is not None:
                module.DATA_IMAGES_DIR = orig_images
            if orig_masks is not None:
                module.DATA_MASKS_DIR = orig_masks

        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not run {module_name} – {e}")

    print("\nAll previous models have been retrained on the SAM True-Positive subset!")
    print("You can now evaluate them on the non-TP images using the list in sam_tp/non_tp_images.txt")


# =============================================
# How to use this module
# =============================================
# In your main.py:
#    from sam_tp_processor import generate_sam_tp_dataset, retrain_previous_models_on_tp
#
#    # Option 1: Just generate the TP dataset
#    generate_sam_tp_dataset(iou_threshold=0.75)
#
#    # Option 2: Generate + automatically retrain ALL previous models on TP data
#    retrain_previous_models_on_tp(iou_threshold=0.75)
#
# After retraining, the models are saved in their usual results/ folders.
# Use sam_tp/non_tp_images.txt to create a custom test set for evaluation on the 1-TP images.