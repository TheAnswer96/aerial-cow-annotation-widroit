import os
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

# =============================================
# GLOBAL CONFIGURATION VARIABLES
# Edit these to match your project paths/setup.
# No command-line arguments or argparsers used.
# =============================================
RAW_IMAGES_DIR = r"raw\images"
RAW_ANNOTATIONS_DIR = r"raw\annotations"
OUTPUT_BASE_DIR = "1k"
NUM_SAMPLES = 1000  # exactly 1K images

# The module will automatically find the single .json file inside RAW_ANNOTATIONS_DIR.
# You can override the name if you prefer a fixed filename:
# ANNOTATION_FILENAME = "instances.json"  # uncomment and set if you want to force a name


def _find_annotation_file() -> str:
    """Find the COCO JSON file inside RAW_ANNOTATIONS_DIR."""
    json_files = [f for f in os.listdir(RAW_ANNOTATIONS_DIR) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No .json annotation file found in {RAW_ANNOTATIONS_DIR}")
    if len(json_files) > 1:
        print(f"Warning: Multiple JSON files found in {RAW_ANNOTATIONS_DIR}. Using the first one: {json_files[0]}")
    return os.path.join(RAW_ANNOTATIONS_DIR, json_files[0])


def _decode_rle(rle_dict: dict, height: int, width: int) -> np.ndarray:
    """
    Decode COCO Run-Length Encoded (RLE) segmentation to a binary mask.
    Returns a (height, width) uint8 array where 1 = object pixel, 0 = background.
    Works with the common list-of-integers format stored in COCO JSON files.
    (Compressed string RLE is NOT supported here to keep the module dependency-free;
     if your dataset uses string RLE, install pycocotools and replace this function.)
    """
    counts = rle_dict.get("counts")
    if not isinstance(counts, list):
        raise ValueError("RLE 'counts' must be a list of integers (string RLE not supported in this pure-Python module).")

    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    val = 0  # COCO RLE starts with background (0)
    for count in counts:
        if pos + count > len(mask):
            break
        mask[pos:pos + count] = val
        pos += count
        val = 1 - val  # toggle between 0 and 1

    return mask.reshape((height, width))


def create_1k_subset() -> None:
    """
    Main function of the module.
    - Randomly selects 1000 images from raw/images
    - Copies them to 1k/images
    - Creates a new COCO JSON with only those images + their annotations in 1k/annotations
    - Generates binary masks (object pixels = black, background = white) in 1k/masks as PNG
    """
    random.seed(42)  # reproducible selection (feel free to remove)

    # Create output directory structure
    images_out = os.path.join(OUTPUT_BASE_DIR, "images")
    anns_out = os.path.join(OUTPUT_BASE_DIR, "annotations")
    masks_out = os.path.join(OUTPUT_BASE_DIR, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(anns_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    # === 1. Randomly select 1K images ===
    image_files = [
        f for f in os.listdir(RAW_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]
    if len(image_files) < NUM_SAMPLES:
        raise ValueError(f"Not enough images in {RAW_IMAGES_DIR} (found {len(image_files)}, need {NUM_SAMPLES})")

    selected_filenames = random.sample(image_files, NUM_SAMPLES)
    print(f"Selected {NUM_SAMPLES} images randomly.")

    # Copy images
    for fname in selected_filenames:
        src = os.path.join(RAW_IMAGES_DIR, fname)
        dst = os.path.join(images_out, fname)
        shutil.copy2(src, dst)

    # === 2. Load original COCO annotations ===
    ann_path = _find_annotation_file()
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build mapping filename -> image_id
    filename_to_id = {img["file_name"]: img["id"] for img in coco["images"]}
    selected_ids = [filename_to_id[fname] for fname in selected_filenames if fname in filename_to_id]

    # Filter images and annotations
    new_images = [img for img in coco["images"] if img["id"] in selected_ids]
    new_annotations = [ann for ann in coco.get("annotations", []) if ann["image_id"] in selected_ids]

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", []),
    }

    # Save new COCO JSON (same filename as original for simplicity)
    original_ann_name = os.path.basename(ann_path)
    new_ann_path = os.path.join(anns_out, original_ann_name)
    with open(new_ann_path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)
    print(f"Created new annotation file: {new_ann_path} ({len(new_images)} images, {len(new_annotations)} annotations)")

    # === 3. Build per-image annotation lookup for mask generation ===
    anns_by_image = defaultdict(list)
    for ann in new_annotations:
        anns_by_image[ann["image_id"]].append(ann)

    # === 4. Generate binary masks (object = black, background = white) ===
    print("Generating binary masks (PNG) in 1k/masks ...")
    for img_info in new_images:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # Start with white background (PIL)
        mask_pil = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(mask_pil)

        # Draw all polygon segmentations
        for ann in anns_by_image[img_id]:
            seg = ann.get("segmentation")
            if isinstance(seg, list) and seg:  # polygon format
                for polygon in seg:  # each annotation can have multiple polygons
                    if len(polygon) < 6 or len(polygon) % 2 != 0:
                        continue
                    points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(points, fill=0)  # black = object

        # Convert to numpy so we can easily merge RLEs
        mask_np = np.array(mask_pil, dtype=np.uint8)

        # Merge any RLE segmentations (union with polygons)
        for ann in anns_by_image[img_id]:
            seg = ann.get("segmentation")
            if isinstance(seg, dict) and "counts" in seg and "size" in seg:
                rle_mask = _decode_rle(seg, height, width)
                mask_np[rle_mask == 1] = 0  # black = object

        # Save as PNG (same base name as the image)
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(masks_out, mask_name)
        Image.fromarray(mask_np).save(mask_path)

    print(f"Done! 1K subset created in '{OUTPUT_BASE_DIR}/'")
    print(f"   • Images   : {images_out} ({NUM_SAMPLES} files)")
    print(f"   • Annotations: {new_ann_path}")
    print(f"   • Masks    : {masks_out} ({NUM_SAMPLES} PNG files)")


# =============================================
# How to use this module
# =============================================
# Option 1: Run directly as a script
#    python create_1k_subset.py
#
# Option 2: Import into your main.py (recommended)
#    from create_1k_subset import create_1k_subset
#    create_1k_subset()   # ← this is what "invoked into the main" means
#
# Just edit the GLOBAL variables at the top if your folder names differ.