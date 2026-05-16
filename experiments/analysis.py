import os
import csv
import numpy as np
from PIL import Image

# =============================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================
IMAGES_DIR = os.path.join("1k", "images")
MASKS_DIR = os.path.join("1k", "masks")
RESULTS_DIR = "quality_analysis"

# =============================================
# QUALITY METRICS HELPERS
# =============================================
def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Pure numpy approximation of Laplacian variance (no-reference blur measure)."""
    # 3x3 Laplacian kernel applied via finite differences
    lap = np.zeros_like(gray, dtype=np.float32)
    lap[1:-1, 1:-1] = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(np.var(lap))


def compute_blurriness_score(pil_img: Image.Image) -> float:
    """
    No-reference blurriness score in [0, 1].
    0 = perfectly sharp, 1 = extremely blurry.
    Based on Laplacian variance (higher variance → sharper → lower score).
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    lap_var = compute_laplacian_variance(gray)
    # Tuned normalization for typical 8-bit images (0–10000+ variance range)
    blur_score = 1.0 / (1.0 + lap_var / 300.0)
    return float(np.clip(blur_score, 0.0, 1.0))


def compute_perceived_brightness(pil_img: Image.Image) -> float:
    """
    Perceived brightness in [-1, 1].
    -1 = completely dark, +1 = completely bright.
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    mean_val = float(gray.mean())
    return (mean_val - 128.0) / 128.0


def compute_mask_geometry(mask_pil: Image.Image) -> tuple:
    """
    Returns:
        center_x_pct, center_y_pct (in [0,1]),
        bbox_width_pct, bbox_height_pct (in percentage of image size).
    Object = black pixels (value 0).
    """
    mask_np = np.array(mask_pil)  # 0 = cow, 255 = background
    fg = (mask_np == 0)

    if not np.any(fg):
        # No object detected
        return 0.5, 0.5, 0.0, 0.0

    ys, xs = np.nonzero(fg)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    w = max_x - min_x + 1
    h = max_y - min_y + 1

    img_w, img_h = mask_pil.size
    center_x_pct = center_x / img_w
    center_y_pct = center_y / img_h
    bbox_width_pct = (w / img_w) * 100.0
    bbox_height_pct = (h / img_h) * 100.0

    return center_x_pct, center_y_pct, bbox_width_pct, bbox_height_pct


# =============================================
# MAIN ANALYSIS MODULE
# =============================================
def run_mask_quality_analysis() -> None:
    """
    Analyzes all 1K images + masks.
    Creates two CSV files:
      - quality_analysis/image_metrics.csv
      - quality_analysis/mask_metrics.csv
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Get all images (assume every image has a matching mask)
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])

    print(f"Found {len(image_files)} images in 1k/images/ – starting quality analysis...")

    image_rows = []
    mask_rows = []

    for idx, fname in enumerate(image_files):
        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx+1}/{len(image_files)}")

        img_path = os.path.join(IMAGES_DIR, fname)
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(MASKS_DIR, mask_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load mask (must exist)
        try:
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            print(f"Warning: Missing mask for {fname}")
            continue

        # === Image metrics ===
        img_blur = compute_blurriness_score(image)
        img_bright = compute_perceived_brightness(image)

        image_rows.append([idx, fname, img_blur, img_bright])

        # === Mask metrics ===
        center_x_pct, center_y_pct, bbox_w_pct, bbox_h_pct = compute_mask_geometry(mask)
        mask_blur = compute_blurriness_score(mask)
        mask_bright = compute_perceived_brightness(mask)

        mask_rows.append([
            idx,
            fname,                     # same filename as image
            center_x_pct,
            center_y_pct,
            bbox_w_pct,
            bbox_h_pct,
            mask_blur,
            mask_bright
        ])

    # === Save Image CSV ===
    image_csv_path = os.path.join(RESULTS_DIR, "image_metrics.csv")
    with open(image_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "blurriness", "brightness"])
        writer.writerows(image_rows)

    # === Save Mask CSV ===
    mask_csv_path = os.path.join(RESULTS_DIR, "mask_metrics.csv")
    with open(mask_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "filename",
            "center_x_pct", "center_y_pct",
            "bbox_width_pct", "bbox_height_pct",
            "blurriness", "brightness"
        ])
        writer.writerows(mask_rows)

    # === Quick summary ===
    print("\n" + "="*80)
    print("MASK QUALITY ANALYSIS COMPLETED!")
    print(f"Processed {len(image_rows)} images/masks")
    print(f"→ image_metrics.csv  : {image_csv_path}")
    print(f"→ mask_metrics.csv   : {mask_csv_path}")
    print("\nColumns explained:")
    print("  • blurriness     : 0.0 (sharp) → 1.0 (very blurry)")
    print("  • brightness     : -1.0 (dark) → +1.0 (bright)")
    print("  • center_x_pct   : horizontal center of cow [0.0 left – 1.0 right]")
    print("  • center_y_pct   : vertical center of cow [0.0 top – 1.0 bottom]")
    print("  • bbox_..._pct   : bounding box size as % of full image")
    print("="*80)


# =============================================
# How to use this module
# =============================================
# In your main.py (or run directly):
#    from mask_quality_analyzer import run_mask_quality_analysis
#    run_mask_quality_analysis()
#
# The two CSV files will be saved in the new folder "quality_analysis/"