"""
SAM2 annotation runner (background thread).
Mask convention: 0 = object (cow), 255 = background.
Falls back to Otsu thresholding if sam2 package not installed.
"""
import queue
import numpy as np
from pathlib import Path
from PIL import Image


def run_sam2(workspace: Path, seed_files: list[str], q: queue.Queue) -> None:
    try:
        _run_sam2(workspace, seed_files, q)
    except Exception as exc:
        q.put({'error': f'SAM2 runner crashed: {exc}', 'done': True})


def _run_sam2(workspace: Path, seed_files: list[str], q: queue.Queue) -> None:
    masks_dir = workspace / 'sam_masks'
    masks_dir.mkdir(exist_ok=True)
    seed_dir = workspace / 'seed'
    total = len(seed_files)

    mask_generator = None
    try:
        import os
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        checkpoint = os.environ.get('SAM2_CHECKPOINT', '')
        cfg = os.environ.get('SAM2_CONFIG', 'sam2_hiera_small.yaml')
        if checkpoint and Path(checkpoint).exists():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sam2 = build_sam2(cfg, checkpoint, device=device)
            mask_generator = SAM2AutomaticMaskGenerator(
                sam2,
                points_per_side=16,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.85,
                min_mask_region_area=300,
            )
            q.put({'msg': 'SAM2 loaded', 'step': 0, 'total': total, 'pct': 0})
        else:
            q.put({'msg': 'No SAM2 checkpoint — using Otsu fallback', 'step': 0, 'total': total, 'pct': 0})
    except Exception:
        q.put({'msg': 'SAM2 unavailable — using Otsu fallback', 'step': 0, 'total': total, 'pct': 0})

    for i, fname in enumerate(seed_files):
        img_path = seed_dir / fname
        out_path = masks_dir / (Path(fname).stem + '.png')
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)

            if mask_generator is not None:
                masks = mask_generator.generate(img_np)
                if masks:
                    best = max(masks, key=lambda m: m['area'])
                    seg = best['segmentation'].astype(np.uint8)
                    out = np.where(seg == 1, 0, 255).astype(np.uint8)
                else:
                    out = np.full(img_np.shape[:2], 255, np.uint8)
            else:
                out = _otsu_mask(img_np)

            Image.fromarray(out, 'L').save(str(out_path))
        except Exception:
            try:
                w, h = Image.open(img_path).size
                Image.new('L', (w, h), 255).save(str(out_path))
            except Exception:
                pass

        q.put({'step': i + 1, 'total': total, 'filename': fname,
               'pct': int((i + 1) / total * 100)})

    q.put({'done': True, 'total': total})


def _otsu_mask(img_np: np.ndarray) -> np.ndarray:
    """Dark regions → object (cow tends to be darker than pen floor from drone view)."""
    gray = (0.299 * img_np[:, :, 0] +
            0.587 * img_np[:, :, 1] +
            0.114 * img_np[:, :, 2]).astype(np.uint8)
    t = _otsu_threshold(gray)
    return np.where(gray < t, 0, 255).astype(np.uint8)


def _otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    best_t, best_var = 0, 0.0
    w0, sum0 = 0, 0.0
    sum_total = float(np.dot(np.arange(256), hist))
    for t in range(256):
        w0 += hist[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        sum0 += t * hist[t]
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t
