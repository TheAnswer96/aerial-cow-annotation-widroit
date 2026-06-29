"""
SAM2 annotation runner (background thread).
Mask convention: 0 = object (cow), 255 = background.

Uses ultralytics SAM2 in "segment everything" mode. Weights live inside the
repo at tool/models/ and auto-download on first run. Falls back to Otsu
thresholding if ultralytics is missing or the model can't be loaded.
"""
import os
import queue
from pathlib import Path

import numpy as np
from PIL import Image

# Weights are kept inside the repo so the tool is self-contained.
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
# sam2_t (tiny, ~39 MB) is the CPU-friendly default; override for quality:
#   SAM2_MODEL=sam2_b.pt  (base, ~162 MB)  /  sam2_s.pt  /  sam2_l.pt
SAM2_MODEL = os.environ.get('SAM2_MODEL', 'sam2_t.pt')


def run_sam2(workspace: Path, seed_files: list[str], q: queue.Queue) -> None:
    try:
        _run_sam2(workspace, seed_files, q)
    except Exception as exc:
        q.put({'error': f'SAM2 runner crashed: {exc}', 'done': True})


def _load_model(q: queue.Queue, total: int):
    """Load ultralytics SAM2, downloading weights into tool/models/ if absent.

    Returns the model, or None to signal the Otsu fallback.
    """
    try:
        from ultralytics import SAM
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weights = MODELS_DIR / SAM2_MODEL
        # SAM(<path>) downloads the named asset to <path> when it doesn't exist.
        model = SAM(str(weights))
        q.put({'msg': f'SAM2 loaded ({SAM2_MODEL})', 'step': 0, 'total': total, 'pct': 0})
        return model
    except Exception as exc:
        q.put({'msg': f'SAM2 unavailable ({exc}) — using Otsu fallback',
               'step': 0, 'total': total, 'pct': 0})
        return None


def _run_sam2(workspace: Path, seed_files: list[str], q: queue.Queue) -> None:
    masks_dir = workspace / 'sam_masks'
    masks_dir.mkdir(exist_ok=True)
    seed_dir = workspace / 'seed'
    total = len(seed_files)

    # Emit immediately so the UI shows life while weights download / model loads
    # (first run fetches ~74 MB; load itself is a few seconds on CPU).
    q.put({'msg': 'Loading SAM2 model…', 'step': 0, 'total': total, 'pct': 0})
    model = _load_model(q, total)

    for i, fname in enumerate(seed_files):
        img_path = seed_dir / fname
        out_path = masks_dir / (Path(fname).stem + '.png')
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)

            if model is not None:
                out = _sam2_mask(model, img_np)
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


def _sam2_mask(model, img_np: np.ndarray) -> np.ndarray:
    """Prompt SAM2 with a centered foreground point and take that mask as the cow.

    A single point prompt is one forward pass (~sub-second on CPU), versus
    segment-everything which runs a dense point grid (~8 s/image on CPU and
    tends to return the background as its largest mask). Drone crops frame the
    cow roughly centrally, and every mask is human-reviewed afterwards.

    Output convention: 0 = object, 255 = background.
    """
    h, w = img_np.shape[:2]
    res = model(img_np, points=[[w // 2, h // 2]], labels=[1], verbose=False)
    masks = res[0].masks if res else None
    if masks is None or len(masks.data) == 0:
        return np.full((h, w), 255, np.uint8)
    data = masks.data.cpu().numpy()  # (N, H, W) at original resolution
    # Point prompt usually yields one mask; if several, take the largest.
    areas = data.reshape(data.shape[0], -1).sum(axis=1)
    seg = data[int(areas.argmax())]
    return np.where(seg > 0.5, 0, 255).astype(np.uint8)


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
