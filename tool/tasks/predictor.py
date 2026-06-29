"""
Segmentor inference runner (background thread).
Rebuilds the architecture chosen at upload time (model_key) and loads
workspace/model/model.pth. Runs on GPU when available, else CPU.
Saves predictions to workspace/predictions/ as PNG masks.
Mask convention: 0=object, 255=background (consistent with training).
"""
import queue
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tasks.models import IMAGE_SIZE, build_model, get_device

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def run_prediction(workspace: Path, filenames: list[str], model_path: str,
                   q: queue.Queue, model_key: str = 'pico') -> None:
    pred_dir = workspace / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    dataset_dir = workspace / 'dataset'
    total = len(filenames)
    device = get_device()

    try:
        model = build_model(model_key).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        q.put({'msg': f'Model loaded ({device.upper()})', 'step': 0, 'total': total, 'pct': 0})
    except Exception as exc:
        q.put({'error': f'Failed to load model: {exc}', 'done': True})
        return

    for i, fname in enumerate(filenames):
        img_path = dataset_dir / fname
        out_path = pred_dir / (Path(fname).stem + '.png')
        try:
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            inp = TRANSFORM(img).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()  # [0,1]: near 1=bg

            # Resize prob map to original resolution, then threshold
            prob_img = Image.fromarray((prob * 255).astype(np.uint8), mode='L')
            prob_img = prob_img.resize((orig_w, orig_h), Image.NEAREST)
            arr = np.array(prob_img)
            # high prob → background=255, low prob → object=0  (project convention)
            out_mask = np.where(arr > 127, 255, 0).astype(np.uint8)
            Image.fromarray(out_mask, 'L').save(str(out_path))
        except Exception:
            pass

        q.put({
            'step': i + 1,
            'total': total,
            'filename': fname,
            'pct': int((i + 1) / total * 100),
        })

    q.put({'done': True, 'total': total})
