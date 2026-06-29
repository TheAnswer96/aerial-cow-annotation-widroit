"""
Segmentor training runner (background thread).
Model architecture is chosen at upload time (see tasks/models.py registry) and
passed in via model_key. Trains on GPU when available, else CPU.
Mask convention: 0=object, 255=background.
After ToTensor: 0.0=object, 1.0=background (consistent with original training code).
"""
import queue
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tasks.models import IMAGE_SIZE, build_model, get_device, model_label

BATCH_SIZE = 16
LEARNING_RATE = 1e-4


# ── Dataset ───────────────────────────────────────────────────────────────────

class BinarySegDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, filenames: list[str]):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = filenames
        self.img_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img = Image.open(self.images_dir / fname).convert('RGB')
        mask_name = Path(fname).stem + '.png'
        mask_path = self.masks_dir / mask_name
        mask = Image.open(mask_path).convert('L') if mask_path.exists() else Image.new('L', img.size, 255)
        return self.img_tf(img), self.mask_tf(mask)


# ── Training runner ───────────────────────────────────────────────────────────

def run_training(workspace: Path, accepted_files: list[str], q: queue.Queue,
                 num_epochs: int = 25, model_key: str = 'pico') -> None:
    try:
        _run_training(workspace, accepted_files, q, num_epochs, model_key)
    except Exception as exc:
        q.put({'error': f'Training crashed: {exc}', 'done': True})


def _run_training(workspace: Path, accepted_files: list[str], q: queue.Queue,
                  num_epochs: int, model_key: str) -> None:
    model_dir = workspace / 'model'
    model_dir.mkdir(exist_ok=True)
    device = get_device()
    label = model_label(model_key)

    if not accepted_files:
        q.put({'error': 'No accepted images to train on.', 'done': True})
        return

    images_dir = workspace / 'accepted_images'
    masks_dir = workspace / 'accepted_masks'

    # Verify masks exist, skip missing
    valid = [f for f in accepted_files if (masks_dir / (Path(f).stem + '.png')).exists()]
    if not valid:
        q.put({'error': 'No accepted masks found. Check annotation step.', 'done': True})
        return

    q.put({'msg': f'Training {label} on {len(valid)} images ({device.upper()})',
           'epoch': 0, 'total': num_epochs, 'pct': 0})

    ds = BinarySegDataset(images_dir, masks_dir, valid)
    bs = min(BATCH_SIZE, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)

    model = build_model(model_key).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), str(model_dir / 'model.pth'))

        q.put({
            'epoch': epoch + 1,
            'total': num_epochs,
            'loss': round(avg_loss, 4),
            'pct': int((epoch + 1) / num_epochs * 100),
        })

    # Final eval on train set (no held-out set in seed; just report train metrics)
    model.eval()
    tp = fp = fn = 0.0
    eps = 1e-6
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            tp += (preds * masks).sum().item()
            fp += (preds * (1 - masks)).sum().item()
            fn += ((1 - preds) * masks).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    q.put({
        'done': True,
        'metrics': {
            'model': label,
            'device': device,
            'iou': round(iou, 4),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'best_loss': round(best_loss, 4),
            'train_images': len(valid),
        },
    })
