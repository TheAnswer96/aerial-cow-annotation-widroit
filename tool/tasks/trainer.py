"""
PicoCowUNet training runner (background thread).
Architecture copied from unet14k.py — ~14.5K parameters.
Mask convention: 0=object, 255=background.
After ToTensor: 0.0=object, 1.0=background (consistent with original training code).
"""
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

IMAGE_SIZE = 128
BASE_CHANNELS = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Model architecture (from unet14k.py) ─────────────────────────────────────

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = torch.sigmoid(self.fc2(torch.relu(self.fc1(y)))).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 2):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = (in_ch == out_ch)
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ) if expand_ratio != 1 else nn.Identity()
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SqueezeExcitation(out_ch) if out_ch >= 8 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.project(self.depthwise(self.expand(x)))
        x = self.se(x)
        return x + residual if self.use_res else x


class PicoCowUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_c: int = BASE_CHANNELS):
        super().__init__()
        self.enc1 = InvertedResidual(in_channels, base_c, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            InvertedResidual(base_c, base_c * 2, 2),
            InvertedResidual(base_c * 2, base_c * 2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.objectness_fc = nn.Sequential(
            nn.Linear(base_c * 2, base_c), nn.ReLU(inplace=True),
            nn.Linear(base_c, base_c * 2),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj_skip1 = nn.Conv2d(base_c, base_c * 2, 1, bias=False)
        self.dec1 = InvertedResidual(base_c * 2, base_c * 2)
        self.se1 = SqueezeExcitation(base_c * 2)
        self.final = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(base_c, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        b = self.bottleneck(self.pool(e1))
        obj = self.objectness_fc(self.global_pool(b).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        b = b + obj
        d1 = self.up1(b) + self.proj_skip1(e1)
        d1 = self.se1(self.dec1(d1))
        return self.final(d1)


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
                 num_epochs: int = 25) -> None:
    model_dir = workspace / 'model'
    model_dir.mkdir(exist_ok=True)

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

    q.put({'msg': f'Training PicoCowUNet on {len(valid)} images', 'epoch': 0, 'total': num_epochs, 'pct': 0})

    ds = BinarySegDataset(images_dir, masks_dir, valid)
    bs = min(BATCH_SIZE, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)

    model = PicoCowUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), str(model_dir / 'pico.pth'))

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
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
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
            'iou': round(iou, 4),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'best_loss': round(best_loss, 4),
            'train_images': len(valid),
        },
    })
