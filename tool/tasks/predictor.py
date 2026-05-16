"""
PicoCowUNet inference runner (background thread).
Loads trained model from workspace/model/pico.pth.
Saves predictions to workspace/predictions/ as PNG masks.
Mask convention: 0=object, 255=background (consistent with training).
"""
import queue
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from PIL import Image

IMAGE_SIZE = 128
BASE_CHANNELS = 12
DEVICE = 'cpu'

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ── Model (same as trainer.py) ────────────────────────────────────────────────

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


# ── Inference runner ──────────────────────────────────────────────────────────

def run_prediction(workspace: Path, filenames: list[str], model_path: str,
                   q: queue.Queue) -> None:
    pred_dir = workspace / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    dataset_dir = workspace / 'dataset'
    total = len(filenames)

    try:
        model = PicoCowUNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()
        q.put({'msg': 'PicoCowUNet loaded', 'step': 0, 'total': total, 'pct': 0})
    except Exception as exc:
        q.put({'error': f'Failed to load model: {exc}', 'done': True})
        return

    for i, fname in enumerate(filenames):
        img_path = dataset_dir / fname
        out_path = pred_dir / (Path(fname).stem + '.png')
        try:
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            inp = TRANSFORM(img).unsqueeze(0).to(DEVICE)

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
