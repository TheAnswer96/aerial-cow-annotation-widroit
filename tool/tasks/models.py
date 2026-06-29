"""
Segmentor model family — shared by trainer.py and predictor.py.

Architectures lifted verbatim from experiments/ (unet14k/80k/200k/800k/unet.py).
All train at 128x128, binary output (1 channel logit). Mask convention after
ToTensor: 0.0 = object, 1.0 = background.

Select a model by its registry key (see MODEL_REGISTRY). build_model(key) returns
a fresh instance; the key is persisted in session state so the predictor rebuilds
the same architecture before loading weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGE_SIZE = 128


def get_device() -> str:
    """CUDA when available, else CPU. Used for both training and inference."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Shared building blocks ────────────────────────────────────────────────────

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
        x = self.se(self.project(self.depthwise(self.expand(x))))
        return x + residual if self.use_res else x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        return self.relu(self.bn2(self.pointwise(x)))


# ── PicoCowUNet ~14.5K (unet14k.py, base_c=12) ────────────────────────────────

class PicoCowUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_c: int = 12):
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
            nn.BatchNorm2d(base_c), nn.ReLU6(inplace=True),
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


# ── NanoCowUNet ~78K (unet80k.py, base_c=12) ──────────────────────────────────

class NanoCowUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_c: int = 12):
        super().__init__()
        self.enc1 = InvertedResidual(in_channels, base_c, expand_ratio=1)
        self.enc2 = InvertedResidual(base_c, base_c * 2, expand_ratio=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            InvertedResidual(base_c * 2, base_c * 4, expand_ratio=4),
            InvertedResidual(base_c * 4, base_c * 4, expand_ratio=4),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.objectness_fc = nn.Sequential(
            nn.Linear(base_c * 4, base_c), nn.ReLU(inplace=True),
            nn.Linear(base_c, base_c * 4),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj_skip2 = nn.Conv2d(base_c * 2, base_c * 4, 1, bias=False)
        self.dec2 = InvertedResidual(base_c * 4, base_c * 4, expand_ratio=4)
        self.se2 = SqueezeExcitation(base_c * 4)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj_skip1 = nn.Conv2d(base_c, base_c * 4, 1, bias=False)
        self.dec1 = InvertedResidual(base_c * 4, base_c * 2, expand_ratio=4)
        self.se1 = SqueezeExcitation(base_c * 2)
        self.final = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_c), nn.ReLU6(inplace=True),
            nn.Conv2d(base_c, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        obj = self.objectness_fc(self.global_pool(b).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        b = b + obj
        d2 = self.se2(self.dec2(self.up2(b) + self.proj_skip2(e2)))
        d1 = self.se1(self.dec1(self.up1(d2) + self.proj_skip1(e1)))
        return self.final(d1)


# ── MicroCowUNet ~191K (unet200k.py, base_c=24) ───────────────────────────────

class MicroCowUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_c: int = 24):
        super().__init__()
        self.enc1 = DepthwiseSeparableConv(in_channels, base_c)
        self.enc2 = DepthwiseSeparableConv(base_c, base_c * 2)
        self.enc3 = DepthwiseSeparableConv(base_c * 2, base_c * 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(base_c * 4, base_c * 8),
            DepthwiseSeparableConv(base_c * 8, base_c * 8),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.objectness_fc = nn.Sequential(
            nn.Linear(base_c * 8, base_c * 2), nn.ReLU(inplace=True),
            nn.Linear(base_c * 2, base_c * 8),
        )
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj_skip3 = nn.Conv2d(base_c * 4, base_c * 8, 1, bias=False)
        self.dec3 = DepthwiseSeparableConv(base_c * 8, base_c * 8)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj_skip2 = nn.Conv2d(base_c * 2, base_c * 8, 1, bias=False)
        self.dec2 = DepthwiseSeparableConv(base_c * 8, base_c * 4)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj_skip1 = nn.Conv2d(base_c, base_c * 4, 1, bias=False)
        self.dec1 = DepthwiseSeparableConv(base_c * 4, base_c * 2)
        self.final = nn.Sequential(
            nn.Conv2d(base_c * 2, base_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_c), nn.ReLU6(inplace=True),
            nn.Conv2d(base_c, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        obj = self.objectness_fc(self.global_pool(b).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        b = b + obj
        d3 = self.dec3(self.up3(b) + self.proj_skip3(e3))
        d2 = self.dec2(self.up2(d3) + self.proj_skip2(e2))
        d1 = self.dec1(self.up1(d2) + self.proj_skip1(e1))
        return self.final(d1)


# ── EfficientUNet ~800K (unet800k.py, base_channels=16) ───────────────────────

def _double_conv_bn(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class EfficientUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = _double_conv_bn(in_channels, c)
        self.enc2 = _double_conv_bn(c, c * 2)
        self.enc3 = _double_conv_bn(c * 2, c * 4)
        self.enc4 = _double_conv_bn(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = _double_conv_bn(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = _double_conv_bn(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = _double_conv_bn(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _double_conv_bn(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _double_conv_bn(c * 2, c)
        self.out_conv = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_conv(d1)


# ── UNet ~8M (unet.py) ────────────────────────────────────────────────────────

def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = _double_conv(in_channels, 64)
        self.enc2 = _double_conv(64, 128)
        self.enc3 = _double_conv(128, 256)
        self.enc4 = _double_conv(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = _double_conv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_conv(d1)


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'pico':      {'cls': PicoCowUNet,   'label': 'PicoCowUNet',   'params': '~12K',   'target': 'MCU (ESP32/STM32)'},
    'nano':      {'cls': NanoCowUNet,   'label': 'NanoCowUNet',   'params': '~78K',   'target': 'Raspberry Pi Zero 2 W'},
    'micro':     {'cls': MicroCowUNet,  'label': 'MicroCowUNet',  'params': '~191K',  'target': 'Raspberry Pi Zero 2 W'},
    'efficient': {'cls': EfficientUNet, 'label': 'EfficientUNet', 'params': '~1.9M',  'target': 'Raspberry Pi / server'},
    'unet':      {'cls': UNet,          'label': 'UNet',          'params': '~31M',   'target': 'GPU / server'},
}

DEFAULT_MODEL = 'pico'


def normalize_key(key: str | None) -> str:
    return key if key in MODEL_REGISTRY else DEFAULT_MODEL


def model_label(key: str | None) -> str:
    return MODEL_REGISTRY[normalize_key(key)]['label']


def build_model(key: str | None) -> nn.Module:
    return MODEL_REGISTRY[normalize_key(key)]['cls']()
