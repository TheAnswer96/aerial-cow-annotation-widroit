import os
import json
import random
import numpy as np
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =============================================
# GLOBAL CONFIGURATION VARIABLES
# Edit these to match your setup.
# Optimized specifically for Raspberry Pi Zero 2 W (512 MB RAM, CPU-only).
# =============================================
DATA_IMAGES_DIR = os.path.join("1k", "images")
DATA_MASKS_DIR = os.path.join("1k", "masks")
RESULTS_DIR = "efficient_results"          # separate folder so previous experiments are untouched

TRAIN_PERCENTAGES = [63]
REPEATS_PER_PERCENTAGE = 1

IMAGE_SIZE = 128                           # smaller resolution = much lower memory & faster inference
BASE_CHANNELS = 16                         # tiny U-Net backbone (≈ 0.8M parameters total)
BATCH_SIZE = 16                            # can be larger thanks to small model
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED_BASE = 42

# =============================================
# EFFICIENT U-Net (designed for Raspberry Pi Zero 2 W)
# ≈ 0.8 million parameters → fits easily in 512 MB RAM
# Uses drastically reduced channel count + same skip connections
# =============================================
class EfficientUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()

        def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = double_conv(in_channels, base_channels)          # 16
        self.enc2 = double_conv(base_channels, base_channels * 2)    # 32
        self.enc3 = double_conv(base_channels * 2, base_channels * 4)# 64
        self.enc4 = double_conv(base_channels * 4, base_channels * 8)# 128

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = double_conv(base_channels * 8, base_channels * 16)  # 256

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = double_conv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = double_conv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = double_conv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = double_conv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# =============================================
# Dataset (updated for 128×128)
# =============================================
class BinarySegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_filenames: list[str]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames

        self.image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        fname = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return self.image_transform(image), self.mask_transform(mask)


# =============================================
# Metrics (same as before)
# =============================================
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> dict:
    preds = (torch.sigmoid(logits) > 0.5).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "f1": f1}


# =============================================
# Training / Evaluation helpers (unchanged)
# =============================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, device):
    model.eval()
    total_tp = total_fp = total_fn = 0.0
    smooth = 1e-6
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_tp += (preds * masks).sum().item()
            total_fp += (preds * (1 - masks)).sum().item()
            total_fn += ((1 - preds) * masks).sum().item()

    precision = total_tp / (total_tp + total_fp + smooth)
    recall = total_tp / (total_tp + total_fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = total_tp / (total_tp + total_fp + total_fn + smooth)
    dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + smooth)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "f1": f1}


# =============================================
# MAIN EXPERIMENT RUNNER (Efficient version)
# =============================================
def run_efficient_unet_experiments() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "training_logs"), exist_ok=True)

    all_images = sorted([
        f for f in os.listdir(DATA_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])
    total_images = len(all_images)
    print(f"Found {total_images} images with masks.\n")
    print(f"Using EfficientUNet (base_channels={BASE_CHANNELS}) → ~0.8M parameters")
    print(f"Input resolution: {IMAGE_SIZE}×{IMAGE_SIZE} (optimized for Raspberry Pi Zero 2 W)\n")

    all_results = []
    learning_curve = []

    for perc_idx, perc in enumerate(TRAIN_PERCENTAGES):
        perc_results = []
        print(f"\n=== Training size: {perc}% ({int(perc/100*total_images)} images) ===")

        for repeat in range(REPEATS_PER_PERCENTAGE):
            seed = RANDOM_SEED_BASE + perc_idx * 100 + repeat
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            shuffled = all_images[:]
            random.shuffle(shuffled)
            n_train = int(perc / 100 * total_images)
            train_files = shuffled[:n_train]
            test_files = shuffled[n_train:]

            train_ds = BinarySegmentationDataset(DATA_IMAGES_DIR, DATA_MASKS_DIR, train_files)
            test_ds = BinarySegmentationDataset(DATA_IMAGES_DIR, DATA_MASKS_DIR, test_files)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model = EfficientUNet(in_channels=3, out_channels=1, base_channels=BASE_CHANNELS).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.BCEWithLogitsLoss()

            print(f"  Repeat {repeat+1}/{REPEATS_PER_PERCENTAGE} | Train: {len(train_files)} | Test: {len(test_files)}")
            best_loss = float('inf')
            best_model_path = None
            # Training
            train_history = []
            for epoch in range(NUM_EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
                train_history.append({"epoch": epoch + 1, "train_loss": loss})
                if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
                    print(f"    Epoch {epoch+1:2d} - Loss: {loss:.4f}")

                if loss < best_loss:
                    best_loss = loss
                    best_model_path = os.path.join(RESULTS_DIR, "models",
                                                   f"efficient_unet_{perc}pct_rep{repeat}_best.pth")
                    torch.save(model.state_dict(), best_model_path)
            # Test evaluation
            test_metrics = evaluate_model(model, test_loader, DEVICE)
            print(f"  → Test  | IoU: {test_metrics['iou']:.4f} | Dice: {test_metrics['dice']:.4f} | "
                  f"F1: {test_metrics['f1']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f}")

            # Save training log
            log_entry = {
                "percentage": perc,
                "repeat": repeat,
                "train_size": len(train_files),
                "test_size": len(test_files),
                "seed": seed,
                "train_history": train_history,
                "test_metrics": test_metrics
            }
            log_path = os.path.join(RESULTS_DIR, "training_logs", f"efficient_{perc}pct_rep{repeat}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)

            all_results.append(log_entry)
            perc_results.append(test_metrics)

            # Save model (for Raspberry Pi deployment)
            model_path = os.path.join(RESULTS_DIR, "models", f"efficient_unet_{perc}pct_rep{repeat}.pth")
            torch.save(model.state_dict(), model_path)

        # Summary with mean ± std
        metrics_list = ["iou", "dice", "precision", "recall", "f1"]
        avg_metrics = {}
        std_metrics = {}
        for m in metrics_list:
            values = [r[m] for r in perc_results]
            avg_metrics[m] = float(np.mean(values))
            std_metrics[m] = float(np.std(values))

        learning_curve.append({
            "percentage": perc,
            "train_size": int(perc / 100 * total_images),
            **{f"avg_{m}": avg_metrics[m] for m in metrics_list},
            **{f"std_{m}": std_metrics[m] for m in metrics_list},
            "num_repeats": REPEATS_PER_PERCENTAGE
        })

    # Final saves
    with open(os.path.join(RESULTS_DIR, "all_raw_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "learning_curve_summary.json"), "w", encoding="utf-8") as f:
        json.dump(learning_curve, f, indent=2)

    # Parameter count (same for every run)
    param_count = sum(p.numel() for p in EfficientUNet(base_channels=BASE_CHANNELS).parameters())
    print("\n" + "="*100)
    print("EFFICIENT U-NET EXPERIMENTS FINISHED!")
    print(f"Model size: {param_count/1_000_000:.2f} million parameters (perfect for Raspberry Pi Zero 2 W)")
    print(f"Input resolution: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("="*100)
    print(f"{'Train %':<8} {'Train imgs':<10} {'IoU':<18} {'Dice':<18} {'F1':<18} {'Precision':<18} {'Recall':<18}")
    print("-" * 100)
    for entry in learning_curve:
        print(f"{entry['percentage']:>6}%   {entry['train_size']:>8}    "
              f"{entry['avg_iou']:.4f}±{entry['std_iou']:.4f}   "
              f"{entry['avg_dice']:.4f}±{entry['std_dice']:.4f}   "
              f"{entry['avg_f1']:.4f}±{entry['std_f1']:.4f}   "
              f"{entry['avg_precision']:.4f}±{entry['std_precision']:.4f}   "
              f"{entry['avg_recall']:.4f}±{entry['std_recall']:.4f}")

    print(f"\nResults saved in '{RESULTS_DIR}/'")
    print("   • Models ready for Pi → efficient_results/models/")
    print("   • Full logs           → efficient_results/training_logs/")


# =============================================
# Raspberry Pi Zero 2 W DEPLOYMENT HELPER
# Call this on the Pi after copying the .pth file
# =============================================
def pi_predict(model_path: str, image_path: str, threshold: float = 0.5, device: str = "cpu"):
    """
    Lightweight inference function for Raspberry Pi Zero 2 W.
    Returns binary mask (PIL Image) where cow = 255 (white), background = 0 (black).
    """
    model = EfficientUNet(in_channels=3, out_channels=1, base_channels=BASE_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Use float16 on CPU for extra speed (optional but recommended on Pi)
    model = model.half()

    # Preprocess
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device).half()

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float()

    # Resize mask back to original image resolution
    mask = transforms.functional.resize(mask.squeeze(0), orig_size, interpolation=transforms.InterpolationMode.NEAREST)
    mask_pil = transforms.ToPILImage()(mask).convert("L")
    mask_pil = mask_pil.point(lambda p: 255 if p > 0 else 0)   # black/white as requested

    inference_time = time.time() - start_time
    print(f"Inference time on {device}: {inference_time*1000:.1f} ms ({1/inference_time:.1f} FPS)")

    return mask_pil


# =============================================
# How to use this module
# =============================================
# In your main.py (PC side - training):
#    from efficient_unet import run_efficient_unet_experiments
#    run_efficient_unet_experiments()
#
# After training, copy the best .pth from efficient_results/models/ to the Pi.
#
# On the Raspberry Pi (inference):
#    from efficient_unet import pi_predict
#    mask = pi_predict("efficient_unet_80pct_rep2.pth", "photo_of_cow.jpg")
#    mask.save("cow_mask.png")