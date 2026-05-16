import os
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =============================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================
DATA_IMAGES_DIR = os.path.join("1k", "images")
DATA_MASKS_DIR = os.path.join("1k", "masks")
RESULTS_DIR = "results"

TRAIN_PERCENTAGES = [63]
REPEATS_PER_PERCENTAGE = 1

IMAGE_SIZE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED_BASE = 42

# =============================================
# U-Net Model
# =============================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = double_conv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
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
# Dataset (unchanged)
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

    def __getitem__(self, idx):
        fname = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask


# =============================================
# Metrics
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
# Training & Evaluation
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
# MAIN EXPERIMENT RUNNER (Updated)
# =============================================
def run_unet_experiments() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "training_logs"), exist_ok=True)

    all_images = sorted([
        f for f in os.listdir(DATA_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])
    total_images = len(all_images)
    print(f"Found {total_images} images with matching masks.\n")

    all_results = []      # Raw results for every run
    learning_curve = []   # Summary with mean + std

    for perc_idx, perc in enumerate(TRAIN_PERCENTAGES):
        perc_results = []
        print(f"\n=== Training size: {perc}% ({int(perc/100*total_images)} images) ===")

        for repeat in range(REPEATS_PER_PERCENTAGE):
            seed = RANDOM_SEED_BASE + perc_idx * 100 + repeat
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Split
            shuffled = all_images[:]
            random.shuffle(shuffled)
            n_train = int(perc / 100 * total_images)
            train_files = shuffled[:n_train]
            test_files = shuffled[n_train:]

            # Datasets
            train_ds = BinarySegmentationDataset(DATA_IMAGES_DIR, DATA_MASKS_DIR, train_files)
            test_ds = BinarySegmentationDataset(DATA_IMAGES_DIR, DATA_MASKS_DIR, test_files)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            # Model
            model = UNet(in_channels=3, out_channels=1).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.BCEWithLogitsLoss()

            print(f"  Repeat {repeat+1}/{REPEATS_PER_PERCENTAGE} | Train: {len(train_files)} | Test: {len(test_files)}")

            # === Training with logging ===
            best_loss = float('inf')
            best_model_path = None
            train_history = []
            for epoch in range(NUM_EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
                train_history.append({"epoch": epoch + 1, "train_loss": loss})

                if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
                    print(f"    Epoch {epoch+1:2d} - Loss: {loss:.4f}")
                if loss < best_loss:
                    best_loss = loss
                    best_model_path = os.path.join(RESULTS_DIR, "models",
                                                   f"unet_{perc}pct_rep{repeat}_best.pth")
                    torch.save(model.state_dict(), best_model_path)
            # === Final evaluation on test set ===
            test_metrics = evaluate_model(model, test_loader, DEVICE)
            print(f"  → Test  | IoU: {test_metrics['iou']:.4f} | Dice: {test_metrics['dice']:.4f} | "
                  f"F1: {test_metrics['f1']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f}")

            # Save training log for this run
            log_entry = {
                "percentage": perc,
                "repeat": repeat,
                "train_size": len(train_files),
                "test_size": len(test_files),
                "seed": seed,
                "train_history": train_history,
                "test_metrics": test_metrics
            }

            log_path = os.path.join(RESULTS_DIR, "training_logs", f"train_{perc}pct_rep{repeat}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)

            # Store for summary
            all_results.append(log_entry)
            perc_results.append(test_metrics)

            # Optional: save model
            # torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "models", f"unet_{perc}pct_rep{repeat}.pth"))

        # === Compute mean and std for this percentage ===
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
            "avg_iou": avg_metrics["iou"],
            "std_iou": std_metrics["iou"],
            "avg_dice": avg_metrics["dice"],
            "std_dice": std_metrics["dice"],
            "avg_precision": avg_metrics["precision"],
            "std_precision": std_metrics["precision"],
            "avg_recall": avg_metrics["recall"],
            "std_recall": std_metrics["recall"],
            "avg_f1": avg_metrics["f1"],
            "std_f1": std_metrics["f1"],
            "num_repeats": REPEATS_PER_PERCENTAGE
        })

    # Save all results
    with open(os.path.join(RESULTS_DIR, "all_raw_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "learning_curve_summary.json"), "w", encoding="utf-8") as f:
        json.dump(learning_curve, f, indent=2)

    # Pretty print summary table
    print("\n" + "="*90)
    print("EXPERIMENT SUMMARY (Mean ± Std over {} repeats)".format(REPEATS_PER_PERCENTAGE))
    print("="*90)
    print(f"{'Train %':<8} {'Train imgs':<10} {'IoU':<18} {'Dice':<18} {'F1':<18} {'Precision':<18} {'Recall':<18}")
    print("-" * 90)

    for entry in learning_curve:
        print(f"{entry['percentage']:>6}%   {entry['train_size']:>8}    "
              f"{entry['avg_iou']:.4f}±{entry['std_iou']:.4f}   "
              f"{entry['avg_dice']:.4f}±{entry['std_dice']:.4f}   "
              f"{entry['avg_f1']:.4f}±{entry['std_f1']:.4f}   "
              f"{entry['avg_precision']:.4f}±{entry['std_precision']:.4f}   "
              f"{entry['avg_recall']:.4f}±{entry['std_recall']:.4f}")

    print(f"\nResults saved in '{RESULTS_DIR}/'")
    print("   • Detailed training logs  → results/training_logs/")
    print("   • All raw results        → results/all_raw_results.json")
    print("   • Learning curve summary → results/learning_curve_summary.json")


# =============================================
# Usage
# =============================================
# In your main.py:
#    from unet_experiment import run_unet_experiments
#    run_unet_experiments()