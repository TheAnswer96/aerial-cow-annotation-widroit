"""
Inference Benchmark Module for Cow Segmentation Models
======================================================
Loads one trained model per architecture (Efficient, Micro, Nano, Pico),
measures average ± std inference time on CPU (batch size 1),
and estimates memory usage (model size + peak activation delta).
"""

import os
import time
import psutil
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Import model classes from previous modules
from unet import UNet
from unet800k import EfficientUNet
from unet200k import MicroCowUNet
from unet80k import NanoCowUNet
from unet14k import PicoCowUNet

# =============================================
# GLOBAL CONFIGURATION
# =============================================
DEVICE = "cpu"
IMAGE_SIZE = 128
NUM_WARMUP = 100
NUM_INFERENCE_RUNS = 100  # higher number → more reliable std
BATCH_SIZE = 1  # single-image inference (realistic for MCU)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =============================================
# MODEL PATHS
# Update these to point to your trained checkpoints
# (recommended: the 80% training size model from the latest SAM-TP run)
# =============================================
MODEL_PATHS = {
"UNET (≈8M params)": {
        "class": UNet,
        "path": "results/models/unet_63pct_rep0_best.pth",
    },
    "EfficientUNet (≈800K params)": {
        "class": EfficientUNet,
        "path": "efficient_results/models/efficient_unet_63pct_rep0.pth",
    },
    "MicroCowUNet (191K params)": {
        "class": MicroCowUNet,
        "path": "micro_cow_results/models/microcow_unet_63pct_rep0.pth",
    },
    "NanoCowUNet (78K params)": {
        "class": NanoCowUNet,
        "path": "nano_cow_results/models/nanocow_unet_63pct_rep0.pth",
    },
    "PicoCowUNet (14.5K params)": {
        "class": PicoCowUNet,
        "path": "pico_cow_results/models/picocow_unet_63pct_rep0.pth",
    },
}


# =============================================
# HELPER FUNCTIONS
# =============================================
def load_model(model_class, checkpoint_path: str):
    """Load a model and its weights on CPU."""
    model = model_class().to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def measure_inference_time(model, input_tensor, num_runs: int = NUM_INFERENCE_RUNS):
    """Return (mean_ms, std_ms) over multiple runs."""
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    return float(np.mean(times)), float(np.std(times))


def estimate_memory_usage(model, input_tensor):
    """Return (model_size_MB, peak_delta_MB) using psutil."""
    # Model size (parameters only)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    with torch.no_grad():
        _ = model(input_tensor)  # forward pass

    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_delta_mb = mem_after - mem_before

    return param_size_mb, peak_delta_mb


# =============================================
# MAIN BENCHMARK
# =============================================
def run_model_benchmark():
    """Benchmark all four models on CPU."""
    # Find one test image
    images_dir = os.path.join("1k", "images")
    test_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not test_files:
        raise FileNotFoundError(f"No test image found in {images_dir}")

    test_image_path = os.path.join(images_dir, test_files[0])
    image = Image.open(test_image_path).convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)  # shape: (1, 3, 128, 128)

    print(f"Test image used: {test_image_path}")
    print(f"Input tensor shape: {input_tensor.shape}\n")
    print(f"Running {NUM_INFERENCE_RUNS} inference runs per model on CPU...\n")

    results = []

    for name, info in MODEL_PATHS.items():
        print(f"→ Loading {name} ...")
        try:
            model = load_model(info["class"], info["path"])
            param_count = sum(p.numel() for p in model.parameters())

            # Warm-up
            with torch.no_grad():
                for _ in range(NUM_WARMUP):
                    _ = model(input_tensor)

            # Measure time
            mean_time, std_time = measure_inference_time(model, input_tensor)

            # Measure memory
            model_size_mb, peak_delta_mb = estimate_memory_usage(model, input_tensor)

            results.append({
                "Model": name,
                "Params": f"{param_count / 1000:.1f}K",
                "Time (ms)": f"{mean_time:.2f} ± {std_time:.2f}",
                "Model Size (MB)": f"{model_size_mb:.2f}",
                "Peak Δ (MB)": f"{peak_delta_mb:.2f}"
            })

            print(f"   Time : {mean_time:.2f} ± {std_time:.2f} ms")
            print(f"   Mem  : {model_size_mb:.2f} MB (model) + {peak_delta_mb:.2f} MB (peak)\n")

        except Exception as e:
            print(f"   Failed: {e}\n")

    # Final summary table
    print("=" * 110)
    print("CPU INFERENCE BENCHMARK SUMMARY")
    print("=" * 110)
    print(f"{'Model':<32} {'Params':<10} {'Time (ms)':<20} {'Model Size (MB)':<18} {'Peak Δ (MB)':<15}")
    print("-" * 110)
    for r in results:
        print(f"{r['Model']:<32} {r['Params']:<10} {r['Time (ms)']:<20} "
              f"{r['Model Size (MB)']:<18} {r['Peak Δ (MB)']:<15}")
    print("=" * 110)
    print("All measurements performed on CPU with batch size 1.")


