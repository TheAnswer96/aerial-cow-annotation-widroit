# Aerial Cow Annotations Tool

Active-learning annotation pipeline for cow segmentation in top-down drone imagery.
Combines SAM2 auto-labelling, human review, and a family of lightweight U-Net models
designed for edge deployment (Raspberry Pi → microcontroller).

---

## Repository structure

```
aerial-cow-annotations-tool/
├── tool/               Flask web app — 4-step annotation pipeline
│   ├── app.py
│   ├── requirements.txt
│   ├── tasks/
│   │   ├── sam2_runner.py   background: SAM2 (or Otsu fallback) per image
│   │   ├── trainer.py       background: PicoCowUNet training loop
│   │   └── predictor.py     background: batch inference
│   ├── templates/           Jinja2 HTML (step1–4 + base)
│   └── static/              CSS + SSE progress JS
│
└── experiments/        Standalone ML scripts (research)
    ├── extractor.py    COCO → 1 K image subset + binary masks
    ├── analysis.py     Image & mask quality metrics (blur, brightness, bbox)
    ├── unet.py         UNet           ~8 M params
    ├── unet800k.py     EfficientUNet  ~800 K params  (Pi Zero 2 W target)
    ├── unet200k.py     MicroCowUNet   ~191 K params  (depthwise-sep + additive skip)
    ├── unet80k.py      NanoCowUNet    ~78 K params   (SE attention + inverted residuals)
    ├── unet14k.py      PicoCowUNet    ~14.5 K params (MCU target, INT8-ready)
    ├── sam_comparison.py   SAM2 YOLO masks vs COCO ground truth metrics
    ├── sam_training.py     Filter SAM2 true-positives → retrain all models
    ├── inference_time.py   CPU inference benchmark (mean ± std ms, RAM delta)
    └── main.py             Experiment runner (uncomment to run each stage)
```

---

## Tool — 4-step annotation pipeline

```
Step 1  Upload a ZIP of drone images → pick segmentor model (Pico → UNet) → set seed percentage
Step 2  SAM2 auto-annotates the seed → review each mask (Accept / Reject / Reject all remaining)
Step 3  Train the chosen model on accepted masks → live loss curve  (GPU if available)
Step 4  Run trained model on remaining images → review predictions → export
```

### Quick start

```bash
cd tool
pip install -r requirements.txt
python app.py          # → http://localhost:5000
```

**SAM2.** Bundled via `ultralytics` (in `requirements.txt`). On the first SAM2 run the
weights auto-download into `tool/models/` (default `sam2_t.pt`, ~74 MB) — no manual setup.
Pick a larger model for quality:

```bash
export SAM2_MODEL=sam2_b.pt   # or sam2_s.pt / sam2_l.pt
```

If `ultralytics` is missing or the model can't load, the pipeline falls back to Otsu
thresholding — still reviewable.

Session workspaces are stored under `tool/workspace/{id}/` and are fully self-contained.

---

## Experiments

All scripts are self-contained with global config variables at the top (no CLI args).

### Dataset setup

```
raw/
  images/       full drone image set
  annotations/  single COCO JSON
```

```bash
cd experiments
pip install -r requirements.txt
# Edit extractor.py paths if needed, then:
python -c "from extractor import create_1k_subset; create_1k_subset()"
```

Output: `1k/images/`, `1k/masks/`, `1k/annotations/`.

### Run experiments

Uncomment the desired call in `main.py`:

```python
# unet.run_unet_experiments()          # ~8 M params baseline
# unet2.run_efficient_unet_experiments()
# unet3.run_micro_cow_unet_experiments()
# unet4.run_nano_cow_unet_experiments()
# unet5.run_pico_cow_unet_experiments()
# run_sam_comparison()
# generate_sam_tp_dataset(iou_threshold=0.25)
run_model_benchmark()
```

---

## Model family

| Model | Params | Target hardware | Key novelty |
|---|---|---|---|
| UNet | ~8 M | GPU / server | Baseline encoder-decoder |
| EfficientUNet | ~800 K | Raspberry Pi Zero 2 W | Reduced channels + BatchNorm |
| MicroCowUNet | ~191 K | Raspberry Pi Zero 2 W | Depthwise-sep convs, additive skip fusion, global objectness prior |
| NanoCowUNet | ~78 K | Raspberry Pi Zero 2 W | Inverted residuals, SE channel attention |
| PicoCowUNet | ~14.5 K | ESP32 / STM32 / RP2040 | Single encoder level, INT8-quantizable, peak activation < 25 KB |

All models trained at 128 × 128 input on a 1 K COCO-format drone image subset.

---

## SAM2 baseline results

Evaluated on 1 000 drone images (zero-shot, automatic mask generator):

| Metric | Mean | Std |
|---|---|---|
| IoU | 0.1503 | 0.1092 |
| Dice / F1 | 0.2466 | 0.1558 |
| Precision | 0.1556 | 0.1082 |
| Recall | 0.7153 | 0.3686 |

High recall, low precision → SAM2 over-segments. The annotation tool filters these with human review before training.

---

## Dataset

Images are **not included** (proprietary drone footage). The pipeline expects any COCO-format segmentation dataset with top-down livestock imagery.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{palazzetti2026drones,
  title     = {From Drones to Labels: A Semi-Automated Pipeline for Efficient Livestock Segmentation Annotation},
  author    = {Palazzetti, Lorenzo and Chen, Kuan-Ling and Casella, Enrico},
  booktitle = {2026 22nd International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)},
  year      = {2026},
}
```

---

## License

MIT
