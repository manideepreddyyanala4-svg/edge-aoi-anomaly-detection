# Edge AOI — Unsupervised Anomaly Detection for Industrial Inspection

A PatchCore-style anomaly detection system built for real manufacturing use cases. No defect labels needed — the model learns what "normal" looks like and flags anything that doesn't fit.

Built with PyTorch, WideResNet50, coreset memory banks, and a live Streamlit UI.

---

## What this project does

Most industrial anomaly detection research assumes you have labeled defect images. In practice, that's rarely true. Defects are rare, inconsistent, and expensive to annotate. Normal samples are easy to collect.

This project takes that constraint seriously. The pipeline:

1. Extracts patch-level features from normal training images using a pretrained WideResNet50
2. Stores those features in a memory bank on disk
3. At inference, compares each test image's patches against the memory bank using nearest-neighbor distance
4. Assigns an anomaly score and generates a localization heatmap

The result is a system that generalizes to unseen defect types without ever training on them.

---

## Results

### 224px Resolution

| Category | Image AUROC | Pixel AUROC | F1 | Threshold |
|----------|------------:|------------:|---:|----------:|
| bottle | 1.0000 | 0.9542 | 1.0000 | 0.6505 |
| carpet | 0.9960 | 0.9588 | 0.9834 | 0.6002 |
| grid | 0.9699 | 0.9639 | 0.9643 | 0.5659 |
| capsule | 0.9446 | 0.9321 | 0.9636 | 0.4484 |
| transistor | 0.9996 | 0.7702 | 0.9877 | 0.6932 |

### 384px Resolution

| Category | Image AUROC | Pixel AUROC | F1 | Threshold |
|----------|------------:|------------:|---:|----------:|
| bottle | 0.9992 | 0.9648 | 0.9921 | 0.7083 |
| carpet | 0.9976 | 0.9587 | 0.9944 | 0.5993 |
| grid | 0.9841 | 0.9894 | 0.9912 | 0.6757 |
| capsule | 0.9597 | 0.9307 | 0.9820 | 0.5749 |
| transistor | 0.9837 | 0.7271 | 0.9620 | 0.7534 |

**Key takeaways:**
- 384px made a big difference on grid — pixel AUROC jumped from 0.9639 to 0.9894
- 224px was more stable for transistor
- Bottle hit perfect image AUROC (1.0) at 224px
- There's a real tradeoff between resolution and robustness depending on the texture type

---

## Demo

### Broken Bottle — Correctly Detected

![Bottle UI](assets/screenshots/bottle_ui.png)

The heatmap highlights the broken area with red/orange. The rest of the bottle scores low (blue). No defect labels were used to produce this.

### Grid Example

![Grid UI](assets/screenshots/grid_ui.png)

### Transistor Example

![Transistor UI](assets/screenshots/transistor_ui.png)

### 224 vs 384 Resolution Comparison

![224](assets/comparisons/comparison_224.png)
![384](assets/comparisons/comparison_384.png)

---

## Architecture

![Architecture](assets/screenshots/architecture.png)

The backbone is WideResNet50_2 pretrained on ImageNet. Features are extracted from `layer2` and `layer3`, pooled to the same spatial resolution, concatenated, and L2-normalized. Each spatial location becomes a patch embedding.

At training time, all patch embeddings from normal images are collected into a memory bank. A greedy coreset algorithm then reduces the bank to ~10% of its original size while keeping the most representative vectors.

At inference, each patch is matched against the coreset via `torch.cdist`. The max patch distance becomes the anomaly score. Patch scores are reshaped into a grid, upsampled to image resolution, and smoothed with a Gaussian kernel.

### Memory Bank Compression

| Category | Full Bank | Coreset | Reduction |
|----------|----------:|--------:|----------:|
| bottle | 240,768 | 24,077 | 10x |
| carpet | 322,560 | 32,256 | 10x |
| grid | 304,128 | 30,413 | 10x |

10x smaller with negligible AUROC drop. The coreset keeps the full bank's coverage without the memory overhead.

---

## Project Structure

```
edge_aoi/
├── src/
│   ├── config.py          # All hyperparameters and paths
│   ├── models.py          # WideResNet feature extractor with hooks
│   ├── preprocess.py      # Image and mask transforms
│   ├── data_loader.py     # MVTec dataset loader
│   ├── build_memory.py    # Build baseline + coreset memory banks
│   ├── inference.py       # AnomalyDetector class
│   ├── evaluate.py        # Full evaluation pipeline
│   ├── metrics.py         # AUROC, AP, F1, confusion matrix
│   ├── visualization.py   # Heatmap, overlay, ROC/PR plots
│   ├── benchmark.py       # Latency benchmarking
│   └── ablation.py        # Backbone / bank / feature ablations
├── app/
│   └── ui.py              # Streamlit web UI
├── run_all.py             # Run all 5 categories end-to-end
└── requirements.txt
```

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/manideepreddyyanala4-svg/edge-aoi-anomaly-detection.git
cd edge-aoi-anomaly-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the MVTec AD dataset

Get it from [mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place it at:

```
data/mvtec/bottle/train/good/
data/mvtec/bottle/test/good/
data/mvtec/bottle/test/broken_large/
data/mvtec/bottle/ground_truth/broken_large/
...
```

### 4. Build memory banks

```bash
# Single category
python -m src.build_memory --category bottle

# All 5 categories at once
python run_all.py
```

### 5. Evaluate

```bash
python -m src.evaluate --category bottle
```

Outputs: Image AUROC, Pixel AUROC, F1, ROC/PR curve plots, false positive/negative examples, pixel-level CSV report.

### 6. Launch the UI

```bash
streamlit run app/ui.py
```

Upload any image and get an anomaly score + heatmap in real time.

### 7. Benchmark latency

```bash
python -m src.benchmark
```

### 8. Run ablation study

```bash
python -m src.ablation
```

Compares ResNet18 vs WideResNet50, baseline vs coreset, layer3-only vs layer2+layer3.

---

## Tech Stack

- **PyTorch** — feature extraction, coreset sampling, nearest-neighbor search
- **TorchVision** — WideResNet50_2 pretrained backbone
- **scikit-learn** — AUROC, AP, F1, threshold tuning
- **OpenCV + Matplotlib** — heatmap rendering and visualization
- **Streamlit** — live inspection UI
- **Pandas** — dataset inspection and CSV reporting

---

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — Paul Bergmann et al., CVPR 2019.

I used five categories: bottle, carpet, grid, capsule, transistor. These cover a good range of difficulty — from easy object-level defects (bottle) to hard texture-level ones (grid).

---

## Why I built this

I wanted to build something that solves a real problem, not just a toy demo. Unsupervised anomaly detection is one of the most practical problems in industrial computer vision — you almost never have defect labels in production. This project is my attempt at building a clean, reproducible implementation that actually works on a standard benchmark and can be tested visually.

---

## License

MIT
