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

### All 15 Categories — 224px Resolution

| Category | Image AUROC | Pixel AUROC | F1 | Threshold |
|----------|------------:|------------:|---:|----------:|
| bottle | 1.0000 | 0.9452 | 1.0000 | 0.6492 |
| cable | 0.9983 | 0.9214 | 0.9840 | 0.7019 |
| capsule | 0.9458 | 0.9332 | 0.9636 | 0.4484 |
| carpet | 0.9952 | 0.9517 | 0.9778 | 0.6055 |
| grid | 0.9616 | 0.9653 | 0.9558 | 0.5670 |
| hazelnut | 1.0000 | 0.9504 | 1.0000 | 0.7932 |
| leather | 1.0000 | 0.9864 | 1.0000 | 0.6544 |
| metal_nut | 0.9995 | 0.9190 | 0.9947 | 0.6817 |
| pill | 0.9607 | 0.8400 | 0.9541 | 0.6071 |
| screw | 0.8832 | 0.8890 | 0.9143 | 0.5676 |
| tile | 0.9917 | 0.9189 | 0.9940 | 0.6999 |
| toothbrush | 0.9889 | 0.9268 | 0.9677 | 0.6100 |
| transistor | 0.9996 | 0.7878 | 0.9877 | 0.6932 |
| wood | 0.9816 | 0.9199 | 0.9587 | 0.7208 |
| zipper | 0.9648 | 0.9272 | 0.9551 | 0.5015 |
| **Mean** | **0.9781** | **0.9188** | **0.9738** | — |

---

### Resolution Experiment — 224px vs 384px (5 categories)

To understand the effect of input resolution, I ran a separate experiment on 5 categories at 384px.

| Category | Image AUROC 224px | Image AUROC 384px | Pixel AUROC 224px | Pixel AUROC 384px |
|----------|------------------:|------------------:|------------------:|------------------:|
| bottle | 1.0000 | 0.9992 | 0.9452 | 0.9648 |
| carpet | 0.9952 | 0.9976 | 0.9517 | 0.9587 |
| grid | 0.9616 | 0.9841 | 0.9653 | 0.9894 |
| capsule | 0.9458 | 0.9597 | 0.9332 | 0.9307 |
| transistor | 0.9996 | 0.9837 | 0.7878 | 0.7271 |

**Key takeaways:**
- 384px helped grid the most — pixel AUROC jumped from 0.9653 to 0.9894
- 224px was more stable for transistor — pixel AUROC actually dropped at 384px
- Bottle hit perfect image AUROC (1.0) at 224px
- Higher resolution doesn't always win — there's a real tradeoff depending on texture type

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
| bottle | 81,928 | 8,193 | 10x |
| cable | 87,808 | 8,781 | 10x |
| capsule | 85,848 | 8,585 | 10x |
| carpet | 109,760 | 10,976 | 10x |
| grid | 103,488 | 10,349 | 10x |
| hazelnut | 153,272 | 15,328 | 10x |
| leather | 96,040 | 9,604 | 10x |
| metal_nut | 86,240 | 8,624 | 10x |
| pill | 104,664 | 10,467 | 10x |
| screw | 125,440 | 12,544 | 10x |
| tile | 90,160 | 9,016 | 10x |
| toothbrush | 23,520 | 2,352 | 10x |
| transistor | 83,496 | 8,350 | 10x |
| wood | 96,824 | 9,683 | 10x |
| zipper | 94,080 | 9,408 | 10x |

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

# All 15 categories at once
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

I ran all 15 categories in the dataset. They cover a wide range of difficulty — from easy object-level defects (bottle, hazelnut) to hard texture-level ones (grid, screw) — which makes the benchmark a solid test of how well the approach actually generalizes.

---

## Why I built this

I wanted to build something that solves a real problem, not just a toy demo. Unsupervised anomaly detection is one of the most practical problems in industrial computer vision — you almost never have defect labels in production. This project is my attempt at building a clean, reproducible implementation that actually works on a standard benchmark and can be tested visually.

---

## License

MIT
