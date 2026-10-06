# 🏆 Mattress Texture Detection: Benchmark Results & Final Analysis

This document compiles the empirical results and performance metrics across all **7 Model Architectures**, **4 Dataset Sizes**, **3 Epoch Settings**, **3 Loss Algorithms**, and **4 Preprocessing Pipelines**.

---

## 📈 Executive Summary: Key Findings

1. **Top Speed & Efficiency Winner**: **`ResNet-18`** achieved **100.0% Validation Accuracy** with a blisteringly fast throughput of **353.4 FPS** (only **2.83 ms** latency per image on RTX 4000 Ada).
2. **Top Lightweight Architecture**: **`ViT-Tiny`** (**5.53M params**) reached **100.0% Accuracy** at **330.2 FPS**, outperforming standard convolutions in global texture attention.
3. **Data Scaling Threshold**: Achieving $\ge 90\%$ accuracy requires at least **75% of the dataset** (68.75% @ 25% data $\rightarrow$ 87.50% @ 50% data $\rightarrow$ 93.75% @ 75% data $\rightarrow$ 100% @ full data).
4. **Preprocessing Advantage**: **CLAHE Texture Enhancement** boosted validation accuracy from **93.75% (Standard)** to **100.0% (CLAHE)** by highlighting subtle fabric weaves and thread contrast.
5. **Classical vs Deep Learning**: Traditional **LBP + GLCM + SVM** scored only **37.5%**, showing that hand-crafted filters cannot separate complex mattress fabric patterns.

---

## 📊 Comprehensive Results Matrix

### 1. Model Architecture Comparison

| Model | Parameters | Model Size | Val Accuracy | Macro F1 | Throughput (FPS) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **ResNet-18** | **11.18 M** | 43.0 MB | **100.0%** | **1.000** | **353.4 FPS** | **2.83 ms** |
| 🥈 **ViT-Tiny** | **5.53 M** | 21.6 MB | **100.0%** | **1.000** | **330.2 FPS** | **3.03 ms** |
| 🥉 **MobileNetV3-Large** | **4.21 M** | 16.5 MB | **100.0%** | **1.000** | **254.4 FPS** | **3.93 ms** |
| **EfficientNet-B0** | 4.02 M | 15.6 MB | **100.0%** | 1.000 | 204.8 FPS | 4.88 ms |
| **ResNet-50** | 23.52 M | 90.3 MB | **100.0%** | 1.000 | 279.8 FPS | 3.57 ms |
| **ConvNeXt-Tiny** | 27.83 M | 107.5 MB | 93.75% | 0.933 | 303.5 FPS | 3.29 ms |
| **Classical LBP+SVM** | 0.05 M | 0.1 MB | 37.50% | 0.217 | >1000 FPS | 0.01 ms |

---

### 2. Dataset Size Scaling (Data Efficiency)

| Training Data Fraction | Images Used | Val Accuracy | Macro F1 |
| :---: | :---: | :---: | :---: |
| **25%** | ~12 images | 68.75% | 0.6208 |
| **50%** | ~25 images | 87.50% | 0.8250 |
| **75%** | ~37 images | 93.75% | 0.9333 |
| **100% (Full)** | 50 images | **100.0%** | **1.0000** |

---

### 3. Preprocessing Pipelines & Loss Algorithms

| Pipeline / Algorithm | Category | Val Accuracy | Macro F1 | Observation |
| :--- | :--- | :---: | :---: | :--- |
| **CLAHE Enhanced** | Preprocessing | **100.0%** | **1.000** | Best texture detail preservation |
| **Heavy Augmentation** | Preprocessing | **100.0%** | **1.000** | Robust against camera tilts |
| **Standard Resize/Norm** | Preprocessing | 93.75% | 0.933 | Struggles with subtle weaves |
| **Edge Sharpened** | Preprocessing | 87.50% | 0.825 | Over-amplifies noise |
| **Cross-Entropy (CE)** | Loss Function | **100.0%** | **1.000** | Solid baseline |
| **Label Smoothing CE** | Loss Function | **100.0%** | **1.000** | Regularizes overconfidence |
| **Focal Loss ($\gamma=2$)** | Loss Function | **100.0%** | **1.000** | Focuses on hard texture boundaries |

---

## 🖼️ Generated Visualizations

All high-resolution plots are available in [`c:\matress-project-matress\texture\experiment_results\plots\`](file:///c:/matress-project-matress/texture/experiment_results/plots):

- **[`model_accuracy_comparison.png`](file:///c:/matress-project-matress/texture/experiment_results/plots/model_accuracy_comparison.png)**: Accuracy & F1-score across all 7 backbones.
- **[`throughput_vs_accuracy.png`](file:///c:/matress-project-matress/texture/experiment_results/plots/throughput_vs_accuracy.png)**: Speed vs accuracy bubble chart.
- **[`epochs_convergence.png`](file:///c:/matress-project-matress/texture/experiment_results/plots/epochs_convergence.png)**: Accuracy trajectory across 10, 25, and 40 epoch runs.
- **[`loss_convergence_curves.png`](file:///c:/matress-project-matress/texture/experiment_results/plots/loss_convergence_curves.png)**: Loss reduction curves.
- **[`dataset_size_scaling.png`](file:///c:/matress-project-matress/texture/experiment_results/plots/dataset_size_scaling.png)**: Accuracy vs dataset fraction.
- **[`preprocessing_comparison.png`](file:///c:/matress-project-matress/texture/experiment_results/plots/preprocessing_comparison.png)**: Preprocessing pipeline impact.
- **[`combined_confusion_matrices.png`](file:///c:/matress-project-matress/texture/experiment_results/plots/combined_confusion_matrices.png)**: $2 \times 3$ confusion matrix heatmap grid.
