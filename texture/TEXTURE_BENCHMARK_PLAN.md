# 🧵 Mattress Texture Detection: Model Comparison & Benchmark Suite

This document defines the complete experimental matrix, model architectures, loss functions, preprocessing pipelines, evaluation protocols, and **all generated visualization graphs** for evaluating mattress fabric texture detection.

---

## 📸 Dataset & Evaluation Overview

- **Primary Texture Dataset**: [`c:\matress-project-matress\texture\Mattress`](file:///c:/matress-project-matress/texture/Mattress)
  - **8 Classes**: *Dual harmony*, *Gravite*, *Maxi plush*, *Maxi pro*, *Memorise*, *Ortholex*, *Purity plus*, *Velvet*.
- **Industrial Camera Test Images**: [`c:\matress-project-matress\side_camera\side_cam_final\results`](file:///c:/matress-project-matress/side_camera/side_cam_final/results)
  - Evaluates real-time side-camera captures under conveyor-belt illumination.

---

## 🧠 Model Architectures Selected & Rationale

| Model Architecture | Parameter Count | Architectural Focus | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | ~5.3M Params | Compound Scaling CNN | Baseline deep transfer model; highly efficient parameter usage. |
| **ResNet-18** | ~11.7M Params | Shallow Residual CNN | Fast convergence with residual connections; prevents overfitting on small datasets. |
| **ResNet-50** | ~25.6M Params | Deep Residual CNN | High feature capacity for capturing intricate multi-layered fabric patterns. |
| **MobileNetV3-Large** | ~5.4M Params | Ultra-Fast Edge Mobile CNN | Industrial edge deployment candidate for high-speed line inspection. |
| **ConvNeXt-Tiny** | ~28.6M Params | Modernized ConvNet | Combines Vision Transformer design principles with pure convolutional execution. |
| **ViT-Tiny (Vision Transformer)** | ~5.7M Params | Patch Self-Attention | Learns global patch-to-patch texture relationships rather than local fields. |
| **Classical LBP + GLCM + SVM** | ~0.05M Params | Handcrafted Feature Baseline | Uses **Local Binary Patterns (LBP)**, **GLCM matrices**, & Color Moments with SVM. |

---

## 🧮 Error Algorithms & Loss Functions

1. **Standard Cross-Entropy Loss (`cross_entropy`)**:
   $$\mathcal{L}_{CE} = -\sum_{i=1}^N y_i \log(\hat{y}_i)$$
   *Baseline multi-class classification metric.*

2. **Label Smoothing Cross-Entropy (`label_smoothing`, $\epsilon=0.1$)**:
   $$\mathcal{L}_{LS} = (1 - \epsilon) \mathcal{L}_{CE} + \frac{\epsilon}{K} \sum_{k=1}^K -\log(\hat{y}_k)$$
   *Softens target targets to prevent model overconfidence on small datasets.*

3. **Focal Loss (`focal_loss`, $\gamma=2.0$)**:
   $$\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   *Downweights easy fabric samples and concentrates gradients on hard texture boundary samples.*

---

## ⚙️ Experimental Parameters & Sweeps

### 1. Training Epoch Configurations
- **10 Epochs**: Fast convergence and under-fitting baseline.
- **25 Epochs**: Standard transfer learning fine-tuning window.
- **40 Epochs**: Extended training with `ReduceLROnPlateau` scheduler.

### 2. Dataset Scaling Fractions
- **25%** Training Data Split
- **50%** Training Data Split
- **75%** Training Data Split
- **100%** Full Training Set

### 3. Preprocessing & Texture Pipelines
- `standard`: Resize (224x224) + ImageNet Normalization.
- `clahe_enhanced`: LAB color space conversion + **Contrast Limited Adaptive Histogram Equalization** on L-channel to highlight fine thread weaves.
- `edge_sharpened`: Unsharp masking + Gaussian edge filtering.
- `heavy_aug`: Random crop, perspective warp, Gaussian blur, and color jitter.

---

## 📈 Visual Graphs & Charts Automatically Produced

Running `generate_report_visuals.py` automatically generates publication-ready high-resolution plots in [`c:\matress-project-matress\texture\experiment_results\plots\`](file:///c:/matress-project-matress/texture/experiment_results/plots):

| Output File Name | Plot Type | What It Shows |
| :--- | :--- | :--- |
| **`model_accuracy_comparison.png`** | Grouped Bar Chart | Side-by-side Top-1 Accuracy (%) and Macro F1-Score (%) for all 7 models. |
| **`throughput_vs_accuracy.png`** | 2D Bubble Scatter Plot | Efficiency vs Accuracy trade-off (FPS on X-axis, Accuracy on Y-axis, bubble size = parameters). |
| **`dataset_size_scaling.png`** | Line Graph | Accuracy curve vs data size (25%, 50%, 75%, 100%) to measure data efficiency. |
| **`epochs_convergence.png`** | Epoch Curve | Validation Accuracy progression over time (10 vs 25 vs 40 epochs). |
| **`loss_convergence_curves.png`** | Epoch Curve | Validation Loss reduction and convergence curves across training durations. |
| **`loss_algorithm_comparison.png`** | Bar Chart | Impact of loss function (Standard CE vs Label Smoothing vs Focal Loss). |
| **`preprocessing_comparison.png`** | Bar Chart | Performance gain of CLAHE texture enhancement vs edge-sharpening vs standard. |
| **`combined_confusion_matrices.png`** | 2x3 Heatmap Grid | Side-by-side confusion matrices for all top Deep Learning models. |
| **`confusion_matrices/<exp_id>_cm.png`** | Individual Heatmaps | Full-resolution confusion matrix with exact per-class true/predicted counts for every single run. |

---

## 🚀 Execution Instructions

To launch the benchmark execution and generate all plots & reports:

```bash
cd c:\matress-project-matress\texture
python -u benchmark_suite.py
python generate_report_visuals.py
```
