import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
from PIL import Image

from texture_preprocessing import CLAHETransform, get_transforms
from model_factory import get_model

OUTPUT_DIR = r"c:\matress-project-matress\texture\experiment_results"
JSON_PATH = os.path.join(OUTPUT_DIR, "benchmark_metrics.json")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
SIDE_CAM_TEST_DIR = r"c:\matress-project-matress\side_camera\side_cam_final\results"

def generate_visualizations():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    sns.set_theme(style="whitegrid")
    
    # 1. Model Accuracy & Efficiency Comparison
    model_exps = [d for d in data if d["exp_id"].startswith("model_")]
    model_names = [d["model_name"] for d in model_exps]
    accs = [d["best_val_accuracy"] for d in model_exps]
    f1s = [d["macro_f1"] for d in model_exps]
    latencies = [d["latency_ms"] for d in model_exps]
    fps_vals = [d["throughput_fps"] for d in model_exps]
    params = [d["param_count_m"] for d in model_exps]
    
    # Chart 1: Model Accuracy & F1 Score Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, accs, width, label='Validation Accuracy (%)', color='#2b5c8f')
    plt.bar(x + width/2, [f*100 for f in f1s], width, label='Macro F1 Score (%)', color='#e76f51')
    plt.xticks(x, model_names, rotation=25, ha='right', fontsize=10)
    plt.ylabel('Percentage (%)')
    plt.title('Model Architecture Comparison: Accuracy & F1-Score', fontsize=13, fontweight='bold')
    plt.ylim(0, 110)
    for i in range(len(model_names)):
        plt.text(i - width/2, accs[i] + 1.5, f"{accs[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        plt.text(i + width/2, f1s[i]*100 + 1.5, f"{f1s[i]*100:.1f}%", ha='center', fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # Chart 2: Throughput (FPS) vs Accuracy Scatter Plot
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(fps_vals, accs, s=[max(50, p*20) for p in params], c=accs, cmap="viridis", alpha=0.9, edgecolors="black", linewidth=1.5)
    plt.colorbar(scatter, label="Accuracy (%)")
    for i, name in enumerate(model_names):
        plt.annotate(f"{name}\n({fps_vals[i]:.0f} FPS, {params[i]:.1f}M)", (fps_vals[i], accs[i]),
                     xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    plt.xlabel("Inference Throughput (FPS - Images / Sec)", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.title("Efficiency vs Accuracy Trade-off (Bubble size = Model Parameters)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "throughput_vs_accuracy.png"), dpi=300)
    plt.close()

    # Chart 3: Data Fraction Scaling Curve
    ds_exps = [d for d in data if d["exp_id"].startswith("datasize_")] + [d for d in data if d["exp_id"] == "model_efficientnet_b0"]
    ds_exps = sorted(ds_exps, key=lambda x: x["data_fraction"])
    fractions = [int(d["data_fraction"]*100) for d in ds_exps]
    ds_accs = [d["best_val_accuracy"] for d in ds_exps]
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(fractions, ds_accs, marker='o', linewidth=2.5, markersize=8, color='#2a9d8f')
    for f, a in zip(fractions, ds_accs):
        plt.annotate(f"{a:.1f}%", (f, a), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
    plt.xlabel("Training Data Fraction (%)", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.title("Impact of Dataset Size on Texture Classification Accuracy", fontsize=12, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_size_scaling.png"), dpi=300)
    plt.close()

    # Chart 4: Epochs Comparison Curves (Differentiated styles & markers so overlapping curves are visible)
    ep_exps = [d for d in data if d["exp_id"] in ["epochs_40", "model_efficientnet_b0", "epochs_10"]]
    # Sort so longest is drawn first as background, then shorter overlays with markers
    styles = {
        40: {"color": "#2ca02c", "linestyle": "-", "linewidth": 4.0, "alpha": 0.5, "marker": None, "label_suffix": "Full (40 Epochs)"},
        25: {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.2, "alpha": 0.9, "marker": "s", "markevery": 2, "label_suffix": "Standard (25 Epochs)"},
        10: {"color": "#d95f02", "linestyle": ":", "linewidth": 3.0, "alpha": 1.0, "marker": "o", "markevery": 1, "label_suffix": "Fast (10 Epochs)"}
    }
    
    plt.figure(figsize=(10, 5.5))
    for d in sorted(ep_exps, key=lambda x: -x["epochs"]):
        ep = d["epochs"]
        st = styles.get(ep, {"color": "gray", "linestyle": "-", "linewidth": 2, "alpha": 0.8, "marker": None, "label_suffix": ""})
        plt.plot(
            d["val_acc_hist"],
            label=f"{ep} Epochs - Best: {d['best_val_accuracy']:.1f}% [{st['label_suffix']}]",
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            alpha=st["alpha"],
            marker=st["marker"],
            markevery=st.get("markevery", 1),
            markersize=6
        )
    
    # Add vertical stop lines
    plt.axvline(x=9, color="#d95f02", linestyle=":", alpha=0.7, label="10-Epoch Run Terminated")
    plt.axvline(x=24, color="#1f77b4", linestyle="--", alpha=0.7, label="25-Epoch Run Terminated")
    
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.title("Validation Accuracy Convergence Across Epoch Configurations\n(Curves overlap during epochs 0-9 because they share identical initialization & seed)", fontsize=11, fontweight='bold')
    plt.legend(loc="lower right", framealpha=0.95)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "epochs_convergence.png"), dpi=300)
    plt.close()

    # Chart 5: Loss Algorithm Comparison
    loss_exps = [d for d in data if d["exp_id"].startswith("loss_")] + [d for d in data if d["exp_id"] == "model_efficientnet_b0"]
    loss_names = [d["loss_algo"] for d in loss_exps]
    loss_accs = [d["best_val_accuracy"] for d in loss_exps]
    loss_f1s = [d["macro_f1"] for d in loss_exps]
    
    plt.figure(figsize=(8, 4.5))
    x_loss = np.arange(len(loss_names))
    plt.bar(x_loss - 0.15, loss_accs, 0.3, label="Val Accuracy (%)", color='#457b9d')
    plt.bar(x_loss + 0.15, [f*100 for f in loss_f1s], 0.3, label="F1 Score (%)", color='#e63946')
    plt.xticks(x_loss, loss_names, fontsize=10)
    plt.ylabel("Percentage (%)")
    plt.title("Loss Algorithm Performance (CE vs Label Smoothing vs Focal Loss)", fontsize=12, fontweight='bold')
    for i in range(len(loss_names)):
        plt.text(i - 0.15, loss_accs[i] + 1, f"{loss_accs[i]:.1f}%", ha='center')
        plt.text(i + 0.15, loss_f1s[i]*100 + 1, f"{loss_f1s[i]*100:.1f}%", ha='center')
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_algorithm_comparison.png"), dpi=300)
    plt.close()

    # Chart 6: Preprocessing Pipeline Impact
    prep_exps = [d for d in data if d["exp_id"].startswith("prep_")] + [d for d in data if d["exp_id"] == "model_efficientnet_b0"]
    prep_names = [d["preprocessing"] for d in prep_exps]
    prep_accs = [d["best_val_accuracy"] for d in prep_exps]
    prep_f1s = [d["macro_f1"] for d in prep_exps]
    
    plt.figure(figsize=(9, 4.5))
    x_prep = np.arange(len(prep_names))
    plt.bar(x_prep - 0.15, prep_accs, 0.3, label="Val Accuracy (%)", color='#2a9d8f')
    plt.bar(x_prep + 0.15, [f*100 for f in prep_f1s], 0.3, label="F1 Score (%)", color='#f4a261')
    plt.xticks(x_prep, prep_names, fontsize=10, rotation=15, ha='right')
    plt.ylabel("Percentage (%)")
    plt.title("Impact of Preprocessing Pipelines (Standard vs CLAHE vs Edge-Sharpened)", fontsize=12, fontweight='bold')
    for i in range(len(prep_names)):
        plt.text(i - 0.15, prep_accs[i] + 1, f"{prep_accs[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        plt.text(i + 0.15, prep_f1s[i]*100 + 1, f"{prep_f1s[i]*100:.1f}%", ha='center', fontsize=9)
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "preprocessing_comparison.png"), dpi=300)
    plt.close()

    # Chart 7: Training & Validation Loss Curves Across Epochs (Distinct styling for overlapping lines)
    plt.figure(figsize=(10, 5.5))
    for d in sorted(ep_exps, key=lambda x: -x["epochs"]):
        ep = d["epochs"]
        st = styles.get(ep, {"color": "gray", "linestyle": "-", "linewidth": 2, "alpha": 0.8, "marker": None, "label_suffix": ""})
        plt.plot(
            d["val_loss_hist"],
            label=f"{ep} Epochs (Min Loss: {min(d['val_loss_hist']):.4f})",
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            alpha=st["alpha"],
            marker=st["marker"],
            markevery=st.get("markevery", 1),
            markersize=6
        )
    plt.axvline(x=9, color="#d95f02", linestyle=":", alpha=0.7, label="10-Epoch Run End")
    plt.axvline(x=24, color="#1f77b4", linestyle="--", alpha=0.7, label="25-Epoch Run End")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Loss", fontsize=11)
    plt.title("Validation Loss Convergence Curves Across Epoch Configurations", fontsize=12, fontweight='bold')
    plt.legend(loc="upper right", framealpha=0.95)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_convergence_curves.png"), dpi=300)
    plt.close()

    # Chart 8: Combined Confusion Matrix Grid for Top 6 Deep Learning Models
    top_models = model_exps[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    class_names = top_models[0]["class_names"]
    
    for i, m_exp in enumerate(top_models):
        cm = np.array(m_exp["cm"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f"{m_exp['model_name']}\nAcc: {m_exp['best_val_accuracy']:.1f}%", fontsize=11, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "combined_confusion_matrices.png"), dpi=300)
    plt.close()

    print("Successfully generated all comparison charts!")

def test_side_camera_captures():
    """Evaluates best model on real side camera captured images."""
    best_pth = os.path.join(OUTPUT_DIR, "model_efficientnet_b0_best.pth")
    if not os.path.exists(best_pth):
        return []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['Dual harmony', 'Gravite', 'Maxi plush', 'Maxi pro', 'Memorise', 'Ortholex', 'Purity plus', 'Velvet']
    
    model = get_model("efficientnet_b0", num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(best_pth, map_location=device))
    model = model.to(device)
    model.eval()
    
    _, val_tf = get_transforms("clahe_enhanced", image_size=224)
    
    results = []
    if os.path.exists(SIDE_CAM_TEST_DIR):
        for root, dirs, files in os.walk(SIDE_CAM_TEST_DIR):
            for f in files:
                if f.lower().endswith(('.jpg', '.png')) and ('corner' in f.lower() or 'bill' in f.lower()):
                    img_path = os.path.join(root, f)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        tensor = val_tf(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1)[0]
                            conf, pred_idx = torch.max(probs, dim=0)
                            
                        results.append({
                            "filename": f,
                            "rel_path": os.path.relpath(img_path, SIDE_CAM_TEST_DIR),
                            "predicted_class": class_names[pred_idx.item()],
                            "confidence": round(conf.item() * 100, 2)
                        })
                    except Exception as e:
                        pass
                    if len(results) >= 10:
                        break
            if len(results) >= 10:
                break
                
    return results

if __name__ == "__main__":
    generate_visualizations()
    side_results = test_side_camera_captures()
    print("Side Camera Test Results:", side_results)
