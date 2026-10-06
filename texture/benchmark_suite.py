import os
import time
import json
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from texture_preprocessing import get_transforms
from loss_functions import get_loss_function
from model_factory import get_model, count_parameters
from classical_texture_model import train_eval_classical_texture_model

# Config
TRAIN_DIR = r"c:\matress-project-matress\texture\dataset_split\train"
VAL_DIR = r"c:\matress-project-matress\texture\dataset_split\val"
OUTPUT_DIR = r"c:\matress-project-matress\texture\experiment_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

def measure_inference_speed(model, sample_input, device=DEVICE, num_iterations=100):
    """Measures single image latency (ms) and throughput (FPS)."""
    model.eval()
    sample_input = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_latency_ms = (total_time / num_iterations) * 1000.0
    fps = num_iterations / total_time
    return avg_latency_ms, fps

def run_single_experiment(
    exp_id,
    model_name="efficientnet_b0",
    epochs=25,
    data_fraction=1.0,
    loss_algo="cross_entropy",
    preprocessing="clahe_enhanced",
    batch_size=8,
    lr=1e-4
):
    print(f"\n==================================================", flush=True)
    print(f"RUNNING EXP [{exp_id}]: Model={model_name}, Data={int(data_fraction*100)}%, Epochs={epochs}, Loss={loss_algo}, Prep={preprocessing}", flush=True)
    print(f"==================================================", flush=True)
    set_seed(SEED)
    
    train_tf, val_tf = get_transforms(preprocessing, image_size=224)
    
    full_train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)
    class_names = full_train_ds.classes
    num_classes = len(class_names)
    
    # Subset train dataset if requested fraction < 1.0
    if data_fraction < 1.0:
        indices = list(range(len(full_train_ds)))
        random.shuffle(indices)
        subset_size = max(num_classes, int(len(full_train_ds) * data_fraction))
        subset_indices = indices[:subset_size]
        train_ds = Subset(full_train_ds, subset_indices)
    else:
        train_ds = full_train_ds
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = get_model(model_name, num_classes=num_classes, pretrained=True).to(DEVICE)
    param_count_m = count_parameters(model)
    criterion = get_loss_function(loss_algo)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []
    
    start_train_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * len(lbls)
            _, preds = torch.max(out, 1)
            t_correct += (preds == lbls).sum().item()
            t_total += len(lbls)
            
        t_loss /= t_total
        t_acc = t_correct / t_total
        
        # Eval
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * len(lbls)
                _, preds = torch.max(out, 1)
                v_correct += (preds == lbls).sum().item()
                v_total += len(lbls)
                
        v_loss /= v_total
        v_acc = v_correct / v_total
        
        train_loss_hist.append(t_loss)
        val_loss_hist.append(v_loss)
        train_acc_hist.append(t_acc)
        val_acc_hist.append(v_acc)
        
        scheduler.step(v_acc)
        if v_acc > best_acc:
            best_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] -> Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.1f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.1f}% (Best: {best_acc*100:.1f}%)", flush=True)
            
    total_train_sec = time.time() - start_train_time
    
    # Load best state
    model.load_state_dict(best_weights)
    model.eval()
    
    # Save model weights & estimate size
    model_pth = os.path.join(OUTPUT_DIR, f"{exp_id}_best.pth")
    torch.save(best_weights, model_pth)
    model_size_mb = os.path.getsize(model_pth) / (1024 * 1024)
    
    # Evaluation on Val dataset
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            y_true.extend(lbls.numpy())
            y_pred.extend(preds.cpu().numpy())
            
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Measure Latency & FPS
    sample_input = torch.randn(1, 3, 224, 224)
    latency_ms, fps = measure_inference_speed(model, sample_input, DEVICE)
    
    # Plot & Save Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {exp_id}\nAcc: {best_acc*100:.1f}% | F1: {f1:.3f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices", f"{exp_id}_cm.png")
    plt.savefig(cm_path)
    plt.close()
    
    metrics = {
        "exp_id": exp_id,
        "model_name": model_name,
        "data_fraction": data_fraction,
        "epochs": epochs,
        "loss_algo": loss_algo,
        "preprocessing": preprocessing,
        "param_count_m": round(param_count_m, 2),
        "model_size_mb": round(model_size_mb, 2),
        "best_val_accuracy": round(best_acc * 100, 2),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "latency_ms": round(latency_ms, 2),
        "throughput_fps": round(fps, 1),
        "total_train_sec": round(total_train_sec, 2),
        "train_loss_hist": [round(x, 4) for x in train_loss_hist],
        "val_loss_hist": [round(x, 4) for x in val_loss_hist],
        "train_acc_hist": [round(x*100, 2) for x in train_acc_hist],
        "val_acc_hist": [round(x*100, 2) for x in val_acc_hist],
        "cm": cm.tolist(),
        "class_names": class_names
    }
    
    return metrics

def run_full_benchmark_suite():
    all_results = []
    
    # ==========================================================
    # 1. MODEL ARCHITECTURE COMPARISON (Full data, 25 epochs, CLAHE prep, CE loss)
    # ==========================================================
    model_list = [
        "efficientnet_b0",
        "resnet18",
        "resnet50",
        "mobilenet_v3_large",
        "convnext_tiny",
        "vit_tiny"
    ]
    
    for m in model_list:
        exp_id = f"model_{m}"
        res = run_single_experiment(
            exp_id=exp_id,
            model_name=m,
            epochs=25,
            data_fraction=1.0,
            loss_algo="cross_entropy",
            preprocessing="clahe_enhanced"
        )
        all_results.append(res)
        
    # Classical LBP+SVM Baseline
    print("\nEvaluating Classical LBP + SVM Baseline...")
    class_res = train_eval_classical_texture_model(TRAIN_DIR, VAL_DIR, classifier_type="svm")
    cm_class = class_res["confusion_matrix"]
    prec, rec, f1, _ = precision_recall_fscore_support(class_res["y_true"], class_res["y_pred"], average="macro", zero_division=0)
    
    classical_metrics = {
        "exp_id": "model_classical_lbp_svm",
        "model_name": "classical_lbp_svm",
        "data_fraction": 1.0,
        "epochs": 0,
        "loss_algo": "n/a",
        "preprocessing": "lbp_glcm_color",
        "param_count_m": 0.05,
        "model_size_mb": 0.1,
        "best_val_accuracy": round(class_res["accuracy"] * 100, 2),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "latency_ms": round(class_res["latency_ms"], 2),
        "throughput_fps": round(1000.0 / (class_res["latency_ms"] + 1e-5), 1),
        "total_train_sec": round(class_res["train_time_sec"], 2),
        "train_loss_hist": [],
        "val_loss_hist": [],
        "train_acc_hist": [],
        "val_acc_hist": [],
        "cm": cm_class.tolist(),
        "class_names": class_res["classes"]
    }
    all_results.append(classical_metrics)

    # Save Classical CM
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_class, annot=True, fmt="d", cmap="Blues", xticklabels=class_res["classes"], yticklabels=class_res["classes"])
    plt.title(f"Confusion Matrix: classical_lbp_svm\nAcc: {class_res['accuracy']*100:.1f}% | F1: {f1:.3f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices", "model_classical_lbp_svm_cm.png"))
    plt.close()

    # ==========================================================
    # 2. TRAINING DATA SIZE COMPARISON (EfficientNet-B0, 25 epochs, CLAHE prep)
    # ==========================================================
    data_sizes = [0.25, 0.50, 0.75]
    for ds_frac in data_sizes:
        exp_id = f"datasize_{int(ds_frac*100)}pct"
        res = run_single_experiment(
            exp_id=exp_id,
            model_name="efficientnet_b0",
            epochs=25,
            data_fraction=ds_frac,
            loss_algo="cross_entropy",
            preprocessing="clahe_enhanced"
        )
        all_results.append(res)
        
    # ==========================================================
    # 3. EPOCHS COMPARISON (EfficientNet-B0, full data)
    # ==========================================================
    epoch_sizes = [10, 40]
    for ep in epoch_sizes:
        exp_id = f"epochs_{ep}"
        res = run_single_experiment(
            exp_id=exp_id,
            model_name="efficientnet_b0",
            epochs=ep,
            data_fraction=1.0,
            loss_algo="cross_entropy",
            preprocessing="clahe_enhanced"
        )
        all_results.append(res)
        
    # ==========================================================
    # 4. LOSS ALGORITHM COMPARISON (EfficientNet-B0, 25 epochs)
    # ==========================================================
    loss_algos = ["label_smoothing", "focal_loss"]
    for loss_name in loss_algos:
        exp_id = f"loss_{loss_name}"
        res = run_single_experiment(
            exp_id=exp_id,
            model_name="efficientnet_b0",
            epochs=25,
            data_fraction=1.0,
            loss_algo=loss_name,
            preprocessing="clahe_enhanced"
        )
        all_results.append(res)

    # ==========================================================
    # 5. PREPROCESSING PIPELINE COMPARISON (EfficientNet-B0, 25 epochs)
    # ==========================================================
    preps = ["standard", "heavy_aug", "edge_sharpened"]
    for prep_name in preps:
        exp_id = f"prep_{prep_name}"
        res = run_single_experiment(
            exp_id=exp_id,
            model_name="efficientnet_b0",
            epochs=25,
            data_fraction=1.0,
            loss_algo="cross_entropy",
            preprocessing=prep_name
        )
        all_results.append(res)
        
    # Save all raw metrics JSON
    json_path = os.path.join(OUTPUT_DIR, "benchmark_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n==================================================")
    print(f"BENCHMARK COMPLETE! Saved raw results to {json_path}")
    print(f"==================================================")
    return all_results

if __name__ == "__main__":
    run_full_benchmark_suite()
