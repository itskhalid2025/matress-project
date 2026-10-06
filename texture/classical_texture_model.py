import os
import time
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def extract_texture_features(image_path):
    """
    Extracts LBP texture histogram, GLCM contrast/homogeneity/energy/correlation features,
    and RGB color moments from a given image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Local Binary Patterns (LBP)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    # 2. GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, 'contrast').ravel()
    glcm_dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    glcm_homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    glcm_energy = graycoprops(glcm, 'energy').ravel()
    glcm_correlation = graycoprops(glcm, 'correlation').ravel()
    
    # 3. Color Moments (Mean & Std in BGR)
    mean, std = cv2.meanStdDev(img)
    color_features = np.concatenate([mean.ravel(), std.ravel()])
    
    feature_vector = np.hstack([
        lbp_hist,
        glcm_contrast,
        glcm_dissimilarity,
        glcm_homogeneity,
        glcm_energy,
        glcm_correlation,
        color_features
    ])
    
    return feature_vector

def train_eval_classical_texture_model(train_dir, val_dir, classifier_type="svm"):
    """
    Trains and evaluates a classical ML texture model (LBP + GLCM + Color + SVM/RandomForest).
    """
    X_train, y_train = [], []
    classes = sorted(os.listdir(train_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_p = os.path.join(cls_path, fname)
                feats = extract_texture_features(img_p)
                X_train.append(feats)
                y_train.append(class_to_idx[cls])
                
    X_val, y_val = [], []
    for cls in classes:
        cls_path = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_p = os.path.join(cls_path, fname)
                feats = extract_texture_features(img_p)
                X_val.append(feats)
                y_val.append(class_to_idx[cls])
                
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    
    if classifier_type == "svm":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale")
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    start_infer = time.time()
    y_pred = clf.predict(X_val)
    infer_time = (time.time() - start_infer) / len(X_val) * 1000.0  # ms per image
    
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=classes, digits=4, output_dict=True)
    
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "train_time_sec": train_time,
        "latency_ms": infer_time,
        "y_true": y_val,
        "y_pred": y_pred,
        "classes": classes
    }
