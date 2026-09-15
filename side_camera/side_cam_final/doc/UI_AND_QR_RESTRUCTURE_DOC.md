# UI Restructure & Multi-Orientation QR Scanner Architecture Document

## 1. Executive Summary

This document specifies the architectural changes to streamline the Master Dashboard UI, remove legacy dimension measurement code, enforce a default `PASS` for Texture AI, and implement 4-orientation scanning for sideways QR code detection.

---

## 2. Requirement Breakdown in Steps

### Step 1: Texture AI Always-PASS Display
- **Goal**: Update the **Cam 2: Texture Pattern AI** card on both the live dashboard UI and the Inspection Details Modal.
- **Behavior**: Instead of displaying classification predictions (e.g. `Maxi plush (18.15%)`), display `Variety: PASS` with `Confidence: 100%`.

### Step 2: Remove Measured vs Expected Dimensions Card
- **Goal**: Remove the `Measured vs Expected Dimensions` card (`TOP DIMS` / `#card-dimensions`) from:
  - Live Dashboard UI (`index.html` & `main.js`)
  - Inspection History Audit Trail Modal (`main.js`)

### Step 3: Remove Product & Batch Metadata Summary Card
- **Goal**: Remove the `Product & Batch Metadata` summary card (`#summary-card`) from the Inspection History Audit Trail Modal.

### Step 4: Remove Dimension Calculation Engine Entirely
- **Goal**: Remove dimension measurement logic and dimension tolerance checks across backend modules:
  - [`verification_engine.py`](file:///c:/matress-project-matress/side_camera/side_cam_final/verification_engine.py)
  - [`top_camera_module.py`](file:///c:/matress-project-matress/side_camera/side_cam_final/top_camera_module.py)
  - [`app.py`](file:///c:/matress-project-matress/side_camera/side_cam_final/app.py)
  - [`storage_manager.py`](file:///c:/matress-project-matress/side_camera/side_cam_final/storage_manager.py)

### Step 5: Multi-Orientation QR Code Detection (0°, 90°, 180°, 270°)
- **Goal**: Fix QR code detection failures when mattress stickers are rotated sideways ($90^\circ$ / $270^\circ$).
- **Behavior**: Update [`qr_module.py`](file:///c:/matress-project-matress/side_camera/side_cam_final/qr_module.py) (`scan_qr` & `detect_qr_presence`) to scan candidate rotations ($0^\circ$, $90^\circ$, $180^\circ$, $270^\circ$) and remap detected polygon coordinates seamlessly.

---

## 3. Revision History
- **Author**: Antigravity AI Pair Programmer
- **Date**: September 15, 2026
- **Target Files**: `qr_module.py`, `verification_engine.py`, `top_camera_module.py`, `texture_module.py`, `app.py`, `storage_manager.py`, `index.html`, `main.js`
