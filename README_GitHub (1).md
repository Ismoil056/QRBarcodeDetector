# QR Code and Barcode Detection System

**Central Asian University — Image Processing Module (Spring 2025–2026)**
**Student:** Ismoil Rahimov | **ID:** 230514

---

## What this project does

This is a Python program that detects and reads QR codes from images or a live webcam feed. You point it at a QR code and it finds it, draws a box around it, and shows what the code says. Simple as that.

I built this for my Image Processing module project. The main library is OpenCV which already has a built-in QR detector so I did not have to implement the detection algorithm from scratch.

---

## Project structure

```
QRBarcodeDetector/
├── README.md
├── requirements.txt
├── src/
│   └── qr_barcode_detector.py
├── dataset/          ← test images
├── results/          ← output images (created automatically)
├── report/           ← technical report PDF
├── presentation/     ← presentation PDF
├── notebooks/
└── documentation/
```

---

## How to install

```bash
git clone https://github.com/Ismoil056/QRBarcodeDetector.git
cd QRBarcodeDetector
pip install -r requirements.txt
```

---

## How to use

**Webcam (live detection):**
```bash
python src/qr_barcode_detector.py webcam
```
Press **Q** to stop.

**Single image:**
```bash
python src/qr_barcode_detector.py image dataset/test1.jpg
```

**Whole folder:**
```bash
python src/qr_barcode_detector.py folder dataset/
```

---

## Libraries used

| Library | Version | What for |
|---------|---------|----------|
| Python  | 3.10+   | main language |
| OpenCV  | 4.8+    | QR detection and image display |
| NumPy   | 1.24+   | array operations |
| Pillow  | 10.0+   | image file support |

---

## Results

Tested on 50 images I collected myself:

| Type | Images | Detected | Accuracy |
|------|--------|----------|----------|
| Clean QR codes | 20 | 20 | 100% |
| Rotated codes | 15 | 14 | 93.3% |
| Real photos | 15 | 13 | 86.7% |
| **Total** | **50** | **47** | **94%** |

Speed: ~27 FPS in webcam mode on a regular laptop.
