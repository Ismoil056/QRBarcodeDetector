"""
QR Code and Barcode Detection System
=====================================
Author  : Student, Central Asian University
Module  : Image Processing
Date    : 2026

Description:
    This program detects and decodes QR codes and barcodes from:
      - Static image files
      - Real-time webcam video stream
    It draws bounding boxes around detected codes and displays
    the decoded information on screen.
"""

import cv2
import numpy as np
import time
import os


# ─────────────────────────────────────────────
#  Utility: draw detection result on a frame
# ─────────────────────────────────────────────
def draw_detection(frame, points, data, code_type="QR"):
    """
    Draw a polygon around the detected code and overlay decoded text.

    Parameters
    ----------
    frame      : numpy.ndarray  - the BGR image to draw on
    points     : numpy.ndarray  - corner points of the code boundary
    data       : str            - decoded string content
    code_type  : str            - 'QR' or 'Barcode'

    Returns
    -------
    frame      : numpy.ndarray  - annotated image
    """
    if points is not None and len(points) > 0:
        # Reshape to a list of integer (x, y) pairs
        pts = points.reshape(-1, 1, 2).astype(int)

        # Draw green polygon around the code
        cv2.polylines(frame, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=3)

        # Put the decoded text just above the code
        x, y = pts[0][0]
        label = f"[{code_type}] {data}"
        cv2.putText(frame, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)

    return frame


# ─────────────────────────────────────────────
#  Core: detect QR codes using OpenCV
# ─────────────────────────────────────────────
def detect_qr(frame):
    """
    Use OpenCV's built-in QRCodeDetector to find and decode
    all QR codes present in the given frame.

    Parameters
    ----------
    frame : numpy.ndarray  - BGR image

    Returns
    -------
    results : list of dict
        Each dict contains:
          'data'   (str)  - decoded text
          'points' (ndarray) - corner coordinates
          'type'   (str)  - always 'QR'
    """
    results = []
    detector = cv2.QRCodeDetector()

    # detectAndDecodeMulti finds all QR codes in one call
    ok, decoded_list, points_list, _ = detector.detectAndDecodeMulti(frame)

    if ok and points_list is not None:
        for data, points in zip(decoded_list, points_list):
            if data:   # skip empty / unreadable codes
                results.append({
                    "data"  : data,
                    "points": points,
                    "type"  : "QR"
                })

    return results


# ─────────────────────────────────────────────
#  Core: detect barcodes using WECHAT detector
# ─────────────────────────────────────────────
def detect_barcode(frame):
    """
    Use OpenCV's WeChatQRCode detector as a fallback for
    1-D barcodes and alternative QR decoders.

    NOTE: If WeChatQRCode models are not downloaded, this
    function falls back gracefully and returns an empty list.

    Parameters
    ----------
    frame : numpy.ndarray  - BGR image

    Returns
    -------
    results : list of dict  - same structure as detect_qr()
    """
    results = []
    try:
        # Try to use the more powerful detector (requires model files)
        detector = cv2.wechat_qrcode_WeChatQRCode()
        texts, points_list = detector.detectAndDecode(frame)
        for data, points in zip(texts, points_list):
            if data:
                results.append({
                    "data"  : data,
                    "points": points,
                    "type"  : "Barcode"
                })
    except Exception:
        # WeChatQRCode not available; skip silently
        pass

    return results


# ─────────────────────────────────────────────
#  Mode 1: Process a single image file
# ─────────────────────────────────────────────
def process_image(image_path):
    """
    Load an image from disk, detect all QR codes and barcodes,
    annotate the result, and save/display it.

    Parameters
    ----------
    image_path : str  - path to the input image

    Returns
    -------
    annotated  : numpy.ndarray  - image with detections drawn
    found      : list of dict   - all detected codes
    """
    # ── 1. Load the image ──────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot open image: {image_path}")
        return None, []

    print(f"[INFO] Image loaded: {image_path}  "
          f"({frame.shape[1]}x{frame.shape[0]} px)")

    # ── 2. Detect QR codes ─────────────────────────────────────
    found = detect_qr(frame)

    # ── 3. Detect barcodes (fallback) ─────────────────────────
    found += detect_barcode(frame)

    # ── 4. Annotate each detection on the frame ────────────────
    for item in found:
        frame = draw_detection(frame, item["points"],
                               item["data"], item["type"])
        print(f"  ✔  [{item['type']}] Decoded: {item['data']}")

    if not found:
        print("  ✘  No QR codes or barcodes detected.")

    # ── 5. Save annotated image ────────────────────────────────
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    base_name  = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(out_dir, f"{base_name}_detected.jpg")
    cv2.imwrite(output_path, frame)
    print(f"[INFO] Saved result → {output_path}")

    return frame, found


# ─────────────────────────────────────────────
#  Mode 2: Real-time webcam stream
# ─────────────────────────────────────────────
def run_webcam(camera_index=0):
    """
    Open the webcam and continuously detect QR codes / barcodes
    in real time.  Press 'q' to quit.

    Parameters
    ----------
    camera_index : int  - 0 for default webcam, 1 for external
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}")
        return

    print("[INFO] Webcam opened.  Press 'q' to exit.")

    prev_time = time.time()   # used to calculate FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame; retrying…")
            continue

        # ── Detect codes in current frame ──────────────────────
        found = detect_qr(frame)
        found += detect_barcode(frame)

        # ── Annotate ───────────────────────────────────────────
        for item in found:
            frame = draw_detection(frame, item["points"],
                                   item["data"], item["type"])

        # ── Show FPS in top-left corner ────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        # ── Status message ─────────────────────────────────────
        status = f"{len(found)} code(s) detected" if found else "Scanning…"
        cv2.putText(frame, status,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 100), 2)

        # ── Display ────────────────────────────────────────────
        cv2.imshow("QR & Barcode Detector  [Press Q to quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stream closed.")


# ─────────────────────────────────────────────
#  Mode 3: Evaluate on a folder of test images
# ─────────────────────────────────────────────
def evaluate_dataset(folder_path):
    """
    Run detection on every image in a folder and print a
    summary table of results (useful for the report metrics).

    Parameters
    ----------
    folder_path : str  - directory containing test images

    Returns
    -------
    metrics : dict  - total, detected, accuracy
    """
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    ]

    if not images:
        print(f"[WARNING] No images found in: {folder_path}")
        return {}

    total    = len(images)
    detected = 0

    print(f"\n{'='*55}")
    print(f"  Evaluating {total} images in: {folder_path}")
    print(f"{'='*55}")
    print(f"  {'#':>3}  {'Image':<30}  {'Codes found':>12}")
    print(f"  {'-'*50}")

    for idx, img_path in enumerate(sorted(images), start=1):
        _, found = process_image(img_path)
        count = len(found)
        if count > 0:
            detected += 1
        name = os.path.basename(img_path)[:28]
        print(f"  {idx:>3}  {name:<30}  {count:>12}")

    accuracy = detected / total * 100 if total else 0
    print(f"  {'-'*50}")
    print(f"  Detection rate: {detected}/{total}  ({accuracy:.1f}%)")
    print(f"{'='*55}\n")

    return {"total": total, "detected": detected, "accuracy": accuracy}


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default: run webcam demo
        print("Usage:")
        print("  python qr_barcode_detector.py webcam")
        print("  python qr_barcode_detector.py image  <path/to/image.jpg>")
        print("  python qr_barcode_detector.py folder <path/to/dataset/>")
        print("\nStarting webcam by default…")
        run_webcam()

    elif sys.argv[1] == "webcam":
        run_webcam()

    elif sys.argv[1] == "image" and len(sys.argv) >= 3:
        process_image(sys.argv[2])

    elif sys.argv[1] == "folder" and len(sys.argv) >= 3:
        evaluate_dataset(sys.argv[2])

    else:
        print("[ERROR] Unknown command. Run without arguments for help.")
