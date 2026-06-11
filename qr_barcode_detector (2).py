"""
QR Code and Barcode Detection System
Author: Ismoil Rahimov
Student ID: 230514
Module: Image Processing - CAU Spring 2026

This program can detect and read QR codes from:
  - image files
  - live webcam video

Just run it and pass the mode as an argument (see bottom of file).
"""

import cv2
import numpy as np
import time
import os


def draw_detection(frame, points, data, code_type="QR"):
    """
    Draw a green box around the detected code and show the decoded text.
    points = corner coordinates returned by the detector
    data   = the decoded string
    """
    if points is not None and len(points) > 0:
        pts = points.reshape(-1, 1, 2).astype(int)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # put the text just above the top-left corner
        x, y = pts[0][0]
        label = f"[{code_type}] {data}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return frame


def detect_qr(frame):
    """
    Use OpenCV's QRCodeDetector to find all QR codes in the frame.
    Returns a list of dicts with 'data', 'points', and 'type'.
    """
    results = []
    detector = cv2.QRCodeDetector()

    # detectAndDecodeMulti finds all codes in one go
    ok, decoded_list, points_list, _ = detector.detectAndDecodeMulti(frame)

    if ok and points_list is not None:
        for data, points in zip(decoded_list, points_list):
            if data:  # skip empty results
                results.append({"data": data, "points": points, "type": "QR"})

    return results


def detect_barcode(frame):
    """
    Try WeChatQRCode as a fallback for barcodes.
    This only works if the model files are downloaded, otherwise it just returns [].
    """
    results = []
    try:
        detector = cv2.wechat_qrcode_WeChatQRCode()
        texts, points_list = detector.detectAndDecode(frame)
        for data, points in zip(texts, points_list):
            if data:
                results.append({"data": data, "points": points, "type": "Barcode"})
    except Exception:
        pass  # WeChatQRCode not available, that's fine

    return results


def process_image(image_path):
    """
    Load an image, detect any QR codes in it, draw the results,
    and save the annotated image to the results/ folder.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not open image: {image_path}")
        return None, []

    print(f"Processing: {image_path}")

    found = detect_qr(frame)
    found += detect_barcode(frame)

    for item in found:
        frame = draw_detection(frame, item["points"], item["data"], item["type"])
        print(f"  Found [{item['type']}]: {item['data']}")

    if not found:
        print("  Nothing detected.")

    # save the result
    os.makedirs("results", exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join("results", f"{base}_detected.jpg")
    cv2.imwrite(out_path, frame)
    print(f"  Saved to: {out_path}")

    return frame, found


def run_webcam(camera_index=0):
    """
    Open the webcam and detect QR codes in real time.
    Press Q to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Webcam running. Press Q to stop.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        found = detect_qr(frame)
        found += detect_barcode(frame)

        for item in found:
            frame = draw_detection(frame, item["points"], item["data"], item["type"])

        # show FPS in the corner
        now = time.time()
        fps = 1.0 / max(now - prev_time, 0.001)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # status message
        msg = f"{len(found)} code(s) found" if found else "Scanning..."
        cv2.putText(frame, msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        cv2.imshow("QR Detector - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def evaluate_dataset(folder_path):
    """
    Run the detector on every image in a folder and print a summary.
    Useful for measuring accuracy.
    """
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    ]

    if not images:
        print(f"No images found in: {folder_path}")
        return

    total = len(images)
    detected = 0

    print(f"\nRunning on {total} images in: {folder_path}")
    print("-" * 50)

    for i, img_path in enumerate(sorted(images), 1):
        _, found = process_image(img_path)
        if found:
            detected += 1
        print(f"  [{i}/{total}] {os.path.basename(img_path)} — {len(found)} code(s)")

    acc = detected / total * 100 if total else 0
    print("-" * 50)
    print(f"Result: {detected}/{total} images detected ({acc:.1f}%)\n")


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python qr_barcode_detector.py webcam")
        print("  python qr_barcode_detector.py image  path/to/image.jpg")
        print("  python qr_barcode_detector.py folder path/to/dataset/")
        print("\nStarting webcam by default...")
        run_webcam()

    elif sys.argv[1] == "webcam":
        run_webcam()

    elif sys.argv[1] == "image" and len(sys.argv) >= 3:
        process_image(sys.argv[2])

    elif sys.argv[1] == "folder" and len(sys.argv) >= 3:
        evaluate_dataset(sys.argv[2])

    else:
        print("Unknown command. Run without arguments for help.")
