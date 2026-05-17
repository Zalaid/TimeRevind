#!/usr/bin/env python3
"""
Gender Detection Script
Detects persons via YOLO, classifies gender + age via InsightFace genderage model.
Shows live bounding boxes: Blue = Male, Pink = Female.

Usage:
    python gender_detect.py --video videos/test.mp4
    python gender_detect.py --video videos/test.mp4 --camera 0   (live camera)
"""

import cv2
import argparse
import sys
import os
from pathlib import Path

import torch
# Expose PyTorch CUDA DLLs so ONNX Runtime uses GPU
os.environ['PATH'] = os.path.join(os.path.dirname(torch.__file__), 'lib') + os.pathsep + os.environ.get('PATH', '')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from insightface.app import FaceAnalysis

# BGR colors
COLOR_MALE    = (219, 112, 50)   # Blue
COLOR_FEMALE  = (147, 20, 255)   # Pink / Magenta
COLOR_UNKNOWN = (128, 128, 128)  # Grey

GENDER_COLORS  = {"M": COLOR_MALE, "F": COLOR_FEMALE}
GENDER_LABELS  = {"M": "Male", "F": "Female"}


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    label_y = max(y1 - 6, th + 6)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def run(source, is_camera=False):
    print("Loading YOLO...")
    yolo = YOLO("yolov8n.pt")

    print("Loading InsightFace gender model...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        root=str(PROJECT_ROOT / "models"),   # uses local models/buffalo_l/
        allowed_modules=["detection", "genderage"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    source_label = f"Camera {source}" if is_camera else Path(str(source)).name
    print(f"\nSource : {source_label}")
    if not is_camera:
        print(f"Frames : {total} @ {fps:.1f} fps")
    print("Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame_display = frame.copy()

        # Detect persons only (class 0)
        yolo_results = yolo(frame, classes=[0], conf=0.35, verbose=False)

        if yolo_results[0].boxes is not None and len(yolo_results[0].boxes):
            for box in yolo_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                gender_key = None
                age = None

                try:
                    faces = face_app.get(crop)
                    if faces:
                        # Pick face with highest detection score
                        face = max(faces, key=lambda f: f.det_score)
                        gender_key = face.sex      # "M" or "F"
                        age = int(face.age)
                except Exception:
                    pass

                if gender_key in GENDER_COLORS:
                    color = GENDER_COLORS[gender_key]
                    label = f"{GENDER_LABELS[gender_key]}, {age}y" if age is not None else GENDER_LABELS[gender_key]
                else:
                    color = COLOR_UNKNOWN
                    label = "No face"

                draw_box(frame_display, x1, y1, x2, y2, label, color)

        # HUD overlay
        frame_info = f"Frame {frame_num}/{total}" if not is_camera else f"Frame {frame_num}"
        cv2.putText(frame_display, frame_info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

        # Legend
        cv2.rectangle(frame_display, (10, 40), (130, 80), (30, 30, 30), -1)
        cv2.putText(frame_display, "Male",   (30, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MALE,   1)
        cv2.putText(frame_display, "Female", (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FEMALE, 1)
        cv2.rectangle(frame_display, (14, 48), (26, 60), COLOR_MALE,   -1)
        cv2.rectangle(frame_display, (14, 65), (26, 77), COLOR_FEMALE, -1)

        cv2.imshow("Gender Detection  |  Q = quit", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_num} frames.")


def main():
    parser = argparse.ArgumentParser(description="Gender Detection with YOLO + InsightFace")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  type=str, help="Path to video file")
    group.add_argument("--camera", type=int, help="Camera device ID (e.g. 0)")
    args = parser.parse_args()

    if args.video:
        path = Path(args.video)
        if not path.exists():
            print(f"Error: Video not found: {path}")
            sys.exit(1)
        run(str(path), is_camera=False)
    else:
        run(args.camera, is_camera=True)


if __name__ == "__main__":
    main()
