"""
LIVE TEST: Real-time camera test for BehaviorDetector
Uses webcam + YOLOv11n-pose + BehaviorDetector
Shows detected behaviors on screen
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import logging

# Setup logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from .behavior_detector import BehaviorDetector
from .logger import BehaviorLogger

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def main():
    print("\n" + "="*70)
    print("LIVE CAMERA TEST: BehaviorDetector with YOLOv11n-pose")
    print("="*70)
    print("\nPress 'q' to quit")
    print("Press 's' to save detected behaviors to log\n")

    # ═══════════════════════════════════════════════════════════════
    # INITIALIZE
    # ═══════════════════════════════════════════════════════════════
    print("[INIT] Loading YOLOv11n-pose model...")
    try:
        model_path = r"d:\BSDS\8th Semester\Projects\TimeRevind\yolo11n-pose.pt"
        yolo = YOLO(model_path)
        print("[OK] YOLOv11n-pose loaded from: {}".format(model_path))
    except Exception as e:
        print("[ERROR] Failed to load YOLO: {}".format(e))
        return

    print("[INIT] Initializing BehaviorDetector...")
    try:
        detector = BehaviorDetector()
        print("[OK] BehaviorDetector initialized")
    except Exception as e:
        print("[ERROR] Failed to initialize BehaviorDetector: {}".format(e))
        return

    print("[INIT] Initializing logger...")
    logger = BehaviorLogger()
    logger.clear()
    print("[OK] Logger initialized")

    print("[INIT] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    print("[OK] Camera opened: {}x{} @ {} FPS".format(frame_width, frame_height, fps))

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════
    frame_count = 0
    total_people = set()

    print("\n" + "="*70)
    print("LIVE PREVIEW - Detecting behaviors...")
    print("="*70 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            frame_count += 1

            # ───────────────────────────────────────────────────────
            # YOLO DETECTION
            # ───────────────────────────────────────────────────────
            results = yolo(frame, verbose=False)

            if results[0].keypoints is None or len(results[0].keypoints.data) == 0:
                # No people detected
                cv2.putText(frame, "No people detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Process each detected person
                for person_idx in range(len(results[0].keypoints.data)):
                    box = results[0].boxes[person_idx]
                    keypoints = results[0].keypoints.data[person_idx]
                    track_id = int(box.id[0]) if box.id is not None else person_idx

                    # Generate person ID
                    person_id = "Person_{}".format(track_id)
                    total_people.add(person_id)

                    # Extract keypoints
                    kps = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints

                    # Extract bounding box
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # Crop person and face (simplified - use bbox as person crop)
                    person_crop = frame[max(0, y1):min(frame.shape[0], y2),
                                       max(0, x1):min(frame.shape[1], x2)]
                    face_crop = None  # Simplified - not extracting actual face

                    # ───────────────────────────────────────────────
                    # DETECT BEHAVIORS
                    # ───────────────────────────────────────────────
                    behaviors = detector.detect_all(person_id, kps, person_crop, face_crop)

                    # Get raw detections for immediate display
                    posture_raw = detector._detect_posture(kps)

                    # ───────────────────────────────────────────────
                    # DRAW BOUNDING BOX
                    # ───────────────────────────────────────────────
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, person_id, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # ───────────────────────────────────────────────
                    # DRAW KEYPOINTS
                    # ───────────────────────────────────────────────
                    for i, (x, y, conf) in enumerate(kps):
                        if conf > 0.3:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

                    # ───────────────────────────────────────────────
                    # DRAW POSTURE (raw detection before window voting)
                    # ───────────────────────────────────────────────
                    y_offset = y2 + 20
                    if posture_raw:
                        text = "POSTURE: {} ({:.0f}%)".format(
                            posture_raw.get("type"),
                            posture_raw.get("confidence", 0) * 100
                        )
                        cv2.putText(frame, text, (x1, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 25

                    # ───────────────────────────────────────────────
                    # DRAW BEHAVIORS (after window voting)
                    # ───────────────────────────────────────────────
                    if behaviors:
                        for behavior in behaviors:
                            text = "{}: {} ({:.0f}%)".format(
                                behavior.get("type"),
                                behavior.get("category"),
                                behavior.get("confidence", 0) * 100
                            )
                            cv2.putText(frame, text, (x1, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            y_offset += 20

                            # Log to file
                            logger.log_event(
                                person_id=person_id,
                                event_category=behavior.get("category"),
                                event_type=behavior.get("type"),
                                confidence=behavior.get("confidence"),
                                duration=behavior.get("duration"),
                                details=behavior.get("details")
                            )
                            print("[DETECTED] {}: {} @ {:.0f}%".format(
                                person_id,
                                behavior.get("type"),
                                behavior.get("confidence", 0) * 100
                            ))

            # ───────────────────────────────────────────────────────
            # DRAW FRAME INFO
            # ───────────────────────────────────────────────────────
            cv2.putText(frame, "Frame: {} | People: {}".format(frame_count, len(total_people)),
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ───────────────────────────────────────────────────────
            # SHOW FRAME
            # ───────────────────────────────────────────────────────
            cv2.imshow("BehaviorDetector Live Test", frame)

            # ───────────────────────────────────────────────────────
            # KEYBOARD CONTROL
            # ───────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[EXIT] User pressed 'q'")
                break
            elif key == ord('s'):
                # Save logs to file (already auto-saved, just confirm)
                logs = logger.get_logs()
                print("\n[SAVED] {} behaviors logged to behavior/logs/behavior_events.json".format(len(logs)))
                for log in logs[-5:]:
                    print("  - {}: {}".format(log['person_id'], log['event_type']))

    except KeyboardInterrupt:
        print("\n[EXIT] User interrupted with Ctrl+C")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ═══════════════════════════════════════════════════════════
        # CLEANUP & SUMMARY
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print("Frames processed: {}".format(frame_count))
        print("Total people detected: {}".format(len(total_people)))
        print("Behaviors logged: {}".format(len(logger.get_logs())))

        logs = logger.get_logs()
        if logs:
            print("\nLogged behaviors:")
            for log in logs:
                print("  - {}: {} ({})".format(
                    log['person_id'],
                    log['event_type'],
                    log['event_category']
                ))
        else:
            print("\nNo behaviors detected yet (skeleton stage)")

        # Cleanup
        for person_id in total_people:
            detector.cleanup(person_id)

        print("\n[OK] Test complete. Logs saved to: behavior/logs/behavior_events.json")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
