"""
STEP 1 TEST: Verify behavior config loads correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import (
    YOLO_MODEL,
    EMOTION_WINDOW_SIZE,
    POSTURE_WINDOW_SIZE,
    ACTIVITY_WINDOW_SIZE,
    GESTURE_WINDOW_SIZE,
    CONFIDENCE_EMOTION,
    CONFIDENCE_POSTURE,
    MIN_DURATION_EMOTION,
    COOLDOWN_EMOTION,
    COOLDOWN_ACTIVITY,
    FALL_VELOCITY_THRESHOLD,
)


def test_config():
    print("\n" + "="*60)
    print("STEP 1: Testing Behavior Config")
    print("="*60)

    # Test YOLO model
    assert YOLO_MODEL == "yolov11n-pose.pt", f"Expected yolov11n-pose.pt, got {YOLO_MODEL}"
    print("[OK] YOLO Model: {}".format(YOLO_MODEL))

    # Test window sizes
    assert EMOTION_WINDOW_SIZE == 5, "Emotion window should be 5"
    print("[OK] Emotion Window: {} frames (0.17s)".format(EMOTION_WINDOW_SIZE))

    assert POSTURE_WINDOW_SIZE == 7, "Posture window should be 7"
    print("[OK] Posture Window: {} frames (0.23s)".format(POSTURE_WINDOW_SIZE))

    assert ACTIVITY_WINDOW_SIZE == 10, "Activity window should be 10"
    print("[OK] Activity Window: {} frames (0.33s)".format(ACTIVITY_WINDOW_SIZE))

    assert GESTURE_WINDOW_SIZE == 5, "Gesture window should be 5"
    print("[OK] Gesture Window: {} frames (0.17s)".format(GESTURE_WINDOW_SIZE))

    # Test confidence thresholds
    assert CONFIDENCE_EMOTION == 0.65, "Emotion confidence should be 0.65"
    print("[OK] Emotion Confidence: {} (approx 3-4 out of 5)".format(CONFIDENCE_EMOTION))

    assert CONFIDENCE_POSTURE == 0.70, "Posture confidence should be 0.70"
    print("[OK] Posture Confidence: {} (approx 5 out of 7)".format(CONFIDENCE_POSTURE))

    # Test durations
    assert MIN_DURATION_EMOTION == 2.0, "Min emotion duration should be 2.0s"
    print("[OK] Min Emotion Duration: {}s".format(MIN_DURATION_EMOTION))

    # Test cooldowns
    assert COOLDOWN_EMOTION == 30, "Emotion cooldown should be 30s"
    print("[OK] Emotion Cooldown: {}s".format(COOLDOWN_EMOTION))

    assert COOLDOWN_ACTIVITY == 45, "Activity cooldown should be 45s"
    print("[OK] Activity Cooldown: {}s (one activity session)".format(COOLDOWN_ACTIVITY))

    # Test detection thresholds
    assert FALL_VELOCITY_THRESHOLD == 8, "Fall velocity threshold should be 8"
    print("[OK] Fall Velocity Threshold: {} pixels/frame".format(FALL_VELOCITY_THRESHOLD))

    print("\n" + "="*60)
    print("PASS: STEP 1 - Config loaded successfully!")
    print("="*60 + "\n")

    # Print summary
    print("CONFIG SUMMARY:")
    print("  Model: {}".format(YOLO_MODEL))
    print("  Windows: Emotion={}, Posture={}, Activity={}".format(
        EMOTION_WINDOW_SIZE, POSTURE_WINDOW_SIZE, ACTIVITY_WINDOW_SIZE))
    print("  Confidence Thresholds: Emotion={}, Posture={}".format(
        CONFIDENCE_EMOTION, CONFIDENCE_POSTURE))
    print("  Min Durations: Emotion={}s".format(MIN_DURATION_EMOTION))
    print("  Cooldowns: Emotion={}s, Activity={}s".format(
        COOLDOWN_EMOTION, COOLDOWN_ACTIVITY))
    print()


if __name__ == "__main__":
    test_config()
