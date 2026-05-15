"""
STEP 1: Behavior Detection Configuration
All thresholds, windows, and parameters for behavior detection
Completely separate from src/config.py
"""

# ═══════════════════════════════════════════════════════════════════
# YOLO SETTINGS
# ═══════════════════════════════════════════════════════════════════
YOLO_MODEL = "yolov11n-pose.pt"  # Upgrade from yolov8n.pt to get 17 keypoints per person


# ═══════════════════════════════════════════════════════════════════
# SLIDING WINDOW SIZES (in frames)
# ═══════════════════════════════════════════════════════════════════
EMOTION_WINDOW_SIZE = 5          # Quick emotion checks (0.17s at 30fps)
POSTURE_WINDOW_SIZE = 10         # Larger window for stability (0.5s at 30fps)
ACTIVITY_WINDOW_SIZE = 10        # Longer for walking/eating (0.33s)
GESTURE_WINDOW_SIZE = 5          # Quick gesture detection (0.17s)
DROWSINESS_WINDOW_SIZE = 8       # Eye closure patterns (0.27s)


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLDS (for logging)
# ═══════════════════════════════════════════════════════════════════
CONFIDENCE_EMOTION = 0.50        # 2-3 out of 5 window votes
CONFIDENCE_POSTURE = 0.50        # 5 out of 10 votes (much lower for real-world detection)
CONFIDENCE_ACTIVITY = 0.50       # 5 out of 10 votes
CONFIDENCE_GESTURE = 0.55        # 3 out of 5 votes
CONFIDENCE_DROWSINESS = 0.60     # 5 out of 8


# ═══════════════════════════════════════════════════════════════════
# MINIMUM DURATION BEFORE LOGGING (seconds)
# ═══════════════════════════════════════════════════════════════════
MIN_DURATION_EMOTION = 2.0       # Mood should persist > 1 window
MIN_DURATION_POSTURE = 0.5       # Body position changes (reduced from 1.0 for faster detection)
MIN_DURATION_ACTIVITY = 3.0      # Activities need confirmation
MIN_DURATION_GESTURE = 0.5       # Gestures brief but intentional
MIN_DURATION_DROWSINESS = 5.0    # Inattention should be sustained


# ═══════════════════════════════════════════════════════════════════
# PER-CATEGORY COOLDOWN (seconds)
# Don't re-log the SAME behavior for X seconds
# ═══════════════════════════════════════════════════════════════════
COOLDOWN_EMOTION = 30            # Don't log same emotion for 30s
COOLDOWN_POSTURE = 20            # Don't log same posture for 20s
COOLDOWN_ACTIVITY = 45           # One activity session = 45s
COOLDOWN_GESTURE = 60            # Same gesture not until 1 min later
COOLDOWN_DROWSINESS = 120        # Alert about drowsiness max once per 2 min


# ═══════════════════════════════════════════════════════════════════
# DETECTION THRESHOLDS (heuristic rules)
# ═══════════════════════════════════════════════════════════════════
FALL_VELOCITY_THRESHOLD = 5      # pixels/frame for fall detection
FALL_HISTORY_FRAMES = 15         # Track last 15 frames (0.5s) for fall velocity
MOUTH_THRESHOLD = 80             # pixels - hand near mouth for eating/drinking
HAND_RAISED_MARGIN = 30          # pixels - wrist above nose
WALKING_MIN_VELOCITY = 2         # pixels/frame minimum walking speed
WALKING_NORMAL_VELOCITY = 6      # pixels/frame normal pace threshold
WALKING_FAST_VELOCITY = 12       # pixels/frame for "fast" pace (nearly running)
WALKING_HISTORY_FRAMES = 15      # Track last 15 frames for walking velocity

# Keypoint confidence - ignore low-confidence keypoints
MIN_KEYPOINT_CONFIDENCE = 0.2    # Ignore if confidence < 0.2 (more lenient)

# Posture detection thresholds
BODY_RATIO_LYING = 1.2           # body_height < body_width * 1.2 → LYING
HIP_KNEE_DISTANCE_SITTING = 50   # hips close to knees (pixels) → SITTING (increased from 30 for better detection)


# ═══════════════════════════════════════════════════════════════════
# MODEL FRAME INTERVALS
# ═══════════════════════════════════════════════════════════════════
EMOTION_FRAME_INTERVAL = 5       # Run HSEmotion every 5th frame
GESTURE_FRAME_INTERVAL = 3       # Run gesture every 3rd frame
DROWSINESS_FRAME_INTERVAL = 2    # Run drowsiness every 2nd frame


# ═══════════════════════════════════════════════════════════════════
# YOLO v11n-pose KEYPOINT REFERENCE
# ═══════════════════════════════════════════════════════════════════
"""
Index   Name              Use in this project
─────   ──────────────    ──────────────────────────────
  0     nose              Eating/drinking (hand-to-mouth target)
  1     left_eye          Drowsiness (eye aspect ratio)
  2     right_eye         Drowsiness (eye aspect ratio)
  3     left_ear          (not used)
  4     right_ear         (not used)
  5     left_shoulder     Posture, fall detection
  6     right_shoulder    Posture, fall detection
  7     left_elbow        (not used)
  8     right_elbow       (not used)
  9     left_wrist        Hand raised, eating/drinking
 10     right_wrist       Hand raised, eating/drinking
 11     left_hip          Posture, fall, walking
 12     right_hip         Posture, fall, walking
 13     left_knee         Posture
 14     right_knee        Posture
 15     left_ankle        Posture, fall
 16     right_ankle       Posture, fall

Each keypoint comes as (x, y, confidence) in pixel coordinates.
"""

print("[CONFIG] Behavior detection configured - yolov11n-pose model ready")
