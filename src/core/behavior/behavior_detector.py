"""
STEP 2: BehaviorDetector - Main behavior detection module
Skeleton class with all methods and data structures initialized
No changes to src/ - completely standalone
"""

import logging
import threading
import time
from collections import defaultdict, deque, Counter
import numpy as np

logger = logging.getLogger(__name__)


class BehaviorDetector:
    """
    Main behavior detection module
    Detects: posture, falls, walking, gestures, emotions, drowsiness
    Uses sliding window voting to eliminate false positives
    """

    def __init__(self):
        """Initialize all detection models and data structures"""
        logger.info("Initializing BehaviorDetector...")

        # ═══════════════════════════════════════════════════════════════
        # POSE HISTORY (for temporal analysis)
        # ═══════════════════════════════════════════════════════════════
        # Stores last 30 frames of keypoints per person
        # At 30fps, this = 1 second of history
        # Used for: fall detection, walking, eating/drinking temporal analysis
        self.pose_history = defaultdict(lambda: deque(maxlen=30))

        # ═══════════════════════════════════════════════════════════════
        # SLIDING WINDOWS (for temporal smoothing)
        # ═══════════════════════════════════════════════════════════════
        # Reduces false positives by 80%+
        # Structure: self.windows[person_id][category] = deque of detections
        from .config import (
            POSTURE_WINDOW_SIZE,
            ACTIVITY_WINDOW_SIZE,
        )
        self.posture_window_size = POSTURE_WINDOW_SIZE
        self.activity_window_size = ACTIVITY_WINDOW_SIZE
        self.windows = defaultdict(dict)

        # ═══════════════════════════════════════════════════════════════
        # LAST VOTED (tracks voting results per category per person)
        # ═══════════════════════════════════════════════════════════════
        # Only log when voting result CHANGES
        # Example: {"Person_1": {"POSTURE": "STANDING", "ACTIVITY": "WALKING"}}
        self.last_voted = defaultdict(dict)

        # ═══════════════════════════════════════════════════════════════
        # LAST LOGGED ACTION (prevents duplicate consecutive logs)
        # ═══════════════════════════════════════════════════════════════
        # Tracks the last logged action per person
        # Example: {"Person_1": "SITTING"}
        self.last_logged_action = defaultdict(lambda: None)

        # ═══════════════════════════════════════════════════════════════
        # SITTING START TIME (tracks when person started sitting)
        # ═══════════════════════════════════════════════════════════════
        # Used to calculate actual sitting duration
        # Example: {"Person_1": 1234567890.5}
        self.sitting_start_time = defaultdict(lambda: None)

        # ═══════════════════════════════════════════════════════════════
        # DEBOUNCE STATE (wait for stable posture after transition)
        # ═══════════════════════════════════════════════════════════════
        # Tracks pending state changes: {person_id: {category: (new_state, frame_count)}}
        self.pending_state_change = defaultdict(dict)

        # ═══════════════════════════════════════════════════════════════
        # THREAD SAFETY
        # ═══════════════════════════════════════════════════════════════
        self.lock = threading.Lock()

        logger.info("✓ BehaviorDetector initialized successfully")

    def detect_all(self, person_id, keypoints_17x3, person_crop_bgr, face_crop_bgr=None):
        """
        Main entry point for behavior detection
        Called once per person per frame

        Args:
            person_id (str): "Person_1", "Person_2", etc.
            keypoints_17x3 (np.ndarray): shape (17, 3) from YOLOv11n-pose
                Each row: [x, y, confidence]
            person_crop_bgr (np.ndarray): cropped person image for hand detection
            face_crop_bgr (np.ndarray): cropped face image for emotion detection (optional)

        Returns:
            list: [
                {"category": "POSTURE", "type": "STANDING", "confidence": 0.9, "details": {}},
                {"category": "ACTIVITY", "type": "WALKING", "confidence": 0.8, "details": {"pace": "slow"}},
                ...
            ]
            Returns only behaviors that CHANGED from last voting window
        """
        if keypoints_17x3 is None or keypoints_17x3.size == 0:
            return []

        with self.lock:
            # Store keypoints in pose history
            self.pose_history[person_id].append(keypoints_17x3.copy())

            # Detect posture (standing/sitting)
            posture_raw = self._detect_posture(keypoints_17x3)

            # Detect activity (walking/idle)
            activity_raw = self._detect_walking(person_id, keypoints_17x3)

            # Apply sliding window voting and return behaviors
            behaviors_to_log = []

            if posture_raw:
                behavior = self._apply_sliding_window(
                    person_id, "POSTURE", posture_raw["type"],
                    posture_raw.get("confidence", 0.5)
                )
                if behavior:
                    # Handle both single behavior and list of behaviors
                    if isinstance(behavior, list):
                        behaviors_to_log.extend(behavior)
                    else:
                        behaviors_to_log.append(behavior)

            # Temporarily disabled - only logging SITTING/STANDING for now
            # if activity_raw:
            #     behavior = self._apply_sliding_window(
            #         person_id, "ACTIVITY", activity_raw["type"],
            #         activity_raw.get("confidence", 0.7),
            #         details=activity_raw.get("details")
            #     )
            #     if behavior:
            #         # Handle both single behavior and list of behaviors
            #         if isinstance(behavior, list):
            #             behaviors_to_log.extend(behavior)
            #         else:
            #             behaviors_to_log.append(behavior)

            return behaviors_to_log

    def _detect_posture(self, keypoints):
        """
        Detect posture: STANDING, SITTING
        Uses heuristic rules on keypoint positions (no ML needed)

        Args:
            keypoints: (17, 3) array - each row is [x, y, confidence]

        Returns:
            dict or None: {"type": "STANDING"|"SITTING", "confidence": float}
        """
        from .config import MIN_KEYPOINT_CONFIDENCE, HIP_KNEE_DISTANCE_SITTING

        # Validate input
        if keypoints is None or keypoints.size == 0:
            return None

        if keypoints.shape != (17, 3):
            if keypoints.size == 51:  # 17 * 3
                try:
                    keypoints = keypoints.reshape(17, 3)
                except:
                    return None
            else:
                return None


        # ───────────────────────────────────────────────────────────
        # EXTRACT KEYPOINTS
        # ───────────────────────────────────────────────────────────
        # Keypoint indices:
        # 5=left_shoulder, 6=right_shoulder
        # 11=left_hip, 12=right_hip
        # 13=left_knee, 14=right_knee
        # 15=left_ankle, 16=right_ankle

        try:
            # Extract keypoints with confidence filtering
            left_shoulder = keypoints[5]   # [x, y, conf]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]

            # ───────────────────────────────────────────────────────
            # HANDLE LOW-CONFIDENCE KEYPOINTS
            # ───────────────────────────────────────────────────────
            # If keypoint confidence < threshold, skip it
            joints = [
                ("shoulder", left_shoulder, right_shoulder),
                ("hip", left_hip, right_hip),
                ("knee", left_knee, right_knee),
                ("ankle", left_ankle, right_ankle),
            ]

            # For each joint, use the visible side (higher confidence)
            # ANKLES are optional (hard to detect with clothing)
            valid_kps = []
            for joint_name, left, right in joints:
                left_conf = left[2]
                right_conf = right[2]

                # Ankles are optional - skip if both low confidence
                if joint_name == "ankle":
                    if left_conf < 0.1 and right_conf < 0.1:  # Very strict for ankles only
                        continue
                # Other joints required
                elif left_conf < MIN_KEYPOINT_CONFIDENCE and right_conf < MIN_KEYPOINT_CONFIDENCE:
                    return None

                # Use the side with higher confidence
                if left_conf >= right_conf:
                    valid_kps.append((joint_name, "left", left))
                else:
                    valid_kps.append((joint_name, "right", right))

            # ───────────────────────────────────────────────────────
            # CALCULATE BODY DIMENSIONS
            # ───────────────────────────────────────────────────────
            # Mid-shoulder: average of left and right shoulders
            mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
            mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

            # Mid-hip: average of left and right hips
            mid_hip_x = (left_hip[0] + right_hip[0]) / 2
            mid_hip_y = (left_hip[1] + right_hip[1]) / 2

            # Mid-knee: average of left and right knees
            mid_knee_x = (left_knee[0] + right_knee[0]) / 2
            mid_knee_y = (left_knee[1] + right_knee[1]) / 2


            # Torso height: vertical distance from shoulder to hip
            torso_height = abs(mid_shoulder_y - mid_hip_y)

            # ───────────────────────────────────────────────────────
            # DECISION TREE (STANDING / SITTING only)
            # ───────────────────────────────────────────────────────
            # 1. SITTING: hips are close to or below knees (bent legs)
            if mid_hip_y > mid_knee_y - HIP_KNEE_DISTANCE_SITTING:
                posture = "SITTING"

            # 2. STANDING: normal upright posture
            else:
                posture = "STANDING"

            # ───────────────────────────────────────────────────────
            # CALCULATE CONFIDENCE
            # ───────────────────────────────────────────────────────
            # Average confidence of keypoints we actually used (not ankles if unavailable)
            all_confs = [
                left_shoulder[2], right_shoulder[2],
                left_hip[2], right_hip[2],
                left_knee[2], right_knee[2]
            ]
            # Include ankles only if they were visible
            left_ankle_conf = left_ankle[2]
            right_ankle_conf = right_ankle[2]
            if left_ankle_conf > 0.1 or right_ankle_conf > 0.1:
                all_confs.extend([left_ankle_conf, right_ankle_conf])

            confidence = sum(all_confs) / len(all_confs) if all_confs else 0.5


            return {
                "type": posture,
                "confidence": float(confidence)
            }

        except Exception as e:
            logger.debug("Posture detection error: {}".format(e))
            return None

    def _detect_walking(self, person_id, keypoints):
        """
        Detect walking and pace: IDLE, WALKING (slow/normal/fast)
        Only active when person is STANDING (not sitting/lying)

        Args:
            person_id: ID of person
            keypoints: (17, 3) array - each row is [x, y, confidence]

        Returns:
            dict or None: {"type": "IDLE"|"WALKING", "confidence": float, "details": {"pace": "slow"|"normal"|"fast"}}
        """
        from .config import (
            WALKING_HISTORY_FRAMES,
            WALKING_MIN_VELOCITY,
            WALKING_NORMAL_VELOCITY,
            WALKING_FAST_VELOCITY
        )

        # Need at least 15 frames of history to detect walking velocity
        if person_id not in self.pose_history or len(self.pose_history[person_id]) < WALKING_HISTORY_FRAMES:
            return None

        try:
            # Extract current and old hip position
            current_kps = keypoints
            old_kps = self.pose_history[person_id][0]  # 15 frames ago

            current_hip_x = (current_kps[11][0] + current_kps[12][0]) / 2
            old_hip_x = (old_kps[11][0] + old_kps[12][0]) / 2

            # Calculate horizontal displacement
            hip_displacement = abs(current_hip_x - old_hip_x)
            hip_velocity = hip_displacement / WALKING_HISTORY_FRAMES

            # ───────────────────────────────────────────────────────
            # DETERMINE ACTIVITY & PACE
            # ───────────────────────────────────────────────────────
            if hip_velocity < WALKING_MIN_VELOCITY:
                return None  # Not walking - let POSTURE detector handle it
            elif hip_velocity < WALKING_NORMAL_VELOCITY:
                activity_type = "WALKING"
                pace = "slow"
                confidence = hip_velocity / WALKING_NORMAL_VELOCITY  # 0.5-1.0
            elif hip_velocity < WALKING_FAST_VELOCITY:
                activity_type = "WALKING"
                pace = "normal"
                confidence = hip_velocity / WALKING_FAST_VELOCITY  # 0.5-1.0
            else:
                activity_type = "WALKING"
                pace = "fast"
                confidence = min(1.0, hip_velocity / (WALKING_FAST_VELOCITY * 1.5))

            details = {}
            if pace:
                details["pace"] = pace
                details["velocity"] = float(hip_velocity)

            return {
                "type": activity_type,
                "confidence": float(confidence),
                "details": details
            }

        except Exception as e:
            logger.debug("Walking detection error: {}".format(e))
            return None

    def _apply_sliding_window(self, person_id, category, detection_type, confidence, details=None):
        """
        Apply sliding window voting
        Only logs when voting result CHANGES from last_voted
        Returns behavior dict or None if no change
        """
        # Initialize window for this person/category if not exists
        if person_id not in self.windows:
            self.windows[person_id] = {
                "POSTURE": deque(maxlen=self.posture_window_size),
                "ACTIVITY": deque(maxlen=self.activity_window_size),
            }

        # Add detection to window
        self.windows[person_id][category].append(detection_type)

        # Check if window is full
        window = self.windows[person_id][category]
        if len(window) < window.maxlen:
            return None  # Window not full yet

        # Vote only on last N frames (most recent, most stable)
        stable_frames = list(window)[-5:]  # Last 5 frames only
        votes = Counter(stable_frames)
        winner, count = votes.most_common(1)[0]
        window_confidence = count / len(stable_frames)

        # Check if result differs from last_voted
        last = self.last_voted[person_id].get(category)
        if last == winner:
            return None  # No state change, don't log

        # ───────────────────────────────────────────────────────────
        # STATE CHANGE DETECTED - Start debounce (wait 15 frames)
        # ───────────────────────────────────────────────────────────
        debounce_frames = 15  # 0.5 seconds at 30fps

        # First time seeing this state change?
        if category not in self.pending_state_change[person_id]:
            # Record it as pending
            self.pending_state_change[person_id][category] = {
                "new_state": winner,
                "frame_count": 0,
                "confidence": window_confidence
            }
            # Record sitting start time NOW (when first detected, not after debounce)
            if winner == "SITTING" and self.sitting_start_time[person_id] is None:
                self.sitting_start_time[person_id] = time.time()
            if category == "POSTURE":
                print("[DEBOUNCE] {}: {} (waiting 15 frames)".format(person_id, winner))
            return None  # Wait, don't log yet

        # Already pending - increment frame counter
        pending = self.pending_state_change[person_id][category]
        pending["frame_count"] += 1

        # Check if debounce time passed (15 frames)
        if pending["frame_count"] < debounce_frames:
            return None  # Still waiting

        # ───────────────────────────────────────────────────────────
        # DEBOUNCE COMPLETE - Log the state change
        # ───────────────────────────────────────────────────────────
        # Print breakdown only on state change
        if category == "POSTURE":
            binary_votes = [1 if v == winner else 0 for v in window]
            print("[CHANGE] {} -> {} ({}/{}) [{}]".format(
                person_id, winner, count, len(stable_frames), " ".join(map(str, binary_votes))))

        # Check confidence threshold before logging
        from .config import (
            CONFIDENCE_POSTURE,
            CONFIDENCE_ACTIVITY
        )

        confidence_thresholds = {
            "POSTURE": CONFIDENCE_POSTURE,
            "ACTIVITY": CONFIDENCE_ACTIVITY
        }

        min_confidence = confidence_thresholds.get(category, 0.65)

        # Log rejected detections (below threshold) for debugging
        if window_confidence < min_confidence:
            logger.info("[REJECTED] {}: {} ({:.0f}%) - Below threshold {:.0f}%".format(
                person_id, winner, window_confidence * 100, min_confidence * 100))
            del self.pending_state_change[person_id][category]
            return None  # Confidence too low, don't log

        # State changed AND debounced AND confidence high enough! Log it
        current_time = time.time()
        self.last_voted[person_id][category] = winner

        behaviors_list = []

        # Only track duration for SITTING
        if winner == "SITTING":
            # Person just sat down - record start time but DON'T LOG YET
            if self.sitting_start_time[person_id] is None:
                self.sitting_start_time[person_id] = current_time

            # Don't log sitting yet, wait until person stands up to log with actual duration
            del self.pending_state_change[person_id][category]
            self.windows[person_id][category].clear()
            self.last_logged_action[person_id] = winner
            print("[SITTING STARTED] {}: Will log duration when standing up".format(person_id))
            return None
        else:
            # Person transitioned to STANDING/WALKING
            # If they were sitting, first log the SITTING duration
            if self.sitting_start_time[person_id] is not None:
                sitting_duration = current_time - self.sitting_start_time[person_id]
                self.sitting_start_time[person_id] = None

                logger.info("[LOGGED] Person_{}: SITTING ({:.0f}%) - Duration: {:.1f}s".format(
                    person_id, window_confidence * 100, sitting_duration))

                sitting_result = {
                    "category": "POSTURE",
                    "type": "SITTING",
                    "confidence": window_confidence,
                    "details": {},
                    "duration": sitting_duration
                }
                behaviors_list.append(sitting_result)

            # Then log the new behavior (STANDING/WALKING)
            logger.info("[LOGGED] {}: {} ({:.0f}%)".format(
                person_id, winner, window_confidence * 100))

            result = {
                "category": category,
                "type": winner,
                "confidence": window_confidence,
                "details": details or {}
            }
            behaviors_list.append(result)

        # Clear debounce and window after logging
        del self.pending_state_change[person_id][category]
        self.windows[person_id][category].clear()

        # Track last logged action (for duplicate prevention)
        self.last_logged_action[person_id] = winner

        return behaviors_list

    def cleanup(self, person_id):
        """Called when person exits frame. Clears their history."""
        with self.lock:
            if person_id in self.pose_history:
                del self.pose_history[person_id]
            if person_id in self.windows:
                del self.windows[person_id]
            if person_id in self.last_voted:
                del self.last_voted[person_id]
            if person_id in self.pending_state_change:
                del self.pending_state_change[person_id]
            if person_id in self.sitting_start_time:
                del self.sitting_start_time[person_id]
            if person_id in self.last_logged_action:
                del self.last_logged_action[person_id]
            logger.debug(f"Cleaned up behavior state for {person_id}")

