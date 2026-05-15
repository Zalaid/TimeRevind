"""
Video Processor Module
Main pipeline: reads frames, runs detection, tracking, and event logging
"""

import logging
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue
import uuid
import time
import hashlib

from scipy.spatial.distance import norm
from ultralytics import YOLO
from src.config import (
    PROJECT_ROOT,
    YOLO_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    VIDEO_FPS,
    VIDEO_RESOLUTION,
    RESOLUTION_FALLBACKS,
    FRAMES_TO_COLLECT,
    FRAME_CAPTURE_STRIDE,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_MATCH_THRESHOLD,
    BEST_FRAMES_DIR,
    FACE_EMBEDDING_L2_THRESHOLD,
    FACE_FRAMES_DIR,
    NEW_VISIT_THRESHOLD_SECONDS,
    LOGS_DIR,
    ACCUMULATOR_ENABLED,
    ACCUMULATOR_TARGET_EMBEDDINGS,
    ACCUMULATOR_MAX_DURATION,
    ACCUMULATOR_MAX_FAILED,
    SAVE_NEW_EMBEDDINGS,
)
from src.core.person_tracker import PersonTracker, Person
from src.core.embeddings import EmbeddingManager
from src.core.event_logger import EventLogger
from src.core.behavior.behavior_detector import BehaviorDetector
from src.core.behavior.behavior_db_logger import get_behavior_db_logger

logger = logging.getLogger(__name__)


class BboxSmoother:
    """Smooths bounding boxes across frames using exponential moving average + detection memory.
    Reduces jitter and handles brief detection dropouts."""

    def __init__(self, alpha=0.3, dropout_frames=15):
        """
        Args:
            alpha: Smoothing factor (0.0-1.0). Higher = more responsive, less smooth
            dropout_frames: Max frames to interpolate during dropout (at 30fps ≈ 0.5s)
        """
        self.alpha = alpha
        self.dropout_frames = dropout_frames
        self.track_history = {}  # track_id -> {bbox, confidence, last_frame}

    def smooth(self, detections, frame_count):
        """
        Smooth detections and fill brief dropouts.

        Args:
            detections: List of {track_id, bbox, confidence}
            frame_count: Current frame number

        Returns:
            Smoothed detections list
        """
        smoothed = []
        current_track_ids = {d['track_id'] for d in detections}

        # Update existing tracks and smooth new detections
        for detection in detections:
            track_id = detection['track_id']
            bbox = np.array(detection['bbox'], dtype=np.float32)
            conf = detection['confidence']

            if track_id in self.track_history:
                # Smooth with EMA: new_val = alpha * current + (1-alpha) * previous
                prev_bbox = np.array(self.track_history[track_id]['bbox'], dtype=np.float32)
                bbox = self.alpha * bbox + (1 - self.alpha) * prev_bbox

            self.track_history[track_id] = {
                'bbox': bbox.tolist(),
                'confidence': conf,
                'last_frame': frame_count
            }

            smoothed.append({
                'track_id': track_id,
                'bbox': bbox.tolist(),
                'confidence': conf,
                'smoothed': track_id in self.track_history  # Flag: was smoothed
            })

        # Handle brief dropouts: interpolate missing tracks for up to dropout_frames
        for track_id, history in list(self.track_history.items()):
            if track_id not in current_track_ids:
                frames_missing = frame_count - history['last_frame']
                if 0 < frames_missing <= self.dropout_frames:
                    # Interpolate: track existed recently, assume it's still nearby
                    smoothed.append({
                        'track_id': track_id,
                        'bbox': history['bbox'],
                        'confidence': history['confidence'] * 0.8,  # Reduce confidence for interpolated
                        'interpolated': True
                    })
                elif frames_missing > self.dropout_frames:
                    # Track gone too long, remove from history
                    del self.track_history[track_id]

        return smoothed

    def reset(self):
        """Clear history"""
        self.track_history.clear()


class FrameBuffer:
    """Buffers frames for embedding collection (up to 50 frames across 10 batches)"""

    def __init__(self, max_frames=55):
        self.frames = []
        self.frame_numbers = []  # Track actual video frame numbers
        self.max_frames = max_frames

    def add(self, frame, frame_number=None):
        """Add frame to buffer with optional video frame number"""
        self.frames.append(frame.copy())
        self.frame_numbers.append(frame_number if frame_number is not None else -1)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
            self.frame_numbers.pop(0)

    def clear(self):
        """Clear buffer"""
        self.frames = []
        self.frame_numbers = []

    def has_enough_frames(self):
        """Check if we have enough frames"""
        return len(self.frames) >= FRAMES_TO_COLLECT


class AccumulatorState:
    """Tracks background embedding collection for a single person."""

    COOLDOWN_SECONDS = 60

    def __init__(self, person_id, instance_id, target_count=12, existing_count=0,
                 max_duration=ACCUMULATOR_MAX_DURATION, max_failed=ACCUMULATOR_MAX_FAILED):
        self.person_id = person_id
        self.instance_id = instance_id       # Unique UUID — never recycled even if person_id is
        self.target_count = target_count
        self.existing_count = existing_count  # Already in Qdrant
        self.new_embeddings = []              # Freshly collected good embeddings
        self.new_embedding_poses = []         # Parallel pose label per new embedding
        self.qdrant_embeddings = []           # Pre-loaded existing embeddings from Qdrant (previous sessions)
        self.qdrant_poses = []                # Parallel pose labels for qdrant_embeddings
        self.qdrant_loaded = False            # True once pre-load is complete
        self.is_running = True
        self.saved = False
        self.flushed_count = 0               # How many new_embeddings already saved to Qdrant (incremental flush)
        self.start_time = time.time()
        self.failed_attempts = 0
        self.gave_up_at = None               # Timestamp when accumulator gave up
        self.retry_count = 0
        self.max_retries = 3
        self.max_duration = max_duration
        self.max_failed = max_failed
        self.frame_counter = 0               # For internal stride management
        self.checks_since_last_summary = 0   # For periodic summary logging
        # Per-reason rejection counters
        self.rej_no_face = 0
        self.rej_blurry = 0
        self.rej_similar = 0
        self.rej_l2 = 0
        self.rej_cohesion = 0
        self.lock = threading.Lock()
        self.acc_logger = None  # Set by _start_background_accumulator

    def rejection_summary(self):
        """Return a compact rejection breakdown string."""
        return (
            f"no_face={self.rej_no_face} | blurry={self.rej_blurry} | "
            f"similar={self.rej_similar} | l2_outlier={self.rej_l2} | cohesion={self.rej_cohesion}"
        )

    @property
    def current_total(self):
        return self.existing_count + len(self.new_embeddings)

    @property
    def reached_target(self):
        return self.current_total >= self.target_count

    @property
    def should_give_up(self):
        elapsed = time.time() - self.start_time
        return elapsed > self.max_duration or self.failed_attempts > self.max_failed

    @property
    def can_retry(self):
        """Check if enough cooldown has passed and retries remain."""
        if self.gave_up_at is None:
            return False
        cooldown_elapsed = time.time() - self.gave_up_at
        return cooldown_elapsed >= self.COOLDOWN_SECONDS and self.retry_count < self.max_retries


class VideoProcessor:
    """Main video processing pipeline"""

    def __init__(self, db_manager, embedding_store, camera_id=0):
        self.db_manager = db_manager
        self.embedding_store = embedding_store
        self.camera_id = camera_id

        # Generate unique session ID for this session
        self.session_id = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
        logger.info(f"Session ID: {self.session_id}")

        # Log session to database
        self.db_manager.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at, camera_id) VALUES (?, ?, ?)",
            (self.session_id, datetime.now().isoformat(), str(self.camera_id))
        )

        # Initialize models
        logger.info("Loading YOLO model...")
        self.yolo_model = YOLO(YOLO_MODEL)
        logger.info("✓ YOLO model loaded")

        logger.info("Loading YOLOv11n-pose model for behavior detection...")
        pose_model_path = PROJECT_ROOT / "yolo11n-pose.pt"
        self.pose_model = YOLO(str(pose_model_path))
        logger.info("✓ Pose model loaded")

        logger.info("Initializing behavior detector...")
        self.behavior_detector = BehaviorDetector()
        logger.info("✓ Behavior detector initialized")

        logger.info("Initializing person tracker...")
        self.tracker = PersonTracker(embedding_store, db_manager)
        logger.info("✓ Person tracker initialized")

        # 🛡️ Clear rejected_track_ids for this session
        self.tracker.rejected_track_ids.clear()

        # Initialize ID counter from SQLite to prevent ID reuse across sessions
        self._initialize_id_counter()
        logger.info("✓ ID counter initialized")

        logger.info("Loading embedding models (may take 30-60 seconds on first run)...")
        self.embeddings = EmbeddingManager()
        logger.info("✓ Embedding models loaded")

        logger.info("Initializing event logger...")
        self.event_logger = EventLogger(db_manager, self.session_id)
        logger.info("✓ Event logger initialized")

        logger.info("Initializing behavior database logger...")
        self.behavior_db_logger = get_behavior_db_logger(event_logger=self.event_logger)
        logger.info("✓ Behavior database logger initialized")

        # Frame processing
        self.frame_count = 0
        self.start_time = None
        self.current_video_file = None
        self.current_video_offset = 0

        # Person buffers (for 6-frame collection)
        self.person_frame_buffers = {}  # person_id -> FrameBuffer
        self.person_first_detection_frame = {}  # person_id -> frame_count (when first detected)
        self.person_frame_counters = {}
        self.buffers_lock = threading.Lock()  # Protect person_frame_buffers and person_first_detection_frame from race conditions

        # Face/body embedding lock (MTCNN and TensorFlow are NOT thread-safe)
        self.embedding_lock = threading.Lock()  # Protect compute_face_embedding calls

        # Entry logging lock (prevent duplicate ENTERED events from concurrent threads)
        self.entry_logging_lock = threading.Lock()

        # Background accumulator tracking
        self.active_accumulators = {}         # instance_id -> AccumulatorState
        self.accumulator_lock = threading.Lock()

        # Video writer
        self.video_writer = None
        self.recording = False

        # Bounding box smoother (reduces jitter, handles brief dropouts)
        self.bbox_smoother = BboxSmoother(alpha=0.35, dropout_frames=20)

        # Background embedding thread
        self.embedding_queue = Queue()
        self.embedding_stop_event = threading.Event()  # Signal to stop embedding thread
        self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.embedding_thread.start()

        logger.info("VideoProcessor initialized")

    def _initialize_id_counter(self):
        """Set _id_counter to max existing person ID from SQLite"""
        try:
            result = self.db_manager.query(
                "SELECT person_id FROM person_profiles ORDER BY person_id DESC"
            )

            max_id = 0
            for row in result:
                pid = row[0]  # e.g. "Person_1", "Person_2"
                try:
                    num = int(pid.split("_")[1])
                    if num > max_id:
                        max_id = num
                except (ValueError, IndexError):
                    continue

            Person._id_counter = max_id
            logger.info(f"✅ ID counter initialized to {max_id} (from SQLite person_profiles)")

        except Exception as e:
            logger.warning(f"Failed to initialize ID counter from SQLite: {e}")

    def start_recording(self, output_path=None, resolution=VIDEO_RESOLUTION):
        """Start recording video"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = Path(f"videos/{timestamp}.mp4")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_video_file = str(output_path)

        # Try common codecs, preference for compatibility on Windows
        logger.info(f"Initializing video writer at resolution {resolution} with FPS {VIDEO_FPS}")
        for codec in ["XVID", "avc1", "MJPG", "h264", "DIVX", "X264"]:
            logger.debug(f"Trying codec: {codec}")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                self.current_video_file,
                fourcc,
                VIDEO_FPS,
                resolution,  # Use actual configured resolution
            )

            if self.video_writer.isOpened():
                logger.info(f"✅ Recording started with codec '{codec}' at {VIDEO_RESOLUTION}: {output_path}")
                self.recording = True
                self.start_time = datetime.now()
                return
            else:
                logger.debug(f"❌ Codec '{codec}' failed")

        # If all codecs failed, log error but don't crash
        logger.error(f"Failed to initialize video writer with any codec. Video recording disabled.")
        logger.error(f"  Resolution: {VIDEO_RESOLUTION}")
        logger.error(f"  FPS: {VIDEO_FPS}")
        logger.error(f"  Output path: {output_path}")
        self.video_writer = None
        self.recording = False

    def stop_recording(self):
        """Stop recording video"""
        if self.video_writer:
            try:
                self.video_writer.release()
                logger.info(f"Recording stopped")
            except Exception as e:
                logger.warning(f"Error releasing video writer: {e}")
            finally:
                self.video_writer = None
                self.recording = False

    def process_frame(self, frame):
        """
        Process a single frame

        Args:
            frame: Input frame (BGR)

        Returns:
            Processed frame with annotations
        """
        self.frame_count += 1

        # 🛡️ Periodically clear rejected_track_ids to prevent unbounded growth
        if self.frame_count % 1000 == 0 and len(self.tracker.rejected_track_ids) > 100:
            self.tracker.rejected_track_ids.clear()
            logger.debug(f"Cleared rejected_track_ids (was {len(self.tracker.rejected_track_ids)} items)")

        # Run YOLO tracking (with ByteTrack for consistent track_ids across frames)
        # Lower confidence threshold to catch more people (reduces 2-4s disappearances)
        results = self.yolo_model.track(
            frame,
            conf=0.25,  # Lowered from 0.65 → 0.45 → 0.35 (catches even more detections)
            persist=True,  # Critical: maintains track IDs across frames
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Extract detections
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                if box.cls == 0:  # Class 0 = person in COCO
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    track_id = int(box.id) if box.id is not None else None

                    detections.append(
                        {
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(conf),
                        }
                    )

            # Debug: log detections for multi-person scenarios
            if len(detections) > 1:
                logger.debug(f"Frame {self.frame_count}: Detected {len(detections)} people - track_ids: {[d['track_id'] for d in detections]}")

            # 🛡️ Filter out detections with track_id=None (ByteTrack not ready yet)
            # These cause duplicate REIDENTIFIED logs when real track_id comes next
            detections = [d for d in detections if d['track_id'] is not None]
            if len(detections) == 0:
                logger.debug(f"Frame {self.frame_count}: All detections skipped (ByteTrack IDs not ready)")

        # ✨ Smooth bounding boxes and handle brief dropouts (max 0.67s @ 30fps)
        detections = self.bbox_smoother.smooth(detections, self.frame_count)

        # Log if detection was interpolated during dropout
        interpolated = [d for d in detections if d.get('interpolated')]
        if interpolated and self.frame_count % 30 == 0:  # Log every 1 second
            logger.debug(f"Frame {self.frame_count}: {len(interpolated)} track(s) interpolated (dropout recovery)")

        # Update person tracking
        current_detections = self.tracker.update(frame, detections, self.embeddings, self.frame_count, self.embedding_lock)

        # Run pose detection for behavior logging (pass original detection boxes)
        if results and len(results) > 0:
            self._detect_and_log_behaviors(frame, results[0].boxes, current_detections)

        # ✨ NEW: Store detections for re-identification
        self._last_detections = detections

        # ✨ NEW: Periodic re-identification (every 10 seconds)
        self._periodic_reidentification(frame, current_detections)

        # Log entry/exit events and update person profiles
        try:
            with self.tracker.people_lock:
                people_to_process = list(self.tracker.people.items())
        except (RuntimeError, KeyError):
            people_to_process = []

        for person_id, person in people_to_process:
            # 🛡️ Safety check: person may have been deleted by _check_timeouts during this frame
            # (snapshot was taken before _check_timeouts ran)
            if self.tracker.get_person(person_id) is None:
                continue

            # Skip temporary persons - they will be logged at Frame 3 when confirmed
            if hasattr(person, '_is_temporary') and person._is_temporary:
                continue

            # Skip if entry already logged by embedding thread (for returning visitors confirmed in _compute_and_store_embeddings)
            # Exception: don't skip if 30s exit is pending — that check needs to run even for already-logged persons
            if hasattr(person, '_entry_logged') and person._entry_logged and not getattr(person, '_30s_exit_pending', False) and not hasattr(person, '_actual_exit'):
                continue

            # Skip if this is a temp person being confirmed as returning visitor
            # Entry will be logged by embedding thread under correct matched person ID
            if hasattr(person, '_matched_person_id') and person._matched_person_id:
                continue

            # 30s exit: person was LOST >= 30s — log DB exit now (before possible re-entry as new visit)
            if getattr(person, '_30s_exit_pending', False):
                person._30s_exit_pending = False
                person._30s_exit_logged = True
                if hasattr(person, '_entry_logged') and person._entry_logged:
                    exit_timestamp = person.lost_time.strftime("%d/%m/%Y %I:%M:%S %p") if person.lost_time else None
                    exit_frame = max(0, self.frame_count - int(NEW_VISIT_THRESHOLD_SECONDS * VIDEO_FPS))
                    self.event_logger.log_exit(person_id, person.visit_number, self.current_video_file, exit_frame, timestamp=exit_timestamp)
                    logger.info(f"⏱️ {person_id} (visit {person.visit_number}) → EXITED logged to DB — absent {person.time_since_lost:.0f}s, new visit if they return")
                    for attr in ('_entry_logged', '_entry_frame', '_entry_logged_session_id'):
                        if hasattr(person, attr):
                            delattr(person, attr)
                # Stop accumulator and save partial results when person fully exits
                if ACCUMULATOR_ENABLED:
                    instance_id = person._instance_id
                    if instance_id in self.active_accumulators:
                        logger.info(f"🔄 ACCUMULATOR {person_id}: 30s exit triggered — stopping and saving")
                        self._stop_accumulator(instance_id, save=True)

            if person.is_in_frame and not hasattr(person, '_entry_logged'):
                # New entry - use lock to prevent concurrent logging
                with self.entry_logging_lock:
                    # Double-check after acquiring lock (another thread might have logged it)
                    # Also check if we're in a new session (entry_logged_session_id is different)
                    if not hasattr(person, '_entry_logged') or person._entry_logged_session_id != self.session_id:
                        # Calculate correct visit_num from database (within this session only)
                        session_visits_result = self.db_manager.query(
                            "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                            (person_id, self.session_id)
                        )
                        current_visit_num = (session_visits_result[0][0] if session_visits_result else 0) + 1
                        person.visit_number = current_visit_num  # Update person object with correct visit_num

                        logger.info(f"Logging ENTRY: {person_id} with visit_num = {person.visit_number} (session {self.session_id})")
                        self.event_logger.log_entry(person_id, person.visit_number, self.current_video_file, self.frame_count)
                        person._entry_logged = True
                        person._entry_logged_session_id = self.session_id  # Track which session entry was logged for
                        person._entry_frame = self.frame_count  # Store frame number for accurate duration calculation
                # Don't reset embedding flag - ByteTrack maintains track_id for same person
                # Embeddings only recomputed after timeout (new visit/session)

                # Count total visits from database (all ENTRY events for this person - all sessions)
                total_visits_result = self.db_manager.query(
                    "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED'",
                    (person_id,)
                )
                total_visits = total_visits_result[0][0] if total_visits_result else 1

                # Create/update person profile on entry
                # Use INSERT OR IGNORE to preserve existing total_time_seconds on re-entry
                self.db_manager.execute(
                    """INSERT OR IGNORE INTO person_profiles
                    (person_id, first_seen, last_seen, total_visits, total_time_seconds, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (person_id, person.first_seen.isoformat(), datetime.now().isoformat(),
                     1, 0, datetime.now().isoformat())
                )
                # Then update to latest values
                self.db_manager.execute(
                    """UPDATE person_profiles
                    SET last_seen = ?, total_visits = ?, updated_at = ?
                    WHERE person_id = ?""",
                    (datetime.now().isoformat(), total_visits, datetime.now().isoformat(), person_id)
                )

            elif not person.is_in_frame and hasattr(person, '_actual_exit'):
                # Skip if this person was merged into another — don't log exit for temp ID
                if hasattr(person, '_matched_person_id') and person._matched_person_id:
                    # Clean up flags (but NOT _exit_processed — keep until mark_online())
                    for attr in ('_entry_logged', '_entry_frame', '_actual_exit', '_entry_logged_session_id'):
                        if hasattr(person, attr):
                            delattr(person, attr)
                    # Clean up buffers
                    with self.buffers_lock:
                        self.person_frame_buffers.pop(person_id, None)
                        self.person_first_detection_frame.pop(person_id, None)
                        self.person_frame_counters.pop(person_id, None)
                    continue

                # 🛡️ CALCULATE DURATION FIRST (before cleaning up _entry_frame)
                duration_seconds = 0
                if hasattr(person, '_entry_frame'):
                    frame_duration = self.frame_count - person._entry_frame
                    duration_seconds = int(frame_duration / VIDEO_FPS)

                # Only log exit if entry was previously logged
                if hasattr(person, '_entry_logged') and person._entry_logged:
                    exit_visit_num = person.visit_number
                    self.event_logger.log_exit(person_id, exit_visit_num, self.current_video_file, self.frame_count)
                    logger.info(f"✓ Events: Exit logged: {person_id} (visit {exit_visit_num})")
                    logger.info(f"✓ DB: {person_id} exited (visit {exit_visit_num})")

                # Always clean up flags (but NOT _exit_processed — that stays until mark_online())
                for attr in ('_entry_logged', '_entry_frame', '_actual_exit', '_entry_logged_session_id'):
                    if hasattr(person, attr):
                        delattr(person, attr)

                # Clean up tracking data for this person
                with self.buffers_lock:
                    self.person_frame_buffers.pop(person_id, None)
                    self.person_first_detection_frame.pop(person_id, None)
                    self.person_frame_counters.pop(person_id, None)

                # Count total visits from database (all ENTRY events for this person)
                total_visits_result = self.db_manager.query(
                    "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED'",
                    (person_id,)
                )
                total_visits = total_visits_result[0][0] if total_visits_result else 1

                # Update person profile on exit
                self.db_manager.execute(
                    """UPDATE person_profiles
                    SET last_seen = ?, total_visits = ?, total_time_seconds = total_time_seconds + ?, updated_at = ?
                    WHERE person_id = ?""",
                    (datetime.now().isoformat(), total_visits, duration_seconds, datetime.now().isoformat(), person_id)
                )

        # Process frame buffers for embeddings
        for detection in detections:
            track_id = detection["track_id"]
            bbox = detection["bbox"]

            # Skip if this track_id was previously rejected (hand, noise, low confidence)
            if track_id in self.tracker.rejected_track_ids:
                continue

            if track_id in current_detections:
                person_id = current_detections[track_id]
                person = self.tracker.get_person(person_id)

                # Feed frame to background accumulator (before final_decision_made skip)
                if ACCUMULATOR_ENABLED and person is not None:
                    instance_id = person._instance_id
                    with self.accumulator_lock:
                        acc = self.active_accumulators.get(instance_id)
                    if acc is not None:
                        # Check retry eligibility first
                        if not acc.is_running and acc.can_retry and not acc.reached_target:
                            with acc.lock:
                                acc.is_running = True
                                acc.start_time = time.time()
                                acc.failed_attempts = 0
                                acc.retry_count += 1
                                acc.gave_up_at = None
                                # Reset per-reason counters for the new attempt
                                acc.rej_no_face = 0
                                acc.rej_blurry = 0
                                acc.rej_similar = 0
                                acc.rej_l2 = 0
                                acc.rej_cohesion = 0
                                acc.checks_since_last_summary = 0
                            logger.info(
                                f"🔄 ACCUMULATOR {person_id}: RETRY #{acc.retry_count}/{acc.max_retries} — "
                                f"resuming at {acc.current_total}/{acc.target_count} embeddings"
                            )
                        if acc.is_running and not acc.reached_target:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            h, w = frame.shape[:2]

                            # 🛡️ Anti-overlap: When people overlap, use center-focused crop
                            # Reduces chance of grabbing neighbor's face (e.g., Person 2's box grabbing Person 1's full face)
                            center_crop_ratio = 0.75  # Use center 75% of bbox (shrink edges by ~12%)
                            bbox_w = x2 - x1
                            bbox_h = y2 - y1
                            margin_x = int(bbox_w * (1 - center_crop_ratio) / 2)
                            margin_y = int(bbox_h * (1 - center_crop_ratio) / 2)
                            x1_centered = x1 + margin_x
                            y1_centered = y1 + margin_y
                            x2_centered = x2 - margin_x
                            y2_centered = y2 - margin_y

                            # Then add small padding for context (but less than before)
                            pad_y = int((y2_centered - y1_centered) * 0.10)
                            pad_x = int((x2_centered - x1_centered) * 0.10)
                            y1_pad = max(0, y1_centered - pad_y)
                            y2_pad = min(h, y2_centered + pad_y)
                            x1_pad = max(0, x1_centered - pad_x)
                            x2_pad = min(w, x2_centered + pad_x)
                            acc_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                            self._feed_accumulator(instance_id, person_id, person, acc_crop)

                # ✅ Skip buffer collection for confirmed persons with final decision
                if person is not None and getattr(person, '_final_decision_made', False):
                    continue

                # Check if we need to collect frames for embeddings
                with self.buffers_lock:
                    if person_id not in self.person_frame_buffers:
                        self.person_frame_buffers[person_id] = FrameBuffer()
                        self.person_first_detection_frame[person_id] = self.frame_count  # Track first detection
                        self.person_frame_counters[person_id] = 0

                    # increment per-person counter
                    self.person_frame_counters[person_id] += 1

                    # only add every Nth frame for diversity (N = FRAME_CAPTURE_STRIDE)
                    if self.person_frame_counters[person_id] % FRAME_CAPTURE_STRIDE == 0:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        h, w = frame.shape[:2]

                        # 🛡️ Anti-overlap: Use center-focused crop to avoid grabbing overlapping neighbor's face
                        center_crop_ratio = 0.75  # Use center 75% of bbox
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        margin_x = int(bbox_w * (1 - center_crop_ratio) / 2)
                        margin_y = int(bbox_h * (1 - center_crop_ratio) / 2)
                        x1_centered = x1 + margin_x
                        y1_centered = y1 + margin_y
                        x2_centered = x2 - margin_x
                        y2_centered = y2 - margin_y

                        # Add ~10% padding for face detection context (reduced from 20%)
                        pad_y = int((y2_centered - y1_centered) * 0.10)
                        pad_x = int((x2_centered - x1_centered) * 0.10)

                        y1_pad = max(0, y1_centered - pad_y)
                        y2_pad = min(h, y2_centered + pad_y)
                        x1_pad = max(0, x1_centered - pad_x)
                        x2_pad = min(w, x2_centered + pad_x)

                        person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                        self.person_frame_buffers[person_id].add(person_crop, self.frame_count)

                # Queue embeddings when buffer has ENOUGH frames for current batch (>= not ==)
                # CRITICAL: Check and set flags INSIDE lock to prevent duplicate queueing
                should_queue = False
                current_batch = 1
                current_size = 0
                frames_needed = 10

                with self.buffers_lock:
                    if person_id not in self.person_frame_buffers:
                        # Buffer was deleted by embedding thread
                        continue

                    buffer = self.person_frame_buffers[person_id]
                    current_batch = getattr(person, '_current_batch', 1) if person is not None else 1
                    current_size = len(buffer.frames)
                    frames_needed = current_batch * FRAMES_TO_COLLECT  # batch 1=5, 2=10, ..., 10=50

                    # Queue when buffer has enough frames for CURRENT batch (not exact count)
                    batch_ready = current_size >= frames_needed

                    # Check flags INSIDE lock (atomic operation)
                    already_queued = getattr(person, '_embeddings_queued', False)
                    already_computed = getattr(person, '_embeddings_computed', False)
                    final_decision_made = getattr(person, '_final_decision_made', False)

                    if batch_ready and not already_queued and not already_computed and not final_decision_made:
                        # Set flags INSIDE lock before queueing
                        if person is not None:
                            person._accumulating = True
                            person._embeddings_queued = True
                        should_queue = True

                if should_queue:
                    self.embedding_queue.put((person_id, person, None))
                    logger.debug(f"📤 Queued batch {current_batch} for {person_id} ({current_size}/{frames_needed} frames)")

        # Periodic orphan accumulator cleanup (every 300 frames)
        if ACCUMULATOR_ENABLED and self.frame_count % 300 == 0:
            self._cleanup_orphaned_accumulators()

        # Draw annotations
        annotated_frame = self._annotate_frame(frame, current_detections, detections)

        # Record frame
        if self.recording and self.video_writer and self.video_writer.isOpened():
            # Ensure frame is proper format (BGR, 3-channel, uint8)
            if annotated_frame is not None and annotated_frame.size > 0:
                try:
                    self.current_video_offset += 1
                    self.video_writer.write(annotated_frame)
                except Exception as e:
                    logger.warning(f"Failed to write frame to video: {e}")
                    self.recording = False
            else:
                logger.warning(f"Invalid frame for video write: shape={annotated_frame.shape if annotated_frame is not None else None}")

        # Write latest annotated frame to file so the API can serve it as live feed
        if annotated_frame is not None and annotated_frame.size > 0:
            try:
                latest_path = PROJECT_ROOT / "latest_frame.jpg"
                _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                latest_path.write_bytes(buf.tobytes())
            except Exception:
                pass

        return annotated_frame

    def _embedding_worker(self):
        """Background worker thread for embedding computation"""
        try:
            logger.info("🧵 Embedding worker thread started")
            while not self.embedding_stop_event.is_set():
                try:
                    person_id, person, is_preliminary = self.embedding_queue.get(timeout=1)
                    logger.debug(f"🧵 Processing embeddings for {person_id}")
                    try:
                        self._compute_and_store_embeddings(person_id, person, is_preliminary)
                    except Exception as e:
                        logger.error(f"Embedding computation error for {person_id}: {e}", exc_info=True)
                except:
                    pass  # Queue timeout, continue waiting

            # Drain remaining items from queue after stop event is set
            logger.info("🧵 Draining remaining queue items...")
            while True:
                try:
                    person_id, person, is_preliminary = self.embedding_queue.get(timeout=0.5)
                    logger.debug(f"🧵 Processing remaining embeddings for {person_id}")
                    try:
                        self._compute_and_store_embeddings(person_id, person, is_preliminary)
                    except Exception as e:
                        logger.error(f"Embedding computation error for {person_id}: {e}", exc_info=True)
                except:
                    break  # Queue empty or timeout, exit

            logger.info("🧵 Embedding worker thread stopping...")
        except Exception as e:
            logger.critical(f"🧵 FATAL ERROR IN EMBEDDING THREAD: {e}", exc_info=True)

    def _save_best_frames(self, person_id, best_face_frame, face_confidence):
        """Save best (highest confidence) face frame to best_one folder"""
        try:
            person_dir = BEST_FRAMES_DIR / person_id
            person_dir.mkdir(exist_ok=True)

            if best_face_frame is not None:
                face_path = person_dir / f"{person_id}_best_face_{face_confidence:.3f}.jpg"
                cv2.imwrite(str(face_path), best_face_frame)
                logger.debug(f"Saved best face frame (confidence: {face_confidence:.3f}): {face_path}")

        except Exception as e:
            logger.warning(f"Error saving best frames for {person_id}: {e}")

    def _save_body_crops(self, person_id: str, person, max_crops: int = 3):
        """Save YOLO body-crop frames to best_one/{person_id}/ for the UI gallery."""
        try:
            person_dir = BEST_FRAMES_DIR / person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            # Only save if we don't already have crops for this person
            existing = list(person_dir.glob("body_*.jpg"))
            if existing:
                return  # Already saved for a previous visit

            # Pull frames from the frame buffer for this person
            pid_for_buf = person.person_id  # may differ from person_id when returning visitor
            with self.buffers_lock:
                buffer = self.person_frame_buffers.get(pid_for_buf)
                frames = list(buffer.frames) if buffer else []

            if not frames:
                return

            # Pick evenly-spaced frames (up to max_crops)
            indices = [int(i * (len(frames) - 1) / max(max_crops - 1, 1)) for i in range(min(max_crops, len(frames)))]
            for n, idx in enumerate(indices):
                frame = frames[idx]
                if frame is None or frame.size == 0:
                    continue
                out_path = person_dir / f"body_{n+1:02d}.jpg"
                cv2.imwrite(str(out_path), frame)

            logger.debug(f"Saved {len(indices)} body crops for {person_id}")
        except Exception as e:
            logger.warning(f"Error saving body crops for {person_id}: {e}")

    def _search_at_checkpoint(self, person, checkpoint):
        """
        Search at checkpoint (10, 20, ..., 100) using face embeddings.
        At each checkpoint, search only with NEW 5 embeddings from current batch (not all accumulated).

        Args:
            person: Person object
            checkpoint: 10, 20, 30, 40, 50, 60, 70, 80, 90, or 100

        Returns:
            True if match found, False if no match
        """
        batch_num_for_checkpoint = checkpoint // 10  # checkpoint 10→1, 20→2, ..., 100→10
        batch_starts = getattr(person, 'batch_embedding_starts', {})
        stored_poses = getattr(person, 'face_embedding_poses', [])

        if batch_num_for_checkpoint in batch_starts:
            # Use recorded start index — accurate even when pose cap trims batches
            start = batch_starts[batch_num_for_checkpoint]
            next_batch = batch_num_for_checkpoint + 1
            end = batch_starts[next_batch] if next_batch in batch_starts else len(person.face_embeddings)
            face_embeds = person.face_embeddings[start:end]
            face_poses = list(stored_poses[start:end]) if stored_poses else []
        else:
            return False

        if not face_embeds:
            return False

        pose_summary = {p: face_poses.count(p) for p in set(face_poses)} if face_poses else {}
        logger.info(f"  Batch {batch_num_for_checkpoint} slice: {len(face_embeds)} embeddings (poses: {pose_summary})")

        # ═══════════════════════════════════════
        # STEP 1 — In-memory search FIRST (fast)
        # ═══════════════════════════════════════
        logger.info(f"  🔍 Searching in-memory offline_persons first...")

        with self.tracker.people_lock:
            offline_persons = {
                pid: p for pid, p in self.tracker.people.items()
                if p.tracking_state in ["LOST", "EXITED"]
                and not getattr(p, '_is_temporary', False)
                and p.face_embeddings
                and not p.is_in_frame
                and pid != person.person_id  # don't match against self
            }

        if offline_persons:
            logger.info(f"  📦 Found {len(offline_persons)} offline persons to search")
            matched_id = self.tracker._match_person(face_embeds, offline_persons, face_embedding_poses=face_poses)
            if matched_id:
                person._search_history[checkpoint] = "MATCH"
                person._matched_person_id = matched_id
                logger.info(f"  ✅ IN-MEMORY MATCH at checkpoint {checkpoint}: {matched_id}")
                return True
            else:
                logger.info(f"  ❌ No match in offline_persons, proceeding to Qdrant...")
        else:
            logger.info(f"  ⚠️ No offline persons found, skipping in-memory search")

        # ═══════════════════════════════════════
        # STEP 2 — Qdrant search (cross-session) — per-pose filtered
        # ═══════════════════════════════════════
        from collections import defaultdict as _dd
        face_best_distance = float('inf')
        face_best_match = None
        face_search_count = 0

        # Group batch embeddings by pose-group → search each group with its filter
        # FRONT / UP / DOWN share one group; LEFT and RIGHT are their own groups
        _POSE_GROUP_FILTER = {
            "FRONT": ["FRONT", "UP", "DOWN"],
            "UP":    ["FRONT", "UP", "DOWN"],
            "DOWN":  ["FRONT", "UP", "DOWN"],
            "LEFT":  ["LEFT"],
            "RIGHT": ["RIGHT"],
        }
        pose_groups = _dd(list)
        for emb, pose in zip(face_embeds, face_poses):
            group_key = tuple(_POSE_GROUP_FILTER.get(pose, [pose]))
            pose_groups[group_key].append(emb)

        # ✨ NEW: Get list of people currently in-frame (to exclude from search)
        in_frame_people = set()
        with self.tracker.people_lock:
            for pid, p in self.tracker.people.items():
                if p.is_in_frame and pid != person.person_id:  # Don't exclude self
                    in_frame_people.add(pid)

        if in_frame_people:
            logger.info(f"  🚫 Excluding in-frame people from Qdrant search: {in_frame_people}")

        for group_key, group_embeds in pose_groups.items():
            pose_label = "/".join(group_key)
            logger.info(f"  🔎 Qdrant search: pose={pose_label}, {len(group_embeds)} query embeddings")
            group_results = self.embedding_store.search_batch(
                "face_embeddings", group_embeds, pose_filter=list(group_key))

            for idx, result in enumerate(group_results):
                if result.get('distances') and result.get('metadatas'):
                    dist = result['distances'][0][0] if isinstance(result['distances'][0], list) else result['distances'][0]
                    meta = result['metadatas'][0][0] if isinstance(result['metadatas'][0], list) else result['metadatas'][0]
                    pid  = meta.get('person_id') if meta else "Unknown"
                    sim  = max(0, (1 - dist) * 100)

                    # ✨ NEW: Skip if this person is already in-frame
                    if pid in in_frame_people:
                        logger.info(f"    [{pose_label}] hit {idx}: {pid} (conf: {sim:.1f}%) ❌ SKIPPED (already in-frame)")
                        continue

                    logger.info(f"    [{pose_label}] hit {idx}: {pid} (conf: {sim:.1f}%)")
                    if pid and pid != "Unknown":
                        face_search_count += 1
                        if dist < face_best_distance:
                            face_best_distance = dist
                            face_best_match = pid

        face_similarity = max(0, (1 - face_best_distance) * 100) if face_best_distance != float('inf') else 0
        logger.info(f"  Face search complete: Searched {face_search_count}/{len(face_embeds)} embeddings, Similarity={face_similarity:.1f}%")

        if face_best_distance <= FACE_EMBEDDING_L2_THRESHOLD:
            person._search_history[checkpoint] = "MATCH"
            person._matched_person_id = face_best_match
            logger.info(f"  ✅ MATCH at checkpoint {checkpoint}: {face_best_match} ({face_similarity:.1f}%)")
            return True
        else:
            person._search_history[checkpoint] = "NO_MATCH"
            logger.info(f"  ❌ NO MATCH at checkpoint {checkpoint} (Face: {face_similarity:.1f}%)")
            return False

    def _finalize_person_id(self, person):
        """
        Determine final person_id after all 50 embeddings searched (10 batches).

        Returns:
            Final person_id (either matched person or the temporary person's ID)
        """
        if person._matched_person_id:
            # Found a match at checkpoint 10, 20, or 30
            final_id = person._matched_person_id
            logger.info(f"✅ FINAL DECISION: {person.person_id} (temp) → {final_id} (RETURNING VISITOR)")
        else:
            # No match found - new person
            final_id = person.person_id  # Keep temporary Person_4 (or whatever the number is)
            logger.info(f"✅ FINAL DECISION: {final_id} (NEW PERSON CONFIRMED)")

        return final_id

    @staticmethod
    def _estimate_pose(kps):
        """Estimate head pose label from 5 SCRFD keypoints (left_eye, right_eye, nose, left_mouth, right_mouth).

        Returns one of: FRONT / LEFT / RIGHT / UP / DOWN
        Uses ratio-based yaw/pitch — no 3D model needed.
        """
        if kps is None or len(kps) < 5:
            return "FRONT"
        le, re, nos, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
        eye_mid     = (le + re) / 2.0
        mouth_mid   = (lm + rm) / 2.0
        eye_width   = float(re[0] - le[0])
        face_height = float(mouth_mid[1] - eye_mid[1])
        if eye_width < 5 or face_height < 5:
            return "FRONT"
        yaw       = (nos[0] - eye_mid[0]) / eye_width
        pitch_dev = (nos[1] - eye_mid[1]) / face_height - 0.50
        if abs(yaw) < 0.12 and abs(pitch_dev) < 0.10:
            return "FRONT"
        if abs(yaw) >= abs(pitch_dev):
            return "RIGHT" if yaw > 0 else "LEFT"
        return "DOWN" if pitch_dev > 0 else "UP"

    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings (0-1)"""
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()

        norm1 = norm(emb1)
        norm2 = norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _compute_cosine_similarity(self, emb1, emb2):
        """Backward compatibility wrapper"""
        return self.cosine_similarity(emb1, emb2)

    def _apply_unsharp_mask(self, face_crop, strength=1.5, blur_radius=5):
        """
        Apply unsharp mask sharpening to face crop

        Process: blur → subtract from original → add back with strength
        Result: Enhanced edges = appears sharper
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop

        blurred = cv2.GaussianBlur(face_crop, (blur_radius, blur_radius), 0)
        sharpened = cv2.addWeighted(face_crop, strength, blurred, 1 - strength, 0)
        return sharpened

    def _compute_tenengrad_sharpness(self, face_crop):
        """
        Compute Tenengrad sharpness on face crop

        Tenengrad: gradient-based focus measure (Gx² + Gy²)
        Higher score = sharper image
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Tenengrad = sum of squared gradients
        g = np.sqrt(gx**2 + gy**2)
        tenengrad = np.sum(g)

        # Normalize by area
        area = face_crop.shape[0] * face_crop.shape[1]
        return tenengrad / area if area > 0 else 0.0

    def apply_filter2_sharpness(self, raw_embeddings, sharpness_threshold=15):
        """
        FILTER 2: SHARPNESS CHECK
        Remove blurry frames using Tenengrad on face crop
        Handles both: dict with metadata (test mode) and raw numpy arrays (production)
        """
        logger.info("\n" + "═" * 80)
        logger.info("🔧 FILTER 2: SHARPNESS CHECK (Tenengrad on Face Crop)")
        logger.info("═" * 80)

        passed = []
        rejected_blur = 0

        for i, raw_emb in enumerate(raw_embeddings):
            # Handle both dict (test mode) and numpy array (production)
            if isinstance(raw_emb, dict):
                frame_num = raw_emb.get('frame', i)
                sharpness = raw_emb.get('sharpness', 0)
                emb = raw_emb['embedding']

                if sharpness >= sharpness_threshold:
                    logger.info(f"  Frame {frame_num}: ✅ PASSED")
                    logger.info(f"             Filter 2 (Sharpness): {sharpness:.4f} ✓ (>= {sharpness_threshold})")
                    passed.append(emb)
                else:
                    rejected_blur += 1
                    logger.info(f"  Frame {frame_num}: ❌ REJECTED")
                    logger.info(f"             Filter 2 (Sharpness): {sharpness:.4f} < {sharpness_threshold} ✗ FAILED")
            else:
                # Production mode: raw numpy array (no metadata available)
                passed.append(raw_emb)

        logger.info(f"\n  📊 Filter 2 Summary: {len(raw_embeddings)} detected → {len(passed)} passed, {rejected_blur} rejected\n")
        return passed

    def apply_diversity_filter(self, embeddings, threshold=0.85):
        """
        FILTER 3: DIVERSITY CHECK
        Remove embeddings too similar to each other
        """
        if len(embeddings) < 2:
            return embeddings

        logger.info("\n" + "═" * 80)
        logger.info("🔧 FILTER 3: DIVERSITY CHECK")
        logger.info("═" * 80)

        diverse_indices = []

        for i in range(len(embeddings)):
            if i == 0:
                diverse_indices.append(0)
                logger.info(f"    Embedding {i:3d}: ✅ KEPT (first embedding, automatically selected)")
            else:
                current_emb = embeddings[i]
                max_sim = 0
                most_similar_idx = -1

                for j in diverse_indices:
                    selected_emb = embeddings[j]
                    sim = self.cosine_similarity(current_emb, selected_emb)
                    if sim > max_sim:
                        max_sim = sim
                        most_similar_idx = j

                if max_sim < threshold:
                    diverse_indices.append(i)
                    logger.info(f"    Embedding {i:3d}: ✅ KEPT (max_sim={max_sim:.4f} < {threshold} threshold)")
                else:
                    logger.info(f"    Embedding {i:3d}: ❌ REJECTED (too similar to embedding {most_similar_idx}, sim={max_sim:.4f} >= {threshold})")

        diverse_embeddings = [embeddings[i] for i in diverse_indices]
        removed = len(embeddings) - len(diverse_embeddings)

        logger.info(f"\n  📊 Filter 3 (Diversity) Summary:")
        logger.info(f"     Input:    {len(embeddings)} embeddings")
        logger.info(f"     Kept:     {len(diverse_embeddings)} embeddings")
        logger.info(f"     Rejected: {removed} embeddings (too similar)")
        logger.info(f"     Threshold: cosine_similarity < {threshold}\n")

        return diverse_embeddings

    def apply_l2_outlier_filter(self, embeddings, threshold=0.60):
        """
        FILTER 4: L2 OUTLIER DETECTION (dynamic threshold)
        Remove embeddings far from mean using mean + 1.5 * std
        """
        if len(embeddings) < 2:
            return embeddings

        logger.info("\n" + "═" * 80)
        logger.info("🔧 FILTER 4: L2 OUTLIER DETECTION (dynamic threshold)")
        logger.info("═" * 80)

        mean_emb = np.mean(embeddings, axis=0)
        l2_distances = [np.linalg.norm(e - mean_emb) for e in embeddings]
        l2_arr = np.array(l2_distances)

        mean_dist = np.mean(l2_arr)
        std_dist  = np.std(l2_arr)
        dynamic_threshold = mean_dist + 1.5 * std_dist

        logger.info(f"  L2 Stats:")
        logger.info(f"    Mean dist: {mean_dist:.4f}")
        logger.info(f"    Std dist:  {std_dist:.4f}")
        logger.info(f"    Threshold: {mean_dist:.4f} + 1.5 × {std_dist:.4f} = {dynamic_threshold:.4f}")

        inlier_indices = []
        for i, (emb, l2_dist) in enumerate(zip(embeddings, l2_distances)):
            if l2_dist < dynamic_threshold:
                inlier_indices.append(i)
                logger.info(f"    Embedding {i:3d}: ✅ KEPT     (L2={l2_dist:.4f} < {dynamic_threshold:.4f})")
            else:
                logger.info(f"    Embedding {i:3d}: ❌ REJECTED (L2={l2_dist:.4f} >= {dynamic_threshold:.4f} — outlier)")

        inlier_embeddings = [embeddings[i] for i in inlier_indices]
        removed = len(embeddings) - len(inlier_embeddings)

        logger.info(f"\n  📊 Filter 4 Summary: {len(embeddings)} → {len(inlier_embeddings)} kept, {removed} rejected\n")

        return inlier_embeddings

    def apply_min_cohesion_filter(self, embeddings, min_threshold=0.7, flog=None):
        """
        FILTER 5: MIN COHESION FILTER (Iterative)
        Remove outlier embeddings until minimum similarity >= threshold
        flog: optional file logger for verbose per-iteration logs (keeps terminal clean)
        """
        if len(embeddings) < 2:
            return embeddings

        def log(msg):
            if flog:
                flog.info(msg)

        log("\n" + "═" * 80)
        log("FILTER 5: MIN COHESION FILTER (Iterative)")
        log("═" * 80)

        filtered = list(embeddings)
        iteration = 0
        min_sim = 1.0

        while len(filtered) >= 2:
            iteration += 1

            min_sim = float('inf')
            worst_pair = (0, 0)

            for i in range(len(filtered)):
                for j in range(i + 1, len(filtered)):
                    sim = self.cosine_similarity(filtered[i], filtered[j])
                    if sim < min_sim:
                        min_sim = sim
                        worst_pair = (i, j)

            log(f"\n  Iteration {iteration}: Min similarity = {min_sim:.4f}")

            if min_sim >= min_threshold:
                log(f"  ✅ DONE! Min {min_sim:.4f} >= {min_threshold} threshold")
                break

            log(f"  ❌ Worst pair: Embedding {worst_pair[0]} & Embedding {worst_pair[1]} — sim={min_sim:.4f}")
            log(f"  Checking avg similarity of each to the whole group to find real outlier:")

            # Remove the embedding with the lowest average similarity to ALL others
            avg_sims = []
            for i in range(len(filtered)):
                others = [filtered[j] for j in range(len(filtered)) if j != i]
                avg_sim = np.mean([self.cosine_similarity(filtered[i], o) for o in others])
                avg_sims.append(avg_sim)
            removed_idx = int(np.argmin(avg_sims))

            for i, avg in enumerate(avg_sims):
                if i == removed_idx:
                    log(f"    Embedding {i:3d}: avg={avg:.4f}  ← lowest fit in group → REMOVE")
                elif i in worst_pair:
                    log(f"    Embedding {i:3d}: avg={avg:.4f}  ← was in worst pair but fits ok with others → keep")
                else:
                    log(f"    Embedding {i:3d}: avg={avg:.4f}")
            log(f"  → Removing embedding {removed_idx} (avg={avg_sims[removed_idx]:.4f})")

            filtered = [e for idx, e in enumerate(filtered) if idx != removed_idx]

            if len(filtered) < 2:
                log(f"  ⚠️  Only 1 embedding left after iteration {iteration}, stopping")
                break

        removed = len(embeddings) - len(filtered)
        logger.info(f"  Filter 5 (MinCohesion): {len(embeddings)} → {len(filtered)} kept, {removed} removed in {iteration} iterations (final min={min_sim:.4f})")
        log(f"\n  📊 Filter 5 Summary: input={len(embeddings)}, output={len(filtered)}, removed={removed}, iterations={iteration}, final_min={min_sim:.4f}\n")

        return filtered

    def _filter_embeddings_final(self, embeddings, frames=None, person_id=None, poses=None):
        """
        Apply 5-step quality filtering pipeline.

        FILTER 2: Sharpness (>= 15)
        FILTER 3: Diversity (cosine_sim < 0.85)
        FILTER 4: L2 Outliers — per-pose pool when multiple poses present
        FILTER 5: Min Cohesion — per-pose pool when multiple poses present

        poses: optional parallel list of pose labels (FRONT/LEFT/RIGHT/UP/DOWN).
               When provided, Filters 4+5 run per-pose so side-view embeddings
               aren't rejected for being dissimilar to the frontal cluster.
        """
        if not embeddings:
            return [], []

        # Per-person batch-filter file logger (verbose filter logs go here, not terminal)
        flog = None
        if person_id:
            import logging as _logging
            bf_log_dir = LOGS_DIR / "batch_filter"
            bf_log_dir.mkdir(exist_ok=True)
            bf_log_path = bf_log_dir / f"{person_id}.log"
            flog = _logging.getLogger(f"bf.{person_id}")
            flog.setLevel(_logging.INFO)
            flog.propagate = False
            if not flog.handlers:
                _fh = _logging.FileHandler(bf_log_path, encoding="utf-8")
                _fh.setFormatter(_logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
                flog.addHandler(_fh)

        logger.info("\n" + "█" * 80)
        logger.info("█ FILTERING PIPELINE (5-STEP)")
        logger.info("█" * 80 + "\n")

        # current_poses tracks pose label for each embedding as it passes through filters
        current_poses = list(poses) if poses and len(poses) == len(embeddings) else ["FRONT"] * len(embeddings)

        video_embeddings = list(embeddings)

        # Check if we have metadata (test mode) or raw arrays (production mode)
        has_metadata = isinstance(embeddings[0], dict) if embeddings else False

        # FILTER 2: Sharpness
        if has_metadata:
            # Test mode: Use stored sharpness metadata
            embeddings_after_filter2 = self.apply_filter2_sharpness(embeddings, sharpness_threshold=15)
            current_poses = current_poses[:len(embeddings_after_filter2)]  # metadata path drops poses (test only)
        else:
            # Production mode: Compute sharpness on-the-fly from frames if available
            if frames is not None and len(frames) == len(embeddings):
                logger.info("\n" + "═" * 80)
                logger.info("🔧 FILTER 2: SHARPNESS CHECK (Computing Tenengrad on-the-fly)")
                logger.info("═" * 80)

                embeddings_after_filter2 = []
                poses_after_filter2     = []
                for i, frame in enumerate(frames):
                    sharpened = self._apply_unsharp_mask(frame, strength=1.5, blur_radius=5)
                    sharpness = self._compute_tenengrad_sharpness(sharpened)

                    if sharpness >= 15:
                        embeddings_after_filter2.append(embeddings[i])
                        poses_after_filter2.append(current_poses[i])
                        logger.info(f"  Frame {i}: ✅ PASSED (sharpness={sharpness:.4f} >= 15, pose={current_poses[i]})")
                    else:
                        logger.info(f"  Frame {i}: ❌ REJECTED (sharpness={sharpness:.4f} < 15)")

                logger.info(f"\n  📊 Filter 2 Summary: {len(embeddings)} detected → {len(embeddings_after_filter2)} passed, {len(embeddings) - len(embeddings_after_filter2)} rejected\n")
                current_poses = poses_after_filter2
            else:
                # No frames available - skip Filter 2
                logger.info(f"⏭️  FILTER 2 SKIPPED (no frames available for sharpness computation)\n")
                embeddings_after_filter2 = embeddings

        logger.info(f"✅ After Filter 2: {len(embeddings_after_filter2)} embeddings\n")

        if len(embeddings_after_filter2) == 0:
            logger.warning(f"⚠️  No embeddings passed Filter 2\n")
            return [], []

        video_embeddings = embeddings_after_filter2

        # FILTER 3: Diversity (flat — cross-pose frames are already different, won't be rejected)
        if len(video_embeddings) >= 2:
            video_embeddings_new = self.apply_diversity_filter(video_embeddings, threshold=0.85)
            # Keep poses in sync: apply_diversity_filter returns same objects by reference (no copies)
            kept_set = set(id(e) for e in video_embeddings_new)
            current_poses = [p for e, p in zip(video_embeddings, current_poses) if id(e) in kept_set]
            video_embeddings = video_embeddings_new
        else:
            logger.info(f"⏭️  FILTER 3 SKIPPED (only {len(video_embeddings)} embedding(s))\n")

        if len(video_embeddings) == 0:
            logger.warning(f"⚠️  No embeddings passed Filter 3\n")
            return [], []

        # Log pose distribution entering Filters 4+5
        pose_dist = {p: current_poses.count(p) for p in set(current_poses)}
        logger.info(f"  Pose distribution entering F4+F5: {pose_dist}")
        use_per_pose = len(set(current_poses)) > 1
        if use_per_pose:
            logger.info(f"  Multiple poses detected — running Filters 4+5 per-pose pool")
        if flog:
            flog.info(f"Pose distribution: {pose_dist}  per_pose_mode={use_per_pose}")

        # FILTER 4+5: Per-pose when multiple poses present, flat otherwise
        final_poses = []
        if use_per_pose:
            from collections import defaultdict
            pose_groups = defaultdict(list)
            for emb, p in zip(video_embeddings, current_poses):
                pose_groups[p].append(emb)

            merged = []
            merged_poses = []
            for p_label, p_embs in pose_groups.items():
                logger.info(f"\n  ── Pose pool: {p_label} ({len(p_embs)} embeddings) ──")
                if flog:
                    flog.info(f"\n{'─'*40}\nPose pool: {p_label} ({len(p_embs)} embeddings)\n{'─'*40}")

                # F4: L2 per-pose
                if len(p_embs) >= 2:
                    p_embs = self.apply_l2_outlier_filter(p_embs, threshold=0.80)
                else:
                    logger.info(f"  ⏭️  F4 {p_label} SKIPPED (only {len(p_embs)} embedding)")

                if len(p_embs) == 0:
                    logger.warning(f"  ⚠️  No embeddings left in {p_label} pool after F4")
                    continue

                # F5: Min cohesion per-pose
                if len(p_embs) >= 2:
                    p_embs = self.apply_min_cohesion_filter(p_embs, min_threshold=0.50, flog=flog)
                else:
                    logger.info(f"  ⏭️  F5 {p_label} SKIPPED (only {len(p_embs)} embedding)")

                merged.extend(p_embs)
                merged_poses.extend([p_label] * len(p_embs))

            video_embeddings = merged
            final_poses = merged_poses
        else:
            # All same pose — flat filters as before
            solo_pose = current_poses[0] if current_poses else "FRONT"
            if len(video_embeddings) >= 2:
                video_embeddings = self.apply_l2_outlier_filter(video_embeddings, threshold=0.80)
            else:
                logger.info(f"⏭️  FILTER 4 SKIPPED (only {len(video_embeddings)} embedding(s))\n")

            if len(video_embeddings) == 0:
                logger.warning(f"⚠️  No embeddings passed Filter 4\n")
                return [], []

            if len(video_embeddings) >= 2:
                video_embeddings = self.apply_min_cohesion_filter(video_embeddings, min_threshold=0.50, flog=flog)
            else:
                logger.info(f"⏭️  FILTER 5 SKIPPED (only {len(video_embeddings)} embedding(s))\n")

            final_poses = [solo_pose] * len(video_embeddings)

        # FILTER 6: Pose cap trim — only runs if any pose has > 7 embeddings after F5
        # Removes the embedding with lowest average similarity to the rest of its pose group
        POSE_CAP = 7
        pose_counts_after_f5 = {p: final_poses.count(p) for p in set(final_poses)}
        needs_trim = any(c > POSE_CAP for c in pose_counts_after_f5.values())
        if needs_trim:
            logger.info(f"\n🔧 FILTER 6: POSE CAP TRIM (max {POSE_CAP} per pose)")
            trimmed_embs = list(video_embeddings)
            trimmed_poses = list(final_poses)
            for p_label, count in pose_counts_after_f5.items():
                if count <= POSE_CAP:
                    continue
                logger.info(f"  Pose {p_label}: {count} > {POSE_CAP} — trimming {count - POSE_CAP}")
                while True:
                    pose_indices = [i for i, p in enumerate(trimmed_poses) if p == p_label]
                    if len(pose_indices) <= POSE_CAP:
                        break
                    pose_vecs = [trimmed_embs[i] for i in pose_indices]
                    # Find the embedding with lowest average similarity to the rest in its group
                    worst_idx_in_group = 0
                    worst_avg = float('inf')
                    for gi, emb in enumerate(pose_vecs):
                        others = [pose_vecs[j] for j in range(len(pose_vecs)) if j != gi]
                        avg_sim = float(np.mean([self.cosine_similarity(emb, o) for o in others]))
                        if avg_sim < worst_avg:
                            worst_avg = avg_sim
                            worst_idx_in_group = gi
                    remove_idx = pose_indices[worst_idx_in_group]
                    logger.info(f"    Removed {p_label}[{worst_idx_in_group}] (avg_sim={worst_avg:.3f}) — least cohesive")
                    trimmed_embs.pop(remove_idx)
                    trimmed_poses.pop(remove_idx)
            video_embeddings = trimmed_embs
            final_poses = trimmed_poses
        else:
            logger.info(f"⏭️  FILTER 6 SKIPPED (no pose exceeds {POSE_CAP})\n")

        logger.info("█" * 80)
        logger.info(f"✅ FILTERING COMPLETE: {len(video_embeddings)} final embeddings | poses: { {p: final_poses.count(p) for p in set(final_poses)} }")
        logger.info("█" * 80 + "\n")

        return video_embeddings, final_poses

    def _store_all_embeddings(self, final_person_id, person):
        """
        Store all filtered embeddings to Qdrant after final decision.

        Args:
            final_person_id: Confirmed person ID to store under
            person: Person object with embeddings
        """
        # Store face embeddings metadata once per person
        if person.face_embeddings:
            logger.info(f"\n🔐 STORAGE PIPELINE START for {final_person_id}")
            logger.info(f"  Original embeddings: {len(person.face_embeddings)}")

            # Apply filtering pipeline (frames optional, will skip Step 0 if unavailable)
            frames = None
            if person.person_id in self.person_frame_buffers:
                buffer_for_filtering = self.person_frame_buffers[person.person_id]
                frames = buffer_for_filtering.frames
                logger.info(f"🔍 DEBUG: Buffer has {len(frames)} frames, {len(person.face_embeddings)} embeddings")
            poses = getattr(person, 'face_embedding_poses', None)
            filtered_embeddings, filtered_poses = self._filter_embeddings_final(
                person.face_embeddings, frames, person_id=final_person_id, poses=poses)

            # Update person with filtered embeddings and their poses
            person.face_embeddings = filtered_embeddings
            person.face_embedding_poses = filtered_poses
            logger.info(f"  Filtered embeddings: {len(filtered_embeddings)} | poses: { {p: filtered_poses.count(p) for p in set(filtered_poses)} if filtered_poses else {} }")

            # ✨ DISABLED: Store filtered face embeddings to Qdrant (only if SAVE_NEW_EMBEDDINGS enabled)
            stored_count = 0
            if SAVE_NEW_EMBEDDINGS:
                for idx, (embedding, pose_label) in enumerate(zip(filtered_embeddings, filtered_poses)):
                    try:
                        self.embedding_store.add_face_embedding(
                            final_person_id,
                            embedding,
                            {"embedding_index": idx + 1, "total_embeddings": len(filtered_embeddings), "pose": pose_label}
                        )
                        stored_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to store face embedding {idx + 1}: {e}")

                logger.info(f"✅ FINALIZED & STORED {stored_count}/{len(filtered_embeddings)} face embeddings to Qdrant as {final_person_id}\n")
            else:
                logger.info(f"⏭️  SKIPPED STORAGE: {len(filtered_embeddings)} embeddings NOT saved (SAVE_NEW_EMBEDDINGS=False)\n")
                logger.info(f"   Embeddings are computed for matching ONLY against existing people (jilani, person 1, person 2)\n")

            # Save body crops for the UI gallery (2-3 frames from the frame buffer)
            self._save_body_crops(final_person_id, person)

            # Update embedding_metadata with actual stored count
            # ON CONFLICT: accumulate frame_count across visits (true Qdrant total per person)
            try:
                self.db_manager.execute(
                    """INSERT INTO embedding_metadata
                        (person_id, embedding_type, first_seen, frame_count, quality)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(person_id, embedding_type) DO UPDATE SET
                        frame_count = frame_count + excluded.frame_count,
                        updated_at  = CURRENT_TIMESTAMP""",
                    (final_person_id, "face", datetime.now().isoformat(), stored_count, "high")
                )
            except Exception as e:
                logger.debug(f"Failed to update embedding metadata: {e}")

    def _compute_and_store_embeddings(self, person_id, person, is_preliminary=False):
        """Compute embeddings and apply progressive matching strategy.

        Searches for person match at 5 decision points (10, 20, 30, 40, 50 embeddings).
        Storage happens ONLY AFTER final decision with correct person_id.
        Filtering applied before storage to reduce false positives.

        Process:
        - Batch 1: Collect 5 face embeddings, search at checkpoint 10
        - Batch 2: Collect 5 more, search at checkpoint 20
        - Batch 3: Collect 5 more, search at checkpoint 30
        - Batch 4: Collect 5 more, search at checkpoint 40
        - Batch 5: Collect 5 more, search at checkpoint 50 (final decision)
        - After Batch 5: Apply 5-step filtering (Sharpness→Diversity→L2→MinCohesion), store filtered embeddings under final person_id

        Args:
            person_id: Person identifier (may be temporary)
            person: Person object
            is_preliminary: Ignored
        """
        try:
            logger.debug(f"✓ _compute_and_store_embeddings called for {person_id}")

            # Check if already made final decision
            if hasattr(person, '_final_decision_made') and person._final_decision_made:
                logger.debug(f"  Skipped: {person_id} already finalized")
                return

            # Get buffer reference atomically
            with self.buffers_lock:
                if person_id not in self.person_frame_buffers:
                    logger.debug(f"  Skipped: {person_id} not in buffers")
                    return
                buffer = self.person_frame_buffers[person_id]

            # Get the current batch number from the person's state (set by _process_batch in person_tracker)
            # This is more reliable than deriving from buffer size
            batch_num = getattr(person, '_current_batch', 1) if person is not None else 1
            frame_count = len(buffer.frames)
            frames_needed = batch_num * FRAMES_TO_COLLECT  # batch 1=5, 2=10, ..., 10=50

            # Validate we have enough frames for this batch
            if frame_count < frames_needed:
                logger.debug(f"Not enough frames for batch {batch_num}: have {frame_count}, need {frames_needed}")
                return  # Wait for more frames to accumulate

            # Get actual video frame numbers for this batch (5 frames per batch)
            frames_per_batch = FRAMES_TO_COLLECT
            start = (batch_num - 1) * frames_per_batch
            batch_frame_numbers = tuple(buffer.frame_numbers[start:start + frames_per_batch])

            # Log motion detection FIRST (batch 1 only says "created", batches 2-3 say "continuing")
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            if batch_num == 1:
                logger.info(f"🆕 MOTION DETECTED - Temporary person created: {person_id}")
            else:
                logger.info(f"🔄 MOTION CONTINUING - Processing batch {batch_num} for: {person_id}")
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Then log batch reception with actual video frame numbers
            frame_range_str = f"{batch_frame_numbers[0]}-{batch_frame_numbers[-1]}"
            logger.info(f"✓ GOT BATCH {batch_num} ({frame_range_str} frames) for {person_id}: {batch_frame_numbers}")

            # Validate person exists
            if person is None:
                logger.warning(f"Person {person_id} not found - skipping embeddings")
                return

            # Check if person is still in tracker
            with self.tracker.people_lock:
                if person_id not in self.tracker.people and not (hasattr(person, '_is_temporary') and person._is_temporary):
                    logger.debug(f"Person {person_id} no longer tracked, skipping embeddings")
                    return

            # Skip if already computed
            if hasattr(person, '_embeddings_computed'):
                return

            # Initialize checkpoint attributes if needed
            if not hasattr(person, '_search_history'):
                person._search_history = {10: None, 20: None, 30: None, 40: None, 50: None, 60: None, 70: None, 80: None, 90: None, 100: None}
            if not hasattr(person, '_matched_person_id'):
                person._matched_person_id = None
            if not hasattr(person, '_final_decision_made'):
                person._final_decision_made = False
            if not hasattr(person, 'face_embeddings'):
                person.face_embeddings = []

            # Take snapshot of ONLY the relevant frames for THIS batch (not all frames in buffer)
            n = FRAMES_TO_COLLECT  # 5 frames per batch, 10 batches = 50 total
            with self.buffers_lock:
                start = (batch_num - 1) * n
                frames_snapshot = list(buffer.frames[start:start + n])

            # COMPUTE FACE EMBEDDINGS FIRST to check quality
            logger.info(f"\n🔄 COMPUTING FACE EMBEDDINGS for {person_id}")
            logger.info(f"  Processing {len(frames_snapshot)} collected frames...")
            face_embeddings = []
            face_confidences = []

            with self.embedding_lock:
                emb_list, conf_list, kps_list = self.embeddings.compute_face_embeddings_batch(frames_snapshot)

            batch_poses = []  # pose per valid embedding in this batch
            for idx, (emb, conf, kps) in enumerate(zip(emb_list, conf_list, kps_list)):
                if emb is not None:
                    face_embeddings.append(emb)
                    face_confidences.append(conf)
                    pose_label = self._estimate_pose(kps)
                    batch_poses.append(pose_label)
                    logger.info(f"  {person_id} Frame {idx+1}/{len(frames_snapshot)}: ✓ Face detected (confidence: {conf:.2%}, pose: {pose_label})")
                else:
                    logger.info(f"  {person_id} Frame {idx+1}/{len(frames_snapshot)}: ❌ No face detected")

            logger.info(f"Face embeddings computed for {person_id}: {len(face_embeddings)}/{len(frames_snapshot)}")

            # Calculate average face confidence
            avg_face_confidence = (sum(face_confidences) / len(face_confidences)) if face_confidences else 0.0
            logger.info(f"Average face confidence for {person_id}: {avg_face_confidence:.2%}")

            # Filter face embeddings (keep up to 50 total across all 10 batches)
            face_embeddings_to_store = face_embeddings
            poses_to_store = list(batch_poses)
            if len(face_embeddings) > 50:
                if face_embeddings:
                    selected_indices   = [0]
                    selected_embeddings = [face_embeddings[0]]
                    selected_poses      = [batch_poses[0]]
                else:
                    selected_embeddings = []
                    selected_poses      = []

                for i in range(1, len(face_embeddings)):
                    if len(selected_indices) >= 50:
                        break
                    current_emb = face_embeddings[i]
                    min_similarity = float('inf')

                    for selected_emb in selected_embeddings:
                        sim = self._compute_cosine_similarity(current_emb, selected_emb)
                        if sim < min_similarity:
                            min_similarity = sim

                    if min_similarity < 0.95:
                        selected_indices.append(i)
                        selected_embeddings.append(current_emb)
                        selected_poses.append(batch_poses[i])

                face_embeddings_to_store = selected_embeddings
                poses_to_store = selected_poses

            # Add face embeddings to person object BEFORE checkpoint logic
            # No pose cap here — collect all frames freely; F6 trims to 7 per pose at storage time
            with self.embedding_lock:
                if not hasattr(person, 'face_embedding_poses'):
                    person.face_embedding_poses = []
                if not hasattr(person, 'batch_embedding_starts'):
                    person.batch_embedding_starts = {}
                batch_start_idx = len(person.face_embeddings)
                person.batch_embedding_starts[batch_num] = batch_start_idx
                for emb, pose_lbl in zip(face_embeddings_to_store, poses_to_store):
                    person.face_embeddings.append(emb)
                    person.face_embedding_poses.append(pose_lbl)
                pose_counts = {p: person.face_embedding_poses.count(p) for p in set(person.face_embedding_poses)}
                logger.info(f"📦 {person_id} in-memory: {len(person.face_embeddings)} face embeddings (batch {batch_num}) | poses: {pose_counts}")

            # 🎯 BATCH PROCESSING - Checkpoint Detection & Matching
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"📊 BATCH {batch_num} PROCESSING for {person_id}")
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Flag already set in process_frame before queueing to prevent race condition with _check_timeouts
            person._current_batch = batch_num

            # Process checkpoints based on BATCH NUMBER (not body embedding count)
            checkpoint_decided = False

            # ===== BATCH 1: CONFIDENCE GATING CHECKPOINT =====
            if batch_num == 1 and person._search_history[10] is None:
                logger.info(f"🔵 BATCH 1 CONFIDENCE CHECK for {person_id}")
                logger.info(f"  Face embeddings collected: {len(face_embeddings)}")
                logger.info(f"  Average face confidence: {avg_face_confidence:.2%}")

                if avg_face_confidence >= FACE_CONFIDENCE_THRESHOLD:
                    # ✅ CONFIRMED: This is a real person
                    logger.info(f"✅ {person_id}: CONFIRMED as REAL PERSON")
                    logger.info(f"  Confidence {avg_face_confidence:.2%} >= {FACE_CONFIDENCE_THRESHOLD:.0%} - Proceeding to search")
                    logger.info(f"🔍 CHECKPOINT 10 SEARCH: Searching with {FRAMES_TO_COLLECT} face embeddings (batch 1)")
                    checkpoint_decided = self._search_at_checkpoint(person, 10)

                    if checkpoint_decided:
                        logger.info(f"✅✅ MATCH FOUND at Checkpoint 10!")
                        logger.info(f"  {person_id} matches existing person: {person._matched_person_id}")
                        logger.info(f"  Result: RETURNING VISITOR DETECTED")

                        # Save matched_person_id BEFORE any state changes
                        matched_person_id = person._matched_person_id
                        temp_person_id = person_id

                        # Set protection flags
                        person._is_temporary = False
                        person._confirming = True
                        person._accumulating = True
                        person._final_decision_made = True

                        # Clear queued flag
                        if hasattr(person, '_embeddings_queued'):
                            delattr(person, '_embeddings_queued')

                        # Finalize person ID
                        final_person_id = self._finalize_person_id(person)

                        # ✨ NEW: Per-frame uniqueness check
                        # Prevent assigning same person_id to two different people in same frame
                        with self.tracker.people_lock:
                            target_person = self.tracker.people.get(final_person_id)

                        if target_person and target_person.is_in_frame and target_person != person:
                            logger.warning(f"🚨 CONFLICT: {final_person_id} already in frame! Re-verifying...")

                            # Get embeddings of both people
                            current_person_embeds = person.face_embeddings if hasattr(person, 'face_embeddings') else []
                            target_person_embeds = target_person.face_embeddings if hasattr(target_person, 'face_embeddings') else []

                            # Resolve: which one is truly final_person_id?
                            final_person_id_for_current, final_person_id_for_target = self.tracker.resolve_identity_conflict(
                                person.person_id, current_person_embeds,
                                final_person_id, target_person_embeds
                            )

                            # Apply resolution
                            if final_person_id_for_current == "NEW":
                                logger.info(f"   → {person.person_id} is FALSE positive, assigning new ID")
                                final_person_id = person.person_id  # Keep as new person
                            else:
                                logger.info(f"   → {person.person_id} is TRUE {final_person_id}, proceeding with merge")

                            # If target person needs new ID (rare case)
                            if final_person_id_for_target == "NEW":
                                logger.info(f"   → {final_person_id} is FALSE positive, needs reassignment")
                                # Note: This would require complex logic to reassign target person
                                # For now, we keep current resolution

                        # Skip storage for returning visitors
                        if final_person_id != person.person_id:
                            logger.info(f"⏭️  Returning visitor {final_person_id} — skipping storage (already in DB)")
                        else:
                            logger.info(f"✅ New person {final_person_id} — will store embeddings")

                        # Clean up temp person — use saved IDs (temp_person_id and matched_person_id)
                        self._cleanup_temp_after_match(temp_person_id, matched_person_id)

                        # Get or create the matched person (reuse if same ID)
                        matched_person = self.tracker.get_person(matched_person_id)
                        if matched_person is None:
                            # Truly different person — create new instance
                            logger.info(f"🔢 ID counter BEFORE create_for_existing: {Person._id_counter}, free_pool: {Person._free_temp_ids}")
                            matched_person = Person.create_for_existing(matched_person_id)
                            logger.info(f"🔢 ID counter AFTER create_for_existing: {Person._id_counter}, free_pool: {Person._free_temp_ids}, created_id: {matched_person.person_id}")
                            try:
                                with self.tracker.people_lock:
                                    self.tracker.people[matched_person_id] = matched_person
                            except RuntimeError:
                                pass

                        absence_seconds = (datetime.now() - matched_person.lost_time).total_seconds() if matched_person.lost_time else (NEW_VISIT_THRESHOLD_SECONDS + 1)
                        same_visit = absence_seconds < NEW_VISIT_THRESHOLD_SECONDS

                        # Initialize visit_number — only increment if this is a new visit
                        if (not hasattr(matched_person, 'visit_number') or matched_person.visit_number == 1) and not same_visit:
                            visit_result = self.db_manager.query(
                                "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                (matched_person_id, self.session_id)
                            )
                            matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1
                        if same_visit:
                            logger.info(f"✅ {matched_person_id} REACTIVATED — returned within {absence_seconds:.1f}s (same visit, no new entry logged)")
                        else:
                            logger.info(f"🔄 {matched_person_id} returned after {absence_seconds:.1f}s — new visit")

                        matched_person.mark_online()
                        matched_person._matched_person_id = None  # ← clear after confirmation
                        matched_person._accumulating = False
                        matched_person._confirming = False
                        matched_person._embeddings_computed = True  # Mark batch complete
                        matched_person._final_decision_made = True  # Mark decision final

                        # Remove temp flags from matched person
                        for attr in ('_is_temporary', '_confirming', '_accumulating', '_embeddings_queued', '_current_batch'):
                            if hasattr(matched_person, attr):
                                delattr(matched_person, attr)

                        # Log entry event for returning visitor (skip if same visit reactivation)
                        with self.entry_logging_lock:
                            already_logged = (
                                hasattr(matched_person, '_entry_logged') and
                                matched_person._entry_logged and
                                getattr(matched_person, '_entry_logged_session_id', None) == self.session_id
                            )
                            if not already_logged and not same_visit:
                                # Recalculate visit_number before logging
                                visit_result = self.db_manager.query(
                                    "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                    (matched_person_id, self.session_id)
                                )
                                matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1

                                self.event_logger.log_entry(matched_person_id, matched_person.visit_number, self.current_video_file, self.frame_count)
                                logger.info(f"✅ Entry logged: {matched_person_id} (visit {matched_person.visit_number})")

                                # Set flags AFTER logging (not before)
                                matched_person._entry_logged = True
                                matched_person._entry_logged_session_id = self.session_id
                                matched_person._entry_frame = self.frame_count

                        # Load face embeddings from Qdrant if not in memory
                        if not matched_person.face_embeddings:
                            try:
                                stored_face = self.embedding_store.get_embeddings_for_person(matched_person_id)
                                if stored_face:
                                    matched_person.face_embeddings = stored_face
                                    logger.info(f"📦 {matched_person_id} in-memory: {len(stored_face)} face embeddings loaded from Qdrant")
                                else:
                                    logger.info(f"📦 {matched_person_id} in-memory: keeping {len(matched_person.face_embeddings)} face embeddings")
                            except Exception as e:
                                logger.warning(f"Failed to load embeddings from Qdrant for {matched_person_id}: {e}")
                        else:
                            logger.info(f"📦 {matched_person_id} already has {len(matched_person.face_embeddings)} face embeddings in memory — skipping Qdrant load")

                        # Start accumulator if returning visitor needs more embeddings
                        if ACCUMULATOR_ENABLED and len(matched_person.face_embeddings) < ACCUMULATOR_TARGET_EMBEDDINGS:
                            self._start_background_accumulator(matched_person_id, matched_person)

                        # Update person profile
                        self.db_manager.execute(
                            """INSERT OR IGNORE INTO person_profiles
                            (person_id, first_seen, last_seen, total_visits, total_time_seconds, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (matched_person_id, datetime.now().isoformat(), datetime.now().isoformat(), 1, 0, datetime.now().isoformat())
                        )
                        self.db_manager.execute(
                            """UPDATE person_profiles SET last_seen=?, total_visits=?, updated_at=? WHERE person_id=?""",
                            (datetime.now().isoformat(), matched_person.visit_number, datetime.now().isoformat(), matched_person_id)
                        )

                        # Mark as fully processed
                        person._embeddings_computed = True
                        person._accumulating = False
                        logger.info(f"✅ Person confirmed as returning visitor — stopping batch collection early")
                        return  # CRITICAL: Stop here, don't continue to batch 2 or 3

                    else:
                        # ❌ NO MATCH — continue to batch 2
                        logger.info(f"❌ No match at Checkpoint 10")
                        logger.info(f"  {person_id} is likely a NEW PERSON")
                        logger.info(f"  Proceeding to collect more frames for batch 2...")

                        # ✅ CLEAR FLAG to allow batch 2 to queue (only if NO match found)
                        if hasattr(person, '_embeddings_queued'):
                            delattr(person, '_embeddings_queued')
                            logger.info(f"✅ Cleared _embeddings_queued flag for {person_id} - batch 2 CAN NOW QUEUE")
                        else:
                            logger.info(f"⚠️ Flag already cleared for {person_id}")

                        # 📊 ADVANCE TO BATCH 2
                        person._current_batch = 2
                        logger.info(f"📊 Advancing to BATCH 2 - {person_id} will collect frames 11-20")
                else:
                    # ❌ REJECTED: Low confidence (hand, noise, etc.)
                    logger.info(f"❌ {person_id}: REJECTED - LOW FACE CONFIDENCE")
                    logger.info(f"  Confidence: {avg_face_confidence:.2%} < {FACE_CONFIDENCE_THRESHOLD:.0%}")
                    logger.info(f"  Classification: HAND/NOISE/FALSE DETECTION")
                    logger.info(f"  Action: Discarding person completely")
                    logger.info(f"  - Clearing frame buffer")
                    logger.info(f"  - Removing from tracker")
                    logger.info(f"  - NOT storing any embeddings")

                    # Clear buffer for low confidence detection
                    with self.buffers_lock:
                        self.person_frame_buffers.pop(person_id, None)
                        self.person_first_detection_frame.pop(person_id, None)
                        self.person_frame_counters.pop(person_id, None)

                    # Remove from tracker
                    try:
                        with self.tracker.people_lock:
                            if person_id in self.tracker.people:
                                del self.tracker.people[person_id]
                    except (KeyError, RuntimeError):
                        pass

                    # Mark all track_ids pointing to this person as rejected
                    # So same hand/noise won't be reprocessed
                    for track_id, pid in list(self.tracker.bytetrack_cache.items()):
                        if pid == person_id:
                            self.tracker.rejected_track_ids.add(track_id)
                            logger.debug(f"Marked track {track_id} as rejected (low face confidence)")
                            break

                    person._search_history[10] = "REJECTED_LOW_CONFIDENCE"
                    return  # Stop processing this person

            # ===== BATCH 2: Continue Search (if batch 1 passed) =====
            elif batch_num == 2 and person._search_history[20] is None:
                logger.info(f"🟡 BATCH 2 PROCESSING for {person_id}")
                logger.info(f"  (Batch 1 already passed confidence check)")
                logger.info(f"  Face embeddings collected (frames 10-19): {len(face_embeddings)}")
                logger.info(f"🔍 CHECKPOINT 20 SEARCH: Searching with {FRAMES_TO_COLLECT} face embeddings (batch 2)")
                checkpoint_decided = self._search_at_checkpoint(person, 20)

                if checkpoint_decided:
                    logger.info(f"✅✅ MATCH FOUND at Checkpoint 20!")
                    logger.info(f"  {person_id} matches existing person: {person._matched_person_id}")
                    logger.info(f"  Result: RETURNING VISITOR DETECTED")

                    # Save matched_person_id BEFORE any state changes
                    matched_person_id = person._matched_person_id
                    temp_person_id = person_id

                    # Set protection flags
                    person._is_temporary = False
                    person._confirming = True
                    person._accumulating = True
                    person._final_decision_made = True

                    # Clear queued flag
                    if hasattr(person, '_embeddings_queued'):
                        delattr(person, '_embeddings_queued')

                    # Finalize person ID
                    final_person_id = self._finalize_person_id(person)

                    # ✨ NEW: Per-frame uniqueness check
                    with self.tracker.people_lock:
                        target_person = self.tracker.people.get(final_person_id)

                    if target_person and target_person.is_in_frame and target_person != person:
                        logger.warning(f"🚨 CONFLICT: {final_person_id} already in frame! Re-verifying...")

                        current_person_embeds = person.face_embeddings if hasattr(person, 'face_embeddings') else []
                        target_person_embeds = target_person.face_embeddings if hasattr(target_person, 'face_embeddings') else []

                        final_person_id_for_current, final_person_id_for_target = self.tracker.resolve_identity_conflict(
                            person.person_id, current_person_embeds,
                            final_person_id, target_person_embeds
                        )

                        if final_person_id_for_current == "NEW":
                            logger.info(f"   → {person.person_id} is FALSE positive, staying as new person")
                            final_person_id = person.person_id

                    # Skip storage for returning visitors
                    if final_person_id != person.person_id:
                        logger.info(f"⏭️  Returning visitor {final_person_id} — skipping storage (already in DB)")
                    else:
                        logger.info(f"✅ New person {final_person_id} — will store embeddings")

                    # Clean up temp person — use saved IDs (temp_person_id and matched_person_id)
                    self._cleanup_temp_after_match(temp_person_id, matched_person_id)

                    # Get or create the matched person (reuse if same ID)
                    matched_person = self.tracker.get_person(matched_person_id)
                    if matched_person is None:
                        # Truly different person — create new instance
                        matched_person = Person.create_for_existing(matched_person_id)
                        try:
                            with self.tracker.people_lock:
                                self.tracker.people[matched_person_id] = matched_person
                        except RuntimeError:
                            pass

                    absence_seconds = (datetime.now() - matched_person.lost_time).total_seconds() if matched_person.lost_time else (NEW_VISIT_THRESHOLD_SECONDS + 1)
                    same_visit = absence_seconds < NEW_VISIT_THRESHOLD_SECONDS

                    # Initialize visit_number — only increment if this is a new visit
                    if (not hasattr(matched_person, 'visit_number') or matched_person.visit_number == 1) and not same_visit:
                        visit_result = self.db_manager.query(
                            "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                            (matched_person_id, self.session_id)
                        )
                        matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1
                    if same_visit:
                        logger.info(f"✅ {matched_person_id} REACTIVATED — returned within {absence_seconds:.1f}s (same visit, no new entry logged)")
                    else:
                        logger.info(f"🔄 {matched_person_id} returned after {absence_seconds:.1f}s — new visit")
                    matched_person.mark_online()
                    matched_person._accumulating = False
                    matched_person._confirming = False
                    matched_person._embeddings_computed = True  # Mark batch complete
                    matched_person._final_decision_made = True  # Mark decision final

                    # Remove temp flags from matched person
                    for attr in ('_is_temporary', '_confirming', '_accumulating', '_embeddings_queued', '_current_batch'):
                        if hasattr(matched_person, attr):
                            delattr(matched_person, attr)

                    # Log entry event for returning visitor
                    with self.entry_logging_lock:
                        already_logged = (
                            hasattr(matched_person, '_entry_logged') and
                            matched_person._entry_logged and
                            getattr(matched_person, '_entry_logged_session_id', None) == self.session_id
                        )
                        if not already_logged and not same_visit:
                            matched_person._entry_logged = True
                            matched_person._entry_logged_session_id = self.session_id
                            matched_person._entry_frame = self.frame_count

                            # Recalculate visit_number before logging
                            visit_result = self.db_manager.query(
                                "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                (matched_person_id, self.session_id)
                            )
                            matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1

                            self.event_logger.log_entry(matched_person_id, matched_person.visit_number, self.current_video_file, self.frame_count)
                            logger.info(f"✅ Entry logged: {matched_person_id} (visit {matched_person.visit_number})")

                    # Load face embeddings from Qdrant if not in memory
                    if not matched_person.face_embeddings:
                        try:
                            stored_face = self.embedding_store.get_embeddings_for_person(matched_person_id)
                            if stored_face:
                                matched_person.face_embeddings = stored_face
                                logger.info(f"📦 {matched_person_id} in-memory: {len(stored_face)} face embeddings loaded from Qdrant")
                            else:
                                logger.info(f"📦 {matched_person_id} in-memory: keeping {len(matched_person.face_embeddings)} face embeddings")
                        except Exception as e:
                            logger.warning(f"Failed to load embeddings from Qdrant for {matched_person_id}: {e}")
                    else:
                        logger.info(f"📦 {matched_person_id} already has {len(matched_person.face_embeddings)} face embeddings in memory — skipping Qdrant load")

                    # Start accumulator if returning visitor needs more embeddings
                    if ACCUMULATOR_ENABLED and len(matched_person.face_embeddings) < ACCUMULATOR_TARGET_EMBEDDINGS:
                        self._start_background_accumulator(matched_person_id, matched_person)

                    # Update person profile
                    self.db_manager.execute(
                        """INSERT OR IGNORE INTO person_profiles
                        (person_id, first_seen, last_seen, total_visits, total_time_seconds, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (matched_person_id, datetime.now().isoformat(), datetime.now().isoformat(), 1, 0, datetime.now().isoformat())
                    )
                    self.db_manager.execute(
                        """UPDATE person_profiles SET last_seen=?, total_visits=?, updated_at=? WHERE person_id=?""",
                        (datetime.now().isoformat(), matched_person.visit_number, datetime.now().isoformat(), matched_person_id)
                    )

                    # Mark as fully processed
                    person._embeddings_computed = True
                    person._accumulating = False
                    logger.info(f"✅ Person confirmed as returning visitor — stopping batch collection early")
                    return  # CRITICAL: Stop here, don't continue to batch 3

                else:
                    # ❌ NO MATCH — continue to batch 3
                    logger.info(f"❌ No match at Checkpoint 20")
                    logger.info(f"  Proceeding to batch 3 (final decision)...")

                    # ✅ CLEAR FLAG to allow batch 3 to queue (only if NO match found)
                    if hasattr(person, '_embeddings_queued'):
                        delattr(person, '_embeddings_queued')
                        logger.info(f"✅ Cleared _embeddings_queued flag for {person_id} - batch 3 CAN NOW QUEUE")
                    else:
                        logger.info(f"⚠️ Flag already cleared for {person_id}")

                    # 📊 ADVANCE TO BATCH 3 (FINAL)
                    person._current_batch = 3
                    logger.info(f"📊 Advancing to BATCH 3 (FINAL) - {person_id} will collect frames 21-30")

            # ===== BATCH 3: FINAL DECISION =====
            elif batch_num == 3 and person._search_history[30] is None:
                logger.info(f"🔴 BATCH 3 FINAL DECISION for {person_id}")
                logger.info(f"  Face embeddings collected (frames 20-29): {len(face_embeddings)}")
                logger.info(f"🔍 CHECKPOINT 30 SEARCH (FINAL): Searching with {FRAMES_TO_COLLECT} face embeddings (batch 3)")
                self._search_at_checkpoint(person, 30)

                # Check if match found at checkpoint 30
                if person._matched_person_id:
                    # RETURNING VISITOR FOUND - FINALIZE NOW
                    person._final_decision_made = True
                    logger.info(f"✅✅✅ FINAL DECISION: RETURNING VISITOR")
                    logger.info(f"  {person_id} (temp) matches: {person._matched_person_id}")

                    # Finalize person ID
                    final_person_id = self._finalize_person_id(person)

                    # ✨ NEW: Per-frame uniqueness check
                    with self.tracker.people_lock:
                        target_person = self.tracker.people.get(final_person_id)

                    if target_person and target_person.is_in_frame and target_person != person:
                        logger.warning(f"🚨 CONFLICT: {final_person_id} already in frame! Re-verifying...")

                        current_person_embeds = person.face_embeddings if hasattr(person, 'face_embeddings') else []
                        target_person_embeds = target_person.face_embeddings if hasattr(target_person, 'face_embeddings') else []

                        final_person_id_for_current, final_person_id_for_target = self.tracker.resolve_identity_conflict(
                            person.person_id, current_person_embeds,
                            final_person_id, target_person_embeds
                        )

                        if final_person_id_for_current == "NEW":
                            logger.info(f"   → {person.person_id} is FALSE positive, staying as new person")
                            final_person_id = person.person_id

                    # Finalize and process returning visitor
                else:
                    # NO MATCH AT CHECKPOINT 30 - CONTINUE TO BATCH 4 & 5
                    logger.info(f"❌ No match at Checkpoint 30 - continuing to batch 4 for more embeddings...")

                    # ✅ CLEAR FLAG to allow batch 4 to queue
                    if hasattr(person, '_embeddings_queued'):
                        delattr(person, '_embeddings_queued')
                        logger.info(f"✅ Cleared _embeddings_queued flag for {person_id} - batch 4 CAN NOW QUEUE")

                    # 📊 ADVANCE TO BATCH 4
                    person._current_batch = 4
                    logger.info(f"📊 Advancing to BATCH 4 - {person_id} will collect frames 31-40")

                    # Return here - batch 4 will handle further processing
                    return
                # continue immediately — don't wait
                # person confirmed, green box shows now

                # Handle temporary person merging if matched to returning visitor
                if hasattr(person, '_is_temporary') and person._is_temporary:
                    if person._matched_person_id:
                        # Matched to returning visitor - merge temporary to existing person
                        temp_person_id = person_id
                        matched_person_id = person._matched_person_id

                        # Complete cleanup of temp person — remove from tracker, cache, buffers
                        self._cleanup_temp_after_match(temp_person_id, matched_person_id)

                        # Get or create the matched person
                        matched_person = self.tracker.get_person(matched_person_id)
                        if matched_person is None:
                            matched_person = Person.create_for_existing(matched_person_id)
                            # Count entries within this session only + 1 = next visit number
                            visit_result = self.db_manager.query(
                                "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                (matched_person_id, self.session_id)
                            )
                            matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1
                            try:
                                with self.tracker.people_lock:
                                    self.tracker.people[matched_person_id] = matched_person
                            except RuntimeError:
                                pass
                        else:
                            pass  # visit_number updated below after same_visit is known

                        absence_seconds = (datetime.now() - matched_person.lost_time).total_seconds() if matched_person.lost_time else (NEW_VISIT_THRESHOLD_SECONDS + 1)
                        same_visit = absence_seconds < NEW_VISIT_THRESHOLD_SECONDS

                        # Update visit_number — only increment if this is a new visit and person was offline
                        if not matched_person.is_in_frame and not same_visit:
                            visit_result = self.db_manager.query(
                                "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                (matched_person_id, self.session_id)
                            )
                            matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1
                        if same_visit:
                            logger.info(f"✅ {matched_person_id} REACTIVATED — returned within {absence_seconds:.1f}s (same visit, no new entry logged)")
                        else:
                            logger.info(f"🔄 {matched_person_id} returned after {absence_seconds:.1f}s — new visit")
                        matched_person.mark_online()

                        # Transfer face embeddings from temp person if matched has none
                        if not matched_person.face_embeddings and person.face_embeddings:
                            matched_person.face_embeddings = list(person.face_embeddings)

                        # Load face embeddings from Qdrant if not in memory
                        if not matched_person.face_embeddings:
                            try:
                                stored_face = self.embedding_store.get_embeddings_for_person(matched_person_id)
                                if stored_face:
                                    matched_person.face_embeddings = stored_face
                                    logger.info(f"📦 {matched_person_id} in-memory: {len(stored_face)} face embeddings loaded from Qdrant")
                                else:
                                    logger.info(f"📦 {matched_person_id} in-memory: keeping {len(matched_person.face_embeddings)} face embeddings")
                            except Exception as e:
                                logger.warning(f"Failed to load embeddings from Qdrant for {matched_person_id}: {e}")
                        else:
                            logger.info(f"📦 {matched_person_id} already has {len(matched_person.face_embeddings)} face embeddings in memory — skipping Qdrant load")

                        # Start accumulator if returning visitor needs more embeddings
                        if ACCUMULATOR_ENABLED and len(matched_person.face_embeddings) < ACCUMULATOR_TARGET_EMBEDDINGS:
                            self._start_background_accumulator(matched_person_id, matched_person)

                        # Log entry event for returning visitor
                        with self.entry_logging_lock:
                            already_logged = (
                                hasattr(matched_person, '_entry_logged') and
                                matched_person._entry_logged and
                                getattr(matched_person, '_entry_logged_session_id', None) == self.session_id
                            )
                            if not already_logged and not same_visit:
                                matched_person._entry_logged = True
                                matched_person._entry_logged_session_id = self.session_id
                                matched_person._entry_frame = self.frame_count
                                logger.info(f"Returning visitor: {matched_person_id} (was temp: {temp_person_id}), logging entry (session {self.session_id})")
                                self.event_logger.log_entry(matched_person_id, matched_person.visit_number, self.current_video_file, self.frame_count)

                        # Update person profile
                        total_visits_result = self.db_manager.query(
                            "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED'",
                            (matched_person_id,)
                        )
                        total_visits = total_visits_result[0][0] if total_visits_result else 1

                        self.db_manager.execute(
                            """INSERT OR IGNORE INTO person_profiles
                            (person_id, first_seen, last_seen, total_visits, total_time_seconds, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (matched_person_id, matched_person.first_seen.isoformat(), datetime.now().isoformat(), 1, 0, datetime.now().isoformat())
                        )
                        self.db_manager.execute(
                            """UPDATE person_profiles
                            SET last_seen = ?, total_visits = ?, updated_at = ?
                            WHERE person_id = ?""",
                            (datetime.now().isoformat(), total_visits, datetime.now().isoformat(), matched_person_id)
                        )

                        # Remove temporary flags
                        if hasattr(person, '_is_temporary'):
                            delattr(person, '_is_temporary')
                        if hasattr(matched_person, '_is_temporary'):
                            delattr(matched_person, '_is_temporary')

                    else:
                        # New person - NOT FINALIZING YET, continue to batch 4
                        logger.info(f"❌ No match at Checkpoint 30 - continuing to batch 4 for more embeddings...")

                        # ✅ CLEAR FLAG to allow batch 4 to queue
                        if hasattr(person, '_embeddings_queued'):
                            delattr(person, '_embeddings_queued')
                            logger.info(f"✅ Cleared _embeddings_queued flag for {person_id} - batch 4 CAN NOW QUEUE")

                        # 📊 ADVANCE TO BATCH 4
                        person._current_batch = 4
                        logger.info(f"📊 Advancing to BATCH 4 - {person_id} will collect frames 31-40")
                        return  # Exit to collect more frames

            # ===== BATCH 4: Continue Search =====
            elif batch_num == 4 and person._search_history.get(40) is None:
                logger.info(f"🟠 BATCH 4 PROCESSING for {person_id}")
                logger.info(f"  Face embeddings collected (frames 30-39): {len(face_embeddings)}")
                logger.info(f"🔍 CHECKPOINT 40 SEARCH: Searching with {FRAMES_TO_COLLECT} face embeddings (batch 4)")
                checkpoint_decided = self._search_at_checkpoint(person, 40)

                if checkpoint_decided:
                    logger.info(f"✅✅ MATCH FOUND at Checkpoint 40!")
                    logger.info(f"  {person_id} matches existing person: {person._matched_person_id}")

                    # Handle match (same as batch 2)
                    matched_person_id = person._matched_person_id
                    temp_person_id = person_id
                    person._is_temporary = False
                    person._confirming = True
                    person._accumulating = True
                    person._final_decision_made = True

                    if hasattr(person, '_embeddings_queued'):
                        delattr(person, '_embeddings_queued')

                    final_person_id = self._finalize_person_id(person)
                    logger.info(f"⏭️  Returning visitor {final_person_id} — skipping storage (already in DB)")

                    self._cleanup_temp_after_match(temp_person_id, matched_person_id)
                    matched_person = self.tracker.get_person(matched_person_id)
                    if matched_person is None:
                        matched_person = Person.create_for_existing(matched_person_id)
                        try:
                            with self.tracker.people_lock:
                                self.tracker.people[matched_person_id] = matched_person
                        except RuntimeError:
                            pass

                    matched_person.mark_online()
                    matched_person._accumulating = False
                    matched_person._confirming = False
                    matched_person._embeddings_computed = True
                    matched_person._final_decision_made = True

                    for attr in ('_is_temporary', '_confirming', '_accumulating', '_embeddings_queued', '_current_batch'):
                        if hasattr(matched_person, attr):
                            delattr(matched_person, attr)

                    person._embeddings_computed = True
                    person._accumulating = False
                    logger.info(f"✅ Person confirmed as returning visitor — stopping batch collection early")
                    return
                else:
                    logger.info(f"❌ No match at Checkpoint 40 - continuing to batch 5 (final)...")

                    if hasattr(person, '_embeddings_queued'):
                        delattr(person, '_embeddings_queued')
                        logger.info(f"✅ Cleared _embeddings_queued flag for {person_id} - batch 5 CAN NOW QUEUE")

                    person._current_batch = 5
                    logger.info(f"📊 Advancing to BATCH 5 - {person_id} will collect frames 41-50")

            # ===== BATCHES 5-9: Continue Search =====
            elif 5 <= batch_num <= 9 and person._search_history.get(batch_num * 10) is None:
                checkpoint = batch_num * 10
                logger.info(f"🟠 BATCH {batch_num} PROCESSING for {person_id}")
                logger.info(f"🔍 CHECKPOINT {checkpoint} SEARCH: Searching with {FRAMES_TO_COLLECT} face embeddings (batch {batch_num})")
                checkpoint_decided = self._search_at_checkpoint(person, checkpoint)

                if checkpoint_decided:
                    logger.info(f"✅✅ MATCH FOUND at Checkpoint {checkpoint}!")
                    logger.info(f"  {person_id} matches existing person: {person._matched_person_id}")

                    matched_person_id = person._matched_person_id
                    temp_person_id = person_id
                    person._is_temporary = False
                    person._confirming = True
                    person._accumulating = True
                    person._final_decision_made = True

                    if hasattr(person, '_embeddings_queued'):
                        delattr(person, '_embeddings_queued')

                    final_person_id = self._finalize_person_id(person)
                    logger.info(f"⏭️  Returning visitor {final_person_id} — skipping storage (already in DB)")

                    self._cleanup_temp_after_match(temp_person_id, matched_person_id)
                    matched_person = self.tracker.get_person(matched_person_id)
                    if matched_person is None:
                        matched_person = Person.create_for_existing(matched_person_id)
                        try:
                            with self.tracker.people_lock:
                                self.tracker.people[matched_person_id] = matched_person
                        except RuntimeError:
                            pass

                    matched_person.mark_online()
                    matched_person._accumulating = False
                    matched_person._confirming = False
                    matched_person._embeddings_computed = True
                    matched_person._final_decision_made = True

                    for attr in ('_is_temporary', '_confirming', '_accumulating', '_embeddings_queued', '_current_batch'):
                        if hasattr(matched_person, attr):
                            delattr(matched_person, attr)

                    person._embeddings_computed = True
                    person._accumulating = False
                    logger.info(f"✅ Person confirmed as returning visitor — stopping batch collection early")
                    return
                else:
                    next_batch = batch_num + 1
                    logger.info(f"❌ No match at Checkpoint {checkpoint} - continuing to batch {next_batch}...")
                    if hasattr(person, '_embeddings_queued'):
                        delattr(person, '_embeddings_queued')
                    person._current_batch = next_batch
                    logger.info(f"📊 Advancing to BATCH {next_batch} - {person_id}")

            # ===== BATCH 10: FINAL DECISION WITH FILTERING =====
            elif batch_num == 10 and person._search_history.get(100) is None:
                logger.info(f"🔴 BATCH 10 FINAL DECISION for {person_id}")
                logger.info(f"  Face embeddings collected (frames 90-99): {len(face_embeddings)}")
                logger.info(f"🔍 CHECKPOINT 100 SEARCH (FINAL): Searching with {FRAMES_TO_COLLECT} face embeddings (batch 10)")
                self._search_at_checkpoint(person, 100)

                # FINAL DECISION
                person._final_decision_made = True

                if person._matched_person_id:
                    logger.info(f"✅✅✅ FINAL DECISION: RETURNING VISITOR")
                    logger.info(f"  {person_id} (temp) matches: {person._matched_person_id}")
                else:
                    logger.info(f"✅✅✅ FINAL DECISION: NEW PERSON")
                    logger.info(f"  {person_id} is a brand new person")
                    logger.info(f"  Collecting {len(person.face_embeddings)} embeddings for filtering and storage...")

                # Finalize person ID
                final_person_id = self._finalize_person_id(person)

                # 🚫 SKIP NEW PEOPLE ENTIRELY (Option 1)
                # If no match found (final_person_id == person.person_id), remove and ignore
                if final_person_id == person.person_id:
                    logger.info(f"🚫 SKIPPING NEW PERSON {final_person_id} - not in known database")
                    logger.info(f"   Reason: User requirement (Option 1) - only track people in database")
                    try:
                        with self.tracker.people_lock:
                            if final_person_id in self.tracker.people:
                                del self.tracker.people[final_person_id]
                                logger.info(f"🗑️ Removed {final_person_id} from tracker")
                    except RuntimeError:
                        pass
                    return  # Skip storage, logging, and all further processing for this person

                # Store all embeddings in background thread, then start accumulator
                storage_thread = threading.Thread(
                    target=self._store_and_start_accumulator,
                    args=(final_person_id, person),
                    daemon=True,
                    name=f"store-{final_person_id}"
                )
                storage_thread.start()
                logger.info(f"📤 Storage + accumulator started in background for {final_person_id}")

                # Handle temporary person merging if matched to returning visitor
                if hasattr(person, '_is_temporary') and person._is_temporary:
                    if person._matched_person_id:
                        # Matched to returning visitor
                        temp_person_id = person_id
                        matched_person_id = person._matched_person_id

                        self._cleanup_temp_after_match(temp_person_id, matched_person_id)

                        matched_person = self.tracker.get_person(matched_person_id)
                        if matched_person is None:
                            matched_person = Person.create_for_existing(matched_person_id)
                            visit_result = self.db_manager.query(
                                "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                (matched_person_id, self.session_id)
                            )
                            matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1
                            try:
                                with self.tracker.people_lock:
                                    self.tracker.people[matched_person_id] = matched_person
                            except RuntimeError:
                                pass

                        absence_seconds = (datetime.now() - matched_person.lost_time).total_seconds() if matched_person.lost_time else (NEW_VISIT_THRESHOLD_SECONDS + 1)
                        same_visit = absence_seconds < NEW_VISIT_THRESHOLD_SECONDS

                        if not matched_person.is_in_frame and not same_visit:
                            visit_result = self.db_manager.query(
                                "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                (matched_person_id, self.session_id)
                            )
                            matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1
                        if same_visit:
                            logger.info(f"✅ {matched_person_id} REACTIVATED — returned within {absence_seconds:.1f}s (same visit, no new entry logged)")
                        else:
                            logger.info(f"🔄 {matched_person_id} returned after {absence_seconds:.1f}s — new visit")
                        matched_person.mark_online()

                        if not matched_person.face_embeddings and person.face_embeddings:
                            matched_person.face_embeddings = list(person.face_embeddings)

                        if not matched_person.face_embeddings:
                            try:
                                stored_face = self.embedding_store.get_embeddings_for_person(matched_person_id)
                                if stored_face:
                                    matched_person.face_embeddings = stored_face
                                    logger.info(f"📦 {matched_person_id} in-memory: {len(stored_face)} face embeddings loaded from Qdrant")
                                else:
                                    logger.info(f"📦 {matched_person_id} in-memory: keeping {len(matched_person.face_embeddings)} face embeddings")
                            except Exception as e:
                                logger.warning(f"Failed to load embeddings from Qdrant for {matched_person_id}: {e}")

                        # Start accumulator if returning visitor needs more embeddings
                        if ACCUMULATOR_ENABLED and len(matched_person.face_embeddings) < ACCUMULATOR_TARGET_EMBEDDINGS:
                            self._start_background_accumulator(matched_person_id, matched_person)

                        with self.entry_logging_lock:
                            already_logged = (
                                hasattr(matched_person, '_entry_logged') and
                                matched_person._entry_logged and
                                getattr(matched_person, '_entry_logged_session_id', None) == self.session_id
                            )
                            if not already_logged and not same_visit:
                                matched_person._entry_logged = True
                                matched_person._entry_logged_session_id = self.session_id
                                matched_person._entry_frame = self.frame_count

                                visit_result = self.db_manager.query(
                                    "SELECT COUNT(*) FROM events WHERE person_id = ? AND event_type = 'ENTERED' AND session_id = ?",
                                    (matched_person_id, self.session_id)
                                )
                                matched_person.visit_number = (visit_result[0][0] if visit_result else 0) + 1

                                self.event_logger.log_entry(matched_person_id, matched_person.visit_number, self.current_video_file, self.frame_count)
                                logger.info(f"✅ Entry logged: {matched_person_id} (visit {matched_person.visit_number})")

                        self.db_manager.execute(
                            """UPDATE person_profiles SET last_seen=?, total_visits=?, updated_at=? WHERE person_id=?""",
                            (datetime.now().isoformat(), matched_person.visit_number, datetime.now().isoformat(), matched_person_id)
                        )

                        if hasattr(person, '_is_temporary'):
                            delattr(person, '_is_temporary')
                        if hasattr(matched_person, '_is_temporary'):
                            delattr(matched_person, '_is_temporary')

                    else:
                        # New person - confirm temporary person
                        logger.debug(f"New person {final_person_id}: confirming temporary")
                        try:
                            with self.tracker.people_lock:
                                self.tracker.people[final_person_id] = person
                        except RuntimeError:
                            pass

                        person.mark_online()

                        # Log entry event for new person
                        with self.entry_logging_lock:
                            has_entry_logged = hasattr(person, '_entry_logged') and person._entry_logged and person._entry_logged_session_id == self.session_id
                            if not has_entry_logged:
                                person._entry_logged = True
                                person._entry_logged_session_id = self.session_id
                                entry_frame = person._creation_frame if hasattr(person, '_creation_frame') and person._creation_frame else self.frame_count
                                person._entry_frame = entry_frame
                                logger.info(f"Confirmed person {final_person_id} at Frame 5, logging entry (created at Frame {entry_frame})")
                                self.event_logger.log_entry(final_person_id, person.visit_number, self.current_video_file, entry_frame)

                        # Update person profile
                        self.db_manager.execute(
                            """INSERT OR IGNORE INTO person_profiles
                            (person_id, first_seen, last_seen, total_visits, total_time_seconds, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (final_person_id, person.first_seen.isoformat(), datetime.now().isoformat(), 1, 0, datetime.now().isoformat())
                        )

                        # Remove temporary flag
                        if hasattr(person, '_is_temporary'):
                            delattr(person, '_is_temporary')

            # Only mark as fully computed if final decision made (all 25 embeddings done)
            # This allows batches 1-5 to all process sequentially
            if getattr(person, '_final_decision_made', False):
                person._embeddings_computed = True
                logger.info(f"✅ All 25 embeddings processed for {person_id} - marked as computed")
                # 🛡️ UNPROTECT from timeout recycling - done accumulating embeddings
                person._accumulating = False
                person._confirming = False
                logger.info(f"✅ Embedding collection complete for {person_id} - OK to recycle if times out")

            # Flag clearing is handled inside batch logic
            # Don't delete here — it would override careful batch-by-batch clearing

            # Clear buffer ONLY after BATCH 5 (final decision), not after every batch
            if batch_num == 5 and getattr(person, '_final_decision_made', False):
                logger.info(f"🧹 Clearing buffer for {person_id} (final decision made)")
                with self.buffers_lock:
                    buffer.clear()

            # Embeddings already computed and accumulated in checkpoint logic
            # Final decision handling is done in checkpoint logic above
            # No additional processing needed here

        except (IOError, ValueError) as e:
            # Handle I/O errors during shutdown (e.g., closed connections)
            logger.debug(f"I/O error during embedding computation for {person_id} (likely shutdown): {e}")
        except Exception as e:
            logger.warning(f"Error computing embeddings for {person_id}: {e}")

    def _cleanup_temp_after_match(self, temp_person_id, matched_person_id):
        """Completely remove temp person after match and transfer everything to matched person"""

        # 🛡️ Guard: prevent duplicate cleanup
        if getattr(self, '_cleaned_up_persons', None) is None:
            self._cleaned_up_persons = set()

        person_obj = self.tracker.get_person(temp_person_id)
        cleanup_key = person_obj._instance_id if person_obj is not None else temp_person_id

        if cleanup_key in self._cleaned_up_persons:
            logger.debug(f"Cleanup already done for {temp_person_id}, skipping duplicate")
            return

        # 1. Update bytetrack cache — all track_ids pointing to temp → matched
        for track_id in list(self.tracker.bytetrack_cache.keys()):
            if self.tracker.bytetrack_cache[track_id] == temp_person_id:
                self.tracker.bytetrack_cache[track_id] = matched_person_id
                logger.debug(f"Cache: track {track_id} → {matched_person_id}")

        # 2. Only delete if temp and matched are DIFFERENT persons
        if temp_person_id != matched_person_id:
            with self.tracker.people_lock:
                if temp_person_id in self.tracker.people:
                    del self.tracker.people[temp_person_id]
                    logger.info(f"Deleted {temp_person_id} from tracker")

            # 🧹 Clear frame buffer for temp person (batch 1 early exit doesn't clear it)
            with self.buffers_lock:
                self.person_frame_buffers.pop(temp_person_id, None)
                self.person_first_detection_frame.pop(temp_person_id, None)
                self.person_frame_counters.pop(temp_person_id, None)
                logger.debug(f"🧹 Cleared frame buffer for {temp_person_id}")

            # 🆔 CRITICAL: Release temp ID back to pool for reuse
            try:
                id_num = int(temp_person_id.split("_")[1])
                Person._release_id(id_num)
            except (ValueError, IndexError):
                pass

            # 3. Clear from rejected track ids if present
            temp_track_ids = [track_id for track_id, pid in list(self.tracker.bytetrack_cache.items())
                             if pid == temp_person_id]
            for track_id in temp_track_ids:
                self.tracker.bytetrack_cache.pop(track_id, None)

            self._cleaned_up_persons.add(cleanup_key)

            logger.info(f"✅ Full cleanup: {temp_person_id} removed, {matched_person_id} active")
        else:
            # Same person — just update cache, don't delete
            logger.info(f"✅ Same ID match ({temp_person_id}) — keeping existing person object")

    def _periodic_reidentification(self, frame, current_detections):
        """
        ✨ NEW: Periodically re-verify person identifications every 10 seconds.
        Compute fresh embeddings and check if person still matches their assigned ID.
        """
        reidentify_interval = 10 * VIDEO_FPS  # 10 seconds at 30fps = 300 frames

        with self.tracker.people_lock:
            people_to_check = [(pid, p) for pid, p in self.tracker.people.items()
                              if p.is_in_frame and pid in ["person 1", "person 2", "jilani"]]  # Only check known people

        for person_id, person in people_to_check:
            # Check if 10 seconds have passed since last verification
            time_since_reid = (datetime.now() - person.last_reidentification_time).total_seconds()

            if time_since_reid >= 10:
                # Get person's bbox from current detection
                for track_id, detected_pid in current_detections.items():
                    if detected_pid == person_id:
                        # Find bbox for this track_id
                        for det in (getattr(self, '_last_detections', []) or []):
                            if det.get('track_id') == track_id:
                                bbox = det.get('bbox')
                                if bbox:
                                    x1, y1, x2, y2 = [int(v) for v in bbox]
                                    h, w = frame.shape[:2]

                                    # Anti-overlap: use center 75% of bbox
                                    center_crop_ratio = 0.75
                                    bbox_w = x2 - x1
                                    bbox_h = y2 - y1
                                    margin_x = int(bbox_w * (1 - center_crop_ratio) / 2)
                                    margin_y = int(bbox_h * (1 - center_crop_ratio) / 2)
                                    x1_c = x1 + margin_x
                                    y1_c = y1 + margin_y
                                    x2_c = x2 - margin_x
                                    y2_c = y2 - margin_y

                                    person_crop = frame[max(0, y1_c):min(h, y2_c), max(0, x1_c):min(w, x2_c)]

                                    # Compute fresh embedding
                                    with self.embedding_lock:
                                        emb, conf, bbox_face, kps = self.embeddings.compute_face_embedding(person_crop)

                                    if emb is not None:
                                        # Search Qdrant for matches to this person's stored embeddings
                                        results = self.embedding_store.search_face_embedding(emb, top_k=5)

                                        if results.get('metadatas') and results['metadatas']:
                                            # Check if top match is still the same person
                                            top_meta = results['metadatas'][0][0] if isinstance(results['metadatas'][0], list) else results['metadatas'][0]
                                            top_pid = top_meta.get('person_id') if top_meta else None
                                            top_dist = results['distances'][0][0] if isinstance(results['distances'][0], list) else results['distances'][0]
                                            confidence = max(0, (1 - top_dist) * 100)

                                            person.reidentification_confidence = confidence

                                            if top_pid == person_id:
                                                person.reidentification_count += 1
                                                logger.info(f"✅ RE-ID CONFIRMED: {person_id} re-verified (confidence: {confidence:.1f}%) [#{person.reidentification_count}]")
                                            else:
                                                logger.warning(f"⚠️ RE-ID MISMATCH: {person_id} now matches {top_pid} (confidence: {confidence:.1f}%) — identity change?")

                                        person.last_reidentification_time = datetime.now()
                                break

    def _annotate_frame(self, frame, current_detections, detections):
        """Draw annotations on frame"""
        annotated = frame.copy()

        # Draw bounding boxes for each detection
        for detection in detections:
            track_id = detection["track_id"]
            bbox = detection["bbox"]
            conf = detection["confidence"]

            if track_id not in current_detections:
                continue

            person_id = current_detections[track_id]
            person = self.tracker.get_person(person_id)

            if person is None:
                continue

            # 🎯 OVERRIDE: If this temp person matched a real person, show the real person's ID
            matched_id = getattr(person, '_matched_person_id', None)

            # DEBUG: Log what's happening with jilani
            if person_id == "jilani" or matched_id == "jilani":
                logger.info(f"DEBUG ANNOTATION: person_id={person_id}, matched_id={matched_id}, is_temp={getattr(person, '_is_temporary', False)}")

            # ✨ NEW: Only show matched ID if that person is NOT already in-frame
            # (if in-frame, the match was excluded, so don't show it as matched)
            matched_person = self.tracker.get_person(matched_id) if matched_id else None
            if matched_person and matched_person.is_in_frame and matched_person != person:
                # Match was excluded (person already in-frame), don't show as matched
                display_id = person_id
            else:
                # Match is valid, show matched ID
                display_id = matched_id or person_id

            # Get the actual person object for display info (visit number, etc.)
            display_person = self.tracker.get_person(display_id) or person

            # ✨ NEW: ONLY show CONFIRMED people from database
            # Hide temporary people ONLY if they didn't match to anyone
            is_temp = hasattr(person, '_is_temporary') and person._is_temporary

            # If temporary person matched to a real person, show the match (not temp)
            # Only skip if temp person with NO valid match
            if is_temp and not matched_id:
                # Skip temporary people that didn't match anyone
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Check if displaying a temporary ID (not matched)
            is_display_temp = is_temp and not matched_id

            if is_display_temp:
                # Draw yellow box for temporary persons (waiting for Frame 3 confirmation)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
                text = f"{display_id} (TEMP) {conf:.2f}"
                cv2.putText(
                    annotated,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),  # Yellow text
                    2,
                )
            else:
                # Draw green box for confirmed persons
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

                # Draw person ID and visit number above box
                text = f"{display_id} (visit {display_person.visit_number}) {conf:.2f}"
                cv2.putText(
                    annotated,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        # Add frame counter
        cv2.putText(
            annotated,
            f"Frame: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return annotated

    def process_video_file(self, video_path):
        """Process video from file"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        logger.info(f"Processing video: {video_path}")

        # Set current video file for event logging
        self.current_video_file = video_path

        frame_idx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display with scaling to fit screen (embeddings already extracted from original frame)
            h, w = processed_frame.shape[:2]
            display_scale = 1280.0 / w if w > 0 else 1.0
            display_frame = cv2.resize(processed_frame, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow(f"TimeRevind - {video_path}", display_frame)

            # Periodic logging
            if frame_idx % 30 == 0:
                logger.debug(
                    f"Processed {frame_idx} frames | "
                    f"People in frame: {len(self.tracker.get_people_in_frame())}"
                )

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        # 🛡️ CRITICAL: Wait for embedding thread before cleanup
        # Prevents batches from being killed mid-process
        self.embedding_stop_event.set()
        logger.info("Waiting for embedding thread to finish (10 batches can take 30-60s)...")
        self.embedding_thread.join(timeout=30)
        if self.embedding_thread.is_alive():
            logger.warning("⚠️ Embedding thread still running after 30s timeout (may be processing batch 3)")
        else:
            logger.info("✅ Embedding thread finished cleanly")

        # Log exits for any people still in frame when video ends
        self._cleanup_remaining_people()
        logger.info(f"Video processing complete. Total frames: {self.frame_count}")

    def _detect_and_log_behaviors(self, frame, yolo_boxes, current_detections):
        """
        Run pose detection and log behaviors for all detected people.

        Args:
            frame: Current frame (BGR)
            yolo_boxes: YOLO detection boxes from tracking
            current_detections: Dict mapping track_id -> person_id
        """
        if not yolo_boxes or len(yolo_boxes) == 0 or not current_detections:
            return

        try:
            # Run pose inference on full frame
            pose_results = self.pose_model(frame, verbose=False)

            if not pose_results or pose_results[0].keypoints is None:
                # print(f"[BEHAVIOR] Frame {self.frame_count}: No keypoints detected")
                return

            keypoints_data = pose_results[0].keypoints.data
            pose_boxes = pose_results[0].boxes

            if len(keypoints_data) == 0:
                # print(f"[BEHAVIOR] Frame {self.frame_count}: Empty keypoints data")
                return

            # print(f"[BEHAVIOR] Frame {self.frame_count}: {len(yolo_boxes)} people, {len(keypoints_data)} poses")

            # Match each YOLO detection with pose keypoints
            for detection in yolo_boxes:
                track_id = int(detection.id[0]) if detection.id is not None else None
                if track_id is None or track_id not in current_detections:
                    continue

                person_id = current_detections[track_id]
                person = self.tracker.get_person(person_id)
                if person is None or not person.is_in_frame:
                    continue

                # Skip temporary people - only log for identified/confirmed people
                if hasattr(person, '_is_temporary') and person._is_temporary:
                    continue

                # Get detection bbox
                detection_bbox = detection.xyxy[0].cpu().numpy() if hasattr(detection.xyxy[0], 'cpu') else detection.xyxy[0]

                # Find matching keypoints from pose inference
                matching_keypoints = None
                for pose_idx, pose_box in enumerate(pose_boxes):
                    pose_bbox = pose_box.xyxy[0].cpu().numpy() if hasattr(pose_box.xyxy[0], 'cpu') else pose_box.xyxy[0]

                    # Check spatial overlap
                    if self._bboxes_overlap(detection_bbox, pose_bbox, threshold=0.1):
                        matching_keypoints = keypoints_data[pose_idx]
                        break

                if matching_keypoints is None:
                    # print(f"[BEHAVIOR] Frame {self.frame_count}: {person_id} - No keypoint match")
                    continue

                # Convert keypoints to numpy array
                kps = matching_keypoints.cpu().numpy() if hasattr(matching_keypoints, 'cpu') else matching_keypoints

                # Extract person crop with anti-overlap protection
                x1, y1, x2, y2 = int(detection_bbox[0]), int(detection_bbox[1]), int(detection_bbox[2]), int(detection_bbox[3])

                # 🛡️ Anti-overlap: Use center-focused crop for behavior detection too
                h, w = frame.shape[:2]
                center_crop_ratio = 0.75
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                margin_x = int(bbox_w * (1 - center_crop_ratio) / 2)
                margin_y = int(bbox_h * (1 - center_crop_ratio) / 2)
                x1 = x1 + margin_x
                y1 = y1 + margin_y
                x2 = x2 - margin_x
                y2 = y2 - margin_y

                person_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                # Detect behaviors
                behaviors = self.behavior_detector.detect_all(person_id, kps, person_crop, face_crop_bgr=None)

                # print(f"[BEHAVIOR] Frame {self.frame_count}: {person_id} - Detected {len(behaviors) if behaviors else 0} behaviors")

                # Log behaviors to database
                if behaviors:
                    for behavior in behaviors:
                        try:
                            visit_num = person.visit_number if hasattr(person, 'visit_number') else None
                            # print(f"[BEHAVIOR] Logging {behavior.get('type')} for {person_id} (confidence: {behavior.get('confidence', 0):.2f})")
                            self.behavior_db_logger.log_behavior(
                                person_id=person_id,
                                behavior=behavior,
                                visit_num=visit_num,
                                video_file=self.current_video_file,
                                video_timestamp=self.frame_count
                            )
                        except Exception as e:
                            # print(f"[BEHAVIOR] Error logging {behavior.get('type')} for {person_id}: {e}")
                            pass

        except Exception as e:
            # print(f"[BEHAVIOR] Error in behavior detection frame {self.frame_count}: {e}")
            import traceback
            # traceback.print_exc()
            pass

    def _bboxes_overlap(self, bbox1, bbox2, threshold=0.1):
        """
        Check if two bboxes overlap by at least threshold percentage.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            threshold: Minimum overlap ratio (0-1)

        Returns:
            True if overlap > threshold
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return False

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        overlap_ratio = inter_area / min(area1, area2)
        return overlap_ratio > threshold

    def _negotiate_camera_resolution(self, cap, camera_id):
        """
        Negotiate best resolution with camera.
        Tries: 1920x1080 → 1280x720 → 640x480

        Returns:
            (width, height) of successfully negotiated resolution
        """
        logger.info(f"Negotiating resolution for camera {camera_id}...")

        for width, height in RESOLUTION_FALLBACKS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

            # Read actual resolution camera supports
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Check if camera accepted this resolution (with 5% tolerance)
            if actual_w >= width * 0.95 and actual_h >= height * 0.95:
                logger.info(f"✓ Camera {camera_id} negotiated to {actual_w}x{actual_h}")
                return actual_w, actual_h
            else:
                logger.debug(f"  Camera rejected {width}x{height}, trying next...")

        # Fallback: just read what camera gives us
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == 0 or actual_h == 0:
            actual_w, actual_h = 640, 480

        logger.warning(f"Camera {camera_id} using fallback resolution: {actual_w}x{actual_h}")
        return actual_w, actual_h

    def process_camera_stream(self, camera_id=None):
        """Process video from camera with live face detection display

        Args:
            camera_id: Override the default camera ID for this session
        """
        if camera_id is not None:
            self.camera_id = camera_id

        logger.info(f"Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return

        # Negotiate best resolution (1080p → 720p → 480p)
        actual_w, actual_h = self._negotiate_camera_resolution(cap, self.camera_id)

        # Set additional camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to prevent lag

        logger.info(f"Processing camera stream {self.camera_id} at {actual_w}x{actual_h}")

        self.start_recording(resolution=(actual_w, actual_h))

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Keep full resolution (1080p/4K) for ArcFace face detection and embedding quality

                # Process frame
                processed_frame = self.process_frame(frame)

                # Note: Face detection for embeddings happens on person_crop, not full frame
                # Drawing face boxes on full frame would be inconsistent with actual embedding computation
                # Actual face detection/alignment occurs in compute_face_embedding() on cropped person regions

                # Display
                # Scale the preview window to a consistent width (1280px) regardless of camera resolution
                # This fixes "zoomed in" high-res feeds and "too small" laptop feeds!
                display_scale = 1280.0 / actual_w if actual_w > 0 else 1.0
                display_frame = cv2.resize(processed_frame, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow(f"TimeRevind - Camera {self.camera_id}", display_frame)

                # Periodic logging
                if self.frame_count % 300 == 0:
                    stats = self.tracker.get_statistics()
                    logger.info(f"Stream stats: {stats}")

                # Press q to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            logger.info("🔴 Starting cleanup sequence...")

            cap.release()
            logger.info("🔴 Released camera")

            cv2.destroyAllWindows()
            logger.info("🔴 Closed windows")

            self.stop_recording()
            logger.info("🔴 Stopped recording")

            # Stop embedding thread gracefully before cleanup
            logger.info("🔴 Signaling embedding thread to stop...")
            self.embedding_stop_event.set()

            logger.info("🔴 Waiting for embedding thread to finish (10 batches can take 30-60s)...")
            # 🛡️ CRITICAL: 30s timeout to allow all batch completions
            # Batches involve face embeddings + checkpoint searches + DB storage for each batch
            # Too short timeout kills thread mid-process, leaving persons unconfirmed
            self.embedding_thread.join(timeout=30)
            if self.embedding_thread.is_alive():
                logger.warning("⚠️ Embedding thread still running after 30s timeout (may be processing batch 3)")
            else:
                logger.info("🔴 Embedding thread finished cleanly")

            # Drain any remaining items in embedding queue (process them now to avoid race condition)
            logger.info("🔴 Draining embedding queue...")
            while not self.embedding_queue.empty():
                try:
                    person_id, person, is_preliminary = self.embedding_queue.get_nowait()
                    if person is not None:
                        try:
                            self._compute_and_store_embeddings(person_id, person, is_preliminary)
                        except Exception as e:
                            logger.debug(f"Error processing remaining embedding for {person_id}: {e}")
                except Exception as e:
                    logger.debug(f"Error draining embedding queue: {e}")
                    break

            # Log exits for any people still in frame when camera closes
            logger.info("🔴 Starting cleanup of remaining people...")
            self._cleanup_remaining_people()
            logger.info("🔴 Cleanup complete")

            logger.info(f"Camera stream processing ended")

    def _cleanup_remaining_people(self):
        """Log exit events for any people still in frame when session ends"""
        # Stop all active accumulators and save partial results
        if ACCUMULATOR_ENABLED:
            with self.accumulator_lock:
                instance_ids = list(self.active_accumulators.keys())
            for instance_id in instance_ids:
                self._stop_accumulator(instance_id, save=True)

        try:
            with self.tracker.people_lock:
                people_to_cleanup = list(self.tracker.people.items())
        except (RuntimeError, KeyError):
            # Handle race condition if dict is modified during iteration
            people_to_cleanup = []

        logger.info(f"🔴 Cleanup: {len(people_to_cleanup)} people in tracker")
        for pid, p in people_to_cleanup:
            logger.info(f"  {pid}: temp={getattr(p, '_is_temporary', False)} "
                       f"entry={getattr(p, '_entry_logged', False)} "
                       f"in_frame={p.is_in_frame} "
                       f"state={p.tracking_state} "
                       f"matched={getattr(p, '_matched_person_id', None)}")

        for person_id, person in people_to_cleanup:
            try:
                # Skip temporary persons (not confirmed yet)
                if hasattr(person, '_is_temporary') and person._is_temporary:
                    continue

                # Skip temp persons that matched a returning visitor
                # Their matched person will be cleaned up separately
                if hasattr(person, '_matched_person_id') and person._matched_person_id:
                    continue

                # 🛡️ Log exit for ACTIVE (in frame) or LOST persons (briefly offline)
                # EXITED persons (>60s timeout) already handled by timeout, and deleted from tracker
                if (person.is_in_frame or person.tracking_state == "LOST") and hasattr(person, '_entry_logged') and person._entry_logged:
                    # Calculate duration using frame numbers
                    duration_seconds = 0
                    if hasattr(person, '_entry_frame'):
                        frame_duration = self.frame_count - person._entry_frame
                        duration_seconds = int(frame_duration / VIDEO_FPS)

                    # Log exit event
                    self.event_logger.log_exit(person_id, person.visit_number, self.current_video_file, self.frame_count)
                    logger.info(f"Cleanup: Logged exit for {person_id} (duration: {duration_seconds}s)")

                    # Update person profile with duration
                    self.db_manager.execute(
                        """UPDATE person_profiles
                        SET last_seen = ?, total_time_seconds = total_time_seconds + ?, updated_at = ?
                        WHERE person_id = ?""",
                        (datetime.now().isoformat(), duration_seconds, datetime.now().isoformat(), person_id)
                    )
            except (KeyError, RuntimeError) as e:
                # Handle if person was deleted from tracker during cleanup
                logger.debug(f"Skipping cleanup for {person_id}: {e}")
                continue

    # ═══════════════════════════════════════════════════════
    # BACKGROUND ACCUMULATOR METHODS
    # ═══════════════════════════════════════════════════════

    def _store_and_start_accumulator(self, final_person_id, person):
        """Store initial embeddings, then start background accumulator if under target."""
        self._store_all_embeddings(final_person_id, person)
        stored_count = len(person.face_embeddings)
        if ACCUMULATOR_ENABLED and stored_count < ACCUMULATOR_TARGET_EMBEDDINGS:
            self._start_background_accumulator(final_person_id, person, existing_count=stored_count)

    def _start_background_accumulator(self, person_id, person, existing_count=None):
        """Start collecting more embeddings in background for a person."""
        if existing_count is None:
            existing_count = len(person.face_embeddings)

        # Always verify against Qdrant — in-memory count can be stale on return visits
        # This prevents embedding_index collisions that would silently overwrite good embeddings
        try:
            qdrant_results, _ = self.embedding_store.client.scroll(
                collection_name="face_embeddings",
                scroll_filter={"must": [{"key": "person_id", "match": {"value": person_id}}]},
                limit=1,
                with_vectors=False,
                with_payload=False
            )
            # Use count_points for an accurate total
            qdrant_count = self.embedding_store.client.count(
                collection_name="face_embeddings",
                count_filter={"must": [{"key": "person_id", "match": {"value": person_id}}]}
            ).count
            if qdrant_count > existing_count:
                logger.info(f"🔄 ACCUMULATOR {person_id}: Qdrant has {qdrant_count} embeddings, updating existing_count from {existing_count}")
                existing_count = qdrant_count
        except Exception as e:
            logger.debug(f"Could not verify Qdrant count for {person_id}: {e}")

        instance_id = person._instance_id
        with self.accumulator_lock:
            if instance_id in self.active_accumulators:
                logger.debug(f"🔄 ACCUMULATOR {person_id}: Already running — skipping duplicate start")
                return  # Already running for this person instance

            acc = AccumulatorState(
                person_id=person_id,
                instance_id=instance_id,
                target_count=ACCUMULATOR_TARGET_EMBEDDINGS,
                existing_count=existing_count,
            )
            self.active_accumulators[instance_id] = acc

        # Create per-person file logger (logs/accumulator/Person_1.log)
        acc_log_dir = LOGS_DIR / "accumulator"
        acc_log_dir.mkdir(exist_ok=True)
        acc_log_path = acc_log_dir / f"{person_id}.log"

        file_logger = logging.getLogger(f"acc.{instance_id}")
        file_logger.setLevel(logging.INFO)
        file_logger.propagate = False  # Don't send to terminal
        if not file_logger.handlers:
            fh = logging.FileHandler(acc_log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
            file_logger.addHandler(fh)
        acc.acc_logger = file_logger

        needed = ACCUMULATOR_TARGET_EMBEDDINGS - existing_count
        msg = f"🔄 ACCUMULATOR STARTED for {person_id} — has {existing_count}/{ACCUMULATOR_TARGET_EMBEDDINGS} embeddings, needs {needed} more"
        logger.info(msg)
        acc.acc_logger.info("=" * 70)
        acc.acc_logger.info(msg)
        acc.acc_logger.info("=" * 70)

        # Pre-load existing Qdrant embeddings in background so quality filters
        # can compare new frames against the full historical pool per pose
        threading.Thread(
            target=self._preload_qdrant_embeddings,
            args=(acc,),
            daemon=True,
            name=f"acc-preload-{person_id}"
        ).start()

    def _preload_qdrant_embeddings(self, acc):
        """Load existing embeddings from Qdrant into accumulator pool so quality
        filters (diversity / L2 / cohesion) compare against the full historical set."""
        try:
            results = self.embedding_store.get_embeddings_for_person(acc.person_id)
            # get_embeddings_for_person returns plain numpy arrays — we also need poses
            # Use scroll directly to get both vectors and payload
            points, _ = self.embedding_store.client.scroll(
                collection_name="face_embeddings",
                scroll_filter={"must": [{"key": "person_id", "match": {"value": acc.person_id}}]},
                limit=200,
                with_vectors=True,
                with_payload=True
            )
            import numpy as np
            qdrant_embs  = []
            qdrant_poses = []
            for p in points:
                if p.vector is not None:
                    qdrant_embs.append(np.array(p.vector, dtype=np.float32))
                    qdrant_poses.append((p.payload or {}).get("pose", "FRONT"))

            with acc.lock:
                acc.qdrant_embeddings = qdrant_embs
                acc.qdrant_poses      = qdrant_poses
                acc.qdrant_loaded     = True

            pose_summary = {}
            for p in qdrant_poses:
                pose_summary[p] = pose_summary.get(p, 0) + 1
            logger.info(f"🔄 ACCUMULATOR {acc.person_id}: 📥 Pre-loaded {len(qdrant_embs)} Qdrant embeddings | {pose_summary}")
            if acc.acc_logger:
                acc.acc_logger.info(f"Pre-loaded {len(qdrant_embs)} Qdrant embeddings: {pose_summary}")
        except Exception as e:
            logger.warning(f"🔄 ACCUMULATOR {acc.person_id}: Pre-load failed — {e}. Filters will use in-memory pool only.")
            with acc.lock:
                acc.qdrant_loaded = True  # Mark done even on failure so accumulator isn't blocked

    def _feed_accumulator(self, instance_id, person_id, person, person_crop):
        """Feed a new frame to the accumulator. Applies quality checks, adds if good."""
        with self.accumulator_lock:
            acc = self.active_accumulators.get(instance_id)
        if acc is None:
            return

        # Stride: process every Nth frame (reuse FRAME_CAPTURE_STRIDE)
        with acc.lock:
            acc.frame_counter += 1
            if acc.frame_counter % FRAME_CAPTURE_STRIDE != 0:
                return
            if not acc.is_running or acc.reached_target or acc.saved:
                return

        alog = acc.acc_logger  # shorthand — writes to logs/accumulator/Person_X.log

        # Check give-up condition
        if acc.should_give_up and not acc.reached_target:
            elapsed = time.time() - acc.start_time
            total_checks = acc.failed_attempts + len(acc.new_embeddings)
            reason = f">{acc.max_duration}s elapsed" if elapsed > acc.max_duration else f">{acc.max_failed} failed quality checks"
            with acc.lock:
                acc.is_running = False
                acc.gave_up_at = time.time()
            msg = (
                f"PAUSED ({reason}) — collected {acc.current_total}/{acc.target_count} "
                f"in {elapsed:.0f}s ({total_checks} frames checked)\n"
                f"  Rejections: {acc.rejection_summary()}\n"
                f"  Will retry #{acc.retry_count + 1}/{acc.max_retries} in {acc.COOLDOWN_SECONDS}s"
            )
            logger.info(f"🔄 ACCUMULATOR {person_id}: {msg}")
            if alog:
                alog.info("-" * 50)
                alog.info(f"PAUSED — {msg}")
                alog.info("-" * 50)
            return

        frame_tag = f"[{acc.current_total}/{acc.target_count}]"

        # Compute embedding
        with self.embedding_lock:
            try:
                emb, conf, bbox, kps = self.embeddings.compute_face_embedding(person_crop)
            except Exception as e:
                if alog: alog.info(f"{frame_tag} ERROR: embedding compute failed — {e}")
                return

        # Step 0: Face detection
        if emb is None:
            with acc.lock:
                acc.failed_attempts += 1
                acc.rej_no_face += 1
                acc.checks_since_last_summary += 1
                should_summarize = acc.checks_since_last_summary >= 20
                if should_summarize:
                    acc.checks_since_last_summary = 0
            if alog: alog.info(f"{frame_tag} Step0 Face:        ❌ NO FACE DETECTED")
            if should_summarize:
                msg = f"Stats ({acc.current_total}/{acc.target_count}) — {acc.rejection_summary()}"
                logger.info(f"🔄 ACCUMULATOR {person_id}: {msg}")
                if alog: alog.info(f"  >> {msg}")
            return

        if alog: alog.info(f"{frame_tag} Step0 Face:        ✅ detected (conf={conf:.2f})")
        emb  = np.array(emb).flatten()
        pose = self._estimate_pose(kps)   # FRONT / LEFT / RIGHT / UP / DOWN
        if alog: alog.info(f"{frame_tag} Pose:              {pose}")

        # Step 1: Sharpness (Tenengrad >= 15)
        if bbox is not None:
            try:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                face_crop = person_crop[y1:y2, x1:x2]
                if face_crop.size > 0:
                    sharpened = self._apply_unsharp_mask(face_crop)
                    sharpness = self._compute_tenengrad_sharpness(sharpened)
                    if sharpness < 15:
                        with acc.lock:
                            acc.failed_attempts += 1
                            acc.rej_blurry += 1
                            acc.checks_since_last_summary += 1
                            should_summarize = acc.checks_since_last_summary >= 20
                            if should_summarize:
                                acc.checks_since_last_summary = 0
                        if alog: alog.info(f"{frame_tag} Step1 Sharpness:   ❌ {sharpness:.1f} < 15 (too blurry)")
                        if should_summarize:
                            msg = f"Stats ({acc.current_total}/{acc.target_count}) — {acc.rejection_summary()}"
                            logger.info(f"🔄 ACCUMULATOR {person_id}: {msg}")
                            if alog: alog.info(f"  >> {msg}")
                        return
                    else:
                        if alog: alog.info(f"{frame_tag} Step1 Sharpness:   ✅ {sharpness:.1f} >= 15")
            except Exception as e:
                if alog: alog.info(f"{frame_tag} Step1 Sharpness:   ⚠️ skipped (crop error: {e})")

        # Build per-pose pools (compare only against same-pose embeddings to allow cross-pose diversity)
        with acc.lock:
            acc_embs      = list(acc.new_embeddings)
            acc_poses     = list(acc.new_embedding_poses)
            qdrant_embs   = list(acc.qdrant_embeddings)
            qdrant_poses  = list(acc.qdrant_poses)
        person_embs  = []
        person_poses = []
        if hasattr(person, 'face_embeddings') and person.face_embeddings:
            person_embs  = [np.array(e).flatten() for e in person.face_embeddings]
            person_poses = list(getattr(person, 'face_embedding_poses', ["FRONT"] * len(person_embs)))
            if len(person_poses) != len(person_embs):
                person_poses = ["FRONT"] * len(person_embs)
        # Full pool: if Qdrant is loaded it already contains everything in person.face_embeddings
        # (batch pipeline + flushed accumulator), so skip person_embs to avoid double-counting
        if qdrant_embs:
            all_embs  = qdrant_embs + acc_embs
            all_poses = qdrant_poses + acc_poses
        else:
            # Qdrant not loaded yet — fall back to in-memory pool
            all_embs  = person_embs + acc_embs
            all_poses = person_poses + acc_poses

        # Same-pose subset — cross-pose frames are intentionally dissimilar (not wrong-person)
        same_pose_embs = [e for e, p in zip(all_embs, all_poses) if p == pose]
        if alog:
            source = f"{len(qdrant_embs)} qdrant" if qdrant_embs else f"{len(person_embs)} in-memory"
            alog.info(f"{frame_tag} Pool:              {source} + {len(acc_embs)} accumulated = {len(all_embs)} total "
                      f"| same-pose({pose})={len(same_pose_embs)}")

        # Per-pose cap: max 7 embeddings per pose (FRONT / LEFT / RIGHT)
        if len(same_pose_embs) >= 7:
            if alog: alog.info(f"{frame_tag} Pose cap:          {pose} slot full ({len(same_pose_embs)}/7) — skipping")
            return

        # Step 2: Diversity — range check against SAME-POSE pool only
        max_sim = max((self.cosine_similarity(emb, e) for e in same_pose_embs), default=0.0)
        if max_sim >= 0.85:
            with acc.lock:
                acc.failed_attempts += 1
                acc.rej_similar += 1
                acc.checks_since_last_summary += 1
                should_summarize = acc.checks_since_last_summary >= 20
                if should_summarize:
                    acc.checks_since_last_summary = 0
            if alog: alog.info(f"{frame_tag} Step2 Diversity:   ❌ max_sim={max_sim:.3f} >= 0.85 (duplicate {pose})")
            if should_summarize:
                msg = f"Stats ({acc.current_total}/{acc.target_count}) — {acc.rejection_summary()}"
                logger.info(f"🔄 ACCUMULATOR {person_id}: {msg}")
                if alog: alog.info(f"  >> {msg}")
            return
        if alog: alog.info(f"{frame_tag} Step2 Diversity:   ✅ max_sim={max_sim:.3f} < 0.85 — not a duplicate (pose={pose})")

        # Step 3: L2 outlier check — DISABLED
        # Dynamic threshold (mean + 1.5*std) is too strict against pre-filtered Qdrant pool
        # (F5 cohesion filter makes pool very tight → tiny std → threshold rejects everything)
        # Step 4 (min cohesion ≥ 0.50) covers outlier rejection. Re-enable with std floor if needed.
        # if len(same_pose_embs) >= 3:
        #     mean_emb   = np.mean(same_pose_embs, axis=0)
        #     pool_dists = np.array([float(np.linalg.norm(e - mean_emb)) for e in same_pose_embs])
        #     dynamic_threshold = pool_dists.mean() + 1.5 * pool_dists.std()
        #     l2_dist = float(np.linalg.norm(emb - mean_emb))
        #     if l2_dist >= dynamic_threshold:
        #         with acc.lock:
        #             acc.failed_attempts += 1
        #             acc.rej_l2 += 1
        #             acc.checks_since_last_summary += 1
        #             should_summarize = acc.checks_since_last_summary >= 20
        #             if should_summarize:
        #                 acc.checks_since_last_summary = 0
        #         if alog: alog.info(f"{frame_tag} Step3 L2:          ❌ dist={l2_dist:.3f} >= {dynamic_threshold:.3f} (outlier | threshold=mean({pool_dists.mean():.3f})+1.5*std({pool_dists.std():.3f}))")
        #         if should_summarize:
        #             msg = f"Stats ({acc.current_total}/{acc.target_count}) — {acc.rejection_summary()"
        #             logger.info(f"🔄 ACCUMULATOR {person_id}: {msg}")
        #             if alog: alog.info(f"  >> {msg}")
        #         return
        #     if alog: alog.info(f"{frame_tag} Step3 L2:          ✅ dist={l2_dist:.3f} < {dynamic_threshold:.3f} (threshold=mean({pool_dists.mean():.3f})+1.5*std({pool_dists.std():.3f}))")
        # else:
        #     if alog: alog.info(f"{frame_tag} Step3 L2:          ⚠️ skipped (same-pose pool={len(same_pose_embs)} < 3)")

        # Step 4: Min cohesion against same-pose pool
        if same_pose_embs:
            min_sim = min((self.cosine_similarity(emb, e) for e in same_pose_embs), default=1.0)
            if min_sim < 0.50:
                with acc.lock:
                    acc.failed_attempts += 1
                    acc.rej_cohesion += 1
                    acc.checks_since_last_summary += 1
                    should_summarize = acc.checks_since_last_summary >= 20
                    if should_summarize:
                        acc.checks_since_last_summary = 0
                if alog: alog.info(f"{frame_tag} Step4 MinCohesion: ❌ min_sim={min_sim:.3f} < 0.50 (outlier in {pose} pool)")
                if should_summarize:
                    msg = f"Stats ({acc.current_total}/{acc.target_count}) — {acc.rejection_summary()}"
                    logger.info(f"🔄 ACCUMULATOR {person_id}: {msg}")
                    if alog: alog.info(f"  >> {msg}")
                return
            if alog: alog.info(f"{frame_tag} Step4 MinCohesion: ✅ min_sim={min_sim:.3f} >= 0.50")
        else:
            if alog: alog.info(f"{frame_tag} Step4 MinCohesion: ⚠️ skipped (no same-pose pool yet — first {pose} frame)")

        # Passed all checks — add to accumulator
        with acc.lock:
            if acc.reached_target or acc.saved:
                return
            acc.new_embeddings.append(emb)
            acc.new_embedding_poses.append(pose)
            current = acc.current_total
            failed  = acc.failed_attempts

            # Pose slot summary — use qdrant_poses (historical) + acc.new_embedding_poses (this session)
            # Do NOT include person_poses: those are already reflected in qdrant_poses after flush
            all_stored_poses = qdrant_poses + list(acc.new_embedding_poses)
            slot_summary = {p: all_stored_poses.count(p) for p in ("FRONT", "LEFT", "RIGHT", "UP", "DOWN")}
            msg = (f"✅✅✅ ACCEPTED ({pose}) — now {current}/{acc.target_count} | "
                   f"slots: F={slot_summary['FRONT']}/7 L={slot_summary['LEFT']}/7 R={slot_summary['RIGHT']}/7 | "
                   f"rejected: {failed}")
            logger.info(f"🔄 ACCUMULATOR {person_id}: {msg}")
            if alog:
                alog.info(f"{frame_tag} {msg}")
                alog.info("")

            # Incremental flush: every 5 new accepted embeddings, save immediately
            unflushed = len(acc.new_embeddings) - acc.flushed_count
            should_flush = unflushed >= 5

            # Stop when total target hit OR all pose slots full
            all_slots_full = all(slot_summary[p] >= 7 for p in ("FRONT", "LEFT", "RIGHT", "UP", "DOWN"))
            should_save = (acc.reached_target or all_slots_full) and not acc.saved

            if should_flush and not should_save:
                # Flush the next 5 accepted embeddings to Qdrant without stopping accumulator
                threading.Thread(
                    target=self._accumulator_flush,
                    args=(instance_id,),
                    daemon=True,
                    name=f"acc-flush-{person_id}"
                ).start()

            if should_save:
                acc.saved = True
                acc.is_running = False
                reason = "ALL POSE SLOTS FULL (7×5)" if all_slots_full else f"TARGET {acc.target_count} REACHED"
                logger.info(f"🔄 ACCUMULATOR {person_id}: 🎯 {reason} — saving to Qdrant...")
                if alog:
                    alog.info("=" * 50)
                    alog.info(f"🎯 {reason} — saving to Qdrant...")
                    alog.info("=" * 50)
                threading.Thread(
                    target=self._accumulator_save,
                    args=(instance_id,),
                    daemon=True,
                    name=f"acc-save-{person_id}"
                ).start()

    def _accumulator_flush(self, instance_id):
        """Incrementally flush the next batch of 5 accepted embeddings to Qdrant."""
        with self.accumulator_lock:
            acc = self.active_accumulators.get(instance_id)
        if acc is None:
            return

        with acc.lock:
            flush_start = acc.flushed_count
            flush_end   = flush_start + 5
            batch_embs  = list(acc.new_embeddings[flush_start:flush_end])
            batch_poses = list(acc.new_embedding_poses[flush_start:flush_end])
            if not batch_embs:
                return
            acc.flushed_count = flush_end  # Mark as in-flight immediately to avoid double flush

        stored = 0
        for idx, (emb, pose_label) in enumerate(zip(batch_embs, batch_poses)):
            try:
                emb_hash = hashlib.md5(emb.tobytes()).hexdigest()[:8]
                emb_index = acc.existing_count + flush_start + idx + 1
                self.embedding_store.add_face_embedding(
                    acc.person_id,
                    emb,
                    {"embedding_index": emb_index, "pose": pose_label, "emb_hash": emb_hash}
                )
                stored += 1
            except Exception as e:
                logger.debug(f"Flush failed for {acc.person_id} emb {flush_start+idx+1}: {e}")

        if stored:
            logger.info(f"🔄 ACCUMULATOR {acc.person_id}: 💾 Flushed {stored} embeddings to Qdrant (indices {acc.existing_count+flush_start+1}–{acc.existing_count+flush_start+stored})")

            # Also append to person's in-memory embeddings so same-session matching sees them
            with self.tracker.people_lock:
                person = self.tracker.people.get(acc.person_id)
            if person is not None:
                with self.embedding_lock:
                    if not hasattr(person, 'face_embedding_poses'):
                        person.face_embedding_poses = []
                    POSE_CAP = 7
                    added = 0
                    for emb, pose_label in zip(batch_embs[:stored], batch_poses[:stored]):
                        if person.face_embedding_poses.count(pose_label) < POSE_CAP:
                            person.face_embeddings.append(emb)
                            person.face_embedding_poses.append(pose_label)
                            added += 1
                    if added:
                        pose_counts = {p: person.face_embedding_poses.count(p) for p in set(person.face_embedding_poses)}
                        logger.info(f"🔄 ACCUMULATOR {acc.person_id}: 📦 +{added} added to in-memory | poses: {pose_counts}")

            try:
                self.db_manager.execute(
                    """INSERT INTO embedding_metadata
                        (person_id, embedding_type, first_seen, frame_count, quality)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(person_id, embedding_type) DO UPDATE SET
                        frame_count = frame_count + excluded.frame_count,
                        updated_at  = CURRENT_TIMESTAMP""",
                    (acc.person_id, "face", datetime.now().isoformat(), stored, "high")
                )
            except Exception as e:
                logger.debug(f"Failed to update embedding metadata after flush: {e}")

    def _accumulator_save(self, instance_id):
        """Save remaining unflushed embeddings to Qdrant (called when accumulator stops)."""
        with self.accumulator_lock:
            acc = self.active_accumulators.get(instance_id)
        if acc is None or not acc.new_embeddings:
            return

        alog = acc.acc_logger
        # Only save what hasn't been flushed yet
        flush_start = acc.flushed_count
        filtered = list(acc.new_embeddings[flush_start:])
        poses_remaining = list(acc.new_embedding_poses[flush_start:])
        start_idx = acc.existing_count + flush_start
        stored = 0
        failed_indices = []

        if alog:
            alog.info("=" * 50)
            alog.info(f"SAVE STARTED — storing {len(filtered)} embeddings to Qdrant (existing: {start_idx})")

        # ✨ DISABLED: Save accumulator embeddings (only if SAVE_NEW_EMBEDDINGS enabled)
        if not SAVE_NEW_EMBEDDINGS:
            if alog:
                alog.info(f"⏭️  SKIPPED STORAGE — {len(filtered)} embeddings NOT saved (SAVE_NEW_EMBEDDINGS=False)")
                alog.info(f"   Using only existing embeddings for matching: jilani, person 1, person 2")
                alog.info("=" * 50)
            logger.info(f"⏭️  ACCUMULATOR SAVE SKIPPED for {acc.person_id} — SAVE_NEW_EMBEDDINGS=False")
        else:
            for idx, emb in enumerate(filtered):
                try:
                    emb_hash = hashlib.md5(emb.tobytes()).hexdigest()[:8]
                    pose_label = poses_remaining[idx] if idx < len(poses_remaining) else "FRONT"
                    self.embedding_store.add_face_embedding(
                        acc.person_id,
                        emb,
                        {"embedding_index": start_idx + idx + 1, "total_embeddings": start_idx + len(filtered), "emb_hash": emb_hash, "pose": pose_label}
                    )
                    stored += 1
                    if alog: alog.info(f"  [{idx+1}/{len(filtered)}] ✅ stored (hash={emb_hash})")
                except Exception as e:
                    failed_indices.append(idx)
                    if alog: alog.info(f"  [{idx+1}/{len(filtered)}] ❌ failed — {e}")

        if failed_indices:
            msg = f"{len(failed_indices)}/{len(filtered)} embeddings failed to save — saved {stored}, will retry next visit"
            logger.warning(f"⚠️ ACCUMULATOR {acc.person_id}: {msg}")
            if alog:
                alog.info(f"⚠️ PARTIAL SAVE — {msg}")
                alog.info("=" * 50)
            with acc.lock:
                acc.saved = False  # Allow retry
        else:
            total = acc.existing_count + flush_start + stored
            reached = acc.reached_target
            status = "TARGET REACHED" if reached else "partial save"
            logger.info(f"✅ ACCUMULATOR SAVED for {acc.person_id} ({status}): +{stored} final → total {total} in Qdrant")
            if alog:
                alog.info(f"✅ SAVE COMPLETE ({status}) — +{stored} new embeddings → total {total} in Qdrant")
                alog.info("=" * 50)

            # Update embedding_metadata with actual stored count
            try:
                self.db_manager.execute(
                    """INSERT INTO embedding_metadata
                        (person_id, embedding_type, first_seen, frame_count, quality)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(person_id, embedding_type) DO UPDATE SET
                        frame_count = frame_count + excluded.frame_count,
                        updated_at  = CURRENT_TIMESTAMP""",
                    (acc.person_id, "face", datetime.now().isoformat(), stored, "high")
                )
            except Exception as e:
                logger.debug(f"Failed to update embedding metadata for {acc.person_id}: {e}")

            with self.accumulator_lock:
                self.active_accumulators.pop(instance_id, None)

    def _stop_accumulator(self, instance_id, save=True):
        """Stop accumulator (person left frame). Save partial results if requested."""
        with self.accumulator_lock:
            acc = self.active_accumulators.get(instance_id)
        if acc is None:
            return

        alog = acc.acc_logger
        with acc.lock:
            acc.is_running = False
            new_count = len(acc.new_embeddings)

        if save and new_count > 0 and not acc.saved:
            msg = f"Person exited — saving {new_count} partial embeddings (total will be {acc.current_total})"
            logger.info(f"🔄 ACCUMULATOR {acc.person_id}: {msg}")
            if alog:
                alog.info("-" * 50)
                alog.info(f"STOPPED — {msg}")
            acc.saved = True
            self._accumulator_save(instance_id)
        elif acc.saved:
            if alog:
                alog.info("-" * 50)
                alog.info("STOPPED — already saved, removing from memory")
            with self.accumulator_lock:
                self.active_accumulators.pop(instance_id, None)
        else:
            msg = f"no new embeddings collected (0/{acc.target_count - acc.existing_count} needed)"
            logger.info(f"🔄 ACCUMULATOR {acc.person_id}: Stopped — {msg}")
            if alog:
                alog.info("-" * 50)
                alog.info(f"STOPPED — {msg}")
            with self.accumulator_lock:
                self.active_accumulators.pop(instance_id, None)

    def _cleanup_orphaned_accumulators(self):
        """
        Remove accumulators for persons fully deleted from the tracker.
        Only cleans up persons no longer in tracker.people — does NOT touch
        active, LOST, or EXITED-in-grace-period persons.
        """
        with self.accumulator_lock:
            to_remove = []
            for iid, acc in self.active_accumulators.items():
                if acc.person_id not in self.tracker.people:
                    to_remove.append(iid)

            for iid in to_remove:
                acc = self.active_accumulators.get(iid)
                if acc and acc.new_embeddings and not acc.saved:
                    acc.saved = True
                    alog = acc.acc_logger
                    if alog:
                        alog.info("-" * 50)
                        alog.info(f"ORPHAN CLEANUP — person no longer in tracker, saving {len(acc.new_embeddings)} embeddings")
                    logger.info(f"🧹 ORPHAN CLEANUP: Saving {len(acc.new_embeddings)} embeddings for {acc.person_id}")
                    threading.Thread(
                        target=self._accumulator_save,
                        args=(iid,),
                        daemon=True,
                        name=f"acc-orphan-{acc.person_id}"
                    ).start()
                else:
                    if acc and acc.acc_logger:
                        acc.acc_logger.info("-" * 50)
                        acc.acc_logger.info("ORPHAN CLEANUP — removed (no embeddings collected)")
                    self.active_accumulators.pop(iid, None)

    def get_statistics(self):
        """Get processing statistics"""
        return {
            "frames_processed": self.frame_count,
            "tracking": self.tracker.get_statistics(),
            "events": self.event_logger.get_statistics(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Video processor module ready")
