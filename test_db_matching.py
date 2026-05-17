#!/usr/bin/env python3
"""Test: Draw boxes ONLY on people MATCHED to database via Qdrant"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from queue import Queue, Empty
import threading

import cv2
import torch
from datetime import datetime

os.environ['PATH'] = os.path.join(os.path.dirname(torch.__file__), 'lib') + os.pathsep + os.environ.get('PATH', '')

from ultralytics import YOLO
from src.config import YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD, FACE_CONFIDENCE_THRESHOLD, FACE_EMBEDDING_L2_THRESHOLD
from src.database.db_init import EmbeddingStore, DatabaseManager
from src.core.embeddings import EmbeddingManager
from src.core.video_processor import BboxSmoother

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class QdrantTestMatcher:
    """Test with real Qdrant matching"""

    def __init__(self):
        logger.info("Loading models...")
        self.yolo = YOLO(YOLO_MODEL)
        self.embeddings = EmbeddingManager()
        self.embedding_store = EmbeddingStore()
        self.embedding_lock = threading.Lock()
        self.bbox_smoother = BboxSmoother(alpha=0.35, dropout_frames=20)
        self.frame_count = 0

        # Matching: track_id -> database_person_id
        self.qdrant_matches = {}  # track_id -> matched_db_person_id
        self.match_distances = {}  # track_id -> qdrant distance float
        self.matched_timestamps = {}  # track_id -> timestamp when matched (for 5-sec re-verification)
        self.queued_track_ids = set()  # Track IDs we've already queued (don't re-queue every frame)
        self.matched_db_people_this_frame = set()  # Database person_ids already matched in current frame
        self.embedding_queue = Queue()
        self.stop_thread = False

        # Visit state: person_id -> state dict
        self.person_visit_state = {}
        self.active_person_ids_last_frame = set()

        # Track if person's total_visits has been incremented for this session
        self.person_session_counted = set()  # (person_id, session_id) tuples

        # Delayed confirmation: track_id -> pending person waiting for 6-sec stability
        self.pending_confirmations = {}  # track_id -> {person_id, timestamp, frame, confirmations}
        self.pending_original_frames = {}  # track_id -> original_frame (persists across identity changes)
        self.CONFIRMATION_DURATION = 6.0  # seconds
        self.MIN_CONFIRMATIONS = 2  # minimum matches needed

        # DB
        self.db_manager = DatabaseManager()
        self.session_id = f"test_{int(time.time())}"
        self.video_file = None
        self.fps = 30.0
        self.db_manager.execute(
            "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
            (self.session_id, datetime.now().isoformat())
        )
        logger.info(f"[DB] Session created: {self.session_id}")


        # Start embedding worker thread
        self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.embedding_thread.start()

        logger.info("✓ Ready\n")

    def _embedding_worker(self):
        """Background thread: compute embeddings and match to Qdrant"""
        logger.debug("[WORKER-THREAD] Worker thread started")
        while not self.stop_thread:
            try:
                item = self.embedding_queue.get(timeout=1)
                if item is None:
                    break

                track_id, crop = item
                logger.debug(f"[WORKER-RECEIVED] Track {track_id}: got item from queue")
                with self.embedding_lock:
                    self.queued_track_ids.discard(track_id)
                logger.debug(f"[WORKER-ENTRY] Track {track_id}: starting, queued_set={self.queued_track_ids}")
                logger.debug(f"[WORKER-START] Track {track_id}: processing crop {crop.shape}")

                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    logger.debug(f"[WORKER-EXIT-SMALL] Track {track_id}: crop too small, will retry next frame")
                    continue

                # Get face embedding (ONNX/CUDA may have threading issues)
                with self.embedding_lock:
                    try:
                        emb, conf, kps, _ = self.embeddings.compute_face_embedding(crop)
                        logger.debug(f"[WORKER-EMB] Track {track_id}: embedding computed (conf={conf})")
                    except Exception as e:
                        logger.error(f"[WORKER-EMB-ERROR] Track {track_id}: {e}", exc_info=True)
                        continue

                if emb is None or conf < FACE_CONFIDENCE_THRESHOLD:
                    logger.debug(f"[WORKER-EXIT-LOWCONF] Track {track_id}: conf={conf}, will retry next frame")
                    continue

                logger.debug(f"[WORKER-OK] Track {track_id}: ready for Qdrant search")

                # Estimate pose
                pose = self._estimate_pose(kps) if kps is not None else "FRONT"

                # Search Qdrant
                matched_id, dist = self._search_qdrant(emb, pose)

                with self.embedding_lock:
                    is_reverify = track_id in self.qdrant_matches
                    old_match = self.qdrant_matches.get(track_id)

                if matched_id:
                    with self.embedding_lock:
                        self.qdrant_matches[track_id] = matched_id
                        self.match_distances[track_id] = dist
                        self.matched_timestamps[track_id] = time.time()

                    if is_reverify:
                        logger.info(f"[RE-VERIFY] Track {track_id}: {old_match} -> {matched_id} (still valid)")
                else:
                    with self.embedding_lock:
                        # If this was a re-verification and no match found, remove the old match
                        if is_reverify:
                            old_match = self.qdrant_matches.pop(track_id, None)
                            self.match_distances.pop(track_id, None)
                            self.matched_timestamps.pop(track_id, None)
                            logger.info(f"[RE-VERIFY] Track {track_id}: {old_match} no longer matches (removed)")

            except Empty:
                continue  # Normal: queue empty during timeout
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

    def _estimate_pose(self, kps):
        """Estimate head pose from keypoints"""
        if kps is None or len(kps) < 5:
            return "FRONT"
        le, re, nos, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
        eye_mid = (le + re) / 2.0
        mouth_mid = (lm + rm) / 2.0
        eye_width = float(re[0] - le[0])
        face_height = float(mouth_mid[1] - eye_mid[1])
        if eye_width < 5 or face_height < 5:
            return "FRONT"
        yaw = (nos[0] - eye_mid[0]) / eye_width
        pitch_dev = (nos[1] - eye_mid[1]) / face_height - 0.50
        if abs(yaw) < 0.12 and abs(pitch_dev) < 0.10:
            return "FRONT"
        if abs(yaw) >= abs(pitch_dev):
            return "RIGHT" if yaw > 0 else "LEFT"
        return "DOWN" if pitch_dev > 0 else "UP"

    def _search_qdrant(self, emb, pose):
        """Search Qdrant for matching face"""
        try:
            # Allow matching across all poses
            pose_filter = {
                "FRONT": ["FRONT", "UP", "DOWN", "LEFT", "RIGHT"],
                "UP": ["FRONT", "UP", "DOWN", "LEFT", "RIGHT"],
                "DOWN": ["FRONT", "UP", "DOWN", "LEFT", "RIGHT"],
                "LEFT": ["FRONT", "UP", "DOWN", "LEFT", "RIGHT"],
                "RIGHT": ["FRONT", "UP", "DOWN", "LEFT", "RIGHT"],
            }.get(pose, ["FRONT", "UP", "DOWN", "LEFT", "RIGHT"])

            logger.debug(f"[SEARCH] Detected pose: {pose} | Searching in poses: {pose_filter}")

            results = self.embedding_store.search_batch(
                "face_embeddings", [emb], top_k=1, pose_filter=pose_filter
            )

            if not results or len(results) == 0:
                logger.debug(f"[SEARCH] No results returned from Qdrant")
                return None, None

            dist = results[0]['distances'][0][0] if isinstance(results[0]['distances'][0], list) else results[0]['distances'][0]
            meta = results[0]['metadatas'][0][0] if isinstance(results[0]['metadatas'][0], list) else results[0]['metadatas'][0]

            person_id = meta.get('person_id') if meta else None
            matched_pose = meta.get('pose', 'UNKNOWN') if meta else 'UNKNOWN'
            point_id = results[0]['ids'][0][0] if isinstance(results[0]['ids'][0], list) else results[0]['ids'][0] if results[0]['ids'] else None

            # Check threshold
            logger.info(f"[QDRANT] Found: {person_id} | Distance: {dist:.4f} (threshold: {FACE_EMBEDDING_L2_THRESHOLD})")
            logger.info(f"         Detected pose: {pose} | Matched pose: {matched_pose} | Point ID: {point_id}")

            if dist <= FACE_EMBEDDING_L2_THRESHOLD:
                logger.info(f"[MATCH-OK] Frame {self.frame_count}: {person_id} ACCEPTED (pose {pose} → {matched_pose})")
                return person_id, dist
            else:
                logger.info(f"[MATCH-REJECT] {person_id} too far (dist {dist:.4f} > threshold {FACE_EMBEDDING_L2_THRESHOLD})")
                return None, dist
        except Exception as e:
            logger.warning(f"Qdrant search error: {e}", exc_info=True)
            return None, None


    def detect(self, frame):
        """YOLO detection"""
        results = self.yolo.track(
            frame,
            conf=0.25,
            iou=YOLO_CONFIDENCE_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                if box.cls == 0:  # person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    track_id = int(box.id) if box.id is not None else None

                    if track_id is not None:
                        detections.append({
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                        })

        # Apply bbox smoothing
        detections = self.bbox_smoother.smooth(detections, self.frame_count)

        return detections

    def _ensure_person_profile(self, person_id):
        existing = self.db_manager.query(
            "SELECT person_id FROM person_profiles WHERE person_id = ?", (person_id,)
        )
        if not existing:
            now_str = datetime.now().isoformat()
            self.db_manager.execute(
                """INSERT INTO person_profiles
                   (person_id, name, first_seen, last_seen, total_visits, total_time_seconds, updated_at)
                   VALUES (?, ?, ?, ?, 0, 0, ?)""",
                (person_id, person_id, now_str, now_str, now_str)
            )
            logger.info(f"[DB-PROFILE] Created profile for {person_id}")

    def _log_event_to_db(self, person_id, event_type, visit_num, frame_count, timestamp_override=None):
        timestamp_str = timestamp_override if timestamp_override else datetime.now().isoformat()
        self.db_manager.execute(
            """INSERT INTO events
               (timestamp, person_id, event_type, visit_num, session_id, video_file, video_timestamp_start)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp_str, person_id, event_type, visit_num, self.session_id, self.video_file, frame_count)
        )
        logger.info(f"[DB-EVENT] {person_id}: {event_type} (visit {visit_num})")

    def _log_activity_to_db(self, person_id, frame_count, dist, visit_num):
        now_str = datetime.now().isoformat()
        frame_duration = int(1.0 / self.fps) if self.fps > 0 else 0
        self.db_manager.execute(
            """INSERT INTO activity_events
               (timestamp, person_id, event_type, action, confidence, visit_num,
                session_id, video_file, video_timestamp_start, duration_seconds)
               VALUES (?, ?, 'INTERACTION', 'MATCHED', ?, ?, ?, ?, ?, ?)""",
            (now_str, person_id, dist, visit_num, self.session_id, self.video_file, frame_count, frame_duration)
        )

    def process_video(self, video_path, output_path=None):
        """Process video"""
        logger.info(f"Opening: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error("Failed to open video")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = fps if fps > 0 else 30.0
        self.video_file = os.path.basename(video_path) if isinstance(video_path, str) else f"camera_{video_path}"
        self.db_manager.execute(
            "UPDATE sessions SET video_file = ? WHERE session_id = ?",
            (self.video_file, self.session_id)
        )

        is_camera = isinstance(video_path, int)
        if is_camera:
            logger.info(f"Camera {video_path}: {width}x{height} @ {fps:.1f} FPS (live)")
            logger.info("Keyboard: [q] Quit\n")
        else:
            logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total} frames")
            logger.info("Keyboard: [a/f] +-10 frames | [w/e] +-100 frames | [q] Quit\n")

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        match_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.frame_count = frame_count

                # YOLO detection only (not using PersonTracker for this test)
                detections = self.detect(frame)

                if detections:
                    logger.debug(f"Frame {frame_count}: {len(detections)} detections")

                # Queue people for embedding matching
                now_time = time.time()
                for detection in detections:
                    track_id = detection["track_id"]
                    bbox = detection["bbox"]
                    x1, y1, x2, y2 = [int(v) for v in bbox]

                    with self.embedding_lock:
                        already_matched = track_id in self.qdrant_matches
                        already_queued = track_id in self.queued_track_ids
                        match_time = self.matched_timestamps.get(track_id)

                    # Check if in pending confirmation
                    in_pending = track_id in self.pending_confirmations

                    needs_reverify = (already_matched and match_time and
                                     (now_time - match_time) >= 2.0)

                    if not already_queued:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            # Queue if: not matched yet, or needs re-verify, or in pending but not confirmed yet
                            if not already_matched or needs_reverify:
                                self.embedding_queue.put((track_id, crop))
                                with self.embedding_lock:
                                    self.queued_track_ids.add(track_id)
                                logger.debug(f"[QUEUE] Track {track_id} queued")
                                if needs_reverify:
                                    logger.debug(f"[RE-VERIFY-QUEUE] Track {track_id} re-verification")
                        else:
                            logger.debug(f"[SKIP] Track {track_id}: crop too small")

                # --- Add new matches to pending confirmation ---
                now = time.time()
                with self.embedding_lock:
                    for track_id, person_id in list(self.qdrant_matches.items()):
                        # If newly matched (not in pending, not in visit_state)
                        if track_id not in self.pending_confirmations and person_id not in self.person_visit_state:
                            self.pending_confirmations[track_id] = {
                                'person_id': person_id,
                                'timestamp': now,
                                'frame': frame_count,
                                'confirmations': 1
                            }
                            # Save original frame for this track (persists across identity changes)
                            if track_id not in self.pending_original_frames:
                                self.pending_original_frames[track_id] = frame_count
                            logger.info(f"[PENDING] Track {track_id}: {person_id} detected, waiting 6s for confirmation")
                            # Don't log ENTERED yet - wait for 6-second confirmation!

                # --- Check pending confirmations (6-sec stability) ---
                confirmed_this_frame = {}  # track_id -> (person_id, timestamp, frame) (newly confirmed)

                for track_id, pending in list(self.pending_confirmations.items()):
                    elapsed = now - pending['timestamp']
                    detected_person = pending['person_id']

                    # Check if this track_id still has a match in this frame
                    current_match = None
                    with self.embedding_lock:
                        current_match = self.qdrant_matches.get(track_id)

                    if current_match == detected_person:
                        # Still matching same person
                        pending['confirmations'] += 1

                        # Only confirm after 6 seconds (time-based, not confirmation count)
                        if elapsed >= self.CONFIRMATION_DURATION:
                            # Confirmed! Use original detection time and frame
                            orig_frame = self.pending_original_frames.get(track_id, pending['frame'])
                            confirmed_this_frame[track_id] = (detected_person, pending['timestamp'], orig_frame)
                            del self.pending_confirmations[track_id]
                            if track_id in self.pending_original_frames:
                                del self.pending_original_frames[track_id]
                            logger.info(f"[CONFIRMED] Track {track_id}: {detected_person} confirmed after {elapsed:.1f}s")
                    else:
                        # Changed to different person or lost
                        if current_match:
                            logger.info(f"[RE-VERIFY-UPDATE] Track {track_id}: changed from {detected_person} to {current_match}, keeping original timestamp")
                            # Update person_id but preserve original detection timestamp and frame
                            orig_frame = self.pending_original_frames.get(track_id, pending['frame'])
                            self.pending_confirmations[track_id] = {
                                'person_id': current_match,
                                'timestamp': pending['timestamp'],  # Keep original timestamp
                                'frame': orig_frame,  # Use saved original frame
                                'confirmations': 1
                            }
                        else:
                            # No match - discard pending (but keep original frame for potential re-match)
                            del self.pending_confirmations[track_id]
                            # Note: pending_original_frames[track_id] is preserved for when match returns

                # --- Visit state: detect currently matched person_ids this frame ---
                current_matched_person_ids = set()
                with self.embedding_lock:
                    for det in detections:
                        pid = self.qdrant_matches.get(det["track_id"])
                        if pid:
                            current_matched_person_ids.add(pid)

                # Start 30s loss timer for people who were active last frame but not this frame
                lost_person_ids = self.active_person_ids_last_frame - current_matched_person_ids
                for person_id in lost_person_ids:
                    state = self.person_visit_state.get(person_id)
                    if state and state.get('loss_timer_start') is None and not state.get('exited_logged'):
                        state['loss_timer_start'] = time.time()
                        logger.info(f"[LOSS-TIMER] {person_id}: started 30s timer")

                # Check 30s timers — keep accumulating time during loss window, stop at EXITED
                now = time.time()
                frame_time = 1.0 / self.fps
                for person_id, state in self.person_visit_state.items():
                    if state.get('loss_timer_start') and not state.get('exited_logged'):
                        if (now - state['loss_timer_start']) >= 30.0:
                            self._log_event_to_db(person_id, 'EXITED', state['visit_num'], frame_count)
                            state['exited_logged'] = True
                            logger.info(f"[EXITED] {person_id}: visit {state['visit_num']} closed after 30s absence")
                        else:
                            # Still in loss window — keep time running
                            self.db_manager.execute(
                                """UPDATE person_profiles SET total_time_seconds = total_time_seconds + ?,
                                   updated_at = ? WHERE person_id = ?""",
                                (frame_time, datetime.now().isoformat(), person_id)
                            )

                # Draw annotations
                annotated = frame.copy()
                self.matched_db_people_this_frame = set()

                # Handle newly confirmed people from 6-sec stability check
                for track_id, (person_id, orig_timestamp, orig_frame) in confirmed_this_frame.items():
                    # Ensure profile exists and increment total_visits
                    self._ensure_person_profile(person_id)
                    person_session_key = (person_id, self.session_id)
                    if person_session_key not in self.person_session_counted:
                        self.db_manager.execute(
                            """UPDATE person_profiles SET total_visits = total_visits + 1,
                               updated_at = ? WHERE person_id = ?""",
                            (datetime.now().isoformat(), person_id)
                        )
                        self.person_session_counted.add(person_session_key)
                        logger.info(f"[SESSION-COUNT] {person_id}: total_visits incremented (new session)")

                    # Initialize visit state for confirmed person (if not already done)
                    if person_id not in self.person_visit_state:
                        # Calculate time already elapsed since original detection
                        elapsed_since_detection = now - orig_timestamp

                        # Add initial elapsed time to database immediately
                        self.db_manager.execute(
                            """UPDATE person_profiles SET total_time_seconds = total_time_seconds + ?,
                               updated_at = ? WHERE person_id = ?""",
                            (elapsed_since_detection, datetime.now().isoformat(), person_id)
                        )

                        self.person_visit_state[person_id] = {
                            'visit_num': 1,
                            'visit_start': orig_timestamp,
                            'last_seen': now,
                            'accumulated_time': elapsed_since_detection,  # Start with elapsed time since first detection
                            'entered_logged': False,
                            'loss_timer_start': None,
                            'exited_logged': False,
                        }

                    # Check person's last event in database
                    last_event_result = self.db_manager.query(
                        "SELECT event_type FROM events WHERE person_id = ? ORDER BY id DESC LIMIT 1",
                        (person_id,)
                    )
                    last_event = last_event_result[0][0] if last_event_result else None

                    # Log ENTERED only if last event is not ENTERED
                    if last_event != 'ENTERED':
                        orig_timestamp_str = datetime.fromtimestamp(orig_timestamp).isoformat()
                        self._log_event_to_db(person_id, 'ENTERED', 1, orig_frame, timestamp_override=orig_timestamp_str)
                        self.person_visit_state[person_id]['entered_logged'] = True
                        logger.info(f"[DB-CHECK] {person_id}: logged ENTERED (last_event was {last_event})")
                    else:
                        logger.info(f"[DB-CHECK] {person_id}: skipped ENTERED (already entered, last_event={last_event})")

                    logger.info(f"[NEW-CONFIRMATION] Track {track_id}: {person_id} confirmed (detected at frame {orig_frame})")

                for detection in detections:
                    track_id = detection["track_id"]
                    bbox = detection["bbox"]
                    x1, y1, x2, y2 = [int(v) for v in bbox]

                    with self.embedding_lock:
                        matched_id = self.qdrant_matches.get(track_id)
                        dist = self.match_distances.get(track_id)

                    # Use confirmed match or current match
                    if track_id in confirmed_this_frame:
                        matched_id, orig_timestamp, orig_frame = confirmed_this_frame[track_id]

                    if matched_id:
                        if matched_id in self.matched_db_people_this_frame:
                            # Duplicate in same frame — draw YELLOW, skip DB logging
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        else:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                annotated, matched_id, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                            )
                            self.matched_db_people_this_frame.add(matched_id)
                            match_count += 1

                            # --- Visit state logic ---
                            now = time.time()
                            state = self.person_visit_state.get(matched_id)

                            # Don't create visit_state if person is still in pending confirmation
                            is_pending = any(p['person_id'] == matched_id for p in self.pending_confirmations.values())
                            if is_pending:
                                continue  # Skip visit state logic until confirmed

                            if state is None:
                                # First time seeing this person in this session
                                self._ensure_person_profile(matched_id)

                                # Increment total_visits once per session for this person
                                person_session_key = (matched_id, self.session_id)
                                if person_session_key not in self.person_session_counted:
                                    self.db_manager.execute(
                                        """UPDATE person_profiles SET total_visits = total_visits + 1,
                                           updated_at = ? WHERE person_id = ?""",
                                        (datetime.now().isoformat(), matched_id)
                                    )
                                    self.person_session_counted.add(person_session_key)
                                    logger.info(f"[SESSION-COUNT] {matched_id}: total_visits incremented (new session)")

                                state = {
                                    'visit_num': 1,
                                    'visit_start': now,
                                    'last_seen': now,
                                    'accumulated_time': 0.0,
                                    'entered_logged': False,
                                    'loss_timer_start': None,
                                    'exited_logged': False,
                                }
                                self.person_visit_state[matched_id] = state

                            elif state.get('loss_timer_start') is not None:
                                elapsed_loss = now - state['loss_timer_start']
                                if elapsed_loss >= 30.0:
                                    # 30s gap = new visit
                                    if not state.get('exited_logged'):
                                        self._log_event_to_db(matched_id, 'EXITED', state['visit_num'], frame_count)
                                    state['visit_num'] += 1
                                    state['visit_start'] = now
                                    state['accumulated_time'] = 0.0
                                    state['entered_logged'] = False
                                    state['exited_logged'] = False
                                    self.db_manager.execute(
                                        """UPDATE person_profiles SET total_visits = total_visits + 1,
                                           updated_at = ? WHERE person_id = ?""",
                                        (datetime.now().isoformat(), matched_id)
                                    )
                                    logger.info(f"[NEW-VISIT] {matched_id}: starting visit {state['visit_num']} (was absent {elapsed_loss:.1f}s)")
                                else:
                                    # Returned within 30s — same visit, reset timer
                                    logger.info(f"[RETURN] {matched_id}: back within {elapsed_loss:.1f}s, same visit {state['visit_num']}")
                                state['loss_timer_start'] = None
                                state['exited_logged'] = False

                            # Log ENTERED on first frame of this visit (but only if confirmed, not pending)
                            if not state.get('entered_logged'):
                                # Check if this person is still in pending confirmation
                                is_pending = any(p['person_id'] == matched_id for p in self.pending_confirmations.values())

                                if not is_pending:
                                    self._log_event_to_db(matched_id, 'ENTERED', state['visit_num'], frame_count)
                                    state['entered_logged'] = True

                            # Accumulate time and update profile
                            frame_time = 1.0 / self.fps
                            state['accumulated_time'] += frame_time
                            state['last_seen'] = now

                            self.db_manager.execute(
                                """UPDATE person_profiles SET last_seen = ?,
                                   total_time_seconds = total_time_seconds + ?,
                                   updated_at = ? WHERE person_id = ?""",
                                (datetime.now().isoformat(), frame_time,
                                 datetime.now().isoformat(), matched_id)
                            )


                    else:
                        # Draw YELLOW box for people not in database yet
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)

                self.active_person_ids_last_frame = current_matched_person_ids

                # Frame counter
                cv2.putText(
                    annotated, f"Frame: {frame_count}/{total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                cv2.imshow("Qdrant Matching Test", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('a'):  # Rewind 10 frames
                    new_frame = max(0, frame_count - 10)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    frame_count = new_frame
                    logger.info(f"Jumped to frame {frame_count}")
                elif key == ord('f'):  # Forward 10 frames
                    new_frame = min(total - 1, frame_count + 10)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    frame_count = new_frame
                    logger.info(f"Jumped to frame {frame_count}")
                elif key == ord('w'):  # Rewind 100 frames
                    new_frame = max(0, frame_count - 100)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    frame_count = new_frame
                    logger.info(f"Jumped to frame {frame_count}")
                elif key == ord('e'):  # Forward 100 frames
                    new_frame = min(total - 1, frame_count + 100)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    frame_count = new_frame
                    logger.info(f"Jumped to frame {frame_count}")

                if out:
                    out.write(annotated)

                if frame_count % 100 == 0:
                    with self.embedding_lock:
                        queue_size = self.embedding_queue.qsize()
                    logger.info(f"Frame {frame_count}/{total} | Queue: {queue_size} | Matches: {match_count}")

        finally:
            # Log EXITED for all people still active at video end
            for person_id, state in self.person_visit_state.items():
                if state.get('entered_logged') and not state.get('exited_logged'):
                    self._log_event_to_db(person_id, 'EXITED', state['visit_num'], frame_count)
                    logger.info(f"[CLEANUP] {person_id}: EXITED logged at video end")

            self.db_manager.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (datetime.now().isoformat(), self.session_id)
            )

            self.stop_thread = True
            self.embedding_thread.join(timeout=5)
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        logger.info(f"\nDone! Qdrant Matches: {match_count}")


def main():
    parser = argparse.ArgumentParser(description="Test: Real Qdrant matching")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  type=str, help="Video file path")
    group.add_argument("--camera", type=int, help="Camera device ID (e.g. 0)")
    parser.add_argument("--output", type=str, default=None, help="Output path (video mode only)")
    args = parser.parse_args()

    matcher = QdrantTestMatcher()

    if args.video:
        if not Path(args.video).exists():
            logger.error("Video not found")
            sys.exit(1)
        matcher.process_video(str(args.video), args.output)
    else:
        matcher.process_video(args.camera, args.output)


if __name__ == "__main__":
    main()
