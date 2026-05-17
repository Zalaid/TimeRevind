#!/usr/bin/env python3
"""
TimeRevind Demo Notification System - Fast Threading Approach
Uses background thread for embedding computation while main thread displays video smoothly
"""

import cv2
import argparse
import logging
import json
import sqlite3
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from queue import Queue, Empty

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.core.embeddings import get_embedding_manager
from src.database.db_init import initialize_databases
from src.core.notifier import send_notification

# Configuration
START_TIME = "15:59:30"
EXCLUDED_PERSONS = {"jilani", "malaika"}
VERIFY_INTERVAL_SEC = 1
VERIFY_DURATION_SEC = 3
MATCH_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.3


class NotificationDemo:
    def __init__(self, source, dry_run: bool = False):
        """Initialize demo with video file or camera ID."""
        self.dry_run = dry_run
        self.is_camera = isinstance(source, int)

        if self.is_camera:
            self.source_label = f"camera_{source}"
            logger.info(f"Opening camera {source}")
            self.cap = cv2.VideoCapture(source)
        else:
            self.video_path = Path(source)
            self.source_label = self.video_path.name
            if not self.video_path.exists():
                raise FileNotFoundError(f"Video not found: {source}")
            logger.info(f"Loading video: {self.video_path.name}")
            self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.is_camera:
            logger.info(f"Camera ready @ {self.fps:.1f} fps")
        else:
            logger.info(f"Video: {self.total_frames} frames @ {self.fps:.1f} fps")

        # Load YOLO
        logger.info("Loading YOLO model...")
        self.yolo = YOLO("yolov8n.pt")

        # Load embeddings
        logger.info("Loading embedding manager...")
        self.embedding_manager = get_embedding_manager()

        # Initialize database
        logger.info("Initializing database...")
        self.db_manager, self.embedding_store = initialize_databases()

        # Simulated time only used for video mode
        if not self.is_camera:
            start_dt = datetime.strptime(START_TIME, "%H:%M:%S")
            self.start_datetime = start_dt.replace(year=2026, month=5, day=16)

        # State tracking
        self.pending_verifications = {}
        self.logged_persons = set()  # Track persons already logged (person_id)
        self.frame_number = 0
        self.stop_thread = False
        self.embedding_lock = threading.Lock()
        self.embedding_queue = Queue()
        self.matched_tracks = {}  # track_id -> (person_id, distance, timestamp)
        self.queued_track_ids = set()
        self.entry_confirmed_persons = set()  # person_id values that got ENTRY notification, prevents re-logging

        # Start embedding worker thread
        logger.info("Starting embedding worker thread...")
        self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.embedding_thread.start()

    def _embedding_worker(self):
        """Background thread: compute embeddings and match to Qdrant."""
        logger.debug("[WORKER] Worker thread started")
        while not self.stop_thread:
            try:
                item = self.embedding_queue.get(timeout=1)
                if item is None:
                    break

                track_id, crop, frame_num, current_time = item

                # Compute embedding
                try:
                    embedding, face_conf, _, _ = self.embedding_manager.compute_face_embedding(
                        crop, pick_center_face=True
                    )
                    if embedding is None or face_conf < FACE_CONFIDENCE_THRESHOLD:
                        continue
                except Exception as e:
                    logger.debug(f"[WORKER] Embedding error: {e}")
                    continue

                # Search Qdrant
                try:
                    results = self.embedding_store.search_face_embedding(embedding, top_k=1)
                    if not results["metadatas"] or not results["metadatas"][0]:
                        continue

                    person_id = results["metadatas"][0][0].get("person_id")
                    distance = results["distances"][0][0]

                    if person_id is None or distance > MATCH_THRESHOLD:
                        continue
                    
                    # Store match
                    with self.embedding_lock:
                        self.matched_tracks[track_id] = (person_id, distance, time.time())
                        self.queued_track_ids.discard(track_id)

                    logger.info(f"[MATCH] Track {track_id}: {person_id} @ {current_time.strftime('%H:%M:%S')} (dist={distance:.4f})")

                except Exception as e:
                    logger.debug(f"[WORKER] Qdrant error: {e}")
                    continue

            except Empty:
                continue
            except Exception as e:
                logger.error(f"[WORKER] Error: {e}")

    def get_current_time(self, frame_num: int) -> datetime:
        """Real wall-clock time for camera; simulated frame-based time for video."""
        if self.is_camera:
            return datetime.now()
        elapsed_seconds = frame_num / self.fps
        return self.start_datetime + timedelta(seconds=elapsed_seconds)

    def is_after_4pm(self, dt: datetime) -> bool:
        """Check if time is after 4:00pm."""
        return dt.hour >= 16

    def log_to_db(self, person_id: str, simulated_time: str, frame_num: int,
                  entry_type: str, verification_checks: list, verified_as: str = None,
                  notified: int = 0, reason_skipped: str = None):
        """Log notification event to database."""
        try:
            sql = """
                INSERT INTO notification_logs
                (person_id, simulated_time, video_frame, entry_type, verification_checks,
                 verified_as, notified, reason_skipped, video_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                person_id, simulated_time, frame_num, entry_type,
                json.dumps(verification_checks), verified_as, notified,
                reason_skipped, self.source_label
            )
            self.db_manager.execute(sql, params)
        except Exception as e:
            logger.error(f"Database error: {e}")

    def process_frame(self):
        """Process single frame."""
        ret, frame = self.cap.read()
        if not ret:
            return False

        self.frame_number += 1
        current_time = self.get_current_time(self.frame_number - 1)
        time_str = current_time.strftime("%H:%M:%S")
        after_4pm = self.is_after_4pm(current_time)

        # Display frame
        frame_display = frame.copy()
        hud_top = f"Camera | Time: {time_str}" if self.is_camera else f"Frame: {self.frame_number}/{self.total_frames} | Time: {time_str}"
        status = "MONITORING ACTIVE" if after_4pm else "Waiting for 4 PM..."
        status_color = (0, 200, 255) if after_4pm else (180, 180, 180)
        cv2.putText(frame_display, hud_top, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.imshow("TimeRevind Demo - Press Q to quit", frame_display)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        # YOLO detection with tracking
        try:
            results = self.yolo.track(frame, persist=True, verbose=False, conf=0.3)
            if results[0].boxes.id is None:
                return True
            track_ids_raw = results[0].boxes.id.int().cpu().tolist()
            detections = results[0].boxes.xyxy.cpu().numpy()
        except Exception as e:
            logger.debug(f"YOLO error: {e}")
            return True

        # Queue detections for embedding (smart queuing: only queue if new or needs re-verify)
        now_time = time.time()
        for track_id, bbox in zip(track_ids_raw, detections):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            with self.embedding_lock:
                already_matched = track_id in self.matched_tracks
                already_queued = track_id in self.queued_track_ids
                match_time = self.matched_tracks.get(track_id, (None, None, 0))[2] if already_matched else 0

            # Only queue if: not yet matched, or matched but needs re-verify (2+ sec interval)
            needs_reverify = (already_matched and (now_time - match_time) >= VERIFY_INTERVAL_SEC)

            if not already_queued and (not already_matched or needs_reverify):
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    self.embedding_queue.put((track_id, crop, self.frame_number, current_time))
                    with self.embedding_lock:
                        self.queued_track_ids.add(track_id)

        # Loop 1: Update verification state from matched tracks
        with self.embedding_lock:
            matched_items = dict(self.matched_tracks)

        for track_id, (person_id, distance, match_time) in matched_items.items():
            # Start verification regardless of time — only check time at confirmation
            if track_id not in self.pending_verifications:
                self.pending_verifications[track_id] = {
                    "person_id": person_id,
                    "checks": [person_id],
                    "start_frame": self.frame_number,
                    "start_time": current_time,
                    "last_check_frame": self.frame_number,
                    "entry_type": "ENTRY",
                }
                logger.info(f"[{time_str}] Starting verification for track {track_id}: {person_id}")
            else:
                state = self.pending_verifications[track_id]
                # Stop re-checking after 3 checks
                if len(state["checks"]) >= 3:
                    continue
                if (self.frame_number - state["last_check_frame"]) / self.fps >= VERIFY_INTERVAL_SEC:
                    state["checks"].append(person_id)
                    state["last_check_frame"] = self.frame_number
                    logger.info(f"[{time_str}] Re-check track {track_id}: {person_id} ({len(state['checks'])}/3)")

        # Loop 2: Evaluate completed verifications
        for track_id, state in list(self.pending_verifications.items()):
            frames_since_start = self.frame_number - state["start_frame"]
            seconds_since_start = frames_since_start / self.fps

            if seconds_since_start >= VERIFY_DURATION_SEC and len(state["checks"]) >= 3:
                from collections import Counter

                checks = state["checks"]
                checks_lower = [c.lower() for c in checks]

                # Majority vote instead of requiring unanimous agreement
                vote_counts = Counter(checks_lower)
                confirmed_person, top_votes = vote_counts.most_common(1)[0]
                total_votes = len(checks_lower)
                confidence = top_votes / total_votes

                logger.info(f"[{time_str}] Vote result: {dict(vote_counts)} → {confirmed_person} ({confidence:.0%})")

                # Require at least 60% agreement to confirm
                if confidence < 0.6:
                    logger.info(f"[{time_str}] ❌ UNCLEAR - low confidence {confidence:.0%}: {checks_lower}")
                    del self.pending_verifications[track_id]
                    continue

                if confirmed_person in self.entry_confirmed_persons:
                    del self.pending_verifications[track_id]
                    continue

                # Wait until after 4 PM to send notification (but keep verification pending)
                if not after_4pm:
                    logger.debug(f"[{time_str}] Verified {confirmed_person} but before 4 PM — waiting")
                    continue

                if confirmed_person in EXCLUDED_PERSONS:
                    logger.info(f"[{time_str}] ❌ {confirmed_person} - EXCLUDED")
                else:
                    logger.info(f"[{time_str}] ✅ {confirmed_person} - NOTIFIED ({confidence:.0%} confidence)")
                    if not self.dry_run:
                        send_notification(confirmed_person, time_str, state["entry_type"])
                    self.log_to_db(confirmed_person, time_str, self.frame_number, state["entry_type"],
                                  checks, verified_as=confirmed_person, notified=1)

                self.entry_confirmed_persons.add(confirmed_person)
                del self.pending_verifications[track_id]

        return True

    def run(self):
        """Process entire video."""
        logger.info("=" * 60)
        logger.info("Starting video processing...")
        logger.info("=" * 60)

        try:
            while self.process_frame():
                pass
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            self.stop_thread = True
            self.embedding_thread.join(timeout=5)
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("=" * 60)
            logger.info(f"Processing complete! Processed {self.frame_number} frames")
            logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TimeRevind Demo Notification System")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  type=str, help="Path to video file")
    group.add_argument("--camera", type=int, help="Camera device ID (e.g. 0)")
    parser.add_argument("--dry-run", action="store_true", help="Log only, don't send notifications")

    args = parser.parse_args()
    source = args.camera if args.camera is not None else args.video

    try:
        demo = NotificationDemo(source, dry_run=args.dry_run)
        demo.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
