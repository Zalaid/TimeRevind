#!/usr/bin/env python3
"""
TimeRevind Demo Notification System - Instant Alert (No Verification)
Sends notification immediately on first detection of non-excluded persons after 4 PM
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
MATCH_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.3


class NotificationDemoInstant:
    def __init__(self, video_path: str, dry_run: bool = False):
        """Initialize demo with video file and database."""
        self.video_path = Path(video_path)
        self.dry_run = dry_run

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load video
        logger.info(f"Loading video: {self.video_path.name}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            logger.error(f"❌ Failed to open video: {self.video_path}")
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

        # Parse start time
        start_dt = datetime.strptime(START_TIME, "%H:%M:%S")
        self.start_datetime = start_dt.replace(year=2026, month=5, day=16)

        # State tracking
        self.frame_number = 0
        self.stop_thread = False
        self.embedding_lock = threading.Lock()
        self.embedding_queue = Queue()
        self.matched_tracks = {}  # track_id -> (person_id, distance, timestamp)
        self.queued_track_ids = set()
        self.notified_persons = set()  # person_id values that already got notified

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
        """Get simulated current time based on frame number."""
        elapsed_seconds = frame_num / self.fps
        return self.start_datetime + timedelta(seconds=elapsed_seconds)

    def is_after_4pm(self, dt: datetime) -> bool:
        """Check if time is after 4:00pm."""
        return dt.hour >= 16

    def log_to_db(self, person_id: str, simulated_time: str, frame_num: int,
                  entry_type: str, notified: int = 0):
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
                json.dumps([person_id]), person_id, notified,
                None, str(self.video_path)
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
        cv2.putText(frame_display, f"Frame: {self.frame_number}/{self.total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Time: {time_str}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("TimeRevind Demo (Instant) - Press Q to quit", frame_display)

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

        # Queue detections for embedding (smart queuing)
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

            # Only queue if new or needs re-verify (2 sec interval)
            needs_reverify = (already_matched and (now_time - match_time) >= 2.0)

            if not already_queued and (not already_matched or needs_reverify):
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    self.embedding_queue.put((track_id, crop, self.frame_number, current_time))
                    with self.embedding_lock:
                        self.queued_track_ids.add(track_id)

        # Process matches — instant notification for non-excluded persons
        with self.embedding_lock:
            matched_items = dict(self.matched_tracks)

        for track_id, (person_id, distance, match_time) in matched_items.items():
            person_id_lower = person_id.lower()

            # Skip before 4pm
            if not after_4pm:
                continue

            # Check if already notified this person
            if person_id_lower in self.notified_persons:
                continue

            # Excluded persons
            if person_id_lower in EXCLUDED_PERSONS:
                logger.info(f"[{time_str}] ❌ {person_id_lower} - EXCLUDED")
                self.notified_persons.add(person_id_lower)
                continue

            # Non-excluded: send notification immediately
            logger.info(f"[{time_str}] ✅ {person_id_lower} - NOTIFIED (instant)")
            if not self.dry_run:
                send_notification(person_id_lower, time_str, "ENTRY")
            self.log_to_db(person_id_lower, time_str, self.frame_number, "ENTRY", notified=1)
            self.notified_persons.add(person_id_lower)

        return True

    def run(self):
        """Process entire video."""
        logger.info("=" * 60)
        logger.info("Starting video processing (INSTANT mode - no verification)...")
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
    parser = argparse.ArgumentParser(description="TimeRevind Demo Instant Notification (No Verification)")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--dry-run", action="store_true", help="Log only, don't send notifications")

    args = parser.parse_args()

    try:
        demo = NotificationDemoInstant(args.video, dry_run=args.dry_run)
        demo.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
