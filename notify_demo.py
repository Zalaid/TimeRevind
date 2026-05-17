#!/usr/bin/env python3
"""
TimeRevind Demo Notification System - Fast Threading Approach
Uses background thread for embedding computation while main thread displays video smoothly
Supports both video files and live camera input
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
from src.core.video_processor import BboxSmoother

# Configuration
START_TIME = "15:59:45"
EXCLUDED_PERSONS = {"jilani", "malaika"}
VERIFY_INTERVAL_SEC = 2.0
VERIFY_DURATION_SEC = 6
MATCH_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.5


class NotificationDemo:
    def __init__(self, source, dry_run: bool = False):
        """Initialize demo with video file or camera index and database."""
        self.dry_run = dry_run
        self.is_camera = isinstance(source, int)

        if self.is_camera:
            self.source_label = f"camera_{source}"
            logger.info(f"Opening camera {source}")
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        logger.info(f"{'Camera' if self.is_camera else 'Video'} ready @ {self.fps:.1f} fps")

        # Load YOLO
        logger.info("Loading YOLO model...")
        self.yolo = YOLO("yolov8n.pt")

        # Load embeddings
        logger.info("Loading embedding manager...")
        self.embedding_manager = get_embedding_manager()

        # Initialize database
        logger.info("Initializing database...")
        self.db_manager, self.embedding_store = initialize_databases()

        # Time tracking — wall-clock based for camera, frame-based for video
        now = datetime.now()
        start_dt = datetime.strptime(START_TIME, "%H:%M:%S")
        self.start_datetime = start_dt.replace(year=now.year, month=now.month, day=now.day)
        self.wall_start = time.time()

        # State tracking
        self.pending_verifications = {}
        self.logged_persons = set()
        self.frame_number = 0
        self.stop_thread = False
        self.embedding_lock = threading.Lock()
        self.embedding_queue = Queue()
        self.matched_tracks = {}       # track_id -> (person_id, distance, wall_time)
        self.queued_track_ids = set()
        self.entry_confirmed_persons = set()

        self.bbox_smoother = BboxSmoother(alpha=0.35, dropout_frames=20)

        # Start embedding worker thread
        logger.info("Starting embedding worker thread...")
        self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.embedding_thread.start()

    def _estimate_pose(self, kps):
        """Estimate head pose from face keypoints."""
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

    def _embedding_worker(self):
        """Background thread: compute embeddings and match to Qdrant."""
        logger.debug("[WORKER] Worker thread started")
        while not self.stop_thread:
            try:
                item = self.embedding_queue.get(timeout=1)
                if item is None:
                    break

                track_id, crop, frame_num, current_time = item

                # Compute embedding (no pick_center_face — avoids wrong face on overlap)
                try:
                    embedding, face_conf, kps, _ = self.embedding_manager.compute_face_embedding(crop)
                    if embedding is None or face_conf < FACE_CONFIDENCE_THRESHOLD:
                        with self.embedding_lock:
                            self.queued_track_ids.discard(track_id)
                        continue
                except Exception as e:
                    logger.debug(f"[WORKER] Embedding error: {e}")
                    with self.embedding_lock:
                        self.queued_track_ids.discard(track_id)
                    continue

                # Estimate pose for better matching accuracy
                pose = self._estimate_pose(kps) if kps is not None else "FRONT"

                # Search Qdrant with pose-aware search
                try:
                    pose_filter = ["FRONT", "UP", "DOWN", "LEFT", "RIGHT"]
                    results = self.embedding_store.search_batch(
                        "face_embeddings", [embedding], top_k=1, pose_filter=pose_filter
                    )

                    if not results or len(results) == 0:
                        with self.embedding_lock:
                            self.matched_tracks[track_id] = (f"unknown_{track_id}", 1.0, time.time())
                            self.queued_track_ids.discard(track_id)
                        logger.info(f"[WORKER] Track {track_id}: UNKNOWN (no DB results)")
                        continue

                    dist = results[0]['distances'][0][0] if isinstance(results[0]['distances'][0], list) else results[0]['distances'][0]
                    meta = results[0]['metadatas'][0][0] if isinstance(results[0]['metadatas'][0], list) else results[0]['metadatas'][0]
                    person_id = meta.get("person_id") if meta else None

                    logger.info(f"[WORKER] Track {track_id}: person={person_id}, dist={dist:.4f}, pose={pose}, face_conf={face_conf:.4f}")

                    if person_id is None or dist > MATCH_THRESHOLD:
                        with self.embedding_lock:
                            self.matched_tracks[track_id] = (f"unknown_{track_id}", dist, time.time())
                            self.queued_track_ids.discard(track_id)
                        logger.info(f"[WORKER] Track {track_id}: UNKNOWN (closest={person_id} dist={dist:.4f})")
                        continue

                    # Store match
                    with self.embedding_lock:
                        self.matched_tracks[track_id] = (person_id, dist, time.time())
                        self.queued_track_ids.discard(track_id)

                    logger.info(f"[MATCH] Track {track_id}: {person_id} @ {current_time.strftime('%H:%M:%S')} (dist={dist:.4f}, pose={pose})")

                except Exception as e:
                    logger.debug(f"[WORKER] Qdrant error: {e}")
                    with self.embedding_lock:
                        self.queued_track_ids.discard(track_id)
                    continue

            except Empty:
                continue
            except Exception as e:
                logger.error(f"[WORKER] Error: {e}")

    def get_current_time(self) -> datetime:
        """Get current time — wall-clock based for both camera and video."""
        elapsed = time.time() - self.wall_start
        return self.start_datetime + timedelta(seconds=elapsed)

    def is_after_4pm(self, dt: datetime) -> bool:
        """Check if time is after 4:00pm."""
        return dt.hour >= 16

    def _run_vote(self, track_id: int, state: dict, time_str: str, after_4pm: bool):
        """Run majority vote on a verification state. Returns True if the state was resolved."""
        from collections import Counter
        checks = state["checks"]
        checks_lower = [c.lower() for c in checks]
        vote_counts = Counter(checks_lower)
        confirmed_person, top_votes = vote_counts.most_common(1)[0]
        confidence = top_votes / len(checks_lower)

        logger.info(f"[{time_str}] VOTE track={track_id}: {dict(vote_counts)} → {confirmed_person} ({confidence:.0%})")

        if confidence < 0.6:
            logger.info(f"[{time_str}] ❌ UNCLEAR - low confidence {confidence:.0%}: {checks_lower}")
            self.log_to_db(state["person_id"], time_str, self.frame_number, state["entry_type"],
                           checks, reason_skipped="UNCLEAR")
            return True

        if confirmed_person in self.entry_confirmed_persons:
            return True

        if not after_4pm:
            state["checks"] = [confirmed_person]
            state["start_wall"] = time.time()
            state["last_check_wall"] = time.time()
            return False

        if confirmed_person in EXCLUDED_PERSONS:
            logger.info(f"[{time_str}] ❌ {confirmed_person} - EXCLUDED")
            self.log_to_db(confirmed_person, time_str, self.frame_number, state["entry_type"],
                           checks, verified_as=confirmed_person, notified=0, reason_skipped="EXCLUDED")
        else:
            logger.info(f"[{time_str}] ✅ {confirmed_person} ({confidence:.0%}) - NOTIFIED")
            if not self.dry_run:
                send_notification(confirmed_person, time_str, state["entry_type"])
            self.log_to_db(confirmed_person, time_str, self.frame_number, state["entry_type"],
                           checks, verified_as=confirmed_person, notified=1)

        self.entry_confirmed_persons.add(confirmed_person)
        return True

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
            if self.is_camera:
                time.sleep(0.01)
                return True
            return False

        self.frame_number += 1
        current_time = self.get_current_time()
        time_str = current_time.strftime("%H:%M:%S")
        after_4pm = self.is_after_4pm(current_time)
        now_wall = time.time()

        # YOLO detection with tracking
        raw_detections = []
        try:
            results = self.yolo.track(frame, persist=True, verbose=False, conf=0.25, iou=0.2, tracker="bytetrack.yaml", classes=[0])
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                for track_id, bbox, conf in zip(ids, results[0].boxes.xyxy.cpu().numpy(), confs):
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    raw_detections.append({
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    })
        except Exception as e:
            logger.debug(f"YOLO error: {e}")

        smoothed = self.bbox_smoother.smooth(raw_detections, self.frame_number)
        track_ids_raw = [d["track_id"] for d in smoothed]
        detections = []
        for d in smoothed:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            detections.append((x1, y1, x2, y2))

        # Clear state for tracks no longer in frame — flush vote if enough checks collected
        current_track_ids = set(track_ids_raw)
        for track_id in list(self.pending_verifications.keys()):
            if track_id not in current_track_ids:
                state = self.pending_verifications[track_id]
                checks = state.get("checks", [])
                person = state.get("person_id", "?")
                if len(checks) >= 3 and after_4pm:
                    logger.info(f"[{time_str}] Track {track_id} ({person}) left frame — flushing vote ({len(checks)} checks)")
                    self._run_vote(track_id, state, time_str, after_4pm)
                else:
                    logger.info(f"[{time_str}] Track {track_id} ({person}) left frame — dropped ({len(checks)} checks)")
                del self.pending_verifications[track_id]
                with self.embedding_lock:
                    self.matched_tracks.pop(track_id, None)
                    self.queued_track_ids.discard(track_id)

        # Queue detections for embedding (smart queuing with 2-sec re-verify)
        for track_id, bbox in zip(track_ids_raw, detections):
            x1, y1, x2, y2 = bbox

            with self.embedding_lock:
                already_matched = track_id in self.matched_tracks
                already_queued = track_id in self.queued_track_ids
                match_wall_time = self.matched_tracks.get(track_id, (None, None, 0))[2]

            needs_reverify = already_matched and (now_wall - match_wall_time) >= VERIFY_INTERVAL_SEC

            if not already_queued and (not already_matched or needs_reverify):
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    self.embedding_queue.put((track_id, crop, self.frame_number, current_time))
                    with self.embedding_lock:
                        self.queued_track_ids.add(track_id)

        # Loop 1: Update verification state from matched tracks
        with self.embedding_lock:
            matched_items = dict(self.matched_tracks)

        for track_id, (person_id, distance, match_wall_time) in matched_items.items():
            if track_id not in self.pending_verifications:
                self.pending_verifications[track_id] = {
                    "person_id": person_id,
                    "checks": [person_id],
                    "start_wall": now_wall,
                    "last_check_wall": now_wall,
                    "entry_type": "ENTRY",
                }
                logger.info(f"[{time_str}] Starting verification for track {track_id}: {person_id}")
            else:
                state = self.pending_verifications[track_id]
                if (now_wall - state["last_check_wall"]) >= VERIFY_INTERVAL_SEC:
                    state["checks"].append(person_id)
                    state["last_check_wall"] = now_wall
                    logger.info(f"[{time_str}] Re-check track {track_id}: {person_id} ({len(state['checks'])}/3)")

        # Loop 2: Evaluate completed verifications
        for track_id, state in list(self.pending_verifications.items()):
            seconds_since_start = now_wall - state["start_wall"]

            if seconds_since_start >= VERIFY_DURATION_SEC and len(state["checks"]) >= 3:
                resolved = self._run_vote(track_id, state, time_str, after_4pm)
                if resolved:
                    del self.pending_verifications[track_id]

        # Draw bounding boxes
        frame_display = frame.copy()
        matched_person_ids_this_frame = set()  # dedup: same person_id can't own two boxes
        for track_id, bbox in zip(track_ids_raw, detections):
            x1, y1, x2, y2 = bbox
            with self.embedding_lock:
                match = self.matched_tracks.get(track_id)
                is_queued = track_id in self.queued_track_ids

            if match is None:
                color = (0, 255, 255) if is_queued else (128, 128, 128)
                label = f"Track #{track_id} | Identifying..." if is_queued else f"Track #{track_id} | No face"
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            else:
                person_id = match[0].lower()
                is_unknown = person_id.startswith("unknown_")

                if is_unknown:
                    color = (0, 0, 255)
                    label = f"UNKNOWN #{track_id} | Verifying..."
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                elif person_id in matched_person_ids_this_frame:
                    # Duplicate match — thin yellow box, no label (same person seen twice due to overlap)
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    continue
                elif person_id in EXCLUDED_PERSONS:
                    color = (0, 255, 0)
                    label = f"{person_id.upper()} | Excluded"
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                    matched_person_ids_this_frame.add(person_id)
                else:
                    color = (255, 100, 0)
                    label = f"{person_id.upper()} | Verifying..."
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                    matched_person_ids_this_frame.add(person_id)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            if y1 - th - 8 >= 0:
                cv2.rectangle(frame_display, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame_display, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame_display, (x1, y1), (x1 + tw + 6, y1 + th + 8), color, -1)
                cv2.putText(frame_display, label, (x1 + 3, y1 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD
        status_color = (0, 200, 255) if after_4pm else (180, 180, 180)
        status_text = "MONITORING ACTIVE" if after_4pm else "Waiting for 4 PM..."
        hud = (f"Camera | {time_str} | {status_text}" if self.is_camera
               else f"Frame: {self.frame_number}/{self.total_frames} | {time_str} | {status_text}")
        cv2.putText(frame_display, hud, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        cv2.imshow("TimeRevind Demo - Press Q to quit", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    def run(self):
        """Process entire video or camera stream."""
        logger.info("=" * 60)
        logger.info(f"Starting {'CAMERA' if self.is_camera else 'VIDEO'} processing...")
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
    group.add_argument("--video", type=str, help="Path to video file")
    group.add_argument("--camera", type=int, help="Camera device index (e.g. 0)")
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