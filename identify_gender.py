#!/usr/bin/env python3
"""
Identity + Gender Detection
Same Qdrant matching as test_db_matching.py — gender label added on top.

Usage:
    python identify_gender.py --video "C:/Users/Zalaid/Desktop/noti2.mp4"
"""

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

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.config import YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD, FACE_CONFIDENCE_THRESHOLD, FACE_EMBEDDING_L2_THRESHOLD
from src.database.db_init import EmbeddingStore, DatabaseManager
from src.core.embeddings import EmbeddingManager
from src.core.video_processor import BboxSmoother

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colors (BGR)
COLOR_MATCHED = (0, 255, 0)      # green  — known person
COLOR_UNKNOWN = (0, 255, 255)    # yellow — not in DB yet


def draw_box(frame, x1, y1, x2, y2, label, box_color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Draw label bar just above the box; if too close to top, draw inside
    if y1 - th - 10 >= 0:
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 8, y1 + th + 10), box_color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


class IdentifyGender:

    def __init__(self):
        logger.info("Loading YOLO...")
        self.yolo = YOLO(YOLO_MODEL)

        logger.info("Loading embedding manager...")
        self.embeddings = EmbeddingManager()

        logger.info("Loading Qdrant + DB...")
        self.embedding_store = EmbeddingStore()
        self.db_manager = DatabaseManager()

        self.embedding_lock = threading.Lock()
        self.bbox_smoother   = BboxSmoother(alpha=0.35, dropout_frames=20)
        self.frame_count     = 0
        self.fps             = 30.0

        # Qdrant matching state (identical to test_db_matching.py)
        self.qdrant_matches      = {}   # track_id -> person_id
        self.match_distances     = {}   # track_id -> distance
        self.matched_timestamps  = {}   # track_id -> time.time()
        self.queued_track_ids    = set()
        self.embedding_queue     = Queue()
        self.stop_thread         = False

        # Gender cache so we don't re-run InsightFace every frame
        self.gender_cache = {}   # track_id -> "Male" / "Female" / "?"

        self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.embedding_thread.start()
        logger.info("Ready — press Q to quit\n")

    # ------------------------------------------------------------------ #
    #  Background embedding + Qdrant matching thread                       #
    # ------------------------------------------------------------------ #

    def _embedding_worker(self):
        while not self.stop_thread:
            try:
                item = self.embedding_queue.get(timeout=1)
                if item is None:
                    break
                track_id, crop = item
                with self.embedding_lock:
                    self.queued_track_ids.discard(track_id)

                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue

                with self.embedding_lock:
                    try:
                        emb, conf, kps, _ = self.embeddings.compute_face_embedding(crop)
                    except Exception as e:
                        logger.debug(f"Embedding error track {track_id}: {e}")
                        continue

                if emb is None or conf < FACE_CONFIDENCE_THRESHOLD:
                    continue

                pose = self._estimate_pose(kps) if kps is not None else "FRONT"
                matched_id, dist = self._search_qdrant(emb, pose)

                with self.embedding_lock:
                    if matched_id:
                        self.qdrant_matches[track_id]     = matched_id
                        self.match_distances[track_id]    = dist
                        self.matched_timestamps[track_id] = time.time()
                    elif track_id in self.qdrant_matches:
                        # re-verify failed — remove stale match
                        self.qdrant_matches.pop(track_id, None)
                        self.match_distances.pop(track_id, None)
                        self.matched_timestamps.pop(track_id, None)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _estimate_pose(self, kps):
        if kps is None or len(kps) < 5:
            return "FRONT"
        le, re, nos, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
        eye_mid    = (le + re) / 2.0
        mouth_mid  = (lm + rm) / 2.0
        eye_width  = float(re[0] - le[0])
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

    def _search_qdrant(self, emb, pose):
        try:
            results = self.embedding_store.search_batch(
                "face_embeddings", [emb], top_k=1,
                pose_filter=["FRONT", "UP", "DOWN", "LEFT", "RIGHT"]
            )
            if not results:
                return None, None
            dist = results[0]['distances'][0][0] if isinstance(results[0]['distances'][0], list) else results[0]['distances'][0]
            meta = results[0]['metadatas'][0][0] if isinstance(results[0]['metadatas'][0], list) else results[0]['metadatas'][0]
            person_id = meta.get('person_id') if meta else None
            if dist <= FACE_EMBEDDING_L2_THRESHOLD:
                return person_id, dist
            return None, dist
        except Exception as e:
            logger.warning(f"Qdrant search error: {e}")
            return None, None

    # ------------------------------------------------------------------ #
    #  Gender helpers                                                       #
    # ------------------------------------------------------------------ #

    def _get_gender(self, track_id: int, person_id) -> str:
        """Return gender from person_profiles DB. Only cached once a real value is found."""
        cached = self.gender_cache.get(track_id)
        if cached and cached != "?":
            return cached

        gender = "?"
        if person_id:
            try:
                rows = self.db_manager.query(
                    "SELECT gender FROM person_profiles WHERE person_id = ?", (person_id,)
                )
                if rows and rows[0][0]:
                    g = rows[0][0].strip().upper()
                    gender = "Male" if g in ("M", "MALE") else "Female" if g in ("F", "FEMALE") else "?"
            except Exception:
                pass

        self.gender_cache[track_id] = gender
        return gender

    # ------------------------------------------------------------------ #
    #  YOLO detect                                                          #
    # ------------------------------------------------------------------ #

    def _detect(self, frame):
        results = self.yolo.track(
            frame, conf=0.25, iou=YOLO_CONFIDENCE_THRESHOLD,
            persist=True, tracker="bytetrack.yaml",
            verbose=False, classes=[0]
        )
        detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id) if box.id is not None else None
                    if track_id is not None:
                        detections.append({
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(box.conf[0].cpu().numpy()),
                        })
        return self.bbox_smoother.smooth(detections, self.frame_count)

    # ------------------------------------------------------------------ #
    #  Main video loop                                                      #
    # ------------------------------------------------------------------ #

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            sys.exit(1)

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps
        logger.info(f"Video: {Path(video_path).name}  |  {total} frames @ {fps:.1f} fps")
        logger.info("Keys: [a/f] ±10 frames  [w/e] ±100 frames  [q] quit\n")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count   += 1
                self.frame_count = frame_count

                detections = self._detect(frame)
                now_time   = time.time()

                # Queue crops for embedding matching
                for det in detections:
                    track_id = det["track_id"]
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    with self.embedding_lock:
                        already_matched = track_id in self.qdrant_matches
                        already_queued  = track_id in self.queued_track_ids
                        match_time      = self.matched_timestamps.get(track_id, 0)

                    needs_reverify = already_matched and (now_time - match_time) >= 5.0

                    if not already_queued and (not already_matched or needs_reverify):
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            self.embedding_queue.put((track_id, crop))
                            with self.embedding_lock:
                                self.queued_track_ids.add(track_id)

                # Draw boxes
                annotated = frame.copy()
                with self.embedding_lock:
                    matches_snapshot = dict(self.qdrant_matches)

                for det in detections:
                    track_id = det["track_id"]
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    matched_id = matches_snapshot.get(track_id)
                    gender     = self._get_gender(track_id, matched_id)

                    if matched_id:
                        label = f"{matched_id}  |  {gender}" if gender != "?" else matched_id
                        draw_box(annotated, x1, y1, x2, y2, label, COLOR_MATCHED)
                    else:
                        draw_box(annotated, x1, y1, x2, y2, "Identifying...", COLOR_UNKNOWN)

                # HUD
                cv2.putText(annotated, f"Frame {frame_count}/{total}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Identity + Gender  |  Q=quit", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    frame_count = max(0, frame_count - 10)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                elif key == ord('f'):
                    frame_count = min(total - 1, frame_count + 10)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                elif key == ord('w'):
                    frame_count = max(0, frame_count - 100)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                elif key == ord('e'):
                    frame_count = min(total - 1, frame_count + 100)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        finally:
            self.stop_thread = True
            self.embedding_thread.join(timeout=5)
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"Done. Processed {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="Identity + Gender Detection")
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    if not Path(args.video).exists():
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)

    IdentifyGender().process_video(args.video)


if __name__ == "__main__":
    main()
