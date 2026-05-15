"""
Manual Labeling Script for Demo Videos
Detects cache miss, asks user for person ID, collects embeddings, saves to Qdrant
"""

import cv2
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from src.config import YOLO_MODEL, FACE_CONFIDENCE_THRESHOLD
from src.core.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ManualLabeler:
    """Interactive manual labeling for cache miss re-identifications"""

    def __init__(self, video_path, output_dir="manual_labels_output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Person tracking: person_label -> {embeddings, poses, frames, confidence}
        self.labeled_persons = {}
        self.new_embeddings_count = 0  # Track how many embeddings are NEW this session

        # Initialize models
        logger.info("Loading YOLO model...")
        self.yolo_model = YOLO(YOLO_MODEL)

        logger.info("Loading embedding model...")
        self.embeddings_manager = EmbeddingManager()

        # Load existing embeddings from Qdrant
        logger.info("Loading embeddings from Qdrant...")
        self._load_from_qdrant()

        logger.info(f"Output directory: {self.output_dir}")

    def _load_from_qdrant(self):
        """Load all existing embeddings from Qdrant to show matching scores"""
        try:
            from src.database.db_init import EmbeddingStore
            import numpy as np

            store = EmbeddingStore()
            offset = 0

            while True:
                points, next_offset = store.client.scroll(
                    collection_name="face_embeddings",
                    offset=offset,
                    limit=100,
                    with_vectors=True,
                    with_payload=True
                )

                if not points:
                    break

                for point in points:
                    person_id = point.payload.get("person_id")
                    if person_id:
                        if person_id not in self.labeled_persons:
                            self.labeled_persons[person_id] = {
                                'embeddings': [],
                                'poses': [],
                                'frame_numbers': [],
                                'confidence': [],
                                'from_qdrant': True  # Mark as loaded from DB
                            }

                        vector = np.array(point.vector, dtype=np.float32) if point.vector else None
                        self.labeled_persons[person_id]['embeddings'].append(vector)
                        self.labeled_persons[person_id]['poses'].append(point.payload.get('pose', 'FRONT'))
                        self.labeled_persons[person_id]['frame_numbers'].append(-1)  # From DB, no frame number
                        self.labeled_persons[person_id]['confidence'].append(point.payload.get('confidence', 0.0))

                if next_offset is None:
                    break
                offset = next_offset

            # Print loaded summary
            if self.labeled_persons:
                total = sum(len(data['embeddings']) for data in self.labeled_persons.values())
                print(f"\n✓ Loaded {total} existing embeddings from Qdrant:")
                for person, data in self.labeled_persons.items():
                    pose_counts = {}
                    for pose in data['poses']:
                        pose_counts[pose] = pose_counts.get(pose, 0) + 1
                    print(f"  {person}: {len(data['embeddings'])} | Poses: {pose_counts}")
                print()

        except Exception as e:
            logger.warning(f"Could not load embeddings from Qdrant: {e}")

    def process_video_with_labels(self):
        """Process video with manual seeking and labeling"""

        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        previous_people = {}  # track_id -> person_label mapping

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("\n" + "="*80)
        print("MANUAL LABELING MODE - Interactive Video Navigation")
        print("="*80)
        print(f"Video: {self.video_path}")
        print(f"Total frames: {total_frames} | FPS: {fps}")
        print("\n📌 CONTROLS:")
        print("  [SPACE]     → Pause/Play current frame")
        print("  [a/f]       → Skip backward/forward 10 frames")
        print("  [w/e]       → Skip backward/forward 100 frames")
        print("  [d]         → Process detections at current frame")
        print("  [s]         → Show current statistics")
        print("  [q]         → Quit and save")
        print("="*80 + "\n")

        paused = False
        manual_seek = False
        frame = None
        cached_results = None  # Cache YOLO results to avoid rerunning on same frame
        last_frame_count = -1

        # Suppress YOLO verbose logging
        import logging as py_logging
        py_logging.getLogger('ultralytics').setLevel(py_logging.WARNING)

        while True:
            # Only read next frame if not paused
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

            if frame is None:
                break

            # Display frame with instructions
            display_frame = frame.copy()

            # Run YOLO only if frame changed (avoid recomputing on same paused frame)
            if frame_count != last_frame_count:
                cached_results = self.yolo_model(display_frame)
                last_frame_count = frame_count

            results = cached_results

            # Draw YOLO person detections only (class 0 = person in COCO)

            if results and len(results) > 0:
                boxes = results[0].boxes
                for idx, box in enumerate(boxes):
                    # Only draw PERSONS (class 0 in COCO dataset)
                    if box.cls[0].item() == 0:  # 0 = person class
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else -idx

                        # Draw green box around person
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label
                        label = f"Track {track_id} ({conf:.0%})"
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Parse detections (also filters to persons only)
            detections = self._parse_detections(results)

            self._add_frame_info_overlay(display_frame, frame_count, total_frames, paused)

            cv2.imshow("Video Navigator", display_frame)
            key = cv2.waitKey(1)
            key_char = key & 0xFF  # For ASCII keys

            # Handle keyboard controls
            if key_char == ord('q'):  # Quit
                print("\n\n✓ Quitting and saving...")
                break
            elif key_char == ord(' '):  # Space - Pause/Play
                paused = not paused
                if paused:
                    print(f"\n⏸️  PAUSED at frame {frame_count}")
                    self._print_statistics()
                    self._print_options()
                continue
            elif key_char == ord('r'):  # Resume from pause
                if paused:
                    paused = False
                    print("▶️  Resuming...")
                continue
            elif key_char == ord('d'):  # Process detections at current frame
                print(f"\n🔍 Processing detections at frame {frame_count}...")
                results = self.yolo_model(frame)
                detections = self._parse_detections(results)

                if detections:
                    previous_people = self._process_detections(frame, detections, previous_people, frame_count)
                else:
                    print("  No detections found")
                continue
            elif key_char == ord('s'):  # Show statistics
                print("\n")
                self._print_statistics()
                continue
            elif key_char == ord('a'):  # 'a' - skip back 10 frames
                new_frame = max(0, frame_count - 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_count = new_frame
                print(f"Skipped back 10 frames to frame {frame_count}")
                continue
            elif key_char == ord('f'):  # 'f' - skip forward 10 frames
                new_frame = min(total_frames - 1, frame_count + 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_count = new_frame
                print(f"Skipped forward 10 frames to frame {frame_count}")
                continue
            elif key_char == ord('w'):  # 'w' - skip back 100 frames
                new_frame = max(0, frame_count - 100)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_count = new_frame
                print(f"Skipped back 100 frames to frame {frame_count}")
                continue
            elif key_char == ord('e'):  # 'e' - skip forward 100 frames
                new_frame = min(total_frames - 1, frame_count + 100)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_count = new_frame
                print(f"Skipped forward 100 frames to frame {frame_count}")
                continue
            elif key != -1:  # Debug: print unrecognized keys
                print(f"[DEBUG] Key pressed: {key} (char: {key_char})")
                continue

            # Only increment frame count when not paused
            if not paused:
                frame_count += 1

        cv2.destroyAllWindows()
        return self.labeled_persons

    def _process_detections(self, frame, detections, previous_people, frame_count):
        """Process detections at current frame and ask for labels"""

        current_people_in_frame = {}

        for detection in detections:
            track_id = detection.get("track_id")
            bbox = detection.get("bbox")
            conf = detection.get("confidence", 0.5)

            if track_id is None:
                continue

            # Check cache hit (track seen before)
            if track_id in previous_people:
                person_label = previous_people[track_id]
                current_people_in_frame[track_id] = person_label
                # logger.debug(f"Cache HIT: track {track_id} → {person_label}")
                continue

            # CACHE MISS: Extract face crop with embedding info
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            # Validate crop
            if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
                print(f"   ⚠️  Crop too small: {crop.shape if crop is not None else 'None'}")
                continue

            # Compute face embedding to get confidence and pose
            try:
                embedding, face_conf, _, kps = self.embeddings_manager.compute_face_embedding(crop)
                pose = self._estimate_pose(kps) if kps is not None else "FRONT"
            except Exception as e:
                logger.debug(f"Error computing embedding: {e}")
                embedding = None
                face_conf = 0.0
                pose = "UNKNOWN"

            # Ensure face_conf is not None
            if face_conf is None:
                face_conf = 0.0

            # Display the crop (always, even if embedding failed)
            try:
                display_crop = cv2.resize(crop, (400, 400))

                # Add info overlay
                info_text = [
                    f"Confidence: {face_conf:.1%}",
                    f"Pose: {pose}",
                    f"Size: {x2-x1}x{y2-y1}px"
                ]

                # Add background for text readability
                for idx, text in enumerate(info_text):
                    y_pos = 35 + (idx * 35)
                    # Draw background rectangle
                    cv2.rectangle(display_crop, (5, y_pos - 25), (400, y_pos + 10), (0, 0, 0), -1)
                    # Draw text
                    cv2.putText(display_crop, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                window_name = f"Track {track_id} - {face_conf:.0%} | {pose}"
                cv2.imshow(window_name, display_crop)
                cv2.waitKey(500)  # Wait longer to ensure window renders

                # Show matching scores against already-labeled persons
                if embedding is not None and self.labeled_persons:
                    self._show_matching_scores(embedding, pose)

            except Exception as e:
                print(f"   ❌ Error displaying crop: {e}")
                logger.debug(f"Display error: {e}")

            # Ask user for label
            print(f"\n{'─'*80}")
            print(f"🔴 CACHE MISS at Frame {frame_count}")
            print(f"   Track ID: {track_id}")
            print(f"   Confidence: {conf:.2%}")
            print(f"\n   👉 Face displayed in popup window above 👈")
            print(f"\n   Existing persons: {list(self.labeled_persons.keys())}")
            print(f"   Or enter NEW person name")

            # Keep window visible while waiting for input
            cv2.waitKey(1)
            user_input = input("   → Enter person ID: ").strip().lower()

            cv2.destroyAllWindows()  # Close the face display window

            if user_input == 'skip' or user_input == '':
                print("   ⊘ Skipped")
                continue

            person_label = user_input
            print(f"   ✓ Labeled as: {person_label}")
            current_people_in_frame[track_id] = person_label

            # Compute face embedding
            try:
                embedding, conf_face, _, kps = self.embeddings_manager.compute_face_embedding(crop)

                if embedding is not None:
                    # Estimate pose from keypoints
                    pose = self._estimate_pose(kps)

                    # Initialize person entry if new
                    if person_label not in self.labeled_persons:
                        self.labeled_persons[person_label] = {
                            'embeddings': [],
                            'poses': [],
                            'frame_numbers': [],
                            'confidence': [],
                            'from_qdrant': False  # Mark as new (not from Qdrant)
                        }

                    # Store embedding
                    self.labeled_persons[person_label]['embeddings'].append(embedding)
                    self.labeled_persons[person_label]['poses'].append(pose)
                    self.labeled_persons[person_label]['frame_numbers'].append(frame_count)
                    self.labeled_persons[person_label]['confidence'].append(float(conf_face))
                    self.labeled_persons[person_label]['from_qdrant'] = False  # Mark as modified (has new data)
                    self.new_embeddings_count += 1

                    count = len(self.labeled_persons[person_label]['embeddings'])
                    print(f"   ✓ Embedding collected (pose: {pose}, {count} total)")
                else:
                    print(f"   ✗ Could not compute face embedding")

            except Exception as e:
                logger.error(f"Error processing detection: {e}")

        # Return updated previous_people tracking
        return current_people_in_frame

    def save_embeddings_local(self):
        """Save embeddings to local file as backup (timestamped for accumulation)"""
        import pickle
        from datetime import datetime

        if not self.labeled_persons:
            logger.warning("No labeled persons to save")
            return

        # Create timestamped backup filename for accumulation
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_file = self.output_dir / f"embeddings_backup_{timestamp}.pkl"

        try:
            with open(backup_file, 'wb') as f:
                pickle.dump(self.labeled_persons, f)
            print(f"\n  💾 Local backup saved: {backup_file}")
            logger.info(f"Embeddings backed up to {backup_file}")
        except Exception as e:
            print(f"  ❌ Error saving local backup: {e}")
            logger.error(f"Error saving embeddings backup: {e}")

    def save_to_qdrant(self):
        """Save labeled embeddings to Qdrant database"""

        if not self.labeled_persons:
            logger.warning("No labeled persons to save")
            return

        print("\n" + "="*80)
        print("SAVING TO QDRANT")
        print("="*80)

        from src.database.db_init import EmbeddingStore
        import time

        # Generate unique session ID using timestamp
        session_id = int(time.time() * 1000) % (2**31)

        # Initialize embedding store
        embedding_store = EmbeddingStore()

        for person_label, data in self.labeled_persons.items():
            try:
                embeddings_list = data['embeddings']
                poses_list = data['poses']
                frames_list = data['frame_numbers']

                # Count existing vs new
                existing_count = sum(1 for frame in frames_list if frame == -1)
                new_count = sum(1 for frame in frames_list if frame >= 0)

                if new_count == 0:
                    print(f"\n  👤 {person_label}: No new embeddings (all {existing_count} from Qdrant)")
                    continue

                print(f"\n  👤 {person_label}: {new_count} NEW embeddings (+ {existing_count} existing)")

                # Count per pose for NEW embeddings only
                pose_counts = {}
                for pose, frame in zip(poses_list, frames_list):
                    if frame >= 0:  # Only NEW ones
                        pose_counts[pose] = pose_counts.get(pose, 0) + 1
                print(f"     Poses: {pose_counts}")

                # Save only NEW embeddings to Qdrant
                saved_count = 0
                failed_count = 0
                for idx, (embedding, pose, frame) in enumerate(zip(embeddings_list, poses_list, frames_list)):
                    if frame < 0:  # Skip existing (loaded from Qdrant)
                        continue
                    metadata = {
                        "pose": pose,
                        "embedding_index": idx,
                        "total_embeddings": len(embeddings_list),
                        "session_id": session_id
                    }

                    try:
                        embedding_store.add_face_embedding(
                            person_id=person_label,
                            embedding=embedding,
                            metadata=metadata
                        )
                        saved_count += 1
                    except Exception as e:
                        failed_count += 1
                        print(f"     ❌ Error embedding {idx}: {e}")
                        logger.warning(f"  Error saving embedding {idx} for {person_label}: {e}")

                if failed_count == 0:
                    print(f"     ✅ {saved_count} embeddings saved to Qdrant")
                else:
                    print(f"     ⚠️  {saved_count} saved, {failed_count} FAILED!")

            except Exception as e:
                print(f"  ❌ Error saving {person_label}: {e}")
                logger.error(f"Error saving {person_label}: {e}")

        logger.info("\n✓ Qdrant save complete!")

    def save_metadata(self):
        """Save labeling metadata to JSON for reference"""

        metadata = {
            'video_path': str(self.video_path),
            'timestamp': datetime.now().isoformat(),
            'total_persons': len(self.labeled_persons),
            'persons': {}
        }

        for person_label, data in self.labeled_persons.items():
            pose_counts = {}
            for pose in data['poses']:
                pose_counts[pose] = pose_counts.get(pose, 0) + 1

            metadata['persons'][person_label] = {
                'total_embeddings': len(data['embeddings']),
                'poses_breakdown': pose_counts,
                'frames_collected': data['frame_numbers'],
                'avg_confidence': float(np.mean(data['confidence'])),
                'min_confidence': float(np.min(data['confidence'])),
                'max_confidence': float(np.max(data['confidence']))
            }

        metadata_file = self.output_dir / "labeling_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Metadata saved to {metadata_file}")

    def _parse_detections(self, results):
        """Parse YOLO results - PERSONS ONLY (class 0 in COCO)"""
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                # Only include PERSON class (0 in COCO dataset)
                # Skip chairs, tables, and other objects
                if box.cls[0].item() == 0:  # 0 = person
                    detection = {
                        'track_id': int(box.id[0]) if box.id is not None else -idx,
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf[0])
                    }
                    detections.append(detection)

        return detections

    def _show_matching_scores(self, new_embedding, new_pose):
        """Show similarity scores between new embedding and ALL stored persons/poses"""
        print(f"\n{'─'*80}")
        print(f"📊 MATCHING SCORES (new detection: {new_pose} pose):")
        print(f"{'─'*80}")

        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        # Compare against ALL persons and ALL poses
        for person_label, data in self.labeled_persons.items():
            embeddings = data['embeddings']
            poses = data['poses']

            # Group embeddings by pose
            poses_dict = {}
            for emb, pose in zip(embeddings, poses):
                if pose not in poses_dict:
                    poses_dict[pose] = []
                poses_dict[pose].append(emb)

            print(f"  👤 {person_label}:")
            for pose, pose_embeddings in poses_dict.items():
                # Calculate similarities for this pose
                sims = [cosine_sim(new_embedding, stored_emb) for stored_emb in pose_embeddings]
                best_sim = max(sims)

                # Convert to percentage (0-100%)
                best_pct = best_sim * 100

                # Match icon and pose marker
                match_icon = "✅" if best_sim > 0.75 else "⚠️ " if best_sim > 0.50 else "❌"
                pose_marker = "→" if pose == new_pose else " "
                print(f"      {match_icon} {pose_marker} {pose:6s} | Match: {best_pct:.0f}%")

        print(f"{'─'*80}\n")

    def _estimate_pose(self, kps):
        """Estimate face pose (FRONT/LEFT/RIGHT) from keypoints"""
        if kps is None or len(kps) < 5:
            return "FRONT"

        try:
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]

            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose[0]

            # Simple heuristic based on nose position relative to eyes
            if nose_x < eye_center_x - 5:
                return "LEFT"
            elif nose_x > eye_center_x + 5:
                return "RIGHT"
            else:
                return "FRONT"
        except:
            return "FRONT"

    def _add_frame_info_overlay(self, frame, frame_count, total_frames, paused):
        """Add frame info overlay to the displayed frame"""
        h, w = frame.shape[:2]

        # Status bar
        status_text = "⏸️  PAUSED" if paused else "▶️  Playing"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Frame counter
        progress_text = f"Frame {frame_count}/{total_frames}"
        cv2.putText(frame, progress_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Progress bar
        progress = frame_count / total_frames if total_frames > 0 else 0
        bar_width = 300
        bar_height = 20
        x, y = 10, 120
        filled = int(bar_width * progress)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), 2)
        cv2.rectangle(frame, (x, y), (x + filled, y + bar_height), (0, 255, 0), -1)

        # Instructions
        instructions = [
            "[SPACE] Pause  [d] Process  [s] Stats  [q] Quit",
            "[a/f] ±10 frames  [w/e] ±100 frames"
        ]
        for idx, text in enumerate(instructions):
            cv2.putText(frame, text, (10, h - 40 + (idx * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def _print_statistics(self):
        """Print current statistics for all labeled persons"""
        print("\n" + "─"*80)
        print("📊 CURRENT STATISTICS")
        print("─"*80)

        if not self.labeled_persons:
            print("  No persons labeled yet")
        else:
            for person_label, data in self.labeled_persons.items():
                pose_counts = {}
                for pose in data['poses']:
                    pose_counts[pose] = pose_counts.get(pose, 0) + 1

                print(f"\n  {person_label.upper()}:")
                print(f"    ✓ Embeddings: {len(data['embeddings'])}")
                print(f"    ✓ Poses: {pose_counts}")
                print(f"    ✓ Avg confidence: {np.mean(data['confidence']):.2%}")

        print("─"*80 + "\n")

    def _print_options(self):
        """Print available options while paused"""
        print("\n📌 OPTIONS WHILE PAUSED:")
        print("  [d] Process detections at this frame")
        print("  [s] Show statistics")
        print("  [r] Resume playback")
        print("  [q] Quit\n")

    def _print_summary(self):
        """Print summary of labeled data"""

        print("\n" + "="*80)
        print("LABELING SUMMARY")
        print("="*80)

        total_embeddings = 0

        for person_label, data in self.labeled_persons.items():
            pose_counts = {}
            for pose in data['poses']:
                pose_counts[pose] = pose_counts.get(pose, 0) + 1

            total_embeddings += len(data['embeddings'])

            print(f"\n{person_label.upper()}:")
            print(f"  Total embeddings: {len(data['embeddings'])}")
            print(f"  Poses breakdown: {pose_counts}")
            print(f"  Frames: {data['frame_numbers'][:3]}... (showing first 3)")
            print(f"  Avg face confidence: {np.mean(data['confidence']):.2%}")

        print(f"\n{'─'*80}")
        print(f"Total embeddings collected: {total_embeddings}")
        print(f"Total persons labeled: {len(self.labeled_persons)}")
        print("="*80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Manual Labeling for Demo Videos")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--output-dir",
        default="manual_labels_output",
        help="Output directory for metadata"
    )
    parser.add_argument(
        "--save-qdrant",
        action="store_true",
        help="Save embeddings to Qdrant after processing"
    )

    args = parser.parse_args()

    # Check if video exists
    if not Path(args.video_path).exists():
        logger.error(f"Video not found: {args.video_path}")
        return

    # Create labeler
    labeler = ManualLabeler(args.video_path, args.output_dir)

    try:
        # Process video
        logger.info("\n📹 STEP 1: Processing video with manual labeling...")
        labeled_data = labeler.process_video_with_labels()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        labeled_data = labeler.labeled_persons

    if not labeled_data:
        logger.warning("No data was labeled")
        return

    # Save metadata (ALWAYS save, even on interrupt)
    logger.info("\n📋 STEP 2: Saving metadata...")
    labeler.save_metadata()

    # Save local backup (ALWAYS save, even on interrupt)
    logger.info("\n💾 STEP 3: Saving local backup...")
    labeler.save_embeddings_local()

    # Save to Qdrant if requested (ALWAYS save, even on interrupt)
    if args.save_qdrant:
        logger.info("\n🔗 STEP 4: Saving embeddings to Qdrant...")
        labeler.save_to_qdrant()
    else:
        logger.info("\n💡 Tip: Run with --save-qdrant flag to save to Qdrant")

    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nNext steps:")
    logger.info(f"1. Check metadata in: {labeler.output_dir}/labeling_metadata.json")
    logger.info(f"2. Run your main.py to match against saved embeddings")
    logger.info("="*80)


if __name__ == "__main__":
    main()
