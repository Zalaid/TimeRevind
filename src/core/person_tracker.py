"""
Person Tracker Module
Handles person detection, tracking, and identification across frames
"""

import logging
import uuid
import numpy as np
import threading
from collections import defaultdict
from datetime import datetime, timedelta

from src.config import (
    PERSON_TIMEOUT_SECONDS,
    EXITED_PERSON_MEMORY_SECONDS,
    NEW_VISIT_THRESHOLD_SECONDS,
    TEMP_PERSON_RECYCLE_TIMEOUT,
    FACE_MATCH_THRESHOLD,
    FACE_EMBEDDING_L2_THRESHOLD,
)

logger = logging.getLogger(__name__)


class Person:
    """Represents a tracked person"""

    _id_counter = 0
    _id_lock = threading.Lock()  # Protect ID generation from race conditions
    _free_temp_ids = set()  # Pool of recycled temporary person ID numbers

    def __init__(self, tracker=None):
        # Thread-safe ID generation with reuse from pool
        # ✨ NEW: Avoid reusing IDs that are still active/recent in tracker
        with Person._id_lock:
            reused_num = None
            if Person._free_temp_ids:
                # Try to find a recycled ID that's not in use
                if tracker and hasattr(tracker, 'people'):
                    for candidate_id in sorted(Person._free_temp_ids):
                        candidate_person_id = f"Person_{candidate_id}"
                        # Skip if still tracked (even offline)
                        if candidate_person_id not in tracker.people:
                            reused_num = candidate_id
                            break

                # Fallback: use any from pool if tracker not available or all are in use
                if reused_num is None:
                    reused_num = min(Person._free_temp_ids)

                Person._free_temp_ids.discard(reused_num)
            else:
                Person._id_counter += 1
                reused_num = Person._id_counter
            self.person_id = f"Person_{reused_num}"

        self._instance_id = str(uuid.uuid4())  # unique forever, never reused

        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.visit_number = 1
        self.is_in_frame = True
        self.last_detection_time = datetime.now()
        self.tracking_state = "ACTIVE"  # ACTIVE or LOST (YOLO tracking state)
        self.lost_time = None  # When person was marked as LOST
        self.offline_timer_start = None
        self.offline_frame_count = 0  # Counter for consecutive frames without detection
        self.visited_locations = set()
        self.face_embeddings = []        # Store multiple embeddings for robust matching
        self.face_embedding_poses = []   # Parallel pose label per embedding (FRONT/LEFT/RIGHT/UP/DOWN)
        self.batch_embedding_starts = {}  # batch_num → start index in face_embeddings at time of append
        self.keyframe_path = None
        self.confidence_history = []
        self._creation_frame = None  # Frame number when person was first created (for temporary persons)
        self._final_decision_made = False  # Flag: true when we have 30 embeddings and made final decision (new vs returning)
        self._matched_person_id = None  # Person_1, Person_2, etc. or None if new person
        self._search_history = {10: None, 20: None, 30: None}  # Track search results at each checkpoint
        self._accumulated_frames = []  # Store frame snapshots for saving after final decision
        self._entry_logged_session_id = None  # Track which session the entry was logged for (to allow re-logging in new sessions)

        # ✨ NEW: Periodic re-identification fields
        self.last_reidentification_time = datetime.now()  # When we last verified this person's ID
        self.reidentification_count = 0  # Number of successful re-identifications
        self.reidentification_confidence = 1.0  # Confidence of last re-id check

    @classmethod
    def create_for_existing(cls, person_id: str) -> 'Person':
        """Create Person object for existing known person without consuming an ID."""
        instance = object.__new__(cls)
        instance.person_id = person_id
        instance._instance_id = str(uuid.uuid4())  # unique forever, never reused
        instance.first_seen = datetime.now()
        instance.last_seen = datetime.now()
        instance.visit_number = 1
        instance.is_in_frame = True
        instance.last_detection_time = datetime.now()
        instance.tracking_state = "ACTIVE"
        instance.lost_time = None
        instance.offline_timer_start = None
        instance.offline_frame_count = 0
        instance.visited_locations = set()
        instance.face_embeddings = []
        instance.face_embedding_poses = []
        instance.batch_embedding_starts = {}
        instance.keyframe_path = None
        instance.confidence_history = []
        instance._creation_frame = None
        instance._final_decision_made = False
        instance._matched_person_id = None
        instance._search_history = {10: None, 20: None, 30: None}
        instance._accumulated_frames = []
        instance._entry_logged_session_id = None

        # ✨ NEW: Periodic re-identification fields
        instance.last_reidentification_time = datetime.now()
        instance.reidentification_count = 0
        instance.reidentification_confidence = 1.0
        return instance

    @property
    def time_since_last_detection(self):
        """Get seconds since last detection"""
        return (datetime.now() - self.last_detection_time).total_seconds()

    @property
    def is_timeout(self):
        """Check if person has timed out (> 60 sec offline)"""
        return self.time_since_last_detection > PERSON_TIMEOUT_SECONDS

    def reset_detection(self):
        """Reset detection timer (person re-entered after brief absence)"""
        self.last_detection_time = datetime.now()
        self.offline_timer_start = None
        self.offline_frame_count = 0  # Reset flicker counter when person re-detected

    def start_offline_timer(self):
        """Start offline timer when person leaves frame"""
        if self.offline_timer_start is None:
            self.offline_timer_start = datetime.now()
            self.is_in_frame = False

    def mark_online(self):
        """Mark person as online again"""
        self.is_in_frame = True
        self.last_detection_time = datetime.now()
        self.offline_timer_start = None
        self._exit_processed = False  # Reset guard — allow mark_exited() if person times out again
        if hasattr(self, '_actual_exit'):
            delattr(self, '_actual_exit')  # Clear exit flag so person can be re-detected
        # 🛡️ DO NOT delete _entry_logged here — it's set after matching and should persist
        # Only delete it in mark_exited() when person actually leaves the session

    def mark_lost(self):
        """YOLO lost track but person might still be in frame"""
        # 🛡️ Guard: only mark LOST if truly offline for 12+ frames
        if self.offline_frame_count < 12:
            logger.debug(f"{self.person_id}: not ready for LOST yet ({self.offline_frame_count}/12 frames)")
            return

        self.tracking_state = "LOST"
        self.lost_time = datetime.now()
        self.is_in_frame = False
        # 🛡️ Don't reset offline_frame_count — it's needed for timeout tracking
        logger.debug(f"{self.person_id} marked as LOST (offline {self.offline_frame_count} frames)")

    def mark_active(self):
        """Person re-detected after being lost"""
        self.tracking_state = "ACTIVE"
        self.lost_time = None
        self.is_in_frame = True
        self.last_detection_time = datetime.now()
        self.offline_timer_start = None
        self.offline_frame_count = 0
        logger.info(f"{self.person_id} re-activated from LOST state")

    @property
    def time_since_lost(self):
        """Get seconds since marked as LOST"""
        if self.lost_time:
            return (datetime.now() - self.lost_time).total_seconds()
        return 0

    def mark_exited(self):
        """Mark person as having exited (after 20-second timeout)"""
        # Prevent multiple calls — mark_exited() can be called repeatedly by _check_timeouts every frame
        if getattr(self, '_exit_processed', False):
            return  # Already processed, ignore subsequent calls

        self.visit_number += 1
        self.is_in_frame = False
        self._actual_exit = True  # Flag to indicate real timeout (not brief flicker)
        self._exit_processed = True  # Permanent flag — prevents re-entry until mark_online()

        # Reset embeddings flag so next visit computes fresh embeddings
        if hasattr(self, '_embeddings_computed'):
            delattr(self, '_embeddings_computed')

        # DO NOT delete _entry_logged or _entry_frame — process_frame needs them
        # DO NOT delete _entry_logged_session_id — process_frame needs it for cleanup
        if hasattr(self, '_entry_logged_session_id'):
            self._entry_logged_session_id = None

        logger.debug(f"{self.person_id} marked as exited: visit_number now = {self.visit_number}")

    def __repr__(self):
        return f"{self.person_id} (in_frame={self.is_in_frame}, visits={self.visit_number})"

    @classmethod
    def _release_id(cls, id_num: int) -> None:
        """Return a temporary person's ID to the free pool for reuse."""
        with cls._id_lock:
            cls._free_temp_ids.add(id_num)

    def _discard_auto_id(self) -> None:
        """Return the auto-generated ID back to pool. Call BEFORE overwriting person_id."""
        try:
            id_num = int(self.person_id.split("_")[1])
            Person._release_id(id_num)
        except (ValueError, IndexError):
            pass


class PersonTracker:
    """Manages person detection and re-identification across frames"""

    def __init__(self, embedding_store, db_manager):
        self.embedding_store = embedding_store
        self.db_manager = db_manager

        # Active people in current frame
        self.people = {}  # person_id -> Person
        self.bytetrack_cache = {}  # track_id -> person_id (for fast intra-frame tracking)
        self.rejected_track_ids = set()  # track_ids that failed confidence checks (hands, noise, etc.)
        self.people_lock = threading.Lock()  # Protect access to people dict from race conditions

        # Statistics
        self.frame_count = 0
        self.total_people_detected = 0

        logger.info("PersonTracker initialized")

    def update(self, frame, detections, embeddings_manager, frame_count=None, embedding_lock=None):
        """
        Update person tracking with new detections

        Args:
            frame: Current video frame
            detections: List of detection dicts with keys: track_id, bbox, confidence
            embeddings_manager: Manager for computing embeddings
            frame_count: Current frame number (for tracking temporary person creation)
            embedding_lock: Lock for thread-safe embedding access (prevent races with embedding thread)
        """
        self.frame_count += 1
        current_detections = {}  # track_id -> person_id mapping
        claimed_identities = set()  # Track which known identities are claimed this frame (prevent duplicates)
        used_track_ids = set()  # 🛡️ Track track_ids used this frame (prevent duplicate assignments)

        # Phase 1: Process current detections
        # Sort by confidence to prioritize best matches (highest confidence first)
        detections_sorted = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)

        for detection in detections_sorted:
            track_id = detection.get("track_id")
            bbox = detection.get("bbox")
            conf = detection.get("confidence", 0.5)

            # 🛡️ FIX #3: Check if track_id already used this frame (detect duplicates)
            if track_id in used_track_ids:
                logger.warning(f"⚠️ Duplicate track_id {track_id} in same frame! Forcing embedding search.")
                # Fall through to embedding search below
            # Check if we have a cached ID from ByteTrack (fast path)
            elif track_id in self.bytetrack_cache:
                person_id = self.bytetrack_cache[track_id]
                with self.people_lock:
                    person = self.people.get(person_id)

                # 🛡️ FIX #2: Only trust cache if person is currently in_frame (continuously visible)
                # ByteTrack can reuse track_ids for different objects once person leaves frame
                if person and person.is_in_frame:
                    person.reset_detection()
                    person.confidence_history.append(conf)
                    person.mark_online()
                    current_detections[track_id] = person_id
                    used_track_ids.add(track_id)
                    logger.debug(f"✅ Cache hit: track_id {track_id} → {person_id}")
                    continue
                else:
                    # Cache invalid: person not in frame (went offline)
                    if person:
                        logger.info(f"❌ CACHE MISS: track_id {track_id} ({person_id} is offline) → creating temp person")
                    # Fall through to embedding search

            # No cache hit — create temp person for batch processing
            person = Person(tracker=self)  # Pass tracker to avoid ID conflicts
            person._is_temporary = True
            person._creation_frame = frame_count
            person.mark_online()
            logger.debug(f"Created temporary: {person.person_id} (no cache hit)")
            with self.people_lock:
                self.people[person.person_id] = person
            self.bytetrack_cache[track_id] = person.person_id
            current_detections[track_id] = person.person_id
            used_track_ids.add(track_id)

        # Phase 2: Check for people who left frame
        self._check_timeouts(current_detections)

        return current_detections

    def _match_person(self, face_embeddings, offline_persons, face_embedding_poses=None):
        """
        Match person using COSINE similarity on face embeddings.
        When face_embedding_poses is provided, only compares same-pose embeddings
        (FRONT vs FRONT, LEFT vs LEFT, RIGHT vs RIGHT).

        Args:
            face_embeddings:       List of face embedding arrays
            offline_persons:       Dict of pid -> Person objects to search
            face_embedding_poses:  Optional parallel pose list for face_embeddings

        Returns:
            person_id if matched, None if new person
        """
        logger.debug(f"🔍 _match_person() called - face samples: {len(face_embeddings)}")

        best_face_match = None
        best_face_sim = 0.0

        def cosine_sim(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        try:
            FACE_COSINE_THRESHOLD = 0.47

            for pid, person in offline_persons.items():
                best_face_sim_for_p = 0.0
                all_sims = []

                if face_embeddings and person.face_embeddings:
                    stored_poses = getattr(person, 'face_embedding_poses', [])

                    # Pose-group comparison:
                    #   FRONT / UP / DOWN  → match against each other
                    #   LEFT               → LEFT only
                    #   RIGHT              → RIGHT only
                    _POSE_GROUP = {
                        "FRONT": {"FRONT", "UP", "DOWN"},
                        "UP":    {"FRONT", "UP", "DOWN"},
                        "DOWN":  {"FRONT", "UP", "DOWN"},
                        "LEFT":  {"LEFT"},
                        "RIGHT": {"RIGHT"},
                    }
                    for q_idx, query_emb in enumerate(face_embeddings):
                        q_pose = face_embedding_poses[q_idx] if face_embedding_poses and q_idx < len(face_embedding_poses) else None
                        q_group = _POSE_GROUP.get(q_pose, None) if q_pose else None
                        for s_idx, stored_emb in enumerate(person.face_embeddings):
                            s_pose = stored_poses[s_idx] if s_idx < len(stored_poses) else None
                            if q_group and s_pose and s_pose not in q_group:
                                continue  # skip cross-group pairs
                            all_sims.append(cosine_sim(query_emb, stored_emb))

                    # Fallback: if no same-pose pairs found (old data without pose tags)
                    if not all_sims:
                        for query_emb in face_embeddings:
                            for stored_emb in person.face_embeddings:
                                all_sims.append(cosine_sim(query_emb, stored_emb))

                    if all_sims:
                        all_sims.sort(reverse=True)

                        # ✨ NEW: Require multiple embeddings (top-5 instead of top-3)
                        # This ensures we have enough data before matching
                        min_embeddings = 5
                        top_n = all_sims[:min_embeddings]

                        # Only match if we have at least min_embeddings comparisons
                        if len(top_n) >= min_embeddings:
                            best_face_sim_for_p = sum(top_n) / len(top_n)  # avg of top 5

                            # ✨ NEW: Also require that ALL top embeddings are decent (not just average)
                            min_individual_score = 0.40
                            all_top_decent = all(score >= min_individual_score for score in top_n)

                            if not all_top_decent:
                                best_face_sim_for_p = 0.0  # Reject: not all top matches are decent
                                logger.debug(f"    Rejected {pid}: not all top-5 embeddings above {min_individual_score}")
                        else:
                            # Not enough embeddings to reliably match
                            best_face_sim_for_p = 0.0
                            logger.debug(f"    Insufficient data for {pid}: only {len(top_n)} embeddings")
                    else:
                        best_face_sim_for_p = 0.0

                face_matched = best_face_sim_for_p >= FACE_COSINE_THRESHOLD

                if face_matched and best_face_sim_for_p > best_face_sim:
                    best_face_sim = best_face_sim_for_p
                    best_face_match = pid

                status_icon = "✅" if face_matched else "❌"
                status_text = "MATCH" if face_matched else "no match"
                best_str = f"{all_sims[0]*100:.1f}%" if all_sims else "n/a"
                logger.info(f"    {status_icon} IN-MEMORY: {status_text} with {pid} (top3_avg={best_face_sim_for_p*100:.1f}%, best={best_str})")

            if best_face_match and best_face_sim >= FACE_COSINE_THRESHOLD:
                return best_face_match
            else:
                return None

        except Exception as e:
            logger.error(f"Error in _match_person: {e}")
            return None

    def resolve_identity_conflict(self, candidate_person_id, candidate_embeddings,
                                   incumbent_person_id, incumbent_embeddings):
        """
        ✨ NEW: Resolve conflicts when same person_id is assigned to multiple detections in same frame.

        When two different people match to same person_id, re-verify embeddings in DB
        to determine which one is actually that person.

        Args:
            candidate_person_id: New person trying to claim this ID
            candidate_embeddings: Embeddings of candidate person
            incumbent_person_id: Person already assigned this ID in frame
            incumbent_embeddings: Embeddings of incumbent person

        Returns:
            Tuple: (person_id_for_candidate, person_id_for_incumbent)
                   One keeps the original ID, other becomes new person
        """
        logger.info(f"⚠️  IDENTITY CONFLICT: Both {candidate_person_id} and {incumbent_person_id} match to same person!")
        logger.info(f"   Re-verifying embeddings in DB to find true identity...")

        # Get stored embeddings from DB for this person
        try:
            with self.people_lock:
                target_person = self.people.get(incumbent_person_id)
                if not target_person:
                    logger.warning(f"   Target person {incumbent_person_id} not found - assigning new ID")
                    return candidate_person_id, None

            stored_embeddings = target_person.face_embeddings
            if not stored_embeddings:
                logger.warning(f"   No stored embeddings for {incumbent_person_id} - both are new")
                return candidate_person_id, None

            # Compute similarity of BOTH detections against stored embeddings
            def avg_cosine_sim(query_embeds, stored_embeds):
                """Average cosine similarity of query against all stored embeddings."""
                if not query_embeds or not stored_embeds:
                    return 0.0

                similarities = []
                for q_emb in query_embeds:
                    na = np.linalg.norm(q_emb)
                    if na == 0:
                        continue
                    for s_emb in stored_embeds:
                        nb = np.linalg.norm(s_emb)
                        if nb > 0:
                            sim = float(np.dot(q_emb, s_emb) / (na * nb))
                            similarities.append(sim)

                return np.mean(similarities) if similarities else 0.0

            candidate_sim = avg_cosine_sim(candidate_embeddings, stored_embeddings)
            incumbent_sim = avg_cosine_sim(incumbent_embeddings, stored_embeddings)

            logger.info(f"   📊 Similarity vs stored embeddings:")
            logger.info(f"      {candidate_person_id}: {candidate_sim*100:.1f}%")
            logger.info(f"      {incumbent_person_id}: {incumbent_sim*100:.1f}%")

            # Assign ID to person with BETTER match
            if candidate_sim > incumbent_sim:
                logger.info(f"   ✅ {candidate_person_id} is TRUE match (higher confidence) → keep it")
                logger.info(f"   🆕 {incumbent_person_id} is FALSE positive → assign new ID")
                return candidate_person_id, "NEW"  # candidate keeps ID, incumbent gets new ID
            else:
                logger.info(f"   ✅ {incumbent_person_id} is TRUE match (higher confidence) → keep it")
                logger.info(f"   🆕 {candidate_person_id} is FALSE positive → assign new ID")
                return "NEW", incumbent_person_id  # incumbent keeps ID, candidate gets new ID

        except Exception as e:
            logger.error(f"   Error in conflict resolution: {e}")
            # Fallback: keep incumbent, make candidate new
            return "NEW", incumbent_person_id

    def _check_timeouts(self, current_detections):
        """Check for people who haven't been detected and manage LOST/EXITED states"""
        current_track_ids = set(current_detections.values())

        # Take snapshot of people under lock to prevent race conditions during iteration
        with self.people_lock:
            people_snapshot = list(self.people.items())

        for person_id, person in people_snapshot:
            with self.people_lock:
                still_exists = person_id in self.people

            if not still_exists:
                continue  # Already removed from tracker (e.g. merged temp) — skip

            # Person detected this frame — mark ACTIVE if was LOST
            if person_id in current_track_ids:
                person.offline_frame_count = 0  # Reset flicker counter
                if person.tracking_state == "LOST":
                    time_lost = person.time_since_lost
                    person.mark_active()
                    logger.info(f"🟢 {person_id} REDETECTED in frame (was LOST for {time_lost:.1f}s) → back to ACTIVE")
                elif person.tracking_state == "ACTIVE":
                    logger.debug(f"👁️  {person_id} confirmed in frame - offline counter reset")
                continue

            # Person not detected this frame

            # ═══════════════════════════════════════════════════════
            # TEMPORARY PERSON HANDLING (unchanged)
            # ═══════════════════════════════════════════════════════
            if getattr(person, '_is_temporary', False):
                # Temp person handling - don't mark LOST, just recycle after timeout
                if getattr(person, '_accumulating', False) or \
                   getattr(person, '_confirming', False):
                    logger.debug(f"Skipping recycle of {person_id} — in progress")
                    continue

                if person.time_since_last_detection > TEMP_PERSON_RECYCLE_TIMEOUT:
                    try:
                        id_num = int(person_id.split("_")[1])
                        Person._release_id(id_num)
                    except (ValueError, IndexError):
                        pass
                    with self.people_lock:
                        self.people.pop(person_id, None)
                continue

            # ═══════════════════════════════════════════════════════
            # CONFIRMED PERSON HANDLING
            # ═══════════════════════════════════════════════════════
            if person.tracking_state == "ACTIVE":
                # Still active — use 12-frame buffer before marking LOST
                person.offline_frame_count += 1

                # 🛡️ Mark as offline immediately (not in current frame)
                # This allows offline_persons pool to find them by face/body
                if person.is_in_frame:  # First time going offline
                    logger.info(f"⏱️ {person_id} is NOT in frame - offline counter started (0/12 frames)")
                person.is_in_frame = False

                if person.offline_frame_count >= 12:
                    # YOLO lost track — mark as LOST (not exited yet)
                    person.mark_lost()
                    logger.info(f"🔴 {person_id} moved to offline people — state: LOST | will move to EXITED in {PERSON_TIMEOUT_SECONDS}s if not reidentified")
                elif person.offline_frame_count % 4 == 0:
                    # Log progress toward LOST state
                    frames_remaining = 12 - person.offline_frame_count
                    logger.debug(f"⏱️ {person_id} offline {person.offline_frame_count}/12 frames ({frames_remaining} more → LOST)")

            elif person.tracking_state == "LOST":
                # Flag 30s DB exit — VideoProcessor picks this up and logs EXITED to DB
                if (person.time_since_lost >= NEW_VISIT_THRESHOLD_SECONDS
                        and not getattr(person, '_30s_exit_pending', False)
                        and not getattr(person, '_30s_exit_logged', False)):
                    person._30s_exit_pending = True

                # Already lost — check if truly timed out
                if person.time_since_lost > PERSON_TIMEOUT_SECONDS:
                    # 🛡️ Change state FIRST — prevents re-entry on next frame
                    person.tracking_state = "EXITED"

                    person.mark_exited()
                    logger.info(f"🔴 {person_id} (visit {person.visit_number}) — not reidentified in {person.time_since_lost:.1f}s → tracker state: LOST → EXITED")
                    logger.info(f"⏳ {person_id} will be deleted in 60s if not redetected")

                    # 🛡️ CRITICAL: DON'T delete immediately!
                    # Keep embeddings for 60 more seconds so if person returns,
                    # _match_person can find them via in-memory embedding search (fast)
                    # instead of slow Qdrant search
                    person._delete_after = datetime.now().timestamp() + EXITED_PERSON_MEMORY_SECONDS

                    # Only clean cache (don't need ByteTrack cache for exited people)
                    for track_id in list(self.bytetrack_cache.keys()):
                        if self.bytetrack_cache[track_id] == person_id:
                            del self.bytetrack_cache[track_id]
                            logger.debug(f"♻️ Cleared cache: track {track_id} → {person_id}")

        # ═══════════════════════════════════════════════════════
        # DELAYED CLEANUP — Delete persons marked for deletion
        # ═══════════════════════════════════════════════════════
        for person_id, person in people_snapshot:
            if hasattr(person, '_delete_after'):
                if datetime.now().timestamp() > person._delete_after:
                    # 🛡️ ID already released in _cleanup_temp_after_match() — don't release again
                    with self.people_lock:
                        self.people.pop(person_id, None)
                    logger.debug(f"🗑️ Deleted {person_id} from tracker (grace period expired)")

    def get_person(self, person_id):
        """Get person object by ID"""
        with self.people_lock:
            return self.people.get(person_id)

    def get_all_people(self):
        """Get all tracked people"""
        with self.people_lock:
            try:
                return list(self.people.values())
            except (RuntimeError, KeyError):
                # Handle race condition if dict is modified during iteration
                return []

    def get_people_in_frame(self):
        """Get people currently in frame"""
        with self.people_lock:
            try:
                return [p for p in list(self.people.values()) if p.is_in_frame]
            except (RuntimeError, KeyError):
                # Handle race condition if dict is modified during iteration
                return []

    def clear_cache(self):
        """Clear ByteTrack cache (useful between video segments)"""
        self.bytetrack_cache.clear()

    def merge_temporary_person(self, temp_person_id, matched_person_id):
        """Replace temporary person ID with matched person ID in cache and tracking.

        Called when Frame 3 embeddings reveal a temporary person is actually a returning visitor.
        Transfers any embeddings collected during temp period to matched person.
        """
        # Update all cache entries pointing to temp_person_id
        for track_id, person_id in list(self.bytetrack_cache.items()):
            if person_id == temp_person_id:
                self.bytetrack_cache[track_id] = matched_person_id
                logger.debug(f"Updated bytetrack_cache: track {track_id} → {matched_person_id} (was {temp_person_id})")

        # Transfer embeddings from temp person to matched person, then remove temp person
        with self.people_lock:
            temp_person = self.people.get(temp_person_id)
            matched_person = self.people.get(matched_person_id)

            # Transfer embeddings if both exist
            if temp_person and matched_person:
                matched_person.face_embeddings.extend(temp_person.face_embeddings)

            # Remove temp person from tracking
            if temp_person_id in self.people:
                del self.people[temp_person_id]
                logger.debug(f"Merged {temp_person_id} → {matched_person_id}")
                # 🛡️ ID release is now handled in video_processor._cleanup_temp_after_match()
                # Don't release here to avoid duplicate release

    def find_person_by_stored_embeddings(self, query_embedding, embeddings_manager):
        """
        Search for person match using stored face embeddings.

        Args:
            query_embedding: Current face embedding to match
            embeddings_manager: Manager with L2 distance computation

        Returns:
            Tuple of (person_id, distance, confidence) or (None, None, 0.0) if no match
        """
        best_person_id = None
        best_distance = float('inf')
        best_confidence = 0.0

        with self.people_lock:
            for person_id, person in list(self.people.items()):
                if person.face_embeddings:
                    distance, confidence = embeddings_manager.find_best_embedding_match(
                        query_embedding, person.face_embeddings, threshold=FACE_EMBEDDING_L2_THRESHOLD
                    )

                    if distance is not None and distance < best_distance:
                        best_distance = distance
                        best_person_id = person_id
                        best_confidence = confidence

        return best_person_id, best_distance, best_confidence

    def get_statistics(self):
        """Get tracking statistics"""
        try:
            with self.people_lock:
                # Don't call get_people_in_frame() - it tries to acquire people_lock again (deadlock)
                # Inline the logic instead
                in_frame_count = len([p for p in self.people.values() if p.is_in_frame])
                return {
                    "frames_processed": self.frame_count,
                    "total_people_detected": self.total_people_detected,
                    "current_people_in_frame": in_frame_count,
                    "total_tracked_people": len(self.people),
                }
        except (RuntimeError, KeyError):
            # Handle race condition if dict is modified during call
            return {
                "frames_processed": self.frame_count,
                "total_people_detected": self.total_people_detected,
                "current_people_in_frame": 0,
                "total_tracked_people": 0,
            }

    def __repr__(self):
        return f"PersonTracker(frame={self.frame_count}, people={len(self.people)})"
