"""
Behavior Database Logger
Logs behavior detection events to SQLite database instead of JSON
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

TS_FMT = "%d/%m/%Y %I:%M:%S %p"  # e.g. 22/03/2026 02:30:45 PM


class BehaviorDatabaseLogger:
    """Logs behavior events (posture, activity) to SQLite database"""

    def __init__(self, event_logger):
        """
        Args:
            event_logger: EventLogger instance from src.core.event_logger
        """
        self.event_logger = event_logger

    def log_behavior(self, person_id, behavior, visit_num=None, video_file=None, video_timestamp=None):
        """
        Log a behavior detection to database

        Args:
            person_id: ID of the person
            behavior: Dict with keys:
                - category: "POSTURE" or "ACTIVITY"
                - type: "STANDING", "SITTING", "WALKING", etc.
                - confidence: float 0-1
                - duration: seconds (only for SITTING)
                - details: dict with extra info (pace, velocity, etc.)
            visit_num: Visit number (optional)
            video_file: Video file path (optional)
            video_timestamp: Timestamp in video (optional)
        """
        try:
            timestamp = datetime.now().strftime(TS_FMT)
            category = behavior.get("category", "UNKNOWN")
            behavior_type = behavior.get("type", "UNKNOWN")
            confidence = behavior.get("confidence", 0.5)
            duration_seconds = behavior.get("duration")
            details = behavior.get("details", {})

            # Log as POSE event if category is POSTURE
            if category == "POSTURE":
                self.event_logger.log_pose_event(
                    person_id=person_id,
                    visit_num=visit_num,
                    pose=behavior_type,
                    duration_seconds=duration_seconds,
                    timestamp=timestamp,
                    video_file=video_file,
                    video_timestamp=video_timestamp
                )
                logger.info(f"Behavior logged: {person_id} - {behavior_type} ({confidence*100:.0f}%)")

            # Log as INTERACTION event if category is ACTIVITY
            elif category == "ACTIVITY":
                action_name = behavior_type
                if details.get("pace"):
                    action_name = f"{behavior_type}_{details['pace'].upper()}"

                self.event_logger.log_interaction_event(
                    person_id=person_id,
                    visit_num=visit_num,
                    action=action_name,
                    duration_seconds=duration_seconds,
                    timestamp=timestamp,
                    video_file=video_file,
                    video_timestamp=video_timestamp,
                    confidence=confidence
                )
                logger.info(f"Behavior logged: {person_id} - {action_name} ({confidence*100:.0f}%)")

        except Exception as e:
            logger.error(f"Error logging behavior: {e}")
            raise

    def log_sitting_duration(self, person_id, duration_seconds, visit_num=None,
                            video_file=None, video_timestamp=None):
        """
        Log sitting duration milestone

        Args:
            person_id: ID of the person
            duration_seconds: How long they were sitting
            visit_num: Visit number (optional)
            video_file: Video file path (optional)
            video_timestamp: Timestamp in video (optional)
        """
        try:
            timestamp = datetime.now().strftime(TS_FMT)

            self.event_logger.log_duration_milestone(
                person_id=person_id,
                visit_num=visit_num,
                action=f"SITTING_FOR_{int(duration_seconds)}_SECONDS",
                duration_seconds=duration_seconds,
                timestamp=timestamp,
                video_file=video_file,
                video_timestamp=video_timestamp
            )
            logger.info(f"Sitting duration logged: {person_id} - {duration_seconds:.1f}s")

        except Exception as e:
            logger.error(f"Error logging sitting duration: {e}")
            raise

    def get_person_behavior_summary(self, person_id, visit_num=None):
        """Get behavior summary for a person"""
        return self.event_logger.get_person_activities(person_id, visit_num)

    def get_behavior_statistics(self):
        """Get overall behavior statistics"""
        return self.event_logger.get_statistics()


# Global instance (lazy initialization)
_behavior_db_logger = None


def get_behavior_db_logger(event_logger=None):
    """
    Get or create global behavior database logger (thread-safe)

    Args:
        event_logger: EventLogger instance. Required on first call.

    Returns:
        BehaviorDatabaseLogger instance
    """
    global _behavior_db_logger

    if _behavior_db_logger is None:
        if event_logger is None:
            from src.core.event_logger import get_event_logger
            event_logger = get_event_logger()

        _behavior_db_logger = BehaviorDatabaseLogger(event_logger)
        logger.info("Behavior database logger initialized")

    return _behavior_db_logger


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Behavior database logger module ready")
