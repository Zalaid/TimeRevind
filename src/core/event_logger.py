"""
Event Logger Module
Logs entry/exit events and detailed activity events
"""

import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

TS_FMT = "%d/%m/%Y %I:%M:%S %p"  # e.g. 22/03/2026 02:30:45 PM


class EventLogger:
    """Handles logging of all events (entry, exit, activities)"""

    def __init__(self, db_manager, session_id=None):
        self.db = db_manager
        self.session_id = session_id or "default"
        self.person_entry_times = {}  # person_id -> entry_timestamp
        self.activity_states = {}  # person_id -> current activity state

    def log_entry(self, person_id, visit_num, video_file=None, video_timestamp=None):
        """Log person entry event"""
        timestamp = datetime.now().strftime(TS_FMT)
        self.person_entry_times[person_id] = timestamp

        sql = """
            INSERT INTO events
            (timestamp, person_id, event_type, visit_num, video_file, video_timestamp_start, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (timestamp, person_id, "ENTERED", visit_num, video_file, video_timestamp, self.session_id)

        self.db.execute(sql, params)
        logger.info(f"Entry logged: {person_id} (visit {visit_num})")

        return timestamp

    def log_exit(self, person_id, visit_num, video_file=None, video_timestamp=None, timestamp=None):
        """Log person exit event"""
        timestamp = timestamp if timestamp else datetime.now().strftime(TS_FMT)

        # Update events table
        sql = """
            INSERT INTO events
            (timestamp, person_id, event_type, visit_num, video_file, video_timestamp_start, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (timestamp, person_id, "EXITED", visit_num, video_file, video_timestamp, self.session_id)
        self.db.execute(sql, params)

        # Update person_profiles
        self._update_person_profile(person_id, timestamp)

        # Clear state
        if person_id in self.person_entry_times:
            del self.person_entry_times[person_id]

        logger.debug(f"Exit logged: {person_id} (visit {visit_num})")

        return timestamp

    def log_location_event(
        self,
        person_id,
        visit_num,
        location,
        timestamp=None,
        video_file=None,
        video_timestamp=None,
    ):
        """Log location change event"""
        if timestamp is None:
            timestamp = datetime.now().strftime(TS_FMT)

        sql = """
            INSERT INTO activity_events
            (timestamp, person_id, event_type, action, location, visit_num,
             video_file, video_timestamp_start)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            timestamp,
            person_id,
            "LOCATION",
            f"AT_{location.upper()}",
            location,
            visit_num,
            video_file,
            video_timestamp,
        )

        self.db.execute(sql, params)

    def log_pose_event(
        self,
        person_id,
        visit_num,
        pose,
        duration_seconds=None,
        timestamp=None,
        video_file=None,
        video_timestamp=None,
    ):
        """Log pose change event (SITTING, STANDING, LYING_DOWN)"""
        if timestamp is None:
            timestamp = datetime.now().strftime(TS_FMT)

        sql = """
            INSERT INTO activity_events
            (timestamp, person_id, event_type, action, duration_seconds, visit_num,
             video_file, video_timestamp_start, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            timestamp,
            person_id,
            "POSE",
            pose.upper(),
            duration_seconds,
            visit_num,
            video_file,
            video_timestamp,
            0.95,  # Default pose confidence
        )

        self.db.execute(sql, params)

    def log_interaction_event(
        self,
        person_id,
        visit_num,
        action,
        object_id=None,
        location=None,
        duration_seconds=None,
        timestamp=None,
        video_file=None,
        video_timestamp=None,
        confidence=0.9,
    ):
        """Log interaction event (USING_PHONE, USING_LAPTOP, PICKING_UP_OBJECT, etc)"""
        if timestamp is None:
            timestamp = datetime.now().strftime(TS_FMT)

        sql = """
            INSERT INTO activity_events
            (timestamp, person_id, event_type, action, object_id, location,
             duration_seconds, visit_num, video_file, video_timestamp_start, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            timestamp,
            person_id,
            "INTERACTION",
            action.upper(),
            object_id,
            location,
            duration_seconds,
            visit_num,
            video_file,
            video_timestamp,
            confidence,
        )

        self.db.execute(sql, params)
        logger.debug(f"Interaction logged: {person_id} - {action}")

    def log_duration_milestone(
        self,
        person_id,
        visit_num,
        action,
        duration_seconds,
        timestamp=None,
        video_file=None,
        video_timestamp=None,
    ):
        """Log duration milestone events (SITTING_FOR_5_MIN, etc)"""
        if timestamp is None:
            timestamp = datetime.now().strftime(TS_FMT)

        sql = """
            INSERT INTO activity_events
            (timestamp, person_id, event_type, action, duration_seconds, visit_num,
             video_file, video_timestamp_start)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            timestamp,
            person_id,
            "DURATION",
            action.upper(),
            duration_seconds,
            visit_num,
            video_file,
            video_timestamp,
        )

        self.db.execute(sql, params)
        logger.debug(f"Duration milestone logged: {person_id} - {action}")

    def get_person_activities(self, person_id, visit_num=None, time_range_seconds=3600):
        """Get all activities for a person"""
        if visit_num:
            sql = """
                SELECT * FROM activity_events
                WHERE person_id = ? AND visit_num = ?
                ORDER BY timestamp ASC
            """
            params = (person_id, visit_num)
        else:
            sql = """
                SELECT * FROM activity_events
                WHERE person_id = ? AND timestamp >= datetime('now', '-' || ? || ' seconds')
                ORDER BY timestamp ASC
            """
            params = (person_id, time_range_seconds)

        results = self.db.query(sql, params)
        return results

    def get_activities_summary(self, person_id, time_range_seconds=3600):
        """Get summary of activities for Groq processing"""
        sql = """
            SELECT
                action,
                COUNT(*) as count,
                SUM(duration_seconds) as total_seconds,
                AVG(duration_seconds) as avg_seconds,
                location,
                object_id
            FROM activity_events
            WHERE person_id = ? AND timestamp >= datetime('now', '-' || ? || ' seconds')
            GROUP BY action, location, object_id
            ORDER BY total_seconds DESC
        """
        params = (person_id, time_range_seconds)

        results = self.db.query(sql, params)
        return results

    def _update_person_profile(self, person_id, exit_timestamp):
        """Update person profile with visit info"""
        try:
            # Get entry time
            entry_sql = """
                SELECT timestamp FROM events
                WHERE person_id = ? AND event_type = 'ENTERED'
                ORDER BY timestamp DESC LIMIT 1
            """
            entry_result = self.db.query(entry_sql, (person_id,))

            if entry_result:
                # Calculate duration
                entry_time = datetime.strptime(entry_result[0][0], TS_FMT)
                exit_time = datetime.strptime(exit_timestamp, TS_FMT)
                duration = int((exit_time - entry_time).total_seconds())

                # Update profile
                update_sql = """
                    UPDATE person_profiles
                    SET last_seen = ?, total_time_seconds = total_time_seconds + ?
                    WHERE person_id = ?
                """
                self.db.execute(update_sql, (exit_timestamp, duration, person_id))

        except Exception as e:
            logger.warning(f"Error updating person profile: {e}")

    def get_statistics(self):
        """Get event logging statistics"""
        entry_count = self.db.query("SELECT COUNT(*) FROM events WHERE event_type = 'ENTERED'")[0][0]
        exit_count = self.db.query("SELECT COUNT(*) FROM events WHERE event_type = 'EXITED'")[0][0]
        activity_count = self.db.query("SELECT COUNT(*) FROM activity_events")[0][0]

        return {
            "total_entries": entry_count,
            "total_exits": exit_count,
            "total_activities_logged": activity_count,
        }


# Global instance (thread-safe singleton)
event_logger = None
_event_logger_lock = threading.Lock()


def get_event_logger():
    """Get or create global event logger (thread-safe)"""
    global event_logger
    with _event_logger_lock:
        if event_logger is None:
            from src.database.db_init import db_manager

            event_logger = EventLogger(db_manager)
    return event_logger


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Event logger module ready")
