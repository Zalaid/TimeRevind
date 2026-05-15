"""
STEP 0: Simple JSON-based logger for behavior events
No database dependency during development
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class BehaviorLogger:
    """Simple JSON-based logger for behavior events (no database)"""

    def __init__(self, log_file="behavior/logs/behavior_events.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.write_text(json.dumps([], indent=2))
        logger.info(f"BehaviorLogger initialized: {self.log_file}")

    def log_event(self, person_id, event_category, event_type,
                  confidence=None, duration=None, details=None):
        """Log a behavior event to JSON file"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "person_id": person_id,
            "event_category": event_category,
            "event_type": event_type,
            "confidence": confidence,
            "duration_seconds": duration,
            "details": details
        }

        # Read existing logs
        events = json.loads(self.log_file.read_text())

        # Add new event
        events.append(event)

        # Write back
        self.log_file.write_text(json.dumps(events, indent=2))
        logger.debug(f"Logged: {person_id} - {event_category}:{event_type}")

    def get_logs(self):
        """Read all logged events"""
        return json.loads(self.log_file.read_text())

    def clear(self):
        """Clear all logs (for testing)"""
        self.log_file.write_text(json.dumps([], indent=2))
        logger.info("Behavior logs cleared")

    def get_person_logs(self, person_id):
        """Get logs for specific person"""
        return [e for e in self.get_logs() if e['person_id'] == person_id]

    def get_category_logs(self, category):
        """Get logs for specific category"""
        return [e for e in self.get_logs() if e['event_category'] == category]
