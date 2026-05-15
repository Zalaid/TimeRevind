"""
Pushover notification sender for TimeRevind alerts
Sends push notifications to user's phone via Pushover API
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PUSHOVER_TOKEN = os.getenv("PUSHOVER_API_KEY")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_notification(person_id: str, simulated_time: str, entry_type: str = "ENTRY") -> bool:
    """
    Send Pushover push notification.

    Args:
        person_id: Name of the person (e.g., "shiza")
        simulated_time: Time string (e.g., "16:05:23")
        entry_type: "ENTRY" or "RE_ENTRY"

    Returns:
        True if sent successfully, False otherwise
    """
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        logger.warning("Pushover credentials not configured in .env")
        return False

    label = "entered" if entry_type == "ENTRY" else "re-entered"
    message = f"⚠️ Alert: {person_id} {label} at {simulated_time}"

    try:
        response = requests.post(
            PUSHOVER_URL,
            data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": message,
                "title": "TimeRevind Alert",
                "sound": "pushover",
                "priority": 1
            },
            timeout=10
        )

        if response.status_code == 200:
            logger.info(f"✓ Pushover notification sent: {person_id} @ {simulated_time}")
            return True
        else:
            logger.error(f"✗ Pushover API error ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to send notification: {e}")
        return False
