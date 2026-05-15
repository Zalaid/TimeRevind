"""
STEP 0 TEST: Verify BehaviorLogger works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .logger import BehaviorLogger


def test_logger():
    print("\n" + "="*60)
    print("STEP 0: Testing BehaviorLogger")
    print("="*60)

    # Initialize logger
    logger = BehaviorLogger()
    print("[OK] Logger initialized")

    # Clear previous logs
    logger.clear()
    print("[OK] Logs cleared")

    # Log some events
    logger.log_event('Person_1', 'SAFETY', 'FALLING', confidence=0.95)
    logger.log_event('Person_1', 'EMOTION', 'HAPPY', confidence=0.82)
    logger.log_event('Person_2', 'ACTIVITY', 'WALKING', duration=5.2)
    logger.log_event('Person_1', 'POSTURE', 'SITTING', confidence=0.88, details={"reason": "test"})
    print("[OK] 4 events logged")

    # Read back all logs
    logs = logger.get_logs()
    print("[OK] Retrieved {} events from JSON".format(len(logs)))

    # Verify structure
    assert len(logs) == 4, f"Expected 4 logs, got {len(logs)}"
    assert logs[0]['event_type'] == 'FALLING', "First should be FALLING"
    assert logs[1]['event_type'] == 'HAPPY', "Second should be HAPPY"
    assert logs[2]['event_type'] == 'WALKING', "Third should be WALKING"
    assert logs[3]['event_type'] == 'SITTING', "Fourth should be SITTING"
    print("[OK] All events logged correctly")

    # Test person-specific queries
    person1_logs = logger.get_person_logs('Person_1')
    assert len(person1_logs) == 3, f"Expected 3 logs for Person_1, got {len(person1_logs)}"
    print("[OK] Person_1 has {} events".format(len(person1_logs)))

    person2_logs = logger.get_person_logs('Person_2')
    assert len(person2_logs) == 1, f"Expected 1 log for Person_2, got {len(person2_logs)}"
    print("[OK] Person_2 has {} event".format(len(person2_logs)))

    # Test category-specific queries
    safety_logs = logger.get_category_logs('SAFETY')
    assert len(safety_logs) == 1, f"Expected 1 SAFETY log, got {len(safety_logs)}"
    print("[OK] Found {} SAFETY event".format(len(safety_logs)))

    # Verify log file exists
    log_path = Path("behavior/logs/behavior_events.json")
    assert log_path.exists(), "Log file should exist"
    print("[OK] Log file exists at {}".format(log_path))

    # Print sample log
    print("\nSample event:")
    print("  {}".format(logs[0]))

    print("\n" + "="*60)
    print("PASS: STEP 0 - BehaviorLogger working!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_logger()
