"""
STEP 2 TEST: Verify BehaviorDetector skeleton works
Tests initialization and basic detect_all() functionality
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .behavior_detector import BehaviorDetector


def test_detector_init():
    """Test BehaviorDetector initialization"""
    print("\n" + "="*60)
    print("STEP 2: Testing BehaviorDetector Skeleton")
    print("="*60)

    try:
        detector = BehaviorDetector()
        print("[OK] BehaviorDetector initialized")
        return detector
    except Exception as e:
        print("[FAIL] BehaviorDetector initialization failed: {}".format(e))
        raise


def test_detect_all(detector):
    """Test detect_all() with fake data"""
    print("\n--- Testing detect_all() with fake keypoints ---")

    # Create fake keypoints (17, 3) - zeros with some confidence
    fake_kps = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        fake_kps[i, 2] = 0.5  # Set confidence to 0.5

    # Create fake person crop (100x50 BGR image)
    fake_person_crop = np.zeros((100, 50, 3), dtype=np.uint8)

    # Create fake face crop (48x48 BGR image)
    fake_face_crop = np.zeros((48, 48, 3), dtype=np.uint8)

    try:
        # Call detect_all with fake data
        result = detector.detect_all(
            'Person_1',
            fake_kps,
            fake_person_crop,
            fake_face_crop
        )

        # Verify result is a list
        assert isinstance(result, list), "detect_all should return a list"
        print("[OK] detect_all() returns list: {}".format(result))

        return result
    except Exception as e:
        print("[FAIL] detect_all() test failed: {}".format(e))
        raise


def test_data_structures(detector):
    """Test that internal data structures are properly initialized"""
    print("\n--- Testing internal data structures ---")

    # Test that windows are initialized
    assert isinstance(detector.windows, dict), "windows should be dict-like"
    print("[OK] windows structure initialized")

    # Test that pose_history is initialized
    assert isinstance(detector.pose_history, dict), "pose_history should be dict-like"
    print("[OK] pose_history structure initialized")

    # Test that last_voted is initialized
    assert isinstance(detector.last_voted, dict), "last_voted should be dict-like"
    print("[OK] last_voted structure initialized")

    # Test that behavior_state is initialized
    assert isinstance(detector.behavior_state, dict), "behavior_state should be dict-like"
    print("[OK] behavior_state structure initialized")

    # Test that last_logged is initialized
    assert isinstance(detector.last_logged, dict), "last_logged should be dict-like"
    print("[OK] last_logged structure initialized")

    # Test that fall_state is initialized
    assert isinstance(detector.fall_state, dict), "fall_state should be dict-like"
    print("[OK] fall_state structure initialized")


def test_cleanup(detector):
    """Test cleanup() method"""
    print("\n--- Testing cleanup() ---")

    detector.detect_all(
        'Person_1',
        np.zeros((17, 3), dtype=np.float32),
        np.zeros((100, 50, 3), dtype=np.uint8)
    )

    # Verify person was added to tracking
    assert 'Person_1' in detector.pose_history, "Person should be in pose_history after detect_all"
    print("[OK] Person_1 added to tracking")

    # Cleanup
    detector.cleanup('Person_1')

    # Verify person was removed
    assert 'Person_1' not in detector.pose_history, "Person should be removed from pose_history after cleanup"
    assert 'Person_1' not in detector.windows, "Person should be removed from windows after cleanup"
    print("[OK] Person_1 cleaned up successfully")


def test_multiple_people(detector):
    """Test tracking multiple people"""
    print("\n--- Testing multiple people ---")

    fake_kps = np.zeros((17, 3), dtype=np.float32)
    fake_crop = np.zeros((100, 50, 3), dtype=np.uint8)

    # Detect for 3 different people
    for person_id in ['Person_1', 'Person_2', 'Person_3']:
        result = detector.detect_all(person_id, fake_kps, fake_crop)
        assert isinstance(result, list), "detect_all should return list for {}".format(person_id)

    print("[OK] All 3 people tracked separately")

    # Verify they're all in the tracking dict
    for person_id in ['Person_1', 'Person_2', 'Person_3']:
        assert person_id in detector.pose_history, "{} should be in pose_history".format(person_id)

    print("[OK] All people in pose_history")

    # Cleanup all
    for person_id in ['Person_1', 'Person_2', 'Person_3']:
        detector.cleanup(person_id)

    print("[OK] All people cleaned up")


def test_placeholder_methods(detector):
    """Test that placeholder methods exist and don't crash"""
    print("\n--- Testing placeholder methods ---")

    fake_kps = np.zeros((17, 3), dtype=np.float32)
    fake_crop = np.zeros((100, 50, 3), dtype=np.uint8)

    # These should return None or dict, not crash
    assert detector._detect_posture(fake_kps) is None or isinstance(detector._detect_posture(fake_kps), (dict, type(None)))
    print("[OK] _detect_posture() callable")

    assert detector._detect_fall('Person_1', fake_kps) is None or isinstance(detector._detect_fall('Person_1', fake_kps), (dict, type(None)))
    print("[OK] _detect_fall() callable")

    assert detector._detect_walking('Person_1', fake_kps) is None or isinstance(detector._detect_walking('Person_1', fake_kps), (dict, type(None)))
    print("[OK] _detect_walking() callable")

    assert detector._detect_gestures('Person_1', fake_kps, fake_crop) is None or isinstance(detector._detect_gestures('Person_1', fake_kps, fake_crop), (dict, type(None)))
    print("[OK] _detect_gestures() callable")

    assert detector._detect_emotion('Person_1', fake_crop) is None or isinstance(detector._detect_emotion('Person_1', fake_crop), (dict, type(None)))
    print("[OK] _detect_emotion() callable")

    assert detector._detect_drowsiness(fake_kps, fake_crop) is None or isinstance(detector._detect_drowsiness(fake_kps, fake_crop), (dict, type(None)))
    print("[OK] _detect_drowsiness() callable")

    assert detector._detect_eating_drinking(fake_kps) is None or isinstance(detector._detect_eating_drinking(fake_kps), (dict, type(None)))
    print("[OK] _detect_eating_drinking() callable")


def main():
    print("\n" + "="*60)
    print("STEP 2: BehaviorDetector Skeleton Tests")
    print("="*60)

    # Test 1: Initialization
    detector = test_detector_init()

    # Test 2: detect_all() with fake data
    result = test_detect_all(detector)

    # Test 3: Data structures
    test_data_structures(detector)

    # Test 4: Cleanup
    test_cleanup(detector)

    # Test 5: Multiple people
    test_multiple_people(detector)

    # Test 6: Placeholder methods
    test_placeholder_methods(detector)

    print("\n" + "="*60)
    print("PASS: STEP 2 - BehaviorDetector skeleton working!")
    print("="*60)
    print("\nReady for STEP 3: Implement posture detection")
    print()


if __name__ == "__main__":
    main()
