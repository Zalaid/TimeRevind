"""
TimeRevind Configuration Module
Centralized settings for the entire system
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
VIDEO_DIR = PROJECT_ROOT / "videos"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_DIR = PROJECT_ROOT / "db"
BEST_FRAMES_DIR = PROJECT_ROOT / "best_one"  # Best face and body frames
EMBEDDING_FRAMES_DIR = PROJECT_ROOT / "embedding_frames"  # Frames used for embeddings
FACE_FRAMES_DIR = EMBEDDING_FRAMES_DIR / "face"

# Create directories if they don't exist
VIDEO_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
BEST_FRAMES_DIR.mkdir(exist_ok=True)
EMBEDDING_FRAMES_DIR.mkdir(exist_ok=True)
FACE_FRAMES_DIR.mkdir(exist_ok=True)

# Database paths
SQLITE_DB_PATH = DB_DIR / "timerevind.db"
# CHROMADB_PATH = DB_DIR / "chromadb"  # Deprecated - using Qdrant Cloud instead

# Video settings
VIDEO_CODEC = "H.264"
VIDEO_FPS = 30
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_RESOLUTION = (VIDEO_WIDTH, VIDEO_HEIGHT)  # 1080p (fallback: 720p → 480p)
RESOLUTION_FALLBACKS = [(1920, 1080), (1280, 720), (640, 480)]  # Try in order
VIDEO_BITRATE = "3000k"          # 3000 kbps for quality/size balance
VIDEO_SEGMENT_MINUTES = 60

# Detection settings
YOLO_MODEL = "yolov8n.pt"  # nano model for speed (person detection only)
INSIGHTFACE_MODEL = str(PROJECT_ROOT / "models" / "buffalo_l" / "w600k_r50.onnx")  # ArcFace ResNet50 via DirectML
SCRFD_MODEL = str(PROJECT_ROOT / "models" / "buffalo_l" / "det_10g_dynamic.onnx")   # SCRFD-10G face detector — dynamic batch
SCRFD_CONFIDENCE_THRESHOLD = 0.5      # Detection threshold for SCRFD (separate from ArcFace quality threshold)
YOLO_CONFIDENCE_THRESHOLD = 0.2       # Lowered from 0.65 → 0.45 → 0.35 → 0.2 (catches even more people, reduces dropouts)
YOLO_IOU_THRESHOLD = 0.80

# Person tracking settings
BYTETRACK_TRACK_THRESH = 0.6
BYTETRACK_TRACK_BUFFER = 30  # frames
PERSON_TIMEOUT_SECONDS = 60  # 60-second timeout in LOST state before marking EXITED
EXITED_PERSON_MEMORY_SECONDS = 1200  # 20 minutes in EXITED state before deleting from memory
NEW_VISIT_THRESHOLD_SECONDS = 30  # Absence >= 30s = new visit (EXITED + new ENTERED); < 30s = same visit (reactivate only)
TEMP_PERSON_RECYCLE_TIMEOUT = 30  # 30-second timeout for recycling temporary person IDs (prevents duplicate IDs for new people)

# Embedding settings
FACE_CONFIDENCE_THRESHOLD = 0.50  # SCRFD face detection confidence (50% = SCRFD standard scale)
FRAMES_TO_COLLECT = 5   # Collect 5 frames per batch (5 batches = 25 total)
FRAME_CAPTURE_STRIDE = 8  # Capture every 8th frame for diversity (lower = more frames, higher = more spread)

# Background accumulator settings
ACCUMULATOR_ENABLED = True           # Enable/disable background embedding collection
ACCUMULATOR_TARGET_EMBEDDINGS = 35   # Target: 7 FRONT + 7 LEFT + 7 RIGHT + 7 UP + 7 DOWN
ACCUMULATOR_MAX_DURATION = 300       # Max seconds to collect before giving up (5 minutes)
# At 30fps / stride 8 = 3.75 checks/sec → 300 checks ≈ 80 seconds of active attempts
ACCUMULATOR_MAX_FAILED = 300         # Max rejected quality checks before giving up
FACE_MATCH_THRESHOLD = 0.47  # Face embedding match confidence
# L2 distance threshold for face embedding matching (ArcFace standard = 1.24)
FACE_EMBEDDING_L2_THRESHOLD = 0.5   # cosine distance threshold for face embeddings (50% similarity minimum)

# Event detection settings
LOCATION_CHECK_INTERVAL = 1  # Check location every frame
POSE_CHECK_INTERVAL = 10  # Check pose every 10 frames (lighter)
INTERACTION_CHECK_INTERVAL = 1  # Check interaction every frame
DURATION_CHECK_INTERVAL = 100  # Log milestones every 100 frames

# Pose estimation settings (MediaPipe)
POSE_CONFIDENCE_THRESHOLD = 0.5
STANDING_THRESHOLD = 0.3  # Hip-to-knee ratio
SITTING_THRESHOLD = 0.6
LYING_THRESHOLD = 0.8

# Zone settings
ZONES = {
    "desk_area": [(100, 100), (400, 100), (400, 400), (100, 400)],  # Example zone
    "sofa_zone": [(500, 200), (800, 200), (800, 500), (500, 500)],
    "bed_zone": [(100, 500), (400, 500), (400, 800), (100, 800)],
}

# API settings
GROQ_API_TIMEOUT = 30  # seconds
GROQ_MAX_RETRIES = 3

# Voice settings
WAKE_WORD = "hey sentinel"
WHISPER_MODEL = "base"  # tiny, base, small, medium
TTS_PROVIDER = "gtts"  # gtts or elevenlabs

# Porcupine wake word settings
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_API_KEY", "")
PORCUPINE_KEYWORDS = ["jarvis"]  # Custom wake word

# logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance settings
ENABLE_GPU = True  # Use CUDA if available
MAX_BATCH_SIZE = 4  # Batch process up to 4 people
CACHE_SIZE_MB = 500  # ChromaDB cache size

# Activity milestones (in seconds) to log
ACTIVITY_MILESTONES = {
    "SITTING": [300, 900, 3600],  # 5min, 15min, 1hour
    "STANDING": [120, 600, 1800],  # 2min, 10min, 30min
    "USING_PHONE": [600, 1800, 3600],  # 10min, 30min, 1hour
    "USING_LAPTOP": [600, 1800, 3600],  # 10min, 30min, 1hour
    "IN_ZONE": [600, 1800, 3600],  # 10min, 30min, 1hour
}

# Video buffer settings
VIDEO_CONTEXT_BUFFER_SECONDS = 10  # Show 10 seconds before event
VIDEO_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for video extraction

# Debug settings
DEBUG = False
SAVE_DEBUG_FRAMES = False
DEBUG_FRAMES_DIR = LOGS_DIR / "debug_frames"

# Embedding storage settings
SAVE_NEW_EMBEDDINGS = False  # Only match against existing embeddings, don't save new ones

print(f"[CONFIG] TimeRevind configured")
print(f"  Database: {SQLITE_DB_PATH}")
print(f"  Video dir: {VIDEO_DIR}")
print(f"  Vector DB: Qdrant Cloud (via environment variables)")
