# TimeRevind

AI-powered intelligent surveillance system with real-time person detection, multi-person tracking, face recognition, behavior analysis, and smart voice query interface.

## Overview

TimeRevind is a comprehensive surveillance solution that combines state-of-the-art computer vision models with intelligent behavior analysis to provide actionable insights from video streams. Built with YOLOv8/v11 for detection, ArcFace for face embeddings, and Qdrant vector database for efficient similarity search, TimeRevind enables fast, accurate person re-identification and behavior tracking.

## Features

- **Real-time Person Detection**: YOLOv8 nano model for efficient multi-person detection
- **Multi-Person Tracking**: ByteTrack-based tracking with 60-second timeout and visit differentiation
- **Face Recognition & Re-ID**: ArcFace embeddings (ResNet50) with Qdrant vector database for fast similarity matching
- **Pose Estimation**: MediaPipe-based pose detection for activity classification (sitting, standing, lying)
- **Zone-based Monitoring**: Configurable zones with entry/exit detection and dwell time tracking
- **Activity Logging**: Automatic logging of key milestones (sitting duration, device usage, zone presence)
- **Voice Query Interface**: WebSocket-based voice commands with wake word detection ("hey sentinel")
- **Web Dashboard**: Real-time monitoring with person list, activity logs, and event timeline
- **SQLite + Qdrant**: Dual storage for structured events and vector embeddings

## Tech Stack

### Core Vision & ML
- **YOLOv8/v11** — Object detection and person tracking
- **ArcFace (ONNX)** — Face embedding generation via InsightFace
- **SCRFD (ONNX)** — Face detection via SCRFD-10G
- **MediaPipe** — Pose estimation (optional)

### Backend
- **FastAPI** — Async web framework with WebSocket support
- **Python 3.8+** — Core language
- **Uvicorn** — ASGI server

### Databases & Search
- **SQLite** — Event and activity logging
- **Qdrant Cloud** — Vector database for face embeddings and similarity search

### Additional Libraries
- **OpenCV** — Image processing and video handling
- **PyTorch/TorchVision** — GPU acceleration (CUDA-compatible)
- **ONNX Runtime** — Inference for face detection/embedding models
- **Groq API** — LLM integration for intelligent query processing
- **OpenWakeword** — Wake word detection
- **PyAudio & gTTS** — Voice I/O and text-to-speech

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU fallback
- ~2GB free disk space for models
- FFmpeg installed and in PATH

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/TimeRevind.git
cd TimeRevind
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download models**
Models will auto-download on first run:
- `yolov8n.pt` — YOLO nano model
- `det_10g_dynamic.onnx` — SCRFD face detector
- `w600k_r50.onnx` — ArcFace embedding model

5. **Configure environment**
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```env
QDRANT_URL=your-qdrant-cloud-url
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
```

6. **Initialize database**
```bash
python -c "from src.database import init_db; init_db()"
```

7. **Run the system**
```bash
python main.py
```

Access the dashboard at `http://localhost:8000`

## Usage

### Starting the Surveillance System

```bash
python main.py
```

The system will:
1. Start video capture from configured camera
2. Initialize YOLO, face detection, and tracking models
3. Launch FastAPI backend on port 8000
3. Serve web dashboard on http://localhost:8000
4. Activate voice query interface (listen for "hey sentinel")

### Web Dashboard

Access at `http://localhost:8000`:
- **Person List** — Current tracked individuals with confidence scores
- **Activity Log** — Real-time event stream (entries, exits, activity milestones)
- **Timeline** — Historical events with timestamps and duration
- **Person Details** — Click any person to view:
  - Best face frame
  - Visit history
  - Activity timeline
  - Inferred tags

### Voice Queries

Speak after hearing the wake word "hey sentinel":
- "Who was here yesterday?" → Returns persons with timestamps
- "Show me sitting events" → Filters by activity type
- "When did person X leave?" → Query-specific person timelines
- "List all activities in zone 1" → Zone-filtered events

## Configuration

Key settings in `src/config.py`:

### Detection Thresholds
```python
YOLO_CONFIDENCE_THRESHOLD = 0.2       # Person detection confidence
FACE_CONFIDENCE_THRESHOLD = 0.50      # Face detection threshold
FACE_EMBEDDING_L2_THRESHOLD = 0.5     # Face matching threshold
```

### Tracking
```python
BYTETRACK_TRACK_BUFFER = 30           # Frames to retain inactive tracks
PERSON_TIMEOUT_SECONDS = 60           # Seconds before marking LOST
NEW_VISIT_THRESHOLD_SECONDS = 30      # Absence threshold for new visit
```

### Zones
```python
ZONES = {
    "desk_area": [(100, 100), (400, 100), (400, 400), (100, 400)],
    "sofa_zone": [(500, 200), (800, 200), (800, 500), (500, 500)],
}
```

### Video Settings
```python
VIDEO_FPS = 30
VIDEO_RESOLUTION = (1920, 1080)       # 1080p
VIDEO_BITRATE = "3000k"
```

See `src/config.py` for complete configuration options.

## API Endpoints

### REST API

**Get all persons**
```
GET /api/persons
Response: [{"id": int, "name": str, "confidence": float, "status": str, ...}]
```

**Get person details**
```
GET /api/persons/{person_id}
Response: {"id": int, "visits": [...], "activities": [...], "best_frame": str}
```

**Get activity logs**
```
GET /api/logs?start_time=2024-01-01&end_time=2024-01-02&person_id=1
Response: [{"id": int, "timestamp": str, "activity": str, "person_id": int, ...}]
```

### WebSocket API

**Voice Query Interface**
```
ws://localhost:8000/ws/query
```

Send JSON:
```json
{"query": "Who was here yesterday?", "person_id": null}
```

Receive JSON:
```json
{"response": "Person John was here on Jan 15 from 9:00 AM to 5:30 PM", "timestamp": "2024-01-16T10:30:00"}
```

## Database Schema

### `persons` table
```sql
id INTEGER PRIMARY KEY
name TEXT
confidence REAL
status TEXT (ACTIVE, LOST, EXITED)
first_seen TIMESTAMP
last_seen TIMESTAMP
visit_count INTEGER
best_face_frame BLOB
```

### `activities` table
```sql
id INTEGER PRIMARY KEY
person_id INTEGER
activity_type TEXT (SITTING, STANDING, USING_PHONE, USING_LAPTOP, IN_ZONE)
start_time TIMESTAMP
end_time TIMESTAMP
duration INTEGER (seconds)
zone_id INTEGER
metadata TEXT (JSON)
```

### `embeddings` table (Qdrant)
```
{
  "id": integer,
  "vector": [float, ...],  // 512-dim ArcFace embedding
  "payload": {
    "person_id": integer,
    "frame_type": string,   // "FRONT", "LEFT", "RIGHT", "UP", "DOWN"
    "timestamp": string
  }
}
```

## Architecture

```
┌─────────────────┐
│  Video Stream   │
└────────┬────────┘
         │
    ┌────▼─────────────┐
    │  Frame Extraction │ (30 FPS)
    └────┬─────────────┘
         │
    ┌────▼──────────────────┐
    │  YOLOv8 Detection      │ → Person bboxes
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  ByteTrack Tracking    │ → Track IDs, status
    └────┬──────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  Face Detection (SCRFD)        │ → Face regions
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  Face Embedding (ArcFace)      │ → 512-dim vectors
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  Qdrant Similarity Search      │ → Person re-ID
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  Pose Estimation (MediaPipe)   │ → Activity classification
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  Zone Check & Event Logging    │ → SQLite events
    └────┬──────────────────────────┘
         │
    ┌────▼────────────────┐
    │  FastAPI Backend     │ → REST + WebSocket
    └─────────────────────┘
         │
    ┌────▼────────────────┐
    │  Web Dashboard       │
    └─────────────────────┘
```

## Performance

- **Detection**: ~15-20 FPS on RTX 3060
- **Face Embedding**: ~50-100 embeddings/sec
- **Qdrant Search**: <5ms per query
- **API Response**: <100ms for typical queries

Optimize with:
```python
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Lower = more detections, slower
MAX_BATCH_SIZE = 8               # Increase for faster batching
ENABLE_GPU = True                # Use CUDA acceleration
```

## Troubleshooting

### PyAudio Device Error
```
OSError: (-9998, 'Invalid number of channels')
```
**Solution**: System auto-detects compatible audio devices. Check device list:
```bash
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```

### Face Embedding Memory Issues
**Solution**: Reduce `FRAMES_TO_COLLECT` in config.py:
```python
FRAMES_TO_COLLECT = 3  # Default: 5
```

### Slow Video Processing
**Solution**: Increase YOLO confidence threshold to skip low-confidence detections:
```python
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Fewer detections, faster
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see LICENSE file for details.

## Citation

If you use TimeRevind in research, please cite:
```bibtex
@software{timerevind2024,
  title={TimeRevind: AI-Powered Surveillance System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TimeRevind}
}
```

## Contact & Support

- **Email**: zalaidbutt2@gmail.com
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

---

**Disclaimer**: This system is designed for authorized surveillance use only. Ensure compliance with local privacy laws and obtain necessary permissions before deployment.
