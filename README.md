# TimeRevind — AI Surveillance System

> **"Who was here? When? For how long? Do I know them?"**
> TimeRevind answers all of these — in real time and from recorded video.

---

## What Is TimeRevind?

TimeRevind is a smart surveillance system that goes far beyond basic CCTV. It:

- **Detects** every person in a camera feed using YOLO
- **Identifies** them by their face using ArcFace (even if they leave and come back)
- **Tracks** when they entered, exited, and what they were doing
- **Alerts** you via push notification when an unknown or specific person arrives after hours
- **Lets you query** your footage in plain English — *"Who was sitting at 4pm?"*

---

## How Data Flows — Big Picture

```
Camera / Video File
        │
        ▼
 ┌─────────────┐
 │  YOLOv8n    │  ← Step 1: Detects people in the frame (bounding boxes)
 └──────┬──────┘
        │  person crops
        ▼
 ┌─────────────┐
 │  ByteTrack  │  ← Step 2: Assigns each person a Track ID and follows them frame-to-frame
 └──────┬──────┘
        │  track_id + crop
        ▼
 ┌─────────────────────┐
 │  SCRFD-10G          │  ← Step 3: Finds the face inside the person crop
 │  (Face Detector)    │
 └──────┬──────────────┘
        │  face crop + 5 key points (eyes, nose, mouth corners)
        ▼
 ┌─────────────────────┐
 │  ArcFace ResNet50   │  ← Step 4: Converts face → 512 numbers (unique identity vector)
 │  (Face Embedding)   │
 └──────┬──────────────┘
        │  512D vector
        ▼
 ┌─────────────────────┐
 │  Qdrant (Cloud DB)  │  ← Step 5: Searches "Does this face match anyone we know?"
 └──────┬──────────────┘
        │  match result
        ▼
 ┌─────────────────────┐
 │  Identity Decision  │  ← Step 6: New person? Returning? Unknown intruder?
 └──────┬──────────────┘
        │
   ┌────┴─────────┐
   ▼              ▼
SQLite DB     Pushover App
(Events Log)  (Phone Alert)
   │
   ▼
FastAPI + Web Dashboard
(View history, query in English)
```

---

## Module-by-Module Breakdown

---

### 1. `main.py` — The Starter

This is where the program begins. Run it to start the whole system.

```bash
# Live camera
python main.py --mode camera

# Process a recorded video file
python main.py --mode video --video videos/test.mp4

# Enable verbose debug logs
python main.py --mode camera --debug
```

**What it does internally:**
- Sets up clean, readable log formatting (timestamps + icons)
- Fixes a Windows GPU issue: exposes PyTorch's CUDA DLLs so ONNX models (SCRFD, ArcFace) run on GPU instead of silently falling back to CPU
- Loads the database, video processor
- Calls either `process_camera_stream()` or `process_video_file()` based on the flag
- Prints a summary when done (frames processed, people detected, entries logged)

---

### 2. `src/core/video_processor.py` — The Brain

This is the main pipeline. Every single frame from the camera passes through here in order.

**Frame-by-frame flow:**

```
Frame arrives (30 per second)
      │
      ▼
[1] YOLOv8n → finds all people → returns bounding boxes + confidence scores
      │
      ▼
[2] ByteTrack → assigns/maintains Track IDs across frames
      │           (same person keeps same ID even if temporarily occluded)
      ▼
[3] PersonTracker.update() → updates each Person object's state
      │                        (ACTIVE / LOST / EXITED)
      ▼
[4] For each person: crop the frame to their bounding box
      │
      ▼
[5] SCRFD face detector → finds face within that crop
      │
      ▼
[6] ArcFace embedder → 512D identity vector
      │
      ▼
[7] Every 5 frames collected → search Qdrant vector DB
      │
      ├── Match found (similarity ≥ 0.47)?
      │       → Label track as known person (e.g., "jilani")
      │
      └── No match after 30 embeddings?
              → Create new Person_N (new unknown person)
      │
      ▼
[8] EventLogger → write ENTERED / EXITED to SQLite
      │
      ▼
[9] Pushover → send phone notification if conditions met
```

**Why collect 5 frames before deciding identity?**
A single frame can be unreliable — bad lighting, motion blur, side profile. Collecting embeddings from multiple frames and taking a majority vote gives a much more confident identity match.

---

### 3. `src/core/person_tracker.py` — The Memory of Each Person

This module maintains a `Person` object for every individual being tracked.

**The `Person` object stores:**

| Field | What It Means |
|---|---|
| `person_id` | e.g., `"Person_3"` or a name like `"jilani"` |
| `first_seen` | When they first appeared |
| `last_seen` | Last frame they were visible |
| `visit_number` | How many times they've entered (this session) |
| `tracking_state` | `ACTIVE`, `LOST`, or `EXITED` |
| `face_embeddings` | List of 512D face vectors collected so far |
| `face_embedding_poses` | Pose for each embedding: FRONT/LEFT/RIGHT/UP/DOWN |

**Tracking States — State Machine:**

```
Person walks in
      → ACTIVE  (YOLO can see them)

Person steps behind a pillar (< 12 frames)
      → still ACTIVE  (ByteTrack keeps track)

Gone for 12+ consecutive frames
      → LOST  (might be temporary occlusion)


Person comes back later
      → Re-identified by face embeddings
      → New ENTERED event logged
      → visit_number incremented
```

---

### 4. `src/core/embeddings.py` — The Face Recognition Engine

This is the most technically complex module. It runs two AI models back-to-back.

#### Step A: SCRFD-10G (Face Detector)

- Input: a cropped image of a person (from YOLO bbox)
- Output: bounding box around the face + 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner)
- Runs at 640×640 resolution
- Uses anchor-based detection across 3 scales — good at catching small/distant faces

#### Step B: ArcFace ResNet50 (Face Embedder)

- Takes the detected face crop
- Uses the 5 landmarks to **align** the face to a standard front-facing position (112×112 pixels)
- This alignment step is critical — it removes the effect of head tilt or rotation
- Passes through a 50-layer neural network trained on millions of faces
- Outputs: a **512-dimensional vector** — a unique "fingerprint" for that face

**Pose-Aware Matching:**

Before comparing two embeddings, the system checks whether their poses are compatible:

```
FRONT, UP, DOWN  →  compare with each other freely
LEFT             →  only compare with other LEFT embeddings
RIGHT            →  only compare with other RIGHT embeddings
```

This prevents a false rejection when someone turns their head — a LEFT profile won't be wrongly compared against a FRONT-facing stored embedding.

---

### 5. `src/database/db_init.py` — Where Everything Is Stored

The system uses two databases together:

#### SQLite (Local File: `db/timerevind.db`)

A regular relational database stored on your computer. Fast, private, no internet needed.

| Table | What's In It |
|---|---|
| `person_profiles` | Every person ever seen — ID, name, first/last seen, total visits, total time spent |
| `events` | Every ENTERED and EXITED event with exact timestamps and video file references |
| `sessions` | Each recording session (start/end time, camera used, video filename) |
| `activity_events` | Behavioral events — sitting, walking, using phone, etc. with durations |
| `notification_logs` | Every notification sent — who, when, video frame number, whether it was sent or skipped |


### 6. `src/core/event_logger.py` — Writing History

Every time something meaningful happens, `EventLogger` writes it to SQLite.

```python
log_entry(person_id, visit_num, video_file, video_timestamp)   # Person walked in
log_exit(person_id, visit_num, video_file, video_timestamp)    # Person walked out
```

These logs are what power the web dashboard and the NLP query system.

---

### 7. `src/core/notifier.py` — Push Notifications
Uses the **Pushover** service to send an instant notification to your phone.

```python
send_notification("shiza", "16:03:22", "ENTRY")
# → Phone receives: "⚠️ Alert: shiza entered at 16:03:22"
```

Configuration in `.env`:
```
PUSHOVER_API_KEY=...   (your app's key from pushover.net)
PUSHOVER_USER=...      (your account's user key)
```

The notification is sent via a simple HTTP POST request — no complex setup needed. Works on both iOS and Android instantly.

---

---


### 10. `api.py` — The Web Backend

A **FastAPI** server that powers the web dashboard and exposes REST endpoints.

**Key Endpoints:**

| URL | Method | What It Does |
|---|---|---|
| `/api/persons` | GET | List all tracked people with photos |
| `/api/persons/{id}/rename` | PATCH | Give someone a real name (e.g., rename Person_3 → "Ahmed") |
      | `/api/events` | GET | Browse all entry/exit history (filterable) |
| `/api/video-event/{event_id}` | GET | Get the video file and timestamp for a specific event |
| `/api/query-nlp` | POST | Ask a question in plain English |
| `/api/activity-timeline/{id}` | GET | See full activity history for one person |
| `/ws/query` | WebSocket | Voice query interface |


**NLP Query System (powered by Groq LLM):**

```
You type: "Who was sitting between 3pm and 5pm?"
      │
      ▼
Groq LLM receives: your question + the database schema description
      │
      ▼
Groq generates: the correct SQL query
      │
      ▼
SQLite executes the query → returns results
      │
      ▼
You receive: plain English answer + structured data
```

**Run it:**
```bash
python api.py
# Open http://localhost:8000 in browser
```

---

### 11. `frontend/index.html` — The Web Dashboard

A single-page web app with a dark theme — no external framework needed, just plain HTML/CSS/JS.

**Layout:**
- **Left panel (50%):** Video player — click any event to jump to that moment in the recording
- **Right panel (50%):** Person list + activity timeline + NLP chat bar

**Features:**
- Person cards with face photos
- Click a person → see all their visits, events, activity history
- Filter events: Today / Last 7 days / All time
- Rename any person by clicking their name

---

## Models Used

| Model | File | What It Does |
|---|---|---|
| YOLOv8n | `yolov8n.pt` | Detects people (bounding boxes) |
| SCRFD-10G | `det_10g_dynamic.onnx` | Finds faces inside person crops |
| ArcFace ResNet50 | `w600k_r50.onnx` | Converts face → 512D identity vector |
| YOLOv11n-pose | auto-downloaded | Detects 17 body keypoints for behavior analysis |
| Groq LLM | cloud API | Converts English questions → database queries |

---

## Environment Setup

Create a `.env` file in the project root:

```env
# Groq (NLP queries — free tier available at console.groq.com)
GROQ_API_KEY=gsk_...

# Qdrant Cloud (vector DB — free tier at cloud.qdrant.io)
QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key

# Pushover (phone notifications — one-time $5 app purchase)
PUSHOVER_API_KEY=your_app_key
PUSHOVER_USER=your_user_key

# Optional
DEBUG=false
LOG_LEVEL=INFO
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Live camera surveillance
python main.py --mode camera

# Process a recorded video
python main.py --mode video --video videos/myvideo.mp4

# Demo notification system on a video
python notify_demo_instant.py --video videos/test.mp4

# Start the web dashboard
python api.py
# Then open http://localhost:8000
```

---

## Key Design Decisions (and Why)

### Why Qdrant instead of a regular database for face embeddings?
Face embeddings are 512-number vectors. Finding the "most similar" vector by scanning all rows in SQLite would be O(n) — slow with many people. Qdrant uses HNSW (Hierarchical Navigable Small World) indexing, which finds the best match in milliseconds even with millions of stored faces.

### Why collect multiple frames before deciding someone's identity?
A single frame can be unreliable — motion blur, side angle, bad lighting. The system collects embeddings from multiple frames and votes on the result. This makes identification far more accurate and avoids false mismatches.

### Why ByteTrack for multi-person tracking?
ByteTrack maintains a persistent `track_id` per person across frames — even through brief occlusions (someone walking in front). Without it, the same person walking behind a pillar would appear as a "new person" every time, creating duplicate records.

### Why the 4pm time gate in the demo?
It simulates real-world after-hours monitoring — a home or office that only needs alerts when no one should be there. The system watches silently during work hours and becomes alert at night.

---

## Project Structure

```
TimeRevind/
├── main.py                        ← Start here: live camera or video file
├── api.py                         ← Web dashboard backend (FastAPI)
├── notify_demo_instant.py         ← Demo: instant alerts with visual bounding boxes
├── .env                           ← API keys (never commit this file)
├── requirements.txt
│
├── src/
│   ├── config.py                  ← All thresholds and tunable settings
│   ├── core/
│   │   ├── video_processor.py     ← Main pipeline: frame → identity → event
│   │   ├── person_tracker.py      ← Person objects + ACTIVE/LOST/EXITED state machine
│   │   ├── embeddings.py          ← SCRFD face detect + ArcFace 512D embed
│   │   ├── event_logger.py        ← Writes ENTERED/EXITED/activity to SQLite
│   │   ├── notifier.py            ← Pushover push notifications
│   │       └── config.py             ← Behavior thresholds and cooldowns
│   └── database/
│       └── db_init.py             ← SQLite schema creation + Qdrant Cloud connection
│
├── models/buffalo_l/
│   ├── w600k_r50.onnx             ← ArcFace ResNet50 (512D face embeddings)
│   └── det_10g_dynamic.onnx       ← SCRFD-10G face detector
│
├── frontend/
│   └── index.html                 ← Web dashboard (dark theme, single page)
│
├── db/
│   └── timerevind.db              ← SQLite database (events, profiles, activity logs)
│
└── videos/                        ← Put your input video files here
```

---


---

## How TimeRevind Compares to Existing Solutions

| Feature | Basic CCTV | Cloud AI (AWS/Azure) | TimeRevind |
|---|---|---|---|
| Detects people | Yes | Yes | Yes |
| Recognizes faces | No | Yes (cloud only) | Yes (runs locally) |
| Re-identifies across sessions | No | No | Yes |
| Alerts on unknown intruders | No | No | Yes |
| Query footage in plain English | No | No | Yes |
| Runs on-device (private) | Yes | No | Yes |
| Time-gated alerts | No | No | Yes |
| Cost per detection | Free | Pay per API call | Free after setup |

---

#

---

*This system is intended for authorized surveillance use only. Ensure compliance with local privacy laws before deployment.*
