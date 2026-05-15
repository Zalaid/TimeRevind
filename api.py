"""
TimeRevind API Server
FastAPI backend for the surveillance dashboard UI
Run: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
import openwakeword
from openwakeword.model import Model as WakeWordModel
import pyaudio as _pyaudio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from groq import Groq
from gtts import gTTS
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("timerevind.api")

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent
DB_PATH         = PROJECT_ROOT / "db" / "timerevind.db"
BEST_FRAMES_DIR = PROJECT_ROOT / "best_one"
VIDEOS_DIR      = Path(r"d:\BSDS\8th Semester\Projects\TimeRevind\videos")  # All videos pre-converted to H.264
TTS_DIR         = PROJECT_ROOT / "tts_cache"
TTS_DIR.mkdir(exist_ok=True)
BEST_FRAMES_DIR.mkdir(exist_ok=True)

# ─── Clients ─────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
PORCUPINE_API_KEY = os.getenv("PORCUPINE_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)


app = FastAPI(title="TimeRevind")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_shutdown_event = threading.Event()

@app.on_event("shutdown")
async def _on_shutdown():
    _shutdown_event.set()


# ════════════════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════════════════

def get_db():
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def query_db(sql: str, params=None) -> list:
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute(sql, params or []).fetchall()]
    finally:
        conn.close()

def execute_db(sql: str, params=None):
    import sqlite3
    conn = get_db()
    try:
        cur = conn.execute(sql, params or [])
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error:
        conn.rollback(); raise
    finally:
        conn.close()


def _format_timestamps(results: list) -> list:
    """Post-process query results to format timestamps to 12-hour format."""
    from datetime import datetime

    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

    for row in results:
        for key, value in row.items():
            if isinstance(value, str) and (' ' in value or 'T' in value):
                # Try to parse as timestamp (formats: "YYYY-MM-DD HH:MM:SS" or ISO 8601 "2026-05-14T15:48:00")
                try:
                    if 'T' in value:
                        dt = datetime.fromisoformat(value.replace('Z', '+00:00').split('.')[0])
                    else:
                        dt = datetime.strptime(value.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    # Format as "May 14, 2026 at 03:48 PM"
                    formatted = dt.strftime(f'{month_names[dt.month]} %d, %Y at %I:%M %p')
                    row[key] = formatted
                except (ValueError, AttributeError):
                    pass  # Not a timestamp, leave as is

    return results


# ════════════════════════════════════════════════════════════════════════════
# PERSONS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/persons")
async def get_persons():
    rows = query_db("""
        SELECT person_id, name, first_seen, last_seen, total_visits, total_time_seconds
        FROM person_profiles ORDER BY last_seen DESC
    """)
    for row in rows:
        pid = row["person_id"]
        d   = BEST_FRAMES_DIR / pid
        photos = []
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    photos.append(f"/photos/{pid}/{f.name}")
                if len(photos) >= 3:
                    break
        row["photos"] = photos
    return rows

class RenameRequest(BaseModel):
    name: str

@app.patch("/api/persons/{person_id}/rename")
async def rename_person(person_id: str, body: RenameRequest):
    name = body.name.strip() or None
    execute_db(
        "UPDATE person_profiles SET name=?, updated_at=CURRENT_TIMESTAMP WHERE person_id=?",
        (name, person_id)
    )
    return {"ok": True, "person_id": person_id, "name": name}

@app.get("/photos/{person_id}/{filename}")
async def serve_photo(person_id: str, filename: str):
    path = BEST_FRAMES_DIR / person_id / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(str(path))


# ════════════════════════════════════════════════════════════════════════════
# EVENTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/events")
async def get_events(limit: int = 80, person_id: Optional[str] = None):
    sql = """
        SELECT e.id, e.timestamp, e.person_id, e.event_type, e.visit_num,
               e.video_file, e.video_timestamp_start,
               COALESCE(p.name, e.person_id) AS display_name
        FROM events e
        LEFT JOIN person_profiles p ON e.person_id = p.person_id
    """
    params: list = []
    if person_id:
        sql += " WHERE e.person_id = ?"; params.append(person_id)
    sql += " ORDER BY e.timestamp DESC LIMIT ?"; params.append(limit)
    return query_db(sql, params)

@app.get("/api/video-event/{event_id}")
async def video_for_event(event_id: int):
    rows = query_db("SELECT video_file, video_timestamp_start FROM events WHERE id=?", [event_id])
    if not rows: raise HTTPException(404)
    r = rows[0]
    # Keep relative path from videos/ onwards (e.g., "m/IMG_6315.mp4")
    video_path = r["video_file"]
    if video_path:
        # Extract path relative to PROJECT_ROOT
        try:
            rel_path = Path(video_path).relative_to(VIDEOS_DIR)
            fn = str(rel_path).replace("\\", "/")  # Convert Windows paths to forward slashes
        except:
            fn = Path(video_path).name  # Fallback to just filename
    else:
        fn = None
    return {"video_url": f"/videos/{fn}" if fn else None, "timestamp_seconds": r["video_timestamp_start"] or 0}

@app.get("/api/videos")
async def list_videos():
    files = []
    if VIDEOS_DIR.exists():
        for f in sorted(VIDEOS_DIR.iterdir(), reverse=True):
            if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                files.append({"name": f.name, "url": f"/videos/{f.name}", "size_mb": round(f.stat().st_size / 1e6, 1)})
    return files

@app.get("/videos/{file_path:path}")
async def serve_video(file_path: str):
    """Serve pre-converted H.264 video files (supports subfolders like m/IMG_6315.mp4)"""
    video_file = VIDEOS_DIR / file_path
    if not video_file.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(str(video_file), media_type="video/mp4")


# ════════════════════════════════════════════════════════════════════════════
# NLP QUERY
# ════════════════════════════════════════════════════════════════════════════

_SCHEMA_DESCRIPTION = """
DATABASE SCHEMA (SQLite — timerevind.db):

=== TABLE: person_profiles ===
COLUMNS:
- person_id (TEXT, PRIMARY KEY): Unique ID like "Person_1", "person_002"
- name (TEXT, nullable): Human label like "John Doe", "Zelaid". NULL if unnamed
- first_seen (TIMESTAMP): Very first moment person was detected across all sessions
- last_seen (TIMESTAMP): Most recent moment person was detected across all sessions
- total_visits (INT): TOTAL NUMBER OF ENTRIES across ALL sessions combined
  * This counts every time person ENTERED (every ENTERED event = 1 visit)
  * If user entered 7 times total across all sessions, total_visits = 7
  * NOT the number of sessions attended
- total_time_seconds (INT): Total time person was visible in camera across ALL sessions
  * Sum of all time spent in ALL sessions
  * E.g., if 5 mins in session 1 + 3 mins in session 2 = 480 + 180 = 660 seconds
- gender (TEXT, nullable): Gender label - "M" for Male, "F" for Female. NULL if unknown
- OTHER: keyframe_path, notes, created_at, updated_at (rarely needed)

IMPORTANT:
- total_visits = TOTAL ENTRIES across all sessions (count of ENTERED events)
- total_time_seconds = TOTAL DURATION across all sessions (sum of all time spent)
- Always COALESCE(name, person_id) to show name if available, else ID

=== TABLE: events (Entry/Exit Log) ===
COLUMNS:
- id (INT, PRIMARY KEY): Event ID
- timestamp (TEXT): Exact moment of entry/exit - STORED FORMAT: "YYYY-MM-DD HH:MM:SS" (e.g., "2026-05-14 15:48:00")
- person_id (TEXT, FK): Reference to person_profiles
- event_type (TEXT): Either 'ENTERED' or 'EXITED'
- visit_num (INT): Visit number WITHIN THIS SESSION ONLY
  * Tracks how many times person entered/exited within a specific session
  * Session 1: ENTER (visit_num=1), EXIT, ENTER (visit_num=2), EXIT
  * Session 2: ENTER (visit_num=1), EXIT, ENTER (visit_num=2), EXIT, ENTER (visit_num=3)
  * visit_num resets to 1 for each new session
- session_id (TEXT): Which recording session this belongs to
- video_file (TEXT): Path to video file where this was recorded
- video_timestamp_start (INT): FRAME NUMBER in video (divide by 30 for seconds)

IMPORTANT:
- visit_num is PER SESSION - resets for each session_id
- event_type = 'ENTERED' means person came into frame
- event_type = 'EXITED' means person left frame (away for >30 seconds)
- Do NOT count events as visits - use person_profiles.total_visits
- Do NOT count sessions as visits - use person_profiles.total_visits
- Do NOT sum time from events - use person_profiles.total_time_seconds

=== TABLE: sessions ===
COLUMNS:
- session_id (TEXT, PRIMARY KEY): Recording session ID
- started_at (TIMESTAMP): Recording start time
- ended_at (TIMESTAMP): Recording end time
- camera_id (TEXT): Which camera
- video_file (TEXT): Path to video
- created_at (TIMESTAMP)

IMPORTANT:
- Do NOT query sessions when asking about "visitors" or "people"
- Sessions are recording sessions, NOT visit counts

=== TABLE: embedding_metadata ===
- For face/body recognition data, rarely needed for basic queries

=== BUSINESS LOGIC ===

VISIT TRACKING:
- Each time person ENTERS camera frame = 1 visit (event_type='ENTERED')
- If person leaves frame for >30 seconds = EXIT logged (event_type='EXITED')
- If person comes back WITHIN SAME SESSION = visit_num increments
- When new session starts = visit_num resets to 1

EXAMPLE - One person in 2 sessions:
SESSION 1 (session_id='sess_001'):
  - Enters: event_type='ENTERED', visit_num=1
  - Leaves for 30s: event_type='EXITED', visit_num=1
  - Comes back: event_type='ENTERED', visit_num=2
  - Leaves: event_type='EXITED', visit_num=2
SESSION 2 (session_id='sess_002'):
  - Enters: event_type='ENTERED', visit_num=1  (reset to 1)
  - Leaves: event_type='EXITED', visit_num=1

PERSON PROFILE TOTALS:
- total_visits = 3 (total ENTERS: 1+1+1)
- total_time_seconds = sum of all time in both sessions

=== WHICH TABLE TO USE FOR WHAT ===

⚠️ CRITICAL: person_profiles = ALL-TIME CUMULATIVE | events = DATE-FILTERED QUERIES

PERSON_PROFILES (for ALL-TIME totals across entire history):
- "How many times has X visited in total (all time)?" → total_visits (ALL-TIME)
- "How long has X spent in total (all time)?" → total_time_seconds (ALL-TIME)
- "When was X first/last seen?" → first_seen, last_seen
- "Who are the top visitors overall?" → ORDER BY total_visits DESC
- "Who spent the most time overall?" → ORDER BY total_time_seconds DESC

EVENTS TABLE (for TIME-SPECIFIC queries - TODAY, LAST 24H, LAST WEEK, etc):
- "Summary of TODAY?" → Filter events WHERE timestamp >= datetime('now', 'start of day')
  * COUNT events WHERE event_type='ENTERED' = visits today
  * SUM duration between ENTERED-EXITED pairs = time spent today
- "How many times did X visit TODAY?" → Count ENTERED events for X where timestamp is today
- "How long did X spend TODAY?" → Sum time between ENTERED-EXITED pairs for X today
- "Last 24 hours summary?" → Filter events WHERE timestamp >= datetime('now', '-24 hours')
- "Show entry/exit log for X?" → SELECT all events for X (optionally filtered by date)
- "When exactly did X enter?" → timestamp + event_type from events table
- "Get video for entry?" → video_file, video_timestamp_start from events table

SESSIONS (for recording details - rarely needed):
- "Which recordings were made?" → session details
- NEVER use for counting people or visits

=== CRITICAL RULES ===

1. WHEN ASKED "Summary of TODAY":
   - Filter events table by TODAY's date, count ENTERED events
   - Example:
     SELECT COALESCE(p.name, p.person_id) as person,
            COUNT(CASE WHEN e.event_type='ENTERED' THEN 1 END) as visits_today,
            MAX(e.timestamp) as last_visit
     FROM person_profiles p
     LEFT JOIN events e ON p.person_id = e.person_id
        AND DATE(e.timestamp) = DATE('now', 'localtime')
     WHERE DATE(e.timestamp) = DATE('now', 'localtime')
     GROUP BY p.person_id

2. WHEN ASKED "Summary of last 24 hours":
   - Filter events table by timestamp >= 24 hours ago
   - Count ENTERED events for visits in last 24 hours
   - Example:
     SELECT COALESCE(p.name, p.person_id) as person,
            COUNT(CASE WHEN e.event_type='ENTERED' THEN 1 END) as visits_24h,
            MAX(e.timestamp) as last_visit
     FROM person_profiles p
     LEFT JOIN events e ON p.person_id = e.person_id
        AND e.timestamp >= datetime('now', '-24 hours', 'localtime')
     WHERE e.timestamp >= datetime('now', '-24 hours', 'localtime')
     GROUP BY p.person_id

DATE COMPARISONS - TIMESTAMPS ARE STORED AS YYYY-MM-DD HH:MM:SS FORMAT:
- Timestamps in events table are TEXT format: "YYYY-MM-DD HH:MM:SS" (e.g., "2026-05-14 15:48:00")
- Use SQLite's DATE() function directly - it parses this format correctly
- CORRECT: SELECT * FROM events WHERE DATE(timestamp) = '2026-05-14'
- CORRECT: SELECT * FROM events WHERE DATE(timestamp) >= '2026-05-14'
- For comparisons by day: Use DATE(timestamp) = '2026-05-14'
- For comparisons by month: Use strftime('%Y-%m', timestamp) = '2026-05'
- For comparisons by year: Use strftime('%Y', timestamp) = '2026'

3. WHEN ASKED "How many times has X visited TOTAL (all-time)":
   - Use person_profiles.total_visits DIRECTLY (ALL-TIME cumulative)
   - Example: SELECT COALESCE(name, person_id) as person, total_visits
              FROM person_profiles WHERE LOWER(name) LIKE LOWER('%X%')

4. WHEN ASKED "How many times did X visit TODAY":
   - Count ENTERED events for X filtered by today's date
   - Example: SELECT COUNT(*) as visits_today
              FROM events WHERE person_id = 'X' AND event_type = 'ENTERED'
                 AND DATE(timestamp) = DATE('now', 'localtime')

5. WHEN ASKED "How long did X spend TODAY":
   - Sum durations between ENTERED-EXITED pairs for X filtered by today
   - Example: SELECT SUM(CAST((julianday(CASE WHEN e2.event_type='EXITED' THEN e2.timestamp
                                              ELSE datetime('now') END)
                              - julianday(e1.timestamp)) AS INTEGER) * 86400) as seconds_today
              FROM events e1
              WHERE e1.person_id = 'X' AND e1.event_type = 'ENTERED'
                 AND DATE(e1.timestamp) = DATE('now')

6. WHEN ASKED "Entry/exit log for X":
   - JOIN person_profiles with events (optionally filter by date)
   - Example: SELECT COALESCE(p.name, p.person_id) as person,
                    e.event_type,
                    e.timestamp,
                    e.visit_num,
                    e.session_id
             FROM person_profiles p
             JOIN events e ON p.person_id = e.person_id
             WHERE p.person_id = 'X'
             ORDER BY e.timestamp DESC

7. WHEN ASKED "Get video of X entering":
   - FIRST query to get the frame number, then call play_video with that frame number
   - Query example: SELECT e.video_timestamp_start, e.timestamp
                   FROM events e
                   WHERE LOWER(e.person_id) = LOWER('Jilani') AND e.event_type = 'ENTERED'
                   ORDER BY e.timestamp DESC LIMIT 1
   - IMPORTANT: Use LOWER() for case-insensitive person_id matching
   - ONLY call play_video if the SQL query returns results (has data)
   - If SQL returns 0 rows: Tell user "No entries found for that person/date/time"
   - If SQL returns data: Extract video_timestamp_start and call play_video with that frame number
   - NEVER pass placeholder strings or "result from execute_sql query" to play_video

8. TIMESTAMP FORMATTING:
   - Return timestamps as-is from the database (no strftime needed)
   - The backend automatically converts them to "March 27, 2026 at 07:45 PM" format
   - Just SELECT the timestamp columns directly

9. WHEN ASKED "What did person X do?" or "What actions did X perform?":
   - Query activity_events WHERE person_id = X AND action = 'SITTING'
   - Only report SITTING actions with duration_seconds > 0
   - NEVER report STANDING (no duration is tracked for standing)
   - Example query: SELECT action, duration_seconds, timestamp FROM activity_events
                   WHERE person_id = 'Person_1' AND action = 'SITTING' AND duration_seconds > 0
   - Example answer: "Zelaid sat for 22 seconds at 6:05 PM"

10. OUTPUT FORMAT:
   - Show person names when available
   - Show time in seconds (not minutes)
   - Use COALESCE(name, person_id) always

NEVER:
- ❌ Use person_profiles.total_visits for TODAY/LAST 24H queries (it's ALL-TIME cumulative)
- ❌ Use person_profiles.total_time_seconds for TODAY/LAST 24H queries (it's ALL-TIME cumulative)
- ❌ Count sessions as visits (use person_profiles.total_visits for ALL-TIME only)
- ❌ Use visit_num for overall stats (it's per-session only)
- ❌ Forget to filter events by DATE when asking about specific time periods
- ❌ Forget COALESCE for names
- ❌ Query embedding_metadata table
- ❌ Make up column names or table names

REMEMBER:
- person_profiles = ALL-TIME historical totals
- events table = raw dated records for filtering by time period
- TODAY query? → Filter events by DATE('now'), NOT person_profiles
- LAST 24H query? → Filter events by timestamp >= datetime('now', '-24 hours'), NOT person_profiles
"""

def _execute_sql_safe(sql: str) -> list:
    """Execute SQL query safely with validation"""
    import sqlite3 as _sq

    if not sql:
        raise ValueError("Empty SQL query")

    # Validate it's a SELECT query
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")

    conn = _sq.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = _sq.Row
    try:
        results = [dict(r) for r in conn.execute(sql).fetchall()]
        # Format any ISO timestamp columns to 12-hour format
        results = _format_timestamps(results)
        return results
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise
    finally:
        conn.close()


_VIDEO_FPS    = 30
_PRE_ROLL_FRAMES = 60  # frames before entry to start playback

def _entry_seek(frame_num) -> float:
    """Convert frame number to seconds, subtract pre-roll, floor at 0."""
    entry_sec = (frame_num or 0) / _VIDEO_FPS
    return round(max(0.0, entry_sec - _PRE_ROLL_FRAMES / _VIDEO_FPS), 2)

def _video_intent(person_name: str = "", event_timestamp: str = "", video_frame_number: int = None) -> Optional[dict]:
    """Fetch video for the person/event mentioned"""
    # If specific frame number provided, use it directly
    if video_frame_number is not None:
        rows = query_db("""
            SELECT e.video_file, e.video_timestamp_start, e.timestamp, p.person_id,
                   COALESCE(p.name, e.person_id) AS display_name
            FROM events e
            LEFT JOIN person_profiles p ON e.person_id = p.person_id
            WHERE e.video_timestamp_start = ? AND e.video_file IS NOT NULL
            LIMIT 1
        """, [video_frame_number])
        if rows:
            r = rows[0]
            video_path = r["video_file"]
            try:
                rel_path = Path(video_path).relative_to(VIDEOS_DIR)
                fn = str(rel_path).replace("\\", "/")
            except:
                fn = Path(video_path).name
            return {"video_url": f"/videos/{fn}",
                    "timestamp_seconds": _entry_seek(r["video_timestamp_start"]),
                    "event_time": r["timestamp"], "display_name": r["display_name"]}

    # If specific timestamp provided, fetch that exact event
    if event_timestamp:
        rows = query_db("""
            SELECT e.video_file, e.video_timestamp_start, e.timestamp, p.person_id,
                   COALESCE(p.name, e.person_id) AS display_name
            FROM events e
            LEFT JOIN person_profiles p ON e.person_id = p.person_id
            WHERE e.timestamp = ? AND e.event_type='ENTERED' AND e.video_file IS NOT NULL
            LIMIT 1
        """, [event_timestamp])
        if rows:
            r = rows[0]
            video_path = r["video_file"]
            try:
                rel_path = Path(video_path).relative_to(VIDEOS_DIR)
                fn = str(rel_path).replace("\\", "/")
            except:
                fn = Path(video_path).name
            return {"video_url": f"/videos/{fn}",
                    "timestamp_seconds": _entry_seek(r["video_timestamp_start"]),
                    "event_time": r["timestamp"], "display_name": r["display_name"]}

    # If person_name provided, fetch their most recent video
    if person_name:
        persons = query_db("SELECT person_id, name FROM person_profiles")
        q = person_name.lower()
        for p in persons:
            lbl = (p["name"] or "").lower()
            pid = p["person_id"].lower()
            if (lbl and lbl in q) or pid in q:
                rows = query_db("""
                    SELECT id, video_file, video_timestamp_start, timestamp FROM events
                    WHERE person_id=? AND event_type='ENTERED' AND video_file IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 1
                """, [p["person_id"]])
                if rows:
                    r = rows[0]
                    video_path = r["video_file"]
                    try:
                        rel_path = Path(video_path).relative_to(VIDEOS_DIR)
                        fn = str(rel_path).replace("\\", "/")
                    except:
                        fn = Path(video_path).name
                    return {"video_url": f"/videos/{fn}",
                            "timestamp_seconds": _entry_seek(r["video_timestamp_start"]),
                            "event_time": r["timestamp"], "display_name": p["name"] or p["person_id"]}

    # No specific info — return the most recent video with an entry event
    rows = query_db("""
        SELECT e.video_file, e.video_timestamp_start, e.timestamp,
               COALESCE(p.name, e.person_id) AS display_name
        FROM events e
        LEFT JOIN person_profiles p ON e.person_id = p.person_id
        WHERE e.video_file IS NOT NULL AND e.event_type='ENTERED'
        ORDER BY e.timestamp DESC LIMIT 1
    """)
    if rows:
        r = rows[0]
        video_path = r["video_file"]
        try:
            rel_path = Path(video_path).relative_to(VIDEOS_DIR)
            fn = str(rel_path).replace("\\", "/")
        except:
            fn = Path(video_path).name
        return {"video_url": f"/videos/{fn}",
                "timestamp_seconds": _entry_seek(r["video_timestamp_start"]),
                "event_time": r["timestamp"], "display_name": r["display_name"]}
    return None


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute a SELECT query against the TimeRevind SQLite database "
                "to answer questions about persons, visits, timestamps, and events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A valid SQLite SELECT statement"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_video",
            "description": (
                "Look up and return footage for a person. "
                "Call this when the user asks to see/watch/play a video or clip. "
                "Prefer video_frame_number for precise seeking. Fallback to person_name for most recent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "Optional: Name or person_id to look up. Empty string for most recent."
                    },
                    "video_frame_number": {
                        "type": ["integer", "string"],
                        "description": "Optional: Frame number from video_timestamp_start column (e.g., 22317 or '22317'). Most reliable method."
                    },
                    "event_timestamp": {
                        "type": "string",
                        "description": "Optional: Exact event timestamp like '2026-05-14 16:02:55' (fallback if frame number unavailable)"
                    }
                },
                "required": []
            }
        }
    }
]


def _run_tool_loop(question: str, history: list, with_tts: bool = False) -> dict:
    """Agentic loop: Call Groq with tools, execute tool calls, loop until final answer."""
    # Build messages: system schema + history + user question
    system_msg = f"""{_SCHEMA_DESCRIPTION}

IMPORTANT FOR TOOL USE:
- After calling a tool and getting results, analyze if you have enough information to answer the user's question
- DO NOT call the same tool multiple times with the same arguments - reuse the results you already have
- Once you have the data needed to answer, provide the FINAL ANSWER directly without making more tool calls
- Stop when you can answer the question, do not over-call tools
- WHEN YOU HAVE SPECIFIC EVENT DETAILS FROM execute_sql AND NEED TO PLAY VIDEO:
  * STEP 1: After calling execute_sql, examine the results list carefully
  * STEP 2: If results list is EMPTY (0 rows): STOP and tell user "No matches found for [person/date/criteria]"
  * STEP 3: If results list has data: Extract the video_timestamp_start value (a number like 22317)
  * STEP 4: Call play_video with video_frame_number=22317 (use the actual number from results)
  * CRITICAL: NEVER pass strings like "result from execute_sql query" - those are NOT data
  * NEVER call play_video if execute_sql returned 0 rows
  * Example sequence:
    1. execute_sql → Returns rows with video_timestamp_start (number) and timestamp (string)
    2. play_video(video_frame_number=22317) ← Use the actual number from step 1

CRITICAL - VALIDATE TOOL RESULTS BEFORE USING THEM:
- After calling execute_sql: Check if results list is empty or has data
- If 0 rows returned: Tell user "No results found for that query" - DO NOT call other tools
- If data returned: Extract the values and use them in subsequent tool calls
- NEVER pass placeholder strings like "result from execute_sql" to other tools
- ALWAYS check: if rows, use the actual data; else, stop and report no results

CRITICAL - REUSE CONVERSATION HISTORY:
- If user asks a question you ALREADY ANSWERED in this conversation → DO NOT query database again
- Example: User asked "has anyone visited last 24 hrs?" and you answered "Zelaid 6 times"
- User asks SAME QUESTION again → ANSWER DIRECTLY from your previous response in history
- NEVER make the same database query twice in one conversation
- Check the conversation history - if the exact answer is there, use it immediately (no tools)

CONTEXT & HISTORY PRIORITY - ALWAYS CHECK CONVERSATION HISTORY FIRST:
1. FIRST CHECK conversation history before any database queries
2. If user already asked about an action/behavior and model answered it, REUSE that answer from history
3. Example:
   - User: "what did he do?" → Model queries DB, answers "Zelaid sat for 22 seconds at 06:06 PM on March 30"
   - User: "what did he do?" (asked again) → ANSWER DIRECTLY FROM HISTORY without re-querying
   - This saves time and provides instant response
4. For video requests: Track timestamps mentioned in conversation
   - If user said "sat at 06:06 PM" and then says "show me the video", use that timestamp from history
5. Only query database if:
   - The specific information is NOT in conversation history
   - User asks about a DIFFERENT person/date than what was discussed
   - User asks for NEW information (time period, different action type)
6. IF FOUND IN HISTORY → Use it immediately, NO tool calls needed
7. IF NOT IN HISTORY → Query database
8. IF STILL UNSURE → Ask for clarification

FINAL ANSWER FORMAT:
- Write answers as plain, conversational text
- For simple answers: use natural language sentences
- For people counts: ALWAYS use COUNT(DISTINCT person_id) to get unique person count
  * Wrong: SELECT COUNT(*) - counts entries/events, not unique people
  * Correct: SELECT COUNT(DISTINCT p.person_id) - counts unique individuals
- For counts: If asking "how many people/males/females" → use DISTINCT, return unique count only
- For lists (visits, entries, events): USE LINE BREAKS AND BULLET POINTS
- NEVER put list items on one paragraph line
- Do NOT use markdown (**, ##, etc) - just use plain bullet format with dashes or numbers

Example for visit list (CORRECT):
Zelaid entered 7 times in the past week:
- March 30 at 6:05 PM
- March 30 at 6:01 PM
- March 30 at 5:54 PM
- March 27 at 7:45 PM

DO NOT INCLUDE in list output:
- ❌ Video file paths (videos\2026-03-30_18-05-44.mp4)
- ❌ Frame numbers or timestamps
- ❌ Session IDs (optional - only if specifically asked)
- ✅ Person name, date, time ONLY

Example for summary (CORRECT):
John visited 5 times and spent 120 seconds here. Last seen on March 28 at 10:45 PM.

WRONG FORMAT - don't do this:
- Zelaid entered 7 times: - 30 Mar at 6:05 PM - 30 Mar at 6:01 PM - 30 Mar at 5:54 PM

CRITICAL FOR SQL:
- NEVER use reserved keywords as column aliases: when, where, time, date, order, group, select, from, etc.
- For timestamp columns, use simple names like: entry_time, occurred_at, last_seen_time, OR just omit the alias
- Example: SELECT e.timestamp (NOT: SELECT e.timestamp as when)
- Test your SQL syntax before returning it

JSON TOOL CALL FORMAT - MUST BE VALID:
- When calling execute_sql, ensure the JSON is COMPLETE and VALID
- Always close all quotes and brackets properly
- For SQL strings with quotes, use: "query": "SELECT ... WHERE person_id = 'Person_1'"
- NEVER generate incomplete JSON
- ALWAYS verify: opening quote matches closing quote, all brackets closed, valid JSON
- If unsure, use parameter binding with ? placeholders where possible"""

    messages = [{"role": "system", "content": system_msg}]
    tool_call_cache = {}  # Cache for identical tool calls
    for msg in (history or []):
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    video_event = None
    max_iterations = 8

    for i in range(max_iterations):
        print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] Iteration {i+1}: Calling Groq...", flush=True)
        try:
            print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}]   Sending request with {len(_TOOLS)} tools...", flush=True)
            resp = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=_TOOLS,
                max_tokens=1500,
                temperature=0.1
            )
            print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}]   ✓ Got response from Groq", flush=True)
        except Exception as e:
            print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] ✗ GROQ API ERROR:", flush=True)
            print(f"       {type(e).__name__}: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return {"question": question, "answer": f"API Error: {str(e)}", "audio_url": None, "video_event": None}

        assistant_msg = resp.choices[0].message
        messages.append(assistant_msg)  # required for multi-turn tool use

        # No tool calls → final text answer
        if not assistant_msg.tool_calls:
            answer = (assistant_msg.content or "").strip()
            if not answer:
                answer = "I couldn't find an answer to your question. Could you ask differently?"
            print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] ✓ Final answer (no more tool calls):", flush=True)
            print(f"       {answer}", flush=True)
            audio_url = _tts(answer) if with_tts else None
            return {"question": question, "answer": answer, "audio_url": audio_url, "video_event": video_event}

        # Execute each tool call
        for tool_call in assistant_msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Create cache key for this tool call
            cache_key = f"{name}:{json.dumps(args, sort_keys=True)}"

            print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] 🔧 TOOL CALL:", flush=True)
            print(f"       Function: {name}", flush=True)
            print(f"       Arguments: {json.dumps(args, indent=14)}", flush=True)

            # Check if this exact tool call was already made
            if cache_key in tool_call_cache:
                tool_result = tool_call_cache[cache_key]
                print(f"       [Using cached result from previous call]", flush=True)
            elif name == "execute_sql":
                print(f"       [Executing SQL query...]", flush=True)
                try:
                    results = _execute_sql_safe(args["query"])
                    print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] ✓ TOOL RESPONSE:", flush=True)
                    print(f"       Status: Success ({len(results)} rows)", flush=True)
                    print(f"       Data: {json.dumps(results, default=str, indent=14)}", flush=True)
                    tool_result = json.dumps(results, default=str)
                    tool_call_cache[cache_key] = tool_result
                except Exception as e:
                    tool_result = json.dumps({"error": str(e)})
                    print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] ✗ TOOL ERROR:", flush=True)
                    print(f"       Error: {str(e)}", flush=True)
                    tool_call_cache[cache_key] = tool_result

            elif name == "play_video":
                print(f"       [Looking up video...]", flush=True)
                frame_num = args.get("video_frame_number")
                try:
                    if frame_num is not None and isinstance(frame_num, str):
                        # Only convert if it's a valid number string
                        if frame_num.isdigit():
                            frame_num = int(frame_num)
                        else:
                            frame_num = None
                except (ValueError, AttributeError):
                    frame_num = None
                    print(f"       [Invalid frame number: {args.get('video_frame_number')}]", flush=True)

                video_data = _video_intent(
                    args.get("person_name", ""),
                    args.get("event_timestamp", ""),
                    frame_num
                )
                if video_data:
                    video_event = video_data
                    tool_result = json.dumps({"status": "found", **video_data}, default=str)
                    print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] ✓ TOOL RESPONSE:", flush=True)
                    print(f"       Status: Found", flush=True)
                    print(f"       Video URL: {video_data.get('video_url')}", flush=True)
                    print(f"       Display Name: {video_data.get('display_name')}", flush=True)
                    print(f"       Timestamp: {video_data.get('timestamp_seconds')}s", flush=True)
                else:
                    tool_result = json.dumps({"status": "not_found"})
                    print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] ✗ TOOL RESPONSE:", flush=True)
                    print(f"       Status: Video not found", flush=True)
                tool_call_cache[cache_key] = tool_result
            else:
                tool_result = json.dumps({"error": f"Unknown tool: {name}"})
                print(f"[LOOP] [{datetime.now().strftime('%H:%M:%S')}] ✗ Unknown tool: {name}", flush=True)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

    # Safety fallback
    return {"question": question, "answer": "Sorry, could not process your request.", "audio_url": None, "video_event": None}


def _tts(text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:12]
    p = TTS_DIR / f"tts_{h}.mp3"
    if not p.exists():
        gTTS(text=text, lang="en", slow=False).save(str(p))
    return f"/tts/tts_{h}.mp3"

@app.get("/tts/{filename}")
async def serve_tts(filename: str):
    p = TTS_DIR / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/mpeg")

class HistoryMsg(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[HistoryMsg] = []

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    q = req.question.strip()
    if not q: raise HTTPException(400, "Empty question")

    print(f"[QUERY] [{datetime.now().strftime('%H:%M:%S')}] Processing: {q}", flush=True)
    result = _run_tool_loop(q, req.history, with_tts=False)
    print(f"[QUERY] [{datetime.now().strftime('%H:%M:%S')}] Answer: {result['answer']}", flush=True)
    return result


# ════════════════════════════════════════════════════════════════════════════
# VOICE WEBSOCKET
# ════════════════════════════════════════════════════════════════════════════

async def _qput(q, status, msg):
    q.put({"type": "status", "status": status, "message": msg})

def _record_and_answer(stream, status_q, loop, dyn_silence: float,
                       idle_timeout: float = 30, history: list = None) -> dict:
    """Record speech then return answer dict. Returns {"idle": True} on silence timeout."""
    CHUNK        = 1280
    RATE         = 16000
    SILENCE_NEED = 12
    MIN_SPEECH   = 5

    buf = []; silence_cnt = 0; speech_cnt = 0; got_speech = False
    deadline = time.time() + idle_timeout

    while time.time() < deadline and not _shutdown_event.is_set():
        raw         = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(raw, dtype=np.int16)
        buf.append(audio_chunk.copy())
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))
        if rms >= dyn_silence:
            speech_cnt += 1; silence_cnt = 0
            if not got_speech:
                got_speech = True
                deadline = time.time() + 30  # extend once speech starts
        else:
            silence_cnt += 1
        if got_speech and silence_cnt >= SILENCE_NEED and speech_cnt >= MIN_SPEECH:
            break

    if speech_cnt < MIN_SPEECH:
        return {"idle": True}

    asyncio.run_coroutine_threadsafe(
        _qput(status_q, "processing", "Processing your question..."), loop)

    audio_data = np.concatenate(buf).astype(np.int16)
    wav = io.BytesIO()
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(RATE); wf.writeframes(audio_data.tobytes())

    # Detect language first (transcriptions returns detected language)
    tr_detect = groq_client.audio.transcriptions.create(
        file=("speech.wav", wav.getvalue()), model="whisper-large-v3", response_format="verbose_json")
    detected_lang = tr_detect.language if hasattr(tr_detect, 'language') else 'en'
    detected_lang_lower = detected_lang.lower() if detected_lang else 'en'

    # Only allow English or Urdu
    if detected_lang_lower not in ['en', 'english', 'ur', 'urdu']:
        raise Exception(f"Language '{detected_lang}' not supported. Only English and Urdu allowed.")

    # If Urdu, translate to English; if English, use as-is
    if detected_lang_lower in ['ur', 'urdu']:
        tr = groq_client.audio.translations.create(
            file=("speech.wav", wav.getvalue()), model="whisper-large-v3", response_format="text")
        question = (tr if isinstance(tr, str) else tr.text).strip()
    else:
        # Already English
        tr = groq_client.audio.transcriptions.create(
            file=("speech.wav", wav.getvalue()), model="whisper-large-v3", response_format="text")
        question = (tr if isinstance(tr, str) else tr.text).strip()

    # ── Process query with tool use ──────────────────────────────────
    print(f"[VOICE] [{datetime.now().strftime('%H:%M:%S')}] Processing: {question}", flush=True)
    result = _run_tool_loop(question, history or [], with_tts=True)
    print(f"[VOICE] [{datetime.now().strftime('%H:%M:%S')}] Answer: {result['answer']}", flush=True)
    return result


def _voice_sync(status_q: queue.Queue, loop) -> dict:
    """Single-shot: wake word → one Q&A → done."""
    RATE = 16000; CHUNK = 1280; WAKE_THRESH = 0.5
    pa = _pyaudio.PyAudio(); stream = None
    try:
        model  = WakeWordModel(wakeword_models=["hey_jarvis_v0.1.onnx"], inference_framework="onnx")
        # Find the first available input device that supports mono
        device_index = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] >= 1:
                device_index = i
                break
        if device_index is None:
            device_index = None  # Use default
        stream = pa.open(rate=RATE, channels=1, format=_pyaudio.paInt16,
                         input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
        asyncio.run_coroutine_threadsafe(
            _qput(status_q, "wake_word_listening", "Listening for 'Hey Jarvis'..."), loop)
        deadline = time.time() + 90
        while time.time() < deadline and not _shutdown_event.is_set():
            raw = stream.read(CHUNK, exception_on_overflow=False)
            scores = model.predict(np.frombuffer(raw, dtype=np.int16))
            if any(s > WAKE_THRESH for s in scores.values()):
                break
        else:
            return {"error": "Wake word not detected within 90 s"}
        asyncio.run_coroutine_threadsafe(
            _qput(status_q, "listening_query", "Hey Jarvis! Ask your question..."), loop)
        result = _record_and_answer(stream, status_q, loop, dyn_silence=200)
        if result.get("idle"):
            return {"error": "No speech detected after wake word"}
        return result
    except Exception as e:
        logger.error(f"Voice error: {e}", exc_info=True); return {"error": str(e)}
    finally:
        if stream:
            try: stream.stop_stream(); stream.close()
            except: pass
        pa.terminate()


def _voice_sync_continuous(status_q: queue.Queue, loop) -> None:
    """Continuous: wake word once → loop Q&A → idle 20 s → back to wake word → repeat."""
    RATE = 16000; CHUNK = 1280; WAKE_THRESH = 0.5
    IDLE_TIMEOUT = 20
    pa = _pyaudio.PyAudio(); stream = None
    try:
        # Find the first available input device that supports mono
        device_index = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] >= 1:
                device_index = i
                break
        stream = pa.open(rate=RATE, channels=1, format=_pyaudio.paInt16,
                         input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
        history = []
        while not _shutdown_event.is_set():
            # ── Wake word phase ──────────────────────────────────────────────
            model = WakeWordModel(wakeword_models=["hey_jarvis_v0.1.onnx"], inference_framework="onnx")
            asyncio.run_coroutine_threadsafe(
                _qput(status_q, "wake_word_listening", "Listening for 'Hey Jarvis'..."), loop)
            deadline = time.time() + 3600  # wait indefinitely (1h max)
            detected = False
            while time.time() < deadline and not _shutdown_event.is_set():
                raw = stream.read(CHUNK, exception_on_overflow=False)
                scores = model.predict(np.frombuffer(raw, dtype=np.int16))
                if any(s > WAKE_THRESH for s in scores.values()):
                    detected = True; break
            if not detected or _shutdown_event.is_set():
                break
            # ── Conversation loop ────────────────────────────────────────────
            asyncio.run_coroutine_threadsafe(
                _qput(status_q, "listening_query", "Hey Jarvis! Ask your question..."), loop)
            while not _shutdown_event.is_set():
                result = _record_and_answer(stream, status_q, loop,
                                            dyn_silence=200, idle_timeout=IDLE_TIMEOUT,
                                            history=history)
                if result.get("idle"):
                    # 20 s silence — break inner loop → go back to wake word
                    asyncio.run_coroutine_threadsafe(
                        _qput(status_q, "wake_word_listening", "Session ended. Listening for 'Hey Jarvis'..."), loop)
                    break
                # Push answer into queue so WS handler forwards it
                history.append({"role": "user", "content": result["question"]})
                history.append({"role": "assistant", "content": result["answer"]})
                if len(history) > 20: history[:] = history[-20:]
                status_q.put({"type": "answer", "continuous": True, **result})
                # Brief pause so TTS starts before we record again
                time.sleep(1.5)
                if _shutdown_event.is_set(): break
                asyncio.run_coroutine_threadsafe(
                    _qput(status_q, "listening_query", "Ask your next question..."), loop)
    except Exception as e:
        logger.error(f"Continuous voice error: {e}", exc_info=True)
        status_q.put({"type": "error", "error": str(e)})
    finally:
        if stream:
            try: stream.stop_stream(); stream.close()
            except: pass
        pa.terminate()


@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    loop     = asyncio.get_event_loop()
    status_q = queue.Queue()
    # Check if client wants continuous mode
    try:
        init     = await asyncio.wait_for(ws.receive_text(), timeout=2.0)
        continuous = json.loads(init).get("continuous", False)
    except Exception:
        continuous = False

    async def drain():
        while True:
            try: await ws.send_text(json.dumps(status_q.get_nowait()))
            except queue.Empty: break

    try:
        if continuous:
            future = loop.run_in_executor(None, _voice_sync_continuous, status_q, loop)
            while not future.done():
                await drain(); await asyncio.sleep(0.05)
            await drain()
        else:
            future = loop.run_in_executor(None, _voice_sync, status_q, loop)
            while not future.done():
                await drain(); await asyncio.sleep(0.05)
            await drain()
            result = await future
            await ws.send_text(json.dumps({"type": "error" if "error" in result else "answer", **result}))
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except: pass


# ════════════════════════════════════════════════════════════════════════════
# FRONTEND
# ════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def frontend():
    p = PROJECT_ROOT / "frontend" / "index.html"
    if not p.exists(): raise HTTPException(503, "Frontend missing")
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
