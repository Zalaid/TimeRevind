# TimeRevind Demo Notification System

## Overview
This system demonstrates the notification feature without needing live video capture:
- Takes any video file as input
- Assumes video starts at **3:59:30 PM**
- Sends notifications when non-excluded persons are detected after **4:00 PM**
- Uses **6-second identity verification** before sending (checks every 2 seconds × 3 checks)

## Files Created

### 1. **Database Table: `notification_logs`**
Location: `db/timerevind.db`

Fields:
- `person_id` — Name of detected person (e.g., "shiza", "jilani")
- `simulated_time` — Time at detection (e.g., "16:05:23")
- `video_frame` — Frame number in video
- `entry_type` — "ENTRY" or "RE_ENTRY"
- `verification_checks` — JSON array of 3 identity checks (e.g., `["shiza", "shiza", "shiza"]`)
- `verified_as` — Final confirmed person_id
- `notified` — 1 if notification sent, 0 if skipped
- `reason_skipped` — Why notification wasn't sent:
  - `BEFORE_4PM` — detected before 4:00 PM
  - `EXCLUDED` — person is jilani or malaika
  - `UNCLEAR` — verification checks disagreed

### 2. **Notification Sender: `src/core/notifier.py`**
Sends push notifications via Pushover API.

Requires `.env`:
```
PUSHOVER_API_KEY=aembc711aahh5bs7fikj38i71qxzca
PUSHOVER_USER=u6oexwt2cborog21wdtqhxzn5gj37p
```

### 3. **Demo Script: `notify_demo.py`**
Main script that processes video and handles notification logic.

## How It Works

### Time Simulation
```
Video Frame 0   → Simulated Time: 3:59:30
Video Frame 30  → Simulated Time: 4:00:00 (at 30fps)
Video Frame 60  → Simulated Time: 4:00:02
...and so on
```

### 6-Second Verification
When person detected after 4:00 PM:
1. **Frame N**: Person detected → Start verification → Check #1 ✓
2. **Frame N+60** (2 sec later @ 30fps): Re-identify → Check #2 ✓
3. **Frame N+120** (4 sec later): Re-identify → Check #3 ✓
4. **Frame N+180** (6 sec later): Evaluate results
   - All 3 checks = same person → Decision made
   - If person NOT jilani/malaika → **SEND NOTIFICATION** 📱
   - If person IS jilani/malaika → **SKIP** (excluded)
   - If checks mixed → **SKIP** (unclear identity)

### Re-Entry Notification (Option B2)
If person exits frame then comes back after 4 PM:
- Reset verification state
- Run 6-second verification again
- If confirmed non-excluded → Send NEW notification
- Entry logged as `RE_ENTRY` type

## Running the Demo

### Basic Usage
```bash
python notify_demo.py --video path/to/demo_video.mp4
```

### With Options
```bash
# Dry-run (log only, don't send notifications)
python notify_demo.py --video demo.mp4 --dry-run

# Custom start time (optional, default is 3:59:30)
python notify_demo.py --video demo.mp4 --start-time "15:59:30"
```

## Example Video for Demo

Record a **1-2 minute video** showing:
- **0-30 seconds**: Different people in frame → No notifications (before 4 PM)
- **30-60 seconds**: People still in frame (after 4 PM) → Notifications sent to phone for non-excluded
- **60+ seconds** (optional): Someone exits and re-enters → New RE_ENTRY notification

## Console Output Example

```
[15:59:31] Frame 30  | shiza detected | START verify (1/3)
[15:59:33] Frame 90  | shiza verified | progress (2/3) | distance=0.3219
[15:59:35] Frame 150 | shiza verified | progress (3/3) | distance=0.3145
[16:00:05] Frame 210 | shiza | verification COMPLETE
         Checks: ['shiza', 'shiza', 'shiza']
         ✅ CONFIRMED & NOTIFIED
[16:00:08] Frame 270 | jilani detected | START verify (1/3)
[16:00:10] Frame 330 | jilani verified | progress (2/3) | distance=0.2737
[16:00:12] Frame 390 | jilani verified | progress (3/3) | distance=0.2801
[16:00:12] Frame 390 | jilani | verification COMPLETE
         Checks: ['jilani', 'jilani', 'jilani']
         ❌ EXCLUDED (person in exclude list)
```

## What Gets Sent via Pushover

**Example notification on your phone:**

```
TimeRevind Alert
⚠️ Alert: shiza entered at 16:00:05
```

or on re-entry:

```
TimeRevind Alert
⚠️ Alert: shiza re-entered at 16:02:15
```

## Database Queries

### View all notifications from demo
```sql
SELECT person_id, simulated_time, entry_type, verified_as, notified, reason_skipped
FROM notification_logs
ORDER BY simulated_time DESC;
```

### View only sent notifications
```sql
SELECT person_id, simulated_time, entry_type
FROM notification_logs
WHERE notified = 1
ORDER BY simulated_time DESC;
```

### View skipped/rejected notifications
```sql
SELECT person_id, simulated_time, reason_skipped
FROM notification_logs
WHERE notified = 0
ORDER BY simulated_time DESC;
```

## Demo for External Evaluation

### During Presentation
1. **Show code**: Review `notify_demo.py` verification logic
2. **Show database**: Query `notification_logs` table with historical data
3. **Run script**: Execute on 1-2 min test video
4. **Show phone**: Display Pushover notifications arriving in real-time
5. **Explain logic**: Walk through 6-second verification process

### What It Demonstrates
✅ Real face embedding + Qdrant matching (reuses existing system)  
✅ Time-based filtering (only after 4 PM)  
✅ Person exclusion (jilani/malaika skip notifications)  
✅ Identity verification (6-second window prevents false positives)  
✅ Database logging (persists all decisions)  
✅ Push notifications (real alerts to phone)  
✅ Re-entry handling (notifies again if person comes back)

## Troubleshooting

### "YOLO model not found"
```bash
python -m pip install ultralytics
```

### "Pushover notification failed"
- Check `.env` has `PUSHOVER_API_KEY` and `PUSHOVER_USER`
- Verify credentials are correct
- Try with `--dry-run` to check logic without sending

### "No faces detected in video"
- Use a video with clear face visibility
- Check face confidence threshold (default: 0.3)
- Use closer shots of people

### "Database locked"
- Close any other connections to `db/timerevind.db`
- Restart the script

## Notes

- Video can be **any length**; logic is based on frame count, not actual time
- System handles **multiple people simultaneously** in same frame
- **Excluded persons** (`jilani`, `malaika`) never trigger notifications
- **Identity verification** is strict: requires all 3 checks to agree
- **Re-entry** resets verification state (person must be re-verified)
