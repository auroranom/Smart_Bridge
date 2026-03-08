import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  SESSION LOGGER
# ─────────────────────────────────────────────

LOG_FILE = "medsafe_session_log.json"


def _load_log() -> list:
    """Loads existing log file or returns empty list."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log(data: list):
    """Saves log data to file."""
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Logging error: {e}")


def log_session_event(event_type: str, data: dict):
    """
    Logs a session event with timestamp.

    event_type examples:
    - "prescription_upload"
    - "interaction_check"
    - "symptom_analysis"
    - "risk_score"
    - "side_effect_check"
    """
    log = _load_log()
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "data": data
    }
    log.append(entry)
    _save_log(log)


def get_session_history(limit: int = 10) -> list:
    """Returns the most recent session log entries."""
    log = _load_log()
    return log[-limit:]


def clear_session_log():
    """Clears the session log."""
    _save_log([])


def get_session_summary() -> dict:
    """Returns a summary of session activity."""
    log = _load_log()
    if not log:
        return {"total_events": 0, "message": "No session activity recorded yet."}

    event_counts = {}
    for entry in log:
        event = entry.get("event", "unknown")
        event_counts[event] = event_counts.get(event, 0) + 1

    return {
        "total_events": len(log),
        "first_event": log[0]["timestamp"] if log else None,
        "last_event": log[-1]["timestamp"] if log else None,
        "event_breakdown": event_counts
    }