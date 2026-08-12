"""
Memory Agent — Persistent Cross-Transaction Intelligence.

The Memory Agent gives the multi-agent system a "long-term memory". It:
  1. Stores every investigation result in a local SQLite database.
  2. On each new investigation, retrieves the customer's history (last 5 cases).
  3. Provides the Supervisor with temporal pattern intelligence:
     - "This customer was held 3 times in the last 7 days" → escalate
     - "Previous investigations found this customer clean" → lower weight on weak signals
     - "Customer was BLOCKED last month but is transacting again" → critical flag

This is a core property of autonomous agents: they maintain state across interactions.
"""
from __future__ import annotations
import os
import json
import sqlite3
import asyncio
from datetime import datetime, timezone
from threading import Lock

# ── Database Setup ────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "memory.db")
_db_lock = Lock()


def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode for concurrent read safety."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the investigations table if it doesn't exist."""
    with _db_lock:
        conn = _get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                risk_score  REAL NOT NULL,
                action      TEXT NOT NULL,
                flags       TEXT NOT NULL,  -- JSON array
                explanation TEXT NOT NULL,
                tx_amount   REAL,
                tx_type     TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_customer ON investigations(customer_id, timestamp)")
        conn.commit()
        conn.close()


# Initialize on import
init_db()


# ── Write ─────────────────────────────────────────────────────────────────────

def store_investigation(
    customer_id: str,
    risk_score: float,
    action: str,
    flags: list[str],
    explanation: str,
    tx_amount: float = 0.0,
    tx_type: str = "",
):
    """Persist an investigation result to SQLite (thread-safe)."""
    with _db_lock:
        try:
            conn = _get_connection()
            conn.execute(
                """
                INSERT INTO investigations
                    (customer_id, timestamp, risk_score, action, flags, explanation, tx_amount, tx_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer_id,
                    datetime.now(timezone.utc).isoformat(),
                    risk_score,
                    action,
                    json.dumps(flags),
                    explanation,
                    tx_amount,
                    tx_type,
                ),
            )
            conn.commit()
            conn.close()
            print(f"[MemoryAgent] Stored investigation for {customer_id} → {action}")
        except Exception as e:
            print(f"[MemoryAgent] Failed to store investigation: {e}")


# ── Read ──────────────────────────────────────────────────────────────────────

def get_customer_history(customer_id: str, limit: int = 5) -> list[dict]:
    """
    Retrieve the most recent investigations for a customer.
    Returns a list of dicts sorted newest-first.
    """
    with _db_lock:
        try:
            conn = _get_connection()
            cursor = conn.execute(
                """
                SELECT customer_id, timestamp, risk_score, action, flags, explanation, tx_amount, tx_type
                FROM investigations
                WHERE customer_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (customer_id, limit),
            )
            rows = cursor.fetchall()
            conn.close()
            results = []
            for row in rows:
                results.append({
                    "customer_id": row["customer_id"],
                    "timestamp": row["timestamp"],
                    "risk_score": row["risk_score"],
                    "action": row["action"],
                    "flags": json.loads(row["flags"]),
                    "explanation": row["explanation"],
                    "tx_amount": row["tx_amount"],
                    "tx_type": row["tx_type"],
                })
            return results
        except Exception as e:
            print(f"[MemoryAgent] Failed to retrieve history: {e}")
            return []


def format_history_for_prompt(history: list[dict]) -> str:
    """Format customer investigation history into a readable prompt snippet."""
    if not history:
        return "No prior investigations found for this customer. This is their first review."

    lines = [f"Customer has {len(history)} prior investigation(s) on record:"]
    for i, h in enumerate(history, 1):
        ts = h["timestamp"][:10]  # Just the date
        flags_summary = h["flags"][0] if h["flags"] else "no specific flags"
        lines.append(
            f"  [{i}] {ts} — Action: {h['action']} | "
            f"Risk: {h['risk_score']:.1f}% | "
            f"Amount: ${h['tx_amount']:,.2f} ({h['tx_type']}) | "
            f"Primary flag: {flags_summary}"
        )

    # Synthesize a pattern warning
    actions = [h["action"].lower() for h in history]
    if any("block" in a for a in actions):
        lines.append(
            "\n⚠️  MEMORY ALERT: This customer has a PRIOR BLOCK on record. "
            "Re-appearance after block is a strong indicator of persistent fraud attempt."
        )
    elif sum(1 for a in actions if "hold" in a or "review" in a) >= 2:
        lines.append(
            "\n⚠️  MEMORY ALERT: Repeated HOLD decisions suggest a pattern of borderline behavior "
            "that may warrant escalation on this investigation."
        )

    return "\n".join(lines)


# ── Async Wrapper for Workflow ────────────────────────────────────────────────

async def run_memory_agent(customer_id: str) -> dict:
    """
    Async wrapper so the Memory Agent can run in parallel with other agents
    via asyncio.gather inside the LlamaIndex workflow.
    """
    loop = asyncio.get_event_loop()
    history = await loop.run_in_executor(None, get_customer_history, customer_id)
    formatted = format_history_for_prompt(history)
    return {
        "history": history,
        "formatted_context": formatted,
        "prior_investigation_count": len(history),
        "was_previously_blocked": any("block" in h["action"].lower() for h in history),
        "repeated_holds": sum(1 for h in history if "hold" in h["action"].lower() or "review" in h["action"].lower()),
    }
