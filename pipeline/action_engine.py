"""
Action Agent — The "Executor" of the Multi-Agent System.

After the Supervisor Reasoning Agent produces a decision, the Action Agent
autonomously carries it out:
  - Block  → Adds customer to the blocked-accounts registry, freezes all outbound transfers
  - Hold   → Generates a dynamic OTP challenge & queues for HITL human review
  - Approve → Logs an audit entry confirming clearance

New in v3 (HITL & Risk-Based Step-Up):
  - Autonomously generates a 6-digit OTP challenge for medium-risk / HELD cases.
  - Exposes REST endpoints for HITL human overrides (verify OTP, manual approve, force block).
"""
from __future__ import annotations
import os
import json
import random
from datetime import datetime, timezone

# ── In-Memory State (shared with api.py via import) ──────────────────────────

blocked_accounts: list[dict] = []
alert_feed: list[dict] = []
audit_log: list[dict] = []

# Active OTP challenges: customer_id -> { code, status, tx_amount, timestamp }
active_otp_challenges: dict[str, dict] = {}

# One current outcome per unique transaction. Unlike audit_log, this is not an
# event log: OTP/HITL follow-ups update the original transaction instead of
# counting it again.
transaction_outcomes: dict[str, str] = {}
customer_latest_transaction: dict[str, str] = {}


def register_transaction(
    transaction_id: str,
    customer_id: str,
    is_flagged: bool,
) -> None:
    """Register every fast-path transaction exactly once."""
    if transaction_id not in transaction_outcomes:
        # A flagged transaction is under review until the Action Agent replaces
        # this provisional outcome with its final decision.
        transaction_outcomes[transaction_id] = "held" if is_flagged else "approved"
    customer_latest_transaction[customer_id] = transaction_id


def _set_transaction_outcome(transaction_id: str | None, outcome: str) -> None:
    if transaction_id:
        transaction_outcomes[transaction_id] = outcome


def _upsert_blocked_account(entry: dict) -> None:
    """Keep one current blocked-account registry entry per customer."""
    blocked_accounts[:] = [
        existing for existing in blocked_accounts
        if existing.get("customer_id") != entry.get("customer_id")
    ]
    blocked_accounts.append(entry)
    if len(blocked_accounts) > 50:
        blocked_accounts.pop(0)


# ── Audit Log Persistence ─────────────────────────────────────────────────────

AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "audit_log.jsonl")

def _append_audit(entry: dict):
    """Persist audit entry to disk for durability."""
    try:
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[ActionAgent] Audit log write failed: {e}")


# ── Core Action Executor ──────────────────────────────────────────────────────

async def execute_action(
    customer_id: str,
    recommended_action: str,
    risk_score: float,
    reasons: list[str],
    sar_explanation: str,
    transaction_amount: float,
    transaction_type: str,
    transaction_id: str | None = None,
) -> dict:
    """
    The Action Agent's main entry point. Evaluates the Supervisor's recommendation
    and autonomously executes the appropriate action.
    """
    now = datetime.now(timezone.utc).isoformat()
    action_lower = recommended_action.lower()

    action_result = {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "timestamp": now,
        "recommended_action": recommended_action,
        "risk_score": risk_score,
        "executed": True,
        "action_taken": "",
        "action_description": "",
        "alert_id": None,
        "otp_challenge": None,
    }

    if "block" in action_lower:
        _set_transaction_outcome(transaction_id, "blocked")
        # ── BLOCK: Autonomously freeze account ────────────────────────────────
        block_entry = {
            "customer_id": customer_id,
            "transaction_id": transaction_id,
            "blocked_at": now,
            "risk_score": risk_score,
            "transaction_amount": transaction_amount,
            "transaction_type": transaction_type,
            "primary_reason": reasons[0] if reasons else "Multiple severe risk flags",
            "sar_summary": sar_explanation[:200] + "..." if len(sar_explanation) > 200 else sar_explanation,
            "all_flags": reasons,
            "action_by": "Autonomous Action Agent",
        }
        _upsert_blocked_account(block_entry)

        alert_id = f"ALERT-BLOCK-{len(alert_feed)+1:04d}"
        alert = {
            "id": alert_id,
            "severity": "CRITICAL",
            "type": "ACCOUNT_BLOCKED",
            "customer_id": customer_id,
            "timestamp": now,
            "message": f"Account autonomously BLOCKED. Risk: {risk_score:.1f}%. {reasons[0] if reasons else ''}",
            "details": sar_explanation,
            "acknowledged": False,
        }
        alert_feed.append(alert)
        if len(alert_feed) > 100:
            alert_feed.pop(0)

        action_result.update({
            "action_taken": "ACCOUNT_BLOCKED",
            "action_description": (
                f"Outbound transfers frozen on account {customer_id}. "
                f"Risk score {risk_score:.1f}% exceeded block threshold. "
                f"SAR draft generated for compliance review."
            ),
            "alert_id": alert_id,
        })
        print(f"[ActionAgent] 🔴 BLOCKED account {customer_id} (risk={risk_score:.1f}%)")

    elif "hold" in action_lower or "review" in action_lower:
        _set_transaction_outcome(transaction_id, "held")
        # ── HOLD: Generate OTP Challenge + Queue for Human Review ─────────────
        otp_code = f"{random.randint(100000, 999999)}"
        active_otp_challenges[customer_id] = {
            "customer_id": customer_id,
            "code": otp_code,
            "status": "PENDING_VERIFICATION",
            "tx_amount": transaction_amount,
            "timestamp": now,
            "transaction_id": transaction_id,
        }

        alert_id = f"ALERT-HOLD-{len(alert_feed)+1:04d}"
        alert = {
            "id": alert_id,
            "severity": "HIGH",
            "type": "HOLD_FOR_REVIEW",
            "customer_id": customer_id,
            "timestamp": now,
            "message": f"Transaction HELD. Risk: {risk_score:.1f}%. Sent SMS OTP Challenge {otp_code} to customer.",
            "details": sar_explanation,
            "flags": reasons,
            "acknowledged": False,
        }
        alert_feed.append(alert)
        if len(alert_feed) > 100:
            alert_feed.pop(0)

        action_result.update({
            "action_taken": "TRANSACTION_HELD",
            "action_description": (
                f"Transaction held for verification. "
                f"Autonomous Agent dispatched SMS OTP challenge ({otp_code}) to customer {customer_id}. "
                f"Awaiting customer OTP input or HITL investigator sign-off."
            ),
            "alert_id": alert_id,
            "otp_challenge": {
                "code": otp_code,
                "status": "PENDING_VERIFICATION",
                "message": f"SMS OTP Challenge {otp_code} dispatched to customer device",
            },
        })
        print(f"[ActionAgent] 🟡 HELD transaction for {customer_id} (risk={risk_score:.1f}%) -> Generated OTP {otp_code}")

    else:
        _set_transaction_outcome(transaction_id, "approved")
        # ── APPROVE: Clear transaction ────────────────────────────────────────
        action_result.update({
            "action_taken": "TRANSACTION_APPROVED",
            "action_description": (
                f"Transaction cleared by agent review. "
                f"Risk score {risk_score:.1f}% within acceptable limits. "
                f"Logged to audit trail."
            ),
        })
        print(f"[ActionAgent] ✅ APPROVED transaction for {customer_id} (risk={risk_score:.1f}%)")

    audit_entry = {**action_result, "reasons": reasons}
    audit_log.append(audit_entry)
    if len(audit_log) > 500:
        audit_log.pop(0)
    _append_audit(audit_entry)

    return action_result


# ── HITL & OTP Handlers ───────────────────────────────────────────────────────

def verify_customer_otp(customer_id: str, entered_code: str) -> dict:
    """Verify customer entered OTP for a held transaction."""
    challenge = active_otp_challenges.get(customer_id)
    if not challenge:
        return {"success": False, "message": "No active OTP challenge found for this customer."}

    if challenge["code"] == entered_code.strip():
        challenge["status"] = "VERIFIED"
        del active_otp_challenges[customer_id]
        _set_transaction_outcome(challenge.get("transaction_id"), "approved")

        hitl_entry = {
            "transaction_id": challenge.get("transaction_id"),
            "customer_id": customer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommended_action": "Approve",
            "risk_score": 0.0,
            "executed": True,
            "action_taken": "OTP_VERIFIED_RELEASED",
            "action_description": f"Customer successfully verified SMS OTP ({entered_code}). Transaction released automatically.",
            "reasons": ["SMS 2FA Challenge Verified by Customer"],
        }
        audit_log.append(hitl_entry)
        _append_audit(hitl_entry)
        print(f"[ActionAgent] 🔑 Customer {customer_id} verified OTP {entered_code}. Released!")
        return {"success": True, "action_result": hitl_entry}
    else:
        return {"success": False, "message": "Incorrect OTP code. Please try again."}


def hitl_override_action(customer_id: str, new_action: str, officer_notes: str = "") -> dict:
    """Human-In-The-Loop (HITL) compliance officer override."""
    now = datetime.now(timezone.utc).isoformat()
    action_taken = f"HITL_OVERRIDE_{new_action.upper()}"
    desc = f"Compliance Officer manually overrode decision to '{new_action}'. Notes: {officer_notes or 'None'}"
    transaction_id = customer_latest_transaction.get(customer_id)
    target_outcome = "blocked" if new_action.lower() == "block" else "approved"
    already_applied = bool(
        transaction_id and transaction_outcomes.get(transaction_id) == target_outcome
    )

    if customer_id in active_otp_challenges:
        del active_otp_challenges[customer_id]

    if new_action.lower() == "block":
        _set_transaction_outcome(transaction_id, "blocked")
        block_entry = {
            "customer_id": customer_id,
            "transaction_id": transaction_id,
            "blocked_at": now,
            "risk_score": 99.0,
            "transaction_amount": 0.0,
            "transaction_type": "HITL Manual Block",
            "primary_reason": f"Compliance officer manual block: {officer_notes}",
            "action_by": "Human Compliance Officer",
        }
        if not already_applied:
            _upsert_blocked_account(block_entry)

    else:
        _set_transaction_outcome(transaction_id, "approved")
        # An explicit human approval reverses any current account block.
        blocked_accounts[:] = [
            entry for entry in blocked_accounts
            if entry.get("customer_id") != customer_id
        ]

    override_entry = {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "timestamp": now,
        "recommended_action": new_action,
        "risk_score": 0.0,
        "executed": True,
        "action_taken": action_taken,
        "action_description": (
            f"{new_action} was already applied to this transaction."
            if already_applied else desc
        ),
        "reasons": [f"HITL Manual Override: {officer_notes or 'Officer sign-off'}"],
    }
    if not already_applied:
        audit_log.append(override_entry)
        _append_audit(override_entry)
    print(f"[ActionAgent] 👤 HITL Override for {customer_id} -> {new_action}")
    return {"success": True, "action_result": override_entry}


# ── Stats Helper ──────────────────────────────────────────────────────────────

def get_action_stats() -> dict:
    """Return aggregate statistics for the dashboard stats bar."""
    outcomes = list(transaction_outcomes.values())
    total = len(outcomes)
    blocked = outcomes.count("blocked")
    held = outcomes.count("held")
    approved = outcomes.count("approved")
    return {
        "total_processed": total,
        "blocked": blocked,
        "held": held,
        "approved": approved,
        "fraud_rate_pct": round((blocked + held) / total * 100, 1) if total > 0 else 0.0,
    }
