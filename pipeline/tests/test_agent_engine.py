import asyncio
import os

import pydantic
import pytest

from agent_engine import (
    TransactionData,
    CustomerData,
    FraudInvestigationWorkflow,
    investigate,
)

requires_gemini = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="No GEMINI_API_KEY/GOOGLE_API_KEY set in environment or .env",
)


def make_transaction(**overrides):
    base = dict(
        customer_id="C1", type="TRANSFER", amount=1000.0,
        oldbalanceOrg=1000.0, oldbalanceDest=0.0, risk_score=95.0, ml_flagged=True,
    )
    base.update(overrides)
    return TransactionData(**base)


def make_customer(**overrides):
    base = dict(
        customer_id="C1", kyc_status="Verified", pep_status=False, risk_tier="Low",
        account_age_days=500, unique_counterparties_30d=3, shared_ip_count=1,
        recent_failed_logins=0, session_velocity_seconds=45, historical_avg_tx_amount=200.0,
    )
    base.update(overrides)
    return CustomerData(**base)


def test_transaction_data_valid():
    tx = make_transaction()
    assert tx.amount == 1000.0


def test_transaction_data_rejects_missing_fields():
    with pytest.raises(pydantic.ValidationError):
        TransactionData(customer_id="C1", type="TRANSFER")  # missing required fields


def test_customer_data_rejects_bad_types():
    with pytest.raises(pydantic.ValidationError):
        CustomerData(
            customer_id="C1", kyc_status="Verified", pep_status="not-a-bool",
            risk_tier="Low", account_age_days="not-a-number",
            unique_counterparties_30d=3, shared_ip_count=1,
            recent_failed_logins=0, session_velocity_seconds=45,
            historical_avg_tx_amount=200.0,
        )


def test_workflow_graph_validates():
    workflow = FraudInvestigationWorkflow(timeout=60)
    workflow.validate()  # raises if steps/events don't connect start -> stop


# ---------------------------------------------------------------------
# Live integration test -- requires a real Gemini API key and makes real
# LLM + embedding calls. Kept to a SINGLE investigate() call (rather than
# one per assertion group) because the Gemini free tier caps at 5
# requests/minute for gemini-2.5-flash -- a few workflow runs (each
# making 2-3 LLM calls) exhausts that quickly.
# ---------------------------------------------------------------------

@requires_gemini
def test_live_investigate_end_to_end():
    tx = make_transaction(
        type="TRANSFER", amount=1241118.86, oldbalanceOrg=1241118.86,
        oldbalanceDest=0.0, risk_score=77.5, ml_flagged=True,
    )
    cust = make_customer(
        kyc_status="Pending", risk_tier="High (Network Risk)",
        shared_ip_count=20, unique_counterparties_30d=38,
    )
    report = asyncio.run(investigate(tx, cust))

    # Well-formed report
    assert isinstance(report.reasons, list)
    assert isinstance(report.explanation, str)
    assert isinstance(report.policy_citation, str)
    assert report.recommended_action in {"Approve", "Hold for Review", "Block"}

    # Risk score is passed through, not re-derived by the LLM
    assert report.risk_score == 77.5

    # Correct judgment on an obviously bad transaction/customer pairing
    assert report.recommended_action != "Approve", "Should not approve an account-draining transfer from a high-risk customer"
    assert len(report.reasons) > 0
    assert len(report.explanation) > 0
    assert "1.1" in report.policy_citation or "10,000" in report.policy_citation or "high-value" in report.policy_citation.lower()
