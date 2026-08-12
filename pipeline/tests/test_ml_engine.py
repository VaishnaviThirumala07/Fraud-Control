import os

import pandas as pd
import pytest

from ml_engine import predict_risk, load_model, FLAG_THRESHOLD

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "aligned_transactions.csv")


def base_transaction(**overrides):
    tx = dict(
        step=1, type="PAYMENT", amount=100.0,
        oldbalanceOrg=1000.0, oldbalanceDest=0.0, isFlaggedFraud=0,
    )
    tx.update(overrides)
    return tx


def test_model_loads():
    model = load_model()
    assert model is not None
    assert model.get_booster().feature_names is not None


def test_threshold_is_a_valid_probability():
    assert 0.0 < FLAG_THRESHOLD < 1.0


@pytest.mark.parametrize("tx_type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
def test_all_transaction_types_produce_valid_scores(tx_type):
    tx = base_transaction(type=tx_type)
    result = predict_risk(tx)
    assert 0.0 <= result["risk_score"] <= 100.0
    assert isinstance(result["is_flagged"], bool)
    assert result["is_flagged"] == (result["risk_score"] >= FLAG_THRESHOLD * 100)


def test_missing_isflaggedfraud_key_defaults_safely():
    tx = base_transaction()
    del tx["isFlaggedFraud"]
    result = predict_risk(tx)  # should not raise KeyError
    assert 0.0 <= result["risk_score"] <= 100.0


def test_account_draining_transfer_scores_high():
    # Classic PaySim fraud signature: TRANSFER that empties the origin account
    tx = base_transaction(type="TRANSFER", amount=50000.0, oldbalanceOrg=50000.0, oldbalanceDest=0.0)
    result = predict_risk(tx)
    assert result["risk_score"] > 50.0, f"Expected elevated risk, got {result['risk_score']}%"


def test_small_payment_scores_low():
    tx = base_transaction(type="PAYMENT", amount=25.0, oldbalanceOrg=5000.0, oldbalanceDest=200.0)
    result = predict_risk(tx)
    assert result["risk_score"] < 50.0, f"Expected low risk, got {result['risk_score']}%"


def test_zero_amount_does_not_crash():
    tx = base_transaction(amount=0.0, oldbalanceOrg=0.0, oldbalanceDest=0.0)
    result = predict_risk(tx)
    assert 0.0 <= result["risk_score"] <= 100.0


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="aligned_transactions.csv not generated yet")
def test_statistical_recall_on_aligned_sample():
    """Sanity check against the real aligned sample: the fast path should
    still catch the large majority of known fraud (matches the ~98%
    recall measured in results_realistic.txt)."""
    df = pd.read_csv(DATA_PATH)
    fraud_rows = df[df["isFraud"] == 1]
    assert len(fraud_rows) > 0

    flagged = sum(predict_risk(row.to_dict())["is_flagged"] for _, row in fraud_rows.iterrows())
    recall = flagged / len(fraud_rows)
    assert recall > 0.90, f"Recall on sampled fraud dropped to {recall:.2%}, expected >90%"
