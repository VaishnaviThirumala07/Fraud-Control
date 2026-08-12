import os

import numpy as np
import pandas as pd
import pytest
from faker import Faker

from align_datasets import generate_profiles, OUT_DIR

VALID_RISK_TIERS = {"Low", "Medium (Compliance)", "High (Network Risk)", "High (ATO Risk)"}


def test_generate_profiles_unit():
    """Unit test against the profile-generation logic directly, without
    touching the multi-million-row paysim.csv."""
    rng = np.random.default_rng(0)
    Faker.seed(0)
    fake = Faker()

    customer_ids = [f"C{i}" for i in range(200)]
    fraud_flags = np.array([True] * 50 + [False] * 150)

    profiles = generate_profiles(customer_ids, fraud_flags, fake, rng)

    assert len(profiles) == 200
    assert set(profiles["customer_id"]) == set(customer_ids)
    assert profiles["risk_tier"].isin(VALID_RISK_TIERS).all()
    assert profiles["kyc_status"].isin(["Verified", "Pending", "Failed"]).all()
    assert profiles["pep_status"].isin([True, False]).all()
    assert (profiles["declared_monthly_income"] > 0).all()
    assert (profiles["account_age_days"] >= 1).all()
    assert (profiles["account_age_days"] <= 3650).all()


def test_generate_profiles_no_duplicate_customer_ids():
    rng = np.random.default_rng(1)
    Faker.seed(1)
    fake = Faker()
    customer_ids = [f"C{i}" for i in range(100)]
    fraud_flags = np.zeros(100, dtype=bool)
    profiles = generate_profiles(customer_ids, fraud_flags, fake, rng)
    assert profiles["customer_id"].is_unique


def test_fraud_accounts_skew_riskier_than_legit():
    """Statistical check: fraud-linked profiles should be flagged
    non-Low risk_tier more often than legit ones (not deterministic,
    but should hold at scale)."""
    rng = np.random.default_rng(2)
    Faker.seed(2)
    fake = Faker()
    n = 4000
    customer_ids = [f"C{i}" for i in range(n)]
    fraud_flags = rng.random(n) < 0.5
    profiles = generate_profiles(customer_ids, fraud_flags, fake, rng)

    fraud_risky_rate = (profiles.loc[fraud_flags, "risk_tier"] != "Low").mean()
    legit_risky_rate = (profiles.loc[~fraud_flags, "risk_tier"] != "Low").mean()
    assert fraud_risky_rate > legit_risky_rate


DATA_DIR = OUT_DIR
TX_PATH = os.path.join(DATA_DIR, "aligned_transactions.csv")
PROFILES_PATH = os.path.join(DATA_DIR, "customer_profiles.csv")


@pytest.mark.skipif(not os.path.exists(TX_PATH) or not os.path.exists(PROFILES_PATH),
                     reason="Run align_datasets.py first to generate the demo data")
class TestGeneratedOutputFiles:
    def test_join_integrity(self):
        tx = pd.read_csv(TX_PATH)
        profiles = pd.read_csv(PROFILES_PATH)
        assert set(tx["customer_id"]).issubset(set(profiles["customer_id"]))
        assert profiles["customer_id"].is_unique

    def test_risk_tiers_valid(self):
        profiles = pd.read_csv(PROFILES_PATH)
        assert profiles["risk_tier"].isin(VALID_RISK_TIERS).all()

    def test_transaction_types_are_valid_paysim_types(self):
        tx = pd.read_csv(TX_PATH)
        valid_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
        assert tx["type"].isin(valid_types).all()

    def test_isfraud_is_binary(self):
        tx = pd.read_csv(TX_PATH)
        assert tx["isFraud"].isin([0, 1]).all()
