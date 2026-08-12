"""
Bridges Dataset A (real PaySim transactions) and Dataset B (synthetic
customer/KYC profiles) as described in the MVP plan: "one original
dataset (PaySim) and one synthetic simulated dataset (customer
profiles)".

PaySim's nameOrig is (almost) 1:1 with transactions -- there's no
real repeat-customer history to mine -- so instead of treating it as
a join key into a pre-existing customer table, we draw a stratified
sample of transactions (oversampling the rare fraud class so the demo
has enough positive cases) and GENERATE one synthetic KYC/behavioral
profile per sampled nameOrig. This keeps the join key real (an actual
PaySim account ID) while the profile fields themselves are synthetic,
matching the two-dataset MVP architecture.

Outputs (into pipeline/data/):
  aligned_transactions.csv  - sampled PaySim rows + customer_id column
  customer_profiles.csv     - one synthetic profile per customer_id
"""
import os
import numpy as np
import pandas as pd
from faker import Faker

PAYSIM_PATH = os.path.expanduser("~/Downloads/paysim.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

N_LEGIT = 4500
N_FRAUD = 500  
SEED = 42


def sample_transactions():
    print(f"Loading {PAYSIM_PATH} ...")
    df = pd.read_csv(PAYSIM_PATH)

    fraud = df[df["isFraud"] == 1].sample(n=N_FRAUD, random_state=SEED)
    legit = df[df["isFraud"] == 0].sample(n=N_LEGIT, random_state=SEED)
    sample = pd.concat([fraud, legit], ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

    sample = sample.rename(columns={"nameOrig": "customer_id"})
    print(f"Sampled {len(sample)} transactions ({sample['isFraud'].sum()} fraud, "
          f"{len(sample) - sample['isFraud'].sum()} legit)")
    return sample


def generate_profiles(customer_ids, fraud_flags, fake, rng):
    """One synthetic profile per customer_id. Fraud-linked accounts are
    given a moderately elevated (not deterministic) chance of risky
    attributes -- real fraud rings don't always trip every KYC flag,
    but they skew that way, which is what makes the Customer Agent's
    cross-reference meaningful rather than a trivial giveaway."""
    n = len(customer_ids)

    kyc_status = np.where(
        fraud_flags & (rng.random(n) < 0.35),
        rng.choice(["Pending", "Failed"], size=n, p=[0.6, 0.4]),
        rng.choice(["Verified", "Pending", "Failed"], size=n, p=[0.90, 0.08, 0.02]),
    )
    pep_status = np.where(
        fraud_flags & (rng.random(n) < 0.10),
        True,
        rng.choice([True, False], size=n, p=[0.01, 0.99]),
    )
    device_id_count = np.where(
        fraud_flags & (rng.random(n) < 0.30),
        rng.choice([3, 10], size=n, p=[0.5, 0.5]),
        rng.choice([1, 2, 3, 10], size=n, p=[0.65, 0.30, 0.04, 0.01]),
    )
    unique_counterparties_30d = np.where(
        fraud_flags & (rng.random(n) < 0.25),
        rng.integers(25, 60, size=n),
        rng.lognormal(mean=2.0, sigma=0.8, size=n).astype(int),
    )
    shared_ip_count = np.where(
        fraud_flags & (rng.random(n) < 0.25),
        rng.choice([5, 20], size=n, p=[0.6, 0.4]),
        rng.choice([1, 2, 5, 20], size=n, p=[0.80, 0.15, 0.04, 0.01]),
    )
    recent_failed_logins = np.where(
        fraud_flags & (rng.random(n) < 0.20),
        rng.choice([5, 10], size=n, p=[0.6, 0.4]),
        rng.choice([0, 1, 5, 10], size=n, p=[0.85, 0.10, 0.04, 0.01]),
    )
    is_bot_session = rng.random(n) < 0.05
    session_velocity_seconds = np.where(
        is_bot_session,
        rng.integers(1, 5, size=n),
        rng.integers(30, 120, size=n),
    )

    profiles = pd.DataFrame({
        "customer_id": customer_ids,
        "full_name": [fake.name() for _ in range(n)],
        "account_age_days": rng.integers(1, 3650, size=n),
        "kyc_status": kyc_status,
        "pep_status": pep_status,
        "declared_monthly_income": np.round(rng.lognormal(mean=8.0, sigma=1.0, size=n), 2),
        "device_id_count": device_id_count,
        "unique_counterparties_30d": unique_counterparties_30d,
        "shared_ip_count": shared_ip_count,
        "historical_avg_tx_amount": np.round(rng.lognormal(mean=5.0, sigma=1.5, size=n), 2),
        "session_velocity_seconds": session_velocity_seconds,
        "recent_failed_logins": recent_failed_logins,
    })

    def calc_risk_tier(row):
        if row["shared_ip_count"] >= 5 or row["unique_counterparties_30d"] > 30:
            return "High (Network Risk)"
        elif row["recent_failed_logins"] >= 5 and row["session_velocity_seconds"] < 5:
            return "High (ATO Risk)"
        elif row["pep_status"] or row["kyc_status"] == "Failed":
            return "Medium (Compliance)"
        return "Low"

    profiles["risk_tier"] = profiles.apply(calc_risk_tier, axis=1)
    return profiles


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    Faker.seed(SEED)
    fake = Faker()
    rng = np.random.default_rng(SEED)

    tx = sample_transactions()
    profiles = generate_profiles(
        tx["customer_id"].values, tx["isFraud"].astype(bool).values, fake, rng
    )

    tx_path = os.path.join(OUT_DIR, "aligned_transactions.csv")
    profiles_path = os.path.join(OUT_DIR, "customer_profiles.csv")
    tx.to_csv(tx_path, index=False)
    profiles.to_csv(profiles_path, index=False)

    print(f"Wrote {tx_path} ({len(tx)} rows)")
    print(f"Wrote {profiles_path} ({len(profiles)} rows)")
    print("\nRisk tier distribution among sampled customers:")
    print(profiles["risk_tier"].value_counts())


if __name__ == "__main__":
    main()
