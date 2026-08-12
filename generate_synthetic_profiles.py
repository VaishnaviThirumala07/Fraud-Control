import pandas as pd
import numpy as np
from faker import Faker

# 1. Initialize
fake = Faker()
Faker.seed(42)
np.random.seed(42)

num_customers = 30000
unique_customers = [f"CUST_{i}" for i in range(num_customers)]

# 2. Generate Advanced Features
profiles = {
    "customer_id": unique_customers,
    "full_name": [fake.name() for _ in range(num_customers)],
    "account_age_days": np.random.randint(1, 3650, size=num_customers),
    "kyc_status": np.random.choice(["Verified", "Pending", "Failed"], size=num_customers, p=[0.90, 0.08, 0.02]),
    "pep_status": np.random.choice([True, False], size=num_customers, p=[0.01, 0.99]),
    "declared_monthly_income": np.round(np.random.lognormal(mean=8.0, sigma=1.0, size=num_customers), 2),

    "device_id_count": np.random.choice([1, 2, 3, 10], size=num_customers, p=[0.65, 0.30, 0.04, 0.01]),

    "unique_counterparties_30d": np.random.lognormal(mean=2.0, sigma=0.8, size=num_customers).astype(int),

    
    "shared_ip_count": np.random.choice([1, 2, 5, 20], size=num_customers, p=[0.80, 0.15, 0.04, 0.01]),

    "historical_avg_tx_amount": np.round(np.random.lognormal(mean=5.0, sigma=1.5, size=num_customers), 2),

    "session_velocity_seconds": np.random.choice(
        [np.random.randint(1, 5), np.random.randint(30, 120)],
        size=num_customers,
        p=[0.05, 0.95]
    ),

    "recent_failed_logins": np.random.choice([0, 1, 5, 10], size=num_customers, p=[0.85, 0.10, 0.04, 0.01])
}

customer_df = pd.DataFrame(profiles)

# 3. Derive the Risk Tier dynamically based on the complex new features
def calculate_advanced_risk(row):
    if row['shared_ip_count'] >= 5 or row['unique_counterparties_30d'] > 30:
        return "High (Network Risk)"

    elif row['recent_failed_logins'] >= 5 and row['session_velocity_seconds'] < 5:
        return "High (ATO Risk)"

    elif row['pep_status'] == True or row['kyc_status'] == "Failed":
        return "Medium (Compliance)"

    else:
        return "Low"

customer_df['risk_tier'] = customer_df.apply(calculate_advanced_risk, axis=1)

out_path = f"advanced_synthetic_profiles_{num_customers}.csv"
customer_df.to_csv(out_path, index=False)
print(f"Advanced synthetic dataset generated: {out_path}")
