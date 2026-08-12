"""
Graph Intelligence bonus component (Step 5 / "Graph Agent" from the
design chat). PaySim and the synthetic KYC profiles are both tabular --
neither has the network structure (shared devices/IPs, mule rings)
that a Graph Neural Network or Neo4j-style query needs. This script
generates a standalone synthetic graph dataset (nodes + edges as CSVs)
with a deliberately injected fraud ring: 5 users sharing one device/IP
funneling high-value transactions to a single merchant -- the classic
pattern tabular ML misses but graph traversal catches instantly.

Out of scope for the 3-day MVP (no Neo4j/graph DB is stood up here),
but the CSVs are graph-database-ready: load nodes_*.csv as nodes and
edges_*.csv as relationships in Neo4j, Memgraph, or NetworkX.
"""
import os
import random
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "graph")

NUM_USERS = 200
NUM_DEVICES = 250
NUM_MERCHANTS = 20
NUM_TRANSACTIONS = 1000
SEED = 42


def generate():
    fake = Faker()
    Faker.seed(SEED)
    random.seed(SEED)

    # 1. NODES
    users = [
        {
            "user_id": f"U{1000 + i}",
            "name": fake.name(),
            "kyc_status": random.choices(["Verified", "Pending"], weights=[0.9, 0.1])[0],
            "risk_tier": random.choices(["Low", "Medium", "High"], weights=[0.8, 0.15, 0.05])[0],
        }
        for i in range(NUM_USERS)
    ]

    devices = [
        {
            "device_id": f"DEV{5000 + i}",
            "ip_address": fake.ipv4(),
            "os": random.choice(["iOS", "Android", "Windows", "MacOS"]),
        }
        for i in range(NUM_DEVICES)
    ]

    merchants = [
        {
            "merchant_id": f"M{100 + i}",
            "merchant_name": fake.company(),
            "category": random.choice(["Retail", "Electronics", "Travel", "Crypto Exchange"]),
        }
        for i in range(NUM_MERCHANTS)
    ]

    # 2. EDGES
    has_device = []
    for user in users:
        for dev in random.sample(devices, k=random.randint(1, 2)):
            has_device.append({"user_id": user["user_id"], "device_id": dev["device_id"]})

    start_date = datetime.now() - timedelta(days=30)
    transactions = [
        {
            "tx_id": fake.uuid4(),
            "user_id": random.choice(users)["user_id"],
            "merchant_id": random.choice(merchants)["merchant_id"],
            "amount": round(random.uniform(5.0, 500.0), 2),
            "timestamp": start_date + timedelta(minutes=random.randint(1, 40000)),
            "is_fraud": 0,
        }
        for _ in range(NUM_TRANSACTIONS)
    ]

    # 3. INJECT A FRAUD RING: 5 users sharing one device/IP, funneling
    # high-value transactions to a single merchant. Tabular models see
    # 5 unrelated accounts; a graph traversal sees one ring instantly.
    fraud_device = {"device_id": "DEV_FRAUD_999", "ip_address": "192.168.1.100", "os": "Windows"}
    devices.append(fraud_device)

    fraud_merchant = merchants[-1]
    fraud_users = users[-5:]

    for user in fraud_users:
        has_device.append({"user_id": user["user_id"], "device_id": fraud_device["device_id"]})
        for _ in range(3):
            transactions.append({
                "tx_id": fake.uuid4(),
                "user_id": user["user_id"],
                "merchant_id": fraud_merchant["merchant_id"],
                "amount": round(random.uniform(4000.0, 9999.0), 2),
                "timestamp": start_date + timedelta(minutes=random.randint(1, 40000)),
                "is_fraud": 1,
            })

    users_df = pd.DataFrame(users)
    devices_df = pd.DataFrame(devices)
    merchants_df = pd.DataFrame(merchants)
    has_device_df = pd.DataFrame(has_device)
    transactions_df = pd.DataFrame(transactions)

    os.makedirs(OUT_DIR, exist_ok=True)
    users_df.to_csv(os.path.join(OUT_DIR, "nodes_users.csv"), index=False)
    devices_df.to_csv(os.path.join(OUT_DIR, "nodes_devices.csv"), index=False)
    merchants_df.to_csv(os.path.join(OUT_DIR, "nodes_merchants.csv"), index=False)
    has_device_df.to_csv(os.path.join(OUT_DIR, "edges_has_device.csv"), index=False)
    transactions_df.to_csv(os.path.join(OUT_DIR, "edges_transactions.csv"), index=False)

    print(f"Wrote graph CSVs to {OUT_DIR}")
    print(f"  {len(users_df)} users, {len(devices_df)} devices, {len(merchants_df)} merchants")
    print(f"  {len(has_device_df)} device links, {len(transactions_df)} transactions "
          f"({transactions_df['is_fraud'].sum()} fraud)")
    print(f"  Injected fraud ring: users {[u['user_id'] for u in fraud_users]} "
          f"sharing device {fraud_device['device_id']} -> merchant {fraud_merchant['merchant_id']}")


if __name__ == "__main__":
    generate()
