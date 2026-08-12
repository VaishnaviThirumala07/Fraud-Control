import os
import shutil
import tempfile

import pandas as pd
import pytest

import graph_data_generator as gdg


@pytest.fixture(scope="module")
def graph_dir():
    """Runs the generator into a temp dir so this test doesn't depend on
    (or clobber) whatever is already in pipeline/data/graph/."""
    tmp = tempfile.mkdtemp()
    original_out_dir = gdg.OUT_DIR
    gdg.OUT_DIR = tmp
    try:
        gdg.generate()
        yield tmp
    finally:
        gdg.OUT_DIR = original_out_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_node_counts(graph_dir):
    users = pd.read_csv(os.path.join(graph_dir, "nodes_users.csv"))
    devices = pd.read_csv(os.path.join(graph_dir, "nodes_devices.csv"))
    merchants = pd.read_csv(os.path.join(graph_dir, "nodes_merchants.csv"))
    assert len(users) == gdg.NUM_USERS
    assert len(devices) == gdg.NUM_DEVICES + 1  # +1 for the injected fraud device
    assert len(merchants) == gdg.NUM_MERCHANTS


def test_node_ids_unique(graph_dir):
    users = pd.read_csv(os.path.join(graph_dir, "nodes_users.csv"))
    devices = pd.read_csv(os.path.join(graph_dir, "nodes_devices.csv"))
    merchants = pd.read_csv(os.path.join(graph_dir, "nodes_merchants.csv"))
    assert users["user_id"].is_unique
    assert devices["device_id"].is_unique
    assert merchants["merchant_id"].is_unique


def test_fraud_ring_is_injected(graph_dir):
    has_device = pd.read_csv(os.path.join(graph_dir, "edges_has_device.csv"))
    fraud_links = has_device[has_device["device_id"] == "DEV_FRAUD_999"]
    assert len(fraud_links) == 5, "Expected exactly 5 users linked to the injected fraud device"


def test_fraud_transactions_flow_to_single_merchant(graph_dir):
    tx = pd.read_csv(os.path.join(graph_dir, "edges_transactions.csv"))
    fraud_tx = tx[tx["is_fraud"] == 1]
    assert len(fraud_tx) == 15  # 5 fraud users x 3 transactions each
    assert fraud_tx["merchant_id"].nunique() == 1, "Fraud ring should funnel to a single merchant"
    assert (fraud_tx["amount"] >= 4000.0).all()
    assert (fraud_tx["amount"] <= 9999.0).all()


def test_legit_transactions_not_marked_fraud(graph_dir):
    tx = pd.read_csv(os.path.join(graph_dir, "edges_transactions.csv"))
    legit_tx = tx[tx["is_fraud"] == 0]
    assert len(legit_tx) == gdg.NUM_TRANSACTIONS
    assert (legit_tx["amount"] >= 5.0).all()
    assert (legit_tx["amount"] <= 500.0).all()


def test_all_transaction_user_ids_exist_in_users(graph_dir):
    users = pd.read_csv(os.path.join(graph_dir, "nodes_users.csv"))
    tx = pd.read_csv(os.path.join(graph_dir, "edges_transactions.csv"))
    assert set(tx["user_id"]).issubset(set(users["user_id"]))
