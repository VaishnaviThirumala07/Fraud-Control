import os
import pandas as pd
from neo4j import GraphDatabase

from dotenv import load_dotenv

sys_path_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(sys_path_dir, ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fraudcontrol")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "graph")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} does not exist. Run graph_data_generator.py first.")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("Wiping existing database...")
        session.run("MATCH (n) DETACH DELETE n")

        print("Loading Users...")
        users_df = pd.read_csv(os.path.join(DATA_DIR, "nodes_users.csv"))
        session.run("""
            UNWIND $rows AS row
            CREATE (:User {id: row.user_id, name: row.name, kyc_status: row.kyc_status, risk_tier: row.risk_tier})
        """, parameters={'rows': users_df.to_dict('records')})

        print("Loading Devices...")
        devices_df = pd.read_csv(os.path.join(DATA_DIR, "nodes_devices.csv"))
        session.run("""
            UNWIND $rows AS row
            CREATE (:Device {id: row.device_id, ip_address: row.ip_address, os: row.os})
        """, parameters={'rows': devices_df.to_dict('records')})

        print("Loading Merchants...")
        merchants_df = pd.read_csv(os.path.join(DATA_DIR, "nodes_merchants.csv"))
        session.run("""
            UNWIND $rows AS row
            CREATE (:Merchant {id: row.merchant_id, name: row.merchant_name, category: row.category})
        """, parameters={'rows': merchants_df.to_dict('records')})

        print("Loading HAS_DEVICE edges...")
        has_device_df = pd.read_csv(os.path.join(DATA_DIR, "edges_has_device.csv"))
        session.run("""
            UNWIND $rows AS row
            MATCH (u:User {id: row.user_id})
            MATCH (d:Device {id: row.device_id})
            CREATE (u)-[:HAS_DEVICE]->(d)
        """, parameters={'rows': has_device_df.to_dict('records')})

        print("Loading TRANSACTED_WITH edges...")
        tx_df = pd.read_csv(os.path.join(DATA_DIR, "edges_transactions.csv"))
        session.run("""
            UNWIND $rows AS row
            MATCH (u:User {id: row.user_id})
            MATCH (m:Merchant {id: row.merchant_id})
            CREATE (u)-[:TRANSACTED_WITH {tx_id: row.tx_id, amount: toFloat(row.amount), timestamp: row.timestamp, is_fraud: toInteger(row.is_fraud)}]->(m)
        """, parameters={'rows': tx_df.to_dict('records')})

    driver.close()
    print("Graph data loaded successfully into Neo4j!")

if __name__ == "__main__":
    main()
