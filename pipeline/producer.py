import time
import json
import pandas as pd
try:
    from confluent_kafka import Producer
except ImportError:
    Producer = None
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
CLOUDAMQP_URL = os.getenv("CLOUDAMQP_URL")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions-in")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def main():
    rabbitmq_connection = None
    rabbitmq_channel = None
    producer = None
    
    if CLOUDAMQP_URL:
        import pika
        print(f"Connecting to RabbitMQ/CloudAMQP...")
        try:
            params = pika.URLParameters(CLOUDAMQP_URL)
            rabbitmq_connection = pika.BlockingConnection(params)
            rabbitmq_channel = rabbitmq_connection.channel()
            rabbitmq_channel.queue_declare(queue=TOPIC, durable=True)
            print("Connected to RabbitMQ/CloudAMQP successfully!")
        except Exception as e:
            print(f"Failed to connect to RabbitMQ: {e}")
            return
    else:
        if Producer is None:
            print("confluent-kafka is not installed and CLOUDAMQP_URL is not set.")
            return
        conf = {'bootstrap.servers': KAFKA_BROKER}
        try:
            producer = Producer(**conf)
        except Exception as e:
            print(f"Failed to connect to Kafka at {KAFKA_BROKER}: {e}")
            return

    print("Loading data for simulation...")
    tx_path = os.path.join(DATA_DIR, "aligned_transactions.csv")
    prof_path = os.path.join(DATA_DIR, "customer_profiles.csv")
    
    if not os.path.exists(tx_path):
        print(f"Missing data file: {tx_path}")
        if rabbitmq_connection:
            rabbitmq_connection.close()
        return
        
    tx_df = pd.read_csv(tx_path)
    profiles_df = pd.read_csv(prof_path).set_index("customer_id")
    
    print(f"Starting to stream transactions to '{TOPIC}'...")
    
    try:
        for _, tx_row in tx_df.iterrows():
            cust_id = tx_row['customer_id']
            
            # Handle nan gracefully by replacing with None
            tx_dict = tx_row.to_dict()
            tx_dict = {k: (None if pd.isna(v) else v) for k, v in tx_dict.items()}
            
            if cust_id not in profiles_df.index:
                continue
                
            profile_row = profiles_df.loc[cust_id]
            prof_dict = profile_row.to_dict()
            prof_dict["customer_id"] = cust_id
            prof_dict = {k: (None if pd.isna(v) else v) for k, v in prof_dict.items()}
            
            payload = {
                "transaction": tx_dict,
                "profile": prof_dict
            }
            
            try:
                if rabbitmq_channel:
                    rabbitmq_channel.basic_publish(
                        exchange='',
                        routing_key=TOPIC,
                        body=json.dumps(payload).encode('utf-8'),
                        properties=pika.BasicProperties(
                            delivery_mode=2 # persistent
                        )
                    )
                    print(f"Published TX to RabbitMQ/CloudAMQP for customer {cust_id} (Amount: ${tx_dict['amount']})")
                else:
                    producer.produce(
                        TOPIC, 
                        value=json.dumps(payload).encode('utf-8'),
                        callback=delivery_report
                    )
                    producer.poll(0) # Serve delivery callback queue
                    print(f"Published TX for customer {cust_id} (Amount: ${tx_dict['amount']})")
            except Exception as e:
                print(f"Failed to publish: {e}")
                
            time.sleep(1) # Stream 1 transaction every 25 seconds to avoid Gemini rate limits
    finally:
        if rabbitmq_connection:
            rabbitmq_connection.close()

if __name__ == "__main__":
    main()
