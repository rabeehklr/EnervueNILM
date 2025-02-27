import time
import psycopg2
from datetime import datetime
from psycopg2 import OperationalError
from flask import Flask, request, jsonify

app = Flask(__name__)

# Database connection parameters
db_params = {
    'dbname': 'power_consumption',
    'user': 'postgres',
    'password': 'password',
    'host': 'db',
    'port': '5432'
}

def wait_for_database(params, max_retries=30, retry_interval=2):
    """Wait for database to become available."""
    for attempt in range(max_retries):
        try:
            print(f"Database connection attempt {attempt + 1}/{max_retries}")
            conn = psycopg2.connect(**params)
            conn.close()
            print("Database is ready!")
            return True
        except OperationalError as e:
            print(f"Database not ready yet: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_interval} seconds before next attempt...")
                time.sleep(retry_interval)
    return False

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json
    print(f"Received data: {data}")
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        # Insert into database
        cur.execute("""
        INSERT INTO aggregate_data 
        (timestamp, total_power, voltage, current, active_power)
        VALUES (%s, %s, %s, %s, %s)
        """, (
        datetime.now(),
        float(data['total_power']),
        float(data['voltage']),
        float(data['current']),
        float(data['total_power'])  # active_power is the same as total_power
        ))
        conn.commit()
        cur.close()
        conn.close()
        print("Data inserted into database")
        return jsonify({"status": "success"}), 200
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def main():
    if not wait_for_database(db_params):
        print("Failed to connect to database after maximum retries")
        return
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()