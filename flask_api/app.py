from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import os
import json
from threading import Thread
import time
import logging

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'power_consumption'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'host': os.getenv('DB_HOST', 'db'),
    'port': os.getenv('DB_PORT', '5432')
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')
    send_real_time_data()

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')

def background_task():
    while True:
        send_real_time_data()
        socketio.sleep(2)

def check_anomaly(appliance_name, current_power):
    normal_limits = {
        'bulb': 65,
        'laptop charger': 150,  # Standardized to 150W
        'unknown': float('inf')
    }
    appliance_name = appliance_name.lower()
    limit = normal_limits.get(appliance_name, float('inf'))
    is_anomaly = current_power > limit
    logging.info(f"Anomaly check - {appliance_name}: current_power={current_power}, limit={limit}, is_anomaly={is_anomaly}")
    return is_anomaly

def send_real_time_data():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        all_appliances = get_all_appliances()
        active_appliances = []
        inactive_appliances = []
        for appliance_name in all_appliances:
            latest_reading = get_latest_power_reading(appliance_name)
            is_active = is_appliance_active(appliance_name)
            
            query_consumption = """
            SELECT SUM(appliance_power * 2/3600) as total_kwh
            FROM predictions
            WHERE appliance_name = %s
            AND timestamp >= NOW() - INTERVAL '24 hours'
            """
            cur.execute(query_consumption, (appliance_name,))
            consumption_row = cur.fetchone()
            daily_consumption = float(consumption_row['total_kwh']) if consumption_row['total_kwh'] else 0

            query_weekly = """
            SELECT DATE(timestamp) as day, 
                   SUM(appliance_power * 2/3600) as usage
            FROM predictions 
            WHERE appliance_name = %s 
            AND timestamp >= NOW() - INTERVAL '7 days'
            GROUP BY day
            ORDER BY day;
            """
            cur.execute(query_weekly, (appliance_name,))
            weekly_data = [
                {'day': row['day'].strftime('%a'), 'usage': round(float(row['usage']), 2)}
                for row in cur.fetchall()
            ]

            query_anomalies = """
            SELECT DATE(timestamp) as day, COUNT(*) as count
            FROM predictions
            WHERE appliance_name = %s
            AND timestamp >= NOW() - INTERVAL '7 days'
            AND appliance_power > CASE 
                WHEN appliance_name = 'bulb' THEN 65
                WHEN appliance_name = 'laptop charger' THEN 150
                ELSE appliance_power
            END
            GROUP BY day
            ORDER BY day;
            """
            cur.execute(query_anomalies, (appliance_name,))
            weekly_anomalies = [
                {'day': row['day'].strftime('%a'), 'count': int(row['count'])}
                for row in cur.fetchall()
            ]
            logging.info(f"{appliance_name} - Weekly anomalies: {weekly_anomalies}")

            appliance_data = {
                'name': appliance_name,
                'current_power': float(latest_reading['appliance_power']),
                'last_active': latest_reading['timestamp'].isoformat() if latest_reading['timestamp'] else None,
                'time_used': get_time_used(appliance_name),
                'cycles': get_on_off_cycles(appliance_name),
                'peak_power': get_peak_power(appliance_name),
                'status': 'on' if is_active else 'off',
                'anomaly': check_anomaly(appliance_name, float(latest_reading['appliance_power'])),
                'anomalies_today': get_anomalies_count(appliance_name),
                'consumption': str(round(daily_consumption, 3)),
                'weekly_data': weekly_data,
                'weekly_anomalies': weekly_anomalies
            }
            logging.info(f"{appliance_name} - Anomalies today: {appliance_data['anomalies_today']}")

            if is_active:
                active_appliances.append(appliance_data)
            else:
                inactive_appliances.append(appliance_data)

        socketio.emit('real_time_data', {
            'active_appliances': active_appliances,
            'inactive_appliances': inactive_appliances
        })
        logging.info(f"Emitted: {len(active_appliances)} active, {len(inactive_appliances)} inactive appliances")
    except Exception as e:
        logging.error(f"Error sending real-time data: {e}")
    finally:
        cur.close()
        conn.close()

def get_all_appliances():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        query = """
        SELECT DISTINCT appliance_name 
        FROM predictions 
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """
        cur.execute(query)
        return [row['appliance_name'] for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

def is_appliance_active(appliance_name):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = """
        SELECT EXISTS (
            SELECT 1 
            FROM predictions 
            WHERE appliance_name = %s 
            AND timestamp >= NOW() - INTERVAL '6 seconds'
            AND appliance_power > 0
        )
        """
        cur.execute(query, (appliance_name,))
        return cur.fetchone()[0]
    finally:
        cur.close()
        conn.close()

def get_latest_power_reading(appliance_name):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        query = """
        SELECT appliance_power, timestamp
        FROM predictions
        WHERE appliance_name = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """
        cur.execute(query, (appliance_name,))
        result = cur.fetchone()
        return result if result else {'appliance_power': 0, 'timestamp': None}
    finally:
        cur.close()
        conn.close()

def get_peak_power(appliance_name):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = """
        SELECT MAX(appliance_power) as peak_power
        FROM predictions
        WHERE appliance_name = %s 
        AND timestamp >= NOW() - INTERVAL '1 day';
        """
        cur.execute(query, (appliance_name,))
        result = cur.fetchone()
        return float(result[0]) if result and result[0] else 0
    finally:
        cur.close()
        conn.close()

def get_time_used(appliance_name):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = """
        SELECT COUNT(*) * 2 / 60.0 as minutes
        FROM predictions
        WHERE appliance_name = %s 
        AND appliance_power > 0 
        AND timestamp >= NOW() - INTERVAL '1 day';
        """
        cur.execute(query, (appliance_name,))
        result = cur.fetchone()
        return round(float(result[0]), 2) if result and result[0] else 0
    finally:
        cur.close()
        conn.close()

def get_on_off_cycles(appliance_name):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = """
        WITH status_changes AS (
            SELECT 
                timestamp,
                appliance_power > 0 as is_on,
                LAG(appliance_power > 0) OVER (ORDER BY timestamp) as prev_status
            FROM predictions
            WHERE appliance_name = %s
            AND timestamp >= NOW() - INTERVAL '24 hours'
        )
        SELECT COUNT(*) as cycles
        FROM status_changes
        WHERE is_on != prev_status
        AND prev_status IS NOT NULL
        AND is_on = true;
        """
        cur.execute(query, (appliance_name,))
        result = cur.fetchone()
        return int(result[0]) if result and result[0] else 0
    finally:
        cur.close()
        conn.close()

def get_anomalies_count(appliance_name):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = """
        SELECT COUNT(*) as anomaly_count
        FROM predictions
        WHERE appliance_name = %s
        AND timestamp >= CURRENT_DATE
        AND appliance_power > CASE 
            WHEN appliance_name = 'bulb' THEN 65
            WHEN appliance_name = 'laptop charger' THEN 150
            ELSE appliance_power
        END;
        """
        cur.execute(query, (appliance_name,))
        result = cur.fetchone()
        return int(result[0]) if result and result[0] else 0
    finally:
        cur.close()
        conn.close()

@app.route('/api/historical-data', methods=['GET'])
def get_historical_data():
    appliance_name = request.args.get('appliance_name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        query = """
        SELECT DATE(timestamp) as day, 
               SUM(appliance_power * 2/3600) as usage
        FROM predictions 
        WHERE appliance_name = %s 
        AND timestamp BETWEEN %s AND %s
        GROUP BY day
        ORDER BY day;
        """
        cur.execute(query, (appliance_name, start_date, end_date))
        rows = cur.fetchall()
        historical_data = [{
            'day': row['day'].strftime('%a'),
            'usage': round(float(row['usage']), 2)
        } for row in rows]
        return jsonify({
            'appliance_name': appliance_name,
            'data': historical_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/api/cost-estimation', methods=['POST'])
def get_cost_estimation():
    data = request.json
    appliances = data.get('appliances', [])
    duration = data.get('duration', 'weekly')
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        estimates = []
        total_cost = 0
        for appliance in appliances:
            query = """
            SELECT SUM(appliance_power * 2/3600) as total_consumption
            FROM predictions
            WHERE appliance_name = %s 
            AND timestamp >= NOW() - INTERVAL %s;
            """
            interval = '7 days' if duration == 'weekly' else '30 days'
            cur.execute(query, (appliance['name'], interval))
            result = cur.fetchone()
            if result and result['total_consumption']:
                consumption = float(result['total_consumption'])
                rate = 3.15
                cost = consumption * rate
                estimates.append({
                    'appliance_name': appliance['name'],
                    'consumption': round(consumption, 2),
                    'rate': rate,
                    'total_cost': round(cost, 2)
                })
                total_cost += cost
        return jsonify({
            'estimates': estimates,
            'total_cost': round(total_cost, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    Thread(target=background_task, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)