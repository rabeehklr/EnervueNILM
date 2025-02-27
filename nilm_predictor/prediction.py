import psycopg2
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import os
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class CNNLSTMModel(nn.Module):
    def __init__(self, input_features=3, time_steps=10, num_appliances=6):
        super().__init__()
        self.time_steps = time_steps
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.status_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_appliances),
            nn.Sigmoid()
        )
        self.power_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_appliances),
            nn.ReLU()
        )
    
    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]
        status_pred = self.status_head(lstm_out)
        power_pred = self.power_head(lstm_out)
        return status_pred, power_pred

class NILMModelPredictor:
    def __init__(self, model_path, db_params, device=None, max_retries=30, retry_delay=5):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self.model = None
        self.scaler = None
        self.appliances = None
        self.db_params = db_params
        self.last_processed_timestamp = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.status_threshold = 0.5
        self.min_power_threshold = 5
        self.time_steps = 10
        self.load_model(model_path)
    
    def wait_for_database(self):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                conn = psycopg2.connect(**self.db_params)
                conn.close()
                logging.info("Successfully connected to database")
                return True
            except psycopg2.OperationalError as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    logging.warning(f"Database connection failed. Retrying in {self.retry_delay} seconds... Error: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f"Maximum retries reached.")
                    return False
    
    def load_model(self, model_path):
        try:
            logging.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.appliances = checkpoint.get('appliances')
            self.scaler = checkpoint.get('scaler_state')
            self.model = CNNLSTMModel(input_features=3, time_steps=self.time_steps, num_appliances=len(self.appliances)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logging.info("Model loaded successfully")
            logging.info(f"Appliances: {self.appliances}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_batch(self, features):
        if len(features) < self.time_steps:
            padding = np.zeros((self.time_steps - len(features), features.shape[1]))
            features = np.vstack((padding, features))
        elif len(features) > self.time_steps:
            features = features[-self.time_steps:]  # Take last time_steps
        features_normalized = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        features_tensor = features_tensor.T.unsqueeze(0)  # [1, features, time_steps]
        return features_tensor
    
    def adjust_predictions(self, status_preds, power_preds, total_power):
        status = (status_preds.cpu().numpy() > self.status_threshold).astype(int).tolist()  # Convert to Python int list
        power = power_preds.cpu().numpy()
        adjusted_power = power * status
        predicted_total = np.sum(adjusted_power)
        if predicted_total > 0 and total_power > self.min_power_threshold:
            ratio = total_power / predicted_total
            adjusted_power *= ratio
        return status, adjusted_power

    @contextmanager
    def database_connection(self):
        conn = psycopg2.connect(**self.db_params)
        try:
            yield conn
        finally:
            conn.close()

    def predict_and_store(self, target_latency=2.0):
        logging.info("Starting prediction loop with target latency of 2 seconds per row")
        
        # Persistent connection
        with self.database_connection() as conn:
            cur = conn.cursor()
            last_log_time = time.time()
            
            while True:
                try:
                    start_time = time.time()
                    
                    # Fetch one row at a time
                    query = """
                    SELECT timestamp, voltage, current, active_power
                    FROM aggregate_data
                    WHERE (timestamp > %s OR %s IS NULL)
                    ORDER BY timestamp
                    LIMIT 1
                    """
                    cur.execute(query, (self.last_processed_timestamp, self.last_processed_timestamp))
                    row = cur.fetchone()
                    
                    if not row:
                        # Minimal delay only if no data, check frequently
                        if time.time() - last_log_time > 10:  # Log every 10s
                            logging.info("No new data, checking again soon")
                            last_log_time = time.time()
                        time.sleep(0.1)  # Short polling interval
                        continue
                    
                    # Process single row
                    input_data = pd.DataFrame([row], columns=['timestamp', 'voltage', 'current', 'active_power'])
                    self.last_processed_timestamp = input_data['timestamp'].iloc[0]
                    features = input_data[['voltage', 'current', 'active_power']].values
                    
                    # Maintain a rolling window of features for sequence
                    if not hasattr(self, 'feature_buffer'):
                        self.feature_buffer = np.zeros((self.time_steps, 3))  # [time_steps, features]
                    self.feature_buffer[:-1] = self.feature_buffer[1:]  # Shift left
                    self.feature_buffer[-1] = features[0]  # Add new row
                    
                    X = self.prepare_batch(self.feature_buffer)
                    with torch.no_grad():
                        status_preds, power_preds = self.model(X)
                    
                    total_power = input_data['active_power'].iloc[0]
                    status, adjusted_power = self.adjust_predictions(status_preds[0], power_preds[0], total_power)
                    
                    # Store predictions for this row
                    for j, appliance in enumerate(self.appliances):
                        cur.execute("""
                        INSERT INTO predictions 
                        (timestamp, total_power, voltage, current, active_power,
                        appliance_name, appliance_status, appliance_power)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            input_data['timestamp'].iloc[0], total_power,
                            input_data['voltage'].iloc[0], input_data['current'].iloc[0],
                            input_data['active_power'].iloc[0], appliance, status[j], float(adjusted_power[j])
                        ))
                    
                    conn.commit()
                    
                    # Measure and enforce latency
                    elapsed_time = time.time() - start_time
                    if elapsed_time > target_latency:
                        logging.warning(f"Processing took {elapsed_time:.2f}s, exceeding target of {target_latency}s")
                    elif time.time() - last_log_time > 10:  # Log progress every 10s
                        logging.info(f"Processed row at {input_data['timestamp'].iloc[0]}, took {elapsed_time:.2f}s")
                        last_log_time = time.time()
                    
                    # If we're under 2s, no need to sleep; proceed immediately
                    remaining_time = target_latency - elapsed_time
                    if remaining_time > 0:
                        time.sleep(min(remaining_time, 0.1))  # Small sleep to avoid busy-waiting
                    
                except Exception as e:
                    logging.error(f"Error: {str(e)}")
                    conn.rollback()
                    time.sleep(0.5)  # Short retry delay on error

def main():
    db_params = {
        'dbname': os.getenv('DB_NAME', 'power_consumption'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'host': os.getenv('DB_HOST', 'db'),
        'port': os.getenv('DB_PORT', '5432')
    }
    model_path = '/app/model/nilm_modelv1.pth'
    predictor = NILMModelPredictor(model_path, db_params)
    predictor.predict_and_store(target_latency=2.0)

if __name__ == "__main__":
    main()