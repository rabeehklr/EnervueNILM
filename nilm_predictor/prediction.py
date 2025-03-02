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
        self.status_threshold = 0.5  # Lowered to 0.5 to catch bulb more reliably
        self.min_power_threshold = 5
        self.time_steps = 10
        self.mobile_charger_threshold = 25.0
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
            if 'mobile charger' not in self.appliances or 'bulb' not in self.appliances or 'laptop charger' not in self.appliances:
                missing = [a for a in ['mobile charger', 'bulb', 'laptop charger'] if a not in self.appliances]
                logging.error(f"Missing appliances in model: {missing}")
                raise ValueError("Required appliances not found in model")
            self.mobile_charger_idx = self.appliances.index('mobile charger')
            self.bulb_idx = self.appliances.index('bulb')
            self.laptop_charger_idx = self.appliances.index('laptop charger')
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_batch(self, features):
        if len(features) < self.time_steps:
            padding = np.zeros((self.time_steps - len(features), features.shape[1]))
            features = np.vstack((padding, features))
        elif len(features) > self.time_steps:
            features = features[-self.time_steps:]
        features_normalized = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        features_tensor = features_tensor.T.unsqueeze(0)
        return features_tensor
    
    def adjust_predictions(self, status_preds, power_preds, total_power):
        status_preds_np = status_preds.cpu().numpy()
        power_preds_np = power_preds.cpu().numpy()
        
        # Log raw predictions for debugging
        logging.debug(f"Raw status_preds: {dict(zip(self.appliances, status_preds_np))}, "
                      f"power_preds: {dict(zip(self.appliances, power_preds_np))}, "
                      f"total_power: {total_power}")

        # Initialize status and power arrays
        status = np.zeros(len(self.appliances), dtype=int).tolist()
        adjusted_power = np.zeros(len(self.appliances))

        if total_power < self.mobile_charger_threshold:
            # Exclusive mobile charger case
            status[self.mobile_charger_idx] = 1
            adjusted_power[self.mobile_charger_idx] = min(total_power, power_preds_np[self.mobile_charger_idx])
            
        else:
            # Predict other appliances (excluding mobile charger)
            status = (status_preds_np > self.status_threshold).astype(int).tolist()
            status[self.mobile_charger_idx] = 0  # Force mobile charger off
            adjusted_power = power_preds_np * np.array(status)
            predicted_total = np.sum(adjusted_power)

            # If no appliances are predicted but power is significant, lower threshold for top candidates
            if predicted_total == 0 and total_power > self.min_power_threshold:
                top_indices = np.argsort(status_preds_np)[-2:]  # Take top 2 appliances
                for idx in top_indices:
                    if idx != self.mobile_charger_idx and status_preds_np[idx] > 0.3:  # Relaxed threshold
                        status[idx] = 1
                        adjusted_power[idx] = power_preds_np[idx]
                predicted_total = np.sum(adjusted_power)

            # Scale power to match total_power
            if predicted_total > 0 and total_power > self.min_power_threshold:
                ratio = total_power / predicted_total
                adjusted_power *= ratio
        
        # Log adjusted predictions
        logging.debug(f"Adjusted status: {dict(zip(self.appliances, status))}, "
                      f"power: {dict(zip(self.appliances, adjusted_power))}")
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
        
        with self.database_connection() as conn:
            cur = conn.cursor()
            last_log_time = time.time()
            last_total_power = None
            
            while True:
                try:
                    start_time = time.time()
                    
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
                        if time.time() - last_log_time > 10:
                            logging.info("No new data, checking again soon")
                            last_log_time = time.time()
                        time.sleep(0.1)
                        continue
                    
                    input_data = pd.DataFrame([row], columns=['timestamp', 'voltage', 'current', 'active_power'])
                    self.last_processed_timestamp = input_data['timestamp'].iloc[0]
                    features = input_data[['voltage', 'current', 'active_power']].values
                    total_power = input_data['active_power'].iloc[0]
                    
                    # Reset buffer on significant power drop
                    if last_total_power is not None and total_power < last_total_power * 0.1:
                        logging.info("Significant power drop detected, resetting feature buffer")
                        self.feature_buffer = np.zeros((self.time_steps, 3))
                    last_total_power = total_power
                    
                    # Update feature buffer
                    if not hasattr(self, 'feature_buffer'):
                        self.feature_buffer = np.zeros((self.time_steps, 3))
                    self.feature_buffer[:-1] = self.feature_buffer[1:]
                    self.feature_buffer[-1] = features[0]
                    
                    X = self.prepare_batch(self.feature_buffer)
                    with torch.no_grad():
                        status_preds, power_preds = self.model(X)
                    
                    status, adjusted_power = self.adjust_predictions(status_preds[0], power_preds[0], total_power)
                    
                    # Store predictions
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
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time > target_latency:
                        logging.warning(f"Processing took {elapsed_time:.2f}s, exceeding target of {target_latency}s")
                    elif time.time() - last_log_time > 10:
                        logging.info(f"Processed row at {input_data['timestamp'].iloc[0]}, took {elapsed_time:.2f}s")
                        last_log_time = time.time()
                    
                    remaining_time = target_latency - elapsed_time
                    if remaining_time > 0:
                        time.sleep(min(remaining_time, 0.1))
                    
                except Exception as e:
                    logging.error(f"Error: {str(e)}")
                    conn.rollback()
                    time.sleep(0.5)

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