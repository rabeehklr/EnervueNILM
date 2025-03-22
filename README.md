
# EnervueNILM

**EnervueNILM** is a Non-Intrusive Load Monitoring (NILM) system designed to disaggregate total power consumption into individual appliance-level insights. This project leverages Docker for containerized deployment, a PostgreSQL database for data storage, a Flask-based API for real-time data access, and a machine learning model (CNN-LSTM) for predicting appliance states and power usage. The system includes an ESP32 simulator for generating power consumption data, a predictor service for NILM inference, and a web API for delivering actionable insights to end users.

## Table of Contents
1. [Features](#features)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Database Schema](#database-schema)
8. [API Endpoints](#api-endpoints)
9. [Machine Learning Model](#machine-learning-model)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)

## Features
- **Real-Time Monitoring**: Tracks appliance power consumption and status in real-time using WebSocket updates.
- **Anomaly Detection**: Identifies unusual power usage patterns based on dynamic thresholds.
- **Historical Data Analysis**: Provides daily and weekly power consumption trends per appliance.
- **Cost Estimation**: Estimates electricity costs based on appliance usage over specified periods (weekly/monthly).
- **Scalable Deployment**: Uses Docker Compose to manage multi-service architecture (database, simulator, predictor, API).
- **ESP32 Simulation**: Simulates power data generation for testing and development.
- **Machine Learning**: Employs a CNN-LSTM model for accurate appliance disaggregation.

## Architecture
The system consists of four main services orchestrated via Docker Compose:
1. **PostgreSQL Database (`db`)**: Stores aggregate power data and predictions.
2. **ESP32 Simulator (`esp32-simulator`)**: Generates simulated power consumption data and sends it to the database.
3. **NILM Predictor (`nilm-predictor`)**: Processes aggregate data using a pre-trained CNN-LSTM model to predict appliance-level usage.
4. **Flask API (`flask-api`)**: Provides a RESTful API and WebSocket interface for real-time and historical data access.

The services communicate via a shared PostgreSQL database and are interconnected using environment variables defined in the `docker-compose.yml` file.

## Prerequisites
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher
- **Python**: Version 3.9 (for local development outside Docker)
- **Git**: For cloning the repository
- A pre-trained NILM model file (`nilm_modelv1.pth`) placed in the `nilm_predictor/model/` directory

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/rabeehklr/EnervueNILM.git
cd EnervueNILM
```

### Step 2: Prepare the Model
Place the pre-trained model file (`nilm_modelv1.pth`) in the `nilm_predictor/model/` directory. This file is required by the `nilm-predictor` service and is not included in the repository due to its size.

### Step 3: Build and Run with Docker Compose
From the project root directory, run:
```bash
docker-compose up --build
```
This command builds the Docker images for each service and starts the containers. The `--build` flag ensures that changes to the Dockerfiles or code are reflected.

### Step 4: Verify Services
- **Database**: Accessible at `localhost:5432` (PostgreSQL)
- **ESP32 Simulator**: Runs on `localhost:5000`
- **Flask API**: Available at `localhost:5001`
- **NILM Predictor**: Runs as a background service, logging predictions to the console.

To stop the services, press `Ctrl+C` or run:
```bash
docker-compose down
```

To remove volumes (including the database), use:
```bash
docker-compose down -v
```

## Usage

### Accessing the Database
Connect to the PostgreSQL database using a client like `psql`:
```bash
docker exec -it dockersetup-db-1 psql -U postgres -d power_consumption
```

Export data to CSV (example):
```bash
docker exec -it dockersetup-db-1 psql -U postgres -d power_consumption -c "\COPY (SELECT * FROM aggregate_data) TO '/tmp/aggregate_data.csv' WITH CSV HEADER"
docker cp dockersetup-db-1:/tmp/aggregate_data.csv ./aggregate_data.csv
```

### Simulating Power Data
Send simulated power data to the ESP32 simulator endpoint:
```bash
curl -X POST http://localhost:5000/data \
-H "Content-Type: application/json" \
-d '{"total_power": 100.5, "voltage": 220.0, "current": 0.457}'
```

### Accessing the API
- **Real-Time Updates**: Connect to the WebSocket at `ws://localhost:5001` to receive live appliance data.
- **Historical Data**: Query historical usage via the REST API (see [API Endpoints](#api-endpoints)).

## Project Structure

EnervueNILM/
├── docker-compose.yml         # Docker Compose configuration
├── esp32_simulator/          # ESP32 simulator service
│   ├── Dockerfile
│   ├── esp32_simulator.py
│   └── requirements.txt
├── flask_api/                # Flask API service
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── nilm_predictor/           # NILM predictor service
│   ├── Dockerfile
│   ├── prediction.py
│   ├── requirements.txt
│   └── model/               # Directory for the pre-trained model (not tracked)
├── init.sql                  # Database initialization script
├── full.py                   # Utility script to merge Docker files
└── .git/                     # Git repository metadata


## Database Schema
The database (`power_consumption`) contains two tables defined in `init.sql`:

### `aggregate_data`
| Column         | Type      | Description                   |
|----------------|-----------|-------------------------------|
| `id`           | SERIAL    | Primary key                   |
| `timestamp`    | TIMESTAMP | Time of measurement           |
| `total_power`  | FLOAT     | Total power consumption (W)   |
| `voltage`      | FLOAT     | Voltage (V)                   |
| `current`      | FLOAT     | Current (A)                   |
| `active_power` | FLOAT     | Active power (W)              |

### `predictions`
| Column            | Type      | Description                   |
|-------------------|-----------|-------------------------------|
| `id`              | SERIAL    | Primary key                   |
| `timestamp`       | TIMESTAMP | Time of prediction            |
| `total_power`     | FLOAT     | Total power consumption (W)   |
| `voltage`         | FLOAT     | Voltage (V)                   |
| `current`         | FLOAT     | Current (A)                   |
| `active_power`    | FLOAT     | Active power (W)              |
| `appliance_name`  | TEXT      | Name of the appliance         |
| `appliance_status`| INT       | Status (0 = off, 1 = on)      |
| `appliance_power` | FLOAT     | Predicted power usage (W)     |

## API Endpoints

### REST API
- **`GET /api/historical-data`**
  - Parameters: `appliance_name`, `start_date`, `end_date`
  - Response: Historical usage data for the specified appliance.
  - Example: `GET /api/historical-data?appliance_name=bulb&start_date=2025-03-14&end_date=2025-03-21`

- **`POST /api/cost-estimation`**
  - Body: `{ "appliances": [{"name": "bulb"}], "duration": "weekly" }`
  - Response: Cost estimates based on consumption and a fixed rate (3.15 INR/kWh).

- **`POST /api/update-limit`**
  - Body: `{ "appliance_name": "bulb", "limit": 75.0 }`
  - Response: Updates the anomaly detection threshold for the specified appliance.

### WebSocket
- **Event: `real_time_data`**
  - Payload: `{ "active_appliances": [...], "inactive_appliances": [...] }`
  - Description: Emits real-time appliance status every 2 seconds.

## Machine Learning Model
The `nilm-predictor` uses a CNN-LSTM hybrid model (`CNNLSTMModel`) trained to disaggregate total power into appliance-specific predictions. Key features:
- **Input**: Voltage, current, and active power over 10 time steps.
- **Output**: Appliance status (binary) and power consumption (continuous).
- **Adjustments**: Custom logic ensures predictions align with total power, with special handling for low-power devices like mobile chargers.

The model file (`nilm_modelv1.pth`) must include:
- `model_state_dict`: Trained weights
- `appliances`: List of supported appliances (e.g., `["bulb", "mobile charger", "laptop charger", ...]`)
- `scaler_state`: Normalization parameters

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code adheres to PEP 8 standards and include unit tests where applicable.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details (to be added).

## Contact
For questions or support, contact:
- **Maintainer**: Rabeeh KLR
- **Email**: rabeehkhancoc@gmail.com
- **GitHub**: [rabeehklr](https://github.com/rabeehklr)
