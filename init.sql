-- init.sql

-- Create the aggregate_data table
CREATE TABLE IF NOT EXISTS aggregate_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    total_power FLOAT NOT NULL,
    voltage FLOAT NOT NULL,
    current FLOAT NOT NULL,
    active_power FLOAT NOT NULL
);

-- Create the predictions table with appliance_name
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    total_power FLOAT NOT NULL,
    voltage FLOAT NOT NULL,
    current FLOAT NOT NULL,
    active_power FLOAT NOT NULL,
    appliance_name TEXT NOT NULL,  -- New column for appliance name
    appliance_status INT NOT NULL,
    appliance_power FLOAT NOT NULL
);
