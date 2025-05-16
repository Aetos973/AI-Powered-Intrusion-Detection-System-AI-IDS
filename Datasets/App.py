import pandas as pd
import numpy as np
import joblib
import gradio as gr
import os
import tempfile

# Set a custom directory for Gradio's temporary files
os.environ["GRADIO_TEMP"] = tempfile.mkdtemp()

# Dictionary of IoT devices and their corresponding model files
device_models = {
    "Garage Door": "garage_door_model.pkl",
    "GPS Tracker": "gps_tracker_model.pkl",
    "Weather": "weather_model.pkl",
    "Thermostat": "thermostat_model.pkl",
    "Fridge": "fridge_model.pkl"
}

# Define required numeric features for each device
device_features = {
    "Garage Door": ["date_numeric", "time_numeric", "door_state", "sphone_signal", "label"],
    "GPS Tracker": ["date_numeric", "time_numeric", "latitude", "longitude", "label"],
    "Weather": ["date_numeric", "time_numeric", "temperature", "humidity", "label"],
    "Thermostat": ["date_numeric", "time_numeric", "temp_set", "temp_actual", "label"],
    "Fridge": ["date_numeric", "time_numeric", "temp_inside", "door_open", "label"]
}

# Class labels for attack types (assuming same for all devices; adjust if needed)
class_labels = {
    0: "Normal",
    1: "Backdoor",
    2: "DDoS",
    3: "Injection",
    4: "Password Attack",
    5: "Ransomware",
    6: "Scanning",
    7: "XSS",
}

def convert_datetime_features(log_data):
    """Convert date and time into numeric values."""
    try:
        log_data['date'] = pd.to_datetime(log_data['date'], format='%d-%m-%y', errors='coerce')
        log_data['date_numeric'] = log_data['date'].astype(np.int64) // 10**9  

        time_parsed = pd.to_datetime(log_data['time'], format='%H:%M:%S', errors='coerce')
        log_data['time_numeric'] = (time_parsed.dt.hour * 3600) + (time_parsed.dt.minute * 60) + time_parsed.dt.second
    except Exception as e:
        return f"Error processing date/time: {str(e)}", None
    
    return None, log_data

def detect_intrusion(device, file):
    """Process log file and predict attack type based on selected device."""
    # Load the selected device's model
    try:
        model = joblib.load(device_models[device])
    except Exception as e:
        return f"Error loading model for {device}: {str(e)}", None, None

    # Read the uploaded file
    try:
        log_data = pd.read_csv(file.name)
    except Exception as e:
        return f"Error reading file: {str(e)}", None, None

    # Convert date and time features
    error, log_data = convert_datetime_features(log_data)
    if error:
        return error, None, None

    # Get the required features for the selected device
    required_features = device_features[device]
    missing_features = [feature for feature in required_features if feature not in log_data.columns]
    if missing_features:
        return f"Missing features for {device}: {', '.join(missing_features)}", None, None

    # Preprocess device-specific features
    try:
        if device == "Garage Door":
            log_data['door_state'] = log_data['door_state'].astype(str).str.strip().replace({'closed': 0, 'open': 1})
            log_data['sphone_signal'] = pd.to_numeric(log_data['sphone_signal'], errors='coerce')
        elif device == "GPS Tracker":
            log_data['latitude'] = pd.to_numeric(log_data['latitude'], errors='coerce')
            log_data['longitude'] = pd.to_numeric(log_data['longitude'], errors='coerce')
        elif device == "Weather":
            log_data['temperature'] = pd.to_numeric(log_data['temperature'], errors='coerce')
            log_data['humidity'] = pd.to_numeric(log_data['humidity'], errors='coerce')
        elif device == "Thermostat":
            log_data['temp_set'] = pd.to_numeric(log_data['temp_set'], errors='coerce')
            log_data['temp_actual'] = pd.to_numeric(log_data['temp_actual'], errors='coerce')
        elif device == "Fridge":
            log_data['temp_inside'] = pd.to_numeric(log_data['temp_inside'], errors='coerce')
            log_data['door_open'] = log_data['door_open'].astype(str).str.strip().replace({'closed': 0, 'open': 1})

        # Prepare feature values for prediction
        feature_values = log_data[required_features].astype(float).values
        predictions = model.predict(feature_values)
    except Exception as e:
        return f"Error during prediction for {device}: {str(e)}", None, None

    # Map predictions to attack types
    log_data['Prediction'] = [class_labels.get(pred, 'Unknown Attack') for pred in predictions]

    # Format date for output
    log_data['date'] = log_data['date'].dt.strftime('%Y-%m-%d')

    # Select final output columns
    output_df = log_data[['date', 'time', 'Prediction']]

    # Save the output to a CSV file for download
    output_file = f"intrusion_results_{device.lower().replace(' ', '_')}.csv"
    output_df.to_csv(output_file, index=False)

    return None, output_df, output_file

# Create Gradio interface
def gradio_interface(device, file):
    error, df, output_file = detect_intrusion(device, file)
    if error:
        return error, None, None
    return df, df, output_file

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(choices=list(device_models.keys()), label="Select IoT Device", value="Garage Door"),
        gr.File(label="Upload Log File (CSV format)")
    ],
    outputs=[
        gr.Textbox(label="Status/Error Message", visible=False),
        gr.Dataframe(label="Intrusion Detection Results"),
        gr.File(label="Download Predictions CSV")
    ],
    title="IoT Intrusion Detection System",
    description=(
        """
        Select an IoT device and upload a CSV log file with the appropriate features for that device.
        Example features per device:
        - Garage Door: date,time,door_state,sphone_signal,label (e.g., 26-04-19,13:59:20,1,-85,normal)
        - GPS Tracker: date,time,latitude,longitude,label
        - Weather: date,time,temperature,humidity,label
        - Thermostat: date,time,temp_set,temp_actual,label
        - Fridge: date,time,temp_inside,door_open,label
        """
    )
)

iface.launch()
