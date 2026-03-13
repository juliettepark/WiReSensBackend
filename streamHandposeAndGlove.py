from enum import Enum
import sys
from pathlib import Path

# Ensure project root is on path so "utils" package resolves
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import asyncio
import websockets
import json
import csv
from datetime import datetime
import joblib
from tactile_utils.PressureConverter import PressureConverter
from tactile_utils.DisplacementConverter import DisplacementConverter

from tactile_utils.tactile_handpose_utils import (
    IMPORTANT_REGIONS,
    append_displacement_to_df,
    collapse_recording_data,
    convert_raw_data_to_pressure,
    get_descriptive_headers,
    get_feature_headers,
    get_region_averages_right,
    get_sensor_data,
    save_to_csv,
    recording_buffer_to_df,
    save_to_result_data_csv,
    right_hand_regions,
)

# =============================== CONSTANTS ===============================

# Number of sensors in the tactile glove pressure grid
NUM_SENSORS = 16*16

# Type of grasp to record and predict
# GRASP_TYPE = "Power sphere"
# GRASP_TYPE = "Precision pinch"
GRASP_TYPE = "Heavy Wrap Internal"

# File to save the results of the recording session. Labeled with 1 row per segment recording.
RESULT_CSV = _project_root / "data" / "power_sphere" / "labeled_collapsed_results" / "hwi_labeled_results_collect_from_stream.csv"

# Path to folder to save the raw CSV recordings per segment
RAW_DATA_RECORDINGS_FOLDER = _project_root / "data" / "pinch_dough"

CONVERTED_RECORDINGS_FOLDER = _project_root / "data" / "power_sphere" / "converted_data"

# File to load the model from.
# MODEL_FILE = _project_root / "hwi_dough_model.joblib"
# MODEL_FILE = _project_root / "hwi_dough_model_more_data_more_features_bestsofar.joblib"
# MODEL_FILE = _project_root / "pinch_dough_model.joblib"

# UPDATE WHEN RUNNING
MODEL_FILE = _project_root / "pinch_dough_model_newmediumdata.joblib"

# Feature columns used by pinch_dough_model (must match training CSV and pinch_model.py column order)
PINCH_FEATURE_COLUMNS = [
    "index_avg", "index_min", "index_max",
    "thumb_avg", "thumb_min", "thumb_max",
    "finger_tip_avg", "finger_tip_min", "finger_tip_max",
]

class BackendMode(Enum):
    PREDICT = "predict"
    RECORD = "record"
    SAVE_TO_CSV = "save_to_csv"

# =============================== GLOBAL VARIABLES ===============================

# Reference to the active Quest connection.
active_quest = None

# Buffer for sensor values from the tactile glove.
latest_sensor_values = []

# Buffer to store the full recording data (timestamps, tactile, handpose)
recording_buffer = []

# Whether we are currently recording.
is_recording = False

# If we are recording new data, label with this
# Argument from command line.
current_session_label = "none"

# Mode to run the backend in. Predict or record.
mode = BackendMode.PREDICT

# Model to make the prediction
model = None

# Initialize the pressure and displacement converters
pressure_converter = PressureConverter()
displacement_converter = DisplacementConverter(GRASP_TYPE)

async def quest_handler(websocket):
    """Handles incoming Hand Pose data from Unity."""
    global latest_sensor_values, recording_buffer, is_recording, current_session_label
    print(f"[🌐] Quest connected!")
    
    async for message in websocket:
        try:
            payload = json.loads(message)
            msg_type = payload.get("type")

            if msg_type == "START_RECORDING":
                print("🔴 RECORDING STARTED")
                recording_buffer = []
                is_recording = True
                # current_session_label = payload.get("label", "unknown")

            elif msg_type == "STOP_RECORDING":
                is_recording = False

                if mode == BackendMode.RECORD:

                    # Save the raw data for entire recording into a separate CSV file
                    filename = f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}{current_session_label}.csv"
                    save_to_csv(recording_buffer, filename, RAW_DATA_RECORDINGS_FOLDER)
                    print(f"✅ FULL CSV SAVED TO {RAW_DATA_RECORDINGS_FOLDER} as {filename}")

                    # Save a new line of data for this recording with label
                    # Save the converted CSV data and a line to the result CSV with the label
                    save_to_result_data_csv(recording_buffer, pressure_converter, displacement_converter, current_session_label, CONVERTED_RECORDINGS_FOLDER, RESULT_CSV)
                    # print(f"✅ RECORDING SAVED TO {RESULT_CSV}")
                elif mode == BackendMode.PREDICT:

                    # # Prepare the buffer for collapse and prediction
                    # df = recording_buffer_to_df(recording_buffer)
                    # if df is None:
                    #     raise ValueError("No recording data in buffer to query.")
                    # df = convert_raw_data_to_pressure(df, pressure_converter)
                    # # Append the displacement values to the DataFrame
                    # df = append_displacement_to_df(df, displacement_converter)

                    # # Query the model for a prediction
                    # prediction = query_model(df)

                    prediction = predict_dough_pinch()

                    # Send through WebSocket to Unity
                    # I will have a label prediction "xsoft", "soft", "medium", "firm"
                    # Unity script will have to parse
                    await websocket.send(json.dumps({
                        "type": "PREDICTION",
                        "prediction": prediction
                    }))

                    print(f"✅ PREDICTION {prediction} SENT TO UNITY")

                elif mode == BackendMode.SAVE_TO_CSV:
                    filename = f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}{current_session_label}.csv"
                    save_to_csv(recording_buffer, filename, RAW_DATA_RECORDINGS_FOLDER)
                    print(f"✅ RECORDING SAVED TO {filename}")

            elif msg_type == "HAND_POSE" and is_recording:
                # MASTER ROW CONSTRUCTION
                row = [
                    datetime.now().timestamp(), # Computer Timestamp
                    payload.get("ts"),          # Unity relative timestamp
                    # latest_index_avg,           # Latest index finger pressure value (for debugging)
                ]
                
                # 1. Add full 16x16 grid of sensor values from the glove
                # If sensor data hasn't arrived yet, fill with zeros
                if len(latest_sensor_values) > 0:
                    row.extend(latest_sensor_values)
                else:
                    # Just write zeros for all the sensors
                    row.extend([0] * NUM_SENSORS) 

                # 2. Add all bone values: must be in same order as get_bone_headers() (L then R, each bone Px,Py,Pz,Qx,Qy,Qz,Qw)
                # Add all 334 bone values (2 hands x 26 bones x 7 data points)
                row.extend(payload.get("data"))
                
                recording_buffer.append(row)

        except Exception as e:
            print(f"Error: {e}")

def predict_dough_pinch():
    """
    Collapse the recording buffer into the same 9 features used to train pinch_dough_model,
    then predict. Uses explicit column names so feature order always matches training.
    """
    df = recording_buffer_to_df(recording_buffer)
    if df is None or df.empty:
        raise ValueError("No recording data in buffer to query.")

    # Validate buffer row length matches headers (catches Unity/buffer misalignment)
    expected_len = len(get_descriptive_headers())
    actual_len = len(recording_buffer[0]) if recording_buffer else 0
    if actual_len != expected_len:
        print(f"[⚠️] PREDICT: buffer row length {actual_len} != expected {expected_len}. "
              "Check that Unity sends bone data in the same order as get_bone_headers().")

    sensor_data = get_sensor_data(df)
    index_pressures = get_index_averages_right(sensor_data)
    thumb_pressures = get_thumb_averages_right(sensor_data)

    index_avg = np.mean(index_pressures)
    index_min = np.min(index_pressures)
    index_max = np.max(index_pressures)
    thumb_avg = np.mean(thumb_pressures)
    thumb_min = np.min(thumb_pressures)
    thumb_max = np.max(thumb_pressures)

    finger_tip_distances = calculate_distance(df, "R_XRHand_IndexTip", "R_XRHand_ThumbTip")
    finger_tip_avg = np.mean(finger_tip_distances)
    finger_tip_min = np.min(finger_tip_distances)
    finger_tip_max = np.max(finger_tip_distances)

    # Build row in exact training order and pass as DataFrame so model sees correct columns
    data_row = [index_avg, index_min, index_max, thumb_avg, thumb_min, thumb_max, finger_tip_avg, finger_tip_min, finger_tip_max]
    X = pd.DataFrame([data_row], columns=PINCH_FEATURE_COLUMNS)
    y_predicted = model.predict(X)

    print("Model predicted label: ", y_predicted[0])

    return y_predicted[0]


def get_sensor_data(df):
    """Get only the sensor data columns from the dataframe."""
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    return df[sensor_cols].values 

def calculate_distance(df, bone1_prefix, bone2_prefix):
    """Calculates Euclidean distance between two bones."""
    p1 = df[[f"{bone1_prefix}_Px", f"{bone1_prefix}_Py", f"{bone1_prefix}_Pz"]].values
    p2 = df[[f"{bone2_prefix}_Px", f"{bone2_prefix}_Py", f"{bone2_prefix}_Pz"]].values
    return np.linalg.norm(p1 - p2, axis=1)

def get_index_averages_right(sensor_data):
    """Get the average pressure in the index finger region for the right hand.
    Returns a list of average pressures for each frame.
    """
    index_pressure = []
    for frame in sensor_data:
        grid = frame.reshape(16, 16)
        # BUG: These were flipped. Need rows 0:3 not cols 
        index_pressure.append(np.mean(grid[right_hand_regions['i3'][1], right_hand_regions['i3'][0]]))
    return index_pressure

def get_thumb_averages_right(sensor_data):
    """Get the average pressure in the thumb region for the right hand.
    Returns a list of average pressures for each frame.
    """
    thumb_pressure = []
    for frame in sensor_data:
        grid = frame.reshape(16, 16)
        thumb_pressure.append(np.mean(grid[right_hand_regions['t2'][1], right_hand_regions['t2'][0]]))
    return thumb_pressure


# TODO: Ensure that the features passed in data row are exactly the same as the features used to train the model.
def query_model(df) -> str:
    """Query the model for its prediction given the current data in the buffer."""
    if df is None:
        raise ValueError("Buffer to query model is None.")
    data_row = collapse_recording_data(df)
    # model.predict expects 2D array: (n_samples, n_features)
    y_predicted = model.predict([data_row])

    # Return the single predicted label
    return y_predicted[0]

# def get_index_average(sensors):
#     index_finger_region = (slice(9,11), slice(0,3))
#     grid = sensors[0].pressure.reshape(sensors[0].selWires, sensors[0].readWires)

#     # Pass columns first because grid is shaped as (along the finger - col, which strip - row)
#     return int(np.mean(grid[index_finger_region[1], index_finger_region[0]]))

async def sync_quest_and_glove(sensors):
    """
    The main background loop.
    1. Starts the WebSocket server to listen to the Quest.
    2. Continously saves the tactile sensor grid from the ESP32 stream.
    3. Listens for new packets of handpose data and stamps along with current tactile values into a row
    """
    # global latest_sensor_values, latest_index_avg, model
    global latest_sensor_values, model

    # Load the model
    model = joblib.load(MODEL_FILE)
    print(f"[🤖] Model loaded from {MODEL_FILE}")

    print("[🚀] Sync Server Live on 10.18.58.199:8765")

    # Start the WebSocket server task
    # server = await websockets.serve(quest_handler, "10.18.58.199", 8765)
    server = await websockets.serve(quest_handler, "10.18.81.13", 8765)

    # Process the sensor grid at 100Hz (faster than Quest)
    while True:
        if sensors[0].init:
            try: 
                # Capture the snapshot of the full pressure array
                # .tolist() ensures standard Python list for CSV writing
                latest_sensor_values = sensors[0].pressure.tolist()
                # latest_index_avg = get_index_average(sensors)
            except Exception as e:
                pass # Handle potential momentary reshaping errors

        # Ensure this loop doesn't block the WebSocket server
        await asyncio.sleep(0.01)