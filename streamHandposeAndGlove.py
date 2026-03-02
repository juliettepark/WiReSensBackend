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

from tactile_utils.tactile_handpose_utils import append_displacement_to_df, collapse_recording_data, convert_raw_data_to_pressure, save_to_csv, recording_buffer_to_df, save_to_result_data_csv

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
RAW_DATA_RECORDINGS_FOLDER = _project_root / "data" / "power_sphere" / "raw_data"

CONVERTED_RECORDINGS_FOLDER = _project_root / "data" / "power_sphere" / "converted_data"

# File to load the model from.
MODEL_FILE = _project_root / "hwi_dough_model.joblib"

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

                    # Prepare the buffer for collapse and prediction
                    df = recording_buffer_to_df(recording_buffer)
                    if df is None:
                        raise ValueError("No recording data in buffer to query.")
                    df = convert_raw_data_to_pressure(df, pressure_converter)
                    # Append the displacement values to the DataFrame
                    df = append_displacement_to_df(df, displacement_converter)

                    # Query the model for a prediction
                    prediction = query_model(df)

                    # Send through WebSocket to Unity
                    # I will have a label prediction "xsoft", "soft", "medium", "firm"
                    # Unity script will have to parse
                    await websocket.send(json.dumps({
                        "type": "PREDICTION",
                        "prediction": prediction
                    }))

                    print(f"✅ PREDICTION {prediction} SENT TO UNITY")

                elif mode == BackendMode.SAVE_TO_CSV:
                    filename = f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M')}{current_session_label}.csv"
                    save_to_csv(filename)
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

                # 2. Add all 334 bone values (2 hands x 26 bones x 7 data points)
                row.extend(payload.get("data"))
                
                recording_buffer.append(row)

        except Exception as e:
            print(f"Error: {e}")

# MOVING
# def get_descriptive_headers():
#     """
#     Generates headers matching both the sensor array and bone data for one segment recording.
#     Ex.
#     ["pc_ts", "unity_ts", "index_avg", "s_0", "s_1", ..., "s_255", "L_XRHand_Wrist_Px",...]
#     """
#     headers = ["pc_ts", "unity_ts", "index_avg"]
    
#     # Create headers for the full sensor array
#     # Format is s_0, s_1, ...
#     for i in range(NUM_SENSORS):
#         headers.append(f"s_{i}")
    
#     # Create L_BoneName_Px, etc. for both hands
#     for prefix in ["L", "R"]:
#         for bone in bone_names:
#             headers.append(f"{prefix}_{bone}_Px")
#             headers.append(f"{prefix}_{bone}_Py")
#             headers.append(f"{prefix}_{bone}_Pz")
#             headers.append(f"{prefix}_{bone}_Qx")
#             headers.append(f"{prefix}_{bone}_Qy")
#             headers.append(f"{prefix}_{bone}_Qz")
#             headers.append(f"{prefix}_{bone}_Qw")
#     return headers

# MOVING
# def save_to_csv(filename):
#     """Save the segment recording data into its own CSV file."""
#     # Determine how many sensor columns we have based on the first row of data
#     if not recording_buffer:
#         return
    
#     headers = get_descriptive_headers()
    
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)
#         writer.writerows(recording_buffer)

# MOVING
# def recording_buffer_to_df():
#     """Convert recording_buffer (list of rows) to a DataFrame with descriptive headers."""
#     if not recording_buffer:
#         return None
#     headers = get_descriptive_headers()
#     return pd.DataFrame(recording_buffer, columns=headers)

# MOVING
# def save_to_result_data_csv():
#     """Save the segment recording data collapsed as a single row to the result CSV file WITH labels."""
#     df = recording_buffer_to_df()
#     if df is None:
#         return

#     # Collapse current recording into feature row
#     data_row = collapse_recording_data(df)

#     # Create result file with header if it doesn't exist yet
#     file_exists = RESULT_CSV.exists()
#     with open(RESULT_CSV, 'a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:

#             # Write the headers for a collapsed row of data
#             writer.writerow([
#                 "index_avg",
#                 "index_min",
#                 "index_max",
#                 "thumb_avg",
#                 "thumb_min",
#                 "thumb_max",
#                 "finger_tip_avg",
#                 "finger_tip_min",
#                 "finger_tip_max",
#                 "label",
#             ])

#         # Write the collapsed row for the newest sequence recording with its label
#         writer.writerow(data_row + [current_session_label])
#         print(f"Wrote data row to {RESULT_CSV} with label {current_session_label}")

#     # also save for debugging
#     save_to_csv(f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

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
    server = await websockets.serve(quest_handler, "10.18.95.137", 8765)

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