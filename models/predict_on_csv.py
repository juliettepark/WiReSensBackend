# Test file to make sure that real time prediction works as expected
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
from pathlib import Path
import numpy as np

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


MODEL_FILE = _project_root / "../pinch_dough_model.joblib"
recording_csv_path = _project_root / "../data" / "pinch_dough" / "medium" / "tactile_hand_capture_20260305_180950medium.csv"

right_hand_regions = {
    't2':(slice(13,16), slice(12,16)),
    't1':(slice(11,13), slice(12,16)),
    'pa1':(slice(0,6), slice(12,16)),
    'pa2':(slice(6,11), slice(12,16)),
    'pa3':(slice(6,11), slice(9,12)),
    'pa4':(slice(0,6), slice(9,12)),
    'i1':(slice(9,11),slice(6,9)),
    'i2':(slice(9,11),slice(3,6)),
    'i3':(slice(9,11),slice(0,3)),
    'm1':(slice(6,8),slice(6,9)),
    'm2':(slice(6,8),slice(3,6)),
    'm3':(slice(6,8),slice(0,3)),
    'r1':(slice(3,5),slice(6,9)),
    'r2':(slice(3,5),slice(3,6)),
    'r3':(slice(3,5),slice(0,3)),
    'p1':(slice(0,2),slice(6,9)),
    'p2':(slice(0,2),slice(3,6)),
    'p3':(slice(0,2),slice(0,3)),
}

def predict_dough_pinch():
    df = pd.read_csv(recording_csv_path)
    sensor_data = get_sensor_data(df)
    index_pressures = get_index_averages_right(sensor_data)
    thumb_pressures = get_thumb_averages_right(sensor_data)

    # Get the average, min, and max of the index and thumb pressures
    index_avg = np.mean(index_pressures)
    index_min = np.min(index_pressures)
    index_max = np.max(index_pressures)
    thumb_avg = np.mean(thumb_pressures)
    thumb_min = np.min(thumb_pressures)
    thumb_max = np.max(thumb_pressures)

    # Get the distance between the index and thumb tips
    finger_tip_distances = calculate_distance(df, "R_XRHand_IndexTip", "R_XRHand_ThumbTip")
    # Get the average, min, and max of the finger tip distances
    finger_tip_avg = np.mean(finger_tip_distances)
    finger_tip_min = np.min(finger_tip_distances)
    finger_tip_max = np.max(finger_tip_distances)

    data_row = [index_avg, index_min, index_max, thumb_avg, thumb_min, thumb_max, finger_tip_avg, finger_tip_min, finger_tip_max]
    y_predicted = model.predict([data_row])

    # Return the single predicted label
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


model = joblib.load(MODEL_FILE)
print(f"[🤖] Model loaded from {MODEL_FILE}")

print(f"[🤖] Predicting on {recording_csv_path}")

prediction = predict_dough_pinch()
print(f"[🤖] Prediction: {prediction}")