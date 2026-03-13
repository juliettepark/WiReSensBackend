#!/usr/bin/env python3
"""
Script to process pinch recording files of one type and label them.
Converts tactile and handpose data over time into a single row of data with 
- averages, mins, and maxes of the index and thumb pressures
- the average, min, and max of the distance between the index and thumb tips.

Append the row to the result CSV file with the given label.

Usage:
    python3 processing_scripts/process_pinch.py <data_dir> <label>
    Ex. python3 processing_scripts/process_pinch.py data/pinch_dough/raw_data/medium medium
"""

import sys
from pathlib import Path

# Allow importing plot_pinch and tactile_handpose_utils from project root
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import argparse
import csv
import os
import pandas as pd
import numpy as np

from plot_pinch import calculate_distance

# UPDATE WHEN RUNNING
RESULT_CSV = _project_root / "data" / "pinch_dough" / "pinch_dough_labeled_results_1.csv"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process CSV files of pinch segment recordings then compress and label them.",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the folder holding the pinch recording files of one type to process",
    )
    parser.add_argument(
        "label",
        type=str,
        help="Label to apply to the data",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    if not args.data_dir.is_dir():
        parser.error(f"Not a directory: {args.data_dir}")

    return args

# def collapse_recording_data(df: pd.DataFrame):
#     """
#     Collapse the recording data into a single row of data with averages, mins, and maxes of the index and thumb 
#     pressures and the average, min, and max of the distance between the index and thumb tips.
#     Returns a list of the data.
#     """
#     sensor_data = get_sensor_data(df)
#     index_pressures = get_index_averages_right(sensor_data)
#     thumb_pressures = get_thumb_averages_right(sensor_data)
#     index_avg = np.mean(index_pressures)
#     index_min = np.min(index_pressures)
#     index_max = np.max(index_pressures)
#     thumb_avg = np.mean(thumb_pressures)
#     thumb_min = np.min(thumb_pressures)
#     thumb_max = np.max(thumb_pressures)
#     finger_tip_distances = calculate_distance(df, "R_XRHand_IndexTip", "R_XRHand_ThumbTip")
#     finger_tip_avg = np.mean(finger_tip_distances)
#     finger_tip_min = np.min(finger_tip_distances)
#     finger_tip_max = np.max(finger_tip_distances)

#     return [index_avg, index_min, index_max, thumb_avg, thumb_min, thumb_max, finger_tip_avg, finger_tip_min, finger_tip_max]

# def get_index_average(sensors):
#     index_finger_region = (slice(9,11), slice(0,3))
#     grid = sensors[0].pressure.reshape(sensors[0].selWires, sensors[0].readWires)

#     # Pass columns first because grid is shaped as (along the finger - col, which strip - row)
#     return int(np.mean(grid[index_finger_region[1], index_finger_region[0]]))


# def process_pinch_folder(data_dir: Path, label: str):
#     """For each segment recording, read the CSV file, collapse the data into a single row, and write the row to the result CSV."""
    
#     # Open the result CSV to append new rows
#     with open(RESULT_CSV, 'a') as f:
#         writer = csv.writer(f)

#         # Iterate over each segment recording
#         for csv_file in os.listdir(data_dir):
#             recording_csv_path = os.path.join(data_dir, csv_file)
#             with open(recording_csv_path, 'r') as f:
#                 df = pd.read_csv(recording_csv_path)
#                 data_row = collapse_recording_data(df)

#                 # Write the row to the result CSV with the label
#                 writer.writerow(data_row + [label])

def process_pinch_folder(data_dir: Path, label: str):
    """For each segment recording, read the CSV file, collapse the data into a single row, and write the row to the result CSV."""
    
    # Open the result CSV to append new rows
    with open(RESULT_CSV, 'a') as f:
        writer = csv.writer(f)

        # Iterate over each segment recording
        for csv_file in os.listdir(data_dir):
            recording_csv_path = os.path.join(data_dir, csv_file)
            with open(recording_csv_path, 'r') as f:
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

                # Write the row to the result CSV with the label
                writer.writerow([index_avg, index_min, index_max, thumb_avg, thumb_min, thumb_max, finger_tip_avg, finger_tip_min, finger_tip_max, label])

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

def main():
    args = parse_args()
    data_dir = args.data_dir
    label = args.label

    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}")
    print(f"Labeling as: {label}")

    # If the result CSV doesn't exist, write the header
    if not os.path.exists(RESULT_CSV):
        print(f"No CSV result file found.\nCreating result CSV: {RESULT_CSV}")
        with open(RESULT_CSV, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['index_avg', 'index_min', 'index_max', 'thumb_avg', 'thumb_min', 'thumb_max', 'finger_tip_avg', 'finger_tip_min', 'finger_tip_max', 'label'])

    # Process the data
    process_pinch_folder(data_dir, label)


if __name__ == "__main__":
    main()
