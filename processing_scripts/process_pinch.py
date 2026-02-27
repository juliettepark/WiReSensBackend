#!/usr/bin/env python3
"""
Script to process pinch recording files of one type and label them.
Converts tactile and handpose data over time into a single row of data with 
- averages, mins, and maxes of the index and thumb pressures
- the average, min, and max of the distance between the index and thumb tips.

Append the row to the result CSV file with the given label.

Usage:
    python process_pinch.py <data_dir> <label>
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
from tactile_handpose_utils import collapse_recording_data

RESULT_CSV = _project_root / "data" / "pinch_labeled_results.csv"


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
                data_row = collapse_recording_data(df)

                # Write the row to the result CSV with the label
                writer.writerow(data_row + [label])


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
