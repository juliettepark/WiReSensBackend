#!/usr/bin/env python3
"""
Script to process heavy wrap internal recording files.
Converts tactile and handpose data into pressure (N) and displacement values based on the action.
Collapses each recording into a single row of data with 
- averages, mins, and maxes of the import region pressures
- the average, min, and max of the principal component vector

Appends the row to the result CSV file with the given label.

Usage:
    python3 process_heavy_wrap_internal.py <data_dir> <label>

    Ex. python3 processing_scripts/process_heavy_wrap_internal.py data/hwi/raw_data/firm firm
    Ex. python3 processing_scripts/process_heavy_wrap_internal.py data/hwi/devin_data/firm firm
"""

import sys
from pathlib import Path

# Add project root so tactile_utils can be imported
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

from tactile_utils.PressureConverter import PressureConverter
from tactile_utils.DisplacementConverter import DisplacementConverter
from tactile_utils.tactile_handpose_utils import append_displacement_to_df, collapse_recording_data, convert_raw_data_to_pressure, get_feature_headers

import argparse
import csv
import os
import pandas as pd
import numpy as np

RESULT_CSV = _project_root / "data" / "hwi" / "labeled_collapsed_results" / "heavy_wrap_internal_labeled_collapsed_results_v4.csv"
GRASP_TYPE = "Heavy Wrap Internal"

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

def process_raw_data_folder(data_dir: Path, label: str):
    """For each segment recording, read the CSV file, collapse the data into a single row, and write the row to the result CSV."""
    pressure_converter = PressureConverter()
    displacement_converter = DisplacementConverter(GRASP_TYPE)

    # Open the result CSV to append new rows
    with open(RESULT_CSV, 'a') as f:
        writer = csv.writer(f)

        # Iterate over each segment recording
        for csv_file in os.listdir(data_dir):
            recording_csv_path = os.path.join(data_dir, csv_file)
            with open(recording_csv_path, 'r') as f:
                df = pd.read_csv(recording_csv_path)

                # Convert the raw data to pressure and displacement values
                df = convert_raw_data_to_pressure(df, pressure_converter)
                df = append_displacement_to_df(df, displacement_converter)

                # Collapse the data into a single row
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
            writer.writerow(get_feature_headers())

    # Process the data
    process_raw_data_folder(data_dir, label)


if __name__ == "__main__":
    main()