import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Define the regions exactly as they are in your recording script
regions = {
    'i3': (slice(9, 11), slice(0, 3)),
    't2': (slice(13, 16), slice(12, 16))
}

def calculate_distance(df, bone1_prefix, bone2_prefix):
    """Calculates Euclidean distance between two bones."""
    p1 = df[[f"{bone1_prefix}_Px", f"{bone1_prefix}_Py", f"{bone1_prefix}_Pz"]].values
    p2 = df[[f"{bone2_prefix}_Px", f"{bone2_prefix}_Py", f"{bone2_prefix}_Pz"]].values
    return np.linalg.norm(p1 - p2, axis=1)

def main():
    parser = argparse.ArgumentParser(description='Plot pinch data: Pressure vs Distance')
    parser.add_argument('file', type=str, help='Path to the CSV recording')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(args.file)
    
    # 2. Extract Tactile Grid (s_0 to s_255)
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    sensor_data = df[sensor_cols].values 
    
    index_pressure = []
    thumb_pressure = []

    for frame in sensor_data:
        grid = frame.reshape(16, 16)

        # Use the same logic as the recording side:
        # slice 1 (index 1) for rows, slice 0 (index 0) for columns
        idx_region = regions['i3']
        thb_region = regions['t2']
        
        index_pressure.append(np.mean(grid[idx_region[1], idx_region[0]]))
        thumb_pressure.append(np.mean(grid[thb_region[1], thb_region[0]]))

        # BUG: These were flipped. Need rows 0:3 not cols 
        # index_pressure.append(np.mean(grid[9:11, 0:3]))
        # thumb_pressure.append(np.mean(grid[13:16, 12:16]))

    df['index_avg_pressure'] = index_pressure
    df['thumb_avg_pressure'] = thumb_pressure

    # 3. Calculate Bone Distance
    df['tip_distance'] = calculate_distance(df, "R_XRHand_IndexTip", "R_XRHand_ThumbTip")

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Pressure Axis (Left) - Using Red/Orange for heat/pressure
    ax1.set_xlabel('Time (Unity Timestamp)')
    ax1.set_ylabel('Mean Pressure (Raw)', color='tab:red')
    
    # Plotting both Index and Thumb
    ax1.plot(df['unity_ts'], df['index_avg_pressure'], label='Index Pressure (i3)', color='tab:red', linewidth=2)
    ax1.plot(df['unity_ts'], df['thumb_avg_pressure'], label='Thumb Pressure (t2)', color='tab:orange', linewidth=2)
    
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Distance Axis (Right) - Using Blue
    ax2 = ax1.twinx() 
    color_d = 'tab:blue'
    ax2.set_ylabel('Bone Distance (Unity Units)', color=color_d)
    ax2.plot(df['unity_ts'], df['tip_distance'], label='Index-Thumb Tip Distance', color=color_d, linestyle='--', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color_d)

    # Invert Pressure if numbers decrease during touch
    # ax1.invert_yaxis() 

    plt.title(f'Pinch Analysis: {os.path.basename(args.file)}')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.show()

if __name__ == "__main__":
    main()
