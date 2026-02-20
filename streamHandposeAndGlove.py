import numpy as np
import asyncio
import websockets
import json
import csv
from datetime import datetime

# Constants

NUM_SENSORS = 16*16

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

bone_names = [
    # Roots
    "XRHand_Wrist", "XRHand_Palm",

    # Thumb
    "XRHand_ThumbMetacarpal", "XRHand_ThumbProximal", "XRHand_ThumbDistal", "XRHand_ThumbTip",

    # Index
    "XRHand_IndexMetacarpal", "XRHand_IndexProximal", "XRHand_IndexIntermediate", "XRHand_IndexDistal", "XRHand_IndexTip",

    # Middle
    "XRHand_MiddleMetacarpal", "XRHand_MiddleProximal", "XRHand_MiddleIntermediate", "XRHand_MiddleDistal", "XRHand_MiddleTip",

    # Ring
    "XRHand_RingMetacarpal", "XRHand_RingProximal", "XRHand_RingIntermediate", "XRHand_RingDistal", "XRHand_RingTip",

    # Little
    "XRHand_LittleMetacarpal", "XRHand_LittleProximal", "XRHand_LittleIntermediate", "XRHand_LittleDistal", "XRHand_LittleTip"
]

# This variable will hold our single connection
active_quest = None
latest_sensor_values = []
latest_index_avg = 0
recording_buffer = []
is_recording = False

async def quest_handler(websocket):
    """Handles incoming Hand Pose data from Unity."""
    global latest_sensor_values, recording_buffer, is_recording
    print(f"[🌐] Quest connected!")
    
    async for message in websocket:
        try:
            payload = json.loads(message)
            msg_type = payload.get("type")

            if msg_type == "START_RECORDING":
                print("🔴 RECORDING STARTED")
                recording_buffer = []
                is_recording = True

            elif msg_type == "STOP_RECORDING":
                is_recording = False
                filename = f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                save_to_csv(filename)
                print(f"✅ RECORDING SAVED TO {filename}")

            elif msg_type == "HAND_POSE" and is_recording:
                # MASTER ROW CONSTRUCTION
                row = [
                    datetime.now().timestamp(), # Computer Timestamp
                    payload.get("ts"),          # Unity relative timestamp
                    latest_index_avg,           # Latest index finger pressure value (for debugging)
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

def get_descriptive_headers():
    """Generates headers matching both the sensor array and bone data."""
    headers = ["pc_ts", "unity_ts", "index_avg"]
    
    # Create headers for the full sensor array
    # Format is s_0, s_1, ...
    for i in range(NUM_SENSORS):
        headers.append(f"s_{i}")
    
    # Create L_BoneName_Px, etc. for both hands
    for prefix in ["L", "R"]:
        for bone in bone_names:
            headers.append(f"{prefix}_{bone}_Px")
            headers.append(f"{prefix}_{bone}_Py")
            headers.append(f"{prefix}_{bone}_Pz")
            headers.append(f"{prefix}_{bone}_Qx")
            headers.append(f"{prefix}_{bone}_Qy")
            headers.append(f"{prefix}_{bone}_Qz")
            headers.append(f"{prefix}_{bone}_Qw")
    return headers

def save_to_csv(filename):
    # Determine how many sensor columns we have based on the first row of data
    if not recording_buffer:
        return
    
    headers = get_descriptive_headers()
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(recording_buffer)

def get_index_average(sensors):
    index_finger_region = (slice(9,11), slice(0,3))
    grid = sensors[0].pressure.reshape(sensors[0].selWires, sensors[0].readWires)
    return int(np.mean(grid[index_finger_region[1], index_finger_region[0]]))

async def sync_quest_and_glove(sensors):
    """
    The main background loop.
    1. Starts the WebSocket server to listen to the Quest.
    2. Continously saves the tactile sensor grid from the ESP32 stream.
    3. Listens for new packets of handpose data and stamps along with current tactile values into a row
    """
    global latest_sensor_values, latest_index_avg
    print("[🚀] Sync Server Live on 10.18.58.199:8765")

    # Start the WebSocket server task
    server = await websockets.serve(quest_handler, "10.18.58.199", 8765)

    # Process the sensor grid at 100Hz (faster than Quest)
    while True:
        if sensors[0].init:
            try: 
                # Capture the snapshot of the full pressure array
                # .tolist() ensures standard Python list for CSV writing
                latest_sensor_values = sensors[0].pressure.tolist()
                latest_index_avg = get_index_average(sensors)
            except Exception as e:
                pass # Handle potential momentary reshaping errors

        # Ensure this loop doesn't block the WebSocket server
        await asyncio.sleep(0.01)