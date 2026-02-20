import numpy as np
import asyncio
import websockets
import json
import csv
from datetime import datetime

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
latest_sensor_value = 0
recording_buffer = []
is_recording = False

async def quest_handler(websocket):
    """Handles incoming Hand Pose data from Unity."""
    global latest_sensor_value, recording_buffer, is_recording
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
                # MERGE HAPPENS HERE: 
                # We take the Unity Bone data and attach the latest sensor average
                row = [
                    datetime.now().timestamp(), # PC Timestamp
                    payload.get("ts"),          # Unity relative timestamp
                    latest_sensor_value         # The value from your numpy processing
                ]

                # The 334 float values (bones)
                # 2 hands x 26 bones x 7 data points
                row.extend(payload.get("data"))
                recording_buffer.append(row)

        except Exception as e:
            print(f"Error: {e}")

def save_to_csv(filename):
    # Setup headers: Time, Sensor, then all Bone floats
    headers = get_descriptive_headers()
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(recording_buffer)

def get_descriptive_headers():
    """Generates headers matching the Unity HandPoseLogger format."""
    headers = ["pc_ts", "unity_ts", "sensor_avg"]
    
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


async def sync_quest_and_glove(sensors):
    """
    The main background loop.
    1. Starts the WebSocket server to listen to the Quest.
    2. Continously calculates the tactile sensor average from the ESP32 stream.
    """
    global latest_sensor_value
    print("[🚀] Sync Server Live on 10.18.58.199:8765")
    
    # Define the slice for the index finger tip based on your regions
    index_finger_region = (slice(9,11), slice(0,3))

    # Start the WebSocket server task
    server = await websockets.serve(quest_handler, "10.18.58.199", 8765)

    # Process the sensor grid at 100Hz
    while True:
        if sensors[0].init:
            try:
                # Reshape raw 1D array into the 2D pressure grid
                grid = sensors[0].pressure.reshape(sensors[0].selWires, sensors[0].readWires)
                # Calculate mean of the specific finger region
                latest_sensor_value = int(np.mean(grid[index_finger_region[1], index_finger_region[0]]))
            except Exception as e:
                pass # Handle potential momentary reshaping errors

        # Ensure this loop doesn't block the WebSocket server
        await asyncio.sleep(0.01)