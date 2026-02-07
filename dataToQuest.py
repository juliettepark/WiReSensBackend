import numpy as np
import asyncio
import websockets
import json

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

# This variable will hold our single connection
active_quest = None

async def quest_handler(websocket):
    """Triggered whenever a Quest connects."""
    global active_quest
    print(f"[🌐] Quest connected from {websocket.remote_address}")
    active_quest = websocket
    
    try:
        # Keep the connection alive
        await websocket.wait_closed()
    finally:
        # Clean up when they disconnect
        active_quest = None
        print("[🌐] Quest disconnected")

async def stream_to_quest(sensors):
    """Main loop: Starts server and sends data to the single Quest."""
    global active_quest
    print("STARTING QUEST STREAMER...")

    # Start server on your laptop's IP, port 8765
    async with websockets.serve(quest_handler, "10.18.58.199", 8765):
        print("[🚀] Server live. Waiting for Quest to connect...")
        index_finger_region = right_hand_regions['i3']

        while True:
            # Only send if the Quest is connected AND sensor is initialized
            if active_quest and sensors[0].init:
                try:
                    # 1. Process data
                    pressureGrid = sensors[0].pressure.reshape(sensors[0].selWires, sensors[0].readWires)
                    index_finger_avg = int(np.mean(pressureGrid[index_finger_region[1], index_finger_region[0]]))
                    
                    # 2. Create your message
                    payload = {
                        "type": "SENSOR_DATA",
                        # "avg": float(np.mean(pressureGrid)),
                        "avg": index_finger_avg
                        # "grid": pressureGrid.tolist()
                    }

                    # print("Sending avg value: ", index_finger_avg)

                    # 3. Send to the single active quest
                    # We 'await' the send so we don't overwhelm the network
                    await active_quest.send(json.dumps(payload))

                except websockets.ConnectionClosed:
                    active_quest = None

            # IMPORTANT: This 'await' lets the ESP32 data-receiving 
            # tasks run in the background!
            await asyncio.sleep(0.01) # Runs at roughly 100Hz

# Initialize mouse controller
# keyboard = Controller()

# from collections import deque

# class FrameBuffer:
#     def __init__(self, m, n, w):
#         self.m = m  # Number of rows in each frame
#         self.n = n  # Number of columns in each frame
#         self.w = w  # Number of frames to maintain
#         self.buffer = deque(maxlen=w)  # Deque to store the frames

#     def add_frame(self, frame):
#         # Ensure the frame is of correct shape
#         if frame.shape != (self.m, self.n):
#         raise ValueError(f"Frame must be of shape ({self.m}, {self.n})")

#         # Add the new frame to the buffer (automatically removes oldest if needed)
#         self.buffer.append(frame)

#     def get_moving_average(self):
#         # Calculate the moving average if there are frames in the buffer
#         if not self.buffer:
#             raise ValueError("Buffer is empty. Add frames before calculating the average.")

#         # Stack all frames and calculate the mean along the first axis
#         stacked_frames = np.stack(self.buffer)
#         return np.mean(stacked_frames, axis=0)

# Takes in sensors variable to access the latest sensor values
# Will be asynchronously updated in a separate thread by the backend receiver
# def stream_to_quest(sensors):
#     print("ENTERED STREAM TO QUEST")
#     print(len(sensors))
#     print(sensors[0].pressure)

#     index_finger_region = right_hand_regions['i3']
#     running = True
#     while running:
        
#         pressureGrid = sensors[0].pressure.reshape(sensors[0].selWires, sensors[0].readWires)
#         if sensors[0].init:
#             index_finger_avg = np.mean(pressureGrid[index_finger_region[0], index_finger_region[1]])
#             print("Index Finger Avg: ", index_finger_avg)
#             print("Pausing now.")
#             sleep(1) # Pauses the program for 1 second
#             print("Resumed.")


# def startController(sensors):
#     print(len(sensors))
#     # Constants
#     MAT_SIZE = 32
#     MAX_PRESSURE = 1000

#     volumeUpRegion = (slice(0,8), slice(16,32))
#     volumeDownRegion = (slice(24,32), slice(16,32))
#     playRegion = (slice(0,10),slice(0,16))
#     pauseRegion = (slice(20,32),slice(0,10))

#     playPauseThreshold = 2300
#     # (calibrated)
#     # volumeUpThreshold=1800
#     #  volumeDownThreshold = 1900
#     # (uncalibrated)
#     volumeUpThreshold = 1400
#     volumeDownThreshold = 1800
#     minVolumeTheshold = 1550
#     #50 presses in 4 seconds (0-100 volume)
#     # Main drawing loop
#     running = True
#     paused = False
#     while running:
#         pressureGrid = sensors[0].pressure.reshape(sensors[0].selWires, sensors[0].readWires)
#         if sensors[0].init:
#             volumeUpAvg = np.mean(pressureGrid[volumeUpRegion[0],volumeUpRegion[1]])
#             volumeDownAvg = np.mean(pressureGrid[volumeDownRegion[0],volumeDownRegion[1]])
#             playAvg = np.mean(pressureGrid[playRegion[0],playRegion[1]])
#             pauseAvg = np.mean(pressureGrid[pauseRegion[0],pauseRegion[1]])

#             if playAvg <= playPauseThreshold and paused:
#                 print("Play")
#                 print(playAvg)
#                 keyboard.press(Key.media_play_pause)
#                 paused=False
#             elif pauseAvg <= playPauseThreshold and not paused:
#                 print("Pause")
#                 print(pauseAvg)
#                 keyboard.press(Key.media_play_pause)
#                 paused = True

#             if volumeUpAvg <volumeUpThreshold:
#                 print("Volume Up")
#                 sleepFactor = 3
#                 diff = volumeUpThreshold - volumeUpAvg
#                 if diff < 150:
#                     sleepFactor = 5
#                 elif diff < 300:
#                     sleepFactor = 15
#                 else:
#                     sleepFactor = 30
#                 keyboard.press(Key.media_volume_up)
#                 print(sleepFactor)
#                 time.sleep(1/sleepFactor)
#             elif volumeDownAvg < volumeDownThreshold:
#                 sleepFactor = 3
#                 diff = volumeDownThreshold - volumeDownAvg
#                 if diff < 150:
#                     sleepFactor = 5
#                 elif diff < 300:
#                     sleepFactor = 15
#                 else:
#                     sleepFactor = 30
#                 print("Volume Down")
#                 print(sleepFactor)
#                 keyboard.press(Key.media_volume_down)
#                 time.sleep(1/sleepFactor)