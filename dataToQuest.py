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
                    # Otherwise, we would try to send too many packets through
                    # Only send more after we have confirmed this packet has been handed off
                    # await lets the current thread go do other stuff
                    await active_quest.send(json.dumps(payload))

                except websockets.ConnectionClosed:
                    active_quest = None

            # IMPORTANT: This 'await' lets the ESP32 data-receiving 
            # tasks run in the background!
            # Sleep for 10 ms
            # so each second, if we sleep every 0.01 seconds,
            # 1 / 0.01 = 100 maximum run 100 times.
            # Sleeping ensures loop doesn't happen so fast that other tasks handled by OS
            # are not overwhelmed. Sets pace of loop
            await asyncio.sleep(0.01) # Runs at roughly 100Hz