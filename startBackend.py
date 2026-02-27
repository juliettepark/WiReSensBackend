"""
Starts the tactile/handpose backend.

Usage:
    python startBackend.py --label <label> --mode <mode>

Arguments:
    --label: Label for the recording session (e.g. ball_blue)
    --mode: Mode for the backend (e.g. predict, record, or save_to_csv)

Will start the backend and perform the mode action:

- predict: Query the model for a prediction. Send the prediction to the Unity client.
- record: Record the data, collapse into a single row with the given label, and save to the result CSV file.
- save_to_csv: Save the entire sequence ofdata to a separate CSV file

"""

import argparse

from flaskApp.index import start_server_web
from TouchSensorWireless import MultiProtocolReceiver
from dataToQuest import stream_to_quest
import streamHandposeAndGlove
from streamHandposeAndGlove import BackendMode, sync_quest_and_glove


def parse_args():
    parser = argparse.ArgumentParser(description="Start the tactile/handpose backend.")
    parser.add_argument(
        "--label",
        type=str,
        default="none",
        help="Label for the recording session (e.g. ball_blue)",
    )

    valid_modes = [m.value for m in BackendMode]
    parser.add_argument(
        "--mode",
        type=str,
        default=BackendMode.PREDICT.value,
        choices=valid_modes,
        help=f"Mode for the backend: {', '.join(valid_modes)}",
    )
    args = parser.parse_args()
    args.mode = BackendMode(args.mode)  # convert string to enum
    return args


args = parse_args()

# Set the label for the recording session in the streaming script (so we don't have to pass down)
streamHandposeAndGlove.current_session_label = args.label

# Set the mode for the backend
streamHandposeAndGlove.mode = args.mode

myReceiver = MultiProtocolReceiver(configFilePath='./configs/oneGloveSerialReceiverRightSmall.json')

# My custom method to start running loop to receive glove values
# Opens WebSocket server and streams average finger data to Quest
# myReceiver.runCustomMethod(stream_to_quest)

# Listens for handpose data from Quest
# Combines with glove data and saves as CSV file
# when Quest says STOP
myReceiver.runCustomMethod(sync_quest_and_glove)

# start_server_web(myReceiver)