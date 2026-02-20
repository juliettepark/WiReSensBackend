from flaskApp.index import start_server_web
from TouchSensorWireless import MultiProtocolReceiver
from dataToQuest import stream_to_quest
from streamHandposeAndGlove import sync_quest_and_glove


myReceiver = MultiProtocolReceiver(configFilePath='./configs/oneGloveSerialReceiverRightSmall.json')

# My custom method to start running loop to receive glove values
# Opens WebSocket server and streams average finger data to Quest
# myReceiver.runCustomMethod(stream_to_quest)

# Listens for handpose data from Quest
# Combines with glove data and saves as CSV file
# when Quest says STOP
myReceiver.runCustomMethod(sync_quest_and_glove)

# start_server_web(myReceiver)