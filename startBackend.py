from flaskApp.index import start_server_web
from TouchSensorWireless import MultiProtocolReceiver
from dataToQuest import stream_to_quest


myReceiver = MultiProtocolReceiver(configFilePath='./configs/oneGloveSerialReceiverRightSmall.json')
# My custom method to start running loop to receive glove values
myReceiver.runCustomMethod(stream_to_quest)
# start_server_web(myReceiver)