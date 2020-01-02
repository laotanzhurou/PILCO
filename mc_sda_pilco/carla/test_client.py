import zmq
import time
import json

from enum import Enum
from random import randrange

context = zmq.Context()


# Clear Noon
# 15.0, 0.0, 0.0, 0.3499999940395355, 0.0, 75.0

#  Socket to talk to server
print("Connect to server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response

fa = open("action.txt", "a+")
fs = open("state.txt", "a+")

count = 0
while True:
    if count % 150 == 0:
        print("initiating Carla...")
        socket.send_json({"type": "RESTART", "data": ""})
        print("init signal sent")
        init = False
        fa.write("initiating... \n")
        fs.write("initiating... \n")
    else:
        print("Sending the next action...")
        cloudyness = randrange(-5, 5)
        precipitation = randrange(-5, 5)
        precipitation_deposits = randrange(-5, 5)
        wind_intensity = randrange(-5, 5)
        sun_azimuth_angle = randrange(-4, 4)
        sun_altitude_angle = randrange(-15, 15)
        message = {"type":"ACTION", "data":(cloudyness, precipitation, precipitation_deposits, wind_intensity, sun_azimuth_angle, sun_altitude_angle)}
        socket.send_json({"type":"ACTION", "data":(cloudyness, precipitation, precipitation_deposits, wind_intensity, sun_azimuth_angle, sun_altitude_angle)})
        fa.write(json.dumps(message) + "\n")
        print("action sent")

    #  Get the reply.
    message = socket.recv_json()
    print("Received next state: %s " % (message))
    fs.write(json.dumps(message) + "\n")

    # simulate model logic
    # time.sleep(0.1)

    # increase counter
    count += 1
    print("current count: " + str(count))

    if count == 1500:
        break
