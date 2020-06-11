import zmq
import time
import json
from random import randint
from enum import Enum

context = zmq.Context()


# Clear Noon
# 15.0, 0.0, 0.0, 0.3499999940395355, 0.0, 75.0

#  Socket to talk to server
print("Connect to server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Data files
fa = open("../data/test_set/action_new.txt", "w+")
fs = open("../data/test_set/state_new.txt", "w+")

# hyper parameters
count = 0
horizon = 50
total_episodes = 50 * 50

while True:

    if count % horizon == 0:
        print("count: {}".format(count))
        print("initiating Carla...")
        socket.send_json({"type": "RESTART", "data": ""})
        print("init signal sent")
        init = False
        fs.write("initiating... \n")
    else:
        # print("Sending the next action...")

        cloudyness = 0
        precipitation = 0
        precipitation_deposits = 0
        wind_intensity = 0
        sun_azimuth_angle = 0

        sun_altitude_angle = -2 if randint(1, 10) <= 8 else 2

        message = {"type":"ACTION", "data":(cloudyness, precipitation, precipitation_deposits, wind_intensity, sun_azimuth_angle, sun_altitude_angle)}
        socket.send_json({"type":"ACTION", "data":(cloudyness, precipitation, precipitation_deposits, wind_intensity, sun_azimuth_angle, sun_altitude_angle)})
        fa.write(json.dumps(message) + "\n")
        # print("action sent")

    #  Get the reply.
    message = socket.recv_json()
    # print("Received next state: %s " % (message))
    fs.write(json.dumps(message) + "\n")

    # simulate model logic
    # time.sleep(0.1)

    # increase counter
    count += 1
    # print("current count: " + str(count))

    if count == total_episodes:
        print("count: {}".format(count))
        break

print("Total episodes: ".format(count))
