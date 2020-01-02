import zmq
import time

from enum import Enum

context = zmq.Context()

#  Socket to talk to server
print("Connect to server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response

init = True
while True:
    if init:
        print("initiating Carla...")
        socket.send_json({"type": "RESTART", "data": ""})
        print("init signal sent")
        init = False
    else:
        print("Sending the next action...")
        cloudyness = 0
        precipitation = 0
        precipitation_deposits = 0
        wind_intensity = 0
        sun_azimuth_angle = 0
        sun_altitude_angle = 0
        socket.send_json({"type":"ACTION", "data":(cloudyness, precipitation, precipitation_deposits, wind_intensity, sun_azimuth_angle, sun_altitude_angle)})
        print("action sent")

    #  Get the reply.
    message = socket.recv_json()
    print("Received next state: %s " % (message))

    # simulate model logic
    time.sleep(0.1)
