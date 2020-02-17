import zmq
from enum import Enum
import json
import numpy as np


def parse_state(data):
	state = np.hstack(([], data['vehicle']))
	state = np.hstack((state, data['peds']))
	state = np.hstack((state, data['weather']))
	return state


def parse_action(raw_action):
	action_json = json.loads(raw_action)
	data = action_json['data']
	action = np.hstack(([], data))
	return action


class CarlaClient:

	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.context = zmq.Context()
		self.socket = None

	def connect(self):
		self.socket = self.context.socket(zmq.REQ)
		self.socket.connect("tcp://%s:%d" % (self.host, self.port))
		print("connected to Carla...")

	def reset_carla(self, init_params=()):
		# TODO: to define init_params types
		message = {"type": "RESTART", "data": ""}
		# send
		self.socket.send_json(message)
		# wait for reply
		response = self.socket.recv_json()

		state = parse_state(response["data"])
		return state

	def next_episode(self, action):
		message = {"type": CarlaMessage.NEXT_STATE.name, "data": action}
		# send
		self.socket.send_json(message)
		# wait for reply
		response = self.socket.recv_json()

		# parse
		state = parse_state(response["data"])
		return state


class CarlaMessage(Enum):
	# request
	RESTART = 1
	NEXT_STATE = 2
	# response
	STATE = 3
