import zmq
from enum import Enum
import json
import numpy as np

from util import parse_state, parse_action

class CarlaClient:

	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.context = zmq.Context()
		self.socket = None

	def connect(self):
		self.socket = self.context.socket(zmq.REQ)
		self.socket.connect("tcp://%s:%d" % (self.host, self.port))

	def reset_carla(self, episodes_to_skip=0):
		message = {"type": "RESTART", "data": ""}
		# send
		self.socket.send_json(message)
		# wait for reply
		response = self.socket.recv_json()
		# skip the initial episodes after reset
		for i in range(episodes_to_skip):
			message = {"type":"ACTION", "data":(0, 0, 0, 0, 0, 0)}
			self.socket.send_json(message)
			response = self.socket.recv_json()

		state, _ = parse_state(json.dumps(response))
		return state

	def next_episode(self, action):
		message = {"type": CarlaMessage.NEXT_STATE.name, "data": action}
		# send
		self.socket.send_json(message)
		# wait for reply
		response = self.socket.recv_json()
		# parse
		state, _ = parse_state(json.dumps(response))
		return state

class CarlaMessage(Enum):
	# request
	RESTART = 1
	NEXT_STATE = 2
	# response
	STATE = 3
