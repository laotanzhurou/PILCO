import zmq
from enum import Enum

# context = zmq.Context()
#
# #  Socket to talk to server
# print("Connect to server...")
# socket = context.socket(zmq.REQ)
# socket.connect("tcp://localhost:5555")
#
# #  Do 10 requests, waiting each time for a response
# for request in range(10):
# 	print("Sending request %s" % request)
# 	socket.send_string("Hello " + str(request))
#
# 	#  Get the reply.
# 	message = socket.recv()
# 	print("Received reply %s [ %s ]" % (request, message))


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
		message = {"type": CarlaMessage.RESTART, "data": init_params}
		# send
		self.socket.send_pyobj(message)
		# wait for reply
		response = self.socket.recv_pyobj()
		ob, hit_object, hit_pedestrian, is_finished = response["data"]
		return ob, hit_object, hit_pedestrian, is_finished

	def next_episode(self, action):
		message = {"type": CarlaMessage.NEXT_STATE, "data": action}
		# send
		self.socket.send_pyobj(message)
		# wait for reply
		response = self.socket.recv_pyobj()

		ob, hit_object, hit_pedestrian, is_finished = response["data"]
		return ob, hit_object, hit_pedestrian, is_finished


class CarlaMessage(Enum):
	# request
	RESTART = 1
	NEXT_STATE = 2
	# response
	STATE = 3


