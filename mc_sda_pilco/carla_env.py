from environment import SDAEnv
from carla_client import CarlaClient


class CarlaEnv(SDAEnv):

	def __init__(self, carla_client:CarlaClient, action_dimension, state_dimension, episode_horizon):
		super(action_dimension, state_dimension)
		self.time_horizon = episode_horizon
		self.carla_client = carla_client

		# init carla connection
		self.carla_client.connect()

		return

	def reset(self):
		# TODO: to pass init_params
		state = self.carla_client.reset_carla(init_params=())
		return state


	def step(self, action):
		# send next action to Carla and get next observation
		new_state = self.execute_action(action)

		# get reward and is_finished
		# TODO:
		# r = self.reward(ob, hit_object, hit_pedestrian, is_finished)
		return new_state

	def execute_action(self, action):
		new_state = self.carla_client.next_episode(action)
		return new_state


	def reward(self, state):
		# TODO: determine reward based on observations
		return None

