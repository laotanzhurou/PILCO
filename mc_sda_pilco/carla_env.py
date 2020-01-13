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
		ob, hit_object, hit_pedestrian, is_finished = self.carla_client.reset_carla(init_params=())
		return ob, hit_object, hit_pedestrian, is_finished


	def step(self, action):
		# send next action to Carla and get next observation
		ob, hit_object, hit_pedestrian, is_finished = self.execute_action(action)

		# get reward and is_finished
		new_state = self.observation_to_state(ob)
		r = self.reward(ob, hit_object, hit_pedestrian, is_finished)
		return new_state, r, is_finished

	def execute_action(self, action):
		ob, hit_object, hit_pedestrian, is_finished = self.carla_client.next_episode(action)
		return ob, hit_object, hit_pedestrian, is_finished


	def reward(self, observation, hit_object, hit_pedestrian, is_finished):
		# TODO: determine reward based on observations
		return None


	def observation_to_state(self, observations):
		# TODO
		return

