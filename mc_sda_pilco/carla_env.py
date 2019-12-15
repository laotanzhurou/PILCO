from .env import SDAEnv


class CarlaEnv(SDAEnv):

	def __init__(self, action_dimension, state_dimension, episode_horizon):
		super(action_dimension, state_dimension)
		self.time_horizon = episode_horizon
		# TODO
		return

	def step(self, action):
		# convert action to Carla command and execute
		self.execute_action(action)

		# get next state
		ob, r, is_finished = self.next_episode()
		new_state = self.observation_to_state(ob)

		# get reward and is_finished
		return new_state, r, is_finished

	def reset(self):
		# TODO: get an initial state from Carla and convert to env state
		self.pause_world()
		return

	def observation_to_state(self, observations):
		# TODO
		return

	def execute_action(self, action):
		# TODO
		return

	def next_episode(self, time_horizon):
		self.pause_world()
		# TODO: called after command executed, sample the next episode from Carla
		ob = None

		# obtain reward and is_finished from observation
		r = self.reward(ob)
		is_finished = self.is_finished(ob)
		return ob, r, is_finished

	def pause_world(self):
		# TODO
		return

	def reward(self, observations):
		# TODO: determine reward based on observations
		return None

	def is_finished(self, observation):
		# TODO:
		return False
