import numpy as np
from gym import spaces


class SDAEnv:

	def __init__(self, action_dimension, state_dimension):
		self.action_dimension = action_dimension
		self.state_dimension = state_dimension

		self.action_space = spaces.Box(np.empty((1, action_dimension)).fill(-1), np.empty((1, action_dimension)).fill(1), dtype=np.float32)
		self.state_space = spaces.Box(np.empty((0, state_dimension)).fill(-1), np.empty((1, state_dimension)).fill(1), dtype=np.float32)

	def step(self, action):
		# implemented in CarEnv
		return None

	def reset(self):
		# implemented in CarEnv
		return


