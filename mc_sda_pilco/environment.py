import numpy as np
from numpy import linalg as la

from mc_sda_pilco import util

class SDAEnv:


	time_per_step = 0.05
	pedestrian_position = 80
	planning_start_position = 90
	collision_position_threshold = 82.62
	collision_velocity_threshold = 0.1
	proximity_threshold = 0.1

	def __init__(self, pilco):
		self.pilco = pilco

	def transition(self, state, action):
		# sample a delta from GP models
		state_action = np.hstack((state, action))
		delta = self.pilco.sample(np.stack([state_action]))

		# calculate the new state
		new_state = state + delta

		return tuple(new_state)

	def reward(self, state):
		"""
		Reward function is defined as:
		-10 - if vehicle stops without crashing
		10 - crash
		-1 - step penalty
		"""
		raw_position = util.state_position_to_raw(state[0])
		raw_velocity = util.state_velocity_to_raw(state[1])

		# non final state
		if not self.is_final_state(state):
			return self.step_penalty()

		# crash
		if abs(raw_velocity) > self.collision_velocity_threshold:
			return self.max_reward()
		# stopped
		else:
			return self.min_reward()

	def max_reward(self):
		return 10.0

	def min_reward(self):
		return -10

	def step_penalty(self):
		return -1

	def is_final_state(self, state):
		"""
		Either vehicle has stopped or crush happened
		"""
		raw_position = util.state_position_to_raw(state[0])
		return raw_position < self.collision_position_threshold

	def action_space(self):
		"""
		Action of increase/decrease sun altitude
		"""
		return [(util.raw_sun_alt_to_action(2)), (util.raw_sun_alt_to_action(-2))]

	def within_proximity(self, a, b):
		"""
		Calculate Euclidean distance between a and b, if lower than proximity threshold
		a could be considered same as b
		"""
		euclidean_dist = la.norm(np.array(a)-np.array(b))
		return euclidean_dist > self.proximity_threshold
