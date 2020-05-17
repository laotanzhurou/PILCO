import numpy as np
from numpy import linalg as la

class SDAEnv:

	pedestrian_position = 1
	distance_threshold = 0.15
	proximity_threshold = 0.1

	def __init__(self, pilco, distance_threshold=None):
		self.pilco = pilco
		if distance_threshold is not None:
			self.distance_threshold = distance_threshold

	def transition(self, state, action):
		# sample a delta from GP models
		state_action = np.hstack((state, action))
		delta = self.pilco.sample(state_action)

		# calculate the new state
		new_state = state + delta

		return new_state

	def reward(self, state):
		"""
		Reward function is defined as:
		0 - if distance between vehicle and pedestrian is above threshold or vehicle has stopped
		10 - otherwise, i.e. vehicle is close to pedestrian but not stopped
		"""
		distance = 1 - state[0]
		velocity = state[1]
		if distance > self.distance_threshold or velocity <= 0:
			return 0.0
		else:
			return 10.0

	def max_reward(self):
		return 10.0

	def min_reward(self):
		return 0.0

	def is_final_state(self, state):
		"""
		Either vehicle has stopped or distance lower than threshold
		"""
		distance = 1 - state[0]
		velocity = state[1]
		return velocity <= 0 or distance <= self.distance_threshold

	def action_space(self):
		"""
		Action of increase/decrease sun altitude
		"""
		return [[2], [-2]]

	def within_proximity(self, a, b):
		"""
		Calculate Euclidean distance between a and b, if lower than proximity threshold
		a could be considered same as b
		"""
		euclidean_dist = la.norm(a, b)
		return euclidean_dist > self.proximity_threshold
