import random
import math
from enum import Enum

from mc_sda_pilco import environment
from mc_sda_pilco import util

class NodeType (Enum):
	DecisionNode = 1
	ChanceNode = 2


class MCTSNode:
	"""
	A class to represent a node in the Monte Carlo search tree.
	following the definition of rho-UCT from Veness, et al. 2009
	"""
	# hyper parameters
	exploration_constant = 2.0
	discount_factor = 0.95

	def __init__(self, node_type: NodeType):
		self.visits = 0
		self.mean = 0.0
		self.children = {}
		self.node_type = node_type

	def sun_alt_heuristics(self, state, env:environment.SDAEnv):
		"""
		This heuristics assumes a constant velocity as in the current state and a default policy that always reduce
		sun altitude by 2 in each step.
		It predicts the value of sun altitude at the time of vehicle reaches pedestrian given above assumptions and
		compare with a target sun altitude which we draw from experience.

		prediction > target -> min reward
		otherwise -> max reward
		"""
		# parameters
		target = 2
		threshold = 10

		# current state
		velocity = util.state_velocity_to_raw(state[1])
		dist = env.pedestrian_position - util.state_position_to_raw(state[0])
		sun_altitude = util.state_sun_alt_to_raw(state[3])

		# prediction
		steps_til_reach = dist / velocity / env.time_per_step
		prediction = sun_altitude - steps_til_reach * 2
		diff = prediction - target

		# heuristics value
		final_reward = env.min_reward() if diff > threshold else env.max_reward()
		return final_reward + steps_til_reach * env.step_penalty()

	def rollout(self, init_state, horizon, env: environment.SDAEnv):
		"""
		Use random policy for rollout
		"""
		r = 0.0
		k = 0
		action_space = env.action_space()
		state = init_state
		# conduct one path of rollout
		while k < horizon and not env.is_final_state(state):
			# sample a random action
			action = action_space[random.randint(0, len(action_space)-1)]
			# one-step transition
			state = env.transition(state, action)
			step_r = env.reward(state)
			# update reward and horizon
			r += self.discount_factor * step_r
			k += 1

		return r

	def best_action(self):
		"""
		Return the action with the highest mean
		"""
		best_action = None
		max_mean = None
		for action in self.children:
			if max_mean is None or self.children[action].mean > max_mean:
				max_mean = self.children[action].mean
				best_action = action
		return best_action

	def action_selection(self, horizon, env: environment.SDAEnv):
		"""
		Select action based on UCB policy
		"""
		undiscovered = []
		selected_action = None
		max_score = 0.0
		# find all undiscovered chance nodes, i.e. actions
		for action in env.action_space():
			if action not in self.children:
				undiscovered.append(action)
		# randomly select undiscovered action if there's any
		if len(undiscovered) > 0:
			selected_action = undiscovered[random.randint(0, len(undiscovered)-1)]
		# otherwise, select child of highest UCB score
		else:
			for action in self.children:
				exploit_score = self.children[action].mean / (horizon * (env.max_reward() - env.min_reward()))
				explore_score = self.exploration_constant * math.sqrt(math.log(self.visits)/self.children[action].visits)
				score = exploit_score + explore_score
				if score > max_score:
					max_score = score
					selected_action = action
		return selected_action

	def sample(self, horizon, env: environment.SDAEnv, state=None, action=None):
		# reward obtained from this run of sampling, computed either through rollout or back-propagation
		r = 0.0

		# if horizon reached, return 0
		if horizon == 0:
			return r

		# direct evaluation for final state
		if env.is_final_state(state):
			return env.reward(state)

		# decision node
		if self.node_type == NodeType.DecisionNode:
			# perform rollout if decision node is never visited
			if self.visits == 0:
				# r = self.rollout(state, horizon, env)
				r = self.sun_alt_heuristics(state, env)
			# otherwise, select action and recursively sample
			else:
				new_action = self.action_selection(horizon, env)
				# add to children list if chance node has never been discovered
				if new_action not in self.children:
					self.children[new_action] = MCTSNode(NodeType.ChanceNode)
				# reward from recursively sample
				r = self.children[new_action].sample(horizon, env, state=state, action=new_action)

		# chance node
		else:
			# generate next state
			new_state = env.transition(state, action)
			# check if state is within proximity of explored states
			for explored_state in self.children:
				if env.within_proximity(new_state, explored_state):
					# represent using explored state if they are close enough
					new_state = explored_state
					break
			# step reward of next state
			step_r = env.step_penalty()
			# add to children set if not within proximity of any child
			if new_state not in self.children:
				self.children[new_state] = MCTSNode(NodeType.DecisionNode)
			# reward = step reward + sample reward
			r = step_r + self.discount_factor * self.children[new_state].sample(horizon-1, env, state=new_state)

		# update mean given r
		self.mean = (r + (float(self.visits) * self.mean)) / (float(self.visits) + 1.0)

		# update visits
		self.visits += 1

		# return reward
		return r

