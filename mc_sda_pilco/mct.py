import random
import math
from enum import Enum
from mc_sda_pilco import environment

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
	discount_factor = 1.0

	def __init__(self, node_type: NodeType):
		self.visits = 0
		self.mean = 0.0
		self.children = {}
		self.node_type = node_type

	def rollout(self, horizon, env: environment.SDAEnv):
		"""
		Use random policy for rollout
		"""
		r = 0.0
		k = 0
		action_space = env.action_space()
		state = self.state
		# conduct one path of rollout
		while k < horizon and not env.is_final_state(state):
			# sample a random action
			action = action_space[random.randint(0, len(action_space))]
			# one-step transition
			state = env.transition(action, state)
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
		# find all undiscovered chance nodes
		for action in env.action_space():
			if action not in self.children:
				undiscovered.append(action)
		# randomly select undiscovered action if there's any
		if len(undiscovered) > 0:
			selected_action = undiscovered[random.randint(0, len(undiscovered))]
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

		# if horizon reached or state is final, return 0
		if horizon == 0 or env.is_final_state(state):
			return r

		# decision node
		if self.node_type == NodeType.DecisionNode:
			# perform rollout if decision node is never visited
			if self.visits == 0:
				r = self.rollout(horizon, env)
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
			step_r = env.reward(new_state)
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

