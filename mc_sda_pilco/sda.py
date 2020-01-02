import tensorflow as tf
import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
import tensorflow as tf
from .carla_env import CarlaEnv
from .carla_client import CarlaClient
from .env import SDAEnv
from .pilco_gp import PILCOGaussianProcess as pilco_gp

def rollout(env: SDAEnv, horizon, verbose=False):
	# reset environment
	x = env.reset()

	# obtain history from rollouts
	X = []
	Y = []
	for t in range(horizon):
		u = sample_action(env)
		x_new, _, done, _ = env.step(u)
		if verbose:
			print("Action: ", u)
			print("State : ", x_new)
		X.append(np.hstack((x, u)))
		Y.append(x_new - x)
		x = x_new
		if done:
			break
	return np.stack(X), np.stack(Y)


def sample_action(env: SDAEnv):
	return env.action_space.sample()


with tf.Session() as session:
	# hyper parameters
	action_dimension = 5
	n_pedestrian = 1
	state_dimension = n_pedestrian * 2 + 2 + 1 + action_dimension
	carla_episode_time = 1000
	rollout_horizon = 100
	iterations = 10

	# carla connection
	carla_host = "localhost"
	carla_port = 5555


	# setup
	carla_client = CarlaClient(carla_host, carla_port)
	env = CarlaEnv(carla_client, action_dimension, state_dimension, carla_episode_time)

	# collecting data from random rolling out
	X = []
	Y = []
	for _ in range(iterations):
		x, y = rollout(env, rollout_horizon, verbose=True)
		X.append(x)
		Y.append(y)
	X = np.array(X)
	Y = np.array(Y)

	# training transition model
	pilco = pilco_gp(X, Y)
	pilco.init()
	pilco.optimise(restarts=1)

