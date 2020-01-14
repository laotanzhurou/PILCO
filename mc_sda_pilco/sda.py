import numpy as np
import tensorflow as tf
import json
import time
import sys
import argparse
from random import randrange

from carla_client import CarlaClient
from carla_env import CarlaEnv
from environment import SDAEnv
from pilco_gp import PILCOGaussianProcess as pilco_gp


def rollout(env: SDAEnv, horizon, verbose=False):
	# reset environment
	state = env.reset()

	# obtain history from rollouts
	state_actions = []
	diffs = []
	for t in range(horizon):
		new_action = sample_action(env)

		# TODO：　
		new_state = env.step(new_action)

		state_actions.append(np.hstack((state, new_action)))
		diffs.append(new_state - state)

		if verbose:
			print("Action: ", new_action)
			print("State : ", new_state)

		# TODO: determine reward and termination from state
		# if done:
		# 	break

	return np.stack(state_actions), np.stack(diffs)


def sample_action(env: SDAEnv):
	cloudyness = randrange(-5, 5)
	precipitation = randrange(-5, 5)
	precipitation_deposits = randrange(-5, 5)
	wind_intensity = randrange(-5, 5)
	sun_azimuth_angle = randrange(-4, 4)
	sun_altitude_angle = randrange(-15, 15)
	return cloudyness, precipitation, precipitation_deposits, wind_intensity, sun_azimuth_angle, sun_altitude_angle


def train(verbose=False):
	with tf.Session() as session:
		# hyper parameters
		action_dimension = 5
		n_pedestrian = 1
		state_dimension = n_pedestrian * 2 + 2 + 1 + action_dimension
		carla_episode_time = 1000
		rollout_horizon = 150
		iterations = 10
		T = 20

		# carla connection
		carla_host = "localhost"
		carla_port = 5555

		# setup
		carla_client = CarlaClient(carla_host, carla_port)
		carla_client.connect()
		env = CarlaEnv(carla_client, action_dimension, state_dimension, carla_episode_time)

		# train model from random rolling out
		print("Initiate batch...")
		state_actions, diffs = rollout(env, rollout_horizon, verbose)
		pilco = pilco_gp(state_actions, diffs)
		pilco.init()
		pilco.optimise(restarts=1)

		for _ in range(iterations):
			print("Batch {} training start...".format(_ + 1))

			new_state_actions, new_diffs = rollout(env, rollout_horizon, verbose)

			state_actions = np.vstack((state_actions, new_state_actions[:T, :]))
			diffs = np.vstack((diffs, new_diffs[:T, :]))

			pilco.set_XY(state_actions, diffs)
			pilco.optimise(restarts=1)

		# optimise policy
		# TODO:

		print("exiting sda...")


def train_offline(args, file_path="data"):
	if args.files is not None:
		file_path = args.files

	state_file = open(file_path + "/state.txt", "r").readlines()
	action_file = open(file_path + "/action.txt", "r").readlines()

	# initialisation
	pilco = None
	init = True
	count = 1
	T = 20

	# fetch the next batch
	state_actions, diffs = next_batch(state_file, action_file)
	while state_actions is not None and diffs is not None:
		start = time.time()
		print("### Training Batch {} ####".format(count))
		if init:
			pilco = pilco_gp(state_actions, diffs)
			pilco.init()
			init = False
		else:
			new_state_actions, new_diffs = next_batch(state_file, action_file)

			state_actions = np.vstack((state_actions, new_state_actions[:T, :]))
			diffs = np.vstack((diffs, new_diffs[:T, :]))

			pilco.set_XY(state_actions, diffs)

		pilco.optimise(restarts=1)
		count += 1
		print("time taken for batch: {} seconds".format(time.time() - start))

	# Sample from model
	# TODO:
	# if pilco is not None:
	# 	pilco.sample()

	print("Exiting...")



def next_batch(state_file, action_file):
	heading = "initiating"

	# init
	state_actions = []
	diffs = []
	state = None
	if len(state_file) > 0 and len(action_file) > 0 and heading in state_file[0] and heading in action_file[0]:
		state_file.pop(0)
		action_file.pop(0)
		state = parse_state(state_file.pop(0))

	# loop
	while True:
		# Terminal condition: next line is heading or no next line
		if len(state_file) == 0 or heading in state_file[0]:
			break

		# Loop logic: add state and action to list
		new_state = parse_state(state_file.pop(0))
		action = parse_action(action_file.pop(0))

		state_actions.append(np.hstack((state, action)))
		diffs.append(new_state - state)

		state = new_state

	return np.stack(state_actions), np.stack(diffs)


def parse_state(raw_state):
	state_json = json.loads(raw_state)
	data = state_json['data']

	state = np.hstack(([], data['vehicle']))
	state = np.hstack((state, data['peds']))
	state = np.hstack((state, data['weather']))

	return state


def parse_action(raw_action):
	action_json = json.loads(raw_action)
	data = action_json['data']
	action = np.hstack(([], data))
	return action


def main(args):

	if args.offline:

		print("training offline...")
		train_offline(args)
	else:
		print("training online...")
		train(args.verbose)


# Scripts
if __name__ == '__main__':

	try:
		parser = argparse.ArgumentParser()
		parser.add_argument("--offline", "-o", help="train model in offline mode")
		parser.add_argument("--files", "-f", help="path of data files")
		parser.add_argument("--verbose", "-v", help="path of data files")
		args = parser.parse_args()
		main(args)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')

