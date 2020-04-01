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

import matplotlib.pyplot as plt

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
	target_batches = 5 # keep this value less than actual number of batches available for testing sample
	count = 1
	T = 149

	# fetch the next batch
	state_actions, diffs = next_batch(state_file, action_file)
	if state_actions is None or diffs is None:
		raise Exception("State data file is empty")

	while True:
		start = time.time()
		print("### Training Batch {} ####".format(count))
		if init:
			pilco = pilco_gp(state_actions, diffs)
			pilco.init()
			init = False
		else:
			if count > target_batches:
				break
			new_state_actions, new_diffs = next_batch(state_file, action_file)
			new_state_actions = new_state_actions[:T, :]
			new_diffs = new_diffs[:T, :]

			state_actions = np.vstack((state_actions, new_state_actions[:T, :]))
			diffs = np.vstack((diffs, new_diffs[:T, :]))

			# pilco.set_XY(state_actions, diffs)
			print("Size of model: " + str(len(diffs)))
			pilco.set_XY(state_actions, diffs)

		pilco.optimise(restarts=5, verbose=True)
		count += 1
		print("time taken for batch: {} seconds".format(time.time() - start))

	# Sample from model
	if pilco is not None:

		errors = None
		new_state_actions, new_diffs = next_batch(state_file, action_file)
		if new_state_actions is not None:
			for i in range(len(new_state_actions)):
				if i != len(new_state_actions) - 1:
					predicted_diffs = pilco.sample(np.stack([new_state_actions[i]]))
					actual_diff = new_diffs[i]

					print("Diffs predicted: " + str(predicted_diffs))
					print("Actual diff: " + str(actual_diff))

					batch_errors = abs((actual_diff - predicted_diffs) / predicted_diffs)
					print("Error percentage: " + str(batch_errors))

					if errors is None:
						errors = batch_errors
					else:
						errors = np.dstack((errors, np.array(batch_errors)))

		print("Errors percentage average: " + str([np.average(errors[0][i]) for i in range(errors.shape[1])]))

		fig, axs = plt.subplots(errors.shape[1])
		fig.suptitle("Prediction Error in Percentage")
		for i in range(errors.shape[1]):
			axs[i].plot(np.arange(0, errors.shape[2]), np.array(list(map(lambda x: 1 if x > 1 else x, errors[0][i]))))
		plt.show()


	print("Exiting...")


def next_batch(state_file, action_file):
	heading = "initiating"

	# init
	state_actions = []
	diffs = []
	state = None
	prev_raw_state = None
	if len(state_file) == 0 or len(action_file) == 0:
		return None, None

	if heading in state_file[0]:
		state_file.pop(0)
		state, prev_raw_state = parse_state(state_file.pop(0))

	# if heading in state_file[0] and heading in action_file[0]:
	# 	state_file.pop(0)
	# 	action_file.pop(0)
	# 	state, prev_raw_state = parse_state(state_file.pop(0))

	# loop
	while True:
		# Terminal condition: next line is heading or no next line
		if len(state_file) == 0 or heading in state_file[0]:
			break

		# Loop logic: add state and action to list
		action = parse_action(action_file.pop(0), prev_raw_state)
		new_state, _raw = parse_state(state_file.pop(0))


		state_actions.append(np.hstack((state, action)))
		diffs.append(new_state - state)

		state = new_state
		prev_raw_state = _raw

	return np.stack(state_actions), np.stack(diffs)


def parse_state(raw_state):
	state_json = json.loads(raw_state)
	data = state_json['data']

	v = data['vehicle']
	p = data['peds']
	w = data['weather']

	# state = np.hstack(([], [v[0]/100, v[1]/100, v[2]/100, v[3]/5, v[4]/5, v[5]/5]))
	# state = np.hstack((state, [p[0]/100, p[1]/100, p[2]/100, p[3]/5, p[4]/5, p[5]/5]))
	# state = np.hstack(([], [w[0]/100, w[1]/100, w[2]/100, w[3]/100, w[4]/360, w[5]/180 + 0.5]))

	# only pos_x, pos_y, rain possibility
	# state = np.hstack(([], [v[0]/6747.5, v[1]/5656.2, w[2]/100]))

	# only pos_x, rain possibility
	state = np.hstack(([], [(v[0]-2)*100, w[2]/100]))

	# only pos_x
	# max - min = 0.0028672218322753906
	# init = 2
	# state = np.hstack(([], [ (v[0]-2) * 100 ]))

	return state, data


def parse_action(raw_action, raw_state):
	action_json = json.loads(raw_action)
	a = action_json['data']

	# test offsetting invalid actions to 0, e.g. sun altitude angle -5 when it's already 0
	# for i in range(6):
	# 	prev_state_val = raw_state['weather'][i]
	# 	if i < 4:
	# 		# guard for lower bound
	# 		if prev_state_val + a[i] <= 0:
	# 			a[i] = max(a[i], 0 - prev_state_val)
	# 		# guard for upper bound
	# 		if prev_state_val + a[i] >= 100:
	# 			a[i] = min(a[i], 100 - prev_state_val)
	# 	elif i == 4:
	# 		if prev_state_val + a[i] <= 0:
	# 			a[i] = max(a[i], 0 - prev_state_val)
	# 		if prev_state_val + a[i] >= 360:
	# 			a[i] = min(a[i], 360 - prev_state_val)
	# 	elif i == 5:
	# 		if prev_state_val + a[i] <= -90:
	# 			a[i] = max(a[i], (-90) - prev_state_val)
	# 		if prev_state_val + a[i] >= 90:
	# 			a[i] = min(a[i], 90 - prev_state_val)
	#
	# action = np.hstack(([], [a[0] / 100, a[1] / 100, a[2] / 100, a[3] / 100, a[4] / 360, a[5] / 180]))

	prev_vehicle_pos_x = raw_state['vehicle'][0]
	prev_state_val = raw_state['weather'][2]
	a[2] = max(min(a[2], 100 - prev_state_val), 0 - prev_state_val)

	action = np.hstack(([], [a[2] / 100]))
	# action = np.hstack(([], (prev_vehicle_pos_x-2) * 100 ))

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

