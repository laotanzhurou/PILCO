import numpy as np
import tensorflow as tf
import json
import time
from datetime import datetime
import sys
import argparse
from random import randrange

from carla_client import CarlaClient
from carla_env import CarlaEnv
from environment import SDAEnv

from pilco_gp import PILCOGaussianProcess as pilco_gp
from util import load_pilco_from_files, run_test, next_batch, dump_pilco_to_files, Logger

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


def train_offline(args, file_path="data/training_set"):
	if args.files is not None:
		file_path = args.files

	state_file = open(file_path + "/state.txt", "r").readlines()
	action_file = open(file_path + "/action.txt", "r").readlines()

	# initialisation
	training_sets = int(args.trainings) if args.trainings is not None else 5
	test_sets = int(args.tests) if args.tests is not None else 2
	batch_size = int(args.batch) if args.batch is not None else 1
	dump = bool(args.dump) if args.dump is not None else False
	count = 1
	horizon = 35

	# logging
	log_file_name = "logs/" + "train_" + str(training_sets) + "_test_" + str(test_sets) + "_time_" + str(datetime.now()) + ".log"
	sys.stdout = Logger(log_file_name)

	# keep track of time in each step for perf
	set_index = []
	training_sets_time = []
	test_sets_time = []

	# init
	# skip the first batch
	_, __ = next_batch(state_file, action_file)

	print("### Training Set Count {} ####".format(count))
	state_actions, diffs = next_batch(state_file, action_file)
	pilco = pilco_gp(state_actions, diffs)
	pilco.init()
	print("initialisation complete.\n")

	while count < training_sets:
		count += 1

		# obtain data from new batch
		new_state_actions, new_diffs = next_batch(state_file, action_file)
		new_state_actions = new_state_actions[:horizon, :]
		new_diffs = new_diffs[:horizon, :]

		# add to existing data points
		state_actions = np.vstack((state_actions, new_state_actions))
		diffs = np.vstack((diffs, new_diffs))

		if count % batch_size == 0:
			# optimise model with new data
			start = time.time()
			print("### Training Set Count {} ####".format(count))

			pilco.set_XY(state_actions, diffs)
			model_size = diffs.shape[0] * diffs.shape[1]
			print("Size of model: " + str(model_size))

			pilco.optimise(restarts=3, verbose=True)
			end = time.time()
			print("time taken for optimisation: {} seconds".format(end - start))

			if dump:
				dump_pilco_to_files(pilco, count, model_size)

			# measure prediction accuracy against test set
			test_time = run_test(count, test_sets, horizon, pilco)

			# update time
			set_index.append(count)
			training_sets_time.append(end - start)
			test_sets_time.append(test_time)

	print("Exiting...")

	# plot time taken
	plt.subplot(1, 2, 1)
	plt.title("Training Time")
	plt.plot(np.array(set_index), np.array(training_sets_time))
	plt.xlabel("# of Training Sets")
	plt.ylabel("Time in Seconds")


	plt.subplot(1, 2, 2)
	plt.title("Test Time")
	plt.plot(np.array(set_index), np.array(test_sets_time))
	plt.xlabel("# of Training Sets")
	plt.ylabel("Time in Seconds")

	fig_path = "output/"
	plt.savefig(fig_path + "{} Performance.png".format(str(datetime.now())))
	plt.close()

def run_testset(args, file_path="data/training_set"):
	# init
	state_file = open(file_path + "/state.txt", "r").readlines()
	action_file = open(file_path + "/action.txt", "r").readlines()

	horizon = 35

	model_name = args.load_pilco_model if args.load_pilco_model is not None else 'pilco_50_5382_posY_accY_sunAlt'
	trained_size = int(model_name.split("_")[1])
	test_sets = int(args.tests) if args.tests is not None else 2

	# load serialised PILCO parameters
	model_dump = load_pilco_from_files(model_name)
	print("pre-trained PILCO model loaded...")

	# skip the first batch
	_, __ = next_batch(state_file, action_file)

	# load data points from traing sets
	X, Y = next_batch(state_file, action_file)
	trained_size -= 1
	while trained_size > 0:
		x, y = next_batch(state_file, action_file)
		X = np.vstack((X, x[:horizon, :]))
		Y = np.vstack((Y, y[:horizon, :]))
		trained_size -= 1

	# re-create PILCO
	pilco = pilco_gp.from_dump(model_dump, X, Y)

	# run tests
	run_test(trained_size, test_sets, horizon, pilco, file_path=file_path, display=True, verbose=True)

	print("Exiting...")


def main(args):

	mode = int(args.mode) if args.mode is not None else 1

	if mode == 1:
		print("testing against pre-trained model")
		run_testset(args)
	elif mode == 2:
		print("training offline...")
		train_offline(args)
	else:
		print("training online...")
		train(args.verbose)


# Scripts
if __name__ == '__main__':

	try:
		parser = argparse.ArgumentParser()
		parser.add_argument("--mode", "-m", help="1: test against a pre-trained model, 2: offline training. 3: online training ")
		parser.add_argument("--files", "-f", help="path of data files")
		parser.add_argument("--verbose", "-v", help="path of data files")
		parser.add_argument("--trainings", "-r", help="number of training sets to execute")
		parser.add_argument("--batch", "-b", help="number of batches to execute")
		parser.add_argument("--dump", "-d", help="dumps pilco model after each batch of training is done")
		parser.add_argument("--tests", "-t", help="number of batches to execute")
		parser.add_argument("--load_pilco_model", "-l", help="name of model dump to load")
		args = parser.parse_args()
		main(args)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')

