import json
from time import time
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import pickle


def load_pilco_from_files(name, file_path="data/models/"):
	path = file_path + name + '.dump'
	if Path(path).exists():
		with open(path, 'rb') as file:
			return pickle.load(file)


def dump_pilco_to_files(pilco, training_sets, model_size, file_path="output/"):
	with open(file_path + 'pilco_' + str(training_sets) + "_" + str(model_size) + '.dump', 'wb+') as file:
		pickle.dump(pilco.snapshot(training_sets), file)


def run_test(training_set_size, test_sets_size, horizon, pilco, file_path="data/test_set", display=True, verbose=False):

	state_file = open(file_path + "/state.txt", "r").readlines()
	action_file = open(file_path + "/action.txt", "r").readlines()
	all_errors = None
	dimensions = 0

	total_runtime = 0

	# skip the first test case
	_, __ = next_batch(state_file, action_file)

	for t in range(test_sets_size):
		print("\n###Testing set: " + str(t + 1))
		start = time()
		errors = None
		new_state_actions, new_diffs = next_batch(state_file, action_file)

		if dimensions == 0:
			dimensions = new_diffs.shape[1]

		# cap horizon at data size
		horizon = min(len(new_diffs), horizon)
		print("test time steps: " + str(horizon))

		for i in range(horizon):
			sample_start = time()
			predicted_diffs = pilco.sample(np.stack([new_state_actions[i]]))
			sample_end = time()
			if verbose:
				print("Time taken for sample in episode {}: {} seconds".format(i, sample_end - sample_start))

			actual_diff = new_diffs[i]

			predict_error = abs((actual_diff - predicted_diffs) / predicted_diffs)

			# cap the error percentage at 1
			normalised_error = np.array(list(map(lambda x: 1 if x > 1 else x, predict_error)))

			if errors is None:
				errors = normalised_error
			else:
				errors = np.vstack((errors, np.array(normalised_error)))

		end = time()
		batch_average = [np.average(errors[:, i]) for i in range(errors.shape[1])]
		print("Batch average errors percentage: " + str(batch_average))
		print("Time taken for test: {} seconds".format(str(end - start)))
		total_runtime += end - start


		if all_errors is None:
			all_errors = errors
		else:
			all_errors = np.dstack((all_errors, errors))

	# average errors across all test sets for each time step in each dimension
	average_errors = np.array(
		[[np.average(all_errors[i, j, :]) for j in range(all_errors.shape[1])] for i in range(all_errors.shape[0])])
	print("\nOverall average error percentage: \n" + str(
		[np.average(average_errors[:, i]) for i in range(average_errors.shape[1])]))

	if display:
		fig, axs = plt.subplots(average_errors.shape[1])
		fig.suptitle("Error % Test set {} Training set {} Pos_Y, Vel_Y, Acc_Y, Sun_Alti".format(test_sets_size, training_set_size))

		for i in range(average_errors.shape[1]):
			axs[i].plot(np.arange(0, average_errors.shape[0]), average_errors[:, i])
			axs[i].legend(["average error: {}".format(round(np.average(average_errors[:, i]), 3))])

		fig_path = "output/"
		fig.savefig(fig_path + "{} Dimensions {} Training sets {} test sets {}.png".format(str(datetime.now()), str(dimensions), training_set_size, test_sets_size))
		plt.close(fig)

		print("Test completes. \n")

	return total_runtime / test_sets_size


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

	# loop
	while True:
		# Terminal condition: next line is heading or no next line
		if len(state_file) == 0 or heading in state_file[0]:
			break

		# Loop logic: add state and action to list
		action, _raw_action = parse_action(action_file.pop(0), prev_raw_state)
		new_state, _raw_state = parse_state(state_file.pop(0))

		# skip frames of starting as well as after vehicle stopped
		if _raw_state['vehicle'][2] < 95:
			state_actions.append(np.hstack((state, action)))
			diffs.append(new_state - state)

		state = new_state
		prev_raw_state = _raw_state

	return np.stack(state_actions), np.stack(diffs)


def parse_state(raw_state):
	state_json = json.loads(raw_state)
	data = state_json['data']

	v = data['vehicle']
	p = data['peds']
	w = data['weather']

	# all states
	# state = np.hstack(([], [v[0]/100, v[1]/100, v[2]/100, v[3]/5, v[4]/5, v[5]/5]))
	# state = np.hstack((state, [p[0]/100, p[1]/100, p[2]/100, p[3]/5, p[4]/5, p[5]/5]))
	# state = np.hstack(([], [w[0]/100, w[1]/100, w[2]/100, w[3]/100, w[4]/360, w[5]/180 + 0.5]))

	# only pos_y, velocity_y, rain possibility
	pos_y = (-v[1]+100) / 20  # range 80 ~ 100
	vel_y = -v[4]/ 10  # range -10 ~ 0, reversed direction
	acc_y = (-v[7]+20) / 30  # range -10 ~ 20, reversed direction
	sun_altitude = (w[5]+15)/60  # range -15 ~ 45
	state = np.hstack(([], [pos_y, vel_y, acc_y, sun_altitude]))

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

	# precipitation deposit
	# prev_state_val = raw_state['weather'][2]
	# a[2] = max(min(a[2], 100 - prev_state_val), 0 - prev_state_val)
	# action = np.hstack(([], [a[2] / 100]))

	# sun altitude
	action = np.hstack(([], [a[5] / 60]))

	return action, a


def construct_action(action):
	out = (0, 0, 0, 0, 0, action[0])
	return out


class Logger(object):
	def __init__(self, logfile):
		self.terminal = sys.stdout
		self.log = open(logfile, "a+")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		#this flush method is needed for python 3 compatibility.
		#this handles the flush command by doing nothing.
		#you might want to specify some extra behavior here.
		pass