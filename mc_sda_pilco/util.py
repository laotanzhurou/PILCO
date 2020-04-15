import json
import time
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import pickle


def load_pilco_from_files(training_sets, model_size, file_path="data/models/"):
	path = file_path + 'pilco_' + training_sets + "_" + model_size + '.dump'
	if Path(path).exists():
		with open(path, 'rb') as file:
			return pickle.load(file)


def dump_pilco_to_files(pilco, training_sets, model_size, file_path="data/models/"):
	with open(file_path + 'pilco_' + str(training_sets) + "_" + str(model_size) + '.dump', 'wb+') as file:
		pickle.dump(pilco.mgpr, file)


def run_test(training_set_size, test_sets_size, horizon, pilco, state_file, action_file, display=False):
	all_errors = None
	for t in range(test_sets_size):
		print("\n###Testing set: " + str(t + 1))
		errors = None
		new_state_actions, new_diffs = next_batch(state_file, action_file)

		for i in range(horizon):
			predicted_diffs = pilco.sample(np.stack([new_state_actions[i]]))
			actual_diff = new_diffs[i]

			predict_error = abs((actual_diff - predicted_diffs) / predicted_diffs)

			# cap the error percentage at 1
			normalised_error = np.array(list(map(lambda x: 1 if x > 1 else x, predict_error)))

			if errors is None:
				errors = normalised_error
			else:
				errors = np.vstack((errors, np.array(normalised_error)))

		batch_average = [np.average(errors[:, i]) for i in range(errors.shape[1])]
		print("Batch average errors percentage: " + str(batch_average))

		if all_errors is None:
			all_errors = errors
		else:
			all_errors = np.dstack((all_errors, errors))

	# average errors across all test sets for each time step in each dimension
	average_errors = np.array(
		[[np.average(all_errors[i, j, :]) for j in range(all_errors.shape[1])] for i in range(all_errors.shape[0])])
	print("\nOverall average error percentage: \n" + str(
		[np.average(average_errors[:, i]) for i in range(average_errors.shape[1])]))

	fig, axs = plt.subplots(average_errors.shape[1])
	fig.suptitle("Average Prediction Error % Test set {} Training set {}".format(test_sets_size, training_set_size))
	for i in range(average_errors.shape[1]):
		axs[i].plot(np.arange(0, average_errors.shape[0]), average_errors[:, i])
		axs[i].text(0.05, 0.95, "Average error percent: {}".format(round(np.average(average_errors[:, i]), 3)),
					fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.1))

	if display:
		plt.show()
	else:
		fig_path = "output/"
		fig.savefig(fig_path + "Figure Training sets {} test sets {}.png".format(training_set_size, test_sets_size))
		plt.close(fig)

	print("Test completes.")


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
		action = parse_action(action_file.pop(0), prev_raw_state)
		new_state, _raw = parse_state(state_file.pop(0))

		# skip the first a few frames where vehicle is not moving
		if _raw['vehicle'][1] <= 118:
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

	# all states
	# state = np.hstack(([], [v[0]/100, v[1]/100, v[2]/100, v[3]/5, v[4]/5, v[5]/5]))
	# state = np.hstack((state, [p[0]/100, p[1]/100, p[2]/100, p[3]/5, p[4]/5, p[5]/5]))
	# state = np.hstack(([], [w[0]/100, w[1]/100, w[2]/100, w[3]/100, w[4]/360, w[5]/180 + 0.5]))

	# only pos_y, rain possibility
	state = np.hstack(([], [(v[1]-80) / 40, w[2]/100]))

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

	prev_state_val = raw_state['weather'][2]
	a[2] = max(min(a[2], 100 - prev_state_val), 0 - prev_state_val)

	action = np.hstack(([], [a[2] / 100]))

	return action


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