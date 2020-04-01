import numpy as np
from mc_sda_pilco.pilco_gp import PILCOGaussianProcess as pilco_gp

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def test():
	# init
	state_actions, diffs = test_data(s=0.5, T=100)
	gp = pilco_gp(state_actions, diffs)
	gp.init()
	gp.optimise(verbose=False)

	# training
	# state_actions, diffs = test_data(s=0.5, T=1000)
	# gp.set_XY(state_actions, diffs)
	# gp.optimise(restarts=5, verbose=True)

	# testing
	test_X, test_Y = test_data(s=0.8, T=100)
	predicts = gp.samples(test_X)
	errors = np.absolute((test_Y - predicts) / predicts).flatten()

	print("All errors: {}".format(errors))
	print("Average: {}".format(np.average(errors)))

	plt.title("Predicts vs. Actual")
	plt.xlabel("Step")
	plt.ylabel("Value")
	plt.plot(np.arange(0, test_Y.shape[0]), test_Y.flatten())
	plt.plot(np.arange(0, test_Y.shape[0]), predicts.flatten())
	plt.legend(['Predicts', "Actual"])

	plt.show()

def test_data(s=0.5, T=500):
	state_actions = []
	diffs = []
	for i in range(T):
		a = np.random.rand() / 10
		new_s = test_transition(s, a)

		state_actions.append([s, a])
		diffs.append([new_s - s])

		s = new_s

	state_actions = np.array(state_actions)
	diffs = np.array(diffs)
	return state_actions, diffs


def test_transition(state, action):
	return state + action


# Scripts
test()