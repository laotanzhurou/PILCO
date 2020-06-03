import pandas as pd
from pilco import models
import numpy as np
import time

class PILCOGaussianProcess:

	def __init__(self, X, Y):
		self.mgpr = models.MGPR(X, Y)
		return

	def init(self):
		for model in self.mgpr.models:
			model.likelihood.variance = 0.001
			model.likelihood.variance.trainable = False

	def optimise(self, restarts=1, verbose=False):
		self.mgpr.optimize(restarts=restarts, verbose=verbose)
		# if verbose:
		# 	lengthscales = {}
		# 	variances = {}
		# 	noises = {}
		# 	i = 0
		# 	for model in self.mgpr.models:
		# 		lengthscales['GP' + str(i)] = model.kern.lengthscales.value
		# 		variances['GP' + str(i)] = np.array([model.kern.variance.value])
		# 		noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
		# 		i += 1
		# 	print('-----Learned models------')
		# 	pd.set_option('precision', 3)
		# 	# print('---Lengthscales---')
		# 	# print(pd.DataFrame(data=lengthscales))
		# 	print('---Variances---')
		# 	print(pd.DataFrame(data=variances))
		# 	print('---Noises---')
		# 	print(pd.DataFrame(data=noises))

	def set_XY(self, X, Y):
		self.mgpr.set_XY(X, Y)

	def sample(self, x, verbose=False):
		start = time.time()
		y = list(map(lambda m: m.predict_f_samples(x, 1).flat[0], self.mgpr.models))
		if verbose:
			print("Time taken for sampling: " + str(time.time() - start) + " seconds")
		return np.stack(y)

	def sample_f(self, x, verbose=False):
		start = time.time()
		y = []
		for m in self.mgpr.models:
			mu, sigma = m.predict_f(x)
			mu = mu.flatten()[0]
			sigma = sigma.flatten()[0]
			y.append(np.random.normal(mu, sigma, 1))
		if verbose:
			print("Time taken for sampling: " + str(time.time() - start) + " seconds")
		return np.array(y).flatten()

	def samples(self, X):
		return np.array([self.sample( x.reshape(1, len(x)) ) for x in X])

	def snapshot(self, training_set_count):
		parameters = []
		for model in self.mgpr.models:
			session = model.enquire_session(None)
			best_parameters = model.read_values(session=session)
			parameters.append(best_parameters)
		dump = ModelDump(training_set_count, parameters)
		return dump


	@staticmethod
	def from_dump(dump, X, Y):
		pilco = PILCOGaussianProcess(X, Y)
		for model, best_params in zip(pilco.mgpr.models, dump.parameters):
			model.assign(best_params)
		return pilco

class ModelDump:
	def __init__(self, size, parameters):
		self.size = size
		self.parameters = parameters
