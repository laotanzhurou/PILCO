import pandas as pd
from pilco import models
import numpy as np

class PILCOGaussianProcess:

	def __init__(self, X, Y):
		self.mgpr = models.MGPR(X, Y)
		return

	def init(self):
		for model in self.mgpr.models:
			model.likelihood.variance = 0.001
			model.likelihood.variance.trainable = False

	def optimise(self, restarts=1, verbose=True):
		self.mgpr.optimize(restarts=restarts)
		if verbose:
			lengthscales = {}
			variances = {}
			noises = {}
			i = 0
			for model in self.mgpr.models:
				lengthscales['GP' + str(i)] = model.kern.lengthscales.value
				variances['GP' + str(i)] = np.array([model.kern.variance.value])
				noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
				i += 1
			print('-----Learned models------')
			pd.set_option('precision', 3)
			print('---Lengthscales---')
			print(pd.DataFrame(data=lengthscales))
			print('---Variances---')
			print(pd.DataFrame(data=variances))
			print('---Noises---')
			print(pd.DataFrame(data=noises))

	def set_XY(self, X, Y):
		self.mgpr.set_XY(X, Y)

	def sample(self):
		# TODO:
		return None
