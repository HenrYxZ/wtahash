import numpy as np

class RandomDataGenerator:
	"""Generates an array with random numbers using the input parameters

	Attributes:
		data (matrix of floats): contains classifiers models of n dimensions
	"""
	def __init__(self, classifiers_count, n, min_number, max_number):
		random_coefficients = np.random.random_sample((classifiers_count, n))
		self.data = random_coefficients * (max_number - min_number) + min_number

	def get_data(self):
		return self.data

	def export_csv(self, filename):
		np.savetxt(filename, self.data, fmt = "%.6f", delimiter = ",")