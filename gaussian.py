import numpy as np
import math

def multi_gaussian(x, mean, covariance):
	"""
	Probability 
	x: 1xD
	mean: 1xD
	covariance: DxD
	"""
	D = len(x)
	x = np.array(x).reshape(1, D)
	mean = np.array(mean).reshape(1, D)
	covariance = np.array(covariance)
	detCov = np.linalg.det(covariance)
	xhat = x - mean
	A = .5 / ((math.pi ** (.5 * D)) * (detCov ** .5))
	B = -.5 * xhat.dot(np.linalg.inv(covariance).dot(xhat.T));
	E = np.exp(B)

	return A * E 

def gaussian(x, mean, variance):
	"""
	x: 1
	mean: 1
	variance: 1
	"""
	A = (2 * math.pi * variance) ** -.5
	B = -.5 * (x - mean) ** 2 / variance

	return A * np.exp(B)

def multi_gaussian_matrix(X, mean, covariance):
	"""
	Probability of N data points
	X: NxD
	mean: 1xD
	covariance: DxD
	"""
	X = np.array(X)
	N = X.shape[0]
	D = X.shape[1]
	# Check dimension
	if (D != len(mean)):
		try:
			raise Exception('Dimension not valid')
		except Exception as exp:
			print exp
		return

	
	mean = np.array(mean).reshape(1, D)
	Xhat = X - mean
	covariance = np.array(covariance)
	detCov = np.linalg.det(covariance)
	invCov = np.linalg.inv(covariance)
	A = .5 / ((math.pi ** (.5 * D)) * (detCov ** .5))
	B = Xhat.dot(invCov)
	C = B.dot(Xhat.T)
	E = -.5 * np.diagonal(C)
	F = np.exp(E)

	return A * F

def test():
	X = [[1, 2, 3], 
		 [4, 5, 7], 
	     [7, 8, 9]]
	mean = [4, 5, 7]
	covariance = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	print multi_gaussian_matrix(X, mean, covariance)

test()
