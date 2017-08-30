import numpy as np
import math

def multiGaussian(x, mean, covariance):
	x = np.array(x)
	mean = np.array(mean)
	covariance = np.array(covariance)

	D = len(x)
	detCov = float(np.linalg.det(covariance))
	A = .5 / ((math.pi ** (.5 * D)) * (detCov ** .5))
	B = -.5*(x-mean).T.dot(np.linalg.inv(covariance).dot(x-mean));
	E = np.exp(B)

	return A * E 

def gaussian(x, mean, variance):
	A = (2 * math.pi * variance) ** -.5
	B = -.5 * (x - mean) ** 2 / variance

	return A * np.exp(B)