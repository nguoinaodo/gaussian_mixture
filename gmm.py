import numpy as np 
from gaussian import multiGaussian, gaussian

class GaussianMixture:
	def __init__(self, K):
		self.K = K

	# Estimate model parameters with the EM algorithm.
	def fit(X):
		# Data
		self.Xtr = np.array(X)
		# Data dimension
		self.D = self.Xtr.shape[1]
		self.N = self.Xtr.shape[0]
		# Mean index
		self.pi = np.random.rand(K)
		# Means and covariances
		means = []
		covariances = []
		for k in range(K):
			# Random mean
			rand_mean = np.random.rand(D)
			means.append(rand_mean)
			# Random covariance
			var = np.random.rand()
			rand_cov = var * np.eye(K)
			covariances.append(rand_cov)
		self.means = np.array(means)
		self.covariances = np.array(covariances)

		# EM algorithm
		for i in range(10000):
			# Theta fixed, find to
			to_matrix = np.zeros((self.N, self.K))
			for n in range(self.N):
				for k in range(self.K):
					to_matrix[n, k] = self.pi[k] * multiGaussian(X[n], self.means[k], \
							self.covariances[k])
			to_matrix = 1. * to_matrix / np.sum(to_matrix, axis = 0)		
			
			# Fix to, find theta
			for k in range(self.K):
				sum_to_k = np.sum(to_matrix[:, k])
				self.means[k] = to_matrix[:, k].T.dot(X) / sum_to_k
				self.covariances[k] = to_matrix[:, k].T.dot((X - means[k]).dot((X - means[k]).T))
				self.pi[k] = 1. * sum_to_k / self.N
			
			# Check convergence
			# ...

	# Get parameters for this estimator.
	def get_params():
		return pi, means, covariances

	# Predict the labels for the data samples in X using trained model.
	def predict(X):
		N = X.shape[0]
		D = X.shape[1]
		to_matrix = np.zeros((N, self.K))
		# Check dimension
		if (D != self.D):
			try:
				raise Exception('Dimension not valid')
			except Exception as exp:
				print exp
			return
		# Calculate to_matrix
		for n in range(N):
			for k in range(self.K):
				to_matrix[n, k] = self.pi[k] * multiGaussian(X[n], self.means[k], \
						self.covariances[k])
		to_matrix = 1. * to_matrix / np.sum(to_matrix, axis = 0)
		pred = np.argmax(to_matrix, axis = 1)

		return pred

	def predict_proba(X):	# Predict posterior probability of each component given the data.
		pass

	def sample(n_samples):	# Generate random samples from the fitted Gaussian distribution.
		pass

	def score(X):	# Compute the per-sample average log-likelihood of the given data X.
		pass

	def score_samples(X):	# Compute the weighted log probabilities for each sample.
		pass

	def set_params():	# Set the parameters of this estimator.
		pass

