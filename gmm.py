import numpy as np 
import math
from gaussian import multi_gaussian, gaussian, multi_gaussian_matrix
from sklearn.cluster import KMeans

class GaussianMixture:
	def __init__(self, K, tol = 1e-6):
		self._K = K
		self._tol = tol

	# Estimate model parameters with the EM algorithm.
	def fit(self, X):
		# Data
		self._Xtr = np.array(X)
		# Data dimension
		self._D = self._Xtr.shape[1]
		self._N = self._Xtr.shape[0]
		# Init parameters
		self._init_params()
		# EM algorithm
		self._em()

	# Initialize parameters
	def _init_params(self):
		# Weight of mixture
		pi = np.random.rand(self._K)
		pi /= np.sum(pi)
		# Means: init by kmeans
		means = []
		kmeans = KMeans(n_clusters = self._K, random_state = 0)\
				.fit(self._Xtr)
		means = kmeans._cluster_centers
		# Covariances from means
		covariances = self._covariances(means, kmeans.labels_, self._Xtr)

		self._pi, self._means, self._covariances = pi, means, covariances

	# Covariance from data and means
	def _covariances(means, labels, data):
		covariances = []
		for k in range(self._K):
			mean = means[k]
			X_k = data[labels == k]
			N = X_k.shape[0]
			Xbar = X_k - means
			cov = Xbar.dot(Xbar.T)
			covariances.append(cov)

		return np.array(covariances)

	# EM algorithm, with paramaters initialized
	def _em(self):
		old_log_likelihood = self._log_likelihood()
		for i in xrange(2000):
			print 'Iterator number %d' % (i + 1)
			
			# E - step: estimate posterior
			gaussian_mat = []
			for k in range(self._K):
				prob_vec = multi_gaussian_matrix(self._Xtr,\
						self._means[k], self._covariances[k])
				gaussian_mat.append(prob_vec)
			gaussian_mat = np.array(gaussian_mat).T # NxK
			pi_diag = np.diag(self._pi) # KxK
			self._unnormal_posterior_matrix = gaussian_mat.dot(pi_diag) # NxK
			# Normalize
			self._posterior_matrix = 1. * self._unnormal_posterior_matrix /\
					np.sum(self._unnormal_posterior_matrix, axis = 1).reshape(self._N, 1)
			
			# M - step: optimize parameters
			trans_posterior = self._posterior_matrix.T # KxN
			k_sums = np.sum(trans_posterior, axis = 1).reshape(self._K, 1) # Kx1
 			self._means = trans_posterior.dot(self._Xtr) / k_sums
 			self._pi = k_sums / self._N
 			# Covariances
 			covariances = []
 			for k in range(self._K):
 				Xbar_k = self._Xtr - self._means[k] # NxD
 				post_k = self._posterior_matrix[:, k] # N
 				cov = (post_k * Xbar_k.T).dot(Xbar_k) * 1. / k_sums[k]
				covariances.append(cov)
			self._covariances = np.array(covariances)	

			# Check convergence
			new_log_likehood = self._log_likelihood()
			if math.fabs(new_log_likehood - old_log_likelihood) < self._tol:
				break
			old_log_likelihood = new_log_likehood

	# Calculate log likelihood
	def _log_likelihood(self):
		# self._unnormal_posterior_matrix: NxK
		sum_k = np.sum(self._unnormal_posterior_matrix, axis = 1)
		log_sum_k = np.log(sum_k)
		log_likelihood = np.sum(log_sum_k)

		return log_likelihood

	# Get parameters for this estimator.
	def get_params(self):
		return self._pi, self._means, self._covariances

	# Predict the labels for the data samples in X using trained model.
	def predict(self, X):
		X = np.array(X)
		N = X.shape[0]
		D = X.shape[1]
		# Check dimension
		if (D != self._D):
			try:
				raise Exception('Dimension not valid')
			except Exception as exp:
				print exp
			return
		# Predict
		gaussian_mat = []
		for k in range(self._K):
			prob_vec = multi_gaussian_matrix(self._Xtr,\
					self._means[k], self._covariances[k])
			gaussian_mat.append(prob_vec)
		gaussian_mat = np.array(gaussian_mat).T # NxK
		pi_diag = np.diag(self._pi) # KxK
		mat = gaussian_mat.dot(pi_diag) # NxK
		pred = np.argmax(mat, axis = 1)

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

