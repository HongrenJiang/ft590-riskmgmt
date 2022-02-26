import numpy as np


def Covar(m):
	"""
	:param m: A 1-D or 2-D array containing multiple variables and observations.
			Each row of m represents a variable,
			and each column a single observation of all those variables.
	"""
	return np.cov(m)


def ew(lambda_, nlags):
	"""
	
	:param lambda_: lambda
	:param nlags: time lags
	:return: exponential weights numpy series
	"""
	weights = np.array([(1 - lambda_) * (lambda_ ** (lag - 1)) for lag in range(1, nlags + 1)])
	weights /= weights.sum()  # normalized weights
	return weights


def ewCovar(m, lambda_):
	"""
	
	:param m: A 1-D or 2-D array containing multiple variables and observations.
			Each row of m represents a variable,
			and each column a single observation of all those variables.
			The smaller column is, the newer the observation is.
	:param lambda_: lambda
	:return: exponential weighted covariance matrix
	"""
	[r, l] = m.shape
	
	# Remove the mean from the series
	m = m - m.mean(axis=1, keepdims=True)
	weights = ew(lambda_, l)
	return np.cov(m, aweights=weights)


def getCor(cov):
	cov_diag = np.diag(cov)
	invSD = np.diag(np.divide(1, np.sqrt(cov_diag)))
	cor = invSD * cov * invSD
	return cor


def getCov(var, cor):
	std = np.sqrt(var)
	n = len(var)
	cov = np.matrix(np.zeros((n, n)))
	for i in range(n):
		for j in range(n):
			cov[i, j] = cor[i, j] * std[i] * std[j]
	
	return cov