import numpy as np
from scipy import stats


def VaR(X, alpha: float=0.05):
	X = np.sort(X)
	var = (X[int(alpha * X.size)] + X[int(alpha * X.size) + 1]) / 2 * (-1)
	return var


def ES(X, alpha: float=0.05):
	X = np.sort(X)
	es = X[:int(alpha * X.size)].mean() * (-1)
	return es

def norm(df):
	mean_n, var_n = stats.distributions.norm.fit(df)
	nsample = 10000
	mu = mean_n
	sigma = var_n
	X = np.random.normal(mu, sigma, nsample)
	return X

def t(df):
	nsample = 10000
	df, loc, scale = stats.t.fit(df.values)
	X = stats.t(df=df, loc=loc, scale=scale).rvs(nsample)
	return X

def VaR_Norm(df, alpha: float=0.05):
	X = norm(df)
	return VaR(X, alpha)

def ES_Norm(df, alpha: float=0.05):
	X = norm(df)
	return ES(X, alpha)

def VaR_T(df, alpha: float=0.05):
	X = t(df)
	return VaR(X, alpha)

def ES_T(df, alpha: float=0.05):
	X = t(df)
	return ES(X, alpha)

def VaR_History(df, alpha: float=0.05):
	X = np.sort(np.transpose(df.values)).ravel()# ravel: [[1,2,3]] -> [1,2,3]
	return VaR(X, alpha)

def ES_History(df, alpha: float=0.05):
	X = np.sort(np.transpose(df.values)).ravel()
	return ES(X, alpha)
