import numpy as np
from . import non_psd_fix

def simulateNormal(N, cov, mean=np.empty(shape=(0, 0)), seed=1234):
	# Error Checking
	[n, m] = cov.shape
	if n != m:
		raise Exception("Covariance Matrix is not square (" + str(n) + "," + str(m) + ")")
	
	# If the mean is missing then set to 0, otherwise use provided mean
	_mean = np.full(n, 0.0)
	m = mean.shape[0]
	if m != 0:
		if n != m:
			raise Exception("Mean " + str(m) + " is not the size of cov (" + str(n) + "," + str(n) + ")")
		_mean = mean.copy()
	
	# Take the root
	l = non_psd_fix.chol_psd(cov)
	
	# Generate needed random standard normals
	out = np.matrix(np.random.normal(0, 1, (n, N)))
	
	# apply the standard normals to the cholesky root
	out = (l * out)
	
	# Add the mean
	for i in range(n):
		out[:, i] = out[:, i] + _mean[i]
	
	return out


def simulate_pca(a, nsim, target, mean=np.empty(shape=(0, 0)), seed=1234):
	n = a.shape[0]
	
	# If the mean is missing then set to 0, otherwise use provided mean
	_mean = np.full(n, 0.0)
	m = mean.shape[0]
	if m != 0:
		if n != m:
			raise Exception("Mean " + str(m) + " is not the size of cov (" + str(n) + "," + str(n) + ")")
		_mean = mean.copy()
	
	# Eigenvalue decomposition
	vals, vecs = np.linalg.eigh(a)
	vals = vals.real
	vecs = vecs.real
	# python returns values lowest to highest, flip them and the vectors
	vals = vals[::-1]
	vecs = vecs[::-1]
	
	tv = vals.sum()
	
	vals = np.maximum(vals, 0)
	
	cumm_val_explained = np.cumsum(vals) / tv
	i = 0
	for i in range(len(vals)):
		if cumm_val_explained[i] < target:
			i += 1
		else:
			break
	
	vals = vals[0:i + 1]
	vecs = vecs[:, :i + 1]
	
	B = vecs @ np.diag(np.sqrt(vals))
	np.random.seed(1234)
	z = np.random.normal(size=(len(vals), nsim))
	return B @ z


