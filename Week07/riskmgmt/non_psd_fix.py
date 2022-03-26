import sys

import numpy as np


def chol_psd(a):
	[row, col] = a.shape
	root = np.full([row, col], 0.0, dtype='float64')
	n = row
	
	for j in range(n):
		s = 0.0
		
		if j > 0:
			s = np.matmul(root[j, :j].T, root[j, :j])
		
		temp = a[j, j] - s
		if -1e-8 <= temp <= 0:
			temp = 0.0
		
		root[j, j] = np.sqrt(temp)
		
		if root[j, j] == 0:
			root[j, j + 1:n] = 0.0
		else:
			ir = 1.0 / root[j, j]
			for i in range(j + 1, n):
				s = np.matmul(root[j, :j].T, root[i, :j])
				root[i, j] = (a[i, j] - s) * ir
	
	return np.matrix(root)


def near_psd(a):
	epsilon = 0.0
	n = a.shape[0]
	
	inv_sd = None
	out = a.copy()
	
	if np.count_nonzero(np.diag(a) == 1) != n:
		inv_sd = np.diag(np.divide(1, np.sqrt(np.diag(a))))
		out = np.matmul(np.matmul(inv_sd, out), inv_sd)
	
	vals, vecs = np.linalg.eigh(out)
	vals = np.matrix(np.maximum(vals, epsilon))
	vecs = np.matrix(vecs)
	
	t = 1. / np.matmul(np.multiply(vecs, vecs), vals.T)
	t = np.diag(np.sqrt(np.array(t).reshape(n)))
	l = np.diag(np.array(np.sqrt(vals)).reshape(n))
	B = np.matmul(np.matmul(t, vecs), l)
	out = np.matmul(B, B.T)
	
	if inv_sd != None:
		inv_sd = np.diag(np.divide(1, np.diag(inv_sd)))
		out = np.matmul(np.matnul(inv_sd, out), inv_sd)
	
	return out


def getAplus(A):
	vals, vecs = np.linalg.eigh(A)
	vals = np.matrix(np.diag(np.maximum(vals, 0)))
	vecs = np.matrix(vecs)
	return vecs * vals * vecs.T


def getPs(A, W):
	W05 = np.sqrt(W)
	iW = W05.I
	return iW * getAplus(W05 * A * W05) * iW


def getPu(A, W):
	Aret = A.copy()
	for i in range(0, Aret.shape[0]):
		Aret[i, i] = 1.0
	return Aret


def wgtNorm(A, W):
	W05 = np.sqrt(W)
	W05 = W05 * A * W05
	return np.multiply(W05, W05).sum()


def higham_nearestPSD(pc, W=None, epsilion=1e-9, maxIter=100, tol=1e-9):
	n = pc.shape[0]
	if W == None:
		W = np.matrix(np.diag(np.full(n, 1.0)))
	
	deltaS = 0
	
	Yk = pc.copy()
	norml = sys.maxsize
	i = 1
	
	while i <= maxIter:
		Rk = Yk - deltaS
		Xk = getPs(Rk, W)
		deltaS = Xk - Rk
		Yk = getPu(Xk, W)
		norm = wgtNorm(Yk - pc, W)
		
		temvals, temvecs = np.linalg.eigh(Yk)
		minEigVal = np.min(temvals)
		
		if ((norm - norml) < tol) and (minEigVal > -epsilion):
			break
		
		norml = norm
		i += 1
	
	if i < maxIter:
		print("Converged in " + str(i) + " iterations.")
	else:
		print("Convergence failed after " + str(i - 1) + " iterations")
	
	return Yk


def Norms(cov, cov_psd):
	res = ((cov - cov_psd) ** 2).sum()
	
	return res
