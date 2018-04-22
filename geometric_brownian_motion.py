# Geometric Brownian Motion Function

import numpy as np
from scipy.stats import norm

def BRW(drift, sigma, S0, T, paths, steps):
	'''
	Function to generate geometric random brownian motion
	drift - expected return
	sigma - volatility
	S0 - starting price
	T - maturity
	paths - number of replications
	steps - number of discretizations
	'''
	# T = float(T)
	# steps = steps
	interval = float(T)/float(steps)

	RW = np.zeros((paths,steps+1))

	for i in range (paths):
		RW[i,0] = S0
		for j in range (steps):
			Z = norm.rvs()
			RW[i,j+1] = RW[i,j]*np.exp((drift - sigma**2/2) * interval + sigma * np.sqrt(interval) * Z)
	return RW