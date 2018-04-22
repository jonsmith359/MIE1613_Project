#Confidence Interval Function
import numpy as np
import scipy.stats as stats
from math import sqrt
def CI(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    mu,se = np.mean(a),stats.sem(a)
    # z = stats.t.ppf(confidence, n)
    z1,z2 = stats.norm.interval(confidence, loc=0, scale=1)
    h=z2*se
    return mu, h
