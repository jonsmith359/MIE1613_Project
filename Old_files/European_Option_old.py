# European Option Pricing Function

import numpy as np
from scipy.stats import norm
import mean_confidence_interval as conf
import geometric_brownian_motion as gbm

class EuropeanOption(object):
    '''
    Class for European Option valuation
    contract - option contract (put or call)
    S0 - initial stock value
    K - strike price
    T - time to maturity (years)
    r - annual risk free rate 
    mu - expected return
    sigma - volatility
    steps - number of steps in discretization
    reps - number of simulations 
    '''
    # Constructor
    def __init__(self,contract,S0,K,T,r,mu,sigma,steps,reps):
        self.contract = contract
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.steps = steps
        self.reps = reps
        self.interval = float(T/steps)
        if (contract != 'call') & (contract != 'put'):
            raise ValueError('Invalid Contract Type. Specify <call> or <put>')
        self.value = self.Sim_value()
    def BS_value(self):
        '''
        Return European option value using Black-Scholes equation
        '''
        d1 = (1/(self.sigma*self.T))*(np.log(self.S0/self.K)+(self.r+0.5*self.sigma**2)*self.T)
        d2 = d1 - self.sigma*self.T
        if self.contract =='call':
            value = norm.cdf(d1)*self.S0 - norm.cdf(d2)*self.K*np.exp(-self.r*self.T)
        elif self.contract =='put':
            value = norm.cdf(-d2)*self.K*np.exp(-self.r*self.T)-norm.cdf(-d1)*self.S0
        return value
    def Sim_value(self):
        '''
        Return European option value using Brownian Random Walk Monte-Carlo simulation
        '''
        prices = gbm.BRW(self.mu,self.sigma,self.S0,self.T,self.reps,self.steps)
        self.final_price = prices[:,-1]
        self.values=[]
        for val in self.final_price:
            if self.contract =='call':
                self.values.append(np.exp(-self.r*self.T)*np.maximum(0.0,val - self.K))
            elif self.contract =='put':
                self.values.append(np.exp(-self.r*self.T)*np.maximum(0.0,self.K - val))
        value, CI_95 = conf.CI(self.values)
#         error = np.abs(value-self.BS_value())/self.BS_value()
        return value, CI_95 #error