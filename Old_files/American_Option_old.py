# American Option Pricing Function

import pandas as pd

import numpy as np
from scipy.stats import norm
import mean_confidence_interval as conf
import geometric_brownian_motion as gbm

import matplotlib.pyplot as plt

class AmericanOption(object):
    '''
    Class for American Option valuation
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
        self.r = float(r)/steps
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.steps = steps
        self.reps = reps
        self.interval = float(T/steps)
        if (contract != 'call') & (contract != 'put'):
            raise ValueError('Invalid Contract Type. Specify <call> or <put>')
        self.value = self.Sim_value()

    def Sim_value(self):
        '''
        Return European option value using Brownian Random Walk Monte-Carlo simulation
        '''
        prices = pd.DataFrame(gbm.BRW(drift=self.mu,sigma=self.sigma,S0=self.S0,T=self.T,paths=self.reps,steps=self.steps))
        if self.contract =='call':
            payout = prices - self.K
        elif self.contract =='put':
            payout = self.K - prices
        
        # Set negative payoffs to 0, reverse order of dataframes along time axis
        payout[payout < 0] = 0
        paths_rev = prices.iloc[:, ::-1]
        payout_rev = payout.iloc[:, ::-1]
        for i in range(payout_rev.shape[1]-1):
            payout_1 = payout_rev.iloc[:,i]
            payout_2 = payout_rev.iloc[:,i+1]
            
            # x - prices of stocks at timestep t, if non-zero payout at time t-1
            x = paths_rev.iloc[:,i+1].iloc[payout_2.nonzero()]

            # y - holding value from time t-1 to t
            HV = np.exp(-self.r)*payout_1.iloc[payout_2.nonzero()]

            # Fit quadratic regression
            try:
                c,b,a = np.polyfit(x,HV,2)
            # polyfit will fail in the case of no non-zero payouts:
            except:
                c,b,a = 0.0,0.0,0.0

            # Find expected holding value based on regression
            E_HV = a + b * x + c * np.square(x)

            # Find Exercise value at time t-1
            EV = payout_2.iloc[payout_2.nonzero()]

            # indexes of EV>E_HV
            for pos,ev in EV.iteritems():
                # if EV>E_HV, payout at t-1 is corresponding EV, and payout at t = 0
                if ev>E_HV[pos]:
                    payout_1[pos]=0
                    payout_2[pos]=EV[pos]
                # if EV<E_HV, payout (value) at t-1 is corresponding HV
                else:
                    payout_2[pos]=HV[pos]

            # Find cases where holding is optimal, and overwrite t-1 payout with discounted t HV
            for pos,p1 in payout_1.iteritems():
                if p1>payout_2[pos]:
                    payout_2[pos] = np.exp(-self.r)*payout_1.iloc[pos]
        
        values = payout_rev.iloc[:,-1]
        value, CI_95 = conf.CI(values)
            
        return value, CI_95