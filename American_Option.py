# American Option Pricing Function

import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import mean_confidence_interval as conf
import geometric_brownian_motion as gbm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

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
    def __init__(self,contract,S0,K,T,r,mu,sigma,paths,**exercise):
        self.contract = contract
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.steps = paths.shape[1]-1
        self.r = float(r)/self.steps
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.paths = paths
        self.exercise = float(exercise['exercise']['exercise'])
        try:
            self.ex_step = int(math.floor(self.steps * self.exercise/self.T))
        except:
            self.ex_step = int(self.steps)
        if (contract != 'call') & (contract != 'put'):
            raise ValueError('Invalid Contract Type. Specify <call> or <put>')
        self.value = self.Sim_value()

    def Sim_value(self):
        '''
        Return European option value using Brownian Random Walk Monte-Carlo simulation
        '''
        self.paths = pd.DataFrame(self.paths)
        if self.contract =='call':
            payout = self.paths - self.K
        elif self.contract =='put':
            payout = self.K - self.paths

        # store values at exercise
        self.values = np.exp(-self.r * self.ex_step) * np.maximum(0,payout.iloc[:,self.ex_step])
        
        # Set negative payoffs to 0, reverse order of dataframes along time axis
        payout[payout < 0] = 0
        paths_rev = self.paths.iloc[:, ::-1]
        payout_rev = payout.iloc[:, ::-1]
        for i in range(self.steps):
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
        
        end_values = payout_rev.iloc[:,-1]
        value, CI_95 = conf.CI(end_values)
                
        # print (type(self.values))
        # print (self.values)

        return value, CI_95