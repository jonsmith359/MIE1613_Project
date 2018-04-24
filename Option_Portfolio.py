# Standard Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Custom Libraries
import geometric_brownian_motion as gbm
from European_Option import EuropeanOption
from Asian_Option import AsianOption
from American_Option import AmericanOption

class OptionPortfolio(object):
    '''
    Class to track value of stock/option portfolio on single underlying asset
    stocks - dictionary containing stock information in form:
        stocks ={
            'A':{'S0':100., 'mu':0.05, 'sigma':0.1},
            'B':{'S0':50., 'mu':0.1, 'sigma':0.2}
            }
        where
        s0 - starting stock price
        mu - expected return
        sigma - expected volatility
    r - risk-free rate
    T - option maturity
    reps - number of simulations
    steps - number of steps in discretization
    --------------------------------------------------------
    self.put_payoff - payoff of all put options in each scenario
    self.call_payoff - payoff of all call options in each scenario
    self.cost - cost of all products in the portfolio
    self.products - Dataframe storing all products in portfolio
    '''
    def __init__(self, stocks, r, T, reps, steps):
        self.stocks = stocks
        self.r = r
        self.T = T
        self.put_payoff = np.zeros(reps)
        self.call_payoff = np.zeros(reps)
        self.cost = 0
        self.products = pd.DataFrame(columns=['stock','current price','product','type','strike','cost','count','total cost',])
        # Generate sample paths for each stock
        for i,j in stocks.items():
            self.stocks[i]['paths'] = gbm.BRW(stocks[i]['mu'],stocks[i]['sigma'],stocks[i]['S0'],T,reps,steps)
            self.stocks[i]['count'] = 0
    def add_stock(self,stock,num,sense='long'):
        # Select stock parameters
        S0 = self.stocks[stock]['S0']
        
        if sense == 'long':
            self.cost += num*S0
            self.stocks[stock]['count'] = self.stocks[stock]['count'] + num
            self.products = self.products.append({'stock':stock,'current price':S0,'product':'stock','type':sense,'strike':'-','cost':S0,'count':num,'total cost':num*S0}, ignore_index=True)
        elif sense == 'short':
            self.cost -= num*S0
            self.stocks[stock]['count'] = self.stocks[stock]['count'] - num
            self.products = self.products.append({'stock':stock,'current price':S0,'product':'stock','type':sense,'strike':'-','cost':-S0,'count':num,'total cost':-num*S0}, ignore_index=True)
        
    def add_put(self,stock,num,K,sense='buy',op_type='european',**exercise):
        # Select stock parameters
        S0 = self.stocks[stock]['S0']
        mu = self.stocks[stock]['mu']
        sigma = self.stocks[stock]['sigma']
        paths = self.stocks[stock]['paths']
        
        if op_type == 'european':
            put = EuropeanOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type == 'asian':
            put = AsianOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type =='american':
            put = AmericanOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths,exercise=exercise)
        
        if sense=='buy':
            self.cost += num*put.value[0]
            self.put_payoff = self.put_payoff + np.multiply(num,put.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':put.value[0],'count':num,'total cost':num*put.value[0]}, ignore_index=True)
            
        elif sense=='sell':
            self.cost -= num*put.value[0]
            self.put_payoff = self.put_payoff - np.multiply(num,put.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':-put.value[0],'count':num,'total cost':-num*put.value[0]}, ignore_index=True)
    
    def add_call(self,stock,num,K,sense='buy',op_type='european',**exercise):
        # Select stock parameters
        S0 = self.stocks[stock]['S0']
        mu = self.stocks[stock]['mu']
        sigma = self.stocks[stock]['sigma']
        paths = self.stocks[stock]['paths']
        
        if op_type == 'european':
            call = EuropeanOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type == 'asian':
            call = AsianOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type =='american':
            call = AmericanOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths,exercise=exercise)
        
        if sense=='buy':
            self.cost += num*call.value[0]
            self.call_payoff = self.call_payoff + np.multiply(num,call.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':call.value[0],'count':num,'total cost':num*call.value[0]}, ignore_index=True)
        
        elif sense=='sell':
            self.cost -= num*call.value[0]
            self.call_payoff = self.call_payoff - np.multiply(num,call.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':-call.value[0],'count':num,'total cost':-num*call.value[0]}, ignore_index=True)
    
    def net_value(self):
        self.stock_value = 0
        for i,j in self.stocks.items():
            self.stock_value += np.exp(-self.r*self.T)*self.stocks[i]['paths'][:,-1]*self.stocks[i]['count']
        self.put_value = self.put_payoff
        self.call_value = self.call_payoff
        self.net = self.stock_value + self.put_value + self.call_value - self.cost
        return self.net
    
    def stock_plot(self,stock, cumulative=False):
        x = self.stocks[stock]['paths'][:,-1]
        y = self.net_value()
        
        sns.set(rc={'figure.figsize':(14,8)})
        
        if cumulative:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, sharex=False)
            sns.regplot(x,y, fit_reg=False, ax=ax1)
            sns.kdeplot(y, shade=True, vertical=True, ax=ax2, kernel= 'epa', gridsize=100)
            sns.kdeplot(y, shade=True, vertical=True, ax=ax3, cumulative=True, gridsize=100)
            ax1.xaxis.set_label_text('Stock Price at Maturity')
            ax1.yaxis.set_label_text('Portfolio Profit')
            ax1.set_title('Profit Vs. Stock Price')
            ax2.xaxis.set_label_text('Probability')
            ax2.set_title('Probability of Profit')
            ax2.set_xlim(0,0.1)
            ax3.xaxis.set_label_text('Cumulative Probability')
            ax3.set_title('Cumulative Probability of Profit')
            ax3.set_xlim(0,1)
        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, sharex=False)
            sns.regplot(x,y, fit_reg=False, ax=ax1)
            sns.kdeplot(y, shade=True, vertical=True, ax=ax2, kernel= 'epa', gridsize=100)
            ax1.xaxis.set_label_text('Stock Price at Maturity')
            ax1.yaxis.set_label_text('Portfolio Profit')
            ax1.set_title('Profit Vs. Stock Price')
            ax2.xaxis.set_label_text('Probability')
            ax2.set_title('Probability of Profit')
            ax2.set_xlim(0,0.1)