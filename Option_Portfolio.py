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
from mean_confidence_interval import CI
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
    def __init__(self, stocks, r, T, reps, steps, sensitivity=False, **delta):
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
        
        # Create parameters for sensitivity analysis portfolio
        self.sensitivity = sensitivity
        # If sensitivity analysis is specified, create alternate portfolio attributes ending in _delta
        # Generate alternative paths according to alternate expected return
        try:
            self.delta = delta['delta']
            self.stock_delta = delta['stock']
        except:
            self.delta = 0
            self.stock_delta = 0

        if self.sensitivity==True:
            self.put_payoff_delta = np.zeros(reps)
            self.call_payoff_delta = np.zeros(reps)
            self.cost_delta = 0
            self.products_delta = pd.DataFrame(columns=['stock','current price','product','type','strike','cost','count','total cost',])
            self.stocks[self.stock_delta]['paths_delta'] = gbm.BRW((1.+self.delta)*stocks[self.stock_delta]['mu'],stocks[self.stock_delta]['sigma'],stocks[self.stock_delta]['S0'],T,reps,steps)
            
    def add_stock(self,stock,num,sense='long'):
        '''
        Add specified stock to portfolio
        and update all portfolio parameters
        '''

        # Select stock parameters
        S0 = self.stocks[stock]['S0']
        
        # Update portfolio parameters
        
        if sense == 'long':
            self.cost += num*S0
            self.stocks[stock]['count'] = self.stocks[stock]['count'] + num
            self.products = self.products.append({'stock':stock,'current price':S0,'product':'stock','type':sense,'strike':'-','cost':S0,'count':num,'total cost':num*S0}, ignore_index=True)
        elif sense == 'short':
            self.cost -= num*S0
            self.stocks[stock]['count'] = self.stocks[stock]['count'] - num
            self.products = self.products.append({'stock':stock,'current price':S0,'product':'stock','type':sense,'strike':'-','cost':-S0,'count':num,'total cost':-num*S0}, ignore_index=True)
        
        # Generate sensitivity analysis portfolio
        
        if (self.sensitivity==True):
            if sense == 'long':
                self.cost_delta += num*S0
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':'stock','type':sense,'strike':'-','cost':S0,'count':num,'total cost':num*S0}, ignore_index=True)
            elif sense == 'short':
                self.cost_delta -= num*S0
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':'stock','type':sense,'strike':'-','cost':-S0,'count':num,'total cost':-num*S0}, ignore_index=True)
    
    def add_put(self,stock,num,K,sense='buy',op_type='european',**exercise):
        '''
        Add specified put option to portfolio
        and update all portfolio parameters
        '''

        # Select stock parameters
        S0 = self.stocks[stock]['S0']
        mu = self.stocks[stock]['mu']
        sigma = self.stocks[stock]['sigma']
        paths = self.stocks[stock]['paths']
        
        # Generate price of option
        if op_type == 'european':
            put = EuropeanOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type == 'asian':
            put = AsianOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type =='american':
            put = AmericanOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths,exercise=exercise)
        
        # Update portfolio parameters
        if sense=='buy':
            self.cost += num*put.value[0]
            self.put_payoff = self.put_payoff + np.multiply(num,put.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':put.value[0],'count':num,'total cost':num*put.value[0]}, ignore_index=True)
            
        elif sense=='sell':
            self.cost -= num*put.value[0]
            self.put_payoff = self.put_payoff - np.multiply(num,put.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':-put.value[0],'count':num,'total cost':-num*put.value[0]}, ignore_index=True)
        
        # Generate sensitivity analysis portfolio
        if (self.sensitivity==True) & (stock==self.stock_delta) :
            # if option is on sensitivity stock, new option prices must be generated
            paths = self.stocks[stock]['paths_delta']
            if op_type == 'european':
                put = EuropeanOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
            elif op_type == 'asian':
                put = AsianOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
            elif op_type =='american':
                put = AmericanOption(contract='put',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths,exercise=exercise)

            if sense=='buy':
                self.cost_delta += num*put.value[0]
                self.put_payoff_delta = self.put_payoff_delta + np.multiply(num,put.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':put.value[0],'count':num,'total cost':num*put.value[0]}, ignore_index=True)

            elif sense=='sell':
                self.cost_delta -= num*put.value[0]
                self.put_payoff_delta = self.put_payoff_delta - np.multiply(num,put.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':-put.value[0],'count':num,'total cost':-num*put.value[0]}, ignore_index=True)
        
        elif self.sensitivity==True:
            # if option is not on sensitivity stock, use existing option costs
            if sense=='buy':
                self.cost_delta += num*put.value[0]
                self.put_payoff_delta = self.put_payoff_delta + np.multiply(num,put.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':put.value[0],'count':num,'total cost':num*put.value[0]}, ignore_index=True)

            elif sense=='sell':
                self.cost_delta -= num*put.value[0]
                self.put_payoff_delta = self.put_payoff_delta - np.multiply(num,put.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' put option','type':sense,'strike':K,'cost':-put.value[0],'count':num,'total cost':-num*put.value[0]}, ignore_index=True)
            
    def add_call(self,stock,num,K,sense='buy',op_type='european',**exercise):
        '''
        Add specified call option to portfolio
        and update all portfolio parameters
        '''

        # Select stock parameters
        S0 = self.stocks[stock]['S0']
        mu = self.stocks[stock]['mu']
        sigma = self.stocks[stock]['sigma']
        paths = self.stocks[stock]['paths']
        
        # Generate price of option
        if op_type == 'european':
            call = EuropeanOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type == 'asian':
            call = AsianOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
        elif op_type =='american':
            call = AmericanOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths,exercise=exercise)
        
        # Update portfolio parameters
        if sense=='buy':
            self.cost += num*call.value[0]
            self.call_payoff = self.call_payoff + np.multiply(num,call.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':call.value[0],'count':num,'total cost':num*call.value[0]}, ignore_index=True)
        
        elif sense=='sell':
            self.cost -= num*call.value[0]
            self.call_payoff = self.call_payoff - np.multiply(num,call.values)
            self.products = self.products.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':-call.value[0],'count':num,'total cost':-num*call.value[0]}, ignore_index=True)
        
        # Generate sensitivity analysis portfolio
        if (self.sensitivity==True) & (stock==self.stock_delta) :
            # if option is on sensitivity stock, new option prices must be generated
            paths = self.stocks[stock]['paths_delta']
            
            if op_type == 'european':
                call = EuropeanOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
            elif op_type == 'asian':
                call = AsianOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths)
            elif op_type =='american':
                call = AmericanOption(contract='call',S0=S0,K=K,T=self.T,r=self.r,mu=mu,sigma=sigma,paths=paths,exercise=exercise)

            if sense=='buy':
                self.cost_delta += num*call.value[0]
                self.call_payoff_delta = self.call_payoff_delta + np.multiply(num,call.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':call.value[0],'count':num,'total cost':num*call.value[0]}, ignore_index=True)

            elif sense=='sell':
                self.cost_delta -= num*call.value[0]
                self.call_payoff_delta = self.call_payoff_delta - np.multiply(num,call.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':-call.value[0],'count':num,'total cost':-num*call.value[0]}, ignore_index=True)

        elif self.sensitivity==True:
            # if option is not on sensitivity stock, use existing option costs
            if sense=='buy':
                self.cost_delta += num*call.value[0]
                self.call_payoff_delta = self.call_payoff_delta + np.multiply(num,call.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':call.value[0],'count':num,'total cost':num*call.value[0]}, ignore_index=True)

            elif sense=='sell':
                self.cost_delta -= num*call.value[0]
                self.call_payoff_delta = self.call_payoff_delta - np.multiply(num,call.values)
                self.products_delta = self.products_delta.append({'stock':stock,'current price':S0,'product':op_type+' call option','type':sense,'strike':K,'cost':-call.value[0],'count':num,'total cost':-num*call.value[0]}, ignore_index=True)

    def net_value(self):
        '''
        Return net value of the portfolio
        '''
        self.stock_value = 0
        for i,j in self.stocks.items():
            self.stock_value += np.exp(-self.r*self.T)*self.stocks[i]['paths'][:,-1]*self.stocks[i]['count']
        self.put_value = self.put_payoff
        self.call_value = self.call_payoff
        self.net = self.stock_value + self.put_value + self.call_value - self.cost
        self.port_ave, self.port_ci = CI(self.net)

        return self.net, self.port_ave, self.port_ci
    
    def sensitivity_analysis(self):
        '''
        Return net value of the alternative portfolio (ie with perturbed stock)
        and the estimated derivative of effect expected net portfolio with respect to perturbed stock
        '''
        portfolio = np.sort(self.net_value()[0])
        # portfolio = self.net_value()[1]

        # Calculate sensitivity stock returns
        self.stock_value_delta = 0
        for i,j in self.stocks.items():
            if i == self.stock_delta:
                self.stock_value_delta += np.exp(-self.r*self.T)*self.stocks[i]['paths_delta'][:,-1]*self.stocks[i]['count']
            else:
                self.stock_value_delta += np.exp(-self.r*self.T)*self.stocks[i]['paths'][:,-1]*self.stocks[i]['count']
        self.put_value_delta = self.put_payoff_delta
        self.call_value_delta = self.call_payoff_delta
        self.net_delta = self.stock_value_delta + self.put_value_delta + self.call_value_delta - self.cost_delta

        self.port_ave_delta, self.port_ci_delta = CI(self.net_delta)
        
        # Calculate finite difference
        FD = (np.sort(self.net_delta) - portfolio)/self.delta
        # FD = (self.port_ave_delta - portfolio)/self.delta
        FD_ave,FD_CI = CI(FD)
        return  self.net_delta, self.port_ave_delta, self.port_ci_delta, FD, FD_ave, FD_CI
        
    def stock_plot(self,stock, cumulative=False):
        '''
        Plot portfolio payoff vs stock price at maturity
        Also plot vertical histogram of outcomes
        If cumulative=True cumulative return distribution is plotted as well
        '''
        x = self.stocks[stock]['paths'][:,-1]
        y = self.net_value()
        y = y[0]
        
        sns.set(rc={'figure.figsize':(14,8)})
        sns.set(font_scale=1.5)
        
        if cumulative:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, sharex=False)
            sns.regplot(x,y, fit_reg=False, ax=ax1)
            sns.distplot(y, vertical=True, bins=20, kde=False, norm_hist=False, ax=ax2)
            sns.kdeplot(y, shade=True, vertical=True, ax=ax3, cumulative=True, gridsize=100)
            ax1.xaxis.set_label_text('Stock Price at Maturity')
            ax1.yaxis.set_label_text('Portfolio Profit')
            ax1.set_title('Profit Vs. Stock Price')
            ax2.xaxis.set_label_text('Probability')
            ax2.set_title('Probability of Profit')
            ax3.xaxis.set_label_text('Cumulative Probability')
            ax3.set_title('Cumulative Probability of Profit')
            ax3.set_xlim(0,1)
        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, sharex=False)
            sns.regplot(x,y, fit_reg=False, ax=ax1)
            sns.distplot(y, vertical=True, bins=20, kde=False, norm_hist=False, ax=ax2)
            
            ax1.xaxis.set_label_text('Stock Price at Maturity')
            ax1.yaxis.set_label_text('Portfolio Profit')
            ax1.set_title('Profit Vs. Stock Price')
            ax2.xaxis.set_label_text('Probability')
            ax2.set_title('Probability of Profit')
    
    def hist_plot(self):
        '''
        Plot histogram of primary portfolio
        '''
        y = self.net_value()
        y = y[0]
        sns.set(rc={'figure.figsize':(14,8)})
        sns.set(font_scale=1.5)
        ax = sns.distplot(y, vertical=False, bins=20, kde=False, norm_hist=False)
        ax.xaxis.set_label_text('Net Portfolio Value at Maturity')
        ax.yaxis.set_label_text('Occurance Count')
        ax.set_title('Histogram of Portfolio Returns')
        
    def sensitivity_hist_plot(self):
        '''
        Plot histogram of perturbed portfolio
        '''
        y = self.sensitivity_analysis()[0]
        sns.set(rc={'figure.figsize':(14,8)})
        sns.set(font_scale=2)
        ax = sns.distplot(y, vertical=False, bins=20, kde=False, norm_hist=False)
        ax.xaxis.set_label_text('Net Portfolio Value at Maturity After Return Perterbation')
        ax.yaxis.set_label_text('Occurance Count')
        ax.set_title('Histogram of Altered Portfolio Returns')

    def stock_plot_reg(self,stock):
        '''
        Plot portfolio payoff vs stock price at maturity
        Also plot vertical histogram of outcomes
        If cumulative=True cumulative return distribution is plotted as well
        '''
        x = self.stocks[stock]['paths'][:,-1]
        y = self.net_value()
        y = y[0]
        
        sns.set(rc={'figure.figsize':(14,8)})
        sns.set(font_scale=1.5)

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, sharex=False)
        sns.regplot(x,y, order=4, ci=None, truncate=True , ax=ax1)
        sns.distplot(y, vertical=True, bins=20, kde=False, norm_hist=False, ax=ax2)
        
        ax1.xaxis.set_label_text('Stock Price at Maturity')
        ax1.yaxis.set_label_text('Portfolio Profit')
        ax1.set_title('Profit Vs. Stock Price')
        ax2.xaxis.set_label_text('Probability')
        ax2.set_title('Probability of Profit')