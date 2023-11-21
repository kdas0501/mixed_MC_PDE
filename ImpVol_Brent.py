# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Description: Computes implied volatility for a given European Put or Call option price using 
Brent's method.

Methods:

- BS_Formula(S0, sig, Strk, r, T, option)
- ImpVol_Brent(S0, Strk, r, T, option, price)

"""


from scipy import optimize
import copy as cp
from math import sqrt, exp, log
from collections import deque
from scipy.stats import norm 

def BS_Formula(S0, sig, Strk, r, T, option):

    """ 
    Computes the Black Scholes Put/Call option formula.

    S0 (float): initial spot.
    sig (float): initial volatility.
    Strk (float): Strike value of the contract.
    r (float): interest rate.
    T (float): Option maturity.
    option (str): 'Put' or 'Call'.
    """
            
    sqrtT = sqrt(T)
    sigsqrtT = sig*sqrtT
    
    lograt = log(S0/Strk)
    dpl = (lograt + r*T)/sigsqrtT + 0.5*sigsqrtT
    dm = dpl - sigsqrtT
    expmr = exp(-1.0*r*T)
    
    if option == 'Put':
        H  = Strk*expmr*norm.cdf(-1.0*dm) - S0*norm.cdf(-1.0*dpl)
        
    elif option == 'Call':
        H  = S0*norm.cdf(dpl) - Strk*expmr*norm.cdf(dm)

    
    return H

# Example usage:

if __name__ == '__main__':

    S0 = 100
    sig = 0.20
    Strk = S0*1.01
    r = 0.02
    T = 6/12.0
    
    option = 'Put'
    
        
    print(BS_Formula(S0, sig, Strk, r, T, option))







def ImpVol_Brent(S0, Strk, r, T, option, price):

    """
    Computes implied volatility for a given European Put or Call option price using 
    Brent's method.

    S0 (float): initial spot.
    Strk (float): strike value of the contract.
    r (float): interest rate.
    T (float): Option maturity.
    option (str): 'Put' or 'Call'.
    price (float): Price of the Put or Call option.
    
    """
    
    def BS_Formula_sig(sig):
        
        return BS_Formula(S0, sig, Strk, r, T, option) - price

    root = optimize.brentq(BS_Formula_sig, -1, 1)
    
    return root



# Example usage:

if __name__ == '__main__':
    
    S0 = 100
    Strk = S0*1.01
    r = 0.02
    T = 6/12.0
    price = 4.3416939168077135
    
    option = 'Put'
    
    
    print(ImpVol_Brent(S0, Strk, r, T, option, price))