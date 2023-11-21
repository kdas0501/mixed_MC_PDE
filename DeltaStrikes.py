# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Description: Computes the corresponding Strike value for a Put/Call option
for a given Delta value in the Black-Scholes model.

"""

from scipy.stats import norm
import numpy as np

def DeltaStrikes(sig, S0, r, T, y, option):

    """
    Computes the corresponding Strike value for a Put/Call option
    for a given Delta value in the Black-Scholes model.

        sig (float): Volatility.
        S0 (float): Initial spot.
        r (float): Interest rate.
        T (float): Maturity of contract.
        y (float): Delta of contract (0.5 is at the money).
        option (str): 'Put' or 'Call'.
    """

    if option == 'Put':
        Strk = S0*np.exp(sig*np.sqrt(T)*norm.ppf(y)+r*T+.5*sig*sig*T)

    elif option == 'Call':
        Strk = S0*np.exp(-sig*np.sqrt(T)*norm.ppf(y)+(r*T+.5*sig*sig*T))
        
    return Strk
    


# Example usage:

if __name__ == "__main__":
    
    print(DeltaStrikes(0.2, 100, 0.01, 6/12.0, .9, 'Put'))

