# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Description: Computes the Monte Carlo price of a Put/Call option in the model 
dS = rSdt + S(V or sqrt{V})dW
dV = alpha(t,V)dt + beta(t,V) dB

"""

import numpy as np
from timeit import default_timer as timer 

def Full_MC(model_params, global_params, N_PATH, N_TIME, model, option):

    """
    Computes the Monte Carlo price of a Call/Put option in the model 
    dS = rSdt + S(V or sqrt{V})dW
    dV = alpha(t,V)dt + beta(t,V) dB
        
    model_params (list): [kap, the, lam, rho] is a list of floats corresponding to model parameters (see model parameter).
    globals_params (list): [S0, V0, rd_deque, rf_deque, Strk, T].
        S0 (float): initial spot.
        V0 (float): initial variance/volatility.
        r (float): interest rate.
        Strk (float): Strike of the contract.
        T (float): Maturity of contract.
    N_PATH (int): # of Monte-Carlo paths.
    N_TIME (int): # of time discretisation points for SDE integration.
    model (str): specifies model, where
        t gives "test", volatility & dV = dB,
        o gives "Ornstein", volatility & dV = kap(the - V)dt + lam dB,
        g gives "GARCH", variance & dV = kap(the - V)dt + lam V dB,
        i gives "Inverse-Gamma", volatility & dV = kap(the - V)dt + lam V dB,
        h gives "Heston", variance & dV = kap(the - V)dt + lam sqrt{V} dB,
        v gives "Verhulst", volatility & dV = kap*V*(the - V)dt + lam V dB.
    option (str): 'Put' or 'Call'.
    """
    
    # np.random.seed(102)

    kap = model_params[0]
    the = model_params[1]
    lam = model_params[2]
    rho = model_params[3]

    S0 = global_params[0]
    V0 = global_params[1]
    r = global_params[2]
    Strk = global_params[3]
    T = global_params[4]
    
    start = timer()
    
    dt = T/N_TIME
    sqrtdt = np.sqrt(dt)
    sqrtrhobar = np.sqrt(1-rho**2)
    kapdt = kap*dt
    kapthedt = the*kapdt
    
    V = np.ones([N_PATH])*V0
    lnS = np.ones([N_PATH])*np.log(S0)
    
    # Test: Volatility and dV = dB         
    if model == "t":
        for tt in range(N_TIME):
            RND1 = np.random.normal(0,1,[N_PATH])
            RND2 = np.random.normal(0,1,[N_PATH])
            lnS += (r-0.5*V**2)*dt + V*sqrtdt*RND1
            V += sqrtdt*(rho*RND1 + sqrtrhobar*RND2)
    
    # Ornstein Uhlenbeck: Volatility & dV = kap(the - V)dt + lam*dB_t     
    elif model == "o":
        for tt in range(N_TIME):
            RND1 = np.random.normal(0,1,[N_PATH])
            RND2 = np.random.normal(0,1,[N_PATH])
            mV = V
            #mV = np.maximum(V,0)
            lnS += (r-0.5*V**2)*dt + V*sqrtdt*RND1
            #V += kapthedt-mV*kapdt + lam*sqrtdt*(rho*RND1 + sqrtrhobar*RND2)
            V += kapthedt-kapdt*V + lam*sqrtdt*(rho*RND1 + sqrtrhobar*RND2)
            
    # GARCH: Variance & dV = kap(the - V)dt + lam*V*dB_t
    elif model == "g":
        for tt in range(N_TIME):
            RND1 = np.random.normal(0,1,N_PATH)
            RND2 = np.random.normal(0,1,N_PATH)
            mV = np.maximum(V,0)
            sqrtmV = np.sqrt(mV)   
            lnS += (r-0.5*mV)*dt+sqrtmV*sqrtdt*RND1
            V += kapthedt-mV*kapdt + lam*mV*sqrtdt*(rho*RND1 + sqrtrhobar*RND2)
            
    # Inverse Gamma: Volatility & dV = kap(the - V)dt + lam*V*dB_t
    elif model == "i":
        for tt in range(N_TIME):
            RND1 = np.random.normal(0,1,[N_PATH])
            RND2 = np.random.normal(0,1,[N_PATH])
            lnS += (r-0.5*V**2)*dt + V*sqrtdt*RND1
            V += kapthedt-V*kapdt + lam*V*sqrtdt*(rho*RND1 + sqrtrhobar*RND2)

    # Heston: Variance & dV = kap(the - V)dt + lam*sqrt{V}*dB_t 
    elif model == "h":       
        if 2*kap*the < lam**2:
            print("Feller test fail!")
            
        for tt in range(N_TIME):
            RND1 = np.random.normal(0,1,N_PATH)
            RND2 = np.random.normal(0,1,N_PATH)
            mV = np.maximum(V,0)
            sqrtmV = np.sqrt(mV)  
            lnS += (r-0.5*mV)*dt+sqrtmV*sqrtdt*RND1
            V += kapthedt-mV*kapdt + lam*sqrtmV*sqrtdt*(rho*RND1 + sqrtrhobar*RND2) 
            
    # Verhulst: Variance & dV = kap*V*(the - V)dt + lam*V*dB_t 
    elif model == "v":       
        for tt in range(N_TIME):
            RND1 = np.random.normal(0,1,N_PATH)
            RND2 = np.random.normal(0,1,N_PATH)
            lnS += (r-0.5*V**2)*dt + V*sqrtdt*RND1
            V += kapthedt*V-kapdt*V**2 + lam*V*sqrtdt*(rho*RND1 + sqrtrhobar*RND2) 
    
    # Payoff
    S = np.exp(lnS)
    if option == "Call":
        D = np.maximum(S - Strk,0)
    elif option == "Put":
        D = np.maximum(Strk - S,0)
    
    # Average over paths to compute price
    H = np.exp(-r*T)*np.mean(D)
    end = timer()
    
    # Compute standard error
    Sigma = np.exp(-r*T)*np.std(D)
    SE = Sigma/np.sqrt(N_PATH)
        
    elapsed = end - start
    
    return(H, SE, elapsed)


# Example usage:

if __name__ == '__main__':

    # Common parameters
    S0 = 100
    V0 = 0.20       
    # V0 = 0.20**2       # V0 is initial variance for Heston and GARCH
    r = 0.01
    T = 6/12.0
    Strk = S0*1.15
    kap = 5
    the = 0.18
    lam = 0.9
    rho = -0.35

    model = "i"
    option = "Put"

    # Monte Carlo parameters
    N_PATH = 300000
    Steps_day_N = 24
    N_TIME = int(Steps_day_N*253*T) 

    global_params = [S0, V0, r, Strk, T]
    model_params = [kap, the, lam, rho]

    Monte = Full_MC(model_params, global_params, N_PATH, N_TIME, model, option)

    print(f"\nPrice of European {option} option in model {model} is {Monte[0]:.3f}")

    print(f"\nThe Monte Carlo Standard Error is {100*Monte[1]:.3f}%")

    print(f"\nFull Monte Carlo simulation time in seconds is {Monte[2]:.3f}s\n")