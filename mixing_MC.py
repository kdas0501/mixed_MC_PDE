# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Description: Computes the Monte Carlo price of a Put/Call option using the mixing solution in the model 
dS = rSdt + S(V or sqrt{V})dW
dV = alpha(t,V)dt + beta(t,V) dB

"""

import numpy as np
from scipy.stats import norm
from timeit import default_timer as timer 

def mixing_MC(model_params, global_params, N_PATH, N_TIME, model, option):

    """
    Computes the Monte Carlo price of a Put/Call option using the mixing solution in the model 
    dS = rSdt + S(V or sqrt{V})dW
    dV = alpha(t,V)dt + beta(t,V) dB
        
    model_params (list): [kap, the, lam, rho] is a list of floats corresponding to model parameters (see model parameter)
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

    # np.random.seed(101)
    
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
    rhosq = rho**2
    rhobar = 1 - rhosq
    #sqrtrhobar = np.sqrt(1-rhosq)
    kapdt = kap*dt
    kapthedt = the*kapdt
    
    V = np.ones([N_PATH])*V0
    lnS = np.ones([N_PATH])*np.log(S0)
    
    It = np.zeros([N_PATH])
    Re = np.zeros([N_PATH])
    

# Test: Volatility and dV = dB         
    if model == "t":
        for tt in range(N_TIME):
            RND = np.random.normal(0,1,[N_PATH])
            Re += V**2*dt
            It += V*sqrtdt*RND
            V += sqrtdt*RND
    
    # Ornstein Uhlenbeck: Volatility & dV = kap(the - V)dt + lam*dB_t     
    elif model == "o":
        for tt in range(N_TIME+1):
            RND = np.random.normal(0,1,[N_PATH])
            mV = V
            #mV = np.maximum(V,0)
            #V += kapthedt-mV*kapdt + lam*sqrtdt*(rho*RND1 + sqrtrhobar*RND2)
            Re += V**2*dt
            It += V*sqrtdt*RND
            V += kapthedt-kapdt*V + lam*sqrtdt*RND
            
    # GARCH: Variance & dV = kap(the - V)dt + lam*V*dB_t
    elif model == "g":
        for tt in range(N_TIME):
            RND = np.random.normal(0,1,N_PATH)
            mV = np.maximum(V,0)
            sqrtmV = np.sqrt(mV)   
            Re += mV*dt
            It += sqrtmV*sqrtdt*RND
            V += kapthedt-mV*kapdt + lam*mV*sqrtdt*RND

    # Inverse Gamma: Volatility & dV = kap(the - V)dt + lam*V*dB_t
    elif model == "i":
        for tt in range(N_TIME):
            RND = np.random.normal(0,1,[N_PATH])
            Re += V**2*dt
            It += V*sqrtdt*RND
            V += kapthedt-V*kapdt + lam*V*sqrtdt*RND

    # Heston: Variance & dV = kap(the - V)dt + lam*sqrt{V}*dB_t 
    elif model == "h":       
        if 2*kap*the < lam**2:
            print("\nFeller test fail!")
            
        for tt in range(N_TIME):
            RND = np.random.normal(0,1,N_PATH)
            mV = np.maximum(V,0)
            sqrtmV = np.sqrt(mV)  
            Re += V*dt
            It += sqrtmV*sqrtdt*RND
            V += kapthedt-mV*kapdt + lam*sqrtmV*sqrtdt*RND
            
    # Verhulst: Variance & dV = kap*V*(the - V)dt + lam*V*dB_t 
    elif model == "v":       
        for tt in range(N_TIME):
            RND = np.random.normal(0,1,N_PATH)
            Vsqr = V**2
            Re += Vsqr*dt
            It += V*sqrtdt*RND
            V += kapthedt*V-kapdt*Vsqr+ lam*V*sqrtdt*RND
            

            

    xarg = lnS + rho*It - .5*rhosq*Re
    yarg = rhobar*Re
    sqrtyarg = np.sqrt(np.abs(yarg))
    dpl = (xarg - np.log(Strk) + r*T)/sqrtyarg + .5*sqrtyarg
    dm = dpl - sqrtyarg

    
    if option == "Put":
        
        PBS = Strk*np.exp(-r*T)*norm.cdf(-1.0*dm) - np.exp(xarg)*norm.cdf(-1.0*dpl)
        H = np.mean(PBS)
        end = timer()
        
        Sigma = np.std(PBS)
        
    elif option == "Call":
        
        CBS = np.exp(xarg)*norm.cdf(dpl) - Strk*np.exp(-r*T)*norm.cdf(dm)
        H = np.mean(CBS)
        end = timer()
        
        Sigma = np.std(CBS)
        
    SE = Sigma/np.sqrt(N_PATH)

    elapsed = end - start
    
    return(H, SE, elapsed)



# Example usage:

if __name__ == '__main__':


    # Common parameters
    S0 = 100
    V0 = 0.20       
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
    N_PATH = 100000
    Steps_day_N = 24
    N_TIME = int(Steps_day_N*253*T) 

    global_params = [S0, V0, r, Strk, T]
    model_params = [kap, the, lam, rho]


    np.random.seed(101)

    Monte = mixing_MC(model_params, global_params, N_PATH, N_TIME, model, option)

    print(f"\nPrice of European {option} option in model {model} is {Monte[0]:.3f}")

    print(f"\nThe Monte Carlo Standard Error is {100*Monte[1]:.3f}%")

    print(f"\nMonte Carlo mixing simulation time in seconds is {Monte[2]:.3f}s\n")