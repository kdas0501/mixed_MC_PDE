# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Description: Computes solution to the informal SPDE for a Put or Call option in the model 

dS = rSdt + S(V or sqrt{V})dW
dV = alpha(t,V)dt + beta(t,V) dZ

utilising the Crank Nicolson numerical PDE scheme as detailed in Section 6 of the article: 
On Stochastic Partial Differential Equations and their applications to derivative pricing through a conditional Feynman-Kac formula.

"""

import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.sparse import spdiags
from timeit import default_timer as timer

def SPDE_CN(model_params, global_params, X, M_VOL_TIME, M_TIME, M_SPACE, M_PATH, model, option):

    """
    Computes solution to the informal SPDE for Put or Call option utilising the Crank Nicolson
    scheme as detailed in Section 6 of the article.
    
    model_params (list)= [kap, the, lam, rho] is a list of floats corresponding to model parameters (see model parameter).
    global_params (list) = [S0, V0, r, Strk, T]
        S0 (float): Initial value for spot.
        V0 (float): Initial value of Volatility or Variance process.
        r (float): Interest rate.
        Strk (float): Strike of contract.
        T (float): Maturity of contract.

    X (list)= [X_m, X_M]
        X_m = minimum in space grid.
        X_M = maximum in space grid.
    M_TIME (int) = # of time steps for PDE solver.
    M_VOL_TIME (int)= # of time steps for V and Z simulation (usually less than M_TIME).
    M_SPACE (int) = # points in space.
    M_PATH (int)= # of Monte Carlo paths.

    model (str):
        t gives "test", volatility & dV = dZ
        o gives "Ornstein", volatility & dV = kap(the - V)dt + lam dZ
        g gives "GARCH", variance & dV = kap(the - V)dt + lam V dZ
        i gives "Inverse-Gamma", volatility & dV = kap(the - V)dt + lam V dZ
        h gives "Heston", variance & dV = kap(the - V)dt + lam sqrt{V} dZ
        v gives "Verhulst", volatility & dV = kap*V*(the - V)dt + lam V dZ

    option = "Put" gives European put option, "Call" gives European call option

    """

    # np.random.seed(103)
        
    # Extract parameters
    kap = model_params[0]
    the = model_params[1]
    lam = model_params[2]
    rho = model_params[3]
    
    S0 = global_params[0]
    V0 = global_params[1]
    r = global_params[2]
    Strk = global_params[3]
    T = global_params[4]
    
    X_m = X[0]
    X_M = X[1]
    
    # Redefine M_PATH as M_PATH_total, as we will do the Monte Carlo 
    # component of the method in batches (for memory saving reasons). 
    # E.g., if batch size is 250 and we have M_PATH_total = 1000, the routine
    # will be repeated 4 times

    M_PATH_total = M_PATH
    
    # Feller test for Heston model
    if 2*kap*the < lam**2 and model == "h":
        print("Feller test fail!")
        
    # Start timer
    start_whole = timer()
    
    # Time parameters on volatility simulation grid (usually coarse)
    dtvol = T/M_VOL_TIME
    sqrtdtvol = np.sqrt(dtvol)
    
    # Space parameters
    dx = (X_M - X_m)/M_SPACE
    repdx = 1/dx
    repsqdx = repdx**2
    x0 = np.log(S0/Strk)
    
    # SDE parameters
    kapdtvol = kap*dtvol
    kapthedtvol = kapdtvol*the
      
    # The following guarantees x0 is actually in the space grid!
    x_left = np.arange(x0, X_m, -dx)
    x_right = np.arange(x0+dx, X_M+dx, dx)
    x_left = np.flip(x_left)
    x = np.concatenate((x_left, x_right))
    M_SPACE = len(x)-1
    
    # s is the point in the space grid corresponding to x0.
    s = np.searchsorted(x,x0)
    # print(x0,x[s])
    
    # inc refers to the increment size used to sample the coarse versions 
    # of V and Z later on.
    inc = int(M_VOL_TIME/M_TIME)
    M_TIME = int(M_VOL_TIME/inc)
    dt = T/M_TIME
    sqrtdt = np.sqrt(dt)
    
    # We now execute the PDE solving of M_PATH_total paths in batches of batch_size.
    # This is because if M_PATH_total is very large, we will run into memory problems.
    # But if batch_size is too small, it becomes very computationally costly!
    # batch_size = 150 seems optimal for M_SPACE = 250.
    
    batch_size = 150
    M_PATH_remainder = np.remainder(M_PATH_total, batch_size)
    M_PATH_it = int(M_PATH_total/batch_size)    # number of batches
    z_o = 0 if M_PATH_remainder == 0 else 1 #issue with #loop when remainder is 0
    
    
    for l in range(M_PATH_it + z_o):
        
        M_PATH = M_PATH_remainder if l == M_PATH_it else batch_size
    
        xones = np.outer(x,np.ones([M_PATH])) # This is a column stack of x vectors
        expxones = np.exp(xones)
        ones = np.ones([M_SPACE+1, M_PATH])

        # Simulate Z and V on the fine time grid. Since we are solving 
        # backwards in time and V goes forwards, we have to presimulate and store it.    
        
        Z_FINE = np.zeros([M_VOL_TIME +1, M_PATH])
        V_FINE = np.zeros([M_VOL_TIME +1, M_PATH])
        V_FINE[0,:] = V0
        Ztemp = np.zeros(M_PATH)
        Vtemp = np.ones(M_PATH)*V0
      
        # dV = alpha(t,V)dt + beta(t,V)dB
        for tt in range(M_VOL_TIME):
            RND = np.random.normal(0, 1, M_PATH)
            dZ = sqrtdtvol*RND
            Ztemp += dZ
            
            # Test: Volatility and dV = dB         
            if model == "t":
                Vtemp += dZ
            
            # Ornstein Uhlenbeck: Volatility & dV = kap(the - V)dt + lam*dB     
            elif model == "o":
                Vtemp += kapthedtvol - kapdtvol*Vtemp + lam*dZ
                
            # GARCH: Variance & dV = kap(theta - V)dt + lam*V*dB
            # Inverse Gamma: Volatility & dV = kap(the - V)dt + lam*V*dB
            elif model in ["g","i"]:
                Vtemp += kapthedtvol - kapdtvol*Vtemp + lam*Vtemp*dZ
                
            # Heston: Variance & dV = kap(the - V)dt + lam*sqrt{V}*dB
            elif model == "h":          
                mV = np.maximum(Vtemp,0)
                sqrtmV = np.sqrt(mV)
                Vtemp += kapthedtvol - kapdtvol*mV + lam*sqrtmV*dZ
                
            # Verhulst: Volatility & dV = kap*V*(the - V)dt + lam*V*dB
            elif model == "v":
                Vsqr = Vtemp**2
                Vtemp += kapthedtvol*Vtemp - kapdtvol*Vsqr + lam*Vtemp*dZ   
                        
            Z_FINE[tt+1,:] = Ztemp
            V_FINE[tt+1,:] = Vtemp
            
            
        # Now create the coarse versions of V and B by extracting the elements
        # from V_FINE and Z_FINE by stepping with inc.
        
        V = V_FINE[0::inc, :].copy()
        Z = Z_FINE[0::inc,:].copy()
            
        
        # Initilise the array U, which contains M_PATH copies of the solution vector
        U = np.zeros([M_SPACE+1, M_PATH])
        
        # Terminal condition
        if option == "Call":
            U = Strk*np.maximum(expxones - 1.0, 0)
        elif option == "Put":
            U = Strk*np.maximum(1.0 - expxones, 0)
        
        

        # Now all simluation of Z and V is done, we must consider the Crank Nicolson numerical PDE scheme.
        # We have U_i A_i = B_{i+1} U_{i+1} for suitable matrices A_i and B_{i+1} which
        # can be derived from the scheme in Section 6 of the article.

        # Construct all M_PATH realisations of the matrices A_i and B_{i+1} on each time loop i.
        
        # We will use that current V_ii = old V_i, etc. The compiler may get
        # annoyed about this, but the code is fine.
        for i in range(M_TIME-1,-1,-1):
            
            if i == M_TIME-1:
                V_ii = V[i+1,:]
                Z_ii = Z[i+1,:]
            else:
                V_ii = V_i
                Z_ii = Z_i
              
            V_i = V[i,:]
            Z_i = Z[i,:]
                
            DeltaZ = Z_ii - Z_i
              
            # Set up parameters unique to each model
            
            # Test: Volatility and dV = dB         
            if model == "t":
                if i == M_TIME - 1: 
                    mu_ii = r-.5*V_ii**2
                    sig_ii = V_ii
                    beta_ii = 1.0
                    dsig_ii = 1.0            
                else:
                    mu_ii = mu_i
                    sig_ii = sig_i
                    beta_ii = beta_i
                    dsig_ii = dsig_i
                    
                mu_i = r-.5*V_i**2
                sig_i = V_i
                beta_i = 1.0
                dsig_i = 1.0
            
            # Ornstein Uhlenbeck: Volatility & dV = kap(the - V)dt + lam*dB     
            elif model == "o":
                if i == M_TIME - 1: 
                    mu_ii = r-.5*V_ii**2
                    sig_ii = V_ii
                    beta_ii = lam
                    dsig_ii = 1.0
                else:
                    mu_ii = mu_i
                    sig_ii = sig_i
                    beta_ii = beta_i
                    dsig_ii = dsig_i
                    
                mu_i = r-.5*V_i**2
                sig_i = V_i
                beta_i = lam
                dsig_i = 1.0
                  
            # GARCH: Variance & dV = kap(the - V)dt + lam*V*dB
            elif model == "g":            
                  if i == M_TIME - 1: 
                      sqrtV_ii = np.sqrt(np.maximum(V_ii,0))
                      mu_ii = r - .5*V_ii
                      sig_ii = sqrtV_ii
                      beta_ii = lam*V_ii
                      repsqrtV_ii = np.array([val**(-1) if val != 0 else 0 for val in sqrtV_ii])
                      dsig_ii = .5*repsqrtV_ii   
                  else:
                      mu_ii = mu_i
                      sig_ii = sig_i
                      beta_ii = beta_i
                      dsig_ii = dsig_i   
                    
                  sqrtV_i = np.sqrt(np.maximum(V_i,0))
                  mu_i = r - .5*V_i
                  sig_i = sqrtV_i
                  beta_i = lam*V_i
                  repsqrtV_i = np.array([val**(-1) if val != 0 else 0 for val in sqrtV_i])
                  dsig_i = .5*repsqrtV_i     
                      
            # Inverse-Gamma: Volatility & dV = kap(the - V)dt + lam*V*dB
            elif model == "i":
                  if i == M_TIME - 1: 
                      mu_ii = r - .5*V_ii**2
                      sig_ii = V_ii
                      beta_ii = lam*V_ii
                      dsig_ii = 1.0  
                  else:
                      mu_ii = mu_i
                      sig_ii = sig_i
                      beta_ii = beta_i
                      dsig_ii = dsig_i
                    
                  mu_i = r - .5*V_i**2
                  sig_i = V_i
                  beta_i = lam*V_i
                  dsig_i = 1.0
                      
            # Heston: Variance & dV = kap(the - V)dt + lam*sqrt{V}*dB
            elif model == "h":
                  if i == M_TIME - 1: 
                      sqrtV_ii = np.sqrt(np.maximum(V_ii,0))
                      mu_ii = r - .5*V_ii
                      sig_ii = sqrtV_ii
                      beta_ii = lam*sqrtV_ii
                      repsqrtV_ii = np.array([val**(-1) if val != 0 else 0 for val in sqrtV_ii])
                      dsig_ii = .5*repsqrtV_ii 
                  else:
                      mu_ii = mu_i
                      sig_ii = sig_i
                      beta_ii = beta_i
                      dsig_ii = dsig_i
                                         
                  sqrtV_i = np.sqrt(np.maximum(V_i,0))
                  mu_i = r - .5*V_i
                  sig_i = sqrtV_i
                  beta_i = lam*sqrtV_i
                  repsqrtV_i = np.array([val**(-1) if val != 0 else 0 for val in sqrtV_i])
                  dsig_i = .5*repsqrtV_i     
                                  
            # Verhulst: Volatility & dV = kap*V*(the - V)dt + lam*V*dB
            elif model == "v":
                  if i == M_TIME - 1: 
                      mu_ii = r - .5*V_ii**2
                      sig_ii = V_ii
                      beta_ii = lam*V_ii
                      dsig_ii = 1.0
                  else:
                      mu_ii = mu_i
                      sig_ii = sig_i
                      beta_ii = beta_i
                      dsig_ii = dsig_i
                    
                  mu_i = r - .5*V_i**2
                  sig_i = V_i
                  beta_i = lam*V_i
                  dsig_i = 1.0      
            
            # Now that the unique model parameters are set, we do some manipulations
            # common to all models
                
            if i == M_TIME - 1: 
                muondx_ii = mu_ii*repdx*ones
                sigsqonsqdx_ii = sig_ii**2*repsqdx*ones
                rhosigondx_ii = rho*sig_ii*repdx*ones
                rhobetadsigondx_ii = rho*beta_ii*dsig_ii*repdx*ones
            else: 
                muondx_ii = muondx_i
                sigsqonsqdx_ii = sigsqonsqdx_i
                rhosigondx_ii = rhosigondx_i
                rhobetadsigondx_ii = rhobetadsigondx_i
           
            muondx_i = mu_i*repdx*ones
            sigsqonsqdx_i = sig_i**2*repsqdx*ones
            rhosigondx_i = rho*sig_i*repdx*ones
            rhobetadsigondx_i = rho*beta_i*dsig_i*repdx*ones
    
            # For the matrices A and B we first set up vectors a, b, c
              
            if i == M_TIME - 1:
                a_ii = .5*(sigsqonsqdx_ii - muondx_ii + rhobetadsigondx_ii)*dt
                b_ii = -1.0*sigsqonsqdx_ii*dt
                c_ii = .5*(sigsqonsqdx_ii + muondx_ii - rhobetadsigondx_ii)*dt
            else:
                a_ii = a_i
                b_ii = b_i
                c_ii = c_i       
                            
            a_i = .5*(sigsqonsqdx_i - muondx_i + rhobetadsigondx_i)*dt
            b_i = -1.0*sigsqonsqdx_i*dt
            c_i = .5*(sigsqonsqdx_i + muondx_i - rhobetadsigondx_i)*dt
            
            
            # For matrix B: we now set up tilde versions of a, b, c
            # These include the increment of Brownian motion
    
            a_ii_t = a_ii - rhosigondx_ii*DeltaZ
            b_ii_t_p2 = b_ii + 2.0
            c_ii_t_p2 = c_ii + rhosigondx_ii*DeltaZ
                
            # Now we adjust the vectors for matrix B
            
            a_ii_t[M_SPACE-1,:] = 0
            
            b_ii_t_p2[0,:] = 1.0
            b_ii_t_p2[M_SPACE,:] = 1.0
            
            c_ii_t_p2[1,:] = 0
            
                
            # For matrix A: first define b_i_m2
            b_i_m2 = b_i - 2.0
    
            # Now we adjust the vectors for matrix A.
            # cp means copy, we require the originals on the next iteration

            a_i_cp = cp.copy(a_i)
            a_i_cp[M_SPACE-1,:] = 0
            
            b_i_m2[0,:] = -1.0
            b_i_m2[M_SPACE,:] = -1.0
            
            c_i_cp = cp.copy(c_i)
            c_i_cp[1,:] = 0
                        
            # Loop over the Monte Carlo paths and solve the linear system on each iteration,
            # A_i U_i = B_{i+1}U_{i+1}. We solve for every path by looping over k, but only for 
            # the ith time step.
            for k in range(M_PATH):  
                
                # Create matrices
                B = spdiags([a_ii_t[:,k], b_ii_t_p2[:,k], c_ii_t_p2[:,k]], [-1,0,1], M_SPACE+1, M_SPACE+1)
                A = -1.0*np.row_stack((c_i_cp[:,k], b_i_m2[:,k], a_i_cp[:,k]))
                
                # Solve the system AU^i = BU^{i+1}
                m = B @ U[:,k]
                U[:,k] = solve_banded((1,1), A, m)
            
            
            


        # PDE solving is done for the batch of simulations. 
        # Now concantenate the batch of simulations into one array
        U_batches = U[s,:] if l == 0 else np.concatenate((U[s,:], U_batches))


    # Compute price by averaging over all realisations of U_batch
    H = np.exp(-r*T)*np.mean(U_batches)
    
    end_whole = timer()
    


    # Compute the standard deviation and standard error
    Sigma = np.exp(-r*T)*np.std(U_batches)
    SE = Sigma/np.sqrt(M_PATH_total)
    
    elapsed_whole = end_whole - start_whole

    return (H, SE, elapsed_whole)








# Example usage:

if __name__ == '__main__':
    
    from DeltaStrikes import DeltaStrikes

    model = "i"
    option = "Put"

    # Common parameters
    S0 = 100
    V0 = 0.20       
    r = 0.01
    T = 6/12.0
    Delta = 0.5
    impsigguess = V0
    Strk = DeltaStrikes(impsigguess, S0, r, T, Delta, option)
    # Strk = S0*1.01
    kap = 5
    the = 0.18
    lam = 0.9
    rho = -0.35

    # SPDE parameters
    L = 4*V0*np.sqrt(T)
    x0 = np.log(S0/Strk)
    X_m = x0 - L
    X_M = x0 + L
    Steps_day_M_Vol = 1
    Steps_day_M = 1
    M_VOL_TIME = int(Steps_day_M_Vol*253*T)
    M_TIME = int(Steps_day_M*253*T) 
    M_SPACE = 250
    M_PATH = 5000

    global_params = [S0, V0, r, Strk, T]
    model_params = [kap, the, lam, rho]
    X = [X_m, X_M]
    



    SP = SPDE_CN(model_params, global_params, X, M_VOL_TIME, M_TIME, M_SPACE, M_PATH, model, option)
        
    print(f"\nPrice of European put option in model {model}:\n{SP[0]:.3f}")
    
    print(f"\nThe Monte Carlo Standard Error is {100*SP[1]:.3f}%")
    
    print(f"\nSPDE Crank Nicolson solving time in seconds {SP[2]:.3f}s\n")