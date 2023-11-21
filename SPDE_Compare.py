# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Description: Computes the price and implied volatility of a European Put/Call option using the mixed Monte Carlo PDE method and 
Full Monte Carlo method, then compares them to those computed via the mixing Monte Carlo method (benchmark) as well as their run times.  

"""

import numpy as np
from scipy.stats import norm
from Full_MC import Full_MC
from mixing_MC import mixing_MC
from SPDE_CN import SPDE_CN
from SPDE_semiimplicit import SPDE_semiimplicit
from ImpVol_Brent import ImpVol_Brent
from DeltaStrikes import DeltaStrikes

def SPDE_Compare(model_params, global_params, PATH, TIME, M_SPACE, X, model, pde_method, option):

            
      """
      Computes the price and implied volatility of a European Put/Call option using the mixed Monte Carlo PDE method and 
      Full Monte Carlo method, then compares them to those computed via the mixing Monte Carlo method (benchmark) as well as their run times. 

      model_params (list) = [kap, the, lam, rho] is a list of floats corresponding to model parameters (see model parameter).
      global_params (list) = [S0, V0, r, Strk, T]
            S0 (float): Initial value for spot.
            V0 (float): Initial value of Volatility or Variance process.
            r (float): Interest rate.
            Strk (float): Strike of contract.
            T (float): Maturity of contract.

      PATH (list) = [N_PATH_mixing, N_PATH, M_PATH]
            N_PATH_mixing = # paths for mixing Monte Carlo method.
            N_PATH = # paths for Full Monte Carlo method.
            M_PATH = # paths for mixed Monte Carlo PDE method.
      TIME (list) = [N_TIME_mixing, N_TIME, M_TIME, M_VOL_TIME]
            N_TIME_mixing = # of time steps for mixing Monte Carlo method.
            N_TIME = # of time steps for Full Monte Carlo method.
            M_TIME = # of time steps for PDE solver in mixed Monte Carlo PDE method.
            M_VOL_TIME = # of time steps for V and Z simulation in the mixed Monte Carlo PDE method (usually less than M_TIME).
      X (list)= [X_m, X_M]
            X_m = minimum in space grid.
            X_M = maximum in space grid.
      M_SPACE (int) = # points in space for PDE solver in mixed Monte Carlo PDE method.

      model (str):
            t gives "test", volatility & dV = dZ
            o gives "Ornstein", volatility & dV = kap(the - V)dt + lam dZ
            g gives "GARCH", variance & dV = kap(the - V)dt + lam V dZ
            i gives "Inverse-Gamma", volatility & dV = kap(the - V)dt + lam V dZ
            h gives "Heston", variance & dV = kap(the - V)dt + lam sqrt{V} dZ
            v gives "Verhulst", volatility & dV = kap*V*(the - V)dt + lam V dZ

      pde_method (int): 0 gives Crank Nicolson, 1 gives semi-implicit for the PDE solver in the mixed Monte Carlo method.
      option = "Put" gives European put option, "Call" gives European call option.

      """

      # Common parameters
      S0 = global_params[0]
      V0 = global_params[1]       # Note that V0 is initial variance for Heston and GARCH models
      r = global_params[2]
      Strk = global_params[3]
      T = global_params[4]
 
      kap = model_params[0]
      the = model_params[1]
      lam = model_params[2]
      rho = model_params[3]

      # mixing MC parameters
      N_TIME_mixing = TIME[0]
      N_PATH_mixing = PATH[0]

      # Full MC parameters
      N_TIME = TIME[1]
      N_PATH = PATH[1]

      # SPDE parameters
      X_m = X[0]
      X_M = X[1]
      M_VOL_TIME = TIME[2]
      M_TIME = TIME[3]
      # M_TIME = 100
      M_SPACE = M_SPACE
      M_PATH = PATH[2]



      # Compute prices, SE and elapsedtimes for mixingMC, MC and SP
      np.random.seed(101)
      (mixingMC, mixingMC_SE, elapsed_mixingMC) = mixing_MC(model_params, global_params, N_PATH_mixing, N_TIME_mixing, model, option)

      np.random.seed(102)
      (MC, MC_SE, elapsed_MC) = Full_MC(model_params, global_params, N_PATH, N_TIME, model, option)

      np.random.seed(103)
      if pde_method == 0:
            (SP, SP_SE, elapsed_SP) = SPDE_CN(model_params, global_params, X, M_VOL_TIME, M_TIME, M_SPACE, M_PATH, model, option)
      elif pde_method == 1:
            (SP, SP_SE, elapsed_SP) = SPDE_semiimplicit(model_params, global_params, X, M_VOL_TIME, M_TIME, M_SPACE, M_PATH, model, option)

      # Standard Errors in percent:
      mixingMC_SE_cent = 100*mixingMC_SE
      MC_SE_cent = 100*MC_SE
      SP_SE_cent = 100*SP_SE





      # Compute implied volatilities for mixing_MC, MC, SP
      ImpVolmixingMC = ImpVol_Brent(S0, Strk, r, T, option, mixingMC)
      ImpVolMC = ImpVol_Brent(S0, Strk, r, T, option, MC)
      ImpVolSP = ImpVol_Brent(S0, Strk, r, T, option, SP)





      # Compute impvol 'standard error' for mixingMC             
      dpl_mixing = (np.log(S0/Strk) + r*T)/(ImpVolmixingMC*np.sqrt(T)) + 0.5*ImpVolmixingMC*np.sqrt(T)
      Deriv_vega_mixing = S0*np.sqrt(T)*norm.pdf(dpl_mixing)
      SE_ImpVolMC_mixing_bp = 10000*(Deriv_vega_mixing**(-1))*mixingMC_SE






      # Option absolute and relative errors for MC in percent (cent)
      MC_AbsErr_cent = np.abs(MC-mixingMC)*100
      MC_RelErr_cent = MC_AbsErr_cent/mixingMC if mixingMC != 0 else 0

      # ImpVol absolute and relative errors in bp for MC
      ImpVolMC_Abs_Err_bp = 10000*(np.abs(ImpVolMC - ImpVolmixingMC))
      ImpVolMC_Rel_Err_bp = ImpVolMC_Abs_Err_bp/ImpVolmixingMC

      # Compute impvol 'standard error' for MC
      dpl_MC = (np.log(S0/Strk) + r*T)/(ImpVolMC*np.sqrt(T)) + 0.5*ImpVolMC*np.sqrt(T)
      Deriv_vega_MC = S0*np.sqrt(T)*norm.pdf(dpl_MC)
      SE_ImpVolMC_bp = 10000*(Deriv_vega_MC**(-1))*MC_SE







      # Option absolute and relative errors for SP in percent (cent)

      SP_AbsErr_cent = np.abs(SP-mixingMC)*100
      SP_RelErr_cent = SP_AbsErr_cent/mixingMC if mixingMC != 0 else 0

      # ImpVol absolute and relative error in bp for SP
      ImpVolSP_Abs_Err_bp = 10000*(np.abs(ImpVolSP - ImpVolmixingMC))
      ImpVolSP_Rel_Err_bp = ImpVolSP_Abs_Err_bp/ImpVolmixingMC if ImpVolmixingMC !=0 else 0

      # Compute impvol 'standard error' for SP
      dpl_SP = (np.log(S0/Strk) + r*T)/(ImpVolSP*np.sqrt(T)) + 0.5*ImpVolSP*np.sqrt(T)
      Deriv_vega_SP = S0*np.sqrt(T)*norm.pdf(dpl_SP)
      SE_ImpVolSP_bp = 10000*(Deriv_vega_SP**(-1))*SP_SE





   
      print(f"\n{option} option in model {model}\n")

      print("Prices\n")
      print(f"Benchmark (mixing MC): {mixingMC:.3f} S.E. {mixingMC_SE_cent:.3f}%")
      print(f"Full MC: {MC:.3f} S.E. {MC_SE_cent:.3f}% AbsErr {MC_AbsErr_cent:.3f}% RelErr {MC_RelErr_cent:.3f}%")
      print(f"mixed MC PDE: {SP:.3f} S.E. {SP_SE_cent:.3f}% AbsErr {SP_AbsErr_cent:.3f}% RelErr {SP_RelErr_cent:.3f}%\n")
      
      print("Implied volatilities\n")
      print(f"Benchmark (mixing MC): {100*ImpVolmixingMC:.3f}% S.E. {SE_ImpVolMC_mixing_bp:.3f}bp")
      print(f"Full MC: {100*ImpVolMC:.3f}% S.E. {SE_ImpVolMC_bp:.3f}bp AbsErr {ImpVolMC_Abs_Err_bp:.3f}bp RelErr {ImpVolMC_Rel_Err_bp:.3f}bp")
      print(f"mixed MC PDE: {100*ImpVolSP:.3f}% S.E. {SE_ImpVolSP_bp:.3f}bp AbsErr {ImpVolSP_Abs_Err_bp:.3f}bp RelErr {ImpVolSP_Rel_Err_bp:.3f}bp\n")

      
      print("Runtimes\n")
      print(f"Benchmark (mixing MC): {elapsed_mixingMC:.3f}s")
      print(f"Full MC: {elapsed_MC:.3f}s")
      print(f"mixed MC PDE: {elapsed_SP:.3f}s\n")





# Example usage:

if __name__ == "__main__":
      
      option = "Put"

      # Common parameters
      S0 = 100
      V0 = 0.20      
      r = 0.01
      T = 6/12.0
      Delta = 0.5
      impsigguess = V0
      Strk = DeltaStrikes(impsigguess, S0, r, T, Delta, option)
      kap = 5
      the = 0.18
      lam =  0.9
      rho = -0.35

      # mixing MC parameters
      Steps_day_N_mixing = 24 # 24 steps a day (make it less to make it faster)
      N_TIME_mixing = int(Steps_day_N_mixing*253*T)
      N_PATH_mixing = 1000000

      # Full MC parameters
      Steps_day_N = 36 # 24 steps a day (make it less to make it faster)
      N_TIME = int(Steps_day_N*253*T)
      N_PATH = 200000

      # SPDE parameters
      L = 4*V0*np.sqrt(T)
      x0 = np.log(S0/Strk)
      X_m = x0 - L
      X_M = x0 + L
      Steps_day_M_Vol = 1
      Steps_day_M = 1
      M_VOL_TIME = int(Steps_day_M_Vol*253*T)
      M_TIME = int(Steps_day_M*253*T) 
      # M_TIME = 100
      M_SPACE = 250
      M_PATH = 20000

      model = "i"

      # 0 gives Crank Nicolson, 1 gives semi-implicit
      pde_method = 0



  

      global_params = [S0, V0, r, Strk, T]
      model_params = [kap, the, lam, rho]
      X = [X_m, X_M]

      PATH = [N_PATH_mixing, N_PATH, M_PATH]
      TIME = [N_TIME_mixing, N_TIME, M_TIME, M_VOL_TIME]




      SPDE_Compare(model_params, global_params, PATH, TIME, M_SPACE, X, model, pde_method, option)


