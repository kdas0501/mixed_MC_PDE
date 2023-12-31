# mixed_MC_PDE
This repository contains working code for utilising the mixed Monte-Carlo PDE method detailed in Section 6 of the article 'On Stochastic Partial Differential Equations and their applications to derivative pricing via a conditional Feynman-Kac formula'.

Title: 'On Stochastic Partial Differential Equations and their applications to derivative pricing via a conditional Feynman-Kac formula'

email addresses: kaustav.das@monash.edu, ivan.guo@monash.edu, gregoire.loeper@monash.edu

### Quickstart for readers of the article

Simply run SPDE_Compare.py, which computes and compares the price and implied volatility of a European put option in the Heston, GARCH diffusion, Ornstein-Uhlenbeck, Inverse-Gamma, and Verhulst models, where the price is obtained via a Full Monte-Carlo method and our mixed Monte-Carlo PDE method, then benchmarked against the so-called Monte-Carlo Mixing Solution method.

### Main files
The following .py files are required in order to utilise the mixed Monte-Carlo PDE method.

- **SPDE_CN.py**
  Computes the price of a European put/call option in the Heston, GARCH diffusion, Ornstein-Uhlenbeck, Inverse-Gamma, and Verhulst models via the mixed Monte-Carlo PDE method, where the numerical PDE scheme utilised is Crank-Nicolson.
  
- **SPDE_semiimplicit.py**
  Computes the price of a European put/call option in the Heston, GARCH diffusion, Ornstein-Uhlenbeck, Inverse-Gamma, and Verhulst models, where the numerical PDE scheme utilised is semi-implicit.

Details of the numerical SPDE schemes utilised in the above methods are given in the PDF **Numericalschemes_SPDE.pdf**.

### Auxiliary files
The rest of the .py files are auxiliary files that are not required for the mixed Monte-Carlo PDE method.

- **mixing_MC.py**
   Computes the price of a European put/call option in the Heston, GARCH diffusion, Ornstein-Uhlenbeck, Inverse-Gamma, and Verhulst models via the Monte-Carlo Mixing Solution method.
  
- **Full_MC.py**
  Computes the price of a European put/call option in the Heston, GARCH diffusion, Ornstein-Uhlenbeck, Inverse-Gamma, and Verhulst models via a Full Monte-Carlo method.
  
- **SPDE_Compare.py**
  Computes and compares the price and implied volatility of a European put/call option in a variety of models, where the price is obtained via a Full Monte-Carlo method and our mixed Monte-Carlo PDE method, then benchmarked against the so-called Monte-Carlo Mixing Solution method.
  
- **ImpVol_Brent.py:** 
  Computes the implied volatility of a European put/call option via Brent's method.

- **DeltaStrikes.py**
  Computes the strike of an option contract corresponding to a given European put/call option Delta.
  
