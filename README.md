# sticky-HDP-HMM-Vector Autoregression
A python implementation of the sticky-HDP-HMM-VAR model[1].
For HMM stability[2] + speeding up the model some functions were implemented in C++.

## The algorithm in a nutshell
The sticky-HDP-HMM-VAR is an unsupervised machine learning algorithm used to model dynamical phenomena by switching between different linear dynamical systems (SLDS). 
Each different linear dynamical system is called an SLDS mode.  
Vector Autoregression is a special case of an SLDS in which, each observation is modeled as a linear function of the previous observations.   
The switching is governed via an HMM, therefore each SLDS mode corresponds to an HMM state. The HDP is used to model the state-transitions without any a priori knowledge on the number of states . 
  
![VAR_equation](images/var_equation.png?raw=true "Equation")


### setup
As part of the code is run in c++. To compile 
To do this run the following commands from the `utils/math` directory:
> g++ -std=c++11 -Wall -Wextra -pedantic -c -fPIC c_extensions.cpp -o c_extensions.o  
> g++ -shared c_extensions.o -o c_extensions.dylib

code was tested and compiled with  `g++` but will probably work with any other compiler

## Features
As a sanity check one can construct observations from a state-space representation of different sine-waves
in the notebook `sine_wave.ipynb` and train the model on it.
The notebook currently contains an example of observations generated via 2 sine-waves that inter-switch 3 times.
Training a model on these observations generated the following predicted vs observations plot.  
![sine_wave](images/sine_wave.png)


## References
[1] Emily Fox, Erik B. Sudderth, Michael I. Jordan, and Alan S. Willsky. Nonparametric
bayesian learning of switching linear dynamical systems. In D. Koller, D. Schuurmans,
Y. Bengio, and L. Bottou, editors, Advances in Neural Information Processing Systems 21,
pages 457–464. Curran Associates, Inc., 2009.  
https://arxiv.org/abs/1003.3829  
[2] Tobias P. Mann.  Numerically stable hidden markov model implementation.Ms. Feb,2006  
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.322.9645  