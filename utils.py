# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:13:26 2020

@author: sebastian franzen
"""
import numpy as np

'''
def beta_func(t, beta0, beta1, t0, w):
    
    Gives the beta parameter at time t. The function is constant to time t0-w,
    and from time t0+w. In between, it changes linearly from the valu at t1-w
    to the value at t1+w.
    
    if t <= t0 - w:
        return beta0
    if t0 - w < t < t0 + w:
        return beta0 + (t -(t0 - w))*(beta1 -beta0)/(2*w)
    if t0 + w <= t: 
        return beta1
'''

def SEIR_differential(SEIR, t, beta, rho, gamma):
    ''' 
    Caluclates the differential for Euler's forward method to solve
    the SEIR system of differential equations.
    
    SEIR: should  be an array (or list) of 4 elements
    t: point in time (mandatory for integral.odeint)
    beta: infection rate  
    1/rho: transition rate from effected to infectious
    1/gamma: rate of recoverry    
    '''
    # generate current beta coefficient
    #beta = beta_func(t, beta0, beta1, t0, w)
    
    S,E,I,R = SEIR
    dS_dt = -beta*S*I
    dE_dt = beta*S*I - E/rho
    dI_dt = E/rho - I/gamma
    dR_dt = I/gamma
    
    return dS_dt,dE_dt,dI_dt,dR_dt


def solve_seir(T, n, S0, I0, beta, rho, gamma):
    '''
    Gives a solution to the system of differential equations
    
    S'(t) = -betaS(t)I(t)
    E'(t) = betaS(t)I(t) - E(t)/rho
    I'(t) = E(t)/rho - I(t)/gamma
    R'(t) = I(t)/gamma
    
    for n*T points evenly space between 0 and T and with inital values
    S0, I0. Returns every n:th point, that is an array of shape (T, 4).
    
    T, n: intgers
    beta: infection rate  
    1/rho: transition rate from effected to infectious
    1/gamma: rate of recovery   
    
    Note:
    1) E and R are initally set to Zero inside the function. 
    
    2) If any of the solution curves turns negative, the next to last value is
    returned constantly for the rest of the points. This prevents the function
    from blowing up.
    '''    
    dt = T/float(n*T)

    t_points = np.linspace(0, T, n*T +1)
    
    SEIR = np.empty((n*T +1 , 4))
    
    SEIR[0] = [S0, 0, I0, 0]

    #N = S0 + I0
    for k in range(n*T):
        
        new_differential = SEIR_differential(SEIR[k], t_points[k],
                                             beta, rho, gamma)
        SEIR[k+1] = SEIR[k] + dt * np.asarray(new_differential)
        
        # break if solution curve turns negative
        if sum(SEIR[k+1] < 0) >= 1:
            # fill remaining of array with last nonnegative row
            SEIR[k+1 : n*T +1] = SEIR[k]
            
            # return every n:th element to have returns of length T
            return SEIR[::n]
        
    # return every n:th element to have returns of length T
    return SEIR[::n]
