#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 07:07:30 2020

Illustrating use of Euler's methods to solve linear ODEs

@author: zettergm
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt


# Time grid
N=25
tmin=0
tmax=6
t=np.linspace(tmin,tmax,num=N)
dt=t[1]-t[0]


# Analytical solution for comparison
y0=1
alpha=2
ybar=y0*np.exp(-alpha*t)


# Forward Euler solution
yfwd=np.zeros((N))
yfwd[0]=y0
for n in range(1,N):
    yfwd[n]=yfwd[n-1]*(1-alpha*dt)


# Backward Euler solution
ybwd=np.zeros((N))
ybwd[0]=y0
for n in range(1,N):
    ybwd[n]=ybwd[n-1]/(1+alpha*dt)


# Plot results
plt.figure()
plt.plot(t,ybar,"o-")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.plot(t,yfwd,"--")
plt.plot(t,ybwd,"-.")
plt.legend(("exact","fwd Eul.","bwd Eul."))
plt.show()