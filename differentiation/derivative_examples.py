#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 07:39:12 2020

Illustrating various approaches to computing ordinary derivatives

@author: zettergm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Test function and exact derivative
lx=20       #number of points in numerical grid
x=np.linspace(-10,10,lx)
dx=x[1]-x[0]     # grid step size or resolution
y=np.sin(0.5*x)
yprime=0.5*np.cos(0.5*x)
ydprime=-0.25*np.sin(0.5*x)

# Plots
plt.figure(1)
plt.plot(x,y)
plt.plot(x,yprime)
plt.xlabel("x")
plt.ylabel("y and dy/dx")

# Centered difference approximation to the first derivative (recommended approach for most problems)
dydx=np.zeros(lx)
dydx[0]=(y[1]-y[0])/dx                  #cannot do centered difference on the edges; foward difference at the beginning
for ix in range(1,lx-1):
    dydx[ix]=(y[ix+1]-y[ix-1])/2/dx     #centered difference in the interior points
dydx[lx-1]=(y[lx-1]-y[lx-2])/dx         #backward difference on the end

plt.plot(x,dydx,"--")

# Foward difference approximation
dydx_fwd=np.zeros(lx)
dydx_fwd[0]=(y[1]-y[0])/dx                  #foward difference at the beginning
for ix in range(1,lx-1):
    dydx_fwd[ix]=(y[ix+1]-y[ix])/dx     #centered difference in the interior points
dydx_fwd[lx-1]=(y[lx-1]-y[lx-2])/dx         #backward difference on the end

plt.plot(x,dydx_fwd,'.')

# Demo of second derivative, centered formula
d2ydx2=np.zeros(lx)     # note that we are not computing edges values for 2nd deriv.
for ix in range(1,lx-1):
    d2ydx2[ix]=(y[ix+1]-2*y[ix]+y[ix-1])/dx**2

# Plot second derivative and legend
plt.plot(x,ydprime,'-')
plt.plot(x,d2ydx2,'^')
plt.legend(("y","dy/dx","centered","forward","2nd deriv. by hand","2nd deriv."))
plt.show()

