#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:15:32 2020

Illustrate handling multiple time scales (ODE stiffness, book example from Gear's paper)

 dy/dt=-alpha*(y=F(t))+F'(t)
 y(t)=(y0-F(0))exp(-alpha*t)+F(t)

@author: zettergm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt


# System parameters
y0=1
alpha=1000


# Time grid
tsmin=0
tsmax=4
dts=0.0015      # choice of time step
ts=np.arange(tsmin,tsmax,dts)


# True solution
ybar=(y0-2)*np.exp(-alpha*ts)+ts+2;
Ns=ts.size;


# Iterate forward and Backward Euler together in a single loop
yfwds=np.zeros((Ns))
ybwds=np.zeros((Ns))
yfwds[0]=y0
ybwds[0]=y0
for n in range(1,Ns):
    yfwds[n]=yfwds[n-1]+dts*(-1000*(yfwds[n-1]-ts[n-1]-2)+1)
    ybwds[n]=(ybwds[n-1]+1000*ts[n-1]*dts+2001*dts)/(1+1000*dts)


# Plot solutions
plt.figure()
plt.plot(ts,ybar)
plt.plot(ts,yfwds)
plt.plot(ts,ybwds)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend(("true soln.","fwd Eul.","bwd Eul."))
plt.show()
