#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 07:23:35 2020

Illustrate the use of Runge-Kutta methods to solve ODEs

@author: zettergm
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt


# RHS of ODE for use with RK4
def fRK(t,y,alpha):
    fval=-alpha*y
    return fval


# Time grid
N=15
tmin=0
tmax=6
t=np.linspace(tmin,tmax,num=N)
dt=t[1]-t[0]


# Analytical solution for comparison
y0=1
alpha=2
ybar=y0*np.exp(-alpha*t)


# RK2
yRK2=np.zeros((N))
yRK2[0]=y0
for n in range(1,N):
    yhalf=yRK2[n-1]+dt/2*(-alpha*yRK2[n-1])
    yRK2[n]=yRK2[n-1]+dt*(-alpha*yhalf)


# RK4
yRK4=np.zeros((N))
yRK4[0]=y0
for n in range(1,N):
    dy1=dt*fRK(t[n-1],yRK4[n-1],alpha)
    dy2=dt*fRK(t[n-1]+dt/2,yRK4[n-1]+dy1/2,alpha)
    dy3=dt*fRK(t[n-1]+dt/2,yRK4[n-1]+dy2/2,alpha)
    dy4=dt*fRK(t[n-1]+dt,yRK4[n-1]+dy3,alpha)
    yRK4[n]=yRK4[n-1]+1/6*(dy1+2*dy2+2*dy3+dy4)


# Plot results
plt.figure()
plt.plot(t,ybar,"o-")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.plot(t,yRK2,"--")
plt.plot(t,yRK4,"-.")
plt.legend(("exact","RK2","RK4"))
plt.show()


# RK2 stability plot
adt=np.linspace(0.01,3,20)
ladt=adt.size
G=np.zeros((ladt))
for igain in range(0,ladt):
    G[igain]=(1-adt[igain]+1/2*adt[igain]**2)

plt.figure()
plt.plot(adt,G,"o")
plt.xlabel("a*dt")
plt.ylabel("gain factor")
plt.show()