#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 07:44:03 2020

Illustrate use of RK2 to solve a system of equations of motion

charged particle moving in the x-y direction in a magnetic field which has only a z-component...

 m dv/dt = q v x B 
(ma = F)

 v = (vx,vy); B=(0,0,B);

 Resulting in the following system of equations:
    m dvx/dt = q vy B
    m dvy/dt = -q vx B

@author: zettergm
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# Parameters of system
q=-1.6e-19;
m=9.1e-31;
B=10000e-9;
omega=q*B/m;    # frequency of oscillation (can be shown via solution by hand gives a SHO)
tmin=0;
tmax=2*2*np.pi/np.abs(omega);    # follow particle for two oscillation periods
t=np.linspace(tmin,tmax,50);
dt=t[1]-t[0];
lt=t.size;


# Main RK2 loop to solve for v
vx=np.zeros((lt))
vy=np.zeros((lt))
vx[0]=1e3
vy[0]=1e3
for n in range(1,lt):
    # half step update for both components
    vxhalf=vx[n-1]+dt/2*(omega*vy[n-1])
    vyhalf=vy[n-1]-dt/2*(omega*vx[n-1])
    
    # full step update for both components
    vx[n]=vx[n-1]+dt*(omega*vyhalf)
    vy[n]=vy[n-1]-dt*(omega*vxhalf)


# Integrate v to get position (assume particle starts at 0,0)
x=integrate.cumtrapz(vx,t)
y=integrate.cumtrapz(vy,t)
vz=1e3
z=vz*t[0:-1]


# Plot velocity 
plt.figure()
plt.plot(t,vx,"-")
plt.plot(t,vy,"--")
plt.xlabel("t")
plt.ylabel("velocity component")
plt.legend(("vx","vy"))
plt.show()


# Plot position
plt.figure()
ax=plt.axes(projection="3d")
ax.plot3D(x,y,z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()




