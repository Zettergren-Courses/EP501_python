#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 07:17:06 2020

Show how to do simple interpolation using Python

@author: zettergm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../linear_algebra")
from elimtools import Gauss_elim,backsub
import scipy.interpolate as spint

# Manual implmementatino of bilinear interpolation, single point
x=np.array([1,2]);
y=np.array([2,3]);
f=np.array([[10, 11], [12, 13]]);
[X,Y]=np.meshgrid(x,y);

# Point to which we are interpolating
x1=1.5;
y1=2.5;

# Conversion of problem to matrix form
xvec=np.reshape(X,(4,1))
yvec=np.reshape(Y,(4,1))
fvec=np.reshape(f,(4,1))
M=np.concatenate( (np.ones((4,1)),xvec,yvec,xvec*yvec), axis=1)
[Mmod,order]=Gauss_elim(M,fvec,False)
avec=backsub(Mmod[order,:],False)
finterp=avec[0]+avec[1]*x1+avec[2]*y1+avec[3]*x1*y1
print("Coefficients:  ",avec)
print("Interpolated function value:  ",finterp)

# Python cubic splines
x2=np.linspace(-5,5,15)
y2=np.sin(x2)
x3=np.linspace(min(x2),max(x2),64)
f3=spint.interp1d(x2,y2,kind="cubic")
y3=f3(x3)

plt.figure(1)
plt.plot(x2,y2,"o",markersize=20)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x3,y3,'-')
plt.legend(("data","spline"))