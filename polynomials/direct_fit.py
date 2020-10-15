#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:04:58 2020

Illustrates several ways to do direct polynomials fits with python

@author: zettergm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../linear_algebra")
from elimtools import Gauss_elim,backsub

# Data for a direct polynomial fit
#x=np.array([1,2,3,4])
#y=2*x**3-3*x**2+4*x+9
x=np.array([1,2,3,4,5,6])
y=x**5-2*x**4+2*x**3-3*x**2+4*x+9

# Create a plot to show results
plt.figure(1)
plt.plot(x,y,'*',markersize=20)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Illustration of direct fit methods")

# Fit using Python functions (numpy.polyfit)
n=x.size-1     # degree of polynomial to be fitted, compute from data point array size
coeffs=np.polyfit(x,y,n)
xlarge=np.linspace(x.min(),x.max(),24)
polyfun=np.poly1d(coeffs)
plt.plot(xlarge,polyfun(xlarge),'--')

# Execute direct fit using self-coded Gaussian Elimination
A=np.zeros((n+1,n+1))
for icol in range(n,-1,-1):
    newcol=x**icol
    A[:,n-icol]=newcol
[Amod,order]=Gauss_elim(A,y.reshape(-1,1),False)   #note conversion of y to column vector
coeffsGE=backsub(Amod[order,:],False)
polyfunGE=np.poly1d(coeffs)
plt.plot(xlarge,polyfunGE(xlarge),'.')
plt.legend(("original data","built-in fit","manual GE fit"))
plt.show()