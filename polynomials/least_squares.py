#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:53:36 2020

Illustrates linear least squares fitting of data

@author: zettergm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../linear_algebra")
from elimtools import Gauss_elim,backsub

# Grid and indep. vars for problem, linear function y=a+b*x
n=40     #number of data points
a=2      #y-intercept, linear fn.
b=3      #slope, linear fn.
minx=-5
maxx=5
xdata=np.linspace(minx,maxx,n)

# Gemeration of Gaussian random numbers in Python
dev=5.0
mean=0.5      #models callibration error in measurement, offset
noise=dev*np.random.randn(n)+mean
ytrue=a+b*xdata
ydata=ytrue+noise

# Plot of function and noisy data
plt.figure(1)
plt.plot(xdata,ytrue,"--")
plt.plot(xdata,ydata,"o",markersize=6)
plt.xlabel("x")
plt.ylabel("y")

# Solution using least squares
J=np.concatenate(( np.reshape(np.ones(n),(n,1)),np.reshape(xdata,(n,1)) ),axis=1)
M=J.transpose()@J
yprime=J.transpose()@np.reshape(ydata,(n,1))
[Mmod,order]=Gauss_elim(M,yprime,False)
avec=backsub(Mmod[order,:],False)
yfit=avec[0]+avec[1]*xdata
plt.plot(xdata,yfit,'-')
plt.legend(("original function","noisy data","fitted function"))
plt.show()
