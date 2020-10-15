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

# Data for a direct polynomial fit
#x=np.array([[1],[2],[3],[4]])
x=np.array([1,2,3,4])
y=2*x**3-3*x**2+4*x+9

# Create a plot to show results
plt.figure(1)
plt.plot(x,y,'*')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Illustrating direct fit methods")

# Fit using Python functions
n=3     # degree of polynomial to be fitted
coeffs=np.polyfit(x,y,n)
xlarge=np.linspace(1,4,64)
polyfun=np.poly1d(coeffs)
plt.plot(xlarge,polyfun(xlarge),'--')
plt.show()

# Execute direct fit using self-coded Gaussian Elimination
