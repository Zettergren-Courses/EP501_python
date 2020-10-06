#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 07:55:23 2020

Script that performs Newton's method

@author: zettergm
"""

import numpy as np
from nonlinear_fns import fun1 as f
from nonlinear_fns import fun1_derivative as fprime
from newton_methods import newton_exact
import matplotlib.pyplot as plt


# Parameters for Newton iteration
maxit=100
tol=1e-9
verbose=True

# Function we are finding roots for
minx=0
maxx=2*np.pi
x=np.linspace(minx,maxx,64)
y=f(x)

# Plot the objective function
plt.figure(1)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function for which roots are to be found")

# Make the call to do the Newton iterations with different starting points
[root,it,converged]=newton_exact(f,fprime,4,maxit,tol,verbose)
print("Root value:  ",root," achieved after:  ",it," iterations.")

[root,it,converged]=newton_exact(f,fprime,0.5,maxit,tol,verbose)
print("Root value:  ",root," achieved after:  ",it," iterations.")
