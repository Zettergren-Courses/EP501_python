#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:30:46 2020

Illustrate root finding for a 2D problem

@author: zettergm
"""

# Import function we are optimizing over
from nonlinear_fns import fun2D1 as f
from nonlinear_fns import fun2D2 as g
from nonlinear_fns import fun2D1_deriv as gradf
from nonlinear_fns import fun2D2_deriv as gradg
import numpy as np
from newton_methods import newton2D_exact
import matplotlib.pyplot as plt

# For plotting the functions being solved
x=np.linspace(-1.5,1.5,64)
y=np.linspace(-1.5,1.5,64)
[X,Y]=np.meshgrid(x,y)
F=f(X,Y)
G=g(X,Y)

# Newton's method call
x0=0.5
y0=0.1
[x,y,it,converged]=newton2D_exact(f,gradf,g,gradg,x0,y0,100,1e-6,True)

# Plot the result
fig=plt.figure()
ax=fig.gca(projection="3d")
plt.plot([x,x],[y,y],[0,0],'o',markersize=20)
ax.plot_surface(X,Y,F)
ax.plot_surface(X,Y,G)
plt.xlabel("x")
plt.ylabel("y")
plt.show()