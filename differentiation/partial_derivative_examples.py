#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 06:47:19 2020

Script to illustrate calculation of partial derivatives using finite differences

@author: zettergm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Grid and function to be differentiated
lx=20
ly=20
x=np.linspace(-5,5,lx)
y=np.linspace(-5,5,ly)
[X,Y]=np.meshgrid(x,y)
f=np.exp(-X**2/2/2)*np.exp(-Y**2/2/1)
# Possibly compute gradient and laplacian analytically to test...

# Coordinate differences
dx=x[1]-x[0]
dy=y[1]-y[0]


# Compute the gradient of f, centered diffs. with fwd/bwd at edges
gradx=np.zeros((ly,lx))       #note first index is treated as "y"
grady=np.zeros((ly,lx))
gradx[:,0]=(f[:,1]-f[:,0])/dx
for ix in range(1,lx-1):
    gradx[:,ix]=(f[:,ix+1]-f[:,ix-1])/2/dx
gradx[:,lx-1]=(f[:,lx-1]-f[:,lx-2])/dx

grady[0,:]=(f[1,:]-f[0,:])/dy
for iy in range(1,lx-1):
    grady[iy,:]=(f[iy+1,:]-f[iy-1,:])/2/dy
grady[lx-1,:]=(f[ly-1,:]-f[ly-2])/dy


# Compute Laplacian f by taking div(grad(f)), centered diffs + edges as above
divx=np.zeros((ly,lx))
divx[:,0]=(gradx[:,1]-gradx[:,0])/dx
for ix in range(0,lx-1):
    divx[:,ix]=(gradx[:,ix+1]-gradx[:,ix-1])/2/dx
divx[:,lx-1]=(gradx[:,lx-1]-gradx[:,lx-2])/dx

divy=np.zeros((ly,lx))
divy[0,:]=(grady[1,:]-grady[0,:])/dy
for iy in range(0,ly-1):
    divy[iy,:]=(grady[iy+1,:]-grady[iy-1,:])/2/dy
divy[ly-1,:]=(grady[ly-1,:]-grady[iy-2,:])/dy
laplacianf=divx+divy


# Take curl of the gradient (to check that is ~0); only concerned with z-comp
# dvy/dx - dvx/dy is z-comp of curl
curlx=np.zeros((ly,lx))
curlx[:,0]=(grady[:,1]-grady[:,0])/dx
for ix in range(0,lx-1):
    curlx[:,ix]=(grady[:,ix+1]-grady[:,ix-1])/2/dx
curlx[:,lx-1]=(grady[:,lx-1]-grady[:,lx-2])/dx

curly=np.zeros((ly,lx))
curly[0,:]=(gradx[1,:]-gradx[0,:])/dy
for iy in range(0,ly-1):
    curly[iy,:]=(gradx[iy+1,:]-gradx[iy-1,:])/2/dy
curly[ly-1,:]=(gradx[ly-1,:]-gradx[ly-2,:])/dy
curl=curlx-curly


# Plot the function and derivatives
plt.figure(1)
plt.pcolor(X,Y,f)
plt.xlabel("x")
plt.ylabel("y")
plt.title("f(x,y) and grad(f)")
plt.colorbar()
plt.quiver(X,Y,gradx,grady,color="white",scale=10)
plt.show()

plt.figure(2)
plt.pcolor(X,Y,laplacianf)
plt.xlabel("x")
plt.ylabel("y")
plt.title("laplacian(f)")
plt.colorbar()
plt.show()

plt.figure(3)
plt.pcolor(X[1:-2,1:-2],Y[1:-2,1:-2],curl[1:-2,1:-2])
plt.xlabel("x")
plt.ylabel("y")
plt.title("curl(grad(f))")
plt.colorbar()
plt.show()
