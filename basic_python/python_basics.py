#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 06:45:58 2020

A script to illustrate some basic plotting and matrix operations in Python

todo:
1) need to add a section on printing matrices in a nice formatted way...

@author: zettergm
"""

# Load external modules
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.special as sspec
import matplotlib.pyplot as plt

from funexamples import fun1D,fun2D


# Define some variable so we can plot things
x=np.linspace(-10,10,100)
y=np.linspace(-10,10,100)
[X,Y]=np.meshgrid(x,y)


# Example function evaluations to plot
y1=fun1D(x,1,1,3)
y2=fun1D(x,1.5,-1,2)
Z1=fun2D(X,Y,2,0,0,3,10)
Z2=fun2D(X,Y,5,2,-2,2,2)


# Basic 1D Python plotting with matplotlib
plt.figure(1)
plt.plot(x,y1,x,y2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of two Gaussian functions")
plt.legend(["curve 1","curve 2"])
plt.show()


# Basic 2D Python plotting with matplotlib
plt.figure(2)
im1=plt.pcolor(X,Y,Z1)
plt.colorbar(im1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian 1")
plt.show()

fig3=plt.figure(3)
im2=plt.pcolor(X,Y,Z2)
plt.colorbar(im2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian 2")
plt.show()


# this matrix and vector represent the system of equations:
# 1)   x +  2y +  4z = 18
# 2)  2x + 12y -  2z =  9
# 3)  5x + 26y +  5z = 14
A=np.array([[1, 2, 4],[2, 12, -2],[5,26,5]])      #matrix of coefficients
b=np.array([[18], [9], [14]])                     #this is the column vector representing the RHS of the system
B=np.transpose(A)                                 #transpose of A, for illustration of differences from MATLAB


# Some basic commands to printlay variables
print("A = ")
print(A)

print("b = ")
print(b)

# Compute some matrix properties and print them
print("det(A) = ")
print(nla.det(A))     # numpy.linalg.det(A) longhand

print("trace(A) = ")
print(np.trace(A))

print("condition # of A = ")
print(nla.cond(A))

print("transpose(A) = ")
print(np.transpose(A))

print("Aii (diagonal elements of A) = ")
print(np.diag(A))

# Matrix functions
print("A*B (matrix multiplication) = ")
print(A@B)

print("Aij*Bij (scalar multiplication) = ")
print(A*B)

print("5*A (multiplication) = ")
print(5*A)

print("Solution (x) of the system A*x=b:  ")
print("x = ")
xvec=nla.solve(A,b)
print(xvec)

print("I (3x3 identity matrix) = ")
print(np.eye(3,3))

print("A^{-1} (inverse of A) = ")
Ainv=nla.solve(A,np.eye(3,3))
print(Ainv)

# slicing and concatenation of array/matrices
print("2nd row of A = ")
print(A[1,:])     #in Python arrays start at zero...

print("2nd column of A = ")
print(A[:,1])

print("flat list (column vector) of matrix elements of A:  ")
print(A.flatten())

print("Horizontal concatenation of A with itself:  ")
print(np.concatenate((A,A),axis=1))

print("Vertical concatenation of A with itself:  ")
print(np.concatenate((A,A),axis=0))

print("diagonal matrix:  ")
a1=np.arange(10,0,-1)                        # count backward, note endpoint excluded
print(np.diag(a1))

print("LU decomposition of A (A=L*U):  ")
[P,L,U]=sla.lu(A)
print("L = ")
print(L)
print("U = ")
print(U)

print("Eigenvalues of A = ")
[lam,psi]=sla.eig(A)      #note output args swapped from matlab
print(lam)
print("Eigenvectors of A = ")
print(psi[:,0])
print(psi[:,1])
print(psi[:,2])


# Special (Bessel) functions
rho=np.linspace(0,10,num=50)
plt.figure(4)
plt.plot(rho,sspec.jv(0,rho),rho,sspec.jv(1,rho),rho,sspec.jv(2,rho),rho,sspec.jv(3,rho))
plt.xlabel("x")
plt.ylabel("J")
plt.legend(["order 0","1","2","3"])
plt.title("Bessel functions of different order")
plt.show()
