#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 07:02:29 2020

Tests the performance of Gaussian elimination for systems of a variety of sizes

@author: zettergm
"""

# imports
import numpy as np
import time
from elimtools import Gauss_elim,backsub
import matplotlib.pyplot as plt
from ittools import Jacobi


# define a range of system sizes to investigate
nvals=np.arange(50,550,50)         # note arange excluded the end of the range
testtimes=np.zeros(nvals.size)     # time taken for each system size
lrep=1                             # number of times to repeat each step (for consistency)


# Perform the solves for each system size
for ind, n in enumerate(list(nvals)):
    a0=-1*np.ones(n-1)
    a1=4*np.ones(n)
    a2=-1*np.ones(n-1)
    A=np.diag(a1)
    A=np.diag(a0,-1)+np.diag(a1,0)+np.diag(a2,1)
    b=np.random.randn(n,1)
    
    for irep in range(0,lrep):
        tstart=time.time()
        [Amod,order]=Gauss_elim(A,b,False)
        Amodsub=np.copy(Amod[order,:])
        x=backsub(Amodsub,False)
        tend=time.time()
        testtimes[ind]=testtimes[ind]+(tend-tstart)/lrep
    
    print("GE solution for system of size:  ",n," took time:  ", testtimes[ind])
    

# Plot the (average) time elapsed during each GE solve
plt.figure(1)
plt.plot(nvals,testtimes,'o')
plt.xlabel("system size (no. of unknowns)")
plt.ylabel("solution time (s)")
plt.title("Performance of self-coded solution")


# Solve same system sizes (diagonally dominant problem) with Jacobi iteration
tol=1e-9
testtimes=np.zeros(nvals.size)     # time taken for each system size

for ind, n in enumerate(list(nvals)):
    a0=-1*np.ones(n-1)
    a1=4*np.ones(n)
    a2=-1*np.ones(n-1)
    A=np.diag(a1)
    A=np.diag(a0,-1)+np.diag(a1,0)+np.diag(a2,1)
    b=np.random.randn(n,1)
    
    for irep in range(0,lrep):
        tstart=time.time()
        x0=np.random.randn(n,1)
        [x,iteration]=Jacobi(x0,A,b,tol,False)
        tend=time.time()
        testtimes[ind]=testtimes[ind]+(tend-tstart)/lrep
    
    print("JI solution for system of size:  ",n," took time:  ", testtimes[ind])


# Plot the time elapses for each Jacobi solve
plt.plot(nvals,testtimes,'^')
plt.legend(["Gaussian elim.","Jacobi it."])
plt.show()
