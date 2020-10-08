#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 07:43:26 2020

Module containing root finding functions based on Newtons' method

@author: zettergm
"""

# import modules needed
import numpy as np
import sys
sys.path.append("../linear_algebra")
from elimtools import Gauss_elim,backsub


# 1D Newton solver
def newton_exact(f,fprime,x0,maxit,tol,verbose):
    # Guard against starting at an inflection point
    if (abs(fprime(x0))<tol):
        print("!Warning:  starting near inflection point, please change initial guess!")
        sys.exit()
    
    # Newton iteration main loop
    it=1
    root=x0
    fval=f(root)
    converged=False
    while (not converged and it<=maxit):
        derivative=fprime(root)
        if (abs(derivative)<100*tol):
            print("!Warning:  derivative near zero, terminating iterations with failure to converge (try a different starting point)!")
            return [root,it,converged]
        else:
            root=root-fval/derivative
            fval=f(root)
            if (verbose):
                print("Iteration ",it,"; root ",root,"; fval ",fval,"; derivative ",derivative)
            it=it+1
            converged=abs(fval)<tol
    it=it-1
    return [root,it,converged]


# 2D Newton solver
def newton2D_exact(f,gradf,g,gradg,x0,y0,maxit,tol,verbose):
    # Guard against starting near and inflection point
    [gradfx,gradfy]=gradf(x0,y0)
    [gradgx,gradgy]=gradg(x0,y0)
    if (abs(min([gradfx,gradfy,gradgx,gradgy]))<tol):
        print(" Attempting to start Newton iterations near inflection point; consider restarting with another initial guess")
        x0=x0+1
        y0=y0+1
    
    # 2D Newton iterations
    it=1
    rootx=x0
    rooty=y0
    fval=f(rootx,rooty)
    gval=g(rootx,rooty)
    converged=False
    while (not converged and it<=maxit):
        [gradfx,gradfy]=gradf(rootx,rooty)
        [gradgx,gradgy]=gradg(rootx,rooty)
        A=np.array([[gradfx,gradfy],[gradgx,gradgy]])
        fvec=np.array([[fval],[gval]])
        [Amod,order]=Gauss_elim(A,-1.0*fvec,False)
        dxvec=backsub(Amod[order,:],False)
        Areord=Amod[order,0:2]
        detA=Areord[0,0]*Areord[1,1]
        if (abs(detA)<1e-6):
            print(" Ended up at a point where the Jacobian is singular, try a different starting point")
            sys.exit()
        
        rootx=rootx+dxvec[0]
        rooty=rooty+dxvec[1]
        fval=f(rootx,rooty)
        gval=g(rootx,rooty)
        if (verbose):
            print("iteration:  ",it,"x,y= ",rootx,rooty,"f,g= ",fval,gval)
            #print("det(J)= ",detA)
            #print("J= ",A)
        it=it+1
        converged=abs(fval)<tol and abs(gval)<tol
        
    it=it-1;
    if (not converged):
        print(" Used max number of iterations")
    return [rootx,rooty,it,converged]       
        
        
        
        