#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 07:43:26 2020

Module containing root finding functions based on Newtons' method

@author: zettergm
"""

#import numpy as np
import sys

def newton_exact(f,fprime,x0,maxit,tol,verbose):
    # Gaurd against starting at an inflection point
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
