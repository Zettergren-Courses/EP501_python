#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 07:23:32 2020

Module for functions used in nonlinear equation solvers

@author: zettergm
"""

import numpy as np

def fun1(x):
    y=np.cos(x)
    return y

def fun1_derivative(x):
    yprime=-1.0*np.sin(x)
    return yprime

def fun3(x):
    y=x**2+6*x+10
    return y

def fun3_deriv(x):
    yprime=2*x+6
    return yprime

def fun2D1(x,y):
    f=x**3+y**3-3*x*y
    return f

def fun2D2(x,y):
    g=x**2+y**2-1
    return g

def fun2D1_deriv(x,y):
    fx=3*x**2-3*y
    fy=3*y**2-3*x
    return [fx,fy]

def fun2D2_deriv(x,y):
    gx=2*x
    gy=2*y
    return [gx,gy]
