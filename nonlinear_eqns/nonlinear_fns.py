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

