#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 07:55:23 2020

Script that performs Newton's method

@author: zettergm
"""

import numpy as np
from nonlinear_fns import fun1 as f
import newton_methods
import matplotlib.pyplot as plt


# Parameters for Newton iteration
maxit=100
tol=1e-9

# Function we are finding roots for
minx=0
maxx=2*np.pi
x=np.linspace(minx,maxx,64)
y=f(x)

