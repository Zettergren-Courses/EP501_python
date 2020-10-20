#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 07:17:06 2020

Show how to do simple interpolation using Python

@author: zettergm
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../linear_algebra")
from elimtools import Gauss_elim,backsub

# Manual implmementatino of bilinear interpolation, single point
x=np.array([1,2]);
y=np.array([2,3]);
f=[10, 11; 12, 13];
[X,Y]=meshgrid(x,y);
x1=1.5;
y1=2.5;