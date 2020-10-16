#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 07:50:40 2020

This file shows briefly how to load data from a MATLAB .mat file and organize it

@author: zettergm
"""

import scipy.io as spio


# presumes that we've cloned the EP501_assignments repo into ../../
filename="../../EP501_assignments/assignments/HW1/testproblem.mat"
datadictionary=spio.loadmat(filename)
A=datadictionary["A"]
b=datadictionary["b"]
b2=datadictionary["b2"]
b3=datadictionary["b3"]

filename="../../EP501_assignments/assignments/HW2/iterative_testproblem.mat"
datadictionary=spio.loadmat(filename)
Ait=datadictionary["Ait"]
bit=datadictionary["bit"]
