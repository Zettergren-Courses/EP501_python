# EP501_python

Python scripts for EP501.  This repository is very much a work-in-progress.  I will be updating it throughout the semester as I port more of my codes from the MATLAB repository for EP501 over to python.  Many of the MATLAB codes will eventually be ported, but I can't gaurantee that they all will by the end of the semester.  Either way you are free to use python to complete your assignments.  


## Codes updated for FA2020 semester

### Basic python functionality

1.  Located in ./python_basics
2.  Contains scripts showing how to execute basic calculations and plotting in Python (basic\_python.py).
3.  Contains a script showing how to use python to load data from MATLAB .mat files (load\_matlab\_file.py).  

### Numerical linear algebra

1.  Located in ./linear_algebra/
2.  Illustrates and checks various methods for solving matrix problems
3.  Contains python modules for elimination (elimtools.py) and iterative solutions (ittools.py) to linear systems of equations.
4.  Contains example scripts showing use of simple elimination (simple\_elim\_example.py) and Gaussian elimination (Gauss\_elim\_example.py).  
5.  Contains examples of using iterative solutions based on Jacobi iteration (Jacobi\_example.py).  

### Nonlinear equations

1.  Located in ./nonlinear_eqns
2.  Illustrates solutions to various nonlinear equations and systems
3.  Contains examples of interval halving (interval\_halving.py), false position (false\_position.py), Newton's method in 1D (Newton\_Rhapson.py), and Newton's method in 2D (Newton\_Rhapson2D.py)
4.  Contains various functions for exact Newton's method in 1D and 2D in the module (newton\_methods.py)
5.  Contains a module with objective functions that can be used as examples to demonstrate root finding algorithms (nonlinear\_fns.py).

### Polynomials and data fitting

1. Located in ./polynomials
2. Illustrates how to fit various types of polynomials to data
3. Contains examples of direct polynomial fitting (direc_fit.py), linear least squares fitting (least_squares.py), and bilinear/spline interpolation (interpolation_examples.py)

