
##############################
##  Optimization Functions  ##
##############################

"""This script provides a variety of useful optimization functions.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

    1) Brent Search algorithm
    2) 

"""

# Import Modules
import numpy as np

# Define Functions
# =====================================================================
# 1) 
# ---------------------------------------------------------------
def brentSearch(errfun, x0, x0_bracket, args, tol=1e-6, ITMAX=100):
    """This function finds the minimum value of a 1-dimensional 
    error function without derivatives by varying the independent 
    variable x0. 

    INPUT:
        errfun - errfun being minimized
        x0 - initial guess of independent variable
        x0_bracket - bounding bracket of x0
        args - arguments necessary for the errfun
        tol - convergence tolerance 
        ITMAX - maximum number of iterations

    OUTPUT:
        xmin  - value of independent variable that corresponds to 
                minimum errfun value
        fmin  - minimum errfun value
        count - number of iterations for convergence
    
    """

    CGOLD = 0.3819660; TINY = 1e-20
    e = 0.0; d = 0.0;

    # Function inputs: golden(ax, cx, func, tol)
    bx = x0; a = x0_bracket[0]; b = x0_bracket[-1]

    # Initializations
    x = w = v = bx
    fw = fv = fx = errfun(x, *args)

    # Main loop
    count = 1; i = 0
    while i < ITMAX-1:
        xm = 0.5*(a + b)
        tol1 = tol*abs(x) + TINY
        tol2 = 2.0*tol1

        if abs(x - xm) <= (tol2 - 0.5*(b - a)):
            break

        # Trial parabolic fit
        if abs(e) > tol1:
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            p = (x - v)*q - (x - w)*r
            q = 2*(q - r)

            if q > 0.0:
                p = -p

            q = abs(q)
            etemp = e
            e = d

            if abs(p) >= abs(0.5*q*etemp) or p <= q*(a - x) or p >= q*(b - x):
                if x >= xm:
                    e = a - x
                else:
                    e = b - x
                
                d = CGOLD*e

            else:
                d = p/q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = SIGN(tol1, xm - x)
        else:
            if x >= xm:
                e = a - x
            else:
                e = b - x
            d = CGOLD

        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + SIGN(tol1, d)
        fu = errfun(u, *args) # This is the one function eval per iteration

        # House keeping
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x 

            v, w, x = shft3(v, w, x, u)
            fv, fw, fx = shft3(fv, fw, fx, fu)
        
        else:
            if u < x:
                a = u
            else:
                b = u

            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

        count += 1; i += 1
        # print(i)

    fmin = fx; xmin = x
    return xmin, fmin, count
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Subfunctions

def shft3(a, b, c, d):
    """Auxilary Brent Search function"""
    a = b
    b = c
    c = d
    return a, b, c

def SIGN(a, b):
    """Auxilary Brent Search function"""
    return np.sign(b)*abs(a)
# ---------------------------------------------------------------
# =====================================================================

# =====================================================================
# #)

# =====================================================================


