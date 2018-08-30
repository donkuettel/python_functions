# Numeric Integrators

"""This script provides a variety of methods to solve first order 
ordinary differential equations.

AUTHOR:
    Don Kuettel <don.kuettel@gmail.com>
    University of Colorado-Boulder

    1) Euler's Method (euler)
    2) Second-order Runge-Kutta Method (rk2)
    3) Fourth-order Runge-Kutta Method (rk4)
    4) 

    INPUT:
        f     - function of x and t equal to dx/dt. x may be 
                multivalued, in which case it should a list or 
                a NumPy array. In this case f must return a 
                NumPy array with the same dimension as x.

        x0    - the initial condition(s). Specifies the value 
                of x when t = t[0]. Can be either a scalar or a 
                list or NumPy array if a system of equations is 
                being solved.

        t     - list or NumPy array of t values to compute 
                solution at. t[0] is the initial condition 
                point, and the difference h=t[i+1]-t[i] 
                determines the step size h.

        constants - a dictionary of constants necessary for 
                    the derivative function

    OUTPUT:
        x     - NumPy array containing solution values 
                corresponding to each entry in t array. If a 
                system is being solved, x will be an array of 
                arrays.
"""

# Import Modules
import numpy as np

# Define Functions
# ==============================================================
def euler(f, x0, t, constants):
    """Euler's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = euler(f, x0, t, constants)
    """

    n = len(t)
    x = np.array([x0]*n)
    for i in range(n-1):
        x[i+1] = x[i] + (t[i+1] - t[i])*f(x[i], t[i], constants)

    return x
# ==============================================================


# ==============================================================
def rk2(f, x0, t, constants):
    """Second-order Runge-Kutta method to solve x' = f(x,t) 
       with x(t[0]) = x0.

        USAGE:
            x = rk2(f, x0, t, constants)

        NOTES:
            This version is based on the algorithm presented in 
            "Numerical Mathematics and Computing" 4th Edition, 
            by Cheney and Kincaid, Brooks-Cole, 1999.
    """

    n = len(t)
    x = np.array([x0]*n)
    for i in range(n-1):
        h = t[i+1] - t[i]
        k1 = h*f(x[i], t[i], constants)
        k2 = h*f(x[i] + k1, t[i+1], constants)
        x[i+1] = x[i] + (k1 + k2)/2.0

    return x
# ==============================================================


# ==============================================================
def rk4(f, x0, t, constants):
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) 
       with x(t[0]) = x0.

    USAGE:
        x = rk4(f, x0, t, constants)
    """

    n = len(t)
    x = np.array([x0]*n)
    for i in range(n-1):
        h = t[i+1] - t[i]
        k1 = h*f(x[i], t[i], constants)
        k2 = h*f(x[i] + 0.5*k1, t[i] + 0.5*h, constants)
        k3 = h*f(x[i] + 0.5*k2, t[i] + 0.5*h, constants)
        k4 = h*f(x[i] + k3, t[i+1], constants)
        x[i+1] = x[i] + (k1 + 2.0*(k2 + k3) + k4)/6.0

    return x
# ==============================================================


# ==============================================================
