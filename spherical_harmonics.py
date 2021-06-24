
#################################
##     Spherical Harmonics     ##
#################################

"""This script provides the BCBF acceleration equations for 
spherical harmonics

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

NOTES:
    -Do I need to normalize coefficients
"""

####################
## Import modules ##
####################
import numpy as np
from math import factorial


def sphericalHarmonics(r, mu, R, degree, order, gc):
    """
    INPUT:
        X - BCBF State
        mu- gravitational parameter of central body 
        R - radius of central body
        degree - degree of spherical harmonic gravity
        order - order of spherical harmonic gravity
        gc - normalized C_lm and S_lm gravity constants of central body

    OUTPUT:
    """

    # Accelerations
    ax = 0
    ay = 0
    az = 0

    C = gc['C_lm']
    S = gc['S_lm']

    # State
    x = r[0]
    y = r[1]
    z = r[2]
    rmag = np.sqrt(r.dot(r))

    for l in range(2, degree + 1):
        ll = np.copy(l)
        if ll > order:
            ll = order
        for m in range(ll + 1):

            # Normalizing Coefficient
            if m == 0:
                k = 1.0
            else:
                k = 2.0
            N_lm = (factorial(l+m)/factorial(l-m)/k/(2*l+1))**0.5

            # Coefficiens
            C_lm = C[l-2,m]/N_lm
            S_lm = S[l-2,m]/N_lm

            if l == 2 and m == 0:
                ax += C_lm*R**2*mu*x*(1.5*x**2 + 1.5*y**2 - 6.0*z**2)/rmag**7
                ay += C_lm*R**2*mu*y*(1.5*x**2 + 1.5*y**2 - 6.0*z**2)/rmag**7
                az += C_lm*R**2*mu*z*(4.5*x**2 + 4.5*y**2 - 3.0*z**2)/rmag**7

            elif l == 2 and m == 1:
                ax += R**2*mu*rmag**(-19.0)*z*(x**2 + y**2)**(-2.5)*(3.0*rmag**12.0*x*z**2*(x**2 + y**2)**1.5*(C_lm*x + S_lm*y) - 12.0*rmag**12.0*x*(x**2 + y**2)**2.5*(C_lm*x + S_lm*y) - 3.0*rmag**14.0*(x**2 + y**2)**1.5*(-C_lm*(x**2 + y**2) + x*(C_lm*x + S_lm*y)))
                ay += R**2*mu*rmag**(-19.0)*z*(x**2 + y**2)**(-2.5)*(3.0*rmag**12.0*y*z**2*(x**2 + y**2)**1.5*(C_lm*x + S_lm*y) - 12.0*rmag**12.0*y*(x**2 + y**2)**2.5*(C_lm*x + S_lm*y) - 3.0*rmag**14.0*(x**2 + y**2)**1.5*(-S_lm*(x**2 + y**2) + y*(C_lm*x + S_lm*y)))
                az += R**2*mu*rmag**(-21.0)*(x**2 + y**2)**(-1.0)*(C_lm*x + S_lm*y)*(-3.0*rmag**14.0*z**2*(x**2 + y**2) - 12.0*rmag**14.0*z**2*(x**2 + y**2)**1.0 + 3.0*rmag**16.0*(x**2 + y**2)**1.0)
        
            elif l == 2 and m == 2:
                ax += R**2*mu*rmag**(-19.0)*(x**2 + y**2)**(-6.0)*(6.0*rmag**12.0*x*z**2*(x**2 + y**2)**5.0*(C_lm*(x**2 - y**2) + 2*S_lm*x*y) - 6.0*rmag**14.0*y*(x**2 + y**2)**5*(-2*C_lm*x*y + 2*S_lm*x**2 - S_lm*(x**2 + y**2)) - 9.0*x*(C_lm*(x**2 - y**2) + 2*S_lm*x*y)*(x**4 + 2*x**2*y**2 + x**2*z**2 + y**4 + y**2*z**2)**6.0)
                ay += R**2*mu*rmag**(-19.0)*(x**2 + y**2)**(-3.0)*(6.0*rmag**12.0*y*z**2*(x**2 + y**2)**2.0*(C_lm*(x**2 - y**2) + 2*S_lm*x*y) - 9.0*rmag**12.0*y*(x**2 + y**2)**3.0*(C_lm*(x**2 - y**2) + 2*S_lm*x*y) - 6.0*rmag**14.0*x*(x**2 + y**2)**2*(-S_lm*(x**2 + y**2) + 2*y*(C_lm*x + S_lm*y)))
                az += -R**2*mu*rmag**(-16.0)*z*(9.0*rmag**9 + 6.0*rmag**9.0)*(C_lm*(x**2 - y**2) + 2*S_lm*x*y)
        
            elif l == 3 and m == 0:
                ax += C_lm*R**3*mu*x*z*(7.5*x**2 + 7.5*y**2 - 10.0*z**2)/rmag**9
                ay += C_lm*R**3*mu*y*z*(7.5*x**2 + 7.5*y**2 - 10.0*z**2)/rmag**9
                az += C_lm*R**3*mu*(0.5*rmag**2*(-3*x**2 - 3*y**2 + 2*z**2) + 5.0*z**2*(x**2 + y**2) + z**2*(7.5*x**2 + 7.5*y**2 - 5.0*z**2))/rmag**9

    return np.array([ax, ay, az])
