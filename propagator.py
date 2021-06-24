
###################################
##     Spacecraft Propagation    ##
###################################

"""This script is an integration tool used to propagate a spacecraft 
forward or backward in time in the inertial frame of a specified 
rotating body that is frozen in its orbit about the Sun.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA
"""

# Import Modules
import constants as c
from bodies import Body
import astro_functions as af
from legendre_poly import legendre_poly
import integrators as i
from coord_trans import CoordTrans
import numpy as np
from math import factorial

#######################################################################
# Main Function Call

def Astrogator(IC, tspan, integrator=i.rk4, body='earth', extras=0, 
    control=False, grav=False, srp=False, thirdbod=False, coords='RV'):
    """This function propagtes a spacecraft forward of backwards in
    time in the inertial frame of a specified rotating body. The 
    central body is frozen in its orbit about the Sun, and the Sun
    is always located in the -x-direction of the body's inertial
    coordinate system.

    ASSUMPTIONS:
        - The sun is always located along the -x-axis of the central 
          body's inertial coordinate system
        - A cannonball srp model is used
        - body-inertial and heliocentric-inertial coordinate frames
          are aligned
        - The control vector is always represented in the the 
          inertial frame.

    INPUT:
        IC - 6x1 intital state of the spacecraft [x, y, z, vx, vy, vz]
             in [km] and [km/s].
        tspan - Nx1 vector of simulation time in [sec].
        integrator - the integration method (default is an rk4).
        body - central body of the simulation (default is Earth).
        extras - extra constants needed for integration. Detailed
                 in the supporting functions.
        control - flag to turn on control.
        grav - flag to turn on the asymetrical gravity field.
        srp - flag to turn on solar radiation pressure.
        thirdbod - flag to turn on third-body pertubations from the
                   Sun.
        coords - flag to indicate which coordinates to integrate
               -'RV'  = cartesian
               -'OE'  = classical orbital elements
               -'EQU' = modified equinoctial elements

    OUTPUT:
        X - When control is off, 6xN state matrix of a 6x1 state 
            vector for each time step. The state is [x, y, z, vx, 
            vy, vz] in [km] and [km/s]
          - When control is on, 7xN state matrix of a 7x1 state 
            vector for each time step. The state is [x, y, z, vx, 
            vy, vz, m] in [km], [km/s], and [kg]
    """

    # Determining Body
    body_cons = Body(body, grav, extras)
    
    # Determining Control
    int_constants = {}
    if control == False:
        int_constants['u'] = np.zeros(3)

        if grav == False and srp == False and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == False and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']

            # Need to be imported
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_grav_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_grav_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_grav_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == False and srp == True and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['srp_flux'] = c.srp_flux

            # Need to be imported
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_srp_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_srp_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_srp_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == False and srp == False and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']

            # Need to be imported
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['d_3B'] = extras['d_3B']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == True and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']
            int_constants['srp_flux'] = c.srp_flux

            # Need to be imported
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_grav_srp_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_grav_srp_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_grav_srp_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == False and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']

            # Need to be imported
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['d_3B'] = extras['d_3B']
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_grav_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_grav_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_grav_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == False and srp == True and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['R'] = body_cons['R']
            int_constants['srp_flux'] = c.srp_flux

            # Need to be imported
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_srp_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_srp_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_srp_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == True and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']
            int_constants['R'] = body_cons['R']
            int_constants['srp_flux'] = c.srp_flux

            # Need to be imported
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_grav_srp_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_grav_srp_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_grav_srp_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')

    else:
        int_constants['u'] = extras['u']

        if grav == False and srp == False and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['isp'] = extras['isp']
            
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == False and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
            int_constants['isp'] = extras['isp']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_grav_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_grav_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_grav_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == False and srp == True and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['srp_flux'] = c.srp_flux
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
            int_constants['isp'] = extras['isp']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_srp_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_srp_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_srp_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == False and srp == False and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['d_3B'] = extras['d_3B']
            int_constants['isp'] = extras['isp']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == True and thirdbod == False:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']
            int_constants['srp_flux'] = c.srp_flux
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
            int_constants['isp'] = extras['isp']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_grav_srp_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_grav_srp_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_grav_srp_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == False and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['R'] = body_cons['R']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['d_3B'] = extras['d_3B']
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
            int_constants['isp'] = extras['isp']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_grav_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_grav_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_grav_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == False and srp == True and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['R'] = body_cons['R']
            int_constants['srp_flux'] = c.srp_flux
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
            int_constants['isp'] = extras['isp']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_srp_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_srp_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_srp_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')
    
        elif grav == True and srp == True and thirdbod == True:
            # Necessary Constants
            int_constants['mu'] = body_cons['mu']
            int_constants['w_body'] = body_cons['w_body']
            int_constants['gc'] = body_cons['gc']
            int_constants['R'] = body_cons['R']
            int_constants['srp_flux'] = c.srp_flux
            int_constants['g'] = c.g0

            # Need to be imported
            int_constants['mu_3B'] = extras['mu_3B']
            int_constants['d_3B'] = extras['d_3B']
            int_constants['R_3B'] = extras['R_3B']
            int_constants['Cr'] = extras['Cr']
            int_constants['a2m'] = extras['a2m']
            int_constants['theta_gst'] = extras['theta_gst']
            int_constants['degree'] = extras['degree']
            int_constants['order'] = extras['order']
            int_constants['isp'] = extras['isp']
    
            # Dynamics Function
            if coords.lower() == 'rv':
                dynamics = twobod_control_grav_srp_thirdbod_rv
    
            elif coords.lower() == 'oe':
                dynamics = twobod_control_grav_srp_thirdbod_oe
    
            elif coords.lower() == 'equ':
                dynamics = twobod_control_grav_srp_thirdbod_equ
    
            else:
                print('ERROR: Coordinates not yet supported!')

    # Here is the actual integration,
    X = integrator(dynamics, IC, tspan, int_constants)

    return X

#######################################################################


#######################################################################
# Supporting Functions
""" 
    No Control:
        Cartesian Elements
         1) twobod_rv
         2) twobod_grav_rv
         3) twobod_srp_rv
         4) twobod_thirdbod_rv
         5) twobod_grav_srp_rv
         6) twobod_grav_thirdbod_rv
         7) twobod_srp_thirdbod_rv
         8) twobod_grav_srp_thirdbod_rv

        Classical Orbital elements
         9) twobod_oe
        10) twobod_grav_oe
        11) twobod_srp_oe
        12) twobod_thirdbod_oe
        13) twobod_grav_srp_oe
        14) twobod_grav_thirdbod_oe
        15) twobod_srp_thirdbod_oe
        16) twobod_grav_srp_thirdbod_oe

        Modified Equanoctial Elements
        17) twobod_equ
        18) twobod_grav_equ
        19) twobod_srp_equ
        20) twobod_thirdbod_equ
        21) twobod_grav_srp_equ
        22) twobod_grav_thirdbod_equ
        23) twobod_srp_thirdbod_equ
        24) twobod_grav_srp_thirdbod_equ

    Control:
        Cartesian Elements
        25) twobod_control_rv
        26) twobod_control_grav_rv
        27) twobod_control_srp_rv
        28) twobod_control_thirdbod_rv
        29) twobod_control_grav_srp_rv
        30) twobod_control_grav_thirdbod_rv
        31) twobod_control_srp_thirdbod_rv
        32) twobod_control_grav_srp_thirdbod_rv

        Classical Orbital elements
        33) twobod_control_oe
        34) twobod_control_grav_oe
        35) twobod_control_srp_oe
        36) twobod_control_thirdbod_oe
        37) twobod_control_grav_srp_oe
        38) twobod_control_grav_thirdbod_oe
        39) twobod_control_srp_thirdbod_oe
        40) twobod_control_grav_srp_thirdbod_oe

        Modified Equanoctial Elements
        41) twobod_control_equ
        42) twobod_control_grav_equ
        43) twobod_control_srp_equ
        44) twobod_control_thirdbod_equ
        45) twobod_control_grav_srp_equ
        46) twobod_control_grav_thirdbod_equ
        47) twobod_control_srp_thirdbod_equ
        48) twobod_control_grav_srp_thirdbod_equ
"""

# Function Definitions
# ===================================================================
# 1)
def twobod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of the central
                              body in [km^3/s^2]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Derivatives
    accel = a_2B + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 2)
def twobod_grav_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field 
    perturbations.

    ASSUMPTIONS:

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_grav + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 3)
def twobod_srp_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for srp perturbations. 

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # srp_mag = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0/r_sun2sc_mag/r_sun2sc_mag
    # G1 = srp_flux/c.c*c.AU*c.AU/1000.0
    # print(G1, srp_mag)

    # Derivatives
    accel = a_2B + a_srp + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 4)
def twobod_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for the third-body effects 
    from the Sun. 

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Derivatives
    accel = a_2B + a_3B + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 5)
def twobod_grav_srp_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field and srp 
    perturbations.

    ASSUMPTIONS:
        

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
    index = 0
    for l in range(2, degree + 1):
        ll = np.copy(l)
        if ll > order:
            ll = order
        for m in range(ll + 1):

            # Normalizing Coefficient
            if m == 0:
                k = 1
            else:
                k = 2
            N_lm = (factorial(l+m)/factorial(l-m)/k/(2*l+1))**0.5
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_srp + a_grav + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 6)
def twobod_grav_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field and solar 
    gravity perturbations.

    ASSUMPTIONS:
        

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
    index = 0
    for l in range(2, degree + 1):
        ll = np.copy(l)
        if ll > order:
            ll = order
        for m in range(ll + 1):

            # Normalizing Coefficient
            if m == 0:
                k = 1
            else:
                k = 2
            N_lm = (factorial(l+m)/factorial(l-m)/k/(2*l+1))**0.5
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_3B + a_grav + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 7)
def twobod_srp_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for srp and solar gravity 
    perturbations. 

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Derivatives
    accel = a_2B + a_3B + a_srp + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 8)
def twobod_grav_srp_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field, srp, and 
    solar gravity perturbations. 

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz] in [km] and [km/s]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az] in 
             [km/s] and [km/s2]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    # srp from sun
    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
    index = 0
    for l in range(2, degree + 1):
        ll = np.copy(l)
        if ll > order:
            ll = order
        for m in range(ll + 1):

            # Normalizing Coefficient
            if m == 0:
                k = 1
            else:
                k = 2
            N_lm = (factorial(l+m)/factorial(l-m)/k/(2*l+1))**0.5
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_grav + a_3B + a_srp + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]

    return dX
# ===================================================================


# ===================================================================
# 9)
def twobod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion and a control input.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    u - control law acceleration in [km/s^2]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Perturbations
    accel = u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h

    return dX
# ===================================================================


# ===================================================================
# 10)
def twobod_grav_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the central gravity field, and
     a control input.

    ASSUMPTIONS:
        -central body is a sphere
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Perturbations
    accel = a_grav + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h

    return dX
# ===================================================================


# ===================================================================
# 11)
def twobod_srp_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h

    return dX
# ===================================================================


# ===================================================================
# 12)
def twobod_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the solar gravity tide,
     and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    
    return dX
# ===================================================================


# ===================================================================
# 13)
def twobod_grav_srp_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    srp, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_grav + a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h

    return dX
# ===================================================================


# ===================================================================
# 14)
def twobod_grav_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    solar gravity, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h

    return dX
# ===================================================================


# ===================================================================
# 15)
def twobod_srp_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, solar gravity and a 
    control input.

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h

    return dX
# ===================================================================


# ===================================================================
# 16)
def twobod_grav_srp_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field,
    srp, solar gravity, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [a, e, i, rann, w, nu] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h

    return dX
# ===================================================================


# ===================================================================
# 17)
def twobod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion and a control input.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    u - control law acceleration in [km/s^2]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [p, f, g, h, k, L] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Perturbations
    oe = af.equ2oe(X)
    accel = u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================


# ===================================================================
# 18)
def twobod_grav_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the central gravity field, and
     a control input.

    ASSUMPTIONS:
        -central body is a sphere
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [p, f, g, h, k, L] in 
             [km/s] and [rad/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Equinoctial Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Perturbations
    accel = a_grav + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================


# ===================================================================
# 19)
def twobod_srp_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = X = [p, f, g, h, k, L] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Cartesian Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================


# ===================================================================
# 20)
def twobod_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the solar gravity tide,
     and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [p, f, g, h, k, L] in 
             [km/s] and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1 + h*h + k*k
    w = 1 + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Cartesian Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================


# ===================================================================
# 21)
def twobod_grav_srp_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    srp, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [p, f, g, h, k, L] in [km/s] 
             and [rad/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1 + h*h + k*k
    w = 1 + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_grav + a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================


# ===================================================================
# 22)
def twobod_grav_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    solar gravity, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [p, f, g, h, k, L] in [km/s] 
             and [rad/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1 + h*h + k*k
    w = 1 + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================


# ===================================================================
# 23)
def twobod_srp_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, solar gravity and a 
    control input.

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [p, f, g, h, k, L] in [km/s] 
             and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1 + h*h + k*k
    w = 1 + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Cartesian Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================


# ===================================================================
# 24)
def twobod_grav_srp_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field,
    srp, solar gravity, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L] in [km] and [rad]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [p, f, g, h, k, L] in [km/s] 
             and [rad/s]
    """

    dX = np.zeros(6)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]

    s2 = 1 + h*h + k*k
    w = 1 + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

     # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW

    return dX
# ===================================================================

# ===================================================================
# 25)
def twobod_control_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of the central
                              body in [km^3/s^2]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Derivatives
    accel = a_2B + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 26)
def twobod_control_grav_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field 
    perturbations.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm)
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_grav + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 27)
def twobod_control_srp_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for srp perturbations. 

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Derivatives
    accel = a_2B + a_srp + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 28)
def twobod_control_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for the third-body effects 
    from the Sun. 

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Derivatives
    accel = a_2B + a_3B + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 29)
def twobod_control_grav_srp_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field and srp 
    perturbations.

    ASSUMPTIONS:   

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
    index = 0
    for l in range(2, degree + 1):
        ll = np.copy(l)
        if ll > order:
            ll = order
        for m in range(ll + 1):

            # Normalizing Coefficient
            if m == 0:
                k = 1
            else:
                k = 2
            N_lm = (factorial(l+m)/factorial(l-m)/k/(2*l+1))**0.5
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_srp + a_grav + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 30)
def twobod_control_grav_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field and solar 
    gravity perturbations.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
    index = 0
    for l in range(2, degree + 1):
        ll = np.copy(l)
        if ll > order:
            ll = order
        for m in range(ll + 1):

            # Normalizing Coefficient
            if m == 0:
                k = 1
            else:
                k = 2
            N_lm = (factorial(l+m)/factorial(l-m)/k/(2*l+1))**0.5
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_3B + a_grav + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 31)
def twobod_control_srp_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for srp and solar gravity 
    perturbations. 

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Derivatives
    accel = a_2B + a_3B + a_srp + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 32)
def twobod_control_grav_srp_thirdbod_rv(X, t, constants):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state accounting for gravity field, srp, and 
    solar gravity perturbations. 

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [km], [km/s], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [vx, vy, vz, ax, ay, az, mdot]
             in [km/s], [km/s2], and [kg]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    m = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    # srp from sun
    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
    index = 0
    for l in range(2, degree + 1):
        ll = np.copy(l)
        if ll > order:
            ll = order
        for m in range(ll + 1):

            # Normalizing Coefficient
            if m == 0:
                k = 1
            else:
                k = 2
            N_lm = (factorial(l+m)/factorial(l-m)/k/(2*l+1))**0.5
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (
        dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (
        dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_grav + a_3B + a_srp + u
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 33)
def twobod_control_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion and a control input.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    u - control law acceleration in [km/s^2]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Perturbations
    accel = u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 34)
def twobod_control_grav_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the central gravity field, and
     a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Perturbations
    accel = a_grav + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 35)
def twobod_control_srp_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, and a control input.

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 36)
def twobod_control_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the solar gravity tide,
     and a control input.

    ASSUMPTIONS:
        -bodies are point masses
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot
    
    return dX
# ===================================================================


# ===================================================================
# 37)
def twobod_control_grav_srp_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    srp, and a control input.

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm)
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_grav + a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 38)
def twobod_control_grav_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    solar gravity, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm)
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Third-Body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 39)
def twobod_control_srp_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, solar gravity and a 
    control input.

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 40)
def twobod_control_grav_srp_thirdbod_oe(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field,
    srp, solar gravity, and a control input.

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [a, e, i, raan, w, nu, m] in [km], [rad], and 
            [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital elements
             derivatives where dX = [a, e, i, rann, w, nu, m] in 
             [km/s], [rad/s], and [kg]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    a =     X[0]        # semi-major axis [km]
    e =     X[1]        # eccentricity [-]
    inc =   X[2]        # inclination [rad]
    raan =  X[3]        # right ascension of the ascending node [rad]
    w =     X[4]        # argument of periapsis [rad]
    nu =    X[5]        # true anomaly [rad]
    m =     X[6]        # SC Mass [kg]

    p = a*(1 - e*e)
    h = np.sqrt(mu_host*p)
    r_mag = p/(1 + e*np.cos(nu))

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    r, v = af.kep2cart(X, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=X)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = 2*a*a/h*(e*np.sin(nu)*FR + p/r_mag*FS)
    dX[1] = (p*np.sin(nu)*FR + ((p + r_mag)*np.cos(nu) + r_mag*e)*FS)/h
    dX[2] = r_mag*np.cos(w + nu)/h*FW
    dX[3] = r_mag*np.sin(w + nu)/h/np.sin(inc)*FW
    dX[4] = (-p*np.cos(nu)*FR + (p + r_mag)*np.sin(nu)*FS)/h/e - r_mag*np.sin(w + nu)*np.cos(inc)/h/np.sin(inc)*FW
    dX[5] = h/r_mag/r_mag + (p*np.cos(nu)*FR - (p + r_mag)*np.sin(nu)*FS)/e/h
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 41)
def twobod_control_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion and a control input.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    u - control law acceleration in [km/s^2]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Perturbations
    oe = af.equ2oe(X)
    accel = u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 42)
def twobod_control_grav_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the central gravity field, and
     a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm)
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Perturbations
    accel = a_grav + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 43)
def twobod_control_srp_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, and a control input.

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 44)
def twobod_control_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, the solar gravity tide,
     and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 45)
def twobod_control_grav_srp_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    srp, and a control input.

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Perturbations
    accel = a_grav + a_srp + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 46)
def twobod_control_grav_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field, 
    solar gravity, and a control input.

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    degree = constants['degree']
    order = constants['order']
    theta_gst = constants['theta_gst']
    w_body = constants['w_body']
    R = constants['R']
    gc = constants['gc']
    C = gc['C_lm']
    S = gc['S_lm']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 47)
def twobod_control_srp_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, srp, solar gravity and a 
    control input.

    ASSUMPTIONS:
        -Bodies are point masses
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R - radius of central body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']
    mu_3B = constants['mu_3B']
    d_3B = constants['d_3B']
    R_3B = constants['R_3B']
    R = constants['R']
    Cr = constants['Cr']
    srp_flux = constants['srp_flux']
    a2m = constants['a2m']
    isp = constants['isp']
    g = constants['g']

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Cartesian Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    srp_const = l*Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================


# ===================================================================
# 48)
def twobod_contol_grav_srp_thirdbod_equ(X, t, constants):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion, central body gravity field,
    srp, solar gravity, and a control input.

    ASSUMPTIONS:
        -The sun is always located along the -x-axis of the central 
         body's inertial coordinate system
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p, f, g, h, k, L, m] in [km], [rad], and [kg]
        t - time of integration in seconds
        constants - a dictionary of constants necessary for 
                    integration
                    u - control law acceleration in [km/s^2]
                    mu_host - gravitational parameter of central
                              body [km^3/s^2]
                    R - radius of central body in [km]
                    w_body - rotation rate of the central body in
                             [rad/sec]
                    degree - degree of spherical harmonic gravity
                    order - order of spherical harmonic gravity
                    theta_gst - initial angle between body-inertial
                                and body fixed frame
                    gc - normalized C_lm and S_lm gravity constants
                         of host body
                    mu_3B - gravitational parameter of third
                            body in [km^3/s^2]
                    d_3B - distance of third body from central 
                           body in [km]
                    R_3B - radius of thrid body in [km]
                    Cr - coefficient of reflectivity of SC
                    srp_flux - value of the Sun's flux at 1 AU
                               in [W/m2]
                    a2m - area to mass ratio of the satellite
                          in [m2/kg]
                    isp - the specific impulse of the SC in [s-1]
                    g - standard gravity in [m/s2]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft orbital element 
        derivatives where dX = [p, f, g, h, k, L, m] in [km/s], 
        [rad/s], and [kg/s]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm)
    """

    dX = np.zeros(7)

    # Control law
    u = constants['u']

    # Assigning Constants
    mu_host = constants['mu']               # m3/s2
    mu_3B = constants['mu_3B']              # m3/s2
    d_3B = constants['d_3B']                # m
    R_3B = constants['R_3B']                # m
    R = constants['R']                      # m
    Cr = constants['Cr']                    # -
    srp_flux = constants['srp_flux']        # W/m2 J/s/m2 kg/s3
    a2m = constants['a2m']                  # m2/kg
    degree = constants['degree']            # -
    order = constants['order']              # -
    theta_gst = constants['theta_gst']      # rad
    w_body = constants['w_body']            # rad/s
    gc = constants['gc']                    # -
    C = gc['C_lm']                          # -
    S = gc['S_lm']                          # -
    isp = constants['isp']                  # s
    g = constants['g']                      # m/s2
    # c                                     # m/s
    # AU                                    # m
    # Cr * a2m * flux/c                     # m/s2

    srp_const = Cr*a2m*srp_flux/c.c*c.AU*c.AU/1000.0

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    m = X[6]        # SC Mass [kg]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    T = np.sqrt(u.dot(u))*m
    m_dot = T/g/isp

    # Carteisan and Orbital Elements
    oe = af.equ2oe(X)
    r, v = af.kep2cart(oe, mu=mu_host)
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_body*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=R)
    lat, lon, alt = lla

    # Grav Field
    dUdr_sum = 0
    dUdlat_sum = 0
    dUdlon_sum = 0
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
            
            # Normalized Spherical Harmonic Coefficients
            try:
                C_lm = C[l-2,m]
                S_lm = S[l-2,m]

            except:
                print('ERROR: degree or order too high for this body')

            # Normalizing Legedre Polynomials
            P_lm = legendre_poly(l, m, np.sin(lat))
            P_lm = P_lm/N_lm
            P_lm1 = legendre_poly(l, m+1, np.sin(lat))
            P_lm1 = P_lm1/N_lm

            # dU/dr
            dUdr_sum += (R/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (R/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (R/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
                - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

     # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # Determining Shadow Properties
    ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    if ang > 1:
        ang = 1
    elif ang < -1:
        ang = -1
    elif abs(ang) > 1.1:
        print("ERROR: SC angle is doing really weird things!!")

    phi = np.arccos(ang)
    phi_sun = np.arcsin(R_3B/r_sun2sc_mag)
    phi_host = np.arcsin(R/r_mag)

    # no eclipse
    if phi >= (phi_host + phi_sun):
        l = 1

    # partial eclipse
    elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
        l = (phi - phi_host)/2/phi_sun + 0.5

    # total eclipse
    elif phi < (phi_host - phi_sun):
        l = 0

    else:
        print('ERROR: The srp shadow conditions are incorrect!!')
        l = float('NaN')
        if r_mag < R:
            print('ERROR: The SC is inside the central body!!')

    a_srp = l*srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_srp + a_3B + u
    accel_RSW = CoordTrans('BCI', 'RSW', accel, oe=oe)
    FR = accel_RSW[0]
    FS = accel_RSW[1]
    FW = accel_RSW[2]

    # LPEs
    dX[0] = q*2*p/w*FS
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FS - g*b/w*FW)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FS + f*b/w*FW)
    dX[3] = q*s2/2/w*np.cos(L)*FW
    dX[4] = q*s2/2/w*np.sin(L)*FW
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FW
    dX[6] = -m_dot

    return dX
# ===================================================================
