##############
## Dynamics ##
##############

# Don Kuettel
# donald.kuetteliii@colorado.edu

"""
This script contains a variety of dynamic funcitons that use 
scipy.integrate.odeint to integrate
"""

####################
## Import modules ##
####################
import numpy as np
import constants as c
import coord_trans as ct
from math import factorial
import astro_functions as af
from legendre_poly import legendre_poly


# ===================================================================
## RV ##
"""
COAST
(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst0, gc, Cr, a2m, srp_flux, c, AU)

 1) ode_2bod_coast_rv               | 7 dim state
 2) ode_2bod_grav_coast_rv          | 7 dim state
 3) ode_2bod_srp_coast_rv           | 7 dim state
 4) ode_2bod_3bod_coast_rv          | 7 dim state
 5) ode_2bod_grav_srp_coast_rv      | 7 dim state
 6) ode_2bod_grav_3bod_coast_rv     | 7 dim state
 7) ode_2bod_srp_3bod_coast_rv      | 7 dim state
 8) ode_2bod_grav_srp_3bod_coast_rv | 7 dim state

BLT
(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst0, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0)

 9) ode_2bod_blt_rv                 | 7 dim state
10) ode_2bod_grav_blt_rv            | 7 dim state
11) ode_2bod_srp_blt_rv             | 7 dim state
12) ode_2bod_3bod_blt_rv            | 7 dim state
13) ode_2bod_grav_srp_blt_rv        | 7 dim state
14) ode_2bod_grav_3bod_blt_rv       | 7 dim state
15) ode_2bod_srp_3bod_blt_rv        | 7 dim state
16) ode_2bod_grav_srp_3bod_blt_rv   | 7 dim state

U
(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst0, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, u_hat)

17) ode_2bod_u_rv                 | 7 dim state
18) ode_2bod_grav_u_rv            | 7 dim state
19) ode_2bod_srp_u_rv             | 7 dim state
20) ode_2bod_3bod_u_rv            | 7 dim state
21) ode_2bod_grav_srp_u_rv        | 7 dim state
22) ode_2bod_grav_3bod_u_rv       | 7 dim state
23) ode_2bod_srp_3bod_u_rv        | 7 dim state
24) ode_2bod_grav_srp_3bod_u_rv   | 7 dim state
"""

# -------------------------------------------------------------------
# 1)
def ode_2bod_coast_rv(X, t, mu_host):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Derivatives
    accel = a_2B
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 2)
def ode_2bod_grav_coast_rv(X, t, mu_host, r_host, w_host, degree, order, theta_gst, gc):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Derivatives
    accel = a_2B + a_grav
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 3)
def ode_2bod_srp_coast_rv(X, t, mu_host, r_host, d_3B, r_3B, Cr, a2m, srp_flux, c, AU):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Derivatives
    accel = a_2B + a_srp
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 4)
def ode_2bod_3bod_coast_rv(X, t, mu_host, mu_3B, d_3B):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Derivatives
    accel = a_2B + a_3B
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 5)
def ode_2bod_grav_srp_coast_rv(X, t, mu_host, r_host, w_host, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Derivatives
    accel = a_2B + a_grav + a_srp
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 6)
def ode_2bod_grav_3bod_coast_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, degree, order, theta_gst, gc):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # Derivatives
    accel = a_2B + a_grav + a_3B
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 7)
def ode_2bod_srp_3bod_coast_rv(X, t, mu_host, r_host, mu_3B, d_3B, r_3B, Cr, a2m, srp_flux, c, AU):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Derivatives
    accel = a_2B + a_srp + a_3B
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 8)
def ode_2bod_grav_srp_3bod_coast_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_host - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Derivatives
    accel = a_2B + a_grav + a_srp + a_3B
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 9)
def ode_2bod_blt_rv(X, t, mu_host, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 10)
def ode_2bod_grav_blt_rv(X, t, mu_host, r_host, w_host, degree, order, theta_gst, gc, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + a_grav + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 11)
def ode_2bod_srp_blt_rv(X, t, mu_host, r_host, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + a_srp + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 12)
def ode_2bod_3bod_blt_rv(X, t, mu_host, mu_3B, d_3B, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + a_3B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 13)
def ode_2bod_grav_srp_blt_rv(X, t, mu_host, r_host, w_host, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + a_grav + a_srp + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 14)
def ode_2bod_grav_3bod_blt_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, degree, order, theta_gst, gc, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + a_grav + a_3B
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 15)
def ode_2bod_srp_3bod_blt_rv(X, t, mu_host, r_host, mu_3B, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + a_srp + a_3B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 16)
def ode_2bod_grav_srp_3bod_blt_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_host - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + a_grav + a_srp + a_3B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 17)
def ode_2bod_u_rv(X, t, mu_host, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 18)
def ode_2bod_grav_u_rv(X, t, mu_host, r_host, w_host, degree, order, theta_gst, gc, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + a_grav + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 19)
def ode_2bod_srp_u_rv(X, t, mu_host, r_host, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + a_srp + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 20)
def ode_2bod_3bod_u_rv(X, t, mu_host, mu_3B, d_3B, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + a_3B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 21)
def ode_2bod_grav_srp_u_rv(X, t, mu_host, r_host, w_host, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + a_grav + a_srp + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 22)
def ode_2bod_grav_3bod_u_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, degree, order, theta_gst, gc, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + a_grav + a_3B
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 23)
def ode_2bod_srp_3bod_u_rv(X, t, mu_host, r_host, mu_3B, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + a_srp + a_3B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 24)
def ode_2bod_grav_srp_3bod_u_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, u_hat):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_host - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                u_hat - thrust direction

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    if np.sqrt(u_hat.dot(u_hat)) == 0:
        m_dot = 0
    else:
        m_dot = T/g0/isp

    # Derivatives
    accel = a_2B + a_grav + a_srp + a_3B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------
# ===================================================================

# ===================================================================
## MEOE ##
"""
COAST
(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst0, gc, Cr, a2m, srp_flux, c, AU)

 1) ode_2bod_coast_meoe               | 7 dim state
 2) ode_2bod_grav_coast_meoe          | 7 dim state
 3) ode_2bod_srp_coast_meoe           | 7 dim state
 4) ode_2bod_3bod_coast_meoe          | 7 dim state
 5) ode_2bod_grav_srp_coast_meoe      | 7 dim state
 6) ode_2bod_grav_3bod_coast_meoe     | 7 dim state
 7) ode_2bod_srp_3bod_coast_meoe      | 7 dim state
 8) ode_2bod_grav_srp_3bod_coast_meoe | 7 dim state

BLT
(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst0, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0)

 9) ode_2bod_blt_meoe                 | 7 dim state
10) ode_2bod_grav_blt_meoe            | 7 dim state
11) ode_2bod_srp_blt_meoe             | 7 dim state
12) ode_2bod_3bod_blt_meoe            | 7 dim state
13) ode_2bod_grav_srp_blt_meoe        | 7 dim state
14) ode_2bod_grav_3bod_blt_meoe       | 7 dim state
15) ode_2bod_srp_3bod_blt_meoe        | 7 dim state
16) ode_2bod_grav_srp_3bod_blt_meoe   | 7 dim state
"""

# -------------------------------------------------------------------
# 1)
def ode_2bod_coast_meoe(X, t, mu_host):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    L = X[5]
    w = 1. + f*np.cos(L) + g*np.sin(L)

    # LPEs
    dX[0] = 0.
    dX[1] = 0.
    dX[2] = 0.
    dX[3] = 0.
    dX[4] = 0.
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 2)
def ode_2bod_grav_coast_meoe(X, t, mu_host, r_host, w_host, degree, order, theta_gst, gc):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

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
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Perturbations
    accel = a_grav
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 3)
def ode_2bod_srp_coast_meoe(X, t, mu_host, r_host, d_3B, r_3B, Cr, a2m, srp_flux, c, AU):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                    
    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

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
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Perturbations
    accel = a_srp
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 4)
def ode_2bod_3bod_coast_meoe(X, t, mu_host, mu_3B, d_3B):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
  
    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

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
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_3B
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 5)
def ode_2bod_grav_srp_coast_meoe(X, t, mu_host, r_host, w_host, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

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
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Perturbations
    accel = a_grav + a_srp
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 6)
def ode_2bod_grav_3bod_coast_meoe(X, t, mu_host, r_host, w_host, mu_3B, d_3B, degree, order, theta_gst, gc):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

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
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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
    accel = a_grav + a_3B
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 7)
def ode_2bod_srp_3bod_coast_meoe(X, t, mu_host, r_host, mu_3B, d_3B, r_3B, Cr, a2m, srp_flux, c, AU):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                    
    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

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
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_srp + a_3B
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 8)
def ode_2bod_grav_srp_3bod_coast_meoe(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_host - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

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
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Perturbations
    accel = a_grav + a_srp + a_3B
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 9)
def ode_2bod_blt_meoe(X, t, mu_host, g0, T, isp, A, B, t0):
    """This function integrates the classical orbital elements 
    accounting for keplerian motion and a control input.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=af.meoe2oe(X[0:6]))
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 10)
def ode_2bod_grav_blt_meoe(X, t, mu_host, r_host, w_host, degree, order, theta_gst, gc, g0, T, isp, A, B, t0):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Equinoctial Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

    dUdr = -mu_host/r_mag/r_mag*dUdr_sum
    dUdlat = mu_host/r_mag*dUdlat_sum
    dUdlon = mu_host/r_mag*dUdlon_sum

    ax = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*x - (dUdlon/(x*x + y*y))*y
    ay = (dUdr/r_mag - z/r_mag/r_mag/np.sqrt(x*x + y*y)*dUdlat)*y + (dUdlon/(x*x + y*y))*x
    az = dUdr/r_mag*z + np.sqrt(x*x + y*y)/r_mag/r_mag*dUdlat

    a_grav = np.array([ax, ay, az])

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = a_grav + at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 11)
def ode_2bod_srp_blt_meoe(X, t, mu_host, r_host, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants
                    
    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Cartesian Elements
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = a_srp + at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 12)
def ode_2bod_3bod_blt_meoe(X, t, mu_host, mu_3B, d_3B, g0, T, isp, A, B, t0):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants
  
    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Cartesian Elements
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = a_3B + at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 13)
def ode_2bod_grav_srp_blt_meoe(X, t, mu_host, r_host, w_host, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Equinoctial Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = a_grav + a_srp + at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 14)
def ode_2bod_grav_3bod_blt_meoe(X, t, mu_host, r_host, w_host, mu_3B, d_3B, degree, order, theta_gst, gc, g0, T, isp, A, B, t0):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Equinoctial Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = a_grav + a_3B + at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 15)
def ode_2bod_srp_3bod_blt_meoe(X, t, mu_host, r_host, mu_3B, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants
                    
    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]
    """

    dX = np.zeros(len(X))

    # Orbital Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Cartesian Elements
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    r_mag = np.sqrt(r.dot(r))

    # SRP from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = a_srp + a_3B + at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 16)
def ode_2bod_grav_srp_3bod_blt_meoe(X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B, t0):
    """This function integrates the modified equinocital orbital 
    elements accounting for keplerian motion.

    ASSUMPTIONS:
        -All units are m and m/s. This affects the srp constant.

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [p f g h k L m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_host - rotation rate of the central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the spacecraft          
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
                isp - specific impulse of thruster
                T - thrust
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
             acceleration where dX = [dp df dg dh dk dL dm]

    NOTES:
        This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
    """

    dX = np.zeros(len(X))

    # Equinoctial Elements
    p = X[0]
    f = X[1]
    g = X[2]
    h = X[3]
    k = X[4]
    L = X[5]
    mass = X[-1]

    s2 = 1. + h*h + k*k
    w = 1. + f*np.cos(L) + g*np.sin(L)
    q = np.sqrt(p/mu_host)
    b = h*np.sin(L) - k*np.cos(L)

    # Carteisan and Orbital Elements
    oe = af.meoe2oe(X[0:6])
    r, v = af.oe2rv(oe, mu=mu_host)
    x = r[0]; y = r[1]; z = r[2]
    r_mag = np.sqrt(r.dot(r))
    theta = w_host*t + theta_gst
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, theta_gst=theta, mu=mu_host, r_body=r_host)
    lat, lon, alt = lla

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) - C_lm*np.sin(m*lon))

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

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
    a_srp = srp_const*r_sun2sc/r_sun2sc_mag

    # Third-body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Perturbations
    accel = a_grav + a_srp + a_3B + at*u_hat
    accel_RIC = ct.CoordTrans('BCI', 'RIC', accel, oe=oe)
    FR = accel_RIC[0]
    FI = accel_RIC[1]
    FC = accel_RIC[2]

    # LPEs
    dX[0] = q*2*p/w*FI
    dX[1] = q*(np.sin(L)*FR + ((w + 1)*np.cos(L) + f)/w*FI - g*b/w*FC)
    dX[2] = q*(-np.cos(L)*FR + ((w + 1)*np.sin(L) + g)/w*FI + f*b/w*FC)
    dX[3] = q*s2/2/w*np.cos(L)*FC
    dX[4] = q*s2/2/w*np.sin(L)*FC
    dX[5] = np.sqrt(mu_host*p)*w*w/p/p + q*b/w*FC
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------
# ===================================================================


# ===================================================================
## Random ODE functions ##
"""
1) ode_full_blt_AB_rv  |  7 dim state
2) ode_full_BCI_blt_rv | 13 dim state
3) ode_full_RIC_blt_rv | 13 dim state
4) ode_full_lam_rv     |  7 dim state
5) ode_full_zemzev_rv  |  7 dim state
6) ode_full_coast_rv   |  7 dim state
7) ode_kep_blt_AB_rv   |  7 dim state
8) ode_kep_BCI_blt_rv  | 13 dim state
9) ode_kep_RIC_blt_rv  | 13 dim state
10) ode_kep_lam_rv     |  7 dim state
11) ode_kep_zemzev_rv  |  7 dim state
12) ode_kep_coast_rv   |  7 dim state
13) ode_kep_u_rv       |  6 dim state
14) ode_kep_rv         |  6 dim state
15) ode_kep_stm_rv     | 42 dim state
16) ode_bcbf_rv        |  6 dim state
17) ode_bcbf_stm_rv    | 42 dim state
18) ode_J2_rv          |  6 dim state
19) ode_J2_stm_rv      | 42 dim state
"""
# -------------------------------------------------------------------
# 1)
def ode_full_blt_AB_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, 
    r_3B, degree, order, theta_gst0, gc, Cr, a2m, isp, T, A, B, t0):
    """This derivative function computes the velocity and 
    acceleration of a spacecraft's state accounting for the gravity 
    field, srp, solar gravity, and BLT thrust perturbations. 

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [m], [m/s], and 
            [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                w_host - rotation rate of the central body in
                    [rad/sec]
                mu_3B - gravitational parameter of third
                    body in [m^3/s^2]
                d_3B - distance of third body from central 
                    body in [m]
                r_3B - radius of thrid body in [m]
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial
                    and body fixed frame
                gc - normalized C_lm and S_lm gravity constants
                    of host body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the satellite
                    in [m2/kg]
                isp - specific impulse of thruster [1/sec]
                T - thrust [N]
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
        -Make sure units match!
    """

    dX = np.zeros(len(X))
    
    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    mass = X[6]

    r = np.array([x, y, z]); v = np.array([vx, vy, vz])
    r_mag = np.sqrt(r.dot(r))
    oe = af.rv2oe(r, v, mu=mu_host)[0]

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*c.srp_flux/c.c*c.AU*c.AU*1000*1000
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
    theta = w_host*t + theta_gst0
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=r_host)
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
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

    # Mass flow rate
    at = T/mass
    m_dot = T/c.g0/isp

    # Control
    u_RIC = A + B*(t-t0)
    u_hat = u_RIC/np.sqrt(u_RIC.dot(u_RIC))
    # u_RIC_hat = u_RIC/np.sqrt(u_RIC.dot(u_RIC))
    # u_hat = ct.CoordTrans('RIC', 'BCI', u_RIC_hat, oe=oe, mu=mu_host)

    # Derivatives
    if r_mag < r_host:
        u_hat = float('NaN')*np.zeros(3)

    accel = a_2B + a_grav + a_3B + a_srp + at*u_hat
    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 2) 
def ode_full_blt_BCI_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, 
    r_3B, degree, order, theta_gst0, gc, Cr, a2m, isp, T):
    """This derivative function computes the velocity and 
    acceleration of a spacecraft's state accounting for the gravity 
    field, srp, solar gravity, and BLT thrust perturbations. 

    ASSUMPTIONS:
        -BLT guidance parameters are defined in the BCI frame.
        -uses costate
        
    INPUT:
        X - 13 dimensional state and costate of the spacecraft in a 
            numpy array where X = [x, y, z, vx, vy, vz, B1, B2, B3, 
            -A1, -A2, -A3, m] in [m], [m/s], and [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                w_host - rotation rate of the central body in
                    [rad/sec]
                mu_3B - gravitational parameter of third
                    body in [m^3/s^2]
                d_3B - distance of third body from central 
                    body in [m]
                r_3B - radius of thrid body in [m]
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial
                    and body fixed frame
                gc - normalized C_lm and S_lm gravity constants
                    of host body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the satellite
                    in [m2/kg]
                isp - specific impulse of thruster [1/sec]
                T - thrust [N]
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 13x1 matrix of the spacraft velocity and 
            acceleration where dX = [dr, dv, dB, -dA, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
        -Make sure units match!
    """

    dX = np.zeros(len(X))
    
    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    lr = X[6:9]     # B
    lv = X[9:12]    # -A
    mass = X[-1]

    r = np.array([x, y, z]); v = np.array([vx, vy, vz])
    r_mag = np.sqrt(r.dot(r)); lv_mag = np.sqrt(lv.dot(lv))
    oe = af.rv2oe(r, v, mu=mu_host)[0]

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*c.srp_flux/c.c*c.AU*c.AU*1000*1000
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
    theta = w_host*t + theta_gst0
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=r_host)
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
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

    # Mass flow rate
    at = T/mass
    m_dot = T/c.g0/isp

    # Control
    u_hat = -lv/lv_mag

    # Checking for collision
    accel = a_2B + a_grav + a_3B + a_srp + at*u_hat
    # if r_mag < r_host:
    #     accel = float('NaN')*np.zeros(3)

    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6:9] = np.zeros(3)
    dX[9:12] = -lr
    dX[-1] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 3)
def ode_full_blt_RIC_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, 
    r_3B, degree, order, theta_gst0, gc, Cr, a2m, isp, T):
    """This derivative function computes the velocity and 
    acceleration of a spacecraft's state accounting for the gravity 
    field, srp, solar gravity, and BLT thrust perturbations. 

    ASSUMPTIONS:
        -BLT guidance parameters are defined in the RIC frame.
        -uses costate
        
    INPUT:
        X - 13 dimensional state and costate of the spacecraft in a 
            numpy array where X = [x, y, z, vx, vy, vz, B1, B2, B3, 
            -A1, -A2, -A3, m] in [m], [m/s], and [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                w_host - rotation rate of the central body in
                    [rad/sec]
                mu_3B - gravitational parameter of third
                    body in [m^3/s^2]
                d_3B - distance of third body from central 
                    body in [m]
                r_3B - radius of thrid body in [m]
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial
                    and body fixed frame
                gc - normalized C_lm and S_lm gravity constants
                    of host body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the satellite
                    in [m2/kg]
                isp - specific impulse of thruster [1/sec]
                T - thrust [N]
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/dt. 13x1 matrix of the spacraft velocity and 
            acceleration where dX = [dr, dv, dB, -dA, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
        -Make sure units match!
    """

    dX = np.zeros(len(X))
    
    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    lr = X[6:9]     # B
    lv = X[9:12]    # -A
    mass = X[-1]

    r = np.array([x, y, z]); v = np.array([vx, vy, vz])
    r_mag = np.sqrt(r.dot(r)); lv_mag = np.sqrt(lv.dot(lv))
    oe = af.rv2oe(r, v, mu=mu_host)[0]

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*c.srp_flux/c.c*c.AU*c.AU*1000*1000
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
    theta = w_host*t + theta_gst0
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=r_host)
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
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

    # Mass flow rate
    at = T/mass
    m_dot = T/c.g0/isp

    # Control
    u_hat_ric = -lv/lv_mag
    u_hat = ct.CoordTrans('RIC', 'BCI', u_hat_ric, oe=oe, mu=mu_host)

    # Checking for collision
    accel = a_2B + a_grav + a_3B + a_srp + at*u_hat
    # if r_mag < r_host:
    #     accel = float('NaN')*np.zeros(3)

    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6:9] = np.zeros(3)
    dX[9:12] = -lr
    dX[-1] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 4)
def ode_full_lam_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, 
    r_3B, degree, order, theta_gst0, gc, Cr, a2m, mdot, u):
    """This derivative function computes the velocity and 
    acceleration of a spacecraft's state accounting for the gravity 
    field, srp, and solar gravity perturbations along with a 
    provided control vector. 

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [m], [m/s], and 
            [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                w_host - rotation rate of the central body in
                    [rad/sec]
                mu_3B - gravitational parameter of third
                    body in [m^3/s^2]
                d_3B - distance of third body from central 
                    body in [m]
                r_3B - radius of thrid body in [m]
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial
                    and body fixed frame
                gc - normalized C_lm and S_lm gravity constants
                    of host body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the satellite
                    in [m2/kg]
                mdot - mass flow rate [kg/s]
                u - control vector from guidance

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
        -Make sure units match!
    """

    dX = np.zeros(len(X))
    
    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    mass = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*c.srp_flux/c.c*c.AU*c.AU*1000*1000
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
    theta = w_host*t + theta_gst0
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=r_host)
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
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
    if r_mag < r_host:
        accel = float('NaN')*np.zeros(3)

    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -mdot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 5)
def ode_full_zemzev_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, 
    r_3B, degree, order, theta_gst0, gc, Cr, a2m, isp, u):
    """This derivative function computes the velocity and 
    acceleration of a spacecraft's state accounting for the gravity 
    field, srp, and solar gravity perturbations along with a 
    provided control vector. 

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [m], [m/s], and 
            [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                w_host - rotation rate of the central body in
                    [rad/sec]
                mu_3B - gravitational parameter of third
                    body in [m^3/s^2]
                d_3B - distance of third body from central 
                    body in [m]
                r_3B - radius of thrid body in [m]
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial
                    and body fixed frame
                gc - normalized C_lm and S_lm gravity constants
                    of host body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the satellite
                    in [m2/kg]
                isp - specific impulse 
                u - control vector from guidance

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
        -Make sure units match!
    """

    dX = np.zeros(len(X))
    
    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    mass = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*c.srp_flux/c.c*c.AU*c.AU*1000*1000
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
    theta = w_host*t + theta_gst0
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=r_host)
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
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

    # Mass flow rate
    mdot = m*np.sqrt(u.dot(u))/c.g0/isp

    # Derivatives
    accel = a_2B + a_grav + a_3B + a_srp + u
    if r_mag < r_host:
        accel = float('NaN')*np.zeros(3)

    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -mdot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 6)
def ode_full_coast_rv(X, t, mu_host, r_host, w_host, mu_3B, d_3B, 
    r_3B, degree, order, theta_gst0, gc, Cr, a2m):
    """This derivative function computes the velocity and 
    acceleration of a spacecraft's state accounting for the gravity 
    field, srp, and solar gravity perturbations. 

    ASSUMPTIONS:
        
    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [m], [m/s], and 
            [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                w_host - rotation rate of the central body in
                    [rad/sec]
                mu_3B - gravitational parameter of third
                    body in [m^3/s^2]
                d_3B - distance of third body from central 
                    body in [m]
                r_3B - radius of thrid body in [m]
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial
                    and body fixed frame
                gc - normalized C_lm and S_lm gravity constants
                    of host body
                Cr - coefficient of reflectivity of SC
                a2m - area to mass ratio of the satellite
                    in [m2/kg]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -This derivation uses the normalized spherical harmonic 
        coefficents (C_lm and S_lm).
        -Make sure units match!
    """
    
    dX = np.zeros(len(X))
    
    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    mass = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Sun Vectors
    r_host2sun = d_3B*np.array([-1, 0, 0])
    r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
    r_sun2sc = r - r_host2sun
    r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

    # # Determining Shadow Properties
    # ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
    # if ang > 1:
    #     ang = 1
    # elif ang < -1:
    #     ang = -1
    # elif abs(ang) > 1.1:
    #     print("ERROR: SC angle is doing really weird things!!")

    # phi = np.arccos(ang)
    # phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
    # phi_host = np.arcsin(r_host/r_mag)

    # # no eclipse
    # if phi >= (phi_host + phi_sun):
    #     l = 1

    # # partial eclipse
    # elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
    #     l = (phi - phi_host)/2/phi_sun + 0.5

    # # total eclipse
    # elif phi < (phi_host - phi_sun):
    #     l = 0

    # else:
    #     print('ERROR: The srp shadow conditions are incorrect!!')
    #     l = float('NaN')
    #     if r_mag < r_host:
    #         print('ERROR: The SC is inside the central body!!')

    l = 1
    srp_const = l*Cr*a2m*c.srp_flux/c.c*c.AU*c.AU*1000*1000
    a_srp = srp_const/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag*r_sun2sc

    # Third-Body from Sun
    a_3B = -mu_3B*(r_sun2sc/r_sun2sc_mag/r_sun2sc_mag/r_sun2sc_mag + 
        r_host2sun/r_host2sun_mag/r_host2sun_mag/r_host2sun_mag)

    # Grav Field
    C = gc['C_lm']
    S = gc['S_lm']
    theta = w_host*t + theta_gst0
    if theta > 2*np.pi:
        theta -= np.floor(theta/2/np.pi)*2*np.pi
    lla = ct.CoordTrans('BCI', 'LLA', r, 
        theta_gst=theta, mu=mu_host, r_body=r_host)
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
            dUdr_sum += (r_host/r_mag)**l*(l + 1)*P_lm*(C_lm*np.cos(m*lon) 
                + S_lm*np.sin(m*lon))

            # dU/dlat
            dUdlat_sum += (r_host/r_mag)**l*(P_lm1 - m*np.tan(lat)*P_lm)*(
                C_lm*np.cos(m*lon) + S_lm*np.sin(m*lon))

            # dU/dlon
            dUdlon_sum += (r_host/r_mag)**l*m*P_lm*(S_lm*np.cos(m*lon) 
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
    accel = a_2B + a_grav + a_3B + a_srp
    if r_mag < r_host:
        accel = float('NaN')*np.zeros(3)

    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 7)
def ode_kep_blt_AB_rv(X, t, mu_host, g0, T, isp, A, B, t0):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state along with the BLT thrust. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                isp - specific impulse of thruster [1/sec]
                T - thrust [N]
                A - guidance constants
                B - guidance constants

    OUTPUT:
        dX - dX/tfin_int. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[-1]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_temp = A + B*(t-t0)
    u_hat = u_temp/np.sqrt(u_temp.dot(u_temp))

    # Derivatives
    accel = a_2B + at*u_hat
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 8)
def ode_kep_blt_BCI_rv(X, t, mu_host, r_host, g0, T, isp):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state along with the BLT thrust. 

    ASSUMPTIONS:
        -Keplerian Dynamics
        -Guidance Parameters defined in the BCI frame

    INPUT:
        X - 13 dimensional state and costate of the spacecraft in a 
            numpy array where X = [x, y, z, vx, vy, vz, B1, B2, B3, 
            -A1, -A2, -A3, m]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body
                isp - specific impulse of thruster
                g0 - standard gravity
                T - thrust

    OUTPUT:
        dX - dX/dt. 13x1 matrix of the spacraft velocity and 
            acceleration where dX = [dr, dv, dB, -dA, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position, Velocity, Costate (RIC)
    r = X[0:3]
    v = X[3:6]
    lr = X[6:9]     # B
    lv = X[9:12]    # -A
    mass = X[-1]

    r_mag = np.sqrt(r.dot(r))
    lv_mag = np.sqrt(lv.dot(lv))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_hat = -lv/lv_mag

    # Checking for collision
    accel = a_2B + at*u_hat
    if r_mag < r_host:
        accel = float('NaN')*np.zeros(3)

    # Derivatives
    dX[0:3] = v
    dX[3:6] = accel
    dX[6:9] = np.zeros(3)
    dX[9:12] = -lr
    dX[-1] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 9) 
def ode_kep_blt_RIC_rv(X, t, mu_host, r_host, g0, T, isp):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state along with the BLT thrust. 

    ASSUMPTIONS:
        -Keplerian Dynamics
        -Guidance Parameters defined in the RIC frame

    INPUT:
        X - 13 dimensional state and costate of the spacecraft in a 
            numpy array where X = [x, y, z, vx, vy, vz, B1, B2, B3, 
            -A1, -A2, -A3, m] in [m], [m/s], and [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                isp - specific impulse of thruster [1/sec]
                T - thrust [N]

    OUTPUT:
        dX - dX/dt. 13x1 matrix of the spacraft velocity and 
            acceleration where dX = [dr, dv, dB, -dA, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position, Velocity, Costate (RIC)
    r = X[0:3]
    v = X[3:6]
    lr = X[6:9]     # B
    lv = X[9:12]    # -A
    mass = X[-1]

    r_mag = np.sqrt(r.dot(r))
    lv_mag = np.sqrt(lv.dot(lv))
    oe = af.rv2oe(r, v, mu=mu_host)[0]

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    at = T/mass
    m_dot = T/g0/isp

    # Control
    u_hat_ric = -lv/lv_mag
    u_hat = ct.CoordTrans('RIC', 'BCI', u_hat_ric, oe=oe, mu=mu_host)

    # Checking for collision
    accel = a_2B + at*u_hat
    if r_mag < r_host:
        accel = float('NaN')*np.zeros(3)

    # Derivatives
    dX[0:3] = v
    dX[3:6] = accel
    dX[6:9] = np.zeros(3)
    dX[9:12] = -lr
    dX[-1] = -m_dot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 10) 
def ode_kep_lam_rv(X, t, mu_host, r_host, mdot, u):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state with an additional control vector 
    provided. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [m], [m/s], and 
            [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                mdot - mass flow rate [kg/s]
                u - control vector from guidance

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    x = X[0]
    y = X[1]
    z = X[2]
    vx = X[3]
    vy = X[4]
    vz = X[5]
    mass = X[6]

    r = np.array([x, y, z])
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Derivatives
    accel = a_2B + u
    if r_mag < r_host:
        accel = np.zeros(3) + float('NaN')

    dX[0] = vx
    dX[1] = vy
    dX[2] = vz
    dX[3] = accel[0]
    dX[4] = accel[1]
    dX[5] = accel[2]
    dX[6] = -mdot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 11)
def ode_kep_zemzev_rv(X, t, mu_host, r_host, isp, u):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state with an additional control vector 
    provided. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [m], [m/s], and 
            [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]
                mdot - mass flow rate [kg/s]
                u - control vector from guidance

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    v = X[3:6]
    m = X[6]

    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Mass flow rate
    mdot = m*np.sqrt(u.dot(u))/c.g0/isp

    # Derivatives
    accel = a_2B + u
    if r_mag < r_host:
        accel = np.zeros(3)*float('NaN')

    dX[0:3] = v
    dX[3:6] = accel
    dX[6] = -mdot

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 12)
def ode_kep_coast_rv(X, t, mu_host, r_host):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 7 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz, m] in [m], [m/s], and 
            [kg]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central
                    body [m^3/s^2]
                r_host - radius of central body in [m]

    OUTPUT:
        dX - dX/dt. 7x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, mdot] 
            in [m/s], [m/s2], and [kg/s]

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    mass = X[6]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Derivatives
    accel = a_2B
    if r_mag < r_host:
        accel = np.zeros(3) + float('NaN')

    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6] = 0.

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 13)
def ode_kep_u_rv(X, t, mu_host, r_host, u):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body
                r_host - radius of central body in
                u - input control vector

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Derivatives
    accel = a_2B + u
    if r_mag < r_host:
        accel = np.zeros(3) + float('NaN')

    dX[0:3] = X[3:6]
    dX[3:6] = accel

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 14)
def ode_kep_rv(X, t, mu_host, r_host):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [x, y, z, vx, vy, vz]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body
                r_host - radius of central body in

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Derivatives
    accel = a_2B
    if r_mag < r_host:
        accel = np.zeros(3) + float('NaN')

    dX[0:3] = X[3:6]
    dX[3:6] = accel

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 15) 
def ode_kep_stm_rv(X, t, mu_host, r_host):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 6 + 6*6 dimensional state of the spacecraft in a numpy 
            array where X = [x, y, z, vx, vy, vz, STM]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body
                r_host - radius of central body in

    OUTPUT:
        dX - dX/dt. (6 + 6*6)x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, STM] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X)); n = 6

    # Position and Velocity
    x = X[0]; y = X[1]; z = X[2]; r = X[0:3]
    r_mag = np.sqrt(r.dot(r))

    # Two-Body
    a_2B = -mu_host/r_mag/r_mag/r_mag*r

    # Reassigning A matrix
    A = np.zeros((n,n))
    A[0,3] = 1.; A[1,4] = 1.; A[2,5] = 1.
    A[3,0] = 3*mu_host*x*x/r_mag**5 - mu_host/r_mag**3
    A[3,1] = 3*mu_host*x*y/r_mag**5
    A[3,2] = 3*mu_host*x*z/r_mag**5
    A[4,0] = 3*mu_host*y*x/r_mag**5
    A[4,1] = 3*mu_host*y*y/r_mag**5 - mu_host/r_mag**3
    A[4,2] = 3*mu_host*y*z/r_mag**5
    A[5,0] = 3*mu_host*z*x/r_mag**5
    A[5,1] = 3*mu_host*z*y/r_mag**5
    A[5,2] = 3*mu_host*z*z/r_mag**5 - mu_host/r_mag**3

    # STM
    phi = np.reshape(X[6::], (n,n))
    phiDot = np.dot(A, phi)

    # Derivatives
    accel = a_2B
    if r_mag < r_host:
        accel = np.zeros(3) + float('NaN')
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6::] = np.reshape(phiDot, n*n)

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 16)
def ode_bcbf_rv(X, t, r_host, mu_host, w_host):
    """This function contains the dynamics of a spacecraft in the 
    body-frame of a body.
    
    ASSUMPTIONS:
        -Keplerian Body
        -Sphereical Body
        -In the Body Frame

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [r, v] in [m] and [m/s]
        t - time of integration in seconds
        constants - a tuple of constants necessary for integration
                    w_host - the rotational rate vector of the 
                        primary body in [rad/s]
                    r_host - radius of the primary body in [m]
                    mu_host - gravitational parameter in [m3/s2]
                    theta_gst - the intial angle between the body
                        frame and the inertial frame

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [v, a]

    """ 
    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    r_mag = np.sqrt(r.dot(r))
    v = X[3:6]
    w = np.array([0., 0., w_host])

    # Keplerian acceleration
    a_2B = -mu_host/r_mag**3*r

    # Rotational acceleration in the body frame
    a_rot = -2*np.cross(w, v) - np.cross(w, np.cross(w, r))

    # Derivatives
    accel = a_2B + a_rot
    if r_mag < r_host:
        accel = np.zeros(3)*float('NaN')
    dX[0:3] = v
    dX[3:6] = accel

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 17) 
def ode_bcbf_stm_rv(X, t, mu_host, r_host, w_host):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state as well as the state transition matrix. 

    ASSUMPTIONS:
        -Keplerian Body
        -Sphereical Body
        -In the Body Frame

    INPUT:
        X - 6 + 6*6 dimensional state of the spacecraft in a numpy 
            array where X = [x, y, z, vx, vy, vz, STM]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body
                r_host - radius of central body
                w_host - rotation rate of central body

    OUTPUT:
        dX - dX/dt. (6 + 6*6)x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, STM] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X)); n = 6

    # Position and Velocity
    x = X[0]; y = X[1]; z = X[2]; r = X[0:3]
    vx = X[3]; vy = X[4]; vz = X[5]; v = X[3:6]
    r_mag = np.sqrt(r.dot(r))
    v = X[3:6]
    w = np.array([0., 0., w_host])

    # Keplerian acceleration
    a_2B = -mu_host/r_mag**3*r

    # Rotational acceleration in the body frame
    a_rot = -2*np.cross(w,v) - np.cross(w, np.cross(w,r))

    # Reassigning A matrix
    A = np.zeros((n,n))
    A[0,3] = 1.; A[1,4] = 1.; A[2,5] = 1.
    A[3,0] = w_host**2 + 3*mu_host*x*x/r_mag**5 - mu_host/r_mag**3
    A[3,1] = 3*mu_host*x*y/r_mag**5
    A[3,2] = 3*mu_host*x*z/r_mag**5
    A[3,4] = 2*w_host
    A[4,0] = 3*mu_host*y*x/r_mag**5
    A[4,1] = w_host**2 + 3*mu_host*y*y/r_mag**5 - mu_host/r_mag**3
    A[4,2] = 3*mu_host*y*z/r_mag**5
    A[4,3] = -2*w_host
    A[5,0] = 3*mu_host*z*x/r_mag**5
    A[5,1] = 3*mu_host*z*y/r_mag**5
    A[5,2] = 3*mu_host*z*z/r_mag**5 - mu_host/r_mag**3

    # STM
    phi = np.reshape(X[6::], (n,n))
    phiDot = np.dot(A, phi)

    # Derivatives
    accel = a_2B + a_rot
    if r_mag < r_host:
        accel = np.zeros(3)*float('NaN')
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6::] = np.reshape(phiDot, n*n) 

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 18) 
def ode_J2_rv(X, t, r_host, mu_host, J2_host):
    """This function computes the acceleration of a spacecraft using
    Keplerian + J2 dynamics. 
    
    ASSUMPTIONS:

    INPUT:
        X - 6 dimensional state of the spacecraft in a numpy array
            where X = [r, v]
        t - time of integration in seconds
        constants - a tuple of constants necessary for integration
                    J2_host - J2 term of the primary body
                    r_host - radius of the primary body
                    mu_host - gravitational parameter

    OUTPUT:
        dX - dX/dt. 6x1 matrix of the spacraft velocity and 
             acceleration where dX = [v, a]

    """ 
    dX = np.zeros(len(X))

    # Position and Velocity
    r = X[0:3]
    r_mag = np.sqrt(r.dot(r))

    # Keplerian acceleration
    a_2B = -mu_host/r_mag**3*r

    # J2 acceleration
    a_J2 = np.zeros(3)
    a_J2[0] = mu_host*J2_host*r_host**2*(15*z*z*x/r_mag**7 - 3*x/r_mag**5)/2
    a_J2[1] = mu_host*J2_host*r_host**2*(15*z*z*y/r_mag**7 - 3*y/r_mag**5)/2
    a_J2[2] = mu_host*J2_host*r_host**2*(15*z*z*z/r_mag**7 - 9*z/r_mag**5)/2

    # Derivatives
    accel = a_2B + a_J2
    if r_mag < r_host:
        accel = np.zeros(3)*float('NaN')
    dX[0:3] = X[3:6]
    dX[3:6] = accel

    return dX
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 19) 
def ode_J2_stm_rv(X, t, mu_host, r_host, J2_host):
    """This function computes the keplerian velocity and acceleration 
    of a spacecraft's state. 

    ASSUMPTIONS:
        -Keplerian Dynamics

    INPUT:
        X - 6 + 6*6 dimensional state of the spacecraft in a numpy 
            array where X = [x, y, z, vx, vy, vz, STM]
        t - time of integration in seconds
        Constants:
                mu_host - gravitational parameter of central body
                r_host - radius of central body
                w_host - rotation rate of central body

    OUTPUT:
        dX - dX/dt. (6 + 6*6)x1 matrix of the spacraft velocity and 
            acceleration where dX = [vx, vy, vz, ax, ay, az, STM] 

    NOTES:
        -Make sure units match!
    """

    dX = np.zeros(len(X)); n = 6

    # Position and Velocity
    x = X[0]; y = X[1]; z = X[2]; r = X[0:3]
    vx = X[3]; vy = X[4]; vz = X[5]; v = X[3:6]
    r_mag = np.sqrt(r.dot(r))

    # Keplerian acceleration
    a_2B = -mu_host/r_mag**3*r

    # J2 acceleration
    a_J2 = np.zeros(3)
    a_J2[0] = mu_host*J2_host*r_host**2*(15*z*z*x/r_mag**7 - 3*x/r_mag**5)/2
    a_J2[1] = mu_host*J2_host*r_host**2*(15*z*z*y/r_mag**7 - 3*y/r_mag**5)/2
    a_J2[2] = mu_host*J2_host*r_host**2*(15*z*z*z/r_mag**7 - 9*z/r_mag**5)/2
    
    # Reassigning A matrix
    A = np.zeros((n,n))
    A[0,3] = 1.; A[1,4] = 1.; A[2,5] = 1.
    A[3,0] = 3*mu_host*x*x/r_mag**5 - mu_host/r_mag**3 - 3*mu_host*J2_host*r_host**2*(1/r_mag**5 - 5*x*x/r_mag**7 - 5*z*z/r_mag**7 + 35*x*x*z*z/r_mag**9)/2 
    A[3,1] = 3*mu_host*x*y/r_mag**5 + 15*mu_host*J2_host*r_host**2*x*y*(1/r_mag**7 - 7*z*z/r_mag**9)/2
    A[3,2] = 3*mu_host*x*z/r_mag**5 + 15*mu_host*J2_host*r_host**2*x*z*(3/r_mag**7 - 7*z*z/r_mag**9)/2
    A[4,0] = 3*mu_host*y*x/r_mag**5 + 15*mu_host*J2_host*r_host**2*x*y*(1/r_mag**7 - 7*z*z/r_mag**9)/2
    A[4,1] = 3*mu_host*y*y/r_mag**5 - mu_host/r_mag**3 - 3*mu_host*J2_host*r_host**2*(1/r_mag**5 - 5*y*y/r_mag**7 - 5*z*z/r_mag**7 + 35*y*y*z*z/r_mag**9)/2 
    A[4,2] = 3*mu_host*y*z/r_mag**5 + 15*mu_host*J2_host*r_host**2*y*z*(3/r_mag**7 - 7*z*z/r_mag**9)/2
    A[5,0] = 3*mu_host*z*x/r_mag**5 + 15*mu_host*J2_host*r_host**2*x*z*(1/r_mag**7 - 7*z*z/r_mag**9)/2
    A[5,1] = 3*mu_host*z*y/r_mag**5 + 15*mu_host*J2_host*r_host**2*y*z*(3/r_mag**7 - 7*z*z/r_mag**9)/2
    A[5,2] = 3*mu_host*z*z/r_mag**5 - mu_host/r_mag**3 - mu_host*J2_host*r_host**2*(9/r_mag**5 - 90*z*z/r_mag**7 + 105*z*z*z*z/r_mag**9)/2

    # STM
    phi = np.reshape(X[6::], (n,n))
    phiDot = np.dot(A, phi)

    # Derivatives
    accel = a_2B + a_J2
    if r_mag < r_host:
        accel = np.zeros(3)*float('NaN')
    dX[0:3] = X[3:6]
    dX[3:6] = accel
    dX[6::] = np.reshape(phiDot, n*n) 

    return dX
# -------------------------------------------------------------------
# ===================================================================

# # Eclipse SRP Model
# # SRP from Sun
#     r_host2sun = d_3B*np.array([-1, 0, 0])
#     r_host2sun_mag = np.sqrt(r_host2sun.dot(r_host2sun))
#     r_sun2sc = r - r_host2sun
#     r_sun2sc_mag = np.sqrt(r_sun2sc.dot(r_sun2sc))

#     # Determining Shadow Properties
#     ang = r_sun2sc.dot(r)/r_sun2sc_mag/r_mag
#     if ang > 1:
#         ang = 1
#     elif ang < -1:
#         ang = -1
#     elif abs(ang) > 1.1:
#         print("ERROR: SC angle is doing really weird things!!")

#     phi = np.arccos(ang)
#     phi_sun = np.arcsin(r_3B/r_sun2sc_mag)
#     phi_host = np.arcsin(r_host/r_mag)

#     # no eclipse
#     if phi >= (phi_host + phi_sun):
#         l = 1

#     # partial eclipse
#     elif ((phi_host - phi_sun)) <= phi <= (phi_host + phi_sun):
#         l = (phi - phi_host)/2/phi_sun + 0.5

#     # total eclipse
#     elif phi < (phi_host - phi_sun):
#         l = 0

#     else:
#         print('ERROR: The srp shadow conditions are incorrect!!')
#         l = float('NaN')
#         if r_mag < r_host:
#             print('ERROR: The SC is inside the central body!!')

#     srp_const = l*Cr*a2m*srp_flux/c*AU*AU/r_sun2sc_mag/r_sun2sc_mag
#     a_srp = srp_const*r_sun2sc/r_sun2sc_mag
