
#################################
##    Astrodynamic Functions   ##
#################################

"""This script provides a variety of useful functions for astrodynamic
purposes.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

    1) cart2kep (Cartesian Elements --> Keplerian Orbital Elements)
    2) kep2cart (Keplerian Orbital Elements --> Cartesian Elements)
    3) oe2equ (Classical Elements --> Modified Mquinoctial Elements)
    4) equ2oe (Modified Mquinoctial Elements --> Classical Elements)
    5) true2mean (True Anomaly --> Eccentric and Mean Anomaly)
    6) mean2true (Mean Anomaly --> Eccentric and True Anomaly)
    7) hohmann (Find dVs necessary for Hohmann Transfer)
    8) theta_gst (Find theta_GST from Julian Date)
    9) julian_date (Finds Julian Date from Gregorian Date)
   10) gregorian_date (Finds Gregorian Date from Julian Date)
   11) ymd2dayofyear (year/month/day hr:min:sec --> day of year)
   12) dayofyear2ymd (day of year --> year/month/day hr:min:sec)
   13) circ_rendez (computes rendez of two sc in same circular orbit)
   14) hill_eqns (determining proximity motion of two s/c)
   15) hill_rendez (finds dV necessary to return to reference s/c)
   16) fpa (find flight path angle)
   17) meeus (find orbital elements of planets with JD)
   18) lambert_uv (universal variables lambert problem solver)
   19) lambert_PC (Prussing and Conway lambert problem solver)
   20) lambert_uv_multirev (multirev lambert problem solver)
"""

# Import Modules
import constants as c
import numpy as np
from scipy.optimize import fsolve

# Define Functions
# =====================================================================
# 1) 
def cart2kep(r, v, mu=c.mu_earth):
    """This function takes the cartesian orbital elements of a 
    spacecraft (x, y, z, vx, vy, vz) and returns the Kepler orbital 
    elements (a, e, i, RAAN, w, nu).

    ASSUMPTIONS:
    - keplarian motion

    INPUT:
        r - [x, y, z] np.array position vector of sc in [km]
        v - [vx, vy, vz] np.array velocity vector of sc in [km/s]
        mu - gravitational parameter of planet (assumed to be Earth) 
             in [km^3/s^2]

        Where:
            x - x-position of the spacecraft in [km]
            y - y-position of the spacecraft in [km]
            z - z-position of the spacecraft in [km]
            vx - x-direction velocity of the spacecraft in [km/s]
            vy - y-direction velocity of the spacecraft in [km/s]
            vz - z-direction velocity of the spacecraft in [km/s]

    OUTPUT:
        a - semimajor axis of orbit in [km]
        e - eccintricity of the spacecraft 0<e<1
        i - inclination of the spacecraft 0<i<pi [rad]
        w - argument of periapsis of the spacecraft 0<i<2*pi [rad]
        raan - right accension of the accending node of the 
               spacecraft 0<raan<2*pi [rad]
        nu - true anomaly 0<nu<2*pi [rad]
        ea - eccentric anomaly 0<EA<2*pi [rad]
        ma - mean anomaly 0<M<2*pi [rad]
        d_tp - time past since periapse [sec]
    """

    # Position and Velocity
    r_mag = np.sqrt(r.dot(r))         # radius of spacecraft [km]
    v_mag = np.sqrt(v.dot(v))         # velocity of spacecraft [km/s]

    # Angular Momentum
    h_xyz = np.cross(r, v)        # specific angular momentum vector
    h = np.sqrt(h_xyz.dot(h_xyz)) # specific angular momentum

    k_unit = np.array([0, 0, 1.0])
    line_nodes = np.cross(k_unit, h_xyz)        # line of nodes
    line_nodes_mag = np.sqrt(line_nodes.dot(line_nodes))
    se = v_mag*v_mag/2 - mu/r_mag               # specific energy

    # Computing Semimajor Axis
    a = -mu/2.0/se                             # semimajor axis [km]

    # Computing Eccentricity
    e_vec = np.cross(v, h_xyz)/mu - r/r_mag  # eccentricity vector
    e = np.sqrt(e_vec.dot(e_vec))            # eccentricity

    # Computing Inclination
    inc = np.arccos(h_xyz[-1]/h)

    # Computing RAAN
    if line_nodes[1] >= 0:
        raan = np.arccos(line_nodes[0]/line_nodes_mag)
    else:
        raan = 2*np.pi - np.arccos(line_nodes[0]/line_nodes_mag)

    # Computing Argument of Periapse
    if e_vec[-1] >= 0:
        w = np.arccos(np.dot(line_nodes, e_vec)/(line_nodes_mag*e))
    else:
        w = 2*np.pi - np.arccos(
            np.dot(line_nodes, e_vec)/(line_nodes_mag*e))

    # Computing True Anomaly
    if np.dot(r, v) >= 0:
        nu = np.arccos(np.dot(e_vec, r)/(e*r_mag))
    else:
        nu = 2*np.pi - np.arccos(np.dot(e_vec, r)/(e*r_mag))

    # Special Cases
    # If raan is undefined (equatorial orbit)
    if np.isnan(raan) == True: # i = 0 or pi
    # if abs(inc) < 1e-6 or abs(abs(inc) - np.pi) < 1e-6:
        raan = float('NaN')
        # If w and raan are undefined (equatorial circular orbit)
        if np.isnan(w) == True:
        # if abs(e) < 1e-6:
            w = float('NaN')

            if r[1] >= 0:
                gamma_true = np.arccos(r[0]/r_mag)
            else:
                gamma_true = 2*np.pi - np.arccos(r[0]/r_mag)
    
            if abs(inc) < 1:
                inc = 0
            else:
                inc = np.pi
            nu = gamma_true
            # print("Circular Equatorial Orbit: nu = gamma_true")

        # If only raan is undefined (equatorial elliptical orbit)
        else:
            if e_vec[1] >= 0:
                w_true = np.arccos(e_vec[0]/e)
            else:
                w_true = 2*np.pi - np.arccos(e_vec[0]/e)
    
            w = w_true
            # print("Elliptical equatorial Orbit: w = w_true")

    # If w is undefined (circular orbits)
    if np.isnan(w) == True and np.isnan(raan) == False:
    # if abs(e) < 1e-6:
        if r[-1] >= 0:
            u = np.arccos(
                np.dot(line_nodes, r)/(line_nodes_mag*r_mag))
        else:
            u = 2*np.pi - np.arccos(
                np.dot(line_nodes, r)/(line_nodes_mag*r_mag))

        nu = u
        w = float('NaN')
        # print("Circular Orbit: nu = u")

    # Computing Mean Anomaly
    cos_ea = (r_mag*np.cos(nu) + a*e)/a
    sin_ea = r_mag*np.sin(nu)/(a*np.sqrt(1 - e*e))
    ea = np.arctan2(sin_ea, cos_ea)

    if ea < 0:
        ea += 2*np.pi

    ma = ea - e*np.sin(ea)
    n = np.sqrt(mu/a/a/a)                     # 1/sec
    d_tp = ma/n                               # sec
    
    oe = [a, e, inc, raan, w, nu]

    return oe, ea, ma, d_tp
# =====================================================================


# =====================================================================
# 2) 
def kep2cart(oe, mu=c.mu_earth):
    """This function takes the Kepler orbital elements (a, e, i, RAAN,
    w, nu) of a spacecraft and returns the cartesian orbital elements 
    (x, y, z, vx, vy, vz).

    ASSUMPTIONS:
    - keplarian motion

    INPUT:
        a - semimajor axis of orbit in [km]
        e - eccintricity of the spacecraft 0<e<1
        i - inclination of the spacecraft 0<i<pi [rad]
        w - argument of periapsis of the spacecraft 0<i<2*pi [rad]
        raan - right accension of the accending node of the 
               spacecraft 0<raan<2*pi [rad]
        nu - true anomaly 0<nu<2*pi [rad]
        mu - gravitational parameter of planet (assumed to be Earth) 
             in [km^3/s^2]

    OUTPUT:
        r - [x, y, z] np.array position vector of sc in [km]
        v - [vx, vy, vz] np.array velocity vector of sc in [km/s]

        Where:
            x - x-position of the spacecraft in [km]
            y - y-position of the spacecraft in [km]
            z - z-position of the spacecraft in [km]
            vx - x-direction velocity of the spacecraft in [km/s]
            vy - y-direction velocity of the spacecraft in [km/s]
            vz - z-direction velocity of the spacecraft in [km/s]
    """

    # classical elements
    a, e, inc, raan, w, nu = oe

    p = a*(1 - e*e)                          # semiparameter [km]
    r_mag = p/(1 + e*np.cos(nu))             # radius [km]

    # Rotation Matrices
    R3_w = np.array([[np.cos(w), -np.sin(w), 0], 
                     [np.sin(w), np.cos(w), 0], 
                     [0, 0, 1]])
    R1_inc = np.array([[1, 0, 0], 
                       [0, np.cos(inc), -np.sin(inc)], 
                       [0, np.sin(inc), np.cos(inc)]])
    R3_raan = np.array([[np.cos(raan), -np.sin(raan), 0],
                        [np.sin(raan), np.cos(raan), 0], 
                         [0, 0, 1]])

    # Position and Velocity Vectors
    r_pqw = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0])
    v_pqw = np.array([-(np.sqrt(mu/p))*np.sin(nu), 
        np.sqrt(mu/p)*(e + np.cos(nu)), 0])
    
    r = R3_raan @ R1_inc @ R3_w @ r_pqw
    v = R3_raan @ R1_inc @ R3_w @ v_pqw

    return r, v
# =====================================================================


# ===================================================================
# 3)
def oe2equ(oe):
    """This function converts the classical orbital elements to the
    modified equinoctial elements

    ASSUMPTIONS: 
        -keplerian dynamics
        
    INPUT:
        oe - a 6x1 array of the classical orbital elements
            a: semi-major axis
            e: eccentricity
            inc: inclination
            raan: right ascension of the ascending node
            w: argument of periapsis
            nu: true anomaly

    OUTPUT:
        equ - a 6x1 array of the modified equinoctial elements
            p: semilatus rectum
            f: X-component of eccentricity vector in EQU frame
            g: Y-component of eccentricity vector in EQU frame
            h: X-component of ascending node vector in EQU frame
            k: Y-component of ascending node vector in EQU frame
            L: True longitude
    """

    # classical elements
    a, e, inc, raan, w, nu = oe

    # equinoctial elements
    p = a*(1 - e*e)
    f = e*np.cos(w + raan)
    g = e*np.sin(w + raan)
    h = np.tan(inc/2)*np.cos(raan)
    k = np.tan(inc/2)*np.sin(raan)
    L = raan + w + nu

    equ = [p, f, g, h, k, L]

    return equ
# ===================================================================


# ===================================================================
# 4)
def equ2oe(equ):
    """This function converts the classical orbital elements to the
    modified equinoctial elements

    ASSUMPTIONS: 
        -keplerian dynamics
        
    INPUT:
        p: semilatus rectum
            f: X-component of eccentricity vector in EQU frame
            g: Y-component of eccentricity vector in EQU frame
            h: X-component of ascending node vector in EQU frame
            k: Y-component of ascending node vector in EQU frame
            L: True longitude

    OUTPUT:
        oe - a 6x1 array of the classical orbital elements
            a: semi-major axis
            e: eccentricity
            inc: inclination
            raan: right ascension of the ascending node
            w: argument of periapsis
            nu: true anomaly
    """

    # equinoctial elements
    p, f, g, h, k, L = equ

    # classical elements
    a = p/(1 - f*f - g*g)
    
    e = np.sqrt(f*f + g*g)
    
    inc = 2*np.arctan(np.sqrt(h*h + k*k))
    
    sin_raan = k/np.tan(inc/2)
    cos_raan = h/np.tan(inc/2)
    raan = np.arctan2(sin_raan, cos_raan)
    if raan < 0:
        raan += 2*np.pi
    
    sin_w = (g*h - f*k)/e*np.tan(inc/2)
    cos_w = (f*h + g*k)/e*np.tan(inc/2)
    w = np.arctan2(sin_w, cos_w)
    if w < 0:
        w += 2*np.pi
    
    sin_nu = (f*np.sin(L) - g*np.cos(L))/e
    cos_nu = (f*np.cos(L) + g*np.sin(L))/e
    nu = np.arctan2(sin_nu, cos_nu)
    if nu < 0:
        nu += 2*np.pi
    
    eo = [a, e, inc, raan, w, nu]

    return eo
# ===================================================================


# =====================================================================
# 5) 
def true2mean(nu, a, e):
    """This function takes the true anomaly, semimajor axis, and 
    eccentricity of an orbit and returns the eccentric and mean 
    anomaly in radians.

    INPUT:
        a - semimajor axis of orbit in [km]
        e - eccentricity of the spacecraft 0<e<1
        nu - true anomoly 0<nu<2*pi [rad]

    OUTPUT:
        ea - eccentric anomaly 0<ea<2*pi [rad]
        ma - mean anomaly 0<ma<2*pi [rad]
    """

    r_mag = a*(1 - e*e)/(1 + e*np.cos(nu))

    # Computing Eccentric Anomaly
    cos_ea = (r_mag*np.cos(nu) + a*e)/a
    sin_ea = r_mag*np.sin(nu)/(a*np.sqrt(1 - e*e))
    ea = np.arctan2(sin_ea, cos_ea)

    if ea < 0:
        ea += 2*np.pi

    # Computing Mean Anomaly
    ma = ea - e*np.sin(ea)

    return ma, ea
# =====================================================================


# =====================================================================
# 6) 
def mean2true(ma, a, e, tol=1e-12, timeout=200):
    """This function takes the mean anomaly, semimajor axis, and 
    eccentricity of an orbit and returns the eccentric and true 
    anomaly in radians using a Newton-Raphson iteration.

    INPUT:
        ma - mean anomaly 0<M<2*pi [rad]
        a - semimajor axis of orbit in [km]
        e - eccintricity of the spacecraft 0<e<1
        tol - desired tolerance for convergence
        timeout - number of iterations to attempt before 
                  breaking loop

    OUTPUT:
        ea - eccentric anomaly 0<EA<2*pi [rad]
        nu - true anomoly 0<nu<2*pi [rad]
    """

    # Newton-Raphson Loop
    ea_temp = []
    
    # initial guess
    if -np.pi < ma < 0 or ma > np.pi:
        ea_temp.append(ma - e) 
    else:
        ea_temp.append(ma + e)
    
    err = 1
    i = 0
    while err > tol:
        ea_temp.append(ea_temp[i] + (ma - ea_temp[i] + 
            e*np.sin(ea_temp[i]))/(1 - e*np.cos(ea_temp[i])))
        err = abs(ea_temp[i+1] - ea_temp[i])
        i += 1
        if i > timeout:
            print('ERROR: eccentric anomaly did not converge!')
            ea_temp = [float('NaN')]
            break

    ea = ea_temp[-1]
    r = a*(1 - e*np.cos(ea))

    cos_nu = (a*np.cos(ea) - a*e)/r
    sin_nu = (a*np.sqrt(1 - e*e)/r)*np.sin(ea)

    nu = np.arctan2(sin_nu, cos_nu)

    if nu < 0:
        nu += 2*np.pi

    return nu, ea
# =====================================================================


# =====================================================================
# 7) 
def hohmann(a_in, a_fin, r_in, r_fin, mu=c.mu_earth):
    """This function finds a Hohmann transfer between two points on 
    two planar orbits.

    INPUT:
        a_in - The semimajor axis of the initial orbit in [km]
        a_fin - The semimajor axis of the final orbit in [km]
        r_in - The radius of the SC on the initial orbit at the time 
               of burn in [km]
        r_fin - the radius of the SC on the final orbit after the 
                transfer in [km]
        mu - gravitaional parameter of main body [km^3/s^2]

    OUTPUT:
        a_trans - The semimajor axis of the transfer ellipse in [km]
        t_trans - The time of transfer in [sec]
        dv_a - The dV of the first burn [km/s]
        dv_b - The dV of the second burn [km/s]
    """

    # Semimajor axis of transfer orbit
    a_trans = (r_in + r_fin)/2.0

    # Velocity Changes
    v_in = np.sqrt(2*mu/r_in - mu/a_in)
    v_fin = np.sqrt(2* mu/r_fin - mu/a_fin)
    v_trans_a = np.sqrt(2*mu/r_in - mu/a_trans)
    v_trans_b = np.sqrt(2*mu/r_fin - mu/a_trans)

    # DeltaV
    dv_a = v_trans_a - v_in
    dv_b = v_fin - v_trans_b

    t_trans = np.pi*np.sqrt(a_trans*a_trans*a_trans/mu)

    return a_trans, t_trans, dv_a, dv_b
# =====================================================================


# =====================================================================
# 8) 
def theta_gst(jd):
    """This function calculates theta_GST for Earth for a given 
    Julian Date.

    INPUT:
        jd - Julian Date

    OUTPUT:
        theta_gst - Angle between ECEF and ECI coordinate frames 
                    in [rads]
    """

    t_ut1 = (jd - 2451545)/36525.0

    gst = (67310.54841 + (876600*60*60 + 8640184.812866)*t_ut1 + 
           0.093104*t_ut1*t_ut1 - 6.2e-6*t_ut1*t_ut1*t_ut1)

    while abs(gst) > 86400:
        if gst < 0:
            gst += 86400
        else:
            gst -= 86400

    if gst < 0:
        gst = gst/240.0*(np.pi/180) + 2*np.pi
    else:
        gst = gst/240.0 * (np.pi/180)

    return gst
# =====================================================================


# =====================================================================
# 9) 
def julian_date(yr, mo, d, hr, m, s):
    """This function take the Gregorian Date of a specific date 
    between March 1st, 1900 and February 28th, 2100 and returns 
    the Julian Date.

    INPUT:
        yr - four digit year
        mo - two digit month
        d - day
        hr - hours in military time
        m - minutes
        s - seconds

    OUTPUT:
        jd - Julian Date
    """

    jd = (367.0*yr - np.floor(7.0*(yr + 
        np.floor((mo + 9.0)/12.0))/4.0) + np.floor(275.0*mo/9.0) + 
        d + 1721013.5 + ((s/60.0 + m)/60.0 + hr)/24.0)

    return jd
# =====================================================================


# =====================================================================
# 10) 
def gregorian_date(jd):
    """This function take the Julian Date of a specific date between 
    March 1st, 1900 and February 28th, 2100 and returns the Gregorian 
    Date. The Julian Date is converted into the Gregorian Date at the 
    Greenwich Meridain described in Greenwich Mean Time (GMT)

    INPUT:
        jd - Julian Date

    OUTPUT:
        yr - four digit year
        mo - two digit month
        d - day
        hr - hours in military time
        m - minutes
        s - seconds
    """

    t_1900 = (jd - 2415019.5)/365.25
    yr = 1900 + np.floor(t_1900)
    LeapYrs = np.floor((yr - 1900 - 1)/4.0)
    days = (jd - 2415019.5) - ((yr - 1900)*365 + LeapYrs)

    if days < 1.0:
        yr -= 1
        LeapYrs = np.floor((yr - 1900 - 1) * .25)
        days = (jd - 2415019.5) - ((yr - 1900)*365 + LeapYrs)

    LMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if np.mod(yr, 4) == 0:
        LMonth[1] = 29

    DayofYr = np.floor(days)
    index = 0
    Month_sum = LMonth[0]

    while Month_sum < DayofYr:
        index += 1
        Month_sum = Month_sum + LMonth[index]

    Month = Month_sum - LMonth[index]
    mon = index+1
    d = DayofYr - Month
    tow = (days - DayofYr) * 24
    hr = np.floor(tow)
    m = np.floor((tow - hr) * 60)
    s = (tow - hr - m / 60) * 3600

    date = (str(int(mon)) + '/' + str(int(d)) + '/' + str(int(yr)) 
           + ' ' + str(int(hr)) + ':' + str(int(m)) + ':' + str(s))

    return date
# =====================================================================


# =====================================================================
# 11) 
def ymd2dayofyear(yr, mon, d, hr, m, s):
    """The function take the date and returns the day of the year 
    in decimal form.

    INPUT:
        yr - four digit year
        mo - two digit month
        d - day
        hr - hours in military time
        m - minutes
        s - seconds

    OUTPUT:
        days - day of year in decimal form
    """

    LMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if np.mod(yr, 4) == 0:
        LMonth[1] = 29

    if mon == 1:
        month_sum = 0
    else:
        month_sum = sum(LMonth[0:mon-1])

    DayofYr = month_sum + d
    days = DayofYr + hr/24.0 + m/1440.0 + s/86400.0

    return days
# =====================================================================


# =====================================================================
# 12) 
def dayofyear2ymd(days, yr):
    """The function take the day of the year and returns the month, 
    day, hour, minute, and second.

    INPUT:
        days - day of year in decimal form
        yr - four digit year

    OUTPUT:
        mo - two digit month
        d - day
        hr - hours in military time
        m - minutes
        s - seconds
    """
    LMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if np.mod(yr, 4) == 0:
        LMonth[1] = 29

    DayofYr = np.floor(days)
    index = 0
    Month_sum = LMonth[0]

    while Month_sum + 1 < DayofYr:
        Month_sum = Month_sum + LMonth[index + 1]
        index += 1

    Month = Month_sum - LMonth[index]
    mon = index+1
    d = DayofYr - Month
    tow = (days - DayofYr) * 24
    hr = np.floor(tow)
    m = np.floor((tow - hr) * 60)
    s = (tow - hr - m / 60) * 3600

    date = (str(int(mon)) + '/' + str(int(d)) + '/' + str(int(yr)) 
           + ' ' + str(int(hr)) + ':' + str(int(m)) + ':' + str(s))

    return date
# =====================================================================


# =====================================================================
# 13) 
def circ_redez(a_target, theta, r_body=c.r_earth, mu=c.mu_earth, 
    k_target=1, k_int=1):
    """This function compute the rendezvous between two satellits in 
    the same coplanar cicular orbit sepreated by some angle.

    INPUT:
        a_target - semimajor axis of orbit
        theta - angle from target to interceptor in [rads]
        r_planet - radius of body (set to Earth) in [km]
        mu - gravitational parameter of body (set to Earth) 
             in [km^3/s^2]
        k_target - number of revolutions of the taret
        k_int - number of revolutions of the interceptor

    OUTPUT:
        t_trans - time duration of phasing orbit in [sec]
        a_trans - semimajor axis of phasing orbit in [km]
        e_trans - eccentricity of phasing orbit
        dV - Velocity change of one burn (burns will be same 
             magnitude
        but oppisite direction)
    """

    # Calculating Mean Motion
    n_target = np.sqrt(mu/a_target/a_target/a_target)

    if theta > 0:
        # Calculating Transfer time
        t_trans = (2*np.pi*k_target + theta)/n_target

        # Calculating the semimajor axis of the transfer orbit
        a_trans = (mu*(t_trans/(2*np.pi*k_int))**2)**(1/3.0)

        rp_trans = a_target
        e_trans = 1 - rp_trans/a_trans

    else:
        # Calculating Transfer time
        t_trans = (2*np.pi*k_target + theta)/n_target

        # Calculating the semimajor axis of the transfer orbit
        a_trans = (mu*(t_trans/(2*np.pi*k_int))**2)**(1/3.0)

        ra_trans = a_target
        e_trans = ra_trans/a_trans - 1
        rp_trans = a_trans*(1 - e_trans)

    if rp_trans < r_body:
        print("ERROR! Your satellite will crash! [rp < r_body]")

    dV = np.sqrt(2*mu/a_target - mu/a_trans) - np.sqrt(mu/a_target)

    return t_trans, dV, a_trans, e_trans
# =====================================================================


# ===================================================================== 
# 14)
def hill_eqns(xi, yi, zi, vxi, vyi, vzi, n, dt):
    """This function is used for close proximity determination. The 
    CW/Hill equations are used to determine the position of a 
    satellite in orbit relative to another satellite.

    ASSUMPTIONS:
    - cicular orbits

    INPUT:
        xi - initial postion in radial direction from reference 
             s/c in [m]
        yi - initial postion in velocity direction from reference 
             s/c in [m]
        zi - initial postion in angular momentum direction from 
             reference s/c in [m]
        vxi - initial velocity in radial direction from reference 
              s/c in [m/s]
        vyi - initial velocity in velocity direction from reference 
              s/c in [m/s]
        vzi - initial velocity in angular momentum direction from 
              reference s/c in [m/s]
        n - mean motion (angular velocity) of the reference s/c
        dt - time space [sec]

    OUTPUT:
        x - postion in radial direction from reference s/c in [m] 
            after dt [sec]
        y - postion in velocity direction from reference s/c in [m] 
            after dt [sec]
        z - postion in angular momentum direction from reference s/c 
            in [m] after dt [sec]
        vx - velocity in radial direction from reference s/c in [m/s] 
             after dt [sec]
        vy - velocity in velocity direction from reference s/c in 
             [m/s] after dt [sec]
        vz - velocity in angular momentum direction from reference 
             s/c in [m/s] after dt [sec]
    """

    # CW / Hill Equations
    x = (vxi/n*np.sin(n*dt) - (3*xi + 2*vyi/n)*np.cos(n*dt) + 
        (4*xi + 2*vyi/n))
    y = ((6*xi + 4*vyi/n)*np.sin(n*dt) + 2*vxi/n*np.cos(n*dt) - 
        (6*n*xi + 3*vyi)*dt + (yi - 2*vxi/n))
    z = zi*np.cos(n*dt) + vzi/n*np.sin(n*dt)
    vx = vxi*np.cos(n*dt) + (3*n*xi + 2*vyi)*np.sin(n*dt)
    vy = ((6*n*xi + 4*vyi)*np.cos(n*dt) - 2*vxi*np.sin(n*dt) - 
         (6*n*xi + 3*vyi))
    vz = -zi*n*np.sin(n*dt) + vzi*np.cos(n*dt)

    return x, y, z, vx, vy, vz
# =====================================================================


# =====================================================================
# 15) 
def hill_rendez(xi, yi, zi, vxi, vyi, vzi, n, dt):
    """This function is used for close proximity determination. The 
    CW/Hill equations are used to determine the velocity needed to 
    return to the reference s/c.

    ASSUMPTIONS:
    - cicular orbits

    INPUT:
        xi - initial postion in radial direction from reference 
             s/c in [m]
        yi - initial postion in velocity direction from reference 
             s/c in [m]
        zi - initial postion in angular momentum direction from 
             reference s/c in [m]
        vxi - initial velocity in radial direction from reference 
              s/c in [m/s]
        vyi - initial velocity in velocity direction from reference 
              s/c in [m/s]
        vzi - initial velocity in angular momentum direction from 
              reference s/c in [m/s]
        n - mean motion (angular velocity) of the reference s/c
        dt - time space [sec]

    OUTPUT:
        vx - velocity in radial direction from reference s/c in 
             [m/s] after dt [sec]
        vy - velocity in velocity direction from reference s/c in 
             [m/s] after dt [sec]
        vz - velocity in angular momentum direction from reference 
             s/c in [m/s] after dt [sec]
    """

    # CW / Hill Equations
    vy_hill = ((6*xi*(n*dt - np.sin(n*dt)) - yi)*n*np.sin(n*dt) - 
        2*n*xi*(4 - 3*np.cos(n*dt))*(1 - np.cos(n*dt)))/(
        (4*np.sin(n*dt) - 3*n*dt)*np.sin(n*dt) + 
        4*(1 - np.cos(n*dt))**2)
    vx_hill = -(n*xi*(4 - 3*np.cos(n*dt)) + 2*vy*(1 - 
        np.cos(n*dt)))/np.sin(n*dt)
    vz_hill = -zi*n/np.tan(n*dt)

    vy = vy_hill - vyi
    vx = vx_hill - vxi
    vz = vz_hill - vzi

    return vx, vy, vz
# =====================================================================


# =====================================================================
# 16) 
def fpa(e, nu):
    """ This function calculates the flight path angle for an 
    elliptical orbit.

    INPUT:
        e - eccentricity of orbit
        nu - true anomaly 0<nu<2*pi [rad]

    OUTPUT:
        fpa - flight path angle of the spacecraft
    """

    cos_fpa = (1 + e*np.cos(nu))/np.sqrt(1 + 2*e*np.cos(nu) + e*e)
    sin_fpa = e*np.sin(nu)/np.sqrt(1 + 2*e*np.cos(nu) + e*e)

    flight_path_angle = np.arctan2(sin_fpa, cos_fpa)

    return flight_path_angle
# =====================================================================


# =====================================================================
# 17) 
def meeus(JD, planet):
    """ This function calculates the position and velocity of 
    planets wrt sun.

    INPUT:
        JD - Julian date
        planet - The desired planet (i.e., 'Earth')
            > Mercury
            > Venus
            > Earth
            > Mars
            > Jupiter
            > Saturn
            > Uranus
            > Neptune

    OUTPUT:
        a - semi-major axis [AU]
        e - eccentricity
        inc - inclination [rad]
        RAAN - right ascension of the ascending node [rad]
        w - arg of periapsis [rad]
        nu - true anomaly [rad]

    """

    # # Mercury
    # if planet.lower() == 'mercury':
    #     a0_L = 
    #     a1_L = 
    #     a2_L = 
    #     a3_L = 
    
    #     a0_a = 
    #     a1_a = 
    #     a2_a = 
    #     a3_a = 
    
    #     a0_e = 
    #     a1_e = 
    #     a2_e = 
    #     a3_e = 
    
    #     a0_inc = 
    #     a1_inc = 
    #     a2_inc = 
    #     a3_inc = 
    
    #     a0_RAAN = 
    #     a1_RAAN = 
    #     a2_RAAN = 
    #     a3_RAAN = 
    
    #     a0_PI = 
    #     a1_PI = 
    #     a2_PI = 
    #     a3_PI = 

    # Venus
    if planet.lower() == 'venus':
        a0_L = 181.979801
        a1_L = 58517.8156760
        a2_L = 0.00000165
        a3_L = -0.000000002
    
        a0_a = 0.72332982
        a1_a = 0
        a2_a = 0
        a3_a = 0
    
        a0_e = 0.00677188
        a1_e = -0.000047766
        a2_e = 0.0000000975
        a3_e = 0.00000000044
    
        a0_inc = 3.394662
        a1_inc = -0.0008568
        a2_inc = -0.00003244
        a3_inc = 0.000000010
    
        a0_RAAN = 76.679920
        a1_RAAN = -0.2780080
        a2_RAAN = -0.00014256
        a3_RAAN = -0.000000198
    
        a0_PI = 131.563707
        a1_PI = 0.0048646
        a2_PI = -0.00138232
        a3_PI = -0.000005332

    # Earth
    if planet.lower() == 'earth':
        a0_L = 100.466449
        a1_L = 35999.3728519
        a2_L = -0.00000568
        a3_L = 0.0
    
        a0_a = 1.000001018
        a1_a = 0
        a2_a = 0
        a3_a = 0
    
        a0_e = 0.01670862
        a1_e = -0.000042037
        a2_e = -0.0000001236
        a3_e = 0.00000000004
    
        a0_inc = 0
        a1_inc = 0.0130546
        a2_inc = -0.00000931
        a3_inc = -0.000000034
    
        a0_RAAN = 174.873174
        a1_RAAN = -0.2410908
        a2_RAAN = 0.00004067
        a3_RAAN = -0.000001327
    
        a0_PI = 102.937348
        a1_PI = 0.3225557
        a2_PI = 0.00015026
        a3_PI = 0.000000478

    # Mars
    if planet.lower() == 'mars':
        a0_L = 355.433275
        a1_L = 19140.2993313
        a2_L = 0.00000261
        a3_L = -0.000000003
    
        a0_a = 1.523679342
        a1_a = 0
        a2_a = 0
        a3_a = 0
    
        a0_e = 0.09340062
        a1_e = 0.000090483
        a2_e = -0.0000000806
        a3_e = -0.00000000035
    
        a0_inc = 1.849726
        a1_inc = -0.0081479
        a2_inc = -0.00002255
        a3_inc = -0.000000027
    
        a0_RAAN = 49.558093
        a1_RAAN = -0.2949846
        a2_RAAN = -0.00063993
        a3_RAAN = -0.000002143
    
        a0_PI = 336.060234
        a1_PI = 0.4438898
        a2_PI = -0.00017321
        a3_PI = 0.000000300

    # Jupiter
    if planet.lower() == 'jupiter':
        a0_L = 34.351484
        a1_L = 3034.9056746
        a2_L = -0.00008501
        a3_L = 0.000000004
    
        a0_a = 5.202603191
        a1_a = 0.0000001913
        a2_a = 0
        a3_a = 0
    
        a0_e = 0.04849485
        a1_e = 0.000163244
        a2_e = -0.0000004719
        a3_e = -0.00000000197
    
        a0_inc = 1.303270
        a1_inc = -0.0019872
        a2_inc = 0.00003318
        a3_inc = 0.000000092
    
        a0_RAAN = 100.464441
        a1_RAAN = 0.1766828
        a2_RAAN = 0.00090387
        a3_RAAN = -0.000007032
    
        a0_PI = 14.331309
        a1_PI = 0.2155525
        a2_PI = 0.00072252
        a3_PI = -0.000004590

    # Saturn
    if planet.lower() == 'saturn':
        a0_L = 50.077471
        a1_L = 1222.1137943
        a2_L = 0.00021004
        a3_L = -0.000000019
    
        a0_a = 9.554909596
        a1_a = -0.0000021389
        a2_a = 0
        a3_a = 0
    
        a0_e = 0.05550862
        a1_e = -0.000346818
        a2_e = -0.0000006456
        a3_e = 0.00000000338
    
        a0_inc = 2.488878
        a1_inc = 0.0025515
        a2_inc = -0.00004903
        a3_inc = 0.000000018
    
        a0_RAAN = 113.665524
        a1_RAAN = -0.2566649
        a2_RAAN = -0.00018345
        a3_RAAN = 0.000000357
    
        a0_PI = 93.056787
        a1_PI = 0.5665496
        a2_PI = 0.00052809
        a3_PI = 0.000004882

    # Uranus
    if planet.lower() == 'uranus':
        a0_L = 314.055005
        a1_L = 429.8640561
        a2_L = 0.00030434
        a3_L = 0.000000026
    
        a0_a = 19.218446062
        a1_a = -0.0000000372
        a2_a = 0.00000000098
        a3_a = 0.0
    
        a0_e = 0.04629590
        a1_e = -0.000027337
        a2_e = 0.0000000790
        a3_e = 0.00000000025
    
        a0_inc = 0.773196
        a1_inc = 0.0007744
        a2_inc = 0.00003749
        a3_inc = -0.000000092
    
        a0_RAAN = 74.005947
        a1_RAAN = 0.5211258
        a2_RAAN = 0.00133982
        a3_RAAN = 0.000018516
    
        a0_PI = 173.005159
        a1_PI = 1.4863784
        a2_PI = 0.0021450
        a3_PI = 0.000000433

    # Neptune
    if planet.lower() == 'neptune':
        a0_L = 304.348665
        a1_L = 219.8833092
        a2_L = 0.00030926
        a3_L = 0.000000018
    
        a0_a = 30.110386869
        a1_a = -0.0000001663
        a2_a = 0.00000000069
        a3_a = 0.0
    
        a0_e = 0.00898809
        a1_e = 0.000006408
        a2_e = -0.0000000008
        a3_e = -0.00000000005
    
        a0_inc = 1.769952
        a1_inc = -0.0093082
        a2_inc = -0.00000708
        a3_inc = 0.000000028
    
        a0_RAAN = 131.784057
        a1_RAAN = 1.1022057
        a2_RAAN = 0.00026006
        a3_RAAN = -0.000000636
    
        a0_PI = 48.123691
        a1_PI = 1.4262677
        a2_PI = 0.00037918
        a3_PI = -0.000000003

    # Pluto
    if planet.lower() == 'pluto':
        a0_L = 238.92903833
        a1_L = 145.20780515
        a2_L = 0.0
        a3_L = 0.0
    
        a0_a = 39.48211675
        a1_a = -0.00031596
        a2_a = 0.0
        a3_a = 0.0
    
        a0_e = 0.24882730
        a1_e = 0.00005170
        a2_e = 0.0
        a3_e = 0.0
    
        a0_inc = 17.14001206
        a1_inc = 0.00004818
        a2_inc = 0.0
        a3_inc = 0.0
    
        a0_RAAN = 110.30393684
        a1_RAAN = -0.01183482
        a2_RAAN = 0.0
        a3_RAAN = 0.0
    
        a0_PI = 224.06891629
        a1_PI = -0.04062942
        a2_PI = 0.0
        a3_PI = 0.0

    # Orbital Elements
    T = (JD - 2451545.0)/36525.0

    # Mean longitude
    L = a0_L + a1_L*T + a2_L*T*T + a3_L*T*T*T                # [deg]
   
    a = a0_a + a1_a*T + a2_a*T*T + a3_a*T*T*T                # [AU]
    e = a0_e + a1_e*T + a2_e*T*T + a3_e*T*T*T                # [-]
    inc = a0_inc + a1_inc*T + a2_inc*T*T + a3_inc*T*T*T      # [deg]
    RAAN = a0_RAAN + a1_RAAN*T + a2_RAAN*T*T + a3_RAAN*T*T*T # [deg]

    # Longitude of the perihelion
    PI = a0_PI + a1_PI*T + a2_PI*T*T + a3_PI*T*T*T

    # Converting units to km and rad
    L = np.deg2rad(L)
    a *= c.AU
    inc = np.deg2rad(inc)
    RAAN = np.deg2rad(RAAN)
    PI = np.deg2rad(PI)

    # arg of periapsis
    w = PI - RAAN
    
    # Mean anomaly
    M = L - PI

    C_cen = ((2*e - (e**3)/4.0 + 5*(e**5)/96.0)*np.sin(M) + 
        (5*(e**2)/4.0 - 11*(e**4)/24.0)*np.sin(2*M) + 
        (13*(e**3)/12.0 - 43*(e**5)/64.0)*np.sin(3*M) + 
        (103*(e**4)/96.0)*np.sin(4*M) + 
        (1097*(e**5)/960.0)*np.sin(5*M))

    # True anomaly
    nu = M + C_cen

    # Truncating terms
    inc = np.mod(inc, 2*np.pi)
    RAAN = np.mod(RAAN, 2*np.pi)
    w = np.mod(w, 2*np.pi)
    nu = np.mod(nu, 2*np.pi)

    return a, e, inc, RAAN, w, nu
# =====================================================================


# =====================================================================
# 18) 
def lambert_uv(r0, rf, dt, DM=1, mu=c.mu_sun):
    """This function solves the 0-rev lambert problem using the 
    Universal Variables Lambert Algorithm

    INPUT:
        r0 - initial position vector [km]
        rf - final position vector [km]
        dt - time of flight [sec]
        DM - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        mu - gravitational parameter of central body 
             (default is sun) [km^3/s^2]

    OUTPUT:
        v0 - velocity at r0 [km/s]
        vf - velocity at rf [k/s]
        psi_n - value of psi converged
        orb_type - type I/II converged
    """
    
    # Tolerances
    tol_psi = 1e-6
    tol_dt = 1e-6

    # Function Checks
    run = True
    check_dt = 0
    check_traj = 0

    # Check Transfer Time
    if dt < 0:
        print('ERROR: Transfer Time is Negative')
        check_dt = 1

    # Calculating direction of motion
    if DM == 1:
        # Short way
        orb_type = 1

    elif DM == -1:
        # Long way
        orb_type = 2

    else:
        # This only works if both orbits are mostly in the xy-plane
        # such as the case for IMD

        # Calculating delta_nu
        nu1 = np.arctan2(r0[1], r0[0])
        if nu1 < 0:
            nu1 += 2*np.pi
    
        nu2 = np.arctan2(rf[1], rf[0])
        if nu2 < 0:
            nu2 += 2*np.pi
    
        delta_nu = nu2 - nu1
        if delta_nu < 0:
            delta_nu += 2*np.pi
    
        # DM = Direction of Motion (+1 short way, -1 longway)
        # Direction of motion dictated by delta_nu
        if delta_nu > np.pi:
            DM = -1
            orb_type = 2
        else:
            DM = 1
            orb_type = 1

    # Intializing
    r0_mag = np.sqrt(r0.dot(r0))
    rf_mag = np.sqrt(rf.dot(rf))
    
    cos_delnu = np.dot(r0, rf)/abs(r0_mag*rf_mag)
    A = DM*np.sqrt(r0_mag*rf_mag*(1.0 + cos_delnu))

    c2 = 1/2.0
    c3 = 1/6.0

    # Checking possibility of trajectory
    if np.arccos(cos_delnu) == 0 or A == 0:
        print('ERROR: No Possible Lambert Trajectories')
        check_traj = 1

    # Setting initia PSI
    psi_high = 4*np.pi*np.pi
    psi_low = -4*np.pi
    psi_n = 0

    if check_dt == 0 and check_traj == 0:
        loop_count = 0
        
        while run:
            loop_count += 1
    
            if loop_count == 1000:
                # print 'WARNING: Tolerance increased to 1e-5'
                tol_dt = 1e-5
            elif loop_count == 10000:
                # print 'WARNING: Tolerance increased to 1e-4'
                tol_dt = 1e-4
            elif loop_count == 30000:
                # print 'WARNING: Tolerance increased to 1e-3'
                tol_dt = 1e-3
            elif loop_count == 60000:
                # print 'WARNING: Tolerance increased to 1e-2'
                tol_dt = 1e-2
            elif loop_count == 100000:
                print('ERROR: No Convergence for TOF')
                v0 = np.float('NaN')
                vf = np.float('NaN')
                psi_n = np.float('NaN')
                break
    
            y = r0_mag + rf_mag + A*(psi_n*c3 - 1.0)/np.sqrt(c2)
    
            # readjusting psi_low until y > 0
            if A > 0 and y < 0:
                while y < 0:
                    psi_n += 0.1
                    y = (r0_mag + rf_mag + 
                        A*(psi_n*c3 - 1.0)/np.sqrt(c2))
    
            Xi = np.sqrt(y/c2)
            dt_n = ((Xi**3.0)*c3 + A*np.sqrt(y))/np.sqrt(mu)
    
            if dt_n <= dt: 
                psi_low = psi_n
            else:
                psi_high = psi_n
            
            psi_n = (psi_low + psi_high)/2.0

            if psi_n > tol_psi:
                c2 = (1.0 - np.cos(np.sqrt(psi_n)))/psi_n
                c3 = ((np.sqrt(psi_n) - 
                    np.sin(np.sqrt(psi_n)))/np.sqrt(psi_n**3.0))
            elif psi_n < -tol_psi:
                c2 = (1.0 - np.cosh(np.sqrt(-psi_n)))/psi_n
                c3 = ((np.sinh(np.sqrt(-psi_n)) - 
                    np.sqrt(-psi_n))/np.sqrt((-psi_n)**3.0))
            else:
                c2 = 1/2.0
                c3 = 1/6.0
    
            if abs(dt_n - dt) < tol_dt:
                break
    
        f = 1.0 - y/r0_mag
        gdot = 1.0 - y/rf_mag
        g = A*np.sqrt(y/mu)
    
        v0 = (rf - f*r0)/g
        vf = (gdot*rf - r0)/g
    
    else:
        print("Lambert did not work")
        v0 = float('NaN')
        vf = float('NaN')
        psi_n = float('NaN')

    return v0, vf, psi_n, orb_type
# =====================================================================


# =====================================================================
# 19) 

def lambert_pc(r0, rf, dt, DM=1, mu=c.mu_sun):
    """This function solves the 0-rev lambert problem using the 
    method given by Prussing and Conway. 

    INPUT:
        r0 - initial position vector [km]
        rf - final position vector [km]
        dt - time of flight [sec]
        DM - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        mu - gravitational parameter of central body 
             (default is sun) [km^3/s^2]

    OUTPUT:
        v0 - velocity at r0 [km/s]
        vf - velocity at rf [km/s]
        alpha - parameter
        beta - parameter
        a_solve - semimajor axis of transfer orbit [km]

    SUPPORTING FUNCTIONS:
        f1 - time equation used to determine SMA of transfer orbit
        f2 - time equation used to determine SMA of transfer orbit
             shifted by 2pi
    """

    # Function Checks
    check_dt = 0

    # Calculating direction of motion
    if DM == 1:
        # Short way
        orb_type = 1

    elif DM == -1:
        # Long way
        orb_type = 2

    else:
        # This only works if both orbits are mostly in the xy-plane
        # such as the case for IMD

        # Calculating delta_nu
        nu1 = np.arctan2(r0[1], r0[0])
        if nu1 < 0:
            nu1 += 2*np.pi
    
        nu2 = np.arctan2(rf[1], rf[0])
        if nu2 < 0:
            nu2 += 2*np.pi
    
        delta_nu = nu2 - nu1
        if delta_nu < 0:
            delta_nu += 2*np.pi
    
        # DM = Direction of Motion (+1 short way, -1 longway)
        # Direction of motion dictated by delta_nu
        if delta_nu > np.pi:
            DM = -1
            orb_type = 2
        else:
            DM = 1
            orb_type = 1

    # Position vector magnitudes
    r0_mag = np.sqrt(r0.dot(r0))
    rf_mag = np.sqrt(rf.dot(rf))

    # Comupte chord length
    c_vec = rf - r0
    c = np.sqrt(c_vec.dot(c_vec))

    # Compute space triangle semi-parameter
    s = 0.5*(r0_mag + rf_mag + c)

    # Compute desired transfer angle
    theta = np.arccos(np.dot(r0, rf)/abs(r0_mag*rf_mag))
    if DM == -1:
        theta = 2*np.pi - theta

    # Check to make sure dt allows for an elliptic transfer
    dt_parabolic = np.sqrt(2)/3/np.sqrt(mu)*(s**(3/2) - 
        np.sign(np.sin(theta))*(s - c)**(3/2))
    if dt <= dt_parabolic:
        print('ERROR: This function only works on ellitpical orbits. Transfer time requires a hyperbolic orbit.')
        check_dt = 1

    if check_dt == 0:
        # Choose sign for beta based on theta
        if theta < np.pi:
            betasign = 1
        else:
            betasign = -1

        # Calculate minimum SMA axis and associated transfer time
        a_min = s/2
        beta_min = 2*np.arcsin(np.sqrt((s - c)/s))
        t_min = np.sqrt(s*s*s/8/mu)*(np.pi - betasign*beta_min + np.sin(betasign*beta_min))
        if dt <= t_min:
            fun = f1
        else:
            fun = f2

        data = [c, s, betasign, dt, mu]
        a_solve = fsolve(fun, a_min, args=data)

        if dt <= t_min:
            alpha = 2*np.arcsin(np.sqrt(s/2/a_solve))
        else:
            alpha = 2*np.pi - 2*np.arcsin(np.sqrt(s/2/a_solve))

        beta = betasign*2*np.arcsin(np.sqrt((s - c)/2/a_solve))

        # Solve for terminal velocites
        u1 = r0/r0_mag
        u2 = rf/rf_mag
        uc = c_vec/c 

        temp = np.sqrt(mu/4/a_solve)
        A = temp/np.tan(alpha/2)
        B = temp/np.tan(beta/2)

        v0 = (B + A)*uc + (B - A)*u1
        vf = (B + A)*uc - (B - A)*u2

    return v0, vf, alpha, beta, a_solve

# Supporting Functions
def f1(a, data):
    c, s, betasign, dt, mu = data
    return a**(3/2)/np.sqrt(mu)*(2*np.arcsin(np.sqrt(s/2/a)) - 
        betasign*2*np.arcsin(np.sqrt((s - c)/2/a)) - 
        (np.sin(2*np.arcsin(np.sqrt(s/2/a))) - 
        np.sin(betasign*2*np.arcsin(np.sqrt((s - c)/2/a))))) - dt

def f2(a, data):
    c, s, betasign, dt, mu = data
    return a**(3/2)/np.sqrt(mu)*(2*np.pi - 2*np.arcsin(np.sqrt(s/2/a)) - 
        betasign*2*np.arcsin(np.sqrt((s - c)/2/a)) - 
        (np.sin(2*np.pi - 2*np.arcsin(np.sqrt(s/2/a))) - 
        np.sin(betasign*2*np.arcsin(np.sqrt((s - c)/2/a))))) - dt
# =====================================================================


# =====================================================================
# 20)
def lambert_multirev(r0, rf, dt, type, DM=1, mu=c.mu_sun):
    """ This function solves the lambert problem using the Universal 
    Variables Lambert Algorithm

    INPUT:
        r0 - initial position vector [km]
        rf - final position vector [km]
        dt - time of flight [sec]
        DM - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        type - type I, II, III, IV, V, VI, etc...
        mu - gravitational parameter of central body (default is sun) 
             [km^3/s^2]

    OUTPUT:
        v0 - velocity at r0 [km/s]
        vf - velocity at fr [k/s]
        psi_n - value of psi converged
        revs - number of revolutions
        
    NOTES:
        Algorithm is a bit slow. Could lower tolerances and feed 
        psi_bound in manually to speed up.
    """

    # Tolerances
    tol_psi = 1e-6
    tol_dt = 1e-6

    # Function Checks
    true = True
    check_dt = 0
    check_traj = 0
    check_type = 0

    # Check Transfer Time
    if dt < 0:
        print('ERROR: Transfer Time is Negative')
        check_dt = 1

    # Calculating direction of motion
    if DM == 1:
        # Short way
        orb_type = 1

    elif DM == -1:
        # Long way
        orb_type = 2

    else:
        # This only works if both orbits are mostly in the xy-plane
        # such as the case for IMD

        # Calculating delta_nu
        nu1 = np.arctan2(r0[1], r0[0])
        if nu1 < 0:
            nu1 += 2*np.pi
    
        nu2 = np.arctan2(rf[1], rf[0])
        if nu2 < 0:
            nu2 += 2*np.pi
    
        delta_nu = nu2 - nu1
        if delta_nu < 0:
            delta_nu += 2*np.pi
    
        # DM = Direction of Motion (+1 short way, -1 longway)
        # Direction of motion dictated by delta_nu
        if delta_nu > np.pi:
            DM = -1
            orb_type = 2
        else:
            DM = 1
            orb_type = 1

    # Intializing
    r0_mag = np.sqrt(r0.dot(r0))
    rf_mag = np.sqrt(rf.dot(rf))
    
    cos_delnu = np.dot(r0, rf)/abs(r0_mag*rf_mag)
    A = DM*np.sqrt(r0_mag*rf_mag*(1.0 + cos_delnu))

    c2 = 1/2.0
    c3 = 1/6.0

    # Checking possibility of trajectory
    if np.arccos(cos_delnu) == 0 or A == 0:
        print('ERROR: Trajectories cannot be computed')
        check_traj = 1

    # Determining Revs
    if type <= 2:
        revs = 0
    else:
        revs = np.floor((type - 1)/2.0)

    if revs == 0:
        psi_high = 4*np.pi*np.pi
        psi_low = -4*np.pi
        psi_n = 0
    
    else:
        psi_h = 4*((revs + 1)*np.pi)**2
        psi_l = 4*(revs*np.pi)**2
        
        # Need to find the minimum value of Psi
        psi_test = np.linspace(psi_l+1e-3, psi_h-1e-3, 1e3)
        TOF_test = []
        
        for psi in psi_test:
            a = 1
            # Finding c2, c3
            if psi > tol_psi:
                c2_test = (1.0 - np.cos(np.sqrt(psi)))/psi
                c3_test = ((np.sqrt(psi) - 
                    np.sin(np.sqrt(psi)))/np.sqrt(psi**3.0))
            elif psi < -tol_psi:
                c2_test = (1.0 - np.cosh(np.sqrt(-psi)))/psi
                c3_test = ((np.sinh(np.sqrt(-psi)) - 
                    np.sqrt(-psi))/np.sqrt((-psi)**3.0))
            else:
                c2_test = 1/2.0
                c3_test = 1/6.0
            
            y_test = (r0_mag + rf_mag + 
                A*(psi*c3_test - 1.0)/np.sqrt(c2_test))
            Xi_test = np.sqrt(y_test/c2_test)
            TOF_test.append(((Xi_test**3.0)*c3_test + 
                A*np.sqrt(y_test))/np.sqrt(mu))

        TOF_min = min(TOF_test)
        index = TOF_test.index(TOF_min)
        psi_bound = psi_test[index]

        if type % 2 == 0: # even type, positive slope
            psi_high = 4*((revs + 1)*np.pi)**2
            psi_low = psi_bound
        elif type % 2 == 1: # odd type, negative slope
            psi_high = psi_bound
            psi_low = 4*(revs*np.pi)**2
        else:
            print('ERROR: Type not defined well')
            check_type = 1

        psi_n = (psi_high + psi_low)/2.0
    
    if check_dt == 0 and check_traj == 0 and check_type == 0:
        loop_count = 0
        
        while true:
            loop_count += 1
    
            if loop_count == 1000:
                # print 'WARNING: Tolerance increased to 1e-5'
                tol_dt = 1e-5
            elif loop_count == 10000:
                # print 'WARNING: Tolerance increased to 1e-4'
                tol_dt = 1e-4
            elif loop_count == 30000:
                # print 'WARNING: Tolerance increased to 1e-3'
                tol_dt = 1e-3
            elif loop_count == 60000:
                # print 'WARNING: Tolerance increased to 1e-2'
                tol_dt = 1e-2
            elif loop_count == 100000:
                print('ERROR: No Convergence for TOF')
                v0 = np.float('NaN')
                vf = np.float('NaN')
                psi_n = np.float('NaN')
                break
    
            y = r0_mag + rf_mag + A*(psi_n*c3 - 1.0)/np.sqrt(c2)
    
            # readjusting psi_low until y > 0
            if A > 0 and y < 0:
                while y < 0:
                    psi_n += 0.1
                    y = (r0_mag + rf_mag + 
                        A*(psi_n*c3 - 1.0)/np.sqrt(c2))
    
            Xi = np.sqrt(y/c2)
            dt_n = ((Xi**3.0)*c3 + A*np.sqrt(y))/np.sqrt(mu)
    
            if type % 2 == 0 or revs == 0: # even type, positive slope
                if dt_n <= dt: 
                    psi_low = psi_n
                else:
                    psi_high = psi_n
            elif type % 2 == 1:  # odd type, negative slope:
                if dt_n >= dt: 
                    psi_low = psi_n
                else:
                    psi_high = psi_n
            
            psi_n = (psi_low + psi_high)/2.0

            if psi_n > tol_psi:
                c2 = (1.0 - np.cos(np.sqrt(psi_n)))/psi_n
                c3 = ((np.sqrt(psi_n) - 
                    np.sin(np.sqrt(psi_n)))/np.sqrt(psi_n**3.0))
            elif psi_n < -tol_psi:
                c2 = (1.0 - np.cosh(np.sqrt(-psi_n)))/psi_n
                c3 = ((np.sinh(np.sqrt(-psi_n)) - 
                    np.sqrt(-psi_n))/np.sqrt((-psi_n)**3.0))
            else:
                c2 = 1/2.0
                c3 = 1/6.0
    
            if abs(dt_n - dt) < tol_dt:
                break
    
        f = 1.0 - y/r0_mag
        gdot = 1.0 - y/rf_mag
        g = A*np.sqrt(y/mu)
    
        v0 = (rf - f*r0)/g
        vf = (gdot*rf - r0)/g

    else:
        print("Lambert did not work")
        v0 = float('NaN')
        vf = float('NaN')
        psi_n = float('NaN')

    return v0, vf, psi_n, revs
# =====================================================================
