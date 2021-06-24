
###########################
##   Lambert Functions   ##
###########################

"""This script provides a variety of useful functions revolving around 
Lambert's theorem.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

    1) lambert_uv (universal variables lambert problem solver)
    2) lambert_PC (Prussing and Conway lambert problem solver)
    3) lambert_uv_multirev (multirev lambert problem solver)
    4) lambert_izzo (Lambert Solver by Dario Izzo)
    5) long_short_way (deterimes lambert direction)
    6) lambertOptTransfer (find total dV optimal lambert transfer)
    7) 
    8) 
    9) 
    10) 

"""

# Import Modules
import numpy as np
import copy as copy
import constants as c
import scipy.optimize as so
import astro_functions as af
import matplotlib.pyplot as plt
import optimization_functions as of
from mpl_toolkits.mplot3d import Axes3D

# Define Functions
# =====================================================================
# 1) 
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
        # print('ERROR: No Possible Lambert Trajectories', A, cos_delnu)
        check_traj = 1

    # Setting initia PSI
    psi_high = 4*np.pi*np.pi
    psi_low = -4*np.pi
    psi_n = 0

    if check_dt == 0 and check_traj == 0:
        loop_count = 0
        
        while run:
            loop_count += 1

            # if loop_count == 1000:
            #     # print 'WARNING: Tolerance increased to 1e-5'
            #     tol_dt = 1e-5
            # elif loop_count == 10000:
            #     # print 'WARNING: Tolerance increased to 1e-4'
            #     tol_dt = 1e-4
            # elif loop_count == 30000:
            #     # print 'WARNING: Tolerance increased to 1e-3'
            #     tol_dt = 1e-3
            # elif loop_count == 60000:
            #     # print 'WARNING: Tolerance increased to 1e-2'
            #     tol_dt = 1e-2
            # elif loop_count == 100000:
            #     print('ERROR: No Convergence for TOF')
            #     run = False
    
            if loop_count == 100:
                # print 'WARNING: Tolerance increased to 1e-5'
                tol_dt = 1e-5
            elif loop_count == 300:
                # print 'WARNING: Tolerance increased to 1e-4'
                tol_dt = 1e-4
            elif loop_count == 500:
                # print 'WARNING: Tolerance increased to 1e-3'
                tol_dt = 1e-3
            elif loop_count == 1000:
                # print('ERROR: No Convergence for TOF')
                run = False
    
            y = r0_mag + rf_mag + A*(psi_n*c3 - 1.0)/np.sqrt(c2)
    
            # readjusting psi_low until y > 0
            if A > 0 and y < 0:
                while y < 0:
                    psi_n += 0.1
                    y = (r0_mag + rf_mag + 
                        A*(psi_n*c3 - 1.0)/np.sqrt(c2))

            if (y/c2) < 0:
                run = False
                break

            else:
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
    
        if run:
            f = 1.0 - y/r0_mag
            gdot = 1.0 - y/rf_mag
            g = A*np.sqrt(y/mu)
        
            v0 = (rf - f*r0)/g
            vf = (gdot*rf - r0)/g

        else:
            v0 = np.zeros(3)*float('NaN')
            vf = np.zeros(3)*float('NaN')
            psi_n = float('NaN')
    
    else:
        # print("Lambert did not work")
        v0 = np.zeros(3)*float('NaN')
        vf = np.zeros(3)*float('NaN')
        psi_n = float('NaN')

    return v0, vf, psi_n, orb_type
# =====================================================================

# =====================================================================
# 2) 
# -----------------------------------------------------------------
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
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Subfunctions

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
# -----------------------------------------------------------------
# =====================================================================

# =====================================================================
# 3) 
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

# =====================================================================
# 4) 
# -----------------------------------------------------------------
def lambert_izzo(r0, rf, dt, DM=1, mu=c.mu_sun, N=0, branch="l"):
    """This function is a fast, capable lambert problem using a method 
    developed by Dario Izz0.
    
    USAGE:
        (v1, v2, a, p, theta, iter) = lambert(r0, rf, dt, mu, DM, N, branch)
    
    INPUTS:
        - r0     = Position vector at departure
        - rf     = Position vector at arrival
        - dt     = Transfer time [sec]
        - mu_C   = gravitational parameter (scalar, units have to be consistent with r0, dt units)
        - DM     = -1 if long way is chosen, 1 for short way
        - branch = "l" (letter L) if the left branch is selected in a problem where N > 0
        - N      = number of revolutions
    
    OUTPUTS:
        - v1     = Velocity at departure        (consistent units)
        - v2     = Velocity at arrival
        - a      = semi major axis of the solution
        - p      = semi latus rectum of the solution
        - theta  = transfer angle in rad
        - iter   = number of iteration made by the newton solver (usually 6)

    NOTES:
    This routine implements a new algorithm that solves Lambert's problem. The
    algorithm has two major characteristics that makes it favorable to other
    existing ones.
     
        1) It describes the generic orbit solution of the boundary condition
        problem through the variable X=log(1+cos(alpha/2)). By doing so the
        graphs of the time of flight become defined in the entire real axis and
        resembles a straight line. Convergence is granted within few iterations
        for all the possible geometries (except, of course, when the transfer
        angle is zero). When multiple revolutions are considered the variable is
        X=tan(cos(alpha/2)*pi/2).
        
        2) Once the orbit has been determined in the plane, this routine
        evaluates the velocity vectors at the two points in a way that is not
        singular for the transfer angle approaching to pi (Lagrange coefficient
        based methods are numerically not well suited for this purpose).
        
        As a result Lambert's problem is solved (with multiple revolutions
        being accounted for) with the same computational effort for all
        possible geometries. The case of near 180 transfers is also solved
        efficiently.
        
        We note here that even when the transfer angle is exactly equal to pi
        the algorithm does solve the problem in the plane (it finds X), but it
        is not able to evaluate the plane in which the orbit lies. A solution
        to this would be to provide the direction of the plane containing the
        transfer orbit from outside. This has not been implemented in this
        routine since such a direction would depend on which application the
        transfer is going to be used in.  
    """
    
    # Preliminary control on the function call
    if dt <= 0:
        print("ERROR: Negative time as input")
        v1 = float("NaN")*np.ones(3)
        v2 = float("NaN")*np.ones(3)
        a = float("NaN")
        p = float("NaN")
        theta = float("NaN")
        it = float("NaN")
        return v1, v2, a, p, theta, it
    
    """Increasing the tolerance does not bring any advantage as the
    precision is usually greater anyway (due to the rectification of 
    the tof graph) except near particular cases such as parabolas in 
    which cases a lower precision allow for usual convergence."""
    tol = 1e-11
    
    # Non dimensional units
    R = np.sqrt(r0.dot(r0))
    V = np.sqrt(mu/R)
    T = R/V
    
    # Working with non-dimensional radii and time-of-flight
    r0 = r0/R
    rf = rf/R
    dt = dt/T
    
    # Evaluation of the relevant geometry parameters in non dimensional units
    rfmod = np.sqrt(rf.dot(rf))
    theta = np.real(np.arccos(r0.dot(rf)/rfmod)) 
    # the real command is useful when theta is very close to pi and the acos function could return complex numbers
    
    if DM == -1:
        theta = 2*np.pi - theta

    c = np.sqrt(1 + rfmod*rfmod - 2*rfmod*np.cos(theta))  # non dimensional chord
    s = (1 + rfmod + c)/2                                 # non dimensional semi-perimeter
    am = s/2                                              # minimum energy ellipse semi major axis
    lmbd = np.sqrt(rfmod)*np.cos(theta/2)/s               # lambda parameter defined in BATTIN's book
    
    # We start finding the log(x+1) value of the solution conic
    ## NO MULTI REV --> (1 SOL)
    if N == 0:
        inn1 = -0.5233    #first guess point
        inn2 = 0.5233     #second guess point
        x1 = np.log(1 + inn1)
        x2 = np.log(1 + inn2)
        y1 = np.log(x2tof(inn1, s, c, DM, N)) - np.log(dt)
        y2 = np.log(x2tof(inn2, s, c, DM, N)) - np.log(dt)
    
        # Newton iterations
        err = 1; i = 0
        while (err > tol) and (i < 60) and (y1 != y2):
            i += 1
            xnew = (x1*y2 - y1*x2)/(y2 - y1)
            ynew = np.log(x2tof(np.exp(xnew)-1, s, c, DM, N)) - np.log(dt)
            x1 = x2
            y1 = y2
            x2 = xnew
            y2 = ynew
            err = abs(x1 - xnew)

        it = i
        x = np.exp(xnew) - 1
    
        if it == 60: 
            v1 = float("NaN")*np.ones(3)
            v2 = float("NaN")*np.ones(3)
            return v1, v2, a, p, theta, it
    
        ##MULTI REV --> (2 SOL) SEPARATING RIGHT AND LEFT BRANCH
    else:
        if branch == "l":
            inn1 = -0.5234
            inn2 = -0.2234
        else:
            inn1 = 0.7234
            inn2 = 0.5234

        x1 = np.tan(inn1*np.pi/2)
        x2 = np.tan(inn2*np.pi/2)
        y1 = x2tof(inn1, s, c, DM, N) - dt
        y2 = x2tof(inn2, s, c, DM, N) - dt
    
        # Newton Iteration
        err=1; i=0
        while (err > tol) and (i < 60) and (y1 != y2):
            i += 1
            xnew = (x1*y2 - y1*x2)/(y2 - y1)
            ynew= x2tof(np.arctan(xnew)*2/np.pi, s, c, DM, N)-dt
            x1 = x2
            y1 = y2
            x2 = xnew
            y2 = ynew
            err = abs(x1 - xnew)

        it=i
        x = np.arctan(xnew)*2/np.pi
    
        if it == 60: 
            v1 = float("NaN")*np.ones(3)
            v2 = float("NaN")*np.ones(3)
            return v1, v2, a, p, theta, it
    
    """The solution has been evaluated in terms of log(x+1) or 
    tan(x*pi/2), we now need the conic. As for transfer angles near 
    to pi the lagrange coefficient technique goes singular (dg 
    approaches a zero/zero that is numerically bad) we here use a 
    different technique for those cases. When the transfer angle is 
    exactly equal to pi, then the ih unit vector is not determined. 
    The remaining equations, though, are still valid."""
    
    # calcolo psi
    a=am/(1-x*x)                       #solution semimajor axis
    
    # ellisse
    if x < 1: 
        beta = 2*np.arcsin(np.sqrt((s - c)/2/a))
        if DM == -1:
            beta *= -1
        alfa = 2*np.arccos(x)
        psi = (alfa - beta)/2
        eta2 = 2*a*np.sin(psi)**2/s
        eta = np.sqrt(eta2)
    
    # iperbole
    else:
        beta = 2*np.arcsinh(np.sqrt((c-s)/2/a))
        if DM == -1:
            beta *= -1
        alfa = 2*np.arccosh(x)
        psi = (alfa - beta)/2
        eta2 = -2*a*np.sinh(psi)**2/s
        eta = np.sqrt(eta2)

    p = rfmod/am/eta2*np.sin(theta/2)**2     #parameter of the solution
    sigma1 = 1/eta/np.sqrt(am)*(2*lmbd*am - (lmbd + x*eta))
    h = np.cross(r0,rf)
    ih = h/np.sqrt(h.dot(h))
    if DM == -1:
        ih *= -1
    
    vr0 = sigma1
    vt1 = np.sqrt(p)
    v1 = vr0*r0 + vt1*np.cross(ih,r0)
    
    vt2 = vt1/rfmod
    vrf = -vr0 + (vt1 - vt2)/np.tan(theta/2)
    v2 = vrf*rf/rfmod + vt2*np.cross(ih,rf/rfmod)
    v1 = v1*V
    v2 = v2*V
    a = a*R
    p = p*R

    return v1, v2, a, p, theta, it
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Subfunctions

def tofabn(sigma, alfa, beta, N):
    """ This subfunction evaluates the time of flight via Lagrange 
    expression.
    """

    if sigma > 0:
        dt = sigma*np.sqrt(sigma)*((alfa - np.sin(alfa)) - (beta - np.sin(beta)) + N*2*np.pi)
    else:
        dt = -sigma*np.sqrt(-sigma)*((np.sinh(alfa) - alfa) - (np.sinh(beta) - beta))
    
    return dt
    
def x2tof(x, s, c, DM, N):
    """ This subfunction evaluates the time of flight as a 
    function of x.
    """
    
    am=s/2
    a=am/(1-x*x)

    # ELLISSE
    if x < 1:
        alfa = 2*np.arccos(x)
        beta = 2*np.arcsin(np.sqrt((s-c)/2/a))
        if DM == -1:
            beta *= -1
    
    # IPERBOLE
    else:
        alfa = 2*np.arccosh(x)
        beta = 2*np.arcsinh(np.sqrt((s-c)/(-2*a)))   
        if DM == -1:
            beta *= -1
      
    return tofabn(a, alfa, beta, N)
# -----------------------------------------------------------------
# =====================================================================

# =====================================================================
# 5)
def long_short_way(r0, rf):
    """ This function Determines long way or short way for Lambert's 
    problem solutions. Chooses whichever solution gives a prograde orbit
    
    USAGE:
        DM = long_short_way(r0, rf)
    
    INPUTS:
        - r0 - Position vector at departure
        - rf - Position vector at arrival
    
    OUTPUTS:
        - DM - 1 for short way, -1 for long way

    NOTES:
        This assumes that both orbits are near the equatorial plane
    """

    angle1 = np.arctan2(r0[1], r0[0])
    angle2 = np.arctan2(rf[1], rf[0])
    
    angle1 = np.mod(angle1, 2*np.pi) # put angle1 between 0->2*pi
    angle2 = np.mod(angle2, 2*np.pi) # put angle2 between 0->2*pi
    
    temp = angle2 - angle1
    
    # Initialize
    if temp > 0:
        if temp < np.pi:
            DM = 1 # short way
        else:
            DM = -1 # long way

    else:
        if temp < -np.pi:
            DM = 1 # short way
        else:
            DM = -1 # long way
 
    return DM

# =====================================================================

# =====================================================================
# 6)
def lambertOptTransfer(oe0, oef, dt_bracket, mu, timeMod=1e2, DM=[], r_host=0, sc_at=0, mode='o2o', minMode='dvt', manType='bcb', plot=True):
    """This function finds the minimum dV_total (dV_t = dv1 + dv2)
    Lambert transfer between two orbits with several different 
    modes. 

    INPUT:
        oe0 - orbital elements of the first orbit
        oef - orbital elements of the second orbit
        dt_bracket - time bracket for the lambert transfer
            >currently gets modified if optimal lambert transfer
             time is outside this bracket
        mu - gravitational parameter of central body
        r_host - radius of central body
        sc_at - acceleration of spacecraft due to thrust
        mode - transfer search mode
            >'o2o', 'o2s', 's2o', 's2s'
        minMode - flag for choosing Brent Search minimization function
            'dv1' = first Lambert burn
            'dv2' = second Lambert burn
            'dvt' = total Lambert burn
            'time' = minimum time burn
        manType - flag for defining the maneuve type
            >'bcb', 'intercept'
        plot - flag to turn on plotting

    OUTPUT:
        oe1 - updated optimnal orbital elements of the first orbit
        oe2 - updated optimnal orbital elements of the second orbit
        vt1 - sc velocity vector at the start of the lambert transfer
        vt2 - sc velocity vector at the end of the lambert transfer
        dv1 - first impulsive maneuver vector
        dv2 - second impulsive maneuver vector
        dt  - optimal lambert transfer time

    NOTES:
        BrentSearch is called a lot for 'os2', 's2o', and 'o2o'. It
        is a nested BrentSearch. Each time BS is called to find the 
        optimal true anomaly (about 14 times), it is called again to 
        find the optimal time of flight. 

        lambertMinNu is called len(DM)*len(nuRange) times
        lambertMinDv is called ~15 times per lambertMinNu call
        brentSearch is called ~15*5*2 = 150 times.

        The lambertMinTime function is kinda slow (~3 sec per run).
        This leads to long ALTm runs when finding minimum time
        Lambert trajectories. 

        If the initial Lambert guess is exactly 0 or 180 deg apart, 
        the enitre algorithm won't work. This is "solved" by using
        random nu0 guesses for o2s and s2o. This situation could 
        happend, but just very unlikely now. I also put in a catch
        if-statement for s2s that perturbes the target true anomoly 
        by 1e-2 for these cases.

        o2o search is currently using unnecesary resources seaching 
        the corner cases [0, 0] [0, 360] [360, 0] [360, 360] that 
        aren't possible. I am trying to find a way to kill the 
        minimization routine when this occurs. I am just letting this 
        be for now because I don't have control over so.minimize, 
        which is where this error is occuring. 

        The time addition to the cost function is tricky (dt_lam/tconst)
        in dv1t and dvtt lamMinModes. I need it large enough to reward 
        short trajectories, but small enough to not totally dominate 
        the dV of the maneuver. Roughly 10x less than dv works well. I 
        have it set up so that it works well with normalized simulations. 
        1e7 for dimensional sims, 1e3 for normalized sims. These results 
        are based off one test case, so keep that in mind.
    """

    if not DM:
        DM = [-1,1]     # checking both Lambert directions

    if mode == 'o2o' or mode == 'fta2o':
        """This mode has both the initial and final state free, 
        so this code uses a 2-dimensional <NAME> search algorithm
        to find the initial and final state that results in the 
        minimum dV transfer. The transfer time is found using a 
        nested 1-dimensional Brent search algorithm to find
        the transfer time associated with the minimum dV transfer.
        Both Lambert directions are investigated. Furthermore, 
        due to the unpredictable nature of the dV profile, several
        Brent Searches are performed to ensure a global minimum
        (i.e., a partical swarm method).
        """
        
        numSwarm = 10    # partical swarm knob
        nu0_list = []
        for i in range(numSwarm):
            nu0_list.append(np.random.random(2)*2*np.pi)

        if mode == 'fta2o':
            for i in range(len(nu0_list)):
                nu0_list[i][0] = oe0[-1]

        parameterSaveOuter = []; minValOuter = []
        for dm in DM:
            parameterSaveInner = []; minValInner = []
            for nu0 in nu0_list:
                
                oe1, oe2, vt1, vt2, dv1, dv2, dt = lambertMinNu(np.copy(oe0), np.copy(oef), nu0, dt_bracket, dm, mu, timeMod, mode, minMode, r_host, sc_at, manType)
                if minMode.lower() == 'dvtt':
                    minValInner.append(np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)) + dt/timeMod)
                elif minMode.lower() == 'dv1t':
                    minValInner.append(np.sqrt(dv1.dot(dv1)) + dt/timeMod)
                elif minMode.lower() == 'dv2t':
                    minValInner.append(np.sqrt(dv2.dot(dv2)) + dt/timeMod)
                elif minMode.lower() == 'dvt':
                    minValInner.append(np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)))
                elif minMode.lower() == 'dv1':
                    minValInner.append(np.sqrt(dv1.dot(dv1)))
                elif minMode.lower() == 'dv2':
                    minValInner.append(np.sqrt(dv2.dot(dv2)))
                elif minMode.lower() == 'time':
                    minValInner.append(dt)
                else:
                    print('\nERROR: minMode selection not valid!\nPlease choose: "dvt" "dv1" "dv2" or "time"\n')
                parameterSaveInner.append(copy.deepcopy([oe1, oe2, vt1, vt2, dv1, dv2, dt]))

            if len(minValInner) > 0:
                minValOuter.append(min(val for val in minValInner if not np.isnan(val)))
                parameterSaveOuter.append(copy.deepcopy(parameterSaveInner[minValInner.index(minValOuter[-1])]))           
            else:
                print("\nERROR! No Lambert Trajectories Found!\n")

        minValIndex = minValOuter.index(min(minValOuter))
        oe1, oe2, vt1, vt2, dv1, dv2, dt = parameterSaveOuter[minValIndex]
        dm = DM[minValIndex]

    elif mode == 'o2s' or mode == 's2o' or mode == 'st2s' or mode == 'fta2s':
        """This mode has either the initial or final state free, 
        so this code uses a 1-dimensional Brent search algorithm
        to find the initial or final state that results in the 
        minimum dV transfer. The transfer time is found using a 
        nested 1-dimensional Brent search algorithm to find
        the transfer time associated with the minimum dV transfer.
        Both Lambert directions are investigated. Furthermore, 
        due to the unpredictable nature of the dV profile, several
        Brent Searches are performed to ensure a global minimum
        (i.e., a partical swarm method).
        """

        if mode == 'fta2s':
            nu0_list = [oe0[-1]]
        else:
            numSwarm = 5    # partical swarm knob
            nu0_list = []
            for i in range(numSwarm):
                nu0_list.append(np.random.random()*2*np.pi)

        parameterSaveOuter = []; minValOuter = []
        for dm in DM:
            parameterSaveInner = []; minValInner = []
            for nu0 in nu0_list:

                # Making sure cos_delnu is not -1 or 1
                oef_test = np.copy(oef)
                oef_test[-1] = nu0
                r1 = af.oe2rv(oe0, mu=mu)[0]
                r2 = af.oe2rv(oef_test, mu=mu)[0]
                cos_delnu = np.dot(r1, r2)/np.sqrt(r1.dot(r1)*r2.dot(r2))
                A = np.sqrt(r1.dot(r1)*r2.dot(r2)*(1.0 + cos_delnu))
                if np.arccos(cos_delnu) == 0 or A == 0:
                    nu0 += 1e-2

                oe1, oe2, vt1, vt2, dv1, dv2, dt = lambertMinNu(np.copy(oe0), np.copy(oef), nu0, dt_bracket, dm, mu, timeMod, mode, minMode, r_host, sc_at, manType)
                if minMode.lower() == 'dvtt':
                    minValInner.append(np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)) + dt/timeMod)
                elif minMode.lower() == 'dv1t':
                    minValInner.append(np.sqrt(dv1.dot(dv1)) + dt/timeMod)
                elif minMode.lower() == 'dv2t':
                    minValInner.append(np.sqrt(dv2.dot(dv2)) + dt/timeMod)
                elif minMode.lower() == 'dvt':
                    minValInner.append(np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)))
                elif minMode.lower() == 'dv1':
                    minValInner.append(np.sqrt(dv1.dot(dv1)))
                elif minMode.lower() == 'dv2':
                    minValInner.append(np.sqrt(dv2.dot(dv2)))
                elif minMode.lower() == 'time':
                    minValInner.append(dt)
                else:
                    print('\nERROR: minMode selection not valid!\nPlease choose: "dvt" "dv1" "dv2" or "time"\n')
                parameterSaveInner.append(copy.deepcopy([oe1, oe2, vt1, vt2, dv1, dv2, dt]))

            print(minValInner)

            if len(minValInner) > 0:
                minValOuter.append(min(val for val in minValInner if not np.isnan(val)))
                parameterSaveOuter.append(copy.deepcopy(parameterSaveInner[minValInner.index(minValOuter[-1])]))       
            else:
                print("\nERROR! No Lambert Trajectories Found!\n")

        minValIndex = minValOuter.index(min(minValOuter))
        oe1, oe2, vt1, vt2, dv1, dv2, dt = parameterSaveOuter[minValIndex]
        dm = DM[minValIndex]

        # try:
        #     minValIndex = minValOuter.index(min(minValOuter))
        #     oe1, oe2, vt1, vt2, dv1, dv2, dt = parameterSaveOuter[minValIndex]
        #     dm = DM[minValIndex]
        # except:
        #     # Doing a hohmann transfer if o2s didn't find any. This is happening in SK MC.
        #     r1, v1 = af.oe2rv(oe0, mu)
        #     r2, v2 = af.oe2rv(oef, mu)
        #     vt0, vtf, dv1, dv2, dt = lambertMinDv(r1, v1, r2, v2, dt_bracket, minMode, r_host, sc_at, manType, dm, mu)
        #     oe1 = np.copy(oe0); oe2 = np.copy(oef)
        #     vt1 = np.copy(vt0); vt2 = np.copy(vtf)

    elif mode == 's2s':
        """This mode has the initial and final states fixed, so 
        all that needs to vary is the Lambert transfer time. This
        code uses a 1-dimensional Brent search algorithm to find
        the transfer time associated with the minimum dV transfer.
        Both Lambert directions are investigated."""

        # Making sure cos_delnu is not -1 or 1
        r1, v1 = af.oe2rv(oe0, mu)
        r2, v2 = af.oe2rv(oef, mu)
        cos_delnu = np.dot(r1, r2)/np.sqrt(r1.dot(r1)*r2.dot(r2))
        A = np.sqrt(r1.dot(r1)*r2.dot(r2)*(1.0 + cos_delnu))
        if np.arccos(cos_delnu) == 0 or A == 0:
            oef[-1] += 1e-2
            r2, v2 = af.oe2rv(oef, mu)

        # Finding the minimum dV transfer for DM=1 and DM=-1
        parameterSave = []; minVal = []
        for dm in DM:
            vt0, vtf, dv1, dv2, dt = lambertMinDv(r1, v1, r2, v2, dt_bracket, minMode, r_host, sc_at, manType, dm, mu)
            if minMode.lower() == 'dvtt':
                minVal.append(np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)) + dt/timeMod)
            elif minMode.lower() == 'dv1t':
                minVal.append(np.sqrt(dv1.dot(dv1)) + dt/timeMod)
            elif minMode.lower() == 'dv2t':
                minVal.append(np.sqrt(dv2.dot(dv2)) + dt/timeMod)
            elif minMode.lower() == 'dvt':
                minVal.append(np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)))
            elif minMode.lower() == 'dv1':
                minVal.append(np.sqrt(dv1.dot(dv1)))
            elif minMode.lower() == 'dv2':
                minVal.append(np.sqrt(dv2.dot(dv2)))
            elif minMode.lower() == 'time':
                minVal.append(dt)
            else:
                print('\nERROR: minMode selection not valid!\nPlease choose: "dvt" "dv1" "dv2" or "time"\n')
            parameterSave.append(copy.deepcopy([vt0, vtf, dv1, dv2, dt]))

        # Reassigning the minimum parameters
        oe1 = np.copy(oe0); oe2 = np.copy(oef)
        minValIndex = minVal.index(min(val for val in minVal if not np.isnan(val)))
        vt1, vt2, dv1, dv2, dt = parameterSave[minValIndex]
        dm = DM[minValIndex]
        
    else:
        print('\nError in Optimal Lambert Transfer mode!\n')

    if plot:
        # -----------------------------------------------------------
        # Trajectory Plot
    
        # Converting initial orbits to Cartesian
        r0, v0 = af.oe2rv(oe0, mu)
        rf, vf = af.oe2rv(oef, mu)
        r0_vec = []; v0_vec = []
        rf_vec = []; vf_vec = []
        nu_plt = np.linspace(0, 2*np.pi, 100)
        for nu in nu_plt:
            r, v = af.oe2rv(np.hstack((oe0[0:-1], nu)), mu=mu)
            r0_vec.append(r); v0_vec.append(v)
    
            r, v = af.oe2rv(np.hstack((oef[0:-1], nu)), mu=mu)
            rf_vec.append(r); vf_vec.append(v)
        r0_vec = np.asarray(r0_vec); v0_vec = np.asarray(v0_vec)
        rf_vec = np.asarray(rf_vec); vf_vec = np.asarray(vf_vec)

        # Transfer Orbit
        r1, v1 = af.oe2rv(oe1, mu)
        r2, v2 = af.oe2rv(oe2, mu)
        oet1 = af.rv2oe(r1, vt1, mu)[0]
        oet2 = af.rv2oe(r2, vt2, mu)[0]
        if oet2[-1] > oet1[-1]:
            nut = np.linspace(oet1[-1], oet2[-1], 50)
        else:
            nut1_ratio = (2*np.pi - oet1[-1])/2/np.pi
            nut2_ratio = oet2[-1]/2/np.pi
            nut1 = np.linspace(oet1[-1], 2*np.pi, int(50*nut1_ratio))
            nut2 = np.linspace(0, oet2[-1], int(50*nut2_ratio))
            nut = np.hstack((nut1, nut2))
        
        rt_vec = []; vt_vec = []
        for nu in nut:
            r, v = af.oe2rv(np.hstack((oet1[0:-1], nu)), mu=mu)
            rt_vec.append(r); vt_vec.append(v)
        rt_vec = np.asarray(rt_vec); vt_vec = np.asarray(vt_vec)
    
        plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
    
        # Plot
        plt.plot(r0_vec[:,0], r0_vec[:,1], r0_vec[:,2], '-k', label='Initial Orbit', linewidth=1.5)
        plt.plot(rf_vec[:,0], rf_vec[:,1], rf_vec[:,2], '-g', label='Final Orbit', linewidth=1.5)
        plt.plot(rt_vec[:,0], rt_vec[:,1], rt_vec[:,2], '-b', label='Transfer Orbit', linewidth=1.5)
        
        # Points
        ax.plot([0], [0], [0], 'oy')
        ax.plot([r1[0]], [r1[1]], [r1[2]], 'ok')
        ax.plot([r2[0]], [r2[1]], [r2[2]], 'og')
        ax.plot([r0[0]], [r0[1]], [r0[2]], 'sk')
        ax.plot([rf[0]], [rf[1]], [rf[2]], 'sg')
    
        # Formatting
        ax.set_xlim([-max(oe0[0], oef[0]),max(oe0[0], oef[0])])
        ax.set_ylim([-max(oe0[0], oef[0]),max(oe0[0], oef[0])])
        ax.set_zlim([-max(oe0[0], oef[0]),max(oe0[0], oef[0])])
        
        # Plot Labels
        # plt.tight_layout()
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.title(r'Lambert Trajectory', fontsize=14)
        plt.legend(loc=0)
        # -----------------------------------------------------------

        # plt.show()

    return oe1, oe2, vt1, vt2, dv1, dv2, dt, dm
# ---------------------------------------------------------------
# =====================================================================

# =====================================================================
# 7)
# ---------------------------------------------------------------
def lambertMinDv(r1, v1, r2, v2, dt_bracket, minMode, r_host, sc_at, manType, dm=1, mu=c.mu_sun):
    """This function combines the Universal Variables Lambert 
    Algorithm with the Brent Search algorithm to find the minimum
    total dV Lambert transfer.

    INPUT:
        r1 - initial position vector
        v1 - initial velocity vector
        r2 - final position vector
        v2 - final velocity vector
        dt_bracket - bracket of possible dt values
        minMode - flag for choosing Brent Search minimization function
        r_host - radius of central body
        sc_at - acceleration of spacecraft due to thrust
        manType - flag for defining the maneuve type
        dm - direction of motion
             0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
            +1 - short way
            -1 - long way
        mu - gravitational parameter of central body 
             (default is sun)

    OUTPUT:
        vt1 - velocity at r1
        vt2 - velocity at r2
        dv1 - first impulsive dv
        dv2 - second impulsive dv
        dt_opt - time of flight for minimized total dV

    NOTES:
        If dt_bracket only contains one value, that value is
        used for the lambert transfer time.
    """

    if len(dt_bracket) == 1:
        dt_opt  = dt_bracket[0]

    else:
        if minMode.lower() == 'time':
            dt0 = dt_bracket[-1]*100 # This value is important. If it is too small, the function won't work
            dt_opt = lambertMinTime(dt0, r1, v1, r2, v2, dm, mu, r_host, sc_at, manType)
        else:
            args = (r1, v1, r2, v2, dm, mu, minMode)
            while True:
                dt0 = np.average(dt_bracket)
                dt_opt = of.brentSearch(errfunLambertdV, dt0, dt_bracket, args)[0]
                # if dt_opt == dt0: # this is causing problems
                #     dt_bracket[-1] *= 2
                if abs(dt_opt - dt_bracket[-1]) < 0.1*dt_bracket[-1]:
                    dt_bracket[-1] *= 2
                elif abs(dt_opt - dt_bracket[0]) < 0.1*dt_bracket[0]:
                    dt_bracket[0] /= 2
                else:
                    break

    # Using dt from bsearch
    vt1, vt2 = lambert_uv(r1, r2, dt_opt, dm, mu)[0:2]
    dv1 = vt1 - v1
    dv2 = v2 - vt2

    return vt1, vt2, dv1, dv2, dt_opt
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def errfunLambertdV(dt, r1, v1, r2, v2, dm, mu, minMode):
    """This function calculates the norm of the total dV 
    associated with a Lambert trajectory. The free variable is the
    transfer time dt. This error function is 1-dimensional and is 
    used with brentSearch.

    INPUT:
        dt - transfer time being varied to find dV_total min value
        r1 - frist sc position vector
        v1 - frist sc velocity vector
        r2 - second sc position vector
        v2 - second sc velocity vector
        dm - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        mu - gravitational parameter of central body 

    OUTPUT:
        dV_total - the norm of the total lambert dV
    """  

    # The algorithm is getting stalled in lambertMinDv in the brent seach function

    vt1, vt2 = lambert_uv(r1, r2, dt, dm, mu)[0:2]

    dv1 = vt1 - v1
    dv2 = v2 - vt2

    if minMode.lower() == 'dvt' or minMode.lower() == 'dvtt':
        errFunVal = np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2))
    elif minMode.lower() == 'dv1' or minMode.lower() == 'dv1t':
        errFunVal = np.sqrt(dv1.dot(dv1))
    elif minMode.lower() == 'dv2' or minMode.lower() == 'dv2t':
        errFunVal = np.sqrt(dv2.dot(dv2))
    else:
        print('\nERROR: minMode selection not valid in errfunLambertdV!\nPlease choose: "dv1t" dvt" "dv1" "dv2" or "time"\n')

    return errFunVal
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def lambertMinTime(dt, r1, v1, r2, v2, dm, mu, r_host, sc_at, manType):
    """This function calculates the fast Lambert trajectory that does 
    not impact with the central body. Current central body miss distance
    is 10% of the body's radius. 

    INPUT:
        dt - transfer time being varied
        r1 - frist sc position vector
        v1 - frist sc velocity vector
        r2 - second sc position vector
        v2 - second sc velocity vector
        dm - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        mu - gravitational parameter of central body 
        r_host - radius of central body
        sc_at - acceleration of spacecraft due to thrust
        manType - flag for defining the maneuve type

    OUTPUT:
        dtOpt - minimum transfer time with given thrusters
    """

    modFactor = 2
    dtList = [dt]
    dtLowerLim = 0
    if manType.lower() == 'intercept':  
        while True:
            # Lambert Transfer
            dtOpt = dtList[-1]
            vt1, vt2 = lambert_uv(r1, r2, dtOpt, dm, mu)[0:2]
            
            # Radius Check
            oet1 = af.rv2oe(r1, vt1, mu)[0]
            oet2 = af.rv2oe(r2, vt2, mu)[0]
            if oet2[-1] > oet1[-1]:
                nut = np.linspace(oet1[-1], oet2[-1], 1000)
            else:
                nut1_ratio = (2*np.pi - oet1[-1])/2/np.pi
                nut2_ratio = oet2[-1]/2/np.pi
                nut1 = np.linspace(oet1[-1], 2*np.pi, int(1000*nut1_ratio))
                nut2 = np.linspace(0, oet2[-1], 1000 - int(1000*nut1_ratio))
                nut = np.hstack((nut1, nut2))
            
            radiusTooSmall = False
            tranRadius = []
            for nu in nut:
                r = af.oe2rv(np.hstack((oet1[0:-1], nu)), mu=mu)[0]
                tranRadius.append(np.sqrt(r.dot(r)))
                if np.sqrt(r.dot(r)) < r_host*1.1:
                    dtLowerLim = dtOpt
                    radiusTooSmall = True
                    break

            if not radiusTooSmall:
                # Thrust Check
                dv1 = vt1 - v1
                dt_test = dtOpt - np.sqrt(dv1.dot(dv1))/sc_at

                if 0 < dt_test < 1e-3:
                    break
                elif dt_test > 0:
                    dt_new = dtOpt*(1 - 1/modFactor)
                    dtList.append(dt_new)
                else:
                    modFactor *= 2
                    dtNew = dtList[-2]*(1 - 1/modFactor)
                    if dtLowerLim != 0 and dtNew < dtLowerLim:
                        dtNew = dtLowerLim*1.1
                    dtList[-1] = dtNew
            else:
                if 0 < (r_host*1.1-min(tranRadius)) < 1e-3:
                    break
                else:
                    modFactor *= 2
                    dtNew = dtList[-2]*(1 - 1/modFactor)
                    if dtLowerLim != 0 and dtNew < dtLowerLim:
                            dtNew = dtLowerLim*1.1
                    dtList[-1] = dtNew

    elif manType.lower() == 'bcb':
        while True:
            # Lambert Transfer
            dtOpt = dtList[-1]
            vt1, vt2 = lambert_uv(r1, r2, dtOpt, dm, mu)[0:2]
            
            # Radius Check
            oet1 = af.rv2oe(r1, vt1, mu)[0]
            oet2 = af.rv2oe(r2, vt2, mu)[0]
            if oet2[-1] > oet1[-1]:
                nut = np.linspace(oet1[-1], oet2[-1], 1000)
            else:
                nut1_ratio = (2*np.pi - oet1[-1])/2/np.pi
                nut2_ratio = oet2[-1]/2/np.pi
                nut1 = np.linspace(oet1[-1], 2*np.pi, int(1000*nut1_ratio))
                nut2 = np.linspace(0, oet2[-1], 1000 - int(1000*nut1_ratio))
                nut = np.hstack((nut1, nut2))
            
            radiusTooSmall = False
            tranRadius = []
            for nu in nut:
                r = af.oe2rv(np.hstack((oet1[0:-1], nu)), mu=mu)[0]
                tranRadius.append(np.sqrt(r.dot(r)))
                if np.sqrt(r.dot(r)) < r_host*1.1:
                    dtLowerLim = dtOpt
                    radiusTooSmall = True
                    break

            if not radiusTooSmall:
                # Thrust Check
                dv1 = vt1 - v1
                dv2 = v2 - vt2
                dt1 = np.sqrt(dv1.dot(dv1))/sc_at
                dt3 = np.sqrt(dv2.dot(dv2))/sc_at
                dt_test = dtOpt - dt1 - dt3

                if 0 < dt_test < 1e-3:
                    break
                elif dt_test > 0:
                    dt_new = dtOpt*(1 - 1/modFactor)
                    dtList.append(dt_new)
                else:
                    modFactor *= 2
                    dtNew = dtList[-2]*(1 - 1/modFactor)
                    if dtLowerLim != 0 and dtNew < dtLowerLim:
                        dtNew = dtLowerLim*1.1
                    dtList[-1] = dtNew
            else:
                if 0 < (r_host*1.1-min(tranRadius)) < 1e-3:
                    break
                else:
                    modFactor *= 2
                    dtNew = dtList[-2]*(1 - 1/modFactor)
                    if dtLowerLim != 0 and dtNew < dtLowerLim:
                            dtNew = dtLowerLim*1.1
                    dtList[-1] = dtNew
    return dtOpt
# ---------------------------------------------------------------
# =====================================================================

# =====================================================================
# 8)
# ---------------------------------------------------------------
def lambertMinNu(oe1, oe2, nu0, dt_bracket, dm, mu, timeMod, mode, minMode, r_host, sc_at, manType):
    """This function combines the Brent search algorithm for 1D
    and a 2D scipy minimization algorithm for 2D with a Universal
    Lambert algorithm to find the minimum total dV Lambert transfer 
    by varying the true anomaly of the tow orbits. This function 
    uses a nested Brent search minimizaiton to find the optimal 
    Lambert transfer time.

    INPUT:
        oe1 - orbital elements of the first orbit
        oe2 - orbital elements of the second orbit
        nu0 - intial guess at true anomoaly
        dt_bracket - time bracket for the lambert transfer
            >currently gets modified if optimal lambert transfer
             time is outside this bracket
        dm - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        mu - gravitational parameter of central body 
        mode - transfer search mode
            >'o2o', 'o2s', 's2o', 's2s'
        minMode - flag for choosing Brent Search minimization function
        r_host - radius of central body
        sc_at - acceleration of spacecraft due to thrust
        manType - flag for defining the maneuve type

    OUTPUT:
        oe1 - updated optimal orbital elements of first orbit
        oe2 - updated optimal orbital elements of second orbit
        vt1 - velocity at r1
        vt2 - velocity at r2
        dv1 - first impulsive dv
        dv2 - second impulsive dv
        dt_opt - time of flight for minimized total dV
    """
    args = (oe1, oe2, dt_bracket, dm, mu, timeMod, mode, minMode, r_host, sc_at, manType)
    if mode == 'o2o' or mode == 'fta2o':

        if mode == 'o2o':
            bounds = [(0, 2*np.pi), (0, 2*np.pi)]
            # bounds = [(-4*np.pi, 4*np.pi), (-4*np.pi, 4*np.pi)]
        else:
            bounds = [(oe1[-1], oe1[-1] + np.pi), (0, 2*np.pi)]
            # bounds = [(oe1[-1], oe1[-1] + np.pi), (-4*np.pi, 4*np.pi)]
        nu_opt = so.minimize(errfunLambertNu, nu0, args=args, bounds=bounds)
    
        oe1[-1] = nu_opt.x[0]
        oe2[-1] = nu_opt.x[-1]
    
        r1, v1 = af.oe2rv(oe1, mu)
        r2, v2 = af.oe2rv(oe2, mu)

    elif mode == 'o2s' or mode == 's2o' or mode == 'fta2s':
        if mode == 'fta2s':
            bounds = [oe1[-1], oe1[-1] + np.pi]
        else:
            bounds = [-4*np.pi, 4*np.pi]
        nu_opt = of.brentSearch(errfunLambertNu, nu0, bounds, args)[0]
        
        while nu_opt < 0:
            nu_opt += 2*np.pi

        while nu_opt > 2*np.pi:
            nu_opt -= 2*np.pi

        if mode == 'o2s' or mode == 'fta2s':
            oe1[-1] = nu_opt
        elif mode == 's2o':
            oe2[-1] = nu_opt 
    
        r1, v1 = af.oe2rv(oe1, mu)
        r2, v2 = af.oe2rv(oe2, mu)

    elif mode == 'st2s':
        nu_opt = of.brentSearch(errfunLambertNuST, nu0, [-4*np.pi, 4*np.pi], args)[0]

        # New dt_bracket here

        while nu_opt < 0:
            nu_opt += 2*np.pi

        while nu_opt > 2*np.pi:
            nu_opt -= 2*np.pi

        oe1[-1] = nu_opt
        r1, v1 = af.oe2rv(oe1, mu)
        r2, v2 = af.oe2rv(oe2, mu)

    else:
        print('\n\nERROR in Lambert Min Nu Mode!!\n\n')
    
    vt1, vt2, dv1, dv2, dt = lambertMinDv(r1, v1, r2, v2, dt_bracket, minMode, r_host, sc_at, manType, dm, mu) 
    return oe1, oe2, vt1, vt2, dv1, dv2, dt
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def errfunLambertNu(nu, oe1, oe2, dt_bracket, dm, mu, timeMod, mode, minMode, r_host, sc_at, manType):
    """This function calculates the norm of the total dV 
    associated with a Lambert trajectory. The free variable is the
    true anomaly of either the first of second orbit The transfer 
    time dt is also being optimized. This error function is 
    1-dimensional and is used with brentSearch.

    INPUT:
        nu - true anomoly being varied to find dV_total min value
        oe1 - orbital elements of first orbit
        oe2 - orbital elements of second orbit
        dt_bracket - bounding bracket for dt
        DM - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        mu - gravitational parameter of central body 
        mode - transfer search mode
        minMode - flag for choosing Brent Search minimization function
        r_host - radius of central body
        sc_at - acceleration of spacecraft due to thrust
        manType - flag for defining the maneuve type

    OUTPUT:
        dV_total - the norm of the total lambert dV
    """

    if mode == 'o2o' or mode == 'fta2o':
        oe1[-1] = nu[0]
        oe2[-1] = nu[-1]
    else:
        # selecting mode
        if mode == 'o2s' or mode == 'fta2s':
            oe1[-1] = nu
        elif mode == 's2o':
            oe2[-1] = nu 
        else:
            print('\nIncorrect errfunLambertNu Search mode!\nMust be "o2o", "o2s", or "s2o".\n')

    r1, v1 = af.oe2rv(oe1, mu)
    r2, v2 = af.oe2rv(oe2, mu)
    
    vt1, vt2, dv1, dv2, dt_lam = lambertMinDv(r1, v1, r2, v2, dt_bracket, minMode, r_host, sc_at, manType, dm, mu)

    if minMode.lower() == 'dvtt':
        errFunVal = np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)) + dt_lam/timeMod
    elif minMode.lower() == 'dv1t':
        errFunVal = np.sqrt(dv1.dot(dv1)) + dt_lam/timeMod
    elif minMode.lower() == 'dv2t':
        errFunVal = np.sqrt(dv2.dot(dv2)) + dt_lam/timeMod
    elif minMode.lower() == 'dvt':
        errFunVal = np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2))
    elif minMode.lower() == 'dv1':
        errFunVal = np.sqrt(dv1.dot(dv1))
    elif minMode.lower() == 'dv2':
        errFunVal = np.sqrt(dv2.dot(dv2))
    elif minMode.lower() == 'time':
        errFunVal = dt_lam
    else:
        print('\nERROR: minMode selection not valid errfunLambertNu!\nPlease choose: "dv1t" dvt" "dv1" "dv2" or "time"\n')

    return errFunVal
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def errfunLambertNuST(nu, oe1, oe2, dt_bracket, dm, mu, mode, minMode, r_host, sc_at, manType):
    """This function calculates the norm of the total dV associated 
    with a Lambert trajectory. The free variable is the true anomaly 
    of the first orbit The transfer time dt is also being optimized. 
    This error function is 1-dimensional and is used with brentSearch.
    This error function makes sure the spacecraft arrives at its
    target by a specific time.

    INPUT:
        nu - true anomoly being varied to find dV_total min value
        oe1 - orbital elements of first orbit
        oe2 - orbital elements of second orbit
        dt_bracket - bounding bracket for dt
        DM - direction of motion
                 0 - calculate DM based on the assumption that 
                     both orbits are in the equatorial plane
                +1 - short way
                -1 - long way
        mu - gravitational parameter of central body 
        mode - transfer search mode
        minMode - flag for choosing Brent Search minimization function
        r_host - radius of central body
        sc_at - acceleration of spacecraft due to thrust
        manType - flag for defining the maneuve type

    OUTPUT:
        dV_total - the norm of the total lambert dV
    """

    # Adjusting dt_bracket for true anomaly adjustment
    """If the true anomaly change is less than 180, it is forward in time.
    If it is more than 180, it is backwards in time.
    """

    # Adjusting Transfer time based on true anomaly (+/-180 deg)
    n = np.sqrt(mu/oe1[0]**3)
    M1 = af.true2mean(oe1[-1], oe1[0], oe1[1])[0]
    M = af.true2mean(nu, oe1[0], oe1[1])[0]
    nuDiff = nu - oe1[-1]
    if nuDiff < 0:
        nuDiff += 2*np.pi

    if nuDiff < np.pi:
        dM = M - M1
        timeFactor = -1
    else:
        dM = M1 - M
        timeFactor = 1

    if dM < 0:
        dM += 2*np.pi

    dt = timeFactor*dM/n

    dt_bracket_adj = [dt_bracket[0] + dt]

    oe1[-1] = nu

    r1, v1 = af.oe2rv(oe1, mu)
    r2, v2 = af.oe2rv(oe2, mu)
    
    vt1, vt2, dv1, dv2, dt_lam = lambertMinDv(r1, v1, r2, v2, dt_bracket_adj, minMode, r_host, sc_at, manType, dm, mu)

    if minMode.lower() == 'dvtt':
        errFunVal = np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2)) + dt_lam/timeMod
    elif minMode.lower() == 'dv1t':
        errFunVal = np.sqrt(dv1.dot(dv1)) + dt_lam/timeMod
    elif minMode.lower() == 'dv2t':
        errFunVal = np.sqrt(dv2.dot(dv2)) + dt_lam/timeMod
    elif minMode.lower() == 'dvt':
        errFunVal = np.sqrt(dv1.dot(dv1)) + np.sqrt(dv2.dot(dv2))
    elif minMode.lower() == 'dv1':
        errFunVal = np.sqrt(dv1.dot(dv1))
    elif minMode.lower() == 'dv2':
        errFunVal = np.sqrt(dv2.dot(dv2))
    elif minMode.lower() == 'time':
        errFunVal = dt_lam
    else:
        print('\nERROR: minMode selection not valid in errfunLambertNuST!\nPlease choose: "dv1t" dvt" "dv1" "dv2" or "time"\n')

    return errFunVal
# ---------------------------------------------------------------
# =====================================================================

# =====================================================================
# #)


# =====================================================================
