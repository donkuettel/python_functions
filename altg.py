
######################################
##  Autonomous Low-Thrust Maneuvers ##
######################################

"""This script uses shooting methods to autonomously correct 
the BLT guidance parameters of a nominal single-burn maneuver 
to account for any initial state perturbations.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

This code consists of using an 's2s' or 's2o' single-shooter 
predictor corrector to correct nominal BLT guidance parameters. 
This code uses MEOEs and shooter algorithms.

NOTES:
    -Currently working for all dynaimcs.
    -'Free' and 'Fixed' time both working.
    -'s2s' and 's2o' working.

ISSUES:

"""

# Import Modules
import numpy as np
import dynamics as dyn
import constants as con
from bodies import Body
import astro_functions as af
from scipy.integrate import odeint

#######################################################################
# Main Function Call

# -------------------------------------------------------------------
def altg(oe0, oef, scInfo, dynInfo, dt, A, B, time='free', shooter='s2s', plot=True, body='bennu', grav=False, srp=False, thirdbod=False):
    """This function autonomously find continuous, finite maneuvers 
    between states and orbits. 

    ASSUMPTIONS:
        - The sun is always located along the -x-axis of the central 
          body's inertial coordinate system
        - A cannonball srp model is used
        - body-inertial and heliocentric-inertial coordinate frames
          are aligned
        - The control vector is always represented in the the 
          inertial frame
        - All integration in done in MEOE coordinates
        - only accounts for solar third body perturbation

    INPUT:
        oe0 - orbital elements of the initial orbit
        oef - orbital elements of the final orbit
        scInfo - dictionary of necessary spacecraft characteristics
                m0  - mass
                Cr  - coefficient of reflectivity
                a2m - area to mass ratio
                isp - specific impulse
                T   - thrust
        dynInfo - dictionary of dynamimcal characteristics
                odeBurn  - burn dynamics function
                odeCoast - coast dynamics function
                mu_host - gravitational parameter of central body 
                r_host - radius of central body
                w_body - rotation rate of the central body
                mu_3B - gravitational parameter of third body (sun)
                d_3B - distance of third body (sun) from central body
                r_3B - radius of third body (sun)
                degree - degree of spherical harmonic gravity
                order - order of spherical harmonic gravity
                theta_gst - initial angle between body-inertial and body-fixed frame
                gc - normalized C_lm and S_lm gravity constants of central body       
                srp_flux - value of the Sun's flux at 1 AU
                c - speed of light
                AU - astronomical unit
                g0 - standard gravity parameter
        dt - nominal maneuver time
        A  - nominal A BLT parameter 
        B  - nominal B BLT parameter 
        time - flag indicating if total maneuver time is free or fixed
        shooter - control and error vector of shooter: 's2o', 's2s'
        plot - flag to turn on plotting 
        body - central body of the simulation (default is Bennu)
        grav - flag to turn on the asymetrical gravity field
        srp - flag to turn on solar radiation pressure
        thirdbod - flag to turn on third-body pertubations from the Sun

    OUTPUT:
        oe0_fin - initial orbital elements of each segment
        oef_fin - final orbital elements of each segment
        A_fin - final A guidance parameters
        B_fin - final B guidance parameters
        dt_fin - final segment times
        X_plot - spacecraft trajectory for plotting in orbital elements

    NOTES:
        shooter - s2o = state-to-orbit, s2s = state-to-state
        
    """
    # -----------------------------------------------------------
    ## Dynamics ##
    sc_at = scInfo['T']/scInfo['m0']         # accel, m/s2
    body_cons = Body(body, grav, dynInfo)
    dynInfo['mu_host'] = body_cons['mu']*1e9        # m3/s2, gravitational parameter
    dynInfo['r_host'] = body_cons['r_body']*1e3     # m, nominal radius
    dynInfo['w_host'] = body_cons['w_body']         # rad/s, body rotation rate
    if grav:
        dynInfo['gc'] = body_cons['gc']             # sphereical harmonic coefficients

    meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm = shooterPrep(oe0, oef, A, B, dt, scInfo, dynInfo, time, shooter, plot, body, grav, srp, thirdbod)
    # -----------------------------------------------------------
    
    # -----------------------------------------------------------
    ## Guidance ## 
    if time == 'free': # no time specified
        if shooter == 's2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc = b_s2o_freetime_shooter(meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo)

        elif shooter == 's2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc = b_s2s_freetime_shooter(meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "s2o" or "s2s".\n')

    elif time == 'fixed':
        if shooter == 's2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc = b_s2o_fixedtime_shooter(meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo)

        elif shooter == 's2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc = b_s2s_fixedtime_shooter(meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "s2o" or "s2s".\n')

    else:
        print('\nError in maneuver time flag! Please choose "free" or "fixed".\n')

    # Re-dimensionalizing
    A_fin = [np.copy(A_gnc)]
    B_fin = [np.copy(B_gnc)/t_norm]
    dt_fin = [np.copy(dt_gnc)*t_norm]
    
    IC_fin = []
    for state in np.copy(IC_gnc):
        state[0] *= r_norm
        IC_fin.append(np.hstack((af.meoe2oe(state[0:6]), state[-1])))
    
    Xf_fin = []
    for state in np.copy(Xf_gnc):
        state[0] *= r_norm
        Xf_fin.append(np.hstack((af.meoe2oe(state[0:6]), state[-1])))

    # Integration for Plotting
    X_plot = []; tvec_plot = []
    if plot:
        odefun = shooterInfo['ode_burn']
        extras = shooterInfo['extras_burn'] + (A_gnc, B_gnc)
        tvec = np.linspace(shooterInfo['t0'], dt_gnc, 100)
        X_norm = odeint(odefun, IC_gnc[0], tvec, args=extras, 
            rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

        X = []
        for state in X_norm:
            state[0] *= r_norm
            X.append(np.hstack((af.meoe2oe(state[0:6]), state[-1])))
        
        X_plot.append(np.asarray(X))
        tvec_plot.append(tvec*t_norm)
    # -----------------------------------------------------------

    return IC_fin, Xf_fin, A_fin, B_fin, dt_fin, X_plot, tvec_plot
# -------------------------------------------------------------------

#######################################################################


#######################################################################
# Supporting Functions
""" 
    1) shooterPrep               - normalizes dynamics and prepares dictionary for shooter functions
    2) b_s2o_freetime_shooter    - b s2o free-time single-shooter
    3) b_s2s_freetime_shooter    - b s2s free-time single-shooter
    4) b_s2o_fixedtime_shooter   - b s2o fixed-time single-shooter
    5) b_s2s_fixedtime_shooter   - b s2s fixed-time single-shooter
"""

# -------------------------------------------------------------------
def shooterPrep(oe0, oef, A, B, dt, scInfo, dynInfo, time, shooter, plot, body, grav, srp, thirdbod):
    """This script preps the inputs (normalizes) to the bcb multiple 
    shooter function.

    # Dyanmics and Normilization
    
    COAST: (X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU)
    BLT: (X, t, mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp, A, B)

    mu_host             # m3/s2
    r_host              # m
    w_host              # rad/s
    mu_3B               # m3/s2
    d_3B                # m
    r_3B                # m
    degree              # -
    order               # -
    theta_gst           # rad
    gc                  # -
        C = gc['C_lm']  # -
        S = gc['S_lm']  # -
    Cr                  # -
    a2m                 # m2/kg
    srp_flux            # W/m2, kg/s3
    c                   # m/s
    AU                  # m
    g0                  # m/s2
    T                   # (kg)m/s2
    isp                 # s
    A                   # -
    B                   # 1/s
    """

    # Normalizing based off dt1 and mu = 1
    shooterInfo = {}                                # dictionary for shooter functions
    shooterInfo['abtol'] = 1e-12
    shooterInfo['reltol'] = 1e-12
    shooterInfo['m0'] = scInfo['m0']                # kg, spacecraft mass
    t_norm = np.copy(dt)                            # s, time of first burn
    r_norm = (dynInfo['mu_host']*t_norm**2)**(1/3)  # m
    shooterInfo['mu_host'] = dynInfo ['mu_host']/r_norm**3*t_norm**2

    # State
    meoe0_norm = af.oe2meoe(oe0)
    meoe0_norm[0] /= r_norm                     # m
    meoef_norm = af.oe2meoe(oef)
    meoef_norm[0] /= r_norm                     # m

    # Time
    shooterInfo['t0'] = dynInfo['t0']/t_norm    # s
    dt_norm = np.copy(dt)/t_norm                # s
    shooterInfo['manTime'] = np.sum(dt)/t_norm  # s

    # BLT parameters
    A_norm = np.copy(A)                         # -
    B_norm = np.copy(B)*t_norm                  # 1/s

    # Constants
    if grav == False and srp == False and thirdbod == False:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host,)

    elif grav == True and srp == False and thirdbod == False:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_grav_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_grav_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        r_host = dynInfo['r_host']/r_norm                  # m
        w_host = dynInfo['w_host']*t_norm                  # rad/s
        degree = dynInfo['degree']                         # -
        order = dynInfo['order']                           # -
        theta_gst = dynInfo['theta_gst']                   # rad
        gc = dynInfo['gc']                                 # -
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, r_host, w_host, degree, order, theta_gst, gc, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host, r_host, w_host, degree, order, theta_gst, gc)

    elif grav == False and srp == True and thirdbod == False:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_srp_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_srp_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        r_host = dynInfo['r_host']/r_norm                  # m
        d_3B = dynInfo['d_3B']/r_norm                      # m
        r_3B = dynInfo['r_3B']/r_norm                      # m
        Cr = scInfo['Cr']                                  # -
        a2m = scInfo['a2m']/r_norm/r_norm                  # m2/kg
        srp_flux = con.srp_flux*t_norm**3                  # kg/s3
        c = con.c/r_norm*t_norm                            # m/s
        AU = con.AU*1e3/r_norm                             # m
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, r_host, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host, r_host, d_3B, r_3B, Cr, a2m, srp_flux, c, AU)

    elif grav == False and srp == False and thirdbod == True:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_3bod_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_3bod_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        mu_3B = dynInfo['mu_3B']/r_norm**3*t_norm**2       # m3/s2
        d_3B = dynInfo['d_3B']/r_norm                      # m
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, mu_3B, d_3B, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host, mu_3B, d_3B)

    elif grav == True and srp == True and thirdbod == False:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_grav_srp_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_grav_srp_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        r_host = dynInfo['r_host']/r_norm                  # m
        w_host = dynInfo['w_host']*t_norm                  # rad/s
        d_3B = dynInfo['d_3B']/r_norm                      # m
        r_3B = dynInfo['r_3B']/r_norm                      # m
        degree = dynInfo['degree']                         # -
        order = dynInfo['order']                           # -
        theta_gst = dynInfo['theta_gst']                   # rad
        gc = dynInfo['gc']                                 # -
        Cr = scInfo['Cr']                                  # -
        a2m = scInfo['a2m']/r_norm/r_norm                  # m2/kg
        srp_flux = con.srp_flux*t_norm**3                  # kg/s3
        c = con.c/r_norm*t_norm                            # m/s
        AU = con.AU*1e3/r_norm                             # m
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, r_host, w_host, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host, r_host, w_host, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU)

    elif grav == True and srp == False and thirdbod == True:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_grav_3bod_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_grav_3bod_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        r_host = dynInfo['r_host']/r_norm                  # m
        w_host = dynInfo['w_host']*t_norm                  # rad/s
        mu_3B = dynInfo['mu_3B']/r_norm**3*t_norm**2       # m3/s2
        d_3B = dynInfo['d_3B']/r_norm                      # m
        degree = dynInfo['degree']                         # -
        order = dynInfo['order']                           # -
        theta_gst = dynInfo['theta_gst']                   # rad
        gc = dynInfo['gc']                                 # -
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, r_host, w_host, mu_3B, d_3B, degree, order, theta_gst, gc, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host, r_host, w_host, mu_3B, d_3B, degree, order, theta_gst, gc)

    elif grav == False and srp == True and thirdbod == True:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_srp_3bod_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_srp_3bod_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        r_host = dynInfo['r_host']/r_norm                  # m
        mu_3B = dynInfo['mu_3B']/r_norm**3*t_norm**2       # m3/s2
        d_3B = dynInfo['d_3B']/r_norm                      # m
        r_3B = dynInfo['r_3B']/r_norm                      # m
        Cr = scInfo['Cr']                                  # -
        a2m = scInfo['a2m']/r_norm/r_norm                  # m2/kg
        srp_flux = con.srp_flux*t_norm**3                  # kg/s3
        c = con.c/r_norm*t_norm                            # m/s
        AU = con.AU*1e3/r_norm                             # m
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, r_host, mu_3B, d_3B, r_3B, Cr, a2m, srp_flux, c, AU, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host, r_host, mu_3B, d_3B, r_3B, Cr, a2m, srp_flux, c, AU)

    elif grav == True and srp == True and thirdbod == True:
        # Dynamic Files
        shooterInfo['ode_burn'] = dyn.ode_2bod_grav_srp_3bod_blt_meoe 
        shooterInfo['ode_coast'] = dyn.ode_2bod_grav_srp_3bod_coast_meoe

        # Necessary Constants
        mu_host = dynInfo['mu_host']/r_norm**3*t_norm**2   # m3/s2
        r_host = dynInfo['r_host']/r_norm                  # m
        mu_3B = dynInfo['mu_3B']/r_norm**3*t_norm**2       # m3/s2
        w_host = dynInfo['w_host']*t_norm                  # rad/s
        d_3B = dynInfo['d_3B']/r_norm                      # m
        r_3B = dynInfo['r_3B']/r_norm                      # m
        degree = dynInfo['degree']                         # -
        order = dynInfo['order']                           # -
        theta_gst = dynInfo['theta_gst']                   # rad
        gc = dynInfo['gc']                                 # -
        Cr = scInfo['Cr']                                  # -
        a2m = scInfo['a2m']/r_norm/r_norm                  # m2/kg
        srp_flux = con.srp_flux*t_norm**3                  # kg/s3
        c = con.c/r_norm*t_norm                            # m/s
        AU = con.AU*1e3/r_norm                             # m
        g0 = con.g0/r_norm*t_norm*t_norm                   # m/s2
        T = scInfo['T']/r_norm*t_norm*t_norm               # (kg)m/s2
        isp = scInfo['isp']/t_norm                         # s

        shooterInfo['extras_burn'] = (mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU, g0, T, isp)
        shooterInfo['extras_coast'] = (mu_host, r_host, w_host, mu_3B, d_3B, r_3B, degree, order, theta_gst, gc, Cr, a2m, srp_flux, c, AU)

    return meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm
# -------------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-orbit single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        shooterInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        dt_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.hstack((meoe0, shooterInfo['m0']))
    tspan = [shooterInfo['t0'], dt]
    X_pert = odeint(shooterInfo['ode_burn'], IC_pert, tspan, 
        args=shooterInfo['extras_burn'] + (A,B), 
        rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    meoet = meoef[0:5]; beta = np.sqrt(dt)
    error_vec.append(X_pert[-1][0:5] - meoet)
    error_vec.append(dt - beta**2)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    dt_gnc = np.copy(dt)         # integration time of ms segments
    beta_gnc = np.copy(beta)     # time slack variable
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This single-burn single shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        de/du) are shown below.

        e = [p(tf) - pt         u = [A
             f(tf) - ft              B
             g(tf) - gt              dt
             h(tf) - ht              b ]
             k(tf) - kt              
               dt - b^2]
    
        Calculated with Finite Differencing

        Gamma_(6x8) = de/du = [dp(tf)/dA dp(tf)/dB dp(tf)/ddt 0
                               df(tf)/dA df(tf)/dB df(tf)/ddt 0
                               dg(tf)/dA dg(tf)/dB dg(tf)/ddt 0
                               dh(tf)/dA dh(tf)/dB dh(tf)/ddt 0
                               dk(tf)/dA dk(tf)/dB dk(tf)/ddt 0
                               0         0         1          -2b1] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 6; n = 8
        gamma = np.zeros((m,n))
        gamma[5,6] = 1. 
        gamma[5,7] = -2*beta_gnc

        # Finite Differencing
        for j in range(n-1): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B, dt)
            if 0 <= j < 3:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j]) + 1e-7
                A_fd[j] += fd_parameter
            elif 3 <= j < 6:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-3]) + 1e-7
                B_fd[j-3] += fd_parameter
            else:
                # Time
                fd_parameter = 1e-6*abs(dt_fd) + 1e-7
                dt_fd += fd_parameter

            # Integration
            X_fd = odeint(shooterInfo['ode_burn'], IC_fd, [shooterInfo['t0'], dt_fd], 
                args=shooterInfo['extras_burn'] + (A_fd, B_fd), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            for k in range(m-1): # Looping over e
                diff = X_fd[-1][k] - Xf_gnc[k]
                gamma[k,j] = diff/fd_parameter
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        gamma_inv = gamma.transpose() @ np.linalg.inv(gamma @ gamma.transpose())
        du = -np.dot(gamma_inv, error_vec)/du_mod

        print('\nIt:', count, '|', 'Current Error:', '{:.4e}'.format(error_mag))

        # Finding Correction
        inner_count = 0
        error_test = [error_mag]
        while True:

            # Control Parameters
            IC_test = np.copy(IC_gnc)
            A_test = np.copy(A_gnc)
            B_test = np.copy(B_gnc)
            dt_test = np.copy(dt_gnc)
            beta_test = np.copy(beta_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]
            dt_test += du[6]
            beta_test += du[7]

            # Integrating with new initial conditions
            X_test = odeint(shooterInfo['ode_burn'], IC_test, [shooterInfo['t0'], dt_test], 
                args=shooterInfo['extras_burn'] + (A_test, B_test), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            # Calculating new error
            error_vec = []
            error_vec.append(X_test[-1][0:5] - meoet)
            error_vec.append(dt_test - beta_test**2)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1
            
            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag < 1:
                error_test.append(error_check)
                break

            elif error_check/error_mag >= 1:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test

        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-state single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        shooterInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        dt_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.hstack((meoe0, shooterInfo['m0']))
    tspan = [shooterInfo['t0'], dt]
    X_pert = odeint(shooterInfo['ode_burn'], IC_pert, tspan, 
        args=shooterInfo['extras_burn'] + (A,B), 
        rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    meoet = meoef; beta = np.sqrt(dt)
    if abs(X_pert[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
        meoet[-1] += np.sign(X_pert[-1][-2] - meoet[-1])*2*np.pi
    error_vec.append(X_pert[-1][0:6] - meoet)
    error_vec.append(dt - beta**2)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    dt_gnc = np.copy(dt)         # integration time of ms segments
    beta_gnc = np.copy(beta)     # time slack variable
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This single-burn single shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        de/du) are shown below.

        e = [p(tf) - pt         u = [A
             f(tf) - ft              B
             g(tf) - gt              dt
             h(tf) - ht              b ]
             k(tf) - kt              
             L(tf) - Lt  
               dt - b^2]
    
        Calculated with Finite Differencing

        Gamma_(7x8) = de/du = [dp(tf)/dA dp(tf)/dB dp(tf)/ddt 0
                               df(tf)/dA df(tf)/dB df(tf)/ddt 0
                               dg(tf)/dA dg(tf)/dB dg(tf)/ddt 0
                               dh(tf)/dA dh(tf)/dB dh(tf)/ddt 0
                               dk(tf)/dA dk(tf)/dB dk(tf)/ddt 0
                               dL(tf)/dA dL(tf)/dB dL(tf)/ddt 0
                               0         0         1          -2b1] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 7; n = 8
        gamma = np.zeros((m,n))
        gamma[6,6] = 1. 
        gamma[6,7] = -2*beta_gnc

        # Finite Differencing
        for j in range(n-1): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B, dt)
            if 0 <= j < 3:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j]) + 1e-7
                A_fd[j] += fd_parameter
            elif 3 <= j < 6:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-3]) + 1e-7
                B_fd[j-3] += fd_parameter
            else:
                # Time
                fd_parameter = 1e-6*abs(dt_fd) + 1e-7
                dt_fd += fd_parameter

            # Integration
            X_fd = odeint(shooterInfo['ode_burn'], IC_fd, [shooterInfo['t0'], dt_fd], 
                args=shooterInfo['extras_burn'] + (A_fd, B_fd), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            for k in range(m-1): # Looping over e
                diff = X_fd[-1][k] - Xf_gnc[k]
                gamma[k,j] = diff/fd_parameter
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        gamma_inv = gamma.transpose() @ np.linalg.inv(gamma @ gamma.transpose())
        du = -np.dot(gamma_inv, error_vec)/du_mod

        print('\nIt:', count, '|', 'Current Error:', '{:.4e}'.format(error_mag))

        # Finding Correction
        inner_count = 0
        error_test = [error_mag]
        while True:

            # Control Parameters
            IC_test = np.copy(IC_gnc)
            A_test = np.copy(A_gnc)
            B_test = np.copy(B_gnc)
            dt_test = np.copy(dt_gnc)
            beta_test = np.copy(beta_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]
            dt_test += du[6]
            beta_test += du[7]

            # Integrating with new initial conditions
            X_test = odeint(shooterInfo['ode_burn'], IC_test, [shooterInfo['t0'], dt_test], 
                args=shooterInfo['extras_burn'] + (A_test, B_test), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            # Calculating new error
            error_vec = []
            error_vec.append(X_test[-1][0:6] - meoet)
            error_vec.append(dt_test - beta_test**2)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1
            
            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag < 1:
                error_test.append(error_check)
                break

            elif error_check/error_mag >= 1:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test

        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time state-to-orbit single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        shooterInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        dt_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.hstack((meoe0, shooterInfo['m0']))
    tspan = [shooterInfo['t0'], dt]
    X_pert = odeint(shooterInfo['ode_burn'], IC_pert, tspan, 
        args=shooterInfo['extras_burn'] + (A,B), 
        rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

    # Calculating Initial Error
    meoet = meoef[0:5]
    error_vec = X_pert[-1][0:5] - meoet
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    dt_gnc = np.copy(dt)         # integration time of ms segments
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This single-burn single shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        de/du) are shown below.

        e = [p(tf) - pt         u = [A
             f(tf) - ft              B]
             g(tf) - gt
             h(tf) - ht
             k(tf) - kt]
    
        Calculated with Finite Differencing

        Gamma_(5x6) = de/du = [dp(tf)/dA dp(tf)/dB
                               df(tf)/dA df(tf)/dB
                               dg(tf)/dA dg(tf)/dB
                               dh(tf)/dA dh(tf)/dB
                               dk(tf)/dA dk(tf)/dB] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 5; n = 6
        gamma = np.zeros((m,n))

        # Finite Differencing
        for j in range(n): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B)
            if 0 <= j < 3:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j]) + 1e-7
                A_fd[j] += fd_parameter
            elif 3 <= j < 6:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-3]) + 1e-7
                B_fd[j-3] += fd_parameter

            # Integration
            X_fd = odeint(shooterInfo['ode_burn'], IC_fd, [shooterInfo['t0'], dt_fd], 
                args=shooterInfo['extras_burn'] + (A_fd, B_fd), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            for k in range(m): # Looping over e
                diff = X_fd[-1][k] - Xf_gnc[k]
                gamma[k,j] = diff/fd_parameter
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        gamma_inv = gamma.transpose() @ np.linalg.inv(gamma @ gamma.transpose())
        du = -np.dot(gamma_inv, error_vec)/du_mod

        print('\nIt:', count, '|', 'Current Error:', '{:.4e}'.format(error_mag))

        # Finding Correction
        inner_count = 0
        error_test = [error_mag]
        while True:

            # Control Parameters
            IC_test = np.copy(IC_gnc)
            A_test = np.copy(A_gnc)
            B_test = np.copy(B_gnc)
            dt_test = np.copy(dt_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]

            # Integrating with new initial conditions
            X_test = odeint(shooterInfo['ode_burn'], IC_test, [shooterInfo['t0'], dt_test], 
                args=shooterInfo['extras_burn'] + (A_test, B_test), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            # Calculating new error
            error_vec = X_test[-1][0:5] - meoet
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1
            
            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag < 1:
                error_test.append(error_check)
                break

            elif error_check/error_mag >= 1:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test

        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time state-to-state single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        shooterInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        dt_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.hstack((meoe0, shooterInfo['m0']))
    tspan = [shooterInfo['t0'], dt]
    X_pert = odeint(shooterInfo['ode_burn'], IC_pert, tspan, 
        args=shooterInfo['extras_burn'] + (A,B), 
        rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

    # Calculating Initial Error
    meoet = meoef
    if abs(X_pert[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
        meoet[-1] += np.sign(X_pert[-1][-2] - meoet[-1])*2*np.pi
    error_vec = X_pert[-1][0:6] - meoet
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    dt_gnc = np.copy(dt)         # integration time of ms segments
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This single-burn single shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        de/du) are shown below.

        e = [p(tf) - pt         u = [A
             f(tf) - ft              B]
             g(tf) - gt
             h(tf) - ht
             k(tf) - kt
             L(tf) - Lt]
    
        Calculated with Finite Differencing

        Gamma_(6x6) = de/du = [dp(tf)/dA dp(tf)/dB
                               df(tf)/dA df(tf)/dB
                               dg(tf)/dA dg(tf)/dB
                               dh(tf)/dA dh(tf)/dB
                               dk(tf)/dA dk(tf)/dB 
                               dL(tf)/dA dL(tf)/dB] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 6; n = 6
        gamma = np.zeros((m,n))

        # Finite Differencing
        for j in range(n): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B)
            if 0 <= j < 3:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j]) + 1e-7
                A_fd[j] += fd_parameter
            elif 3 <= j < 6:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-3]) + 1e-7
                B_fd[j-3] += fd_parameter

            # Integration
            X_fd = odeint(shooterInfo['ode_burn'], IC_fd, [shooterInfo['t0'], dt_fd], 
                args=shooterInfo['extras_burn'] + (A_fd, B_fd), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            for k in range(m): # Looping over e
                diff = X_fd[-1][k] - Xf_gnc[k]
                gamma[k,j] = diff/fd_parameter
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        gamma_inv = gamma.transpose() @ np.linalg.inv(gamma @ gamma.transpose())
        du = -np.dot(gamma_inv, error_vec)/du_mod

        print('\nIt:', count, '|', 'Current Error:', '{:.4e}'.format(error_mag))

        # Finding Correction
        inner_count = 0
        error_test = [error_mag]
        while True:

            # Control Parameters
            IC_test = np.copy(IC_gnc)
            A_test = np.copy(A_gnc)
            B_test = np.copy(B_gnc)
            dt_test = np.copy(dt_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]

            # Integrating with new initial conditions
            X_test = odeint(shooterInfo['ode_burn'], IC_test, [shooterInfo['t0'], dt_test], 
                args=shooterInfo['extras_burn'] + (A_test, B_test), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            # Calculating new error
            error_vec = X_test[-1][0:6] - meoet
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1
            
            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag < 1:
                error_test.append(error_check)
                break

            elif error_check/error_mag >= 1:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test

        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc
# ---------------------------------------------------------------
