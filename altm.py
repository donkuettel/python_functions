
######################################
##  Autonomous Low-Thrust Maneuvers ##
######################################

"""This script uses Lambert's Method and shooting methods to
autonomously plan continuous, finite maneuvers less than one
revolution.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

This code consists of two parts: finding an initial guess for the 
maneuver and then correcting that initial guess using a 
multiple-shooter predictor corrector. This code uses MEOEs, 
Lambert Solver, Brent Search, and shooter algorithms.

NOTES:
    -Currently working for all dynaimcs.
    -'Free' and 'Fixed' time both working.
    -'bcb', 'intercept', and 'b' working.

ISSUES:
    -------------------------------------------------------------------
    Angle wrapping issue with L in the MEOEs (i.e., Lf = 6.5, Lt = 0.22).
    These values are actually super close, but due to the angle wrap, they
    don't appear to be close and screw up the predictor-corrector. Something
    like this needs to be done:
        
        if meoef[-1] < 2*np.pi:
            meoef[-1] += 2*np.pi

    I fixed this by adding

        if abs(X_int[-1][-2] - meoet[-1]) >= np.pi:
            meoet[-1] += np.sign(X_int[-1][-2] - meoet[-1])*2*np.pi

    to 'o2s' and 's2s' error functions. If the difference between
    X_int[-1][-2] and meoet[-1] is greater than pi, that means adding or 
    subracting 2*pi will reduce the error and target the same true anomaly.
    -------------------------------------------------------------------

    -------------------------------------------------------------------
    Issue with full dynamics and Lambert initial guess. If there is a long 
    coast phase in a full dynamics simulation, the two-body Lambert initial
    guess is not good enough for the mulitple-shooter to converge. This is 
    a larger issue when the thrust is high. 

    I added a function that offsets the Lambert target to account for 
    dynamic drift.
    -------------------------------------------------------------------

    -------------------------------------------------------------------
    Issue with mulitple-shooter having to converge on bad initial guess. There
    is currently a line in the code 

        if error_check/error_mag > ratio:
            du /= 10

    that determines if du should be reduced to promote convergence. The larger
    the "ratio" value is (to a limit), the more freedom the multiple-shooter has 
    to change the control vector. However, that normally adds more computations.
    This is a knob that can be tweaked by ALTa. I am finding that 2 is a pretty
    good value for "ratio".
    -------------------------------------------------------------------

    -------------------------------------------------------------------
    Minimum time BCB transfers do not work the best with this code. The 
    minimum time transfer seeks to have no coast phase. This makes the 
    maneuvers rather long, and the Lambert initial guess isn't always 
    good enough for successful convergence. Works well with intercept.
    -------------------------------------------------------------------
"""

# Import Modules
import numpy as np
import dynamics as dyn
import constants as con
from bodies import Body
import scipy.optimize as so
import astro_functions as af
import lambert_functions as lf
from scipy.integrate import odeint
import optimization_functions as of

#######################################################################
# Main Function Call

# -------------------------------------------------------------------
def altm(oe0, oef, scInfo, dynInfo, dt=[], A=[], B=[], time='free', manType='bcb', shooter='o2o', searchMode='o2o', lamMinMode='dvt', plot=True, body='bennu', grav=False, srp=False, thirdbod=False):
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
        dt - maneuver time options: [], [dtt], [dt1, dt2, dt3], [dt1, dt2, dt3, dt4, dt5]
        A  - can specify the A BLT parameter for maneuver segments
        B  - can specify the B BLT parameter for maneuver segments
        time - flag indicating if total maneuver time is free or fixed
        manType - type of maneuver: 'b', 'bcb', 'bcbcb', 'intercept'
        shooter - control and error vector of shooter: 'o2o', 'o2s', 's2o', 's2s', 'o2r', 's2r'
        searchMode - manuever search type: 'o2o', 'o2s', 's2o', 's2s'
        lamMinMode - flag for choosing Brent Search minimization function
            'dv1' = first Lambert burn
            'dv2' = second Lambert burn
            'dvt' = total Lambert burn
            'time' = minimum time Lambert transfer
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
        manType - 'b' = a single burn, 'bcb' = burn-coast-burn sequence,
                  'bcbcb' = a burn-coast-burn-coast-burn squence, and 
                  'intercept' indicates that is spacecraft is only 
                  targeting a position with a burn-coast maneuver.
        
        shooter - o2o = orbit-to-orbit, o2s = = orbit-to-state, s2o = 
                  state-to-orbit, s2s = state-to-state, o2r = orbit-to-position,
                  s2r = state-to-position. o2r and s2r are only used with
                  'intercept' manType. This parameter determines the shooter
                  control and error vectors.
        
        dt      - If dt is empty, the algorithm works to find the 
                  minimum Lambert dV transfer time. If a single value is
                  specified, it is used as the Lambert transfer time. If a 
                  vector is specified, the components are the maneuver times 
                  of each of the segments.
        
        A/B     - If no BLT parameters are specified, an initial guess will
                  be found using Lambert/Brent search.
        
        - BLT parameters can only be specified if all the time segements are 
          also specified
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
    # -----------------------------------------------------------
    
    # -----------------------------------------------------------
    ## Maneuvers ## 

    # -----------------------------------------------------------
    if manType == 'b':
        
        # Getting intial guess for single shooter
        if len(dt) == 0:
            # Maneuver BLT Initial Guess
            oe0, oef, dv = singleBurnInitialGuess(oe0, oef, dynInfo, searchMode)
            dt = [np.sqrt(dv.dot(dv))/sc_at]       # manevuer time guess, s
            A = dv/np.sqrt(dv.dot(dv))
            B = np.zeros(3)

            print('\n==============================================================')
            print('Single Burn')
            print('\tTransfer Time =', '{:.2f}'.format(dt[0]/60), 'min')
            print('\tdV =', '{:.2f}'.format(np.sqrt(dv.dot(dv))*100), 'cm/s')
            print('==============================================================')

        # B Multiple shooter function
        meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm = shooterPrep(oe0, oef, A, B, dt[0], scInfo, dynInfo, time, shooter, plot, body, grav, srp, thirdbod)
        oe0_fin, oef_fin, A_fin, B_fin, dt_fin, converge, X_plot, tvec_plot = bMultipleShooter(meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm, time, shooter, plot)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    elif manType == 'bcb':

        # Getting initial guess for multiple shooter
        if len(dt) == 3: # Time of all maneuver segments are specified
            dt1 = dt[0]; A1 = A[0]; B1 = B[0]
            dt2 = dt[1]
            dt3 = dt[2]; A3 = A[-1]; B3 = B[-1]

        else: 
            # Find optimal Lambert transfer
            if not dt: # no time specified
                dt_bracket = [100*60, 10000*60]  # Important knob
                oe0, oef, vt1, vt2, dv1, dv2, dt_lam, dm = lf.lambertOptTransfer(oe0, oef, dt_bracket, dynInfo['mu_host'], r_host=dynInfo['r_host'], sc_at=sc_at, mode=searchMode, minMode=lamMinMode, manType=manType, plot=plot)
    
            elif len(dt) == 1: # Lambert transfer time is specified
                oe0, oef, vt1, vt2, dv1, dv2, dt_lam, dm = lf.lambertOptTransfer(oe0, oef, dt, dynInfo['mu_host'], r_host=dynInfo['r_host'], sc_at=sc_at, mode=searchMode, minMode=lamMinMode, manType=manType, plot=plot)
    
            else:
                print('\nError in specifying time segment!\n')

            # Lambert Drift Check
            meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm = shooterPrep(oe0, oef, A, B, dt_lam, scInfo, dynInfo, time, shooter, plot, body, grav, srp, thirdbod)
            dv1, dv2, dt_lam = lambertRetarget(meoe0_norm, meoef_norm, dv1/r_norm*t_norm, dv2/r_norm*t_norm, dt_norm, shooterInfo)

            # Turn Lambert transfer into initial continuous guess
            """
            Option to create a function here if I find a better way to calculate
            the initial guess for the shooter
            
            dt1, A1, B1, dt2, dt3, A3, B3 = FUNCTION(oe1, oe2, dv1, dv2, dt_lam)
            """
        
            # Maneuvers BLT Initial Guess 
            dt1 = np.sqrt(dv1.dot(dv1))/sc_at       # manevuer time guess, s
            A1 = dv1/np.sqrt(dv1.dot(dv1))
            B1 = np.zeros(3)
            A3 = dv2/np.sqrt(dv2.dot(dv2))
            B3 = np.zeros(3)
            dt3 = np.sqrt(dv2.dot(dv2))/sc_at       # manevuer time guess, s
            dt2 = dt_lam - dt1 - dt3                # manevuer time guess, s
            if dt2 < 0:                             # This will be a flag for ALTa
                dt2 = 1
                # print('\nERROR: Not enough thrust for this transfer time!')
                # print('dt2 =', dt2, '\n')

            print('\n==============================================================')
            print('Impulsive Transfer')
            print('\tTransfer Time =', '{:.2f}'.format(dt_lam/60), 'min')
            print('\tdV1 =', '{:.2f}'.format(np.sqrt(dv1.dot(dv1))*100), 'cm/s')
            print('\tdV2 =', '{:.2f}'.format(np.sqrt(dv2.dot(dv2))*100), 'cm/s')
            print('\tdVt =', '{:.2f}'.format((np.sqrt(dv1.dot(dv1))+np.sqrt(dv2.dot(dv2)))*100), 'cm/s')
            print('==============================================================')

        # BCB Multiple shooter function
        dt = np.array([dt1, dt2, dt3])
        A = np.array([A1, np.zeros(3), A3])
        B = np.array([B1, np.zeros(3), B3])
        print(dt)
        print(A)
        print(B)
        meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm = shooterPrep(oe0, oef, A, B, dt, scInfo, dynInfo, time, shooter, plot, body, grav, srp, thirdbod)
        oe0_fin, oef_fin, A_fin, B_fin, dt_fin, converge, X_plot, tvec_plot = bcbMultipleShooter(meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm, time, shooter, plot)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    elif manType == 'intercept':
        
        # Getting initial guess for multiple shooter
        if len(dt) == 2: # Time of all maneuver segments are specified
            dt1 = dt[0]; A1 = A[0]; B1 = B[0]
            dt2 = dt[1]

        else: 
            # Find optimal Lambert transfer
            if not dt: # no time specified
                dt_bracket = [100*60, 10000*60]  # Important knob
                oe0, oef, vt1, vt2, dv1, dv2, dt_lam, dm = lf.lambertOptTransfer(oe0, oef, dt_bracket, dynInfo['mu_host'], r_host=dynInfo['r_host'], sc_at=sc_at, mode=searchMode, minMode=lamMinMode, manType=manType, plot=plot)
    
            elif len(dt) == 1: # Lambert transfer time is specified
                oe0, oef, vt1, vt2, dv1, dv2, dt_lam, dm = lf.lambertOptTransfer(oe0, oef, dt, dynInfo['mu_host'], r_host=dynInfo['r_host'], sc_at=sc_at, mode=searchMode, minMode=lamMinMode, manType=manType, plot=plot)
    
            else:
                print('\nError in specifying time segment!\n')

            # Lambert Drift Check
            meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm = shooterPrep(oe0, oef, A, B, dt_lam, scInfo, dynInfo, time, shooter, plot, body, grav, srp, thirdbod)
            dv1, dv2, dt_lam = lambertRetarget(meoe0_norm, meoef_norm, dv1/r_norm*t_norm, dv2/r_norm*t_norm, dt_norm, shooterInfo)

            # Turn Lambert transfer into initial continuous guess
            """
            Option to create a function here if I find a better way to calculate
            the initial guess for the shooter
            
            dt1, A1, B1, dt2 = FUNCTION(oe0, oef, dv1, dv2, dt_lam)
            """
        
            # Maneuvers BLT Initial Guess 
            dt1 = np.sqrt(dv1.dot(dv1))/sc_at       # manevuer time guess, s
            A1 = dv1/np.sqrt(dv1.dot(dv1))
            B1 = np.zeros(3)
            dt2 = dt_lam - dt1                      # manevuer time guess, s
            if dt2 < 0:                             # This will be a flag for ALTa
                dt2 = 1
                # print('\nERROR: Not enough thrust for this transfer time!')
                # print('dt2 =', dt2, '\n')

            print('\n==============================================================')
            print('Impulsive Transfer')
            print('\tTransfer Time =', '{:.2f}'.format(dt_lam/60), 'min')
            print('\tdV1 =', '{:.2f}'.format(np.sqrt(dv1.dot(dv1))*100), 'cm/s')
            print('\tdV2 =', '{:.2f}'.format(np.sqrt(dv2.dot(dv2))*100), 'cm/s')
            print('\tdVt =', '{:.2f}'.format((np.sqrt(dv1.dot(dv1))+np.sqrt(dv2.dot(dv2)))*100), 'cm/s')
            print('==============================================================')

        # BC Multiple shooter function
        dt = np.array([dt1, dt2])
        A = np.array([A1, np.zeros(3)])
        B = np.array([B1, np.zeros(3)])
        meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm = shooterPrep(oe0, oef, A, B, dt, scInfo, dynInfo, time, shooter, plot, body, grav, srp, thirdbod)
        oe0_fin, oef_fin, A_fin, B_fin, dt_fin, converge, X_plot, tvec_plot = bcMultipleShooter(meoe0_norm, meoef_norm, A_norm, B_norm, dt_norm, shooterInfo, r_norm, t_norm, time, shooter, lamMinMode, plot)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    elif manType == 'bcbcb':
        if shooter == 'o2o':
            print('\n' + manType + '_' + shooter + ' is currently under construction!')
        elif shooter == 'o2s':
            print('\n' + manType + '_' + shooter + ' is currently under construction!')
        elif shooter == 's2o':
            print('\n' + manType + '_' + shooter + ' is currently under construction!')
        elif shooter == 's2s':
            print('\n' + manType + '_' + shooter + ' is currently under construction!')
        else:
            print('\nHey, dummy! Pick a correct BCBCB maneuver type!')
    # -----------------------------------------------------------

    else:
        print('\n\nHey, dummy! Pick a correct manType!\n\n')
    # -----------------------------------------------------------

    return oe0_fin, oef_fin, A_fin, B_fin, dt_fin, converge, X_plot, tvec_plot
# -------------------------------------------------------------------

#######################################################################


#######################################################################
# Supporting Functions
""" 
    1) shooterPrep               - normalizes dynamics and prepares dictionary for shooter functions
    2) bMulitpleShooter          - calls correct 'b' shooter function
    3) bcMultipleShooter         - calls correct 'bc' shooter function
    4) bcbMulitpleShooter        - calls correct 'bcb' shooter function
    5) bcb_o2o_freetime_shooter  - bcb o2o free-time multiple-shooter
    6) bcb_o2s_freetime_shooter  - bcb o2s free-time multiple-shooter
    7) bcb_s2o_freetime_shooter  - bcb s2o free-time multiple-shooter
    8) bcb_s2s_freetime_shooter  - bcb s2s free-time multiple-shooter
    9) bcb_o2o_fixedtime_shooter - bcb o2o fixed-time multiple-shooter
   10) bcb_o2s_fixedtime_shooter - bcb o2s fixed-time multiple-shooter
   11) bcb_s2o_fixedtime_shooter - bcb s2o fixed-time multiple-shooter
   12) bcb_s2s_fixedtime_shooter - bcb s2s fixed-time multiple-shooter
   13) bc_o2r_freetime_shooter   - bc o2r free-time multiple-shooter
   14) bc_s2r_freetime_shooter   - bc s2r free-time multiple-shooter
   15) b_o2r_freetime_shooter    - b o2r free-time single-shooter
   16) b_s2r_freetime_shooter    - b s2r free-time single-shooter
   17) bc_o2r_fixedtime_shooter  - bc o2r fixed-time multiple-shooter
   18) bc_s2r_fixedtime_shooter  - bc s2r fixed-time multiple-shooter
   19) b_o2o_freetime_shooter    - b o2o free-time single-shooter
   20) b_o2s_freetime_shooter    - b o2s free-time single-shooter
   21) b_s2o_freetime_shooter    - b s2o free-time single-shooter
   22) b_s2s_freetime_shooter    - b s2s free-time single-shooter
   23) b_o2o_fixedtime_shooter   - b o2o fixed-time single-shooter
   24) b_o2s_fixedtime_shooter   - b o2s fixed-time single-shooter
   25) b_s2o_fixedtime_shooter   - b s2o fixed-time single-shooter
   26) b_s2s_fixedtime_shooter   - b s2s fixed-time single-shooter
   27) errfunManShift            - error function for shifting single-burn ignition point
   28) singleBurnInitialGuess    - initial guess for a single-burn maneuver
   29) errfunClosestPoints       - error function for determining if single-burn maneuver is possible
   30) lambertRetarget           - accounts for dynamic drift in Lambert transfer with higher fidelity dynamics
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
    try:
        t_norm = np.copy(dt[0])                         # s, time of first burn
    except:
        t_norm = np.copy(dt)
    r_norm = (dynInfo['mu_host']*t_norm**2)**(1/3)  # m
    shooterInfo['mu_host'] = dynInfo ['mu_host']/r_norm**3*t_norm**2
    shooterInfo['r_norm'] = r_norm
    shooterInfo['t_norm'] = t_norm

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_coast_rv

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_grav_coast_rv

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_srp_coast_rv

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_3bod_coast_rv

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_grav_srp_coast_rv

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_grav_3bod_coast_rv

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_srp_3bod_coast_rv

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
        shooterInfo['ode_retarget'] = dyn.ode_2bod_grav_srp_3bod_coast_rv

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

# -------------------------------------------------------------------
def bMultipleShooter(meoe0, meoef, A, B, dt, shooterInfo, r_norm, t_norm, time, shooter, plot):
    """This script calls the correct bcb multiple shooter function and 
    renormalizes the output of the bcb multiple shooter function.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        shooterInfo - dictionary of all necessary integration constants
        time - flag indicating if total maneuver time is free or fixed
        shooter - control and error vector of shooter: 'o2o', 'o2s', 's2o', 's2s', 'o2r', 's2r'
        plot - flag to turn on plotting 

    OUTPUT:
        IC_fin - initial conditions for the BCB segements
        Xf_fin - final state for the BCB segements
        A_fin - converged A BLT parameters
        B_fin - converged B BLT parameters
        dt_fin - converged segment times
        X_plot - full trajectory of BCB segements
        tvec_plot - full time vectors of BCB segments
    """

    # Calculate Manevuer Shift
    args = (meoe0, meoef, A, B, dt, shooterInfo)
    L_bracket = [meoe0[-1]*0.5, meoe0[-1]*1.5]
    L_opt = of.brentSearch(errfunManShift, meoe0[-1], L_bracket, args)[0]
    meoe0[-1] = L_opt

    # -----------------------------------------------------------
    ## Shooter ##

    print('\n==============================================================')
    print('Converging on ' + time + '-time, ' + shooter + ' single-burn trajectory')
    if time == 'free': # no time specified
        if shooter == 'o2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_o2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 'o2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_o2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_s2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_s2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "o2o", "o2s", "s2o", or "s2s".\n')

    elif time == 'fixed':
        if shooter == 'o2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_o2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 'o2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_o2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_s2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_s2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "o2o", "o2s", "s2o", or "s2s".\n')

    else:
        print('\nError in maneuver time flag! Please choose "free" or "fixed".\n')
    print('==============================================================')

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

    return IC_fin, Xf_fin, A_fin, B_fin, dt_fin, converge, X_plot, tvec_plot
# -------------------------------------------------------------------

# -------------------------------------------------------------------
def bcMultipleShooter(meoe0, meoef, A, B, dt, shooterInfo, r_norm, t_norm, time, shooter, lamMinMode, plot):
    """This script calls the correct intercept multiple shooter 
    function and renormalizes the output of the intercept multiple 
    shooter function.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        shooterInfo - dictionary of all necessary integration constants
        time - flag indicating if total maneuver time is free or fixed
        shooter - control and error vector of shooter: 'o2o', 'o2s', 's2o', 's2s', 'o2r', 's2r'
        lamMinMode - flag for choosing Brent Search minimization function
        plot - flag to turn on plotting 

    OUTPUT:
        IC_fin - initial conditions for the BCB segements
        Xf_fin - final state for the BCB segements
        A_fin - converged A BLT parameters
        B_fin - converged B BLT parameters
        dt_fin - converged segment times
        X_plot - full trajectory of BCB segements
        tvec_plot - full time vectors of BCB segments
    """
    # -----------------------------------------------------------
    ## Shooter ##

    print('\n==============================================================')
    print('Converging on ' + time + '-time, ' + shooter + ' intercept trajectory')
    if time == 'free': # no time specified
        if shooter == 'o2r':
            if lamMinMode.lower() == 'time':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_o2r_freetime_shooter(meoe0, meoef, A[0], B[0], dt[0], shooterInfo)
            else:
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bc_o2r_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2r':
            if lamMinMode.lower() == 'time':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = b_s2r_freetime_shooter(meoe0, meoef, A[0], B[0], dt[0], shooterInfo)
            else:
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bc_s2r_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "o2r" or "s2r".\n')

    elif time == 'fixed':
        if shooter == 'o2r':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bc_o2r_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2r':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bc_s2r_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "o2r" or "s2r".\n')

    else:
        print('\nError in maneuver time flag! Please choose "free" or "fixed".\n')
    print('==============================================================')

    # Re-dimensionalizing
    if lamMinMode.lower() == 'time':
        A_fin = [np.copy(A_gnc)]
        B_fin = [np.copy(B_gnc)/t_norm]
        dt_fin = [np.copy(dt_gnc)*t_norm]
    else:
        A_fin = np.copy(A_gnc)
        B_fin = np.copy(B_gnc)/t_norm
        dt_fin = np.copy(dt_gnc)*t_norm
    
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
        X_plot_norm = []; tvec_plot_norm = []
        if lamMinMode.lower() == 'time':
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A_gnc, B_gnc)
            tvec = np.linspace(shooterInfo['t0'], dt_gnc, 100)
            X = odeint(odefun, IC_gnc[0], tvec, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
            X_plot_norm.append(X); tvec_plot_norm.append(tvec)
        else:
            segment_type = [1, 0]
            for i in range(len(segment_type)):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_gnc[i], B_gnc[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Segment Integration
                tvec = np.linspace(shooterInfo['t0']+np.sum(dt_gnc[0:i]), shooterInfo['t0']+np.sum(dt_gnc[0:i+1]), 100)
                X = odeint(odefun, IC_gnc[i], tvec, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                X_plot_norm.append(X); tvec_plot_norm.append(tvec)
        
        for X_norm in X_plot_norm:
            X = []
            for state in X_norm:
                state[0] *= r_norm
                X.append(np.hstack((af.meoe2oe(state[0:6]), state[-1])))
            X_plot.append(np.asarray(X))
    
        for tvec in tvec_plot_norm:
            tvec_plot.append(tvec*t_norm)
    # -----------------------------------------------------------

    return IC_fin, Xf_fin, A_fin, B_fin, dt_fin, converge, X_plot, tvec_plot
# -------------------------------------------------------------------

# -------------------------------------------------------------------
def bcbMultipleShooter(meoe0, meoef, A, B, dt, shooterInfo, r_norm, t_norm, time, shooter, plot):
    """This script calls the correct bcb multiple shooter function and 
    renormalizes the output of the bcb multiple shooter function.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        shooterInfo - dictionary of all necessary integration constants
        time - flag indicating if total maneuver time is free or fixed
        shooter - control and error vector of shooter: 'o2o', 'o2s', 's2o', 's2s', 'o2r', 's2r'
        plot - flag to turn on plotting 

    OUTPUT:
        IC_fin - initial conditions for the BCB segements
        Xf_fin - final state for the BCB segements
        A_fin - converged A BLT parameters
        B_fin - converged B BLT parameters
        dt_fin - converged segment times
        X_plot - full trajectory of BCB segements
        tvec_plot - full time vectors of BCB segments
    """
    # -----------------------------------------------------------
    ## Shooter ##

    print('\n==============================================================')
    print('Converging on ' + time + '-time, ' + shooter + ' burn-coast-burn trajectory')
    if time == 'free': # no time specified
        if shooter == 'o2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_o2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 'o2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_o2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_s2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_s2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "o2o", "o2s", "s2o", or "s2s".\n')

    elif time == 'fixed':
        if shooter == 'o2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_o2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 'o2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_o2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2o':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_s2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        elif shooter == 's2s':
            IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge = bcb_s2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo)

        else:
            print('\nError in shooter flag! Please choose "o2o", "o2s", "s2o", or "s2s".\n')

    else:
        print('\nError in maneuver time flag! Please choose "free" or "fixed".\n')
    print('==============================================================')

    # Re-dimensionalizing
    A_fin = np.copy(A_gnc)
    B_fin = np.copy(B_gnc)/t_norm
    dt_fin = np.copy(dt_gnc)*t_norm
    
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
        segment_type = [1, 0, 1]
        X_plot_norm = []; tvec_plot_norm = []
        for i in range(len(segment_type)):
            # Segement Type
            if segment_type[i] == 1:
                odefun = shooterInfo['ode_burn']
                extras = shooterInfo['extras_burn'] + (A_gnc[i], B_gnc[i])
            else:
                odefun = shooterInfo['ode_coast']
                extras = shooterInfo['extras_coast']
            # Segment Integration
            tvec = np.linspace(shooterInfo['t0']+np.sum(dt_gnc[0:i]), shooterInfo['t0']+np.sum(dt_gnc[0:i+1]), 100)
            X = odeint(odefun, IC_gnc[i], tvec, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
            X_plot_norm.append(X); tvec_plot_norm.append(tvec)
        
        for X_norm in X_plot_norm:
            X = []
            for state in X_norm:
                state[0] *= r_norm
                X.append(np.hstack((af.meoe2oe(state[0:6]), state[-1])))
            X_plot.append(np.asarray(X))
    
        for tvec in tvec_plot_norm:
            tvec_plot.append(tvec*t_norm)
    # -----------------------------------------------------------

    return IC_fin, Xf_fin, A_fin, B_fin, dt_fin, converge, X_plot, tvec_plot
# -------------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_o2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-orbit burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef[0:5]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            error.append(X_int[-1,0:5] - meoet)
            error.append(dt[i] - beta[i]**2)
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)            dt3 - b3^2]
                 dt1 - b1^2  ]           dt2 - b2^2  ]
        
        u1 = [L1(t0)            u2 = [p2(t0)            u3 = [p3(t0)
               A1                     f2(t0)                  f3(t0)
               B1                     g2(t0)                  g3(t0)
               dt1                    h2(t0)                  h3(t0)
               b1   ]                 k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
    
        E = [e1, e2, e3]; U = [u1, u2, u3]
    
        Gamma_(20x31) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
    
        Gamma_(20x31) = dE/dU = [de1/du1_(7x9) de1/du2_(7x8)       0_(7x14)
                                       0_(7x9) de2/du2_(7x8) de2/du3_(7x14)
                                       0_(6x9)       0_(6x8) de3/du3_(6x14)]
    
        de1/de2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
    
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        Calculated with Finite Differencing
    
        de1/du1_(7x9) = [dp1(tf)/dL1(t0) dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dL1(t0) df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dL1(t0) dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dL1(t0) dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dL1(t0) dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dL1(t0) dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0               0           0           1            -2b1] 
    
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
    
        de1/du1_(6x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((20,31))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 9
                gamma = np.zeros((m,n))
                gamma[6,7] = 1. 
                gamma[6,8] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 6; n = 14
                gamma = np.zeros((m,n))
                gamma[5,12] = 1. 
                gamma[5,13] = -2*beta_gnc[i]

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
                    if j == 0:
                        # Initial Burn L0
                        fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                        IC_fd[j+5] += fd_parameter
                    elif 1 <= j < 4:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                        A_fd[j-1] += fd_parameter
                    elif 4 <= j < 7:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                        B_fd[j-4] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                for k in range(m-1): # Looping over e
                    diff = X_fd[-1][k] - Xf_gnc[i][k]
                    gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:9] = gamma
                GAMMA[0:7,9:17] = de1du2
            elif i == N-1:
                GAMMA[14:20,17:31] = gamma
            else:
                GAMMA[7:14,9:17] = gamma
                GAMMA[7:14,17:31] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    IC_test[i][5] += du[0]
                    A_test[i] += du[1:4]
                    B_test[i] += du[4:7]
                    dt_test[i] += du[7]
                    beta_test[i] += du[8]
                elif i == 1:
                    IC_test[i][0:6] += du[9:15]
                    dt_test[i] += du[15]
                    beta_test[i] += du[16]
                else:
                    IC_test[i][0:6] += du[17:23]
                    A_test[i] += du[23:26]
                    B_test[i] += du[26:29]
                    dt_test[i] += du[29]
                    beta_test[i] += du[30]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:5] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            """There is a trade off between the error_check/error_mag value.
            A larger value gives the shooter more freedom to change values 
            and can ultimately converge faster. However, it may never converge 
            if a step is too big. A small value results in more iterations, but
            a higher chance of success. 

            Example)
            sma0 = 2000.                            # m, semi-major axis
            e0 = 0.05                               # -, eccentricity
            inc0 = np.deg2rad(90)                   # rad, inclination
            raan0 = np.deg2rad(200)                 # rad, raan
            w0 = np.deg2rad(10)                     # rad, arg of periapsis
            nu0 = np.deg2rad(10)                    # rad, true anomaly

            smaf = 250.                             # m, semi-major axis
            ef = 0.001                              # -, eccentricity
            incf = np.deg2rad(45)                   # rad, inclination
            raanf = np.deg2rad(270)                 # rad, raan
            wf = np.deg2rad(70)                     # rad, arg of periapsis
            nuf = np.deg2rad(90)                    # rad, true anomaly

            shooter = 'o2o', searchMode = 'o2s'

            error_check/error_mag > 10 --> does not converge
            error_check/error_mag > 1 --> converges

            The higher the number, the more freedom the mulitple shooter has
            to change the control parameters
            """
            
            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_o2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-state burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef[0:6]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            if abs(X_int[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
                meoet[-1] += np.sign(X_int[-1][-2] - meoet[-1])*2*np.pi
            error.append(X_int[-1,0:6] - meoet)
            error.append(dt[i] - beta[i]**2)
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)         L3(tf) - Lt
                 dt1 - b1^2  ]           dt2 - b2^2  ]         dt3 - b3^2]
        
        u1 = [L1(t0)            u2 = [p2(t0)            u3 = [p3(t0)
               A1                     f2(t0)                  f3(t0)
               B1                     g2(t0)                  g3(t0)
               dt1                    h2(t0)                  h3(t0)
               b1   ]                 k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
        
        E = [e1_(7), e2_(7), e3_(7)]; U = [u1_(9), u2_(8), u3_(14)]
        
        Gamma_(21x31) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
        
        Gamma_(21x31) = dE/dU = [de1/du1_(7x9) de1/du2_(7x8)       0_(7x14)
                                       0_(7x9) de2/du2_(7x8) de2/du3_(7x14)
                                       0_(7x9)       0_(7x8) de3/du3_(7x14)]
        
        de1/du2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
        
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x9) = [dp1(tf)/dL1(t0) dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dL1(t0) df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dL1(t0) dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dL1(t0) dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dL1(t0) dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dL1(t0) dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0               0           0           1            -2b1] 
        
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        
        de3/du3_(7x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          dL3(tf)/dp3(t0) dL3(tf)/df3(t0) dL3(tf)/dg3(t0) dL3(tf)/dh3(t0) dL3(tf)/dk3(t0) dL3(tf)/dL3(t0) dL3(tf)/dA3 dL3(tf)/dB3 dL3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((21,31))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 9
                gamma = np.zeros((m,n))
                gamma[6,7] = 1. 
                gamma[6,8] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 7; n = 14
                gamma = np.zeros((m,n))
                gamma[6,12] = 1. 
                gamma[6,13] = -2*beta_gnc[i]

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
                    if j == 0:
                        # Initial Burn L0
                        fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                        IC_fd[j+5] += fd_parameter
                    elif 1 <= j < 4:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                        A_fd[j-1] += fd_parameter
                    elif 4 <= j < 7:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                        B_fd[j-4] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                for k in range(m-1): # Looping over e
                    diff = X_fd[-1][k] - Xf_gnc[i][k]
                    gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:9] = gamma
                GAMMA[0:7,9:17] = de1du2
            elif i == N-1:
                GAMMA[14:21,17:31] = gamma
            else:
                GAMMA[7:14,9:17] = gamma
                GAMMA[7:14,17:31] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    IC_test[i][5] += du[0]
                    A_test[i] += du[1:4]
                    B_test[i] += du[4:7]
                    dt_test[i] += du[7]
                    beta_test[i] += du[8]
                elif i == 1:
                    IC_test[i][0:6] += du[9:15]
                    dt_test[i] += du[15]
                    beta_test[i] += du[16]
                else:
                    IC_test[i][0:6] += du[17:23]
                    A_test[i] += du[23:26]
                    B_test[i] += du[26:29]
                    dt_test[i] += du[29]
                    beta_test[i] += du[30]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:6] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_s2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-orbit burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef[0:5]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            error.append(X_int[-1,0:5] - meoet)
            error.append(dt[i] - beta[i]**2)
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)          dt3 - b3^2]
                 dt1 - b1^2  ]           dt2 - b2^2  ]
        
        u1 = [A1                u2 = [p2(t0)            u3 = [p3(t0)
              B1                      f2(t0)                  f3(t0)
              dt1                     g2(t0)                  g3(t0)
              b1 ]                    h2(t0)                  h3(t0)
                                      k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
        
        E = [e1_(7), e2_(7), e3_(6)]; U = [u1_(8), u2_(8), u3_(14)]
        
        Gamma_(20x30) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
        
        Gamma_(20x30) = dE/dU = [de1/du1_(7x8) de1/du2_(7x8)       0_(7x14)
                                       0_(7x8) de2/du2_(7x8) de2/du3_(7x14)
                                       0_(6x8)       0_(6x8) de3/du3_(6x14)]
        
        de1/du2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
        
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x8) = [dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0           0           1            -2b1] 
        
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        
        de3/du3_(6x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((20,30))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 6; n = 14
                gamma = np.zeros((m,n))
                gamma[5,12] = 1. 
                gamma[5,13] = -2*beta_gnc[i]

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
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
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                for k in range(m-1): # Looping over e
                    diff = X_fd[-1][k] - Xf_gnc[i][k]
                    gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:8] = gamma
                GAMMA[0:7,8:16] = de1du2
            elif i == N-1:
                GAMMA[14:20,16:30] = gamma
            else:
                GAMMA[7:14,8:16] = gamma
                GAMMA[7:14,16:30] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    A_test[i] += du[0:3]
                    B_test[i] += du[3:6]
                    dt_test[i] += du[6]
                    beta_test[i] += du[7]
                elif i == 1:
                    IC_test[i][0:6] += du[8:14]
                    dt_test[i] += du[14]
                    beta_test[i] += du[15]
                else:
                    IC_test[i][0:6] += du[16:22]
                    A_test[i] += du[22:25]
                    B_test[i] += du[25:28]
                    dt_test[i] += du[28]
                    beta_test[i] += du[29]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:5] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_s2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-state burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            if abs(X_int[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
                meoet[-1] += np.sign(X_int[-1][-2] - meoet[-1])*2*np.pi
            error.append(X_int[-1,0:6] - meoet)
            error.append(dt[i] - beta[i]**2)
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)         L3(tf) - Lt
                 dt1 - b1^2  ]           dt2 - b2^2  ]         dt3 - b3^2]
        
        u1 = [A1                u2 = [p2(t0)            u3 = [p3(t0)
              B1                      f2(t0)                  f3(t0)
              dt1                     g2(t0)                  g3(t0)
              b1 ]                    h2(t0)                  h3(t0)
                                      k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
        
        E = [e1_(7), e2_(7), e3_(7)]; U = [u1_(8), u2_(8), u3_(14)]
        
        Gamma_(21x30) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
        
        Gamma_(21x30) = dE/dU = [de1/du1_(7x8) de1/du2_(7x8)       0_(7x14)
                                       0_(7x8) de2/du2_(7x8) de2/du3_(7x14)
                                       0_(7x8)       0_(7x8) de3/du3_(7x14)]
        
        de1/du2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
        
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x8) = [dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0           0           1            -2b1] 
        
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        
        de3/du3_(7x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          dL3(tf)/dp3(t0) dL3(tf)/df3(t0) dL3(tf)/dg3(t0) dL3(tf)/dh3(t0) dL3(tf)/dk3(t0) dL3(tf)/dL3(t0) dL3(tf)/dA3 dL3(tf)/dB3 dL3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((21,30))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 7; n = 14
                gamma = np.zeros((m,n))
                gamma[6,12] = 1. 
                gamma[6,13] = -2*beta_gnc[i]

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
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
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                for k in range(m-1): # Looping over e
                    diff = X_fd[-1][k] - Xf_gnc[i][k]
                    gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:8] = gamma
                GAMMA[0:7,8:16] = de1du2
            elif i == N-1:
                GAMMA[14:21,16:30] = gamma
            else:
                GAMMA[7:14,8:16] = gamma
                GAMMA[7:14,16:30] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    A_test[i] += du[0:3]
                    B_test[i] += du[3:6]
                    dt_test[i] += du[6]
                    beta_test[i] += du[7]
                elif i == 1:
                    IC_test[i][0:6] += du[8:14]
                    dt_test[i] += du[14]
                    beta_test[i] += du[15]
                else:
                    IC_test[i][0:6] += du[16:22]
                    A_test[i] += du[22:25]
                    B_test[i] += du[25:28]
                    dt_test[i] += du[28]
                    beta_test[i] += du[29]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:6] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_o2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time orbit-to-orbit burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef[0:5]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            error.append(X_int[-1,0:5] - meoet)
            error.append(dt[i] - beta[i]**2)
            error.append(shooterInfo['manTime'] - np.sum(dt))
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    de3du1 = np.zeros((7,9)); de3du1[6,7] = -1
    de3du2 = np.zeros((7,8)); de3du2[6,6] = -1
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)         dt3 - b3^2
                 dt1 - b1^2  ]           dt2 - b2^2  ]        T - dt1 - dt2 - dt3]
        
        u1 = [L1(t0)            u2 = [p2(t0)            u3 = [p3(t0)
               A1                     f2(t0)                  f3(t0)
               B1                     g2(t0)                  g3(t0)
               dt1                    h2(t0)                  h3(t0)
               b1   ]                 k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
        
        E = [e1_(7), e2_(7), e3_(7)]; U = [u1_(9), u2_(8), u3_(14)]
        
        Gamma_(21x31) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
        
        Gamma_(21x31) = dE/dU = [de1/du1_(7x9) de1/du2_(7x8)       0_(7x14)
                                       0_(7x9) de2/du2_(7x8) de2/du3_(7x14)
                                 de3/du1_(7x9) de3/du2_(7x8) de3/du3_(7x14)]
        
        de1/du2_(7x8) =  [-1  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0
                           0  0  0 -1  0  0  0  0
                           0  0  0  0 -1  0  0  0
                           0  0  0  0  0 -1  0  0
                           0  0  0  0  0  0  0  0]
        
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        de3/du1_(7x9) =  [ 0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0 -1  0]
         
        de3/du2_(7x8) =  [ 0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0 -1  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x9) = [dp1(tf)/dL1(t0) dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dL1(t0) df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dL1(t0) dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dL1(t0) dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dL1(t0) dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dL1(t0) dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0               0           0           1            -2b1] 
        
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        
        de3/du3_(7x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3
                          0               0               0               0               0               0               0           0           -1           0   ]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((21,31))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 9
                gamma = np.zeros((m,n))
                gamma[6,7] = 1. 
                gamma[6,8] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 7; n = 14
                gamma = np.zeros((m,n))
                gamma[5,12] = 1. 
                gamma[5,13] = -2*beta_gnc[i]
                gamma[6,12] = -1.

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
                    if j == 0:
                        # Initial Burn L0
                        fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                        IC_fd[j+5] += fd_parameter
                    elif 1 <= j < 4:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                        A_fd[j-1] += fd_parameter
                    elif 4 <= j < 7:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                        B_fd[j-4] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                # Looping over e
                if i == N-1:
                    for k in range(m-2): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter
                else:
                    for k in range(m-1): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:9] = gamma
                GAMMA[0:7,9:17] = de1du2
            elif i == N-1:
                GAMMA[14:21,17:31] = gamma
                GAMMA[14:21,0:9] = de3du1
                GAMMA[14:21,9:17] = de3du2
            else:
                GAMMA[7:14,9:17] = gamma
                GAMMA[7:14,17:31] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    IC_test[i][5] += du[0]
                    A_test[i] += du[1:4]
                    B_test[i] += du[4:7]
                    dt_test[i] += du[7]
                    beta_test[i] += du[8]
                elif i == 1:
                    IC_test[i][0:6] += du[9:15]
                    dt_test[i] += du[15]
                    beta_test[i] += du[16]
                else:
                    IC_test[i][0:6] += du[17:23]
                    A_test[i] += du[23:26]
                    B_test[i] += du[26:29]
                    dt_test[i] += du[29]
                    beta_test[i] += du[30]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:5] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                    error.append(shooterInfo['manTime'] - np.sum(dt_test))
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1
            
            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_o2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time orbit-to-state burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            if abs(X_int[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
                meoet[-1] += np.sign(X_int[-1][-2] - meoet[-1])*2*np.pi
            error.append(X_int[-1,0:6] - meoet)
            error.append(dt[i] - beta[i]**2)
            error.append(shooterInfo['manTime'] - np.sum(dt))
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    de3du1 = np.zeros((8,9)); de3du1[7,7] = -1
    de3du2 = np.zeros((8,8)); de3du2[7,6] = -1
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)         L3(tf) - Lt
                 dt1 - b1^2  ]           dt2 - b2^2  ]        dt3 - b3^2
                                                              T - dt1 - dt2 - dt3]
        
        u1 = [L1(t0)            u2 = [p2(t0)            u3 = [p3(t0)
               A1                     f2(t0)                  f3(t0)
               B1                     g2(t0)                  g3(t0)
               dt1                    h2(t0)                  h3(t0)
               b1   ]                 k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
        
        E = [e1_(7), e2_(7), e3_(8)]; U = [u1_(9), u2_(8), u3_(14)]
        
        Gamma_(22x31) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
        
        Gamma_(22x31) = dE/dU = [de1/du1_(7x9) de1/du2_(7x8)       0_(7x14)
                                       0_(7x9) de2/du2_(7x8) de2/du3_(7x14)
                                 de3/du1_(8x9) de3/du2_(8x8) de3/du3_(8x14)]
        
        de1/du2_(7x8) =  [-1  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0
                           0  0  0 -1  0  0  0  0
                           0  0  0  0 -1  0  0  0
                           0  0  0  0  0 -1  0  0
                           0  0  0  0  0  0  0  0]
        
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        de3/du1_(8x9) =  [ 0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0 -1  0]
         
        de3/du2_(8x8) =  [ 0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0 -1  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x9) = [dp1(tf)/dL1(t0) dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dL1(t0) df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dL1(t0) dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dL1(t0) dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dL1(t0) dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dL1(t0) dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0               0           0           1            -2b1] 
        
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        
        de3/du3_(8x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          dL3(tf)/dp3(t0) dL3(tf)/df3(t0) dL3(tf)/dg3(t0) dL3(tf)/dh3(t0) dL3(tf)/dk3(t0) dL3(tf)/dL3(t0) dL3(tf)/dA3 dL3(tf)/dB3 dL3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3
                          0               0               0               0               0               0               0           0           -1           0   ]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((22,31))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 9
                gamma = np.zeros((m,n))
                gamma[6,7] = 1. 
                gamma[6,8] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 8; n = 14
                gamma = np.zeros((m,n))
                gamma[6,12] = 1. 
                gamma[6,13] = -2*beta_gnc[i]
                gamma[7,12] = -1.

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
                    if j == 0:
                        # Initial Burn L0
                        fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                        IC_fd[j+5] += fd_parameter
                    elif 1 <= j < 4:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                        A_fd[j-1] += fd_parameter
                    elif 4 <= j < 7:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                        B_fd[j-4] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                # Looping over e
                if i == N-1:
                    for k in range(m-2): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter
                else:
                    for k in range(m-1): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:9] = gamma
                GAMMA[0:7,9:17] = de1du2
            elif i == N-1:
                GAMMA[14:22,17:31] = gamma
                GAMMA[14:22,0:9] = de3du1
                GAMMA[14:22,9:17] = de3du2
            else:
                GAMMA[7:14,9:17] = gamma
                GAMMA[7:14,17:31] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    IC_test[i][5] += du[0]
                    A_test[i] += du[1:4]
                    B_test[i] += du[4:7]
                    dt_test[i] += du[7]
                    beta_test[i] += du[8]
                elif i == 1:
                    IC_test[i][0:6] += du[9:15]
                    dt_test[i] += du[15]
                    beta_test[i] += du[16]
                else:
                    IC_test[i][0:6] += du[17:23]
                    A_test[i] += du[23:26]
                    B_test[i] += du[26:29]
                    dt_test[i] += du[29]
                    beta_test[i] += du[30]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:6] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                    error.append(shooterInfo['manTime'] - np.sum(dt_test))
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_s2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time state-to-orbit burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef[0:5]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            error.append(X_int[-1,0:5] - meoet)
            error.append(dt[i] - beta[i]**2)
            error.append(shooterInfo['manTime'] - np.sum(dt))
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    de3du1 = np.zeros((7,8)); de3du1[6,6] = -1
    de3du2 = np.zeros((7,8)); de3du2[6,6] = -1
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)         dt3 - b3^2
                 dt1 - b1^2  ]           dt2 - b2^2  ]        T - dt1 - dt2 - dt3]
        
        u1 = [A1                u2 = [p2(t0)            u3 = [p3(t0)
              B1                      f2(t0)                  f3(t0)
              dt1                     g2(t0)                  g3(t0)
              b1 ]                    h2(t0)                  h3(t0)
                                      k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
        
        E = [e1_(7), e2_(7), e3_(7)]; U = [u1_(8), u2_(8), u3_(14)]
        
        Gamma_(21x30) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
        
        Gamma_(21x30) = dE/dU = [de1/du1_(7x8) de1/du2_(7x8)       0_(7x14)
                                       0_(7x8) de2/du2_(7x8) de2/du3_(7x14)
                                 de3/du1_(7x8) de3/du2_(7x8) de3/du3_(7x14)]
        
        de1/du2_(7x8) =  [-1  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0
                           0  0  0 -1  0  0  0  0
                           0  0  0  0 -1  0  0  0
                           0  0  0  0  0 -1  0  0
                           0  0  0  0  0  0  0  0]
        
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        de3/du1_(7x8) =  [ 0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0 -1  0]
         
        de3/du2_(7x8) =  [ 0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0 -1  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x8) = [dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0           0           1            -2b1] 
        
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        
        de3/du3_(7x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3
                          0               0               0               0               0               0               0           0           -1           0   ]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((21,30))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 7; n = 14
                gamma = np.zeros((m,n))
                gamma[5,12] = 1. 
                gamma[5,13] = -2*beta_gnc[i]
                gamma[6,12] = -1.

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
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
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                # Looping over e
                if i == N-1:
                    for k in range(m-2): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter
                else:
                    for k in range(m-1): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:8] = gamma
                GAMMA[0:7,8:16] = de1du2
            elif i == N-1:
                GAMMA[14:21,16:30] = gamma
                GAMMA[14:21,0:8] = de3du1
                GAMMA[14:21,8:16] = de3du2
            else:
                GAMMA[7:14,8:16] = gamma
                GAMMA[7:14,16:30] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    A_test[i] += du[0:3]
                    B_test[i] += du[3:6]
                    dt_test[i] += du[6]
                    beta_test[i] += du[7]
                elif i == 1:
                    IC_test[i][0:6] += du[8:14]
                    dt_test[i] += du[14]
                    beta_test[i] += du[15]
                else:
                    IC_test[i][0:6] += du[16:22]
                    A_test[i] += du[22:25]
                    B_test[i] += du[25:28]
                    dt_test[i] += du[28]
                    beta_test[i] += du[29]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:5] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                    error.append(shooterInfo['manTime'] - np.sum(dt_test))
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            break

        # This ends predictor-corrector is inner loop doesn't reduce error_mag
        elif local_min:
            print('\nUnsuccessful Convergence :(')
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bcb_s2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time state-to-state burn-coast-burn trajectory.

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
    N = 3; segment_type = [1, 0, 1]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt); meoet = meoef
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation (don't need angle wrap correction for o2o)
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            if abs(X_int[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
                meoet[-1] += np.sign(X_int[-1][-2] - meoet[-1])*2*np.pi
            error.append(X_int[-1,0:6] - meoet)
            error.append(dt[i] - beta[i]**2)
            error.append(shooterInfo['manTime'] - np.sum(dt))
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du3 = np.zeros((7,14)); de2du3[0:6,0:6] = -np.eye(6)
    de3du1 = np.zeros((8,8)); de3du1[7,6] = -1
    de3du2 = np.zeros((8,8)); de3du2[7,6] = -1
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast-burn mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)   e2 = [p2(tf) - p3(t0)   e3 = [p3(tf) - pt
              f1(tf) - f2(t0)         f2(tf) - f3(t0)         f3(tf) - ft
              g1(tf) - g2(t0)         g2(tf) - g3(t0)         g3(tf) - gt
              h1(tf) - h2(t0)         h2(tf) - h3(t0)         h3(tf) - ht
              k1(tf) - k2(t0)         k2(tf) - k3(t0)         k3(tf) - kt
              L1(tf) - L2(t0)         L2(tf) - L3(t0)         L3(tf) - Lt
                 dt1 - b1^2  ]           dt2 - b2^2  ]        dt3 - b3^2
                                                              T - dt1 - dt2 - dt3]
        
        u1 = [A1                u2 = [p2(t0)            u3 = [p3(t0)
              B1                      f2(t0)                  f3(t0)
              dt1                     g2(t0)                  g3(t0)
              b1 ]                    h2(t0)                  h3(t0)
                                      k2(t0)                  k3(t0)
                                      L2(t0)                  L3(t0)
                                       dt2                     A3
                                        b2  ]                  B3
                                                               dt3
                                                               b3   ]
        
        E = [e1_(7), e2_(7), e3_(8)]; U = [u1_(8), u2_(8), u3_(14)]
        
        Gamma_(22x30) = dE/dU = [de1/du1 de1/du2 de1/du3
                                 de2/du1 de2/du2 de2/du3
                                 de3/du1 de3/du2 de3/du3]
        
        Gamma_(22x30) = dE/dU = [de1/du1_(7x8) de1/du2_(7x8)       0_(7x14)
                                       0_(7x8) de2/du2_(7x8) de2/du3_(7x14)
                                 de3/du1_(8x8) de3/du2_(8x8) de3/du3_(8x14)]
        
        de1/du2_(7x8) =  [-1  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0
                           0  0  0 -1  0  0  0  0
                           0  0  0  0 -1  0  0  0
                           0  0  0  0  0 -1  0  0
                           0  0  0  0  0  0  0  0]
        
        de2/du3_(7x14) = [-1  0  0  0  0  0  0  0  0  0  0  0  0  0
                           0 -1  0  0  0  0  0  0  0  0  0  0  0  0
                           0  0 -1  0  0  0  0  0  0  0  0  0  0  0
                           0  0  0 -1  0  0  0  0  0  0  0  0  0  0
                           0  0  0  0 -1  0  0  0  0  0  0  0  0  0
                           0  0  0  0  0 -1  0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        
        de3/du1_(8x8) =  [ 0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0 -1  0]
         
        de3/du2_(8x8) =  [ 0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0  0  0
                           0  0  0  0  0  0 -1  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x8) = [dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0           0           1            -2b1] 
        
        de2/du2_(7x8) = [dp2(tf)/dp2(t0) dp2(tf)/df2(t0) dp2(tf)/dg2(t0) dp2(tf)/dh2(t0) dp2(tf)/dk2(t0) dp2(tf)/dL2(t0) dp2(tf)/ddt2 0
                         df2(tf)/dp2(t0) df2(tf)/df2(t0) df2(tf)/dg2(t0) df2(tf)/dh2(t0) df2(tf)/dk2(t0) df2(tf)/dL2(t0) df2(tf)/ddt2 0
                         dg2(tf)/dp2(t0) dg2(tf)/df2(t0) dg2(tf)/dg2(t0) dg2(tf)/dh2(t0) dg2(tf)/dk2(t0) dg2(tf)/dL2(t0) dg2(tf)/ddt2 0
                         dh2(tf)/dp2(t0) dh2(tf)/df2(t0) dh2(tf)/dg2(t0) dh2(tf)/dh2(t0) dh2(tf)/dk2(t0) dh2(tf)/dL2(t0) dh2(tf)/ddt2 0
                         dk2(tf)/dp2(t0) dk2(tf)/df2(t0) dk2(tf)/dg2(t0) dk2(tf)/dh2(t0) dk2(tf)/dk2(t0) dk2(tf)/dL2(t0) dk2(tf)/ddt2 0
                         dL2(tf)/dp2(t0) dL2(tf)/df2(t0) dL2(tf)/dg2(t0) dL2(tf)/dh2(t0) dL2(tf)/dk2(t0) dL2(tf)/dL2(t0) dL2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        
        de3/du3_(8x14) = [dp3(tf)/dp3(t0) dp3(tf)/df3(t0) dp3(tf)/dg3(t0) dp3(tf)/dh3(t0) dp3(tf)/dk3(t0) dp3(tf)/dL3(t0) dp3(tf)/dA3 dp3(tf)/dB3 dp3(tf)/ddt3 0
                          df3(tf)/dp3(t0) df3(tf)/df3(t0) df3(tf)/dg3(t0) df3(tf)/dh3(t0) df3(tf)/dk3(t0) df3(tf)/dL3(t0) df3(tf)/dA3 df3(tf)/dB3 df3(tf)/ddt3 0
                          dg3(tf)/dp3(t0) dg3(tf)/df3(t0) dg3(tf)/dg3(t0) dg3(tf)/dh3(t0) dg3(tf)/dk3(t0) dg3(tf)/dL3(t0) dg3(tf)/dA3 dg3(tf)/dB3 dg3(tf)/ddt3 0
                          dh3(tf)/dp3(t0) dh3(tf)/df3(t0) dh3(tf)/dg3(t0) dh3(tf)/dh3(t0) dh3(tf)/dk3(t0) dh3(tf)/dL3(t0) dh3(tf)/dA3 dh3(tf)/dB3 dh3(tf)/ddt3 0
                          dk3(tf)/dp3(t0) dk3(tf)/df3(t0) dk3(tf)/dg3(t0) dk3(tf)/dh3(t0) dk3(tf)/dk3(t0) dk3(tf)/dL3(t0) dk3(tf)/dA3 dk3(tf)/dB3 dk3(tf)/ddt3 0
                          dL3(tf)/dp3(t0) dL3(tf)/df3(t0) dL3(tf)/dg3(t0) dL3(tf)/dh3(t0) dL3(tf)/dk3(t0) dL3(tf)/dL3(t0) dL3(tf)/dA3 dL3(tf)/dB3 dL3(tf)/ddt3 0
                          0               0               0               0               0               0               0           0           1            -2b3
                          0               0               0               0               0               0               0           0           -1           0   ]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((22,30))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            elif i == N-1: # Final Burn
                m = 8; n = 14
                gamma = np.zeros((m,n))
                gamma[6,12] = 1. 
                gamma[6,13] = -2*beta_gnc[i]
                gamma[7,12] = -1.

            else: # Coast
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oe, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
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
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                elif i == N-1: # Final Burn: oe, A, B, dt need to be perturbed
                    if j < 6:
                        # Final Burn ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    elif 6 <= j < 9:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-6]) + 1e-7
                        A_fd[j-6] += fd_parameter
                    elif 9 <= j < 12:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-9]) + 1e-7
                        B_fd[j-9] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oe, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                # Looping over e
                if i == N-1:
                    for k in range(m-2): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter
                else:
                    for k in range(m-1): 
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:8] = gamma
                GAMMA[0:7,8:16] = de1du2
            elif i == N-1:
                GAMMA[14:22,16:30] = gamma
                GAMMA[14:22,0:8] = de3du1
                GAMMA[14:22,8:16] = de3du2
            else:
                GAMMA[7:14,8:16] = gamma
                GAMMA[7:14,16:30] = de2du3
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    A_test[i] += du[0:3]
                    B_test[i] += du[3:6]
                    dt_test[i] += du[6]
                    beta_test[i] += du[7]
                elif i == 1:
                    IC_test[i][0:6] += du[8:14]
                    dt_test[i] += du[14]
                    beta_test[i] += du[15]
                else:
                    IC_test[i][0:6] += du[16:22]
                    A_test[i] += du[22:25]
                    B_test[i] += du[25:28]
                    dt_test[i] += du[28]
                    beta_test[i] += du[29]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    error.append(X_test[-1,0:6] - meoet)
                    error.append(dt_test[i] - beta_test[i]**2)
                    error.append(shooterInfo['manTime'] - np.sum(dt_test))
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bc_o2r_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-position intercept trajectory.

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

    # Position Target
    rt = af.meoe2rv(meoef, mu=shooterInfo['mu_host'])[0]
    
    # Getting segement initial conditions
    N = 2; segment_type = [1, 0]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt)
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            r_int = af.meoe2rv(X_int[-1,0:6], mu=shooterInfo['mu_host'])[0]
            error.append(r_int - rt)
            error.append(dt[i] - beta[i]**2)
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)    e2 = [x2(tf) - xt
              f1(tf) - f2(t0)          y2(tf) - yt
              g1(tf) - g2(t0)          z2(tf) - zt
              h1(tf) - h2(t0)             dt2 - b2^2]
              k1(tf) - k2(t0)        
              L1(tf) - L2(t0)         
                 dt1 - b1^2  ]
        
        u1 = [L1(t0)            u2 = [p2(t0)
               A1                     f2(t0)
               B1                     g2(t0)
               dt1                    h2(t0)
               b1   ]                 k2(t0) 
                                      L2(t0)
                                       dt2
                                        b2  ]
        
        E = [e1_(7), e2_(4)]; U = [u1_(9), u2_(8)]
        
        Gamma_(11x17) = dE/dU = [de1/du1 de1/du2
                                 de2/du1 de2/du2]
        
        Gamma_(11x17) = dE/dU = [de1/du1_(7x9) de1/du2_(7x8)
                                       0_(4x9) de2/du2_(4x8)]
        
        de1/du2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x9) = [dp1(tf)/dL1(t0) dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dL1(t0) df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dL1(t0) dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dL1(t0) dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dL1(t0) dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dL1(t0) dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0               0           0           1            -2b1] 
        
        de2/du2_(4x8) = [dx2(tf)/dp2(t0) dx2(tf)/df2(t0) dx2(tf)/dg2(t0) dx2(tf)/dh2(t0) dx2(tf)/dk2(t0) dx2(tf)/dL2(t0) dx2(tf)/ddt2 0
                         dy2(tf)/dp2(t0) dy2(tf)/df2(t0) dy2(tf)/dg2(t0) dy2(tf)/dh2(t0) dy2(tf)/dk2(t0) dy2(tf)/dL2(t0) dy2(tf)/ddt2 0
                         dz2(tf)/dp2(t0) dz2(tf)/df2(t0) dz2(tf)/dg2(t0) dz2(tf)/dh2(t0) dz2(tf)/dk2(t0) dz2(tf)/dL2(t0) dz2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((11,17))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 9
                gamma = np.zeros((m,n))
                gamma[6,7] = 1. 
                gamma[6,8] = -2*beta_gnc[i]

            else: # Coast
                m = 4; n = 8
                gamma = np.zeros((m,n))
                gamma[3,6] = 1. 
                gamma[3,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oes, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
                    if j == 0:
                        # Initial Burn L0
                        fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                        IC_fd[j+5] += fd_parameter
                    elif 1 <= j < 4:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                        A_fd[j-1] += fd_parameter
                    elif 4 <= j < 7:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                        B_fd[j-4] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oes, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                for k in range(m-1): # Looping over e
                    if i == N-1:
                        r_fd = af.meoe2rv(X_fd[-1,0:6], mu=shooterInfo['mu_host'])[0]
                        r_gnc = af.meoe2rv(Xf_gnc[-1][0:6], mu=shooterInfo['mu_host'])[0]
                        diff = r_fd[k] - r_gnc[k]
                    else:
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                    gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:9] = gamma
                GAMMA[0:7,9:17] = de1du2
            else:
                GAMMA[7:11,9:17] = gamma
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    IC_test[i][5] += du[0]
                    A_test[i] += du[1:4]
                    B_test[i] += du[4:7]
                    dt_test[i] += du[7]
                    beta_test[i] += du[8]
                else:
                    IC_test[i][0:6] += du[9:15]
                    dt_test[i] += du[15]
                    beta_test[i] += du[16]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    r_test = af.meoe2rv(X_test[-1,0:6], mu=shooterInfo['mu_host'])[0]
                    error.append(r_test - rt)
                    error.append(dt_test[i] - beta_test[i]**2)
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bc_s2r_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-position intercept trajectory.

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

    # Position Target
    rt = af.meoe2rv(meoef, mu=shooterInfo['mu_host'])[0]
    
    # Getting segement initial conditions
    N = 2; segment_type = [1, 0]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt)
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            r_int = af.meoe2rv(X_int[-1,0:6], mu=shooterInfo['mu_host'])[0]
            error.append(r_int - rt)
            error.append(dt[i] - beta[i]**2)
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)    e2 = [x2(tf) - xt
              f1(tf) - f2(t0)          y2(tf) - yt
              g1(tf) - g2(t0)          z2(tf) - zt
              h1(tf) - h2(t0)             dt2 - b2^2]
              k1(tf) - k2(t0)        
              L1(tf) - L2(t0)         
                 dt1 - b1^2  ]
        
        u1 = [A1                 u2 = [p2(t0)
              B1                       f2(t0)
              dt1                      g2(t0)
              b1 ]                     h2(t0)
                                       k2(t0) 
                                       L2(t0)
                                         dt2
                                         b2  ]
        
        E = [e1_(7), e2_(4)]; U = [u1_(8), u2_(8)]
        
        Gamma_(11x16) = dE/dU = [de1/du1 de1/du2
                                 de2/du1 de2/du2]
        
        Gamma_(11x16) = dE/dU = [de1/du1_(7x8) de1/du2_(7x8)
                                       0_(4x8) de2/du2_(4x8)]
        
        de1/du2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x8) = [dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0           0           1            -2b1] 
        
        de2/du2_(4x8) = [dx2(tf)/dp2(t0) dx2(tf)/df2(t0) dx2(tf)/dg2(t0) dx2(tf)/dh2(t0) dx2(tf)/dk2(t0) dx2(tf)/dL2(t0) dx2(tf)/ddt2 0
                         dy2(tf)/dp2(t0) dy2(tf)/df2(t0) dy2(tf)/dg2(t0) dy2(tf)/dh2(t0) dy2(tf)/dk2(t0) dy2(tf)/dL2(t0) dy2(tf)/ddt2 0
                         dz2(tf)/dp2(t0) dz2(tf)/df2(t0) dz2(tf)/dg2(t0) dz2(tf)/dh2(t0) dz2(tf)/dk2(t0) dz2(tf)/dL2(t0) dz2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((11,16))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            else: # Coast
                m = 4; n = 8
                gamma = np.zeros((m,n))
                gamma[3,6] = 1. 
                gamma[3,7] = -2*beta_gnc[i]

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oes, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
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
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oes, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                for k in range(m-1): # Looping over e
                    if i == N-1:
                        r_fd = af.meoe2rv(X_fd[-1,0:6], mu=shooterInfo['mu_host'])[0]
                        r_gnc = af.meoe2rv(Xf_gnc[-1][0:6], mu=shooterInfo['mu_host'])[0]
                        diff = r_fd[k] - r_gnc[k]
                    else:
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                    gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:8] = gamma
                GAMMA[0:7,8:16] = de1du2
            else:
                GAMMA[7:11,8:16] = gamma
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    A_test[i] += du[0:3]
                    B_test[i] += du[3:6]
                    dt_test[i] += du[6]
                    beta_test[i] += du[7]
                else:
                    IC_test[i][0:6] += du[8:14]
                    dt_test[i] += du[14]
                    beta_test[i] += du[15]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    r_test = af.meoe2rv(X_test[-1,0:6], mu=shooterInfo['mu_host'])[0]
                    error.append(r_test - rt)
                    error.append(dt_test[i] - beta_test[i]**2)
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2r_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-state single-burn trajectory.

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

    # Position Target
    rt = af.meoe2rv(meoef, mu=shooterInfo['mu_host'])[0]
    
    # Getting segement initial conditions
    IC_pert = np.hstack((meoe0, shooterInfo['m0']))
    tspan = [shooterInfo['t0'], dt]
    X_pert = odeint(shooterInfo['ode_burn'], IC_pert, tspan, 
        args=shooterInfo['extras_burn'] + (A,B), 
        rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    beta = np.sqrt(dt)
    r_pert = af.meoe2rv(X_pert[-1,0:6], mu=shooterInfo['mu_host'])[0]
    error_vec.append(r_pert - rt)
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

        e = [x(tf) - xt         u = [L(t0)
             y(tf) - yt                A
             z(tf) - zt                B
              dt2 - b2^2]              dt
                                       b  ]
    
        Calculated with Finite Differencing

        Gamma_(4x9) = de/du = [dx(tf)/dL(t0) dx(tf)/dA dx(tf)/dB dx(tf)/ddt 0
                               dy(tf)/dL(t0) dy(tf)/dA dy(tf)/dB dy(tf)/ddt 0
                               dz(tf)/dL(t0) dz(tf)/dA dz(tf)/dB dz(tf)/ddt 0
                               0             0         0         1          -2b1] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 4; n = 9
        gamma = np.zeros((m,n))
        gamma[3,7] = 1. 
        gamma[3,8] = -2*beta_gnc

        # Finite Differencing
        for j in range(n-1): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B, dt)
            if j == 0:
                # Initial Burn L0
                fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                IC_fd[j+5] += fd_parameter
            elif 1 <= j < 4:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                A_fd[j-1] += fd_parameter
            elif 4 <= j < 7:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                B_fd[j-4] += fd_parameter
            else:
                # Time
                fd_parameter = 1e-6*abs(dt_fd) + 1e-7
                dt_fd += fd_parameter

            # Integration
            X_fd = odeint(shooterInfo['ode_burn'], IC_fd, [shooterInfo['t0'], dt_fd], 
                args=shooterInfo['extras_burn'] + (A_fd, B_fd), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            for k in range(m-1): # Looping over e
                r_fd = af.meoe2rv(X_fd[-1,0:6], mu=shooterInfo['mu_host'])[0]
                r_gnc = af.meoe2rv(Xf_gnc[0:6], mu=shooterInfo['mu_host'])[0]
                diff = r_fd[k] - r_gnc[k]
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
            IC_test[5] += du[0]
            A_test += du[1:4]
            B_test += du[4:7]
            dt_test += du[7]
            beta_test += du[8]

            # Integrating with new initial conditions
            X_test = odeint(shooterInfo['ode_burn'], IC_test, [shooterInfo['t0'], dt_test], 
                args=shooterInfo['extras_burn'] + (A_test, B_test), 
                rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

            # Calculating new error
            error_vec = []
            r_test = af.meoe2rv(X_test[-1,0:6], mu=shooterInfo['mu_host'])[0]
            error_vec.append(r_test - rt)
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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2r_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
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

    # Position Target
    rt = af.meoe2rv(meoef, mu=shooterInfo['mu_host'])[0]
    
    # Getting segement initial conditions
    IC_pert = np.hstack((meoe0, shooterInfo['m0']))
    tspan = [shooterInfo['t0'], dt]
    X_pert = odeint(shooterInfo['ode_burn'], IC_pert, tspan, 
        args=shooterInfo['extras_burn'] + (A,B), 
        rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    beta = np.sqrt(dt)
    r_pert = af.meoe2rv(X_pert[-1,0:6], mu=shooterInfo['mu_host'])[0]
    error_vec.append(r_pert - rt)
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

        e = [x(tf) - xt         u = [A
             y(tf) - yt              B
             z(tf) - zt              dt
              dt2 - b2^2]            b ]
    
        Calculated with Finite Differencing

        Gamma_(4x8) = de/du = [dx(tf)/dA dx(tf)/dB dx(tf)/ddt 0
                               dy(tf)/dA dy(tf)/dB dy(tf)/ddt 0
                               dz(tf)/dA dz(tf)/dB dz(tf)/ddt 0
                               0         0         1          -2b1] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 4; n = 8
        gamma = np.zeros((m,n))
        gamma[3,6] = 1. 
        gamma[3,7] = -2*beta_gnc

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
                r_fd = af.meoe2rv(X_fd[-1,0:6], mu=shooterInfo['mu_host'])[0]
                r_gnc = af.meoe2rv(Xf_gnc[0:6], mu=shooterInfo['mu_host'])[0]
                diff = r_fd[k] - r_gnc[k]
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
            r_test = af.meoe2rv(X_test[-1,0:6], mu=shooterInfo['mu_host'])[0]
            error_vec.append(r_test - rt)
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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bc_o2r_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-position intercept trajectory.

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

    # Position Target
    rt = af.meoe2rv(meoef, mu=shooterInfo['mu_host'])[0]
    
    # Getting segement initial conditions
    N = 2; segment_type = [1, 0]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt)
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            r_int = af.meoe2rv(X_int[-1,0:6], mu=shooterInfo['mu_host'])[0]
            error.append(r_int - rt)
            error.append(dt[i] - beta[i]**2)
            error.append(shooterInfo['manTime'] - np.sum(dt))
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du1 = np.zeros((5,9));  de2du1[4,7] = -1
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)    e2 = [x2(tf) - xt
              f1(tf) - f2(t0)          y2(tf) - yt
              g1(tf) - g2(t0)          z2(tf) - zt
              h1(tf) - h2(t0)          dt2 - b2^2
              k1(tf) - k2(t0)          T - dt1 - dt2]
              L1(tf) - L2(t0)         
                 dt1 - b1^2  ]
        
        u1 = [L1(t0)            u2 = [p2(t0)
               A1                     f2(t0)
               B1                     g2(t0)
               dt1                    h2(t0)
               b1   ]                 k2(t0) 
                                      L2(t0)
                                       dt2
                                        b2  ]
        
        E = [e1_(7), e2_(5)]; U = [u1_(9), u2_(8)]
        
        Gamma_(12x17) = dE/dU = [de1/du1 de1/du2
                                 de2/du1 de2/du2]
        
        Gamma_(12x17) = dE/dU = [de1/du1_(7x9) de1/du2_(7x8)
                                 de2/du1_(5x9) de2/du2_(5x8)]
        
        de1/du2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
        
        de1/du2_(5x9) = [ 0  0  0  0  0  0  0  0  0
                          0  0  0  0  0  0  0  0  0
                          0  0  0  0  0  0  0  0  0
                          0  0  0  0  0  0  0  0  0
                          0  0  0  0  0  0  0 -1  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x9) = [dp1(tf)/dL1(t0) dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dL1(t0) df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dL1(t0) dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dL1(t0) dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dL1(t0) dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dL1(t0) dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0               0           0           1            -2b1] 
        
        de2/du2_(5x8) = [dx2(tf)/dp2(t0) dx2(tf)/df2(t0) dx2(tf)/dg2(t0) dx2(tf)/dh2(t0) dx2(tf)/dk2(t0) dx2(tf)/dL2(t0) dx2(tf)/ddt2 0
                         dy2(tf)/dp2(t0) dy2(tf)/df2(t0) dy2(tf)/dg2(t0) dy2(tf)/dh2(t0) dy2(tf)/dk2(t0) dy2(tf)/dL2(t0) dy2(tf)/ddt2 0
                         dz2(tf)/dp2(t0) dz2(tf)/df2(t0) dz2(tf)/dg2(t0) dz2(tf)/dh2(t0) dz2(tf)/dk2(t0) dz2(tf)/dL2(t0) dz2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2
                         0               0               0               0               0               0               -1           0   ]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((12,17))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 9
                gamma = np.zeros((m,n))
                gamma[6,7] = 1. 
                gamma[6,8] = -2*beta_gnc[i]

            else: # Coast
                m = 5; n = 8
                gamma = np.zeros((m,n))
                gamma[3,6] = 1. 
                gamma[3,7] = -2*beta_gnc[i]
                gamma[4,6] = -1

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oes, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
                    if j == 0:
                        # Initial Burn L0
                        fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                        IC_fd[j+5] += fd_parameter
                    elif 1 <= j < 4:
                        # A BLT parameters
                        fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                        A_fd[j-1] += fd_parameter
                    elif 4 <= j < 7:
                        # B BLT parameters
                        fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                        B_fd[j-4] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oes, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                # Looping over e
                if i == N-1:
                    for k in range(m-2): 
                        r_fd = af.meoe2rv(X_fd[-1,0:6], mu=shooterInfo['mu_host'])[0]
                        r_gnc = af.meoe2rv(Xf_gnc[-1][0:6], mu=shooterInfo['mu_host'])[0]
                        diff = r_fd[k] - r_gnc[k]
                        gamma[k,j] = diff/fd_parameter
                else:
                    for k in range(m-1):
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:9] = gamma
                GAMMA[0:7,9:17] = de1du2
            else:
                GAMMA[7:12,9:17] = gamma
                GAMMA[7:12,0:9] = de2du1
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    IC_test[i][5] += du[0]
                    A_test[i] += du[1:4]
                    B_test[i] += du[4:7]
                    dt_test[i] += du[7]
                    beta_test[i] += du[8]
                else:
                    IC_test[i][0:6] += du[9:15]
                    dt_test[i] += du[15]
                    beta_test[i] += du[16]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    r_test = af.meoe2rv(X_test[-1,0:6], mu=shooterInfo['mu_host'])[0]
                    error.append(r_test - rt)
                    error.append(dt_test[i] - beta_test[i]**2)
                    error.append(shooterInfo['manTime'] - np.sum(dt_test))
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def bc_s2r_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-position intercept trajectory.

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

    # Position Target
    rt = af.meoe2rv(meoef, mu=shooterInfo['mu_host'])[0]
    
    # Getting segement initial conditions
    N = 2; segment_type = [1, 0]
    segICs = [np.hstack((meoe0, shooterInfo['m0']))]
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X = odeint(odefun, segICs[-1], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        segICs.append(X[-1])
    segICs = np.asarray(segICs)[0:-1]

    # Calculating Initial Error
    Xf0 = []; error_vec = []
    beta = np.sqrt(dt)
    for i in range(N):
        # Segement Type
        if segment_type[i] == 1:
            odefun = shooterInfo['ode_burn']
            extras = shooterInfo['extras_burn'] + (A[i], B[i])
        else:
            odefun = shooterInfo['ode_coast']
            extras = shooterInfo['extras_coast']
        # Segment Integration
        tspan = [shooterInfo['t0']+np.sum(dt[0:i]), shooterInfo['t0']+np.sum(dt[0:i+1])]
        X_int = odeint(odefun, segICs[i], tspan, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
        Xf0.append(X_int[-1])
        # Error Calculation
        error = []
        if i < N-1:
            error.append(X_int[-1,0:6] - segICs[i+1][0:6])
            error.append(dt[i] - beta[i]**2)
        else:
            r_int = af.meoe2rv(X_int[-1,0:6], mu=shooterInfo['mu_host'])[0]
            error.append(r_int - rt)
            error.append(dt[i] - beta[i]**2)
            error.append(shooterInfo['manTime'] - np.sum(dt))
        error_vec.append(np.hstack(error))
    Xf0 = np.asarray(Xf0)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(segICs)    # initial conditions of ms segments
    Xf_gnc = np.copy(Xf0)       # final values of ms segments
    A_gnc = np.copy(A)          # A BLT parameters
    B_gnc = np.copy(B)          # B BLT parameters
    dt_gnc = np.copy(dt)        # integration time of ms segments
    beta_gnc = np.copy(beta)    # time slack variables of ms segments

    de1du2 = np.zeros((7,8));  de1du2[0:6,0:6] = -np.eye(6)
    de2du1 = np.zeros((5,8));  de2du1[4,6] = -1

    tol = 1e-6; local_min = False
    count = 1; count_tol = 100; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    
    while True:
        """
        This burn-coast mulitple shooter algorithm uses finite 
        differencing to find Gamma, which maps changes in the control 
        to changes in the error vector to null the error vector. The 
        control vectors, error vectors, and Gamma (the Jacobian matrix 
        dE/dU) are shown below.

        e1 = [p1(tf) - p2(t0)    e2 = [x2(tf) - xt
              f1(tf) - f2(t0)          y2(tf) - yt
              g1(tf) - g2(t0)          z2(tf) - zt
              h1(tf) - h2(t0)             dt2 - b2^2
              k1(tf) - k2(t0)          T - dt1 - dt2]
              L1(tf) - L2(t0)         
                 dt1 - b1^2  ]
        
        u1 = [A1                 u2 = [p2(t0)
              B1                       f2(t0)
              dt1                      g2(t0)
              b1 ]                     h2(t0)
                                       k2(t0) 
                                       L2(t0)
                                         dt2
                                         b2  ]
        
        E = [e1_(7), e2_(5)]; U = [u1_(8), u2_(8)]
        
        Gamma_(12x16) = dE/dU = [de1/du1 de1/du2
                                 de2/du1 de2/du2]
        
        Gamma_(12x16) = dE/dU = [de1/du1_(7x8) de1/du2_(7x8)
                                       0_(5x8) de2/du2_(5x8)]
        
        de1/du2_(7x8) = [-1  0  0  0  0  0  0  0
                          0 -1  0  0  0  0  0  0
                          0  0 -1  0  0  0  0  0
                          0  0  0 -1  0  0  0  0
                          0  0  0  0 -1  0  0  0
                          0  0  0  0  0 -1  0  0
                          0  0  0  0  0  0  0  0]
        
        de1/du2_(5x8) = [ 0  0  0  0  0  0  0  0
                          0  0  0  0  0  0  0  0
                          0  0  0  0  0  0  0  0
                          0  0  0  0  0  0  0  0
                          0  0  0  0  0  0 -1  0]
        
        Calculated with Finite Differencing
        
        de1/du1_(7x8) = [dp1(tf)/dA1 dp1(tf)/dB1 dp1(tf)/ddt1 0
                         df1(tf)/dA1 df1(tf)/dB1 df1(tf)/ddt1 0
                         dg1(tf)/dA1 dg1(tf)/dB1 dg1(tf)/ddt1 0
                         dh1(tf)/dA1 dh1(tf)/dB1 dh1(tf)/ddt1 0
                         dk1(tf)/dA1 dk1(tf)/dB1 dk1(tf)/ddt1 0
                         dL1(tf)/dA1 dL1(tf)/dB1 dL1(tf)/ddt1 0
                         0           0           1            -2b1] 
        
        de2/du2_(4x8) = [dx2(tf)/dp2(t0) dx2(tf)/df2(t0) dx2(tf)/dg2(t0) dx2(tf)/dh2(t0) dx2(tf)/dk2(t0) dx2(tf)/dL2(t0) dx2(tf)/ddt2 0
                         dy2(tf)/dp2(t0) dy2(tf)/df2(t0) dy2(tf)/dg2(t0) dy2(tf)/dh2(t0) dy2(tf)/dk2(t0) dy2(tf)/dL2(t0) dy2(tf)/ddt2 0
                         dz2(tf)/dp2(t0) dz2(tf)/df2(t0) dz2(tf)/dg2(t0) dz2(tf)/dh2(t0) dz2(tf)/dk2(t0) dz2(tf)/dL2(t0) dz2(tf)/ddt2 0
                         0               0               0               0               0               0               1            -2b2
                         0               0               0               0               0               0               -1           0   ]
        """

        # -------------------------------------------------------
        # Making Giant Matrix

        GAMMA = np.zeros((12,16))
        for i in range(N):
            # Determining the size of indiviual partial matrices
            if i == 0: # Initial Burn
                m = 7; n = 8
                gamma = np.zeros((m,n))
                gamma[6,6] = 1. 
                gamma[6,7] = -2*beta_gnc[i]

            else: # Coast
                m = 5; n = 8
                gamma = np.zeros((m,n))
                gamma[3,6] = 1. 
                gamma[3,7] = -2*beta_gnc[i]
                gamma[4,6] = -1

            # Finite Differencing
            for j in range(n-1): # looping over u

                # Control Parameters
                IC_fd = np.copy(IC_gnc[i])
                A_fd = np.copy(A_gnc[i])
                B_fd = np.copy(B_gnc[i])
                dt_fd = np.copy(dt_gnc)

                # Perturbing Control Parameters (order: oes, A, B, dt)
                if i == 0: # Initial Burn: L0, A, B, dt need to be perturbed
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
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                else: # coast arc: oes, dt need to be perturbed
                    if j < 6:
                        # Coast arc ICs
                        fd_parameter = 1e-6*abs(IC_fd[j]) + 1e-7
                        IC_fd[j] += fd_parameter
                    else:
                        # Time
                        fd_parameter = 1e-6*abs(dt_fd[i]) + 1e-7
                        dt_fd[i] += fd_parameter

                # Integration
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_fd, B_fd)
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']

                tspan_fd = [shooterInfo['t0']+np.sum(dt_fd[0:i]), shooterInfo['t0']+np.sum(dt_fd[0:i+1])]
                X_fd = odeint(odefun, IC_fd, tspan_fd, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])

                # Looping over e
                if i == N-1:
                    for k in range(m-2): 
                        r_fd = af.meoe2rv(X_fd[-1,0:6], mu=shooterInfo['mu_host'])[0]
                        r_gnc = af.meoe2rv(Xf_gnc[-1][0:6], mu=shooterInfo['mu_host'])[0]
                        diff = r_fd[k] - r_gnc[k]
                        gamma[k,j] = diff/fd_parameter
                else:
                    for k in range(m-1):
                        diff = X_fd[-1][k] - Xf_gnc[i][k]
                        gamma[k,j] = diff/fd_parameter

            if i == 0:
                GAMMA[0:7,0:8] = gamma
                GAMMA[0:7,8:16] = de1du2
            else:
                GAMMA[7:12,8:16] = gamma
                GAMMA[7:12,0:8] = de2du1
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Correction

        # Finding nominal control correction
        GAMMA_inv = GAMMA.transpose() @ np.linalg.inv(GAMMA @ GAMMA.transpose())
        du = -np.dot(GAMMA_inv, error_vec)/du_mod

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
            for i in range(N):
                if i == 0:
                    A_test[i] += du[0:3]
                    B_test[i] += du[3:6]
                    dt_test[i] += du[6]
                    beta_test[i] += du[7]
                else:
                    IC_test[i][0:6] += du[8:14]
                    dt_test[i] += du[14]
                    beta_test[i] += du[15]

            # Calculating Initial Error
            Xf0_test = []; error_vec = []
            for i in range(N):
                # Segement Type
                if segment_type[i] == 1:
                    odefun = shooterInfo['ode_burn']
                    extras = shooterInfo['extras_burn'] + (A_test[i], B_test[i])
                else:
                    odefun = shooterInfo['ode_coast']
                    extras = shooterInfo['extras_coast']
                # Updating Mass
                if i > 0:
                    IC_test[i][-1] = X_test[-1][-1]
                # Segment Integration
                tspan_test = [shooterInfo['t0']+np.sum(dt_test[0:i]), shooterInfo['t0']+np.sum(dt_test[0:i+1])]
                X_test = odeint(odefun, IC_test[i], tspan_test, args=extras, rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
                Xf0_test.append(X_test[-1])
                # Error Calculation
                error = []
                if i < N-1:
                    error.append(X_test[-1,0:6] - IC_test[i+1][0:6])
                    error.append(dt_test[i] - beta_test[i]**2)
                else:
                    r_test = af.meoe2rv(X_test[-1,0:6], mu=shooterInfo['mu_host'])[0]
                    error.append(r_test - rt)
                    error.append(dt_test[i] - beta_test[i]**2)
                    error.append(shooterInfo['manTime'] - np.sum(dt_test))
                error_vec.append(np.hstack(error))
            Xf0_test = np.asarray(Xf0_test)
            error_vec = np.hstack(error_vec)
            error_check = np.sqrt(error_vec.dot(error_vec))

            inner_count += 1

            # Inner loop stopping conditions
            if inner_count > inner_count_tol:
                local_min = True
                break

            elif error_check/error_mag <= 2:
                error_test.append(error_check)
                break

            elif error_check/error_mag > 2:
                print('\tReducing du by', du_reduction)
                du /= du_reduction

        error_mag = error_check
        IC_gnc = IC_test; A_gnc = A_test; B_gnc = B_test; dt_gnc = dt_test; beta_gnc = beta_test; Xf_gnc = Xf0_test
    
        # Stopping Conditions
        if error_mag < tol:
            print('\nSuccessful Convergence :)')
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2o_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-orbit single-burn trajectory.

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

        e = [p(tf) - pt         u = [L(t0)
             f(tf) - ft                A
             g(tf) - gt                B
             h(tf) - ht                dt
             k(tf) - kt                b  ]
               dt - b^2]
    
        Calculated with Finite Differencing

        Gamma_(6x9) = de/du = [dp(tf)/dL(t0) dp(tf)/dA dp(tf)/dB dp(tf)/ddt 0
                               df(tf)/dL(t0) df(tf)/dA df(tf)/dB df(tf)/ddt 0
                               dg(tf)/dL(t0) dg(tf)/dA dg(tf)/dB dg(tf)/ddt 0
                               dh(tf)/dL(t0) dh(tf)/dA dh(tf)/dB dh(tf)/ddt 0
                               dk(tf)/dL(t0) dk(tf)/dA dk(tf)/dB dk(tf)/ddt 0
                               0             0         0         1          -2b1] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 6; n = 9
        gamma = np.zeros((m,n))
        gamma[5,7] = 1. 
        gamma[5,8] = -2*beta_gnc

        # Finite Differencing
        for j in range(n-1): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B, dt)
            if j == 0:
                # Initial Burn L0
                fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                IC_fd[j+5] += fd_parameter
            elif 1 <= j < 4:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                A_fd[j-1] += fd_parameter
            elif 4 <= j < 7:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                B_fd[j-4] += fd_parameter
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
            IC_test[5] += du[0]
            A_test += du[1:4]
            B_test += du[4:7]
            dt_test += du[7]
            beta_test += du[8]

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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2s_freetime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-state single-burn trajectory.

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

        e = [p(tf) - pt         u = [L(t0)
             f(tf) - ft                A
             g(tf) - gt                B
             h(tf) - ht                dt
             k(tf) - kt                b  ]
             L(tf) - Lt  
               dt - b^2]
    
        Calculated with Finite Differencing

        Gamma_(7x9) = de/du = [dp(tf)/dL(t0) dp(tf)/dA dp(tf)/dB dp(tf)/ddt 0
                               df(tf)/dL(t0) df(tf)/dA df(tf)/dB df(tf)/ddt 0
                               dg(tf)/dL(t0) dg(tf)/dA dg(tf)/dB dg(tf)/ddt 0
                               dh(tf)/dL(t0) dh(tf)/dA dh(tf)/dB dh(tf)/ddt 0
                               dk(tf)/dL(t0) dk(tf)/dA dk(tf)/dB dk(tf)/ddt 0
                               dL(tf)/dL(t0) dL(tf)/dA dL(tf)/dB dL(tf)/ddt 0
                               0             0         0         1          -2b1] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 7; n = 9
        gamma = np.zeros((m,n))
        gamma[6,7] = 1. 
        gamma[6,8] = -2*beta_gnc

        # Finite Differencing
        for j in range(n-1): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B, dt)
            if j == 0:
                # Initial Burn L0
                fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                IC_fd[j+5] += fd_parameter
            elif 1 <= j < 4:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                A_fd[j-1] += fd_parameter
            elif 4 <= j < 7:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                B_fd[j-4] += fd_parameter
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
            IC_test[5] += du[0]
            A_test += du[1:4]
            B_test += du[4:7]
            dt_test += du[7]
            beta_test += du[8]

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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2o_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time orbit-to-orbit single-burn trajectory.

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

        e = [p(tf) - pt         u = [L(t0)
             f(tf) - ft                A
             g(tf) - gt                B  ]
             h(tf) - ht
             k(tf) - kt]
    
        Calculated with Finite Differencing

        Gamma_(5x7) = de/du = [dp(tf)/dL(t0) dp(tf)/dA dp(tf)/dB
                               df(tf)/dL(t0) df(tf)/dA df(tf)/dB
                               dg(tf)/dL(t0) dg(tf)/dA dg(tf)/dB
                               dh(tf)/dL(t0) dh(tf)/dA dh(tf)/dB
                               dk(tf)/dL(t0) dk(tf)/dA dk(tf)/dB] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 5; n = 7
        gamma = np.zeros((m,n))

        # Finite Differencing
        for j in range(n): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B)
            if j == 0:
                # Initial Burn L0
                fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                IC_fd[j+5] += fd_parameter
            elif 1 <= j < 4:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                A_fd[j-1] += fd_parameter
            elif 4 <= j < 7:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                B_fd[j-4] += fd_parameter
            else:
                # Time
                fd_parameter = 1e-6*abs(dt_fd) + 1e-7
                dt_fd += fd_parameter

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
            IC_test[5] += du[0]
            A_test += du[1:4]
            B_test += du[4:7]

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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2s_fixedtime_shooter(meoe0, meoef, A, B, dt, shooterInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time orbit-to-state single-burn trajectory.

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

        e = [p(tf) - pt         u = [L(t0)
             f(tf) - ft                A
             g(tf) - gt                B  ]
             h(tf) - ht
             k(tf) - kt
             L(tf) - Lt]
    
        Calculated with Finite Differencing

        Gamma_(6x7) = de/du = [dp(tf)/dL(t0) dp(tf)/dA dp(tf)/dB
                               df(tf)/dL(t0) df(tf)/dA df(tf)/dB
                               dg(tf)/dL(t0) dg(tf)/dA dg(tf)/dB
                               dh(tf)/dL(t0) dh(tf)/dA dh(tf)/dB
                               dk(tf)/dL(t0) dk(tf)/dA dk(tf)/dB
                               dL(tf)/dL(t0) dL(tf)/dA dL(tf)/dB] 
        """
        # -------------------------------------------------------
        # Calculating Gamma

        m = 6; n = 7
        gamma = np.zeros((m,n))

        # Finite Differencing
        for j in range(n): # looping over u

            # Control Parameters
            IC_fd = np.copy(IC_gnc)
            A_fd = np.copy(A_gnc)
            B_fd = np.copy(B_gnc)
            dt_fd = np.copy(dt_gnc)

            # Perturbing Control Parameters (order: oes, A, B)
            if j == 0:
                # Initial Burn L0
                fd_parameter = 1e-6*abs(IC_fd[j+5]) + 1e-7
                IC_fd[j+5] += fd_parameter
            elif 1 <= j < 4:
                # A BLT parameters
                fd_parameter = 1e-6*abs(A_fd[j-1]) + 1e-7
                A_fd[j-1] += fd_parameter
            elif 4 <= j < 7:
                # B BLT parameters
                fd_parameter = 1e-6*abs(B_fd[j-4]) + 1e-7
                B_fd[j-4] += fd_parameter
            else:
                # Time
                fd_parameter = 1e-6*abs(dt_fd) + 1e-7
                dt_fd += fd_parameter

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
            IC_test[5] += du[0]
            A_test += du[1:4]
            B_test += du[4:7]

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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
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
            converge = True
            break

        elif local_min:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        elif count > count_tol:
            print('\nUnsuccessful Convergence :(')
            converge = False
            break

        count += 1
        # -------------------------------------------------------
    # -----------------------------------------------------------

    return [IC_gnc], [Xf_gnc], A_gnc, B_gnc, dt_gnc, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def errfunManShift(L, meoe0, meoef, A, B, dt, shooterInfo):
    """This function calculates the norm of the total dV 
    associated with a Lambert trajectory. The transfer time dt is 
    the free variable.

    INPUT:
        L - free variable
        meoe0 - intitial meoe values
        meoef - target meoes
        A - BLT guidance parameter
        B - BLT guidance parameter
        dt - integration time
        extras - constants needed for integration

    OUTPUT:
        diff: the norm of the meoe difference
    """

    # Perturbed Trajectory
    ic = np.hstack((meoe0, shooterInfo['m0'])); ic[5] = L
    x = odeint(shooterInfo['ode_burn'], ic, [0, dt], 
        args=shooterInfo['extras_burn']+(A,B), rtol=shooterInfo['reltol'], atol=shooterInfo['abtol'])
    diff = x[-1][0:5] - meoef[0:5]

    return np.sqrt(diff.dot(diff))
# ---------------------------------------------------------------

# -------------------------------------------------------------------
def singleBurnInitialGuess(oe0, oef, dynInfo, mode):
    """This function finds the location and initial impulsive burn 
    guess for a single-burn maneuver

    INPUT:
        oe0 - orbital elements of intial orbit
        oef - orbital elements of final orbit
        dynInfo - dynamic information of central body
        mode - search mode: 'o2o', 'o2s', 's2o', 's2s'

    OUTPUT:
        oe0 - updated orbital elements of intial orbit
        oef - updated orbital elements of final orbit
        dv - impulsive intial guess
    """

    mu_host = dynInfo['mu_host']
    r_host = dynInfo['r_host']

    # Initial Guesses
    if mode == 's2s':
        r1 = af.oe2rv(oe0, mu_host)[0]
        r2 = af.oe2rv(oef, mu_host)[0]
        diff = r2 - r1
        costFun = [np.sqrt(diff.dot(diff))]

    elif mode == 'o2o':
        nu0_list = []
        nuSearch = np.linspace(0, 2*np.pi, 9)
        for i in range(1, len(nuSearch), 2):
            for j in range(1, len(nuSearch), 2):
                nu0_list.append([nuSearch[i], nuSearch[j]])
    
        bounds = [(0, 2*np.pi), (0, 2*np.pi)]
        args = (oe0, oef, mu_host, mode)
        nuOpt = []; costFun = []
        for nu0 in nu0_list:
            nuTemp = so.minimize(errfunClosestPoints, nu0, args=args)
            costFun.append(nuTemp.fun)
            nuOpt.append(nuTemp)
        nuOpt = nuOpt[costFun.index(min(costFun))]
        for val in nuOpt.x:
            while val > 2*np.pi:
                val -= 2*np.pi
            while val < 0:
                val += 2*np.pi
        oe0[-1] = nuOpt.x[0]; oef[-1] = nuOpt.x[-1]

    else:
        nu0_list = np.linspace(0, 2*np.pi, 5)
        args = (oe0, oef, mu_host, mode)
        nuOpt = []; costFun = []
        for nu0 in nu0_list:
            nuTemp, costFunTemp = of.brentSearch(errfunClosestPoints, nu0, [0, 2*np.pi], args)[0:2]
            costFun.append(costFunTemp)
            nuOpt.append(nuTemp)
        nuOpt = nuOpt[costFun.index(min(costFun))]
        if mode == 'o2s':
            oe0[-1] = nuOpt
        elif mode == 's2o':
            oef[-1] = nuOpt 

    if min(costFun) > r_host*1e6:
        print('\n\nClosest Distance is greater than r_host*1e6. Single Maneuver not advised.\n\n')
        dv = np.zeros(3)
    else:
        r1, v1 = af.oe2rv(oe0, mu_host)
        r2, v2 = af.oe2rv(oef, mu_host)
        dr_hat = (r2 - r1)/np.sqrt((r2 - r1).dot(r2 - r1))
        dv = v2 - v1

    return oe0, oef, dv
# -------------------------------------------------------------------

# -------------------------------------------------------------------
def errfunClosestPoints(nu, oe0, oef, mu, mode):
    """This error function returns the difference in position between
    two points.
    """

    if mode == 'o2o':
        oe0[-1] = nu[0]
        oef[-1] = nu[-1]
    else:
        # selecting mode
        if mode == 'o2s':
            oe0[-1] = nu
        elif mode == 's2o':
            oef[-1] = nu 
        else:
            print('\nIncorrect errfunClosestPoints Search mode!\nMust be "o2o", "o2s", or "s2o".\n')

    r1 = af.oe2rv(oe0, mu)[0]
    r2 = af.oe2rv(oef, mu)[0]
    diff = r2 - r1

    return np.sqrt(diff.dot(diff))
# -------------------------------------------------------------------

# ----------------------------------------------------
def lambertRetarget(meoe1, meoe2, dv1, dv2, dt, shooterInfo):
    """This function updates vt1 of the Lambert trajectory 
    to account for dynamic drift during the coast phase.

    INPUT:
        meoe1 - orbital elements of initial orbit
        meoe2 - orbital elements of final orbit
        vt1 - initial lambert transfer velocity
        dt - lambert transfer time
        shooterInfo - integration information
        r_norm - distance normalization factor

    OUTPUT:
        dv1 - updated initial maneuver
    """

    odefun = shooterInfo['ode_retarget']
    extras = shooterInfo['extras_coast']
    abtol = shooterInfo['abtol']
    reltol = shooterInfo['reltol']
    t0 = shooterInfo['t0']
    m0 = shooterInfo['m0']

    r0, v0 = af.meoe2rv(meoe1, mu=extras[0])
    rf, vf = af.meoe2rv(meoe2, mu=extras[0])

    # ------------------------------------------------
    ## Single Shooter ##

    rt = np.copy(rf)

    IC_pert = np.hstack((r0, v0+dv1, m0))
    tspan = [t0, dt]
    X_pert = odeint(odefun, IC_pert, tspan, args=extras, rtol=reltol, atol=abtol)

    # Calculating Initial Error
    error_vec = X_pert[-1][0:3] - rt
    error_mag = np.sqrt(error_vec.dot(error_vec))
    
    # Preparing shooter
    X_gnc = np.copy(X_pert)    # final values maneuver, needed for finite differencing
    dv_gnc = np.copy(dv1)       # maneuver
    dt_gnc = np.copy(dt)       # integration time of ms segments
    
    tol = 10/shooterInfo['r_norm']
    count = 1; count_tol = 9; inner_count_tol = 5
    du_reduction = 10.; du_mod = 1.
    if error_mag > tol:
        print('\n==============================================================')
        print('Lambert Re-target')
        print('Inital Pos Error:', '{:.4e}'.format(error_mag))
        while True:
            """
            This single shooter algorithm uses finite differencing to find 
            Gamma, which maps changes in the control to changes in the error 
            vector to null the error vector. The control vectors, error vectors, 
            and Gamma (the Jacobian matrix de/du) are shown below.
    
            e = [x(tf) - xt(t0)   u = [vx(t0)
                 y(tf) - yt(t0)        vy(t0)
                 z(tf) - zt(t0)]       vz(t0)
                                           dt]
    
            Gamma_(3x4) = dE/dU = [dx(tf)/dvx(t0) dx(tf)/dvy(t0) dx(tf)/dvz(t0) dx(tf)/ddt
                                   dy(tf)/dvx(t0) dy(tf)/dvy(t0) dy(tf)/dvz(t0) dy(tf)/ddt
                                   dz(tf)/dvx(t0) dz(tf)/dvy(t0) dz(tf)/dvz(t0) dz(tf)/ddt]
            """
            # -------------------------------------------------------
            # Calculating Gamma
    
            m = 3; n = 4
            gamma = np.zeros((m,n))
    
            # Finite Differencing
            for j in range(n): # looping over u
    
                # Control Parameters
                dv_fd = np.copy(dv_gnc)
                dt_fd = np.copy(dt_gnc)
    
                # Perturbing Control Parameters (order: v, dt)
                if j < n-1:
                    # impulsive maneuver
                    fd_parameter = 1e-6*abs(dv_fd[j]) + 1e-7
                    dv_fd[j] += fd_parameter
                else:
                    # Time
                    fd_parameter = 1e-6*abs(dt_fd) + 1e-7
                    dt_fd += fd_parameter
    
                # Integration
                IC_fd = np.hstack((r0, v0+dv_fd, m0))
                X_fd = odeint(odefun, IC_fd, [t0, dt_fd], args=extras, rtol=reltol, atol=abtol)
    
                for k in range(m): # Looping over e
                    diff = X_fd[-1][k] - X_gnc[-1][k]
                    gamma[k,j] = diff/fd_parameter
            # -------------------------------------------------------
    
            # -------------------------------------------------------
            # Correction
    
            # Finding nominal control correction
            gamma_inv = gamma.transpose() @ np.linalg.inv(gamma @ gamma.transpose())
            du = -np.dot(gamma_inv, error_vec)/du_mod
    
            # Finding Correction
            inner_count = 0
            error_test = [error_mag]
            while True:
    
                # Control Parameters
                dv_test = np.copy(dv_gnc)
                dt_test = np.copy(dt_gnc)
    
                # Applying Updates
                dv_test += du[0:3]
                dt_test += du[3]
    
                # Integrating with new initial conditions
                IC_test = np.hstack((r0, v0+dv_test, m0))
                X_test = odeint(odefun, IC_test, [t0, dt_test], args=extras, rtol=reltol, atol=abtol)
    
                # Calculating new error
                error_vec = X_test[-1][0:3] - rt
                error_check = np.sqrt(error_vec.dot(error_vec))
    
                inner_count += 1
    
                # Inner loop stopping conditions
                if inner_count > inner_count_tol:
                    break
                elif error_check < error_mag:
                    error_test.append(error_check)
                    break
                elif error_check/error_mag > 1:
                    du /= du_reduction
    
            error_mag = error_check
            dv_gnc = dv_test; dt_gnc = dt_test; X_gnc = X_test
        
            # Stopping Conditions
            if error_mag < tol:
                break
            if count > count_tol:
                break
            count += 1
    
        IC = np.hstack((r0, v0+dv_gnc, m0))
        tspan = [t0, dt_gnc]
        X_rv_fin = odeint(odefun, IC, tspan, args=extras, rtol=reltol, atol=abtol)
    
        dv2 = vf - X_rv_fin[-1][3:6]

        # Re-dimensionalizing
        dv1_fin = dv_gnc*shooterInfo['r_norm']/shooterInfo['t_norm']
        dv2_fin = dv2*shooterInfo['r_norm']/shooterInfo['t_norm']
        dt_fin  = dt_gnc*shooterInfo['t_norm']

        print('Final Pos Error: ', '{:.4e}'.format(error_mag))
        print('==============================================================')

    else:
        dv1_fin = dv1*shooterInfo['r_norm']/shooterInfo['t_norm']
        dv2_fin = dv2*shooterInfo['r_norm']/shooterInfo['t_norm']
        dt_fin  = dt*shooterInfo['t_norm']

    return dv1_fin, dv2_fin, dt_fin
# ----------------------------------------------------