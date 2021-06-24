
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
def altg(meoe0, meoef, t0, tf, A, B, mant0, simInfo, time='free', shooter='s2s', mode='b'):
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
        meoe0 - orbital elements of the initial orbit including mass
        meoef - orbital elements of the final orbit
        tf - nominal maneuver time
        A  - nominal A BLT parameter 
        B  - nominal B BLT parameter 
        simInfo - dictionary of necessary simulation characteristics
                m0  - mass
                Cr  - coefficient of reflectivity
                a2m - area to mass ratio
                isp - specific impulse
                T   - thrust
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
        time - flag indicating if total maneuver time is free or fixed
        shooter - control and error vector of shooter: 's2o', 's2s'

    OUTPUT:
        oe0_fin - initial orbital elements of each segment
        oef_fin - final orbital elements of each segment
        A_fin - final A guidance parameters
        B_fin - final B guidance parameters
        tf_fin - final segment times

    NOTES:
        shooter - s2o = state-to-orbit, s2s = state-to-state
        
    """
    ## Guidance ## 

    # -----------------------------------------------------------
    if mode == 'b':
        if time == 'free': # no time specified    
            if shooter == 's2o':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, errorRatio, converge = b_s2o_freetime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo)
    
            elif shooter == 's2s':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, errorRatio, converge = b_s2s_freetime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo)
    
            else:
                print('\nError in shooter flag! Please choose "s2o" or "s2s".\n')
    
        elif time == 'fixed':
            if shooter == 's2o':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, errorRatio, converge = b_s2o_fixetfime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo)
    
            elif shooter == 's2s':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, errorRatio, converge = b_s2s_fixetfime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo)
    
            else:
                print('\nError in shooter flag! Please choose "s2o" or "s2s".\n')
    
        else:
            print('\nError in guidance time flag! Please choose "free" or "fixed".\n')
    
        return IC_gnc, Xf_gnc, tf_gnc, A_gnc, B_gnc, errorRatio, converge 
    # -----------------------------------------------------------
    
    # -----------------------------------------------------------
    else:
        dt = tf - t0
        if time == 'free': # no time specified    
            if shooter == 'o2o':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, errorRatio, converge = b_o2o_freetime_shooter(meoe0, meoef, A, B, dt, simInfo)
    
            elif shooter == 'o2s':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, errorRatio, converge = b_o2s_freetime_shooter(meoe0, meoef, A, B, dt, simInfo)
    
            else:
                print('\nError in shooter flag! Please choose "o2o" or "o2s".\n')
    
        elif time == 'fixed':
            if shooter == 'o2o':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, errorRatio, converge = b_o2o_fixetfime_shooter(meoe0, meoef, A, B, dt, simInfo)
    
            elif shooter == 'o2s':
                IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, errorRatio, converge = b_o2s_fixetfime_shooter(meoe0, meoef, A, B, dt, simInfo)
    
            else:
                print('\nError in shooter flag! Please choose "o2o" or "o2s".\n')
    
        else:
            print('\nError in guidance time flag! Please choose "free" or "fixed".\n')
    
        return IC_gnc, Xf_gnc, dt_gnc, A_gnc, B_gnc, errorRatio, converge 
    # -----------------------------------------------------------
# -------------------------------------------------------------------

#######################################################################


#######################################################################
# Supporting Functions
""" 
    1) b_s2o_freetime_shooter    - b s2o free-time single-shooter
    2) b_s2s_freetime_shooter    - b s2s free-time single-shooter
    3) b_s2o_fixetfime_shooter   - b s2o fixed-time single-shooter
    4) b_s2s_fixetfime_shooter   - b s2s fixed-time single-shooter
"""

# ---------------------------------------------------------------
def b_o2o_freetime_shooter(meoe0, meoef, A, B, dt, simInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-orbit single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        simInfo - dictionary of all necessary integration constants

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
    IC_pert = np.hstack((meoe0, simInfo['m_current']))
    tspan = [0., dt]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    meoet = meoef[0:5]; beta = np.sqrt(dt)
    error_vec.append(X_pert[-1][0:5] - meoet)
    error_vec.append(dt - beta**2)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
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
    count = 1; count_tol = 200; inner_count_tol = 5
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
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [0., dt_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, 0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [0., dt_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, 0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2s_freetime_shooter(meoe0, meoef, A, B, dt, simInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time orbit-to-state single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        simInfo - dictionary of all necessary integration constants

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
    IC_pert = np.hstack((meoe0, simInfo['m_current']))
    tspan = [0., dt]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    meoet = meoef; beta = np.sqrt(dt)
    if abs(X_pert[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
        meoet[-1] += np.sign(X_pert[-1][-2] - meoet[-1])*2*np.pi
    error_vec.append(X_pert[-1][0:6] - meoet)
    error_vec.append(dt - beta**2)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
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
    count = 1; count_tol = 200; inner_count_tol = 5
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
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [0., dt_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, 0.), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [0., dt_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, 0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2o_fixedtime_shooter(meoe0, meoef, A, B, dt, simInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time orbit-to-orbit single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        simInfo - dictionary of all necessary integration constants

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
    IC_pert = np.hstack((meoe0, simInfo['m_current']))
    tspan = [0., dt]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    meoet = meoef[0:5]
    error_vec = X_pert[-1][0:5] - meoet
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
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
    count = 1; count_tol = 200; inner_count_tol = 5
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
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [0., dt_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, 0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [0., dt_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, 0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_o2s_fixedtime_shooter(meoe0, meoef, A, B, dt, simInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time orbit-to-state single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        dt - normalized segment times
        simInfo - dictionary of all necessary integration constants

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
    IC_pert = np.hstack((meoe0, simInfo['m_current']))
    tspan = [0., dt]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    meoet = meoef
    if abs(X_pert[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
        meoet[-1] += np.sign(X_pert[-1][-2] - meoet[-1])*2*np.pi
    error_vec = X_pert[-1][0:6] - meoet
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
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
    count = 1; count_tol = 200; inner_count_tol = 5
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
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [0., dt_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, 0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [0., dt_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, 0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, dt_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------


# ---------------------------------------------------------------
def b_s2o_freetime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-orbit single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        t0 - normalized segment initial time
        tf - normalized segment final time
        simInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        tf_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.copy(meoe0)
    tspan = [t0, tf]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,mant0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    meoet = meoef[0:5]; beta = np.sqrt(tf-t0)
    error_vec.append(X_pert[-1][0:5] - meoet)
    error_vec.append(tf-t0 - beta**2)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    tf_gnc = np.copy(tf)         # integration time of ms segments
    beta_gnc = np.copy(beta)     # time slack variable
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 200; inner_count_tol = 5
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
             g(tf) - gt              tf
             h(tf) - ht              b ]
             k(tf) - kt              
             tf-t0 - b^2]
    
        Calculated with Finite Differencing

        Gamma_(6x8) = de/du = [dp(tf)/dA dp(tf)/dB dp(tf)/dtf 0
                               df(tf)/dA df(tf)/dB df(tf)/dtf 0
                               dg(tf)/dA dg(tf)/dB dg(tf)/dtf 0
                               dh(tf)/dA dh(tf)/dB dh(tf)/dtf 0
                               dk(tf)/dA dk(tf)/dB dk(tf)/dtf 0
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
            tf_fd = np.copy(tf_gnc)

            # Perturbing Control Parameters (order: oes, A, B, tf)
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
                fd_parameter = 1e-6*abs(tf_fd) + 1e-7
                tf_fd += fd_parameter

            # Integration
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [t0, tf_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            tf_test = np.copy(tf_gnc)
            beta_test = np.copy(beta_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]
            tf_test += du[6]
            beta_test += du[7]

            # Integrating with new initial conditions
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [t0, tf_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

            # Calculating new error
            error_vec = []
            error_vec.append(X_test[-1][0:5] - meoet)
            error_vec.append(tf_test-t0 - beta_test**2)
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
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; tf_gnc = tf_test; beta_gnc = beta_test

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2s_freetime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo):
    """This function converges on the BLT guidance parameters for 
    a free-time state-to-state single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        t0 - normalized segment initial time
        tf - normalized segment final time
        simInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        tf_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.copy(meoe0)
    tspan = [t0, tf]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,mant0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    error_vec = []
    meoet = meoef; beta = np.sqrt(tf-t0)
    if abs(X_pert[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
        meoet[-1] += np.sign(X_pert[-1][-2] - meoet[-1])*2*np.pi
    error_vec.append(X_pert[-1][0:6] - meoet)
    error_vec.append(tf-t0 - beta**2)
    error_vec = np.hstack(error_vec)
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    tf_gnc = np.copy(tf)         # integration time of ms segments
    beta_gnc = np.copy(beta)     # time slack variable
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 200; inner_count_tol = 5
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
             g(tf) - gt              tf
             h(tf) - ht              b ]
             k(tf) - kt              
             L(tf) - Lt  
             tf-t0 - b^2]
    
        Calculated with Finite Differencing

        Gamma_(7x8) = de/du = [dp(tf)/dA dp(tf)/dB dp(tf)/dtf 0
                               df(tf)/dA df(tf)/dB df(tf)/dtf 0
                               dg(tf)/dA dg(tf)/dB dg(tf)/dtf 0
                               dh(tf)/dA dh(tf)/dB dh(tf)/dtf 0
                               dk(tf)/dA dk(tf)/dB dk(tf)/dtf 0
                               dL(tf)/dA dL(tf)/dB dL(tf)/dtf 0
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
            tf_fd = np.copy(tf_gnc)

            # Perturbing Control Parameters (order: oes, A, B, tf)
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
                fd_parameter = 1e-6*abs(tf_fd) + 1e-7
                tf_fd += fd_parameter

            # Integration
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [t0, tf_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            tf_test = np.copy(tf_gnc)
            beta_test = np.copy(beta_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]
            tf_test += du[6]
            beta_test += du[7]

            # Integrating with new initial conditions
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [t0, tf_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

            # Calculating new error
            error_vec = []
            error_vec.append(X_test[-1][0:6] - meoet)
            error_vec.append(tf_test-t0 - beta_test**2)
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
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; tf_gnc = tf_test; beta_gnc = beta_test

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2o_fixedtime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time state-to-orbit single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        t0 - normalized segment initial time
        tf - normalized segment final time
        simInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        tf_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.copy(meoe0)
    tspan = [t0, tf]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,mant0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    meoet = meoef[0:5]
    error_vec = X_pert[-1][0:5] - meoet
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    tf_gnc = np.copy(tf)         # integration time of ms segments
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 200; inner_count_tol = 5
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
            tf_fd = np.copy(tf_gnc)

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
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [t0, tf_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            tf_test = np.copy(tf_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]

            # Integrating with new initial conditions
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [t0, tf_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; tf_gnc = tf_test

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def b_s2s_fixedtime_shooter(meoe0, meoef, A, B, t0, tf, mant0, simInfo):
    """This function converges on the BLT guidance parameters for 
    a fixed-time state-to-state single-burn trajectory.

    INPUT:
        meoe0 - normalized MEOEs of initial orbit
        meoef - normalized MEOEs of final orbit
        A - normalized A BLT parameters
        B - normalized B BLT parameters
        t0 - normalized segment initial time
        tf - normalized segment final time
        simInfo - dictionary of all necessary integration constants

    OUTPUT:
        IC_gnc - Normilized initial conditions for the BCB segements
        Xf_gnc - Normilized final state for the BCB segements
        A_gnc - Normalized converged A BLT parameters
        B_gnc - Normalized converged B BLT parameters
        tf_gnc - Normalized converged segment times
    """
    
    # -----------------------------------------------------------
    ## Initial Integration ##
    
    # Getting segement initial conditions
    IC_pert = np.copy(meoe0)
    tspan = [t0, tf]
    X_pert = odeint(simInfo['scOdeBurn'], IC_pert, tspan, 
        args=simInfo['scExtrasBurn'] + (A,B,mant0), 
        rtol=simInfo['reltol'], atol=simInfo['abtol'])

    # Calculating Initial Error
    meoet = meoef
    if abs(X_pert[-1][-2] - meoet[-1]) >= np.pi: # This prevents the angle wrap issue
        meoet[-1] += np.sign(X_pert[-1][-2] - meoet[-1])*2*np.pi
    error_vec = X_pert[-1][0:6] - meoet
    error_mag = np.sqrt(error_vec.dot(error_vec))
    error_mag0 = np.copy(error_mag)
    print('\nInital Error:', '{:.4e}'.format(error_mag))
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    ## Multiple Shooter ##

    # Preparing shooter
    IC_gnc = np.copy(IC_pert)    # initial condition
    Xf_gnc = np.copy(X_pert[-1]) # final values maneuver, needed for finite differencing
    A_gnc = np.copy(A)           # A BLT parameters
    B_gnc = np.copy(B)           # B BLT parameters
    tf_gnc = np.copy(tf)         # integration time of ms segments
    
    tol = 1e-6; local_min = False
    count = 1; count_tol = 200; inner_count_tol = 5
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
            tf_fd = np.copy(tf_gnc)

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
            X_fd = odeint(simInfo['scOdeBurn'], IC_fd, [t0, tf_fd], 
                args=simInfo['scExtrasBurn'] + (A_fd, B_fd, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
            tf_test = np.copy(tf_gnc)

            # Applying Updates
            A_test += du[0:3]
            B_test += du[3:6]

            # Integrating with new initial conditions
            X_test = odeint(simInfo['scOdeBurn'], IC_test, [t0, tf_test], 
                args=simInfo['scExtrasBurn'] + (A_test, B_test, mant0), 
                rtol=simInfo['reltol'], atol=simInfo['abtol'])

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
        IC_gnc = IC_test; Xf_gnc = X_test[-1]; A_gnc = A_test; B_gnc = B_test; tf_gnc = tf_test

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

    return IC_gnc, Xf_gnc, A_gnc, B_gnc, tf_gnc, error_mag0/error_mag, converge
# ---------------------------------------------------------------
