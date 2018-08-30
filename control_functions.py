
###########################
##   Control Functions   ##
###########################

"""This script provides a variety of useful control functions.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

    1) electric propulsion thrust
    2) finite burn acceleration
    3) finite burn mass flow
    4) Lambert control
    5) cross-product control
    6) 

"""

# Import Modules
import numpy as np
import constants as c
import astro_functions as af
import coord_trans as ct

# Define Functions
# =====================================================================
# 1) 
def ep_thrust(Isp, eff, P):
    """This function calculates the thrust magintude associated
    with electic propulsion.

    INPUT:
        Isp - Isp of engine in [sec] :: scalar
        eff - dimensionless efficiency of power supply :: scalar
        P - power supplied to engine in [W] :: scalar

    OUTPUT
        T - thrust magnitude in [N] :: scalar
    """

    # Acceleration from electic propulsion
    T = 2*eff*P/Isp/c.g0

    return T
# =====================================================================


# =====================================================================
# 2) 
def trust2accel(T, m):
    """This function calculates the acceleration magintude associated
    with finite propulsion systems.

    INPUT:
        m - mass of spacecraft in [kg] :: scalar
        T - thrust of the engine in [N] :: scalar

    OUTPUT
        a_t - acceleration magnitude in [km/s^2] :: scalar
    """

    # Acceleration 
    a_t = T/m/1000

    return a_t
# =====================================================================


# =====================================================================
# 3) 
def mass_flow(T, Isp):
    """This function calculates the mass flow rate associated with 
    finite propulsion systems.

    INPUT:
        T - thrust of the engine in [N] :: scalar
        Isp - Isp of engine in [sec] :: scalar

    OUTPUT
        m_dot - mass flow of spacecraft in [kg/s] :: scalar
    """

    # mass flow
    m_dot = T/c.g0/Isp

    return m_dot
# =====================================================================


# =====================================================================
# 4) 
def lambert_control(r_sc, v_sc, r_t, v_t, t_trans, a_t, DM=1,
    mu=c.mu_earth):
    """This function computes the control acceleration using 
    in the Lambert guidance algorithm.

    INPUT:
        r_sc - spacecraft position vector in [km] :: (,3) array
        v_sc - spacecraft velocity vector in [km] :: (,3) array
        r_t - desired position vector in [km] :: (,3) array
        v_t - desired velocity vector in [km] :: (,3) array
        t_trans - transfer time in [sec] :: scalar
        a_t - thrust acceleration magnitude in [km/s^2] :: scalar
        mu - gravitational parameter in [km^3/s^2] :: scalar
    
    OUTPUT:
        u - control accelerations in [km/s^2] :: (,3) array
        dv - lambert calculated instantaneous burn to reach 
             r_dest with t_trans in [km/s] :: (,3) array
    """

    # Angle between current position and desired position
    r_sc_mag = np.sqrt(r_sc.dot(r_sc))
    r_dest_mag = np.sqrt(r_dest.dot(r_dest))
    cos_ang = np.dot(r_sc, r_dest)/r_sc_mag/r_dest_mag

    # If this angle is -1 or 1 the lambert solver won't converge
    # in this scenario a Hohmann transfer is optimal (i.e., burning
    # in the S-direction)
    if abs(cos_ang) == 1:
        # orbital elements
        oe_sc = af.cart2kep(r_sc, v_sc, mu=mu)[0]
        oe_dest = af.cart2kep(r_t, v_t, mu=mu)[0]

        # hohmann transfer
        dv_a = af.hohmann(oe_sc[0], oe_dest[0], 
            r_sc_mag, r_dest_mag, mu=mu)[2]
        dv = np.array([0, dv_a, 0])
        dv_mag = dv_a

        # control acceleration
        a_rsw = np.array([0, a_t, 0])
        u = ct.CoordTrans('RSW', 'BCI', a_rsw, oe_sc)
        
    else:
        # Lambert's solution
        v0 = af.lambert_uv(r_sc, r_t, t_trans, DM=DM, mu=mu)[0]

        # Necessary velocity change
        dv = v0 - v_sc
        dv_mag = np.sqrt(dv.dot(dv))
        u = a_t*dv/dv_mag
        
    return u, dv, dv_mag
# =====================================================================


# =====================================================================
# 5)
def xproduct_control(r, v, at, num, constants, mu=c.mu_earth):
    """This function computes the control acceleration using 
    in the cross-product guidance algorithm.

    INPUT:
        r - position vector of spacecraft :: (,3) array
        v - velocity vector of spacecraft :: (,3) array
        a_t - thrust acceleration magnitude in [km/s^2] :: scalar
        num - determines the method used :: scalar
            1 - Battin's Lambert solution
            2 - Prussing and Conway's Lambert solution
            3 - Battin's constant energy solution
            4 - Battin's orbit solution
        mu - gravitational parameter of central body :: scalar
        constants - necessary variables for the methods
            rt - position vector of target [1, 2, 3]
            dt - transfer time [1, 2]
            DM - direction of motion [1, 2]
            a  - SMA of transfer orbit [3]
               - desired SMA [4]
            e  - desired ECC [4]
            sign - sign for certain parameters [4]

    OUTPUT
        vg - velocity to be gained :: (,3) array
        u - control accelerations in [km/s^2] :: (,3) array
    """

    # get vg and C
    vg, C = get_vg_C(r, v, num, constants, mu)

    # Control law
    i_vg = vg/np.sqrt(vg.dot(vg)) # vg unit vector
    p = -C @ vg
    p_mag = np.sqrt(p.dot(p))
    q = np.sqrt(at*at - p_mag*p_mag + np.dot(i_vg, p)**2)
    u = p + (q - np.dot(i_vg, p))*i_vg

    return u, vg

def get_vg_C(r, v, num, constants, mu):
    """This function calculates the 3x3 dvr/dr partial matrix 
    necessary and the necessary velocity to be gained for 
    cross-product control.

    INPUT:
        r - position vector of spacecraft :: (,3) array
        v - velocity vector of spacecraft :: (,3) array
        num - determines the method used :: scalar
            1 - Battin's Lambert solution
            2 - Prussing and Conway's Lambert solution
            3 - Battin's constant energy solution
            4 - Battin's orbit solution
        mu - gravitational parameter of central body :: scalar
        constants - necessary variables for the methods
            rt - position vector of target [1, 2, 3]
            dt - transfer time [1, 2]
            DM - direction of motion [1, 2]
            a  - SMA of transfer orbit [3]
               - desired SMA [4]
            e  - desired ECC [4]
            sign - sign for certain parameters [4]
            
    OUTPUT
        vg - velocity to be gained :: (,3) array
        C - dvr/dr partial matrix :: 3x3 matrix
    """
    
    # Setting up vectors and maginitudes
    I = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    r_mag = np.sqrt(r.dot(r)); i_r = r/r_mag 

    # Battin Lambert formulation
    if num == 1:

        # Unpacking constants
        rt = constants['TARGET']
        dt = constants['DT']
        DM = constants['DM']

        rt_mag = np.sqrt(rt.dot(rt))
        c_vec = rt - r; c = np.sqrt(c_vec.dot(c_vec)); i_c = c_vec/c
        s = (r_mag + rt_mag + c)/2

        # Required Velocity
        vr, _, _, _, a = af.lambert_pc(r, rt, dt, DM=DM, mu=mu)

        # Partial Matrix
        P = np.sqrt(mu/(r_mag + rt_mag - c) - mu/4/a)
        Q = np.sqrt(mu/(r_mag + rt_mag + c) - mu/4/a)
        delta = 3*P*Q*dt + (s - c)*Q - s*P

        C = ((P - Q)/r_mag - (P + Q)/c)*I + \
        (mu*Q/16/a/P/delta - mu/8/P/(s - c)**2)*np.array([i_c + i_r]) @ np.array([i_c + i_r]).transpose() + \
        (mu*P/16/a/Q/delta + mu/8/Q/s/s)*np.array([i_c - i_r]) @ np.array([i_c - i_r]).transpose() - \
        ((P - Q)/r_mag + mu/8/a/delta)*np.array([i_r]) @ np.array([i_r]).transpose() + \
        ((P + Q)/r_mag + mu/8/a/delta)*np.array([i_c]) @ np.array([i_c]).transpose()

    # Prussing and Conway Lambert partials
    elif num == 2:

        # Unpacking constants
        rt = constants['TARGET']
        dt = constants['DT']
        DM = constants['DM']

        rt_mag = np.sqrt(rt.dot(rt))
        c_vec = rt - r; c = np.sqrt(c_vec.dot(c_vec)); i_c = c_vec/c
        s = (r_mag + rt_mag + c)/2                  # semiparameter

        # Required Velocity
        vr, vf, alpha, beta, a = af.lambert_pc(r, rt, dt, DM=DM, mu=mu)

        # Partial Matrix
        dalpha_dr = 1/np.cos(alpha/2)/np.sqrt(8*a*s)*(np.array([i_r]).transpose() - np.array([i_c]).transpose())
        dbeta_dr = 1/np.cos(beta/2)/np.sqrt(8*a*(s-c))*(np.array([i_r]).transpose() + np.array([i_c]).transpose())
        dA_dr = -0.5*np.sqrt(mu/4/a)/np.sin(alpha/2)/np.sin(alpha/2)*dalpha_dr
        dB_dr = -0.5*np.sqrt(mu/4/a)/np.sin(beta/2)/np.sin(beta/2)*dbeta_dr

        # C matrix
        A = np.sqrt(mu/4/a)/np.tan(alpha/2)
        B = np.sqrt(mu/4/a)/np.tan(beta/2)

        C = i_c@(dB_dr + dA_dr) + (B+A)*(-I/c + i_c@np.array([i_c]).transpose()/c) + \
        i_r@(dB_dr - dA_dr) + (B-A)/r_mag*(I - i_r@np.array([i_r]).transpose())
        
    # Battin constant energy formulation
    elif num == 3:

        # Unpacking constants
        rt = constants['TARGET']
        a = constants['SMA']

        rt_mag = np.sqrt(rt.dot(rt))
        c_vec = rt - r; c = np.sqrt(c_vec.dot(c_vec)); i_c = c_vec/c
        s = (r_mag + rt_mag + c)/2

        # Required Velocity
        P = np.sqrt(mu/(r_mag + rt_mag - c) - mu/4/a)
        Q = np.sqrt(mu/(r_mag + rt_mag + c) - mu/4/a)
        vr = P*(i_c + i_r) + Q*(i_c - i_r)

        # Partial Matrix
        C = (-mu/8/P/(s - c)**2)*np.array([i_c + i_r])@np.array([i_c + i_r]).transpose() + \
        mu/8/Q/s/s*np.array([i_c - i_r])@np.array([i_c - i_r]).transpose() - \
        (P+Q)/c*(I - np.array([i_c])@np.array([i_c]).transpose()) + \
        (P-Q)/r_mag*(I - np.array([i_r])@np.array([i_r]).transpose())

    # Battin specified SMA and ECC
    elif num == 4:

        # Unpacking constants
        a = constants['SMA']
        e = constants['ECC']
        sign = constants['SIGN']
        i_h = constants['I_H']

        sp = a*(1 - e*e)

        # Required Velocity
        vr = sign*np.sqrt(mu/sp*(e*e - (sp/r_mag - 1)**2))*i_r + \
        np.sqrt(mu*sp)/r_mag*np.cross(i_h, i_r)

        # Partial Matrix
        C = sign*np.sqrt(mu*sp)/r_mag/r_mag/np.sqrt((r_mag*e/(sp - r_mag))**2 - 1)*i_r@np.array([i_r]).transpose() + \
        sign*np.sqrt(mu/sp*(e*e - (sp/r_mag - 1)**2))/r_mag*(I - i_r@np.array([i_r]).transpose()) - \
        np.sqrt(mu*sp)/r_mag/r_mag*np.cross(i_h, I - i_r@np.array([i_r]).transpose())

    else:
        print('ERROR: Input number not recognized!')
        vr = float('NaN')
        C = float('NaN')

    # Velocity to be gained
    vg = vr - v

    return vg, C
# =====================================================================

