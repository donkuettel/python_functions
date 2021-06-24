
#################################
##  Coordinate Transformations ##
#################################

"""This script provides a variety of useful coordinate transformations.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA
"""

# Import Modules
import constants as c
import numpy as np

#######################################################################
# Main Function Call

def CoordTrans(frame1, frame2, original_vec, oe=np.zeros(6), 
    theta_gst=float('NaN'), lla_gs=np.zeros(3), mu=c.mu_earth, 
    r_body=c.r_earth):
    """This function rotates 3xN matrix of vectors from one of the
    seven reference frames specified in this function to another. This
    function can only rotate the vectors at a single time (i.e., a 
    single state). You need to provide all of the necessary variables
    to rotate from one frame to another. 

    FRAMES:
        BCI  - Body-Centered Inertial
        BCBF - Body-Centered Body-Fixed
        RIC  - Satellite Radial
        NTW  - Satellite Normal
        PQW  - Perifocal 
        LLA  - geocentric Lat, Lon, Altitude
        SEZ  - Topocentric Horizon

    INPUT:
        frame1 - The coordinate system of original_vec
        frame2 - The frame original_vec is being rotated into
        original_vec - a 3xN matrix of vectors
        oe - A 6x1 vector of the spacecraft orbital elements in [km]
             and radians
        theta_gst - the angle between the BCBF and BCI coordinate 
                    systems in [rads]. Only 
    """

    # Orbital Elements
    a, e, inc, raan, w, nu = oe

    # Warnings
    oe_frames = ['ric', 'ntw', 'pqw']
    if any(frame in oe_frames for frame in (frame1, frame2)):
        if oe.dot(oe) == 0:
            print('ERROR: You forgot to define the orbital elements!')

    topocentric_frames = ['sez']
    if any(frame in topocentric_frames for frame in (frame1, frame2)):
        if lla_gs.dot(lla_gs) == 0:
            print('ERROR: You forgot lla for the ground stations!')

    # Coordinate System Logic
    if frame1.lower() == 'bci':
        if frame2.lower() == 'bcbf':
            rotated_vec = bci2bcbf(original_vec, theta_gst)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'ric':
            rotated_vec = bci2ric(original_vec, raan, inc, w, nu)
    
        elif frame2.lower() == 'ntw':
            rotated_vec = bci2ntw(original_vec, e, raan, inc, w, nu)
    
        elif frame2.lower() == 'pqw':
            rotated_vec = bci2pqw(original_vec, raan, inc, w)
    
        elif frame2.lower() == 'lla':
            rotated_vec1 = bci2bcbf(original_vec, theta_gst)
            rotated_vec = bcbf2lla(rotated_vec1, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'sez':
            rotated_vec1 = bci2bcbf(original_vec, theta_gst)
            rotated_vec = bcbf2sez(rotated_vec1, lla_gs, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')

        else:
            print('ERROR: Frame2 is not included in this function!')

    elif frame1.lower() == 'bcbf':
        if frame2.lower() == 'bci':
            rotated_vec = bcbf2bci(original_vec, theta_gst)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'ric':
            rotated_vec1 = bcbf2bci(original_vec, theta_gst)
            rotated_vec = bci2ric(rotated_vec1, raan, inc, w, nu)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'ntw':
            rotated_vec1 = bcbf2bci(original_vec, theta_gst)
            rotated_vec = bci2ntw(rotated_vec1, e, raan, inc, w, nu)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'pqw':
            rotated_vec1 = bcbf2bci(original_vec, theta_gst)
            rotated_vec = bci2pqw(rotated_vec1, raan, inc, w)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'lla':
            rotated_vec = bcbf2lla(original_vec, r_body=r_body)
    
        elif frame2.lower() == 'sez':
            rotated_vec = bcbf2sez(original_vec, lla_gs, r_body=r_body)

        else:
            print('ERROR: Frame2 is not included in this function!')

    elif frame1.lower() == 'ric':
        rotated_vec1 = ric2bci(original_vec, raan, inc, w, nu)
        if frame2.lower() == 'bcbf':
            rotated_vec = bci2bcbf(rotated_vec1, theta_gst)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'bci':
            rotated_vec = rotated_vec1
    
        elif frame2.lower() == 'ntw':
            rotated_vec = bci2ntw(rotated_vec1, e, raan, inc, w, nu)
    
        elif frame2.lower() == 'pqw':
            rotated_vec = bci2pqw(rotated_vec1, raan, inc, w)
    
        elif frame2.lower() == 'lla':
            rotated_vec2 = bci2bcbf(rotated_vec1, theta_gst)
            rotated_vec = bcbf2lla(rotated_vec2, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'sez':
            rotated_vec2 = bci2bcbf(rotated_vec1, theta_gst)
            rotated_vec = bcbf2sez(rotated_vec2, lla_gs, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')

        else:
            print('ERROR: Frame2 is not included in this function!')

    elif frame1.lower() == 'ntw':
        rotated_vec1 = ntw2bci(original_vec, e, raan, inc, w, nu)
        if frame2.lower() == 'bcbf':
            rotated_vec = bci2bcbf(rotated_vec1, theta_gst)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'ric':
            rotated_vec = bci2ric(rotated_vec1, raan, inc, w, nu)
    
        elif frame2.lower() == 'bci':
            rotated_vec = rotated_vec1
    
        elif frame2.lower() == 'pqw':
            rotated_vec = bci2pqw(rotated_vec1, raan, inc, w)
    
        elif frame2.lower() == 'lla':
            rotated_vec2 = bci2bcbf(rotated_vec1, theta_gst)
            rotated_vec = bcbf2lla(rotated_vec2, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'sez':
            rotated_vec2 = bci2bcbf(rotated_vec1, theta_gst)
            rotated_vec = bcbf2sez(rotated_vec2, lla_gs, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')

        else:
            print('ERROR: Frame2 is not included in this function!')

    elif frame1.lower() == 'pqw':
        rotated_vec1 = pqw2bci(original_vec, raan, inc, w)
        if frame2.lower() == 'bcbf':
            rotated_vec = bci2bcbf(rotated_vec1, theta_gst)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'ric':
            rotated_vec = bci2ric(rotated_vec1, raan, inc, w, nu)
    
        elif frame2.lower() == 'ntw':
            rotated_vec = bci2ntw(rotated_vec1, e, raan, inc, w, nu)
    
        elif frame2.lower() == 'bci':
            rotated_vec = rotated_vec1
    
        elif frame2.lower() == 'lla':
            rotated_vec2 = bci2bcbf(rotated_vec1, theta_gst)
            rotated_vec = bcbf2lla(rotated_vec2, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'sez':
            rotated_vec2 = bci2bcbf(rotated_vec1, theta_gst)
            rotated_vec = bcbf2sez(rotated_vec2, lla_gs, r_body=r_body)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')

        else:
            print('ERROR: Frame2 is not included in this function!')

    elif frame1.lower() == 'lla':
        rotated_vec1 = lla2bcbf(original_vec, r_body=r_body)
        if frame2.lower() == 'bcbf':
            rotated_vec = rotated_vec1
    
        elif frame2.lower() == 'ric':
            rotated_vec2 = bcbf2bci(rotated_vec1, theta_gst)
            rotated_vec = bci2ric(rotated_vec2, raan, inc, w, nu)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'ntw':
            rotated_vec2 = bcbf2bci(rotated_vec1, theta_gst)
            rotated_vec = bci2ntw(rotated_vec2, e, raan, inc, w, nu)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'pqw':
            rotated_vec2 = bcbf2bci(rotated_vec1, theta_gst)
            rotated_vec = bci2pqw(rotated_vec2, raan, inc, w)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'bci':
            rotated_vec = bcbf2bci(rotated_vec1, theta_gst)
            if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
    
        elif frame2.lower() == 'sez':
            rotated_vec = bcbf2sez(rotated_vec1, lla_gs, r_body=r_body)

        else:
            print('ERROR: Frame2 is not included in this function!')

    elif frame1.lower() == 'sez':
        rotated_vec1 = sez2bcbf(original_vec, lla_gs, r_body=r_body)
        rotated_vec2 = bcbf2bci(rotated_vec1, theta_gst)
        if np.isnan(theta_gst):
                print('ERROR: You forgot to define theta_gst!')
                
        if frame2.lower() == 'bcbf':
            rotated_vec = rotated_vec1
    
        elif frame2.lower() == 'ric':
            rotated_vec = bci2ric(rotated_vec2, raan, inc, w, nu)
    
        elif frame2.lower() == 'ntw':
            rotated_vec = bci2ntw(rotated_vec2, e, raan, inc, w, nu)
    
        elif frame2.lower() == 'pqw':
            rotated_vec = bci2pqw(rotated_vec2, raan, inc, w)
    
        elif frame2.lower() == 'lla':
            rotated_vec = bcbf2lla(rotated_vec1, r_body=r_body)
    
        elif frame2.lower() == 'bci':
            rotated_vec = rotated_vec2

        else:
            print('ERROR: Frame2 is not included in this function!')

    else:
        print('ERROR: Frame1 is not included in this function!')

    return rotated_vec

#######################################################################


#######################################################################
# Supporting Functions
""" 
    1) R1       - Rotation about x-axis
    2) R2       - Rotation about y-axis
    3) R3       - Rotation about z-axis
    4) bcbf2bci - Rotating from BCBF to BCI
    5) bci2bcbf - Rotating from BCI to BCBF
    6) ric2bci  - Rotating from Satellite Radial (RIC) to BCI
    7) bci2ric  - Rotating from BCI to Satellite Radial (RIC)
    7) ntw2bci  - Rotating from Satellite Normal (NTW) to BCI
    9) bci2ntw  - Rotating from BCI to Satellite Normal (NTW)
   10) lla2bcbf - Rotating from lat, lon, alt (LLA) to BCBF
   11) bcbf2lla - Rotating from BCBF to lat, lon, alt (LLA)
   12) pqw2bci  - Rotating from PQW to BCI
   13) bci2pqw  - Rotating from BCI to PQW
   14) sez2ecef - Rotating from Topocentric Horizon (SEZ) to BCBF
   15) bcef2sez - Rotating from BCBF to Topocentric Horizon (SEZ)
   16)     
"""

# Function Definitions
# ===================================================================
# 1)
def R1(theta):
    """This function creates a DCM that preforms a CW rotation about
    the x-axis (1 axis).

    INPUT:
        theta - angle to rotate by in [rads]

    OUTPUT:
        DCM - DCM for CW 1st axis rotation
    """

    DCM = np.array([[1, 0, 0], 
        [0, np.cos(theta), np.sin(theta)], 
        [0, -np.sin(theta), np.cos(theta)]])

    return DCM
# ===================================================================


# ===================================================================
# 2)
def R2(theta):
    """This function creates a DCM that preforms a CW rotation about
    the y-axis (2 axis).

    INPUT:
        theta - Angle to rotate by in [rads]

    OUTPUT:
        DCM - DCM for CW 2nd axis rotation
    """

    DCM = np.array([[np.cos(theta), 0, -np.sin(theta)], 
        [0, 1, 0], 
        [np.sin(theta), 0, np.cos(theta)]])

    return DCM
# ===================================================================


# ===================================================================
# 3)
def R3(theta):
    """This function creates a DCM that preforms a rotation about
    the z-axis (3 axis).

    INPUT:
        theta - Angle to rotate by in [rads]

    OUTPUT:
        DCM - DCM for CW 3rd axis rotation
    """

    DCM = np.array([[np.cos(theta), np.sin(theta), 0], 
        [-np.sin(theta), np.cos(theta), 0], 
        [0, 0, 1]])

    return DCM
# ===================================================================


# ===================================================================
# 4)
def bcbf2bci(bcbf_vec, theta_gst):
    """This function takes a 3xN matrix of vectors in Body-Centered 
    Body-Fixed (BCBF) coordinates and rotates it to Body-Centered 
    Inertial (BCI) coordinates. Theta_GST is measured positive CCW.

    ASSUMPTIONS:
    - all angles in radians
    - BCI and BCBF coordinate frames share a common z-axis

    INPUT:
        bcbf_vec - a Nx3 matrix of vectors in the BCBF frame
        theta_gst - the angle between the BCBF and BCI coordinate 
                    systems in [rads]

    OUTPUT:
        bci_vec - a Nx3 matrix of vectors in BCI coordinates
    """

    bci_vec = R3(-theta_gst) @ bcbf_vec

    return bci_vec
# ===================================================================


# ===================================================================
# 5)
def bci2bcbf(bci_vec, theta_gst):
    """This function takes a 3xN matrix of vectors in Body-Centered 
    Inertial (BCI) coordinates and rotates it to Body-Centered 
    Body-Fixed (BCBF) coordinates. Theta_GST is measured positive 
    CCW.

    ASSUMPTIONS:
    - all angles in radians
    - BCI and BCBF coordinate frames share a common z-axis

    INPUT:
        bci_vec - a Nx3 matrix or vectors in the BCI frame
        theta_gst - the angle between the BCBF and BCI coordinate 
                    systems in [rads]

    OUTPUT:
        bcbf_vec - a Nx3 matrix of vectors in BCBF coordinates
    """

    bcbf_vec = R3(theta_gst) @ bci_vec

    return bcbf_vec
# ===================================================================


# ===================================================================
# 6) 
def ric2bci(ric_vec, raan, inc, w, nu):
    """This function takes a 3xN matrix of vectors in Satellite 
    Radial (RIC) coordinates and rotates it to Body-Centered 
    Inertial (BCI) coordinates.
    
    ASSUMPTIONS:
    - all angles in radians

    INPUT:
        ric_vec - a Nx3 matrix of vectors in the RIC frame
        raan - right ascencion of the ascending node in [rad]
        inc - inclination of the orbit in [rad]
        w - argument of periapsis in [rad]
        nu - true anomaly in [rad]

    OUTPUT:
        bci_vec - a Nx3 matrix of vectors in the BCI frame
    """

    # Checking for special orbit cases
    if np.isnan(w) == True:
        w = 0
    if np.isnan(raan) == True:
        raan = 0

    # Argument of Latitude
    u = w + nu

    bci_vec = R3(-raan) @ R1(-inc) @ R3(-u) @ ric_vec

    return bci_vec

# ===================================================================


# ===================================================================
# 7)
def bci2ric(bci_vec, raan, inc, w, nu):
    """This function converts a 3xN matrix of vectors in the
    Body-Centered Inertial (BCI) frame and rotates it to the 
    Satellite Radial (RIC) frame.
    
    ASSUMPTIONS:
    - all angles in radians

    INPUT:
        bci_vec - a Nx3 matrix of vectors in the BCI frame
        raan - right ascencion of the ascending node in [rad]
        inc - inclination of the orbit in [rad]
        w - argument of periapsis in [rad]
        nu - true anomaly in [rad]

    OUTPUT:
        ric_vec - a Nx3 matrix of vectors in the BCI frame
    """

    # Checking for special orbit cases
    if np.isnan(w) == True:
        w = 0
    if np.isnan(raan) == True:
        raan = 0

    # Argument of Latitude
    u = w + nu

    ric_vec = R3(u) @ R1(inc) @ R3(raan) @ bci_vec

    return ric_vec

# ===================================================================


# ===================================================================
# 8)
def ntw2bci(ntw_vec, e, raan, inc, w, nu):
    """This function takes a 3xN matrix of vectors in Satellite 
    Normal (NTW) coordinates and rotates it to Body-Centered 
    Inertial (BCI) coordinates.
    
    ASSUMPTIONS:
    - all angles in radians

    INPUT:
        ntw_vec - a Nx3 matrix of vectors in the NTW frame
        e - eccentricity
        raan - right ascencion of the ascending node in [rad]
        inc - inclination of the orbit in [rad]
        w - argument of periapsis in [rad]
        nu - true anomaly in [rad]

    OUTPUT:
        bci_vec - a Nx3 matrix of vectors in the BCI frame
    """

    # Checking for special orbit cases
    if np.isnan(w) == True: # circular orbit
        w = 0
    if np.isnan(raan) == True: # equatorial orbit
        raan = 0

    # Argument of Latitude
    u = w + nu

    cos_fpa = (1 + e*np.cos(nu))/np.sqrt(1 + 2*e*np.cos(nu) + e*e)
    sin_fpa = e*np.sin(nu)/np.sqrt(1 + 2*e*np.cos(nu) + e*e)
    fpa = np.arctan2(sin_fpa, cos_fpa)

    bci_vec = R3(-raan) @ R1(-inc) @ R3(-u) @ R3(fpa) @ ntw_vec

    return bci_vec
# ===================================================================


# ===================================================================
# 9)
def bci2ntw(bci_vec, e, raan, inc, w, nu):
    """This function converts a 3xN matrix of vectors in the
    Body-Centered Inertial (BCI) frame and rotates it to the 
    Satellite Normal (NTW) frame.
    
    ASSUMPTIONS:
    - all angles in radians

    INPUT:
        bci_vec - a Nx3 matrix of vectors in the BCI frame
        e - eccentricity
        raan - right ascencion of the ascending node in [rad]
        inc - inclination of the orbit in [rad]
        w - argument of periapsis in [rad]
        nu - true anomaly in [rad]

    OUTPUT:
        ntw_vec - a Nx3 matrix of vectors in the NTW frame
    """

    # Checking for special orbit cases
    if np.isnan(w) == True: # circular orbit
        w = 0
    if np.isnan(raan) == True: # equatorial orbit
        raan = 0

    # Argument of Latitude
    u = w + nu

    cos_fpa = (1 + e*np.cos(nu))/np.sqrt(1 + 2*e*np.cos(nu) + e*e)
    sin_fpa = e*np.sin(nu)/np.sqrt(1 + 2*e*np.cos(nu) + e*e)
    fpa = np.arctan2(sin_fpa, cos_fpa)

    ntw_vec = R3(-fpa) @ R3(u) @ R1(inc) @ R3(raan) @ bci_vec

    return ntw_vec
# ===================================================================


# ===================================================================
# 10)
def lla2bcbf(lla, r_body=c.r_earth):
    """This function converts a 3xN matrix of geocentric latitude, 
    longitude, and altitude vectors to Body-Centered Body-Fixed 
    (BCBF) frame position vectors.
    
    ASSUMPTIONS:
    - Host body is a perfect sphere
    - all angles in radians

    INPUT:
        lla - a Nx3 matrix of vectors with: 
            lat - geocentric latitude in [rad]
            lon - the longitude in [rad]
            alt - the altitude from sea-level in [km]
        r_body  - radius of host body (assumed to be Earth) 
                  in [km]

    OUTPUT:
        bcbf_vec - a Nx3 matrix of position vectors in the 
                   BCBF frame in [km]
    """

    # 3xN LLA Matrix
    try:
        bcbf_vec = np.zeros(lla.shape)
    
        for i in range(lla.shape[0]):
            lat = lla[i,0]
            lon = lla[i,1]
            alt = lla[i,2]
    
            # ECEF Position of the surface location
            bcbf_vec[i,:] = np.array(
                [(alt + r_body)*np.cos(lat)*np.cos(lon),
                (alt + r_body)*np.cos(lat)*np.sin(lon),
                (alt + r_body)*np.sin(lat)])

    # 3x1 LLA vector
    except:
        lat = lla[0]
        lon = lla[1]
        alt = lla[2]

        # ECEF Position of the surface location
        bcbf_vec = np.array(
            [(alt + r_body)*np.cos(lat)*np.cos(lon),
            (alt + r_body)*np.cos(lat)*np.sin(lon),
            (alt + r_body)*np.sin(lat)])

    return bcbf_vec
# ===================================================================


# ===================================================================
# 11)
def bcbf2lla(bcbf_vec, r_body=c.r_earth):
    """This function converts a 3xN matrix of Body-Centered 
    Body-Fixed (BCBF) frame position vectors to N geocentric 
    latitude, longitude, and altitude vectors.
    
    ASSUMPTIONS:
    - Host body is a perfect sphere
    - all angles in radians

    INPUT:
        r_body - radius of host body (assumed to be Earth) in [km]
        bcbf_vec - a Nx3 matrix of position vectors in the 
                   BCBF frame in [km]

    OUTPUT:
        lla - a Nx3 matrix of vectors with: 
            lat - geocentric latitude in [rad]
            lon - the longitude in [rad]
            alt - the altitude from sea-level in [km]
    """

    # 3xN BCBF Position Matrix
    try:
        lla = np.zeros(bcbf_vec.shape)
    
        for i in range(bcbf_vec.shape[0]):
            x = bcbf_vec[i,0]
            y = bcbf_vec[i,1]
            z = bcbf_vec[i,2]
            r_mag = np.sqrt(x*x + y*y + z*z)
    
            # lla
            lat = np.arcsin(z/r_mag)
                   
            lon = np.arctan2(y, x)
            # if lon < 0:
            #     lon += 2*np.pi
                   
            alt = r_mag - r_body
    
            lla[i,:] = np.array([lat, lon, alt])

    # 3x1 BCBF Position Vector
    except:
        x = bcbf_vec[0]
        y = bcbf_vec[1]
        z = bcbf_vec[2]
        r_mag = np.sqrt(x*x + y*y + z*z)
        
        # lla
        lat = np.arcsin(z/r_mag)
        
        lon = np.arctan2(y, x)
        # if lon < 0:
        #     lon += 2*np.pi
        
        alt = r_mag - r_body
        lla = np.array([lat, lon, alt])

    return lla
# ===================================================================


# ===================================================================
# 12) pqw3bci
def pqw2bci(pqw_vec, raan, inc, w):
    """This function takes a 3xN matrix of vectors in Perifocal 
    (PQW) coordinates and rotates it to Body-Centered Inertial 
    (BCI) coordinates.
    
    ASSUMPTIONS:
    - all angles in radians

    INPUT:
        ric_vec - a Nx3 matrix of vectors in the RIC frame
        raan - right ascencion of the ascending node in [rad]
        inc - inclination of the orbit in [rad]
        w - argument of periapsis in [rad]
        nu - true anomaly in [rad]

    OUTPUT:
        bci_vec - a Nx3 matrix of vectors in the BCI frame
    """

    # Checking for special orbit cases
    if np.isnan(w) == True: # circular orbit
        w = 0
        print('PWQ is not well defined for circular orbits!')
    if np.isnan(raan) == True: # equatorial orbit
        raan = 0

    bci_vec = R3(-raan) @ R1(-inc) @ R3(-w) @ pqw_vec

    return bci_vec

# ===================================================================


# ===================================================================
# 13) bci2pqw
def bci2pqw(bci_vec, raan, inc, w):
    """This function takes a 3xN matrix of vectors in Body-Centered 
    Inertial (BCI) coordinates and rotates it to Perifocal (PQW) 
    coordinates.

    ASSUMPTIONS:
    - all angles in radians

    INPUT:
        bci_vec - a Nx3 matrix of vectors in the BCI frame    
        raan - right ascencion of the ascending node in [rad]
        inc - inclination of the orbit in [rad]
        w - argument of periapsis in [rad]
        nu - true anomaly in [rad]

    OUTPUT:
        ric_vec - a Nx3 matrix of vectors in the RIC frame
    """

    # Checking for special orbit cases
    if np.isnan(w) == True: # circular orbit
        w = 0
        print('PWQ is not well defined for circular orbits!')
    if np.isnan(raan) == True: # equatorial orbit
        raan = 0

    pqw_vec = R3(w) @ R1(inc) @ R3(raan) @ bci_vec

    return pqw_vec
# ===================================================================


# ===================================================================
# 14)
def sez2bcbf(razel, lla, r_body=c.r_earth):
    """This function takes the range, azimuth, and elevation of a 
    spacecraft and the geocentric latitude, longitude, and altitude 
    of a seperate location and computes the postion of the 
    spacecraft wrt the location in the Body-Centered Body-Fixed 
    (BCBF) frame.

    ASSUMPTIONS:
    - Orbiting body is a perfect sphere
    - all angles in radians
    - razel and lla are the same size

    INPUT:
        razel - a Nx3 matrix of vectors with:
            rng - the distance of the spacecraft from the ground 
                  location in [km]
            az  - the azimuth of the spacecraft wrt the ground 
            location in [rad]
            el -  elevation angle of the spacecraft wrt the ground 
                  location in [rad]         
        lla - a Nx3 matrix of vectors with: 
            lat - geocentric latitude in [rad]
            lon - the longitude in [rad]
            alt - the altitude from sea-level in [km]
        r_body  - radius of host body (assumed to be Earth) 
                  in [km]

    OUTPUT:
        bcbf_vec - a Nx3 matrix of position vectors in the BCBF 
                   frame in [km]
    """

    # BCBF Position of the surface location
    bcbf_ground_pos = lla2bcbf(lla, r_body=r_body)

    # 3xN razel matrix
    try:
        bcbf_vec = np.zeros(razel.shape)

        for i in range(razel.shape[0]):
            rng = razel[i,0]
            az = razel[i,1]
            el = razel[i,2]

            # SEZ range vector of the spacecraft
            p_sez = np.array([-rng*np.cos(el)*np.cos(az),
                               rng*np.cos(el)*np.sin(az),
                               rng*np.sin(el)])
        
            # Rotating SEZ range vector to BCBF range vector
            p_bcbf = R3(-lon) @ R2(lat - np.pi/2.0) @ p_sez
        
            # Final BCBF position vector
            bcbf_vec[i,:] = p_bcbf + bcbf_ground_pos[:,i]

    # 3x1 razel vector
    except:
        rng = razel[0]
        az = razel[1]
        el = razel[2]

        # SEZ range vector of the spacecraft
        p_sez = np.array([-rng*np.cos(el)*np.cos(az),
                           rng*np.cos(el)*np.sin(az),
                           rng*np.sin(el)])
    
        # Rotating SEZ range vector to BCBF range vector
        p_bcbf = R3(-lon) @ R2(lat - np.pi/2.0) @ p_sez
    
        # Final BCBF position vector
        bcbf_vec = p_bcbf + bcbf_ground_pos

    return bcbf_vec

# ===================================================================


# ===================================================================
# 15)
def bcbf2sez(bcbf_vec, lla, r_body=c.r_earth):
    """This function takes the postion of a spacecraft in the 
    Body-Centered Body-Fixed (BCBF) frame and and the geocentric 
    latitude, longitude, and altitude of a seperate location and 
    computes the azimuth, elevation, and range of the spacecraft 
    relative to the location.

    ASSUMPTIONS:
    - Orbiting body is a perfect sphere
    - all angles in radians
    - bcbf_vec and lla are the same size

    INPUT:
        bcbf_vec - a Nx3 matrix of position vectors in the BCBF 
                   frame in [km]
        lla - a Nx3 matrix of vectors with: 
            lat - geocentric latitude in [rad]
            lon - the longitude in [rad]
            alt - the altitude from sea-level in [km]
        r_body  - radius of host body (assumed to be Earth) 
                  in [km]

    OUTPUT:
        razel - a Nx3 matrix of vectors with:
            rng - the distance of the spacecraft from the ground 
                  location in [km]
            az  - the azimuth of the spacecraft wrt the ground 
                  location in [rad]
            el -  elevation angle of the spacecraft wrt the 
                  ground location in [rad] 
    """

    # BCBF Position of the surface location
    bcbf_ground_pos = lla2bcbf(lla, r_body=r_body)

    # 3xN bcbf_vec matrix
    try:
        razel = np.zeros(bcbf_vec.shape)

        for i in range(bcbf_vec.shape[0]):
            # BCBF range = spacecraft positon - ground position
            p_bcbf = bcbf_vec[i,:] - bcbf_ground_pos[i,:]

            # Now we rotate BCBF into SEZ
            p_sez = R2(np.pi/2 - lat) @ R3(lon) @ p_bcbf
            rng = np.sqrt(p_sez.dot(p_sez))

            # Finally computing the azimuth and elevation
            cos_el = np.sqrt(p_sez[0]**2 + p_sez[1]**2)/rng
            sin_el = p_sez[-1]/rng
            el = np.arctan2(sin_el, cos_el)

            cos_az = -p_sez[0]/np.sqrt(p_sez[0]**2 + p_sez[1]**2)
            sin_az = p_sez[1]/np.sqrt(p_sez[0]**2 + p_sez[1]**2)
            az = np.arctan2(sin_az, cos_az)
            # if az < 0:
            #     az += 2*np.pi

            razel[i,:] = np.array([rng, az, el])

    # 3x1 bcbf_vec vector
    except:
        # BCBF range = spacecraft positon - ground position
        p_bcbf = bcbf_vec - bcbf_ground_pos

        # Now we rotate BCBF into SEZ
        p_sez = R2(np.pi/2 - lat) @ R3(lon) @ p_bcbf
        rng = np.sqrt(p_sez.dot(p_sez))

        # Finally computing the azimuth and elevation
        cos_el = np.sqrt(p_sez[0]**2 + p_sez[1]**2)/rng
        sin_el = p_sez[-1]/rng
        el = np.arctan2(sin_el, cos_el)

        cos_az = -p_sez[0]/np.sqrt(p_sez[0]**2 + p_sez[1]**2)
        sin_az = p_sez[1]/np.sqrt(p_sez[0]**2 + p_sez[1]**2)
        az = np.arctan2(sin_az, cos_az)
        # if az < 0:
        #     az += 2*np.pi

        razel = np.array([rng, az, el])

    return razel
# ===================================================================


# ===================================================================
# 16)

# ===================================================================