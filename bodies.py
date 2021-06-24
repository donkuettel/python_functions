
###################################
##             BODIES            ##
###################################

"""This script provides useful constants for a number of 
astronomical bodies.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

NOTES:
    -The sphereical harmonic coefficients are dimensionless and normalized
"""

# Import Modules
from constants import *
import numpy as np

#######################################################################
# Main Function Call

def Body(body, grav=False, extras=0):
    """This function stores main characteristics for many different
    astronomical bodies in a dictionary for later use.    

    INPUT:
        body - a string that corresponds to the body of interest
        grav - flag indicating whether or not to include the bodies
               gravity coefficients in body_cons
        extras - constants necessary for the supporting functions
                 to work (namely degree and order of grav field)

    OUTPUT:
        body_cons - a dictionary containing the body characteristics
                    parameters of the desired body

    NOTES:
        - Most of these constants are gathered from the 4th Edition 
          of Fundamentals of Astodynamics and Applications by David
          Vallado
    """

    try:
        if isinstance(body, str):

            # ================================================================================
            if body.lower() == 'earth':
                # Body Characteristics
                r_body = 6378.1363          # km
                mu = 3.986004415e5          # km3/s2
                mass = 5.9742e24            # kg
                w_body = 7.2921158553e-5    # rad/s
                den = 5.515                 # g/cm3
    
                # Heliocentric orbital elements
                a = 149598023               # km
                e = 0.016708617
                inc = 0                     # rad
                raan = 0                    # rad
                w = 1.7965956472674636      # rad

                # Gravity Field Coefficients
                if grav:
                    try:
                        l = extras['degree']
                        m = extras['order']

                        C_lm, S_lm = gc_extract(l, m, '/earth_GGM02C.txt')
        
                        gc = {}
                        gc['C_lm'] = C_lm
                        gc['S_lm'] = S_lm

                    except:
                        print('ERROR: You need to define degree and order of gravity field!!')
            # ================================================================================


            # ================================================================================
            elif body.lower() == 'bennu':
                # Body Characteristics
                r_body = 0.245                          # km
                mass = 7.793e10                         # kg
                mu = 4.892e-9                           # km3/s2
                w_body = -4.061739e-4                   # rad/sec (retrograde)
                den = 1.26                              # g/cm3
    
                # Heliocentric orbital elements (JPL Small Body Database)
                a = 1.126391026024589*AU                # km
                e = 0.2037451095574896                
                inc = np.deg2rad(6.034939391321328)     # rad
                raan = np.deg2rad(2.060867570687039)    # rad
                w = np.deg2rad(66.22306847249314)       # rad
    
                # Gravity Field Coefficients
                if grav:
                    try:
                        l = extras['degree']
                        m = extras['order']

                        C_lm, S_lm = gc_extract(l, m, '/bennu_16_DRA9_CM.txt')

                        gc = {}
                        gc['C_lm'] = C_lm
                        gc['S_lm'] = S_lm

                    except:
                        print('ERROR: You need to define degree and order of gravity field!!')
            # ================================================================================


            # ================================================================================
            # elif body.lower() == '<BODY>':
            #     # Body Characteristics
            #     R = 
            #     mass = 
            #     mu = 
            #     w_body = 
            #     den = 
    
            #     # Heliocentric orbital elements
            #     a = 
            #     e = 
            #     inc = 
            #     raan = 
            #     w = 

            #     # Gravity Field Coefficients
            #     if grav:
            #         try:
            #             l = extras['degree']
            #             m = extras['order']

            #             C_lm, S_lm = gc_extract(l, m, '<GRAV_FILE.txt>')
        
            #             gc = {}
            #             gc['C_lm'] = C_lm
            #             gc['S_lm'] = S_lm

            #         except:
            #             print('ERROR: You need to define degree and order of gravity field!!')
            # ================================================================================


            # ================================================================================
            else:
                # Body Characteristics
                r_body = 0
                mu = 0
                mass = 0
                w_rot = 0
                den = 0
    
                # Heliocentric orbital elements
                a = 0
                e = 0
                inc = 0
                raan = 0
                w = 0
    
                # Grav coefficients
                gc = []

                print('ERROR: That body is not in this database!!')
            # ================================================================================


            # ================================================================================
            # Adding all the body's parameters to a dictionary
            body_cons = {}
            body_cons['r_body'] = r_body        # km
            body_cons['mu'] = mu                # km3/s2
            body_cons['mass'] = mass            # kg
            body_cons['w_body'] = w_body        # rad/s
            body_cons['den'] = den              # g/cm3
            body_cons['a'] = a                  # km
            body_cons['e'] = e
            body_cons['inc'] = inc              # rad
            body_cons['raan'] = raan            # rad
            body_cons['w'] = w                  # rad
            if grav:
                body_cons['gc'] = gc
            # ================================================================================

    except:
        print('ERROR in BODY(): Is the input a string?')

    return body_cons

#######################################################################


#######################################################################
# Supporting Functions
def gc_extract(l, m, file_name):
    """This function extracts a body's sphereical harmonic gravity
    field coefficients and arranges them into a matrix for use.

    INPUT:
        l - degree of spherical harmonic gravity
        m - order of spherical harmonic gravity
        file_name - name of gravity coefficients text file

    OUTPUT:
        C - (l-1)x(m+1) matrix of C_lm coefficients
        S - (l-1)x(m+1) matrix of S_lm coefficients
    """

    # Reading in grav file
    path = '/Users/donkuettel/Documents/coding/python3/functions/planet_files'
    file = []
    with open(path+file_name) as f:
        for line in f.readlines():
             file.append([x.strip() for x in line.split(',')])

    # Saving Grav Coeffs to a matrix
    C = np.zeros((l-1,m+1))
    S = np.zeros((l-1,m+1))
    for i in range(len(file)):
        row = file[i]
        # Getting desired degree
        if int(row[0]) > l:
            break
        else:
            # getting desired order
            if int(row[1]) > m:
                continue        
            C[int(row[0])-2,int(row[1])] = float(row[2])
            S[int(row[0])-2,int(row[1])] = float(row[3])

    return C, S
#######################################################################
    