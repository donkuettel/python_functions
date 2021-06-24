
#################################
##          Constants          ##
#################################

"""This script provides a variety of useful constants.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA
"""

# Misc 
AU = 1.49597870700e8            # km
DPY = 365.242189                # days
solar_day = 86400               # sec
sideral_day = 86164.090517      # sec
c = 2.99792458e8                # m/s, speed light in a vacuum
G = 6.6726e-20                  # km3/kg/s2, gravitional constant
srp_flux = 1357                 # W/m2 at 1 AU, solar flux
g0 = 9.80665                    # m/s2, standard gravity

# Gravitational Parameters
mu_sun = 1.32712440018e11       # km3/s2
mu_venus = 3.24858599e5         # km3/s2
mu_earth = 3.986004415e5        # km3/s2
mu_mars = 4.28283100e4          # km3/s2
mu_jupiter = 1.26686536e8       # km3/s2
mu_saturn = 3.7931208e7         # km3/s2
mu_uranus = 5.7939513e6         # km3/s2
mu_neptune = 6.835100e6         # km3/s2
mu_bennu = 4.892e-9             # km3/s2
mu_luna = 4.9048695e3           # km3/s2

# Planet Radius
r_sun = 696000                  # km
r_venus = 6051.8                # km
r_earth = 6378.1363             # km
r_mars = 3396.19                # km
r_jupiter = 71492               # km
r_saturn = 60268                # km
r_uranus = 25559                # km
r_neptune = 24764               # km
r_bennu = 245e-3                # km
r_luna = 1737.1                 # km

# Planet Masses
m_sun = 1.9891e30               # km
m_venus = 4.869e24              # km
m_earth = 5.9742e24             # km
m_mars = 6.4191e23              # km
m_jupiter = 1.8988e27           # km
m_saturn = 5.685e26             # km
m_uranus = 8.6625e25            # km
m_neptune = 1.0278e26           # km
m_bennu = 7.329e10              # km

# Distance from Sun (semi-major axis)
a_mercury = AU*0.387098         # km (eccentricity of 0.2)
a_venus = AU*0.72332982         # km
a_earth = AU*1.0                # km
a_mars = AU*1.52367934          # km
a_jupiter = AU*5.202603191      # km
a_saturn = AU*9.554909595       # km
a_uranus = AU*19.218446061      # km
a_neptune = AU*30.11038687      # km

# Earth
P_earth = 365.242189            # days
w_earth = 0.000072921158553     # rad/s
J2_earth = 0.00108263566655
J3_earth = -2.53247369133e-06
J4_earth = -0.0000016196
