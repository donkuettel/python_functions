
#############################
##    Plotting Functions   ##
#############################

"""This script provides a variety of useful functions for plotting.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

    1) get_cmap
    2) labelLine
    3) labelLines
    4) 

"""

# Import Modules
import matplotlib.pyplot as plt
import numpy as np
from math import atan2,degrees

# Define Functions
# =====================================================================
# 1) 
def get_cmap(n, name='jet'):
    """Returns a function that maps each index in 0, 1, ..., n-1 
    to a distinct RGB color; the keyword argument name must be a 
    standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)
# =====================================================================


# =====================================================================
# 2) 
def labelLine(line, x, label=None, align=True, **kwargs):
    """Label line with line2D label data"""

    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    # Find corresponding y coordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)
# =====================================================================


# =====================================================================
# 3) 
def labelLines(lines, align=True, xvals=None, **kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)
# =====================================================================


# =====================================================================
# 4) 

# =====================================================================
