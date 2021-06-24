
#############################
##     LU Decomposition    ##
#############################

"""This script uses Crout's LU Decomposition method to decompose
an nxn matrix into it's upper and lower triangular form.

AUTHOR: 
    Don Kuettel <don.kuettel@gmail.com>
    Univeristy of Colorado-Boulder - ORCCA

    1) LUdcmp
    2) LUdcmp_inv
    3) LUdcmp_det
    4) LUdcmp_solve
    5) LUdcmp_msolve

"""

# Import Modules
import numpy as np

# Define Functions
# ===================================================================
# 1) 
def LUdcmp(a):
    """This function uses Crout's method to perfom the LU 
    decomposition of a matrix a.
    
    INPUT:
        a - (n,n) matrix (np.array)
    
    OUPUT:
        lu - (n,n) LU Crout matrix
    """ 

    # Checking dimensions of a
    if (len(a.shape) != 2) or (a.shape[0] != a.shape[-1]):
        print('A matrix is not square')
        return

    # LU decomposition
    n = len(a)
    TINY = 1e-40
    d = 1.0
    indx = [0]*n
    vv = np.zeros(n)
    lu = np.copy(a)

    # Making sure the matrix doesn't have a row of zeros
    for i in range(n):
        big = 0.0
        for j in range(n):
            temp = abs(lu[i][j])
            if (temp > big):
                big = temp
        if big == 0.0:
            print('Singular Matrix in LUdcmp')
            return
        vv[i] = 1.0/big

    # Outer most kij loop.
    for k in range(n):
        big = 0.0

        # This searches for the largest pivot element.
        for i in range(k, n):
            temp = vv[i]*abs(lu[i][k])
            # print(temp, big)
            if (temp > big):
                big = temp
                imax = i

        # Do we need to interchange rows?
        if (k != imax):
            for j in range(n):
                temp = lu[imax][j]
                lu[imax][j] = lu[k][j]
                lu[k][j] = temp
            d = -d
            vv[imax] = vv[k]

        indx[k] = imax

        # Getting rid of any pivot element zero terms
        if (lu[k][k] == 0.0):
            lu[k][k] = TINY

        # Beta divsion
        for i in range(k+1, n):
            lu[i][k] /= lu[k][k]
            temp = lu[i][k]
            for j in range(k+1, n):
                lu[i][j] -= temp*lu[k][j]

    return lu, indx, d
# ===================================================================


# ===================================================================
# 2) 
def LUdcmp_inv(a):
    """This function finds the inverse of a matrix using Crout's 
    method to perfom the LU decomposition.
    
    INPUT:
        a - (n,n) matrix (np.array)
    
    OUPUT:
        inv_a - (n,n) inverse matrix (np.array)
    """ 

    # Getting LU decomposition of a
    lu, indx, d = LUdcmp(a)

    # Solving for inverse
    n = len(a)
    inv_a = LUdcmp_msolve(a, np.eye(n))

    return inv_a
# ===================================================================    


# ===================================================================
# 3) 
def LUdcmp_det(a):
    """This function finds the inverse of a matrix using Crout's 
    method to perfom the LU decomposition.
    
    INPUT:
        a - (n,n) matrix (np.array)
    
    OUPUT:
        det_a - determinant of a
    """ 

    # Getting LU decomposition of a
    lu, indx, d = LUdcmp(a)

    # Finding determinant
    n = len(a)
    for i in range(n):
        d *= lu[i][i]

    return d
# ===================================================================


# ===================================================================
# 4) 
def LUdcmp_solve(a, b):
    """This function solves A*x = b, where b is one dimensional.
    
    INPUT:
        a - (n,n) matrix (np.array)
        b - (n,) vector (np.array)
    
    OUPUT:
        x - (n,) solution vector (np.array)
    """ 

    # Checking that function will work
    if len(b.shape) > 1:
        print("Use msolve for B")
        return

    if len(a) != len(b):
        print("A and b are not the same size")
        return

    # Getting LU decomposition of a
    lu, indx, d = LUdcmp(a)

    # Solving A*x = b, where b is one dimensional
    ii = 0
    x = np.copy(b)
    n = len(b)

    # Forward subsitution
    for i in range(n):
        ip = indx[i]
        fsum = x[ip]
        x[ip] = x[i]
        if ii != 0:
            for j in range(ii-1, i):
                fsum -= lu[i][j]*x[j]
        elif fsum != 0:
            ii = i +1
        x[i] = fsum
    
    # Backwards substitution
    for i in reversed(range(n)):
        bsum = x[i]
        for j in range(i+1, n):
            bsum -= lu[i][j]*x[j]
        x[i] = bsum/lu[i][i]

    return x
# ===================================================================


# ===================================================================
# 5) 
def LUdcmp_msolve(a, b):
    """This function solves A*x = B, where B is one 
    multi-dimensional.
    
    INPUT:
        a - (n,n) matrix (np.array)
        b - (n,n) matrix (np.array)
    
    OUPUT:
        x - (n,n) solution matrix (np.array)
    """ 

    # Checking that function will work
    if (len(a.shape) != 2) or (len(b.shape) != 2):
        print("A and B must be matrices")

    if len(a) != b.shape[0]:
        print("B does not have n rows")
        return

    # Solving for B
    n = b.shape[0]
    m = b.shape[-1]
    x = np.copy(b)
    xx = np.zeros(n)

    for j in range(m):
        for i in range(n):
            xx[i] = b[i][j]
        temp = LUdcmp_solve(a, xx)
        for i in range(n):
            x[i][j] = temp[i]

    return x
# ===================================================================

