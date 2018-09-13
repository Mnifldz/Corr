# Riemannian Correlation Toolkit
#------------------------------------------------------------------------------
# Last Updated: 9/12/2018
# This file contains my main functions for computations on the correlation 
# manifold under the quotient structure SPD(n)/Diag_+(n) as described in my 
# various writings.  My functions will be here so that I no longer have to refer
# to my Matlab implementations.

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as LA


# Distance Functions
#------------------------------------------------------------------------------
def SPD_Inner(base, V1, V2):
    """Returns the inner product with respect to the affine-invariant metric
       on SPD(n) of two tangent vectors 'V1' and 'V2' at the base point 'base'."""
    return np.trace( np.matmul( LA.solve(base, V1), LA.solve(base, V2)    ))   

def SPD_Norm(base, V):
    """Special case of 'SPD_Inner' where we take the square-root of the norm of
       the inner product of a vector with itself."""
    return np.sqrt( SPD_Inner(base, V,V) )   


# Manifold Tools
#------------------------------------------------------------------------------
def Corr_Proj(P):
    """This function takes a matrix 'P' belonging to SPD(n) and projects it to
       its correlation representative under the quotient SPD(n)/Diag_+(n)."""
    Diag = np.sqrt( np.multiply(P, np.identity(P.shape[0]))  )
    return LA.solve(Diag, np.transpose( LA.solve(Diag, P )) )


# Descent Tools
#------------------------------------------------------------------------------
def SPD_Geo(base, V):
    """Returns the SPD matrix obtained by following the geodesic starting at
       'base' in the direction 'V' for a length of time 1.  NOTE: to change the
       amount of time to follow the geodesic, pre-multiply the tangent vector 
       by an amount 0 < c < 1 in implementation."""
    SQRT = LA.sqrtm(base)
    SOLVE = np.transpose( LA.solve(SQRT, V )) # NOTE: This only works because we have symmetric matrices.
    return np.matmul(  np.matmul(SQRT, LA.expm( LA.solve(SQRT,  SOLVE  ))),    SQRT) 

def Diag_Geo(base, V):
    """This is essentially the same 'SPD_Geo' but we take advantage of the fact
       that the computations simplify a bit."""
    return np.matmul(base, LA.expm(  LA.solve(base, V) ) )

def Diag_Tan(C1, C2, D):
    """Compute tangent vector for the diagonal in the gradient descent in Diag_+(n)."""
    pre_tan = np.matmul(D, LA.logm( np.matmul(C2, np.matmul(D, LA.solve(C1, D)) ) ))  
    return np.multiply( pre_tan + np.transpose(pre_tan), np.identity(D.shape[0]) )

def Diag_Grad(C1, C2, step, thresh):
    """This function helps find the optimal representative D*C2*D over C2 relative
       to C1 with respect to the affine-invariant geometry of Corr(n).  The 
       outputs are a tuple inclduing 'D_opt' the optimal diagonal matrix, 'P_opt'
       the corresponding optimal SPD(n) matrix over 'C2', 'iters' the number of
       iterations needed to converge on the desired value, and 'siz' the size of 
       the final tangent vector before termination of the algorithm."""
    # Initialize diagonal vector   
    D_opt = 0.1*np.identity(C2.shape[0])
    # Initialize tangent vector   
    tan = Diag_Tan(C1, C2, D_opt)
    
    iters = 0
    while SPD_Norm(D_opt, tan) > thresh:
        print(str(SPD_Norm(D_opt, tan)))
        D_opt = Diag_Geo(D_opt, -step*tan)
        tan   = Diag_Tan(C1, C2, D_opt)
        iters += 1
    
    P_opt = np.matmul(D_opt, np.matmul(C2, D_opt))    
    return D_opt, P_opt, iters, SPD_Norm(D_opt, tan)
        
    
       
       
       
       
       
       
       
       
       
       
       