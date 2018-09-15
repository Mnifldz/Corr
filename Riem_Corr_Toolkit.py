# Riemannian Correlation Toolkit
#------------------------------------------------------------------------------
# Last Updated: 9/13/2018
# This file contains my main functions for computations on the correlation 
# manifold under the quotient structure SPD(n)/Diag_+(n) as described in my 
# various writings.  My functions will be here so that I no longer have to refer
# to my Matlab implementations.

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import numpy.linalg as npl
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

def SPD_Hess(start, end):
    """This function computes the approximate Hessian of the mean-squared objective
       function on SPD(n) [which can also be used on Corr(n) and Diag_+(n)].  We
       note here that this utilizes a 3rd-order Taylor approximation of the
       matrix logarithm."""
    n     = start.shape[0]
    S_inv = LA.solve(start, np.identity(n))
    E_inv = LA.solve(end, np.identity(n))
    term_1 = (1/3)*LA.kron( np.matmul(E_inv, npl.matrix_power(np.matmul(start, E_inv), 2) ), S_inv )
    term_2 = LA.kron(   np.matmul(E_inv, np.matmul(start, E_inv) )  , (2/3)*E_inv - (3/2)*S_inv)
    term_3 = LA.kron(E_inv,  3*S_inv - (3/2)*E_inv)
   
    return term_1 + term_2 + term_3

def Diag_Hess(C1, C2, D):
    """This is the Hessian of the diagonal manifold Diag_+(n) in order to perform
       fiber optimization."""
    n = C1.shape[0]
    C1_inv = LA.solve(C1, np.identity(n))
    Q      = np.matmul(C1_inv, np.matmul(D, C2))
    term_1 = LA.kron( np.transpose((4/3)*np.matmul(Q, np.matmul(D,Q)) -3*Q  ), Q)
    term_2 = LA.kron(  np.transpose( 6*C1_inv -3*np.matmul(Q, np.matmul(D, C1_inv) )  + (2/3)*np.matmul( npl.matrix_power( np.matmul(Q,D), 2), C1_inv) )   , C2)
    term_3 = LA.kron(np.transpose(6*Q -3*np.matmul(Q, np.matmul(D,Q)) +(2/3)*np.matmul(npl.matrix_power(np.matmul(Q,D), 2) ,Q)  ),  LA.solve(D,np.identity(n))  )
    term_4 = LA.kron(np.transpose( (2/3)*np.matmul(C2, npl.matrix_power( np.matmul(Q,D), 2)) -3*np.matmul(C2, np.matmul(D,Q)  ) )  , C1_inv)
    term_5 = LA.kron( (2/3)*np.matmul(C2, np.matmul(D,Q)),  np.matmul(Q, np.matmul(D, C1_inv))   )
    
    return term_1 + term_2 + term_3 + term_4 + term_5

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
        #print(str(SPD_Norm(D_opt, tan)))
        D_opt = Diag_Geo(D_opt, -step*tan)
        tan   = Diag_Tan(C1, C2, D_opt)
        iters += 1
    
    P_opt = np.matmul(D_opt, np.matmul(C2, D_opt))    
    return D_opt, P_opt, iters, SPD_Norm(D_opt, tan)

def Diag_Newton(C1, C2, thresh):
    """This function peforms a Newton gradient descent along the fiber over C2."""
    
    n     = C1.shape[0]
    D_opt = 0.1*np.identity(C2.shape[0])
    # Initialize tangent vector   
    tan = Diag_Tan(C1, C2, D_opt)
    
    iters = 0
    while SPD_Norm(D_opt, tan) > thresh:
        #print(str(SPD_Norm(D_opt, tan)))
        D_opt = Diag_Geo(D_opt, -tan)
        grad   = Diag_Tan(C1, C2, D_opt)
        Hess   = Diag_Hess(C1, C2, D_opt)
        
        tan = np.multiply( np.identity(n), np.reshape(  LA.solve(Hess, np.reshape(grad, [n**2, 1]) )   , [n,n]) )
        
        iters += 1
    
    P_opt = np.matmul(D_opt, np.matmul(C2, D_opt))    
    return D_opt, P_opt, iters, SPD_Norm(D_opt, tan)
       
       
# Simulation and Verification
#------------------------------------------------------------------------------
def Corr_Simulation(n, thresh):
    """This function offers a proof of concept and demonstrates the validity
       of the quotient and geodesic structure for Corr(n).  We initialize two
       random correlation matrices `C1' and 'C2' of size 'n', and follow the
       geodesic starting at 'C1' to 'C2' to verify that we get the same matrix.
       Outputs include C1 and C2, 'P_opt' which is the optimal SPD(n) element
       over C2 (minimizing the SPD-distance between C1 and the fiber over C2),
       'P_fin' the final SPD(n) element following the gradient descent, and 
       'Corr_end' the final correlation which is the projection of 'P_fin' back
       to Corr(n)."""
       
    A = np.random.random([n,n])
    B = np.random.random([n,n])
    C1 = Corr_Proj( LA.expm( (A + np.transpose(A))/2  ))
    C2 = Corr_Proj( LA.expm((B + np.transpose(B))/2 )  )
    
    #_, P_opt, _, _ = Diag_Grad(C1, C2, step, thresh)
    D_opt, P_opt, _, _ = Diag_Newton(C1, C2, thresh)
    
    P_sqrt = LA.sqrtm(P_opt)
    tan = np.matmul(P_sqrt,  np.matmul(  LA.logm( LA.solve(P_sqrt,  np.transpose(LA.solve(P_sqrt, C1  )) ) )  , P_sqrt  ))
    P_fin = SPD_Geo(P_opt, tan)
    
    return C1, C2, D_opt, P_opt, P_fin, Corr_Proj(P_fin)
    
       
       
       
       
       