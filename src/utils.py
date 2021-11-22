import numpy as np
from mytypes import ArrayLike
from cat_slice import CatSlice

# %%
def cross_product_matrix(n: ArrayLike, debug: bool = True) -> np.ndarray:
    assert len(
        n) == 3, f"utils.cross_product_matrix: Vector not of length 3: {n}"
    vector = np.array(n, dtype=float).reshape(3)

    S = S = np.array([[0, -vector[2], vector[1]],
                      [vector[2], 0, -vector[0]],
                      [-vector[1], vector[0], 0]])
    if debug:
        assert S.shape == (
            3,
            3,
        ), f"utils.cross_product_matrix: Result is not a 3x3 matrix: {S}, \n{S.shape}"
        assert np.allclose(
            S.T, -S
        ), f"utils.cross_product_matrix: Result is not skew-symmetric: {S}"

    return S

def wrap_to_pi_from_euler(euler_angles): 
    """
    Parameters
    ----------
    euler_angles : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    rads = np.deg2rad(euler_angles)
    return (rads + np.pi) % (2 * np.pi) - np.pi

def wrap_to_pi(rads): 
    return (rads + np.pi) % (2 * np.pi) - np.pi


def UDU_factorization(P: np.ndarray):
    """
            Args:
        -----------
        P (np.ndarray): The covariance matrix. Must be symmetric and positive semidefinite, shape ((15,15))
        rtol, atol: Tolerances for checking if its numerically close to zero, i.e factorization works
        Raises:
        -----------
            AssertionError: If any input is of the wrong shape, and if factorization failes

        Returns:
        -----------
            U, D np.ndarray: The upper triangular matrix U, and Diagonal matrix D
        """
    rtol = 1e-5
    atol = 1e-8
    assert P.shape == (
        15,
        15,
    ), f"utils.UDU-factorization: P incorrect shape {P.shape}"
    assert np.allclose(P, P.T, rtol=rtol, atol=atol) == True, f"utils.UDU-factorization: P Not symmetrical."


    
    n = len(P)
    U = np.zeros((n,n))
    d = np.zeros(n) #Leave as vector and diagonalize later

    U[:,-1] = P[:,-1] / P[-1,-1]
    d[-1] = P[-1, -1]

    for j in range(n-2,-1,-1):
        d[j] = P[j,j] - np.sum(d[j+1:n] * (U[j,j+1:n])**2)

        U[j,j] = 1.0

        for i in range(j-1,-1,-1):

            U[i,j] = (P[i,j] - np.sum(d[j+1:n] * (U[i,j+1:n]) * (U[j,j+1:n]))) / d[j]

    D = np.diag(d)

    assert U.shape == (
            15,
            15,
        ), f"utils.UDU-factorization: U shape incorrect {U.shape}"
    assert D.shape == (
            15,
            15,
        ), f"utils.UDU-factorization: D shape incorrect {D.shape}"
    
    # assert np.allclose(P-U@D@U.T, 0, rtol=rtol, atol=atol) == True, f"utils.UDU-factorization: Factorization failed. P-U@D@U.T not close zero."
    return U, D

def mod_gram_schmidt(A):
    """[summary]

    Args:
        A ([type]): [description]
    """
    d,n = np.shape(A)
    m = np.minimum(d,n)
    
    Q = np.zeros((m,m))
    R = np.zeros((n,n))
    
    for i in range(n):
        v = np.reshape(A[:,i],((n,1)))

        for j in range(i):
            R[j,i] = (np.reshape(Q[:,j],((n,1)))).T @ v
            v = (v)-R[j,i] * np.reshape(Q[:,j],((n,1)))
        
        R[i,i] = np.linalg.norm(v)
        
        Q[:,i] = (v / R[i,i]).T
        
    return Q, R

def mod_gram_NASA(Y, D_tilde):
    
    (n, m) = np.shape(Y)
    b = np.zeros((n, m))
    f = np.zeros((n, m))
    
    D_bar = np.zeros((n,n))
    U_bar = np.zeros((n,n))

    #Fra siste element til det 2. elementet
    # (k = n-1; j>-1, j--)
    for k in range(n-1, -1,-1):
        #Copy the row Y[k] into b[k]. This can be done more efficient and does not need to be its own loop
        b[k,:] = Y[k,:] 
    
    #Fra siste element til det 2. elementet
    #for (j = n-1; j>0, j--)
    for j in range(n-1, 0,-1):             
        # print("j = ", j)
        #Set the diagonal of U_bar to be 0, with the exception of U[0,0]
        U_bar[j,j] = 1

        f[j,:] = D_tilde @ b[j,:]                    #30x30 @ 30x1 = 30x1

        D_bar[j,j] = b[j,:].T @ f[j,:]               #1x30 @ 30x1 = 1x1

        f[j,:] = f[j,:] / D_bar[j,j]                 #30x1 / 1x1 = 30x1
 
        #Fra det 1. elementet til j-1. element
        #for (i = 0; i >j0, i++)
        for i in range(0, j):

            U_bar[i,j] = b[i,:].T @ f[j,:]            #1x30 @ 30x1 = 1x1
            b[i,:] = b[i,:] - U_bar[i,j] * b[j,:]     #30x1 - 1x1*30x1 = 30x1
             
    U_bar[0,0] = 1
    f[0,:] = D_tilde @ b[0,:]
    D_bar[0,0] = b[0,:] @ f[0,:]
    return U_bar, D_bar

    
def create_random_cov_matrix(n):
    """[summary]

    Args:
        n ([int]): [Dimentions of matrix]

    Returns:
        [np.array]: [Random positive definite, symmetric matrix illustrating a cov matrix]
    """

    matrixSize = n 
    A = np.random.rand(matrixSize, matrixSize)
    B = np.dot(A, A.transpose())
    return B


def check_upper_triangular(A):
    n = np.shape(A)[0]
    flag = 0
    for i in range(1, n):
        for j in range(0, i):
            if (A[i][j] != 0):
                flag = 0
            else:
                flag = 1

    if (flag == 1):
        print("Input matrix is Upper Triangular Matrix")
    else:
        print("Input matrix is not an Upper Triangular Matrix")

def check_if_diagonal_matrix(A):
    """[summary]
    Removes the diagonal of an input matrix and counts all non-zero elements.
    Returns 0 if the matrix is diagonal
    Args:
        A ([np.ndarray([][])]): [Input matrix]

    Returns:
        [int]: [Number of nonzero elements]
    """
    nonzero_elems = np.count_nonzero(A - np.diag(np.diagonal(A)))
    print("Number of nonzero elements not including the diagonal is: ", nonzero_elems)
    return nonzero_elems

