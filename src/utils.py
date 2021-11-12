import numpy as np
from mytypes import ArrayLike

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


def UDU_factorization(P: np.ndarray, rtol, atol):
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

    # assert P.shape == (
    #     15,
    #     15,
    # ), f"utils.UDU-factorization: P incorrect shape {P.shape}"
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

    # assert U.shape == (
    #         15,
    #         15,
    #     ), f"utils.UDU-factorization: U shape incorrect {U.shape}"
    # assert D.shape == (
    #         15,
    #         15,
    #     ), f"utils.UDU-factorization: D shape incorrect {D.shape}"

    assert np.allclose(P-U@D@U.T, 0, rtol=rtol, atol=atol) == True, f"utils.UDU-factorization: Factorization failed. P-U@D@U.T not close zero."
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
        v = np.reshape(A[:,i],((3,1)))

        for j in range(i):
            R[j,i] = (np.reshape(Q[:,j],((3,1)))).T @ v
            v = (v)-R[j,i] * np.reshape(Q[:,j],((3,1)))
        
        R[i,i] = np.linalg.norm(v)
        
        Q[:,i] = (v / R[i,i]).T
        
    return Q,R
        

# function [Q,R]=mgs(A)
# [m,n]=size(A);
# V=A;
# Q=zeros(m,n);
# R=zeros(n,n);
# for i=1:n
#   R(i,i)=norm(V(:,i));
#   Q(:,i)=V(:,i)/R(i,i);
#   for j=i+1:n
#     R(i,j)=Q(:,i)'*V(:,j);
#     V(:,j)=V(:,j)-R(i,j)*Q(:,i);
#   end
# end