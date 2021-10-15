import numpy as np
from mytypes import ArrayLike


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
    Factorizes covariance matrix P at current step into U, and D.

    Note that D can also be stored as a vector containing only the diagonal elements of D_matrix, but then
    we need to handle this when using the matrices

    U : output, unit upper triangular matrix
    D : output, diagonal matrix
    P : input, real symmetric matrix

    """
    n = len(P)
    U = np.zeros((n,n))
    D = np.zeros((n,n))

    print(U)
    print(D)

    # np.fill_diagonal(D, np.diag(P))
    # %% From NASA
    U[-1,-1] = 1.0
    D[-1,-1] = P[-1,-1]
    # Denne stemmer
    for j in range(n-1,0,-1):
        U[j,n-1] =  P[j,n-1]/D[n-1,n-1]

    # Her blir det rart
    for j in range(n-2,0,-1):
        print("j=",j)
        D[j,j] = P[j,j]
        for k in range(j,0 -1):
            print("k=",k)
            D[j,j] = D[j,j] + D[k-1,k-1]*U[j,k-1]**2
        U[j,j] = 1.0
        for i in range(j-1,0,-1):
            print("i=",i)
            U[i,j] = P[i,j]
            for k in range(j + 1, n-1):
                U[i,j] = U[i,j] + D[k,k] * U[j,k] * U[i,k]
            U[i,j] = U[i,j]/D[j,j]
    

    # %% From matlab snippet 
    # for j in range(n-1, 0,-1):
    #     D[j,j] = P[j,j]
    #     alpha = 1/D[j,j]
    #     for k in range(0,j):
    #         # print("k=",k)
    #         beta = P[k,j]
    #         U[k,j] = alpha*beta
    #         for i in range(0,k+1):
    #             # print("i=",i)
    #             P[i,k] = P[i,k] - beta * U[i,j]

    # D[1,1] = P[1,1]
    # for i in range (0,n):
    #     U[i,i] = 1




    # assert U.shape ==(
    #         3,
    #         3,
    #         ), f"utils.UDU_factorization: U-matrix shape incorrect {U.shape}"
    # assert D.shape == (
    #         3,
    #         3,
    #         ), f"utils.UDU_factorization: D-matrix shape incorrect {D.shape}"

    # assert np.count_nonzero(D - np.diag(np.diagonal(D))) == (
    #         0,
    #         ), f"utils.UDU_factorization: D matrix is not diagonal and has this amount of nonzero elements: {np.count_nonzero(D - np.diag(np.diagonal(D)))}"
    return U, D
# %% 
P = np.random.randint(1,10,size=((3,3)))

U,D = UDU_factorization(P)
# %%
