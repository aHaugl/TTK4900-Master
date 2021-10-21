import numpy as np

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

    assert np.allclose(P-U@D@U.T, 0, rtol=rtol, atol=atol) == True, f"utils.UDU-factorization: Factorization failed. P-U@D@U.T not close zero."
    return U, D

matrixSize = 15
A = np.random.randint(1,10,size=(matrixSize, matrixSize))
P = np.dot(A, A.transpose())
rtol = 1e-05
atol = 1e-08
np.allclose(P, P.T, rtol=rtol, atol=atol)


U, D = UDU_factorization(P, rtol, atol)
# print("P = ", P)
# print("U = ", U)
# print("D = ", D)
# C = P-U@D@U.T
# print("C =", C)
