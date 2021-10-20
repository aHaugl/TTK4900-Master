import numpy as np

def UDU_factorization(P: np.ndarray):
    n = len(P)
    U = np.zeros((n,n))
    d = np.zeros(n) #Leave as vector

    U[:,-1] = P[:,-1] / P[-1,-1]
    d[-1] = P[-1, -1]

    for j in range(n-2,-1,-1):
        d[j] = P[j,j] - np.sum(d[j+1] * (U[j,j+1:n])**2)
        U[j,j] = 1.0
        print("j = ", j)
        for i in range(j-1,-1,-1):
            print("i = ", i)
            U[i,j] = (P[i,j] - np.sum(d[j+1:n] * (U[i,j+1:n]) * (U[j,j+1:n]))) / d[j]
            print("sum = ", np.sum(d[j+1:n] * (U[i,j+1:n]) * (U[j,j+1:n])) / d[j])
    
    D = np.diag(d)
    
    return U, D

matrixSize = 5
A = np.random.rand(matrixSize, matrixSize)
P = np.dot(A, A.transpose())
rtol = 1e-05
atol = 1e-08
np.allclose(P, P.T, rtol=rtol, atol=atol)

U, D = UDU_factorization(P)
# print("P = ", P)
# print("U = ", U)
# print("D = ", D)

print(P-U@D@U.T)