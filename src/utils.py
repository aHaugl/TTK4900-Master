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
    m = n-1 # Matrix index number
    U = np.zeros((n,n))
    D = np.zeros((n,n))


    print(U)
    print(D)
    print("P = ",P)

    # %% From NASA
    #sorting out the n,n elements
    U[n-1,n-1] = 1.0
    D[n-1,n-1] = P[n-1, n-1]

    #Mattenotasjon og pythonnotasjon: D_{1,1} = D[0,0] = det første elementet

    # %% Last column loop
    #"For j = n-1: -1: 1 do:""
    # For row = nest siste rad til den 1. raden.
    for row in range(n-2, -1, -1):
        U[row,m] = P[row,m] / D[m,m]
        # print("row=", row)
        # print("U= ", U)

    # %% Main loop
    #"for j =n:-1:2 do:"    
    #for row = den siste raden, dekrementert til 2. rad
    #Gjør ting med diagonalen til D, med unntak av siste og første element (D_{n,n} og D_{1,1})
    for row in range(n-2, 1, -1):
        #Sett diagonalen til U = 1.0
        U[row,row] = 1.0

        D[row,row] = P[row,row]
        #"for k = j+1: -n do"
        #for k = row (row+1 er utenfor index?) row +1 dekrementert til det første elementet i kolonnen
        for D_col in range (row + 1, 0, -1):
            D[row,row] = D[row,row] + D[D_col,D_col]*U[row,D_col]**2
            # print("col = ", col)
            # print("D: ", D)
    
        # for i = j-1: -1:1 do
        # En rad opp til 1. rad
        #Gå gjennom de resterende tomme U-elementene og fyll inn
        for i in range(row-1, 0 ,-1):
            U[i,row] = P[i,row]
            print("i = ", i) #Skal outputte i = 4 til 0
            print ("U[i,row] i", U[i,row])
            

            for k in range(row+1,n-1):
                U[i,row] = U[i,row] + D[k,k]*U[row,k]*U[i,k]
                print("k = ", i) #Skal outputte i = 4 til 0
                print("U[i,row] k = ", U[i,row])
            U[i,row] = U[i,row]/D[row,row]

    print("D = ", D)
    print("U = ", U)

    return U, D
    



P = np.array([[1, 5, 9, 7, 6],
       [1, 3, 9, 9, 7],
       [2, 1, 5, 3, 3],
       [6, 1, 5, 7, 5],
       [1, 4, 8, 8, 9]])


P3 = np.array([[1, 5, 9],
       [1, 3, 9],
       [2, 1, 5,]])
       

U, D = UDU_factorization(P)
print(U)
print(D)



    

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
    # return U, D
# %% 
P = np.random.randint(1,10,size=((5,5)))

U,D = UDU_factorization(P)
# %%
