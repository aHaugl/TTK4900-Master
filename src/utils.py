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


