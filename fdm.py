from boundary import Boundary
from typing import List, Tuple, Callable, Dict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg


# def get_neighbors(p: Tuple[int, int]) -> List[Tuple[int, int]]:
#     """
#     Gets the grid neighbors of p in standard Z^2 coordinates (origin is 0).
#     Totally not index-safe! Please don't use this directly on an array...
#     :param p: Grid point in standard Z^2 coordinates
#     :return: Grid neighbors of p in the order [N, S, E, W]
#     """
#     return [(p[0], p[1] + 1), (p[0], p[1] - 1), (p[0] + 1, p[1]), (p[0] - 1, p[1])]


def generate_fdm_components(boundary: Boundary,
                            walker: Tuple[float, float, float, float],
                            interior_function: Callable[[Tuple[float, float]], float],
                            boundary_function: Callable[[Tuple[float, float]], float],
                            ) -> Tuple[sp.csc_matrix, sp.csc_matrix, np.array, np.array]:
    """
    Create the components of "Au = -A_hat*g + f" to run finite difference method with
    this boundary. Users could reference the second edition of "Numerical Methods for
    Elliptic and Parabolic Partial Differential Equations" by Knabner & Angerman to
    understand FDM. This code uses notation from Chapter 1, section 2, pg. 24-25
    of that text.

    Returns matrices in Compressed Sparse Column (CSC) format.

    TODO: Make Walker *class* a parameter!!

    Format of return tuple:

    - [0]    'A': <CSC format A matrix>,
    - [1]    'A_hat': <CSC format A_hat matrix, marks grid points as boundary points>,
    - [2]    'f': <np.array f vector, evals of interior function over interior>,
    - [3]   'g': <np.array g vector, evals of boundary function over boundary>


    :return: Tuple of components of "Au = -A_hat*g + f". See formatting above.
    """
    # Shabam:
    f_vec = boundary.vectorize_interior(interior_function)
    g_vec = boundary.vectorize_boundary(boundary_function)

    # A and A_hat can be made at the same time!
    A_rows = []
    A_cols = []
    A_vals = []
    A_hat_rows = []
    A_hat_cols = []
    # all values in A_hat are -1, so no need for an ``A_hat_vals``

    for ip, idx in boundary.interior_idx.items():

        # The point itself is always marked in A, and it should be -1
        A_rows.append(idx)
        A_cols.append(idx)
        A_vals.append(-1)

        # There will be at most four more, one for each stencil neighbor
        neighbors = [(ip[0], ip[1] + 1), (ip[0], ip[1] - 1), (ip[0] + 1, ip[1]), (ip[0] - 1, ip[1])]
        for i, neighbor in enumerate(neighbors):

            # Stencil point is boundary, belongs on RHS of equation
            if neighbor in boundary.boundary_idx:
                A_hat_rows.append(idx)
                A_hat_cols.append(boundary.boundary_idx[neighbor])

            # Stencil point is interior, belongs on LHS of equation
            else:
                # neighbors and walkers both ordered by [n, s, e, w]!!
                A_rows.append(idx)
                A_cols.append(boundary.interior_idx[neighbor])
                A_vals.append(walker[i])

    # COOrdinate matrices from scipy.sparse allow us to efficiently construct
    # sparse matrices. They're not very good data structures for doing linalg,
    # but we can easily convert them to CSR or CSC (Compressed Pparse Row/Column)
    # matrices down the line, so we're not actually swinging enormous matrices around
    A = sp.coo_matrix((A_vals, (A_rows, A_cols)),
                      shape=(boundary.interior_size, boundary.interior_size))

    A_hat = sp.coo_matrix(([-1] * len(A_hat_rows), (A_hat_rows, A_hat_cols)),
                          shape=(boundary.interior_size, boundary.boundary_size))

    return A.tocsc(), A_hat.tocsc(), f_vec, g_vec


def solve_fdm(A: sp.csc_matrix,
              A_hat: sp.csc_matrix,
              f: np.array,
              g: np.array) -> np.array:

    q = -(A_hat * g) + f
    u = sp.linalg.spsolve(A, q)
    return u


def map_solution(solution: np.array,
                 boundary: Boundary) -> Callable[[tuple[float, float]], float]:
    """
    Turn the FDM solution vector back into a grid function
    :param solution: ``np.ndarray`` solution vector
    :param boundary: Boundary object that FDM was run on
    :return: Grid function for FDM solution that maps to zero outside the boundary
    """

    solution_idx = {ip: solution[idx] for ip, idx in boundary.interior_idx.items()}

    def u(p, _sol=solution_idx) -> float:
        return _sol.get((int(p[0]), int(p[1])), 0.0)

    return u

