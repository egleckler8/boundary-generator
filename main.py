from typing import List, Tuple, Set
import numpy as np
import noise


from boundary import Boundary
import fdm


n = 47

# Walkers with directions N, S, E, W
w1 = (0.125, 0.125, 0.325, 0.325)
w2 = (0.325, 0.325, 0.125, 0.125)
wS = (0.25, 0.25, 0.25, 0.25)


def g(p: Tuple[float, float]) -> float:
    """
    Boundary function, R^2 --> R.
    :param p: plane point in standard (x, y) coords
    :return: g(p)
    """
    # This can help us check that everything is in order
    return p[0]

    # The following will sample from smooth noise:
    scale = 2
    z = 8.88  # seed z for reproducibility?
    # below, adding 0.08 so that it's NEVER an integer: integer coords always return zero
    e = noise.pnoise3((p[0] / n) * scale + 0.08,
                      (p[1] / n) * scale + 0.08,
                      z,
                      octaves=3,
                      persistence=0.75,
                      lacunarity=0.75)
    return e


# This is the function inside the boundary, like f on the LHS of the system in FDM
def f(p: Tuple[int, int]) -> float:
    # This can help us visualize the boundary function:
    # return g(p)
    # This can help us check the orientation:
    # return p[0]
    # ... and this will surely make a fun picture, too:
    # return np.sin(p[0] * p[1] / n)
    return 0.0


b = Boundary(n)

# Test out FDM on the boundary
A, A_hat, f_vec, g_vec = fdm.generate_fdm_components(boundary=b, walker=wS, interior_function=f, boundary_function=g)

u = fdm.solve_fdm(A, A_hat, f_vec, g_vec)
u = fdm.map_solution(u, b)

img = b.make_img(u, g)
img.show()
# Test out image generation:
# img = b.make_img(f, g)
# img.save('imgs/boundary0.png')
# img.show()

# Test out the __str__ method:
# print(b)










