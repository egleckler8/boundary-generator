from typing import List, Tuple, Set
import numpy as np
import noise


import boundary




n = 93


def g(p: Tuple[float, float]) -> float:
    """
    Boundary function, R^2 --> R.
    :param p: plane point in standard (x, y) coords
    :return: g(p)
    """
    # This can help us check that everything is in order
    # return p[0]

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
    return np.sin(p[0] * p[1] / n)


# Walkers with transition probabilities enumerated
# as [n, s, e, w] to match `get_neighbors` function
w_symmetric = (0.25, 0.25, 0.25, 0.25)
w1 = (0.125, 0.125, 0.325, 0.325)
w2 = (0.225, 0.325, 0.125, 0.125)

b = boundary.Boundary(n=n, interior_function=f, boundary_function=g)

yeah = b.vectorize(walker=w1)


img = b.make_img()
# img.save('imgs/boundary0.png')
img.show()
#print(b)










