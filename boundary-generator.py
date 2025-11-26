from typing import List, Tuple, Set
import numpy as np
import noise


import boundary

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (0) Boundary function                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# A map R^2 --> [-1, 1]
#
# Do inquire if curious. I have spent long hours delving deeply into the dark arts
# of "perlin noise." It's just smooth noise, and the boundary samples from it.
# Hence, a smoothly random boundary function. Should be interesting for testing?
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
"""

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (1) Place markers on the corners                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# This part is pretty chill. Rotate by a random amount and put a randomly long
# vector down. Then rotate again and put down another random vector, etc.
# Random ranges for theta and chosen artfully above so that the boundary is in
# a goldilocks zone: not too erratic, and not too regular
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
"""

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (2) Interpolate between corners                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# The way we do this is simply by making a line y = mx + b between each corner
# and then for each x between the points, marking the appropriate y = mx + b.
#
# BUT! This would leave gaps in the boundary. For example, imagine a line y = 0.2x.
# How many times does this hit an integer over the interval [0, 5]? Only at (0,0),
# and (5, 1). But, for the boundary to be closed, we would need at least a full 5
# points marked on the grid.
#
# This can be solved by also making a line x = my + b, and for each y between the
# points, marking the appropriate x = my + b on the grid. The way the rounding
# works sorts this out. The "if" statements make sure the boundary is closed.
#
# ESSENTIALLY, THOUGH--we check the boundary index set so we never overwrite
# points that are already in the boundary! This will totally screw up the
# evaluation of the boundary function because the points from "y = mx + b"
# will end up with a different boundary values than "x = ym + b" because of
# weird rounding stuff. Thankfully checking set membership is O(1).
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
"""

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (3) Make the boundary + interior                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# This is the part that really matters for FDM... We need some kind of output.,
# and it will be the vectorized boundary and interior, ready for an FDM solve.
#
# It was way harder than I thought to round up all the interior points! The
# algorithm I came up with is a "wall insulation"/"quarantined plague" strategy.
#
# (1) We know that the origin is within the boundary, so we put it in a queue.
# (2) Then, until the queue is empty, we take an item out and check if it is a boundary
# point. If not, and we haven't already looked at this point, then add it to the
# interior points set.
# (3) Then we put its four grid neighbors in queue to be checked; return to (2)
#
# Hence, "plague strategy;" each point gets "infected." Thank heavens our boundary
# is closed! It quarantines the exterior, and we end up with all the interior points.
#
# In addition, we build a minimal boundary set. It could happen that above we
# generate a boundary that is "two points thick" at some points," an easy
# example to imagine would be very spiky corners. Now, as we go through all
# the interior points, we can finally choose the boundary to be only the points
# in the generated boundary set that are reachable by moving from the interior
#
# FDM doesn't care what ordering we choose for the interior and boundary points.
#
"""


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










