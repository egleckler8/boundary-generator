from typing import List, Tuple
from scipy.sparse import *
from PIL import Image
import numpy as np
import noise
import queue
import time
import json


n = 25

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


max_theta = np.pi / 2
min_theta = np.arctan2(2, n)  # as n gets bigger, we could afford more minute turns
max_r = 0.5*n - 1
min_r = 0.25*n


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
theta = 0
corners = []

while theta < 2*np.pi - 0.001:

    # To test regular polygons:
    # r = min(m, n)/2
    # theta += (2*np.pi)/k

    r = np.random.uniform(min_r, max_r)
    x, y = int(r * np.cos(theta)), int(r * np.sin(theta))
    corners.append((x, y))
    theta += np.random.uniform(min_theta, max_theta)


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
boundary_set = set()

for i in range(len(corners)):

    # x's different, do "y = mx + b" between corners
    if corners[i][0] != corners[(i+1) % len(corners)][0]:

        # Look at the points left-to-right
        if corners[i][0] < corners[(i+1) % len(corners)][0]:
            p1, p2 = corners[i], corners[(i+1) % len(corners)]
        else:
            p2, p1 = corners[i], corners[(i + 1) % len(corners)]

        # x's different, so slope safe
        m1 = (p2[1] - p1[1])/(p2[0] - p1[0])
        b1 = p2[1] - m1 * p2[0]

        # Increment x one at a time and plot y
        x = p1[0]
        while x < p2[0]:
            y = m1*x + b1
            if (int(x), int(y)) not in boundary_set:
                boundary_set.add((int(x), int(y)))
            x += 1

    # y's different, do "x = my + b" between corners for good measure
    if corners[i][1] != corners[(i+1) % len(corners)][1]:

        # Look at the points bottom-to-top
        if corners[i][1] < corners[(i+1) % len(corners)][1]:
            p1, p2 = corners[i], corners[(i+1) % len(corners)]
        else:
            p2, p1 = corners[i], corners[(i + 1) % len(corners)]

        # y's different, so slope safe
        m2 = (p2[0] - p1[0]) / (p2[1] - p1[1])
        b2 = p2[0] - m2 * p2[1]

        y = p1[1]
        while y < p2[1]:
            x = m2*y + b2
            if (int(x), int(y)) not in boundary_set:
                boundary_set.add((int(x), int(y)))
            y += 1


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


# This is the function inside the boundary, like f on the LHS of the system in FDM
# For now, for us, it can just return zero.
def f(p: Tuple[int, int]) -> float:
    # This can help us visualize the boundary function:
    return g(p)
    # This can help us check the orientation:
    # return p[0]
    # ... and this will surely make a fun picture, too:
    # return np.sin(0.08 * (p[0] * p[1]) / n)


def get_neighbors(p: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the grid neighbors of p in standard Z^2 coordinates (origin is 0).
    Totally not index-safe! Please don't use this directly on an array...
    :param p: Grid point in standard Z^2 coordinates
    :return: Grid neighbors of p in the order [N, S, E, W]
    """
    return [(p[0], p[1] + 1), (p[0], p[1] - 1), (p[0] + 1, p[1]), (p[0] - 1, p[1])]


t0 = time.perf_counter()  # let's see how long this takes in practice... algo is O(n^2) probably

# Origin infects the whole interior. Boundary quarantines effectively.
# Keep on the relevant boundary points.
# Huge fan of constant time dict lookup.
interior_idx = {}
boundary_idx = {}
q = queue.Queue()
q.put((0, 0))

while not q.empty():
    p = q.get()
    if p not in boundary_set:
        if p not in interior_idx:

            # Setting the dict entry to the size of the set at the time gives
            # us an ordering on the interior. Now we can map each point to
            # its index in a vector. Will come in clutch when we vectorize D
            interior_idx[p] = len(interior_idx)

            for neighbor in get_neighbors(p):
                q.put(neighbor)

    # Here we are constructing a minimal boundary, including on the
    # points that the interior points are touching.
    else:
        boundary_idx[p] = len(boundary_idx)


tf = time.perf_counter()
dt = tf - t0

print('****************************************************************')
print('\tBOUNDARY GENERATION COMPLETE. INTERIOR GENERATION COMPLETE.')
print('****************************************************************')
print('Number of interior points:', len(interior_idx), f' --> {len(interior_idx) / (n * n):.2%} of the {n}x{n} grid.')
print('Number of boundary points:', len(boundary_idx), f' --> {len(boundary_idx) / (n * n):.2%} of the {n}x{n} grid.')
print('Time to compute interior:', f'~{dt:.4f}s')
print(f'~{(len(interior_idx) / dt):.2f} points per second.')
print('****************************************************************')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (3) Vectorize interior & boundary                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# We need all the components of "Au = Ahat*g + f"
#

# (1) f vector - interior values
f_vec = np.zeros(len(interior_idx))
for ip in interior_idx:
    f_vec[interior_idx[ip]] = f(ip)

# (2) g vector - boundary values
g_vec = np.zeros(len(boundary_idx))
for bp in boundary_idx:
    g_vec[boundary_idx[bp]] = g(bp)




# (3 + 4) A and Ahat at the same time!

# We shall first assemble COO matrix with (row, col, val),
# then turn that into a CSR matrix to efficiently store sparse data

# Walkers with transition probabilities enumerated
# as [n, s, e, w] to match `get_neighbors` function
w_symmetric = [0.25, 0.25, 0.25, 0.25]
w1 = [0.125, 0.125, 0.325, 0.325]
w2 = [0.225, 0.325, 0.125, 0.125]

# This is the walker we choose just for testing:
walker = w1

A_rows = []
A_cols = []
A_vals = []
Ahat_rows = []
Ahat_cols = []
# all values in Ahat are -1, so no need for an `Ahat_vals`

for ip, idx in interior_idx.items():

    # The point itself is always marked in A, and it should be -1
    A_rows.append(idx)
    A_cols.append(idx)
    A_vals.append(-1)

    # There will be at most four more, one for each stencil neighbor
    neighbors = get_neighbors(ip)
    for i, neighbor in enumerate(neighbors):

        # Stencil point is boundary, belongs on RHS of equation
        if neighbor in boundary_idx:
            Ahat_rows.append(idx)
            Ahat_cols.append(boundary_idx[neighbor])

        # Stencil point is interior, belongs on LHS of equation
        else:
            # neighbors and walkers both ordered by [n, s, e, w]!!
            A_rows.append(idx)
            A_cols.append(interior_idx[neighbor])
            A_vals.append(walker[i])


Ahat_vals = [1] * len(Ahat_rows)

# COOrdinate matrices from scipy.sparse allow us to efficiently construct
# sparse matrices. They're not very good data structures for doing linalg,
# but we can easily convert them to CSR or CSC (Compressed Pparse Row/Column)
# matrices down the line so we're not actually swinging enormous matrices around
A = coo_matrix((A_vals, (A_rows, A_cols)), shape=(len(interior_idx), len(interior_idx)))
Ahat = coo_matrix(([1] * len(Ahat_rows), (Ahat_rows, Ahat_cols)), shape=(len(interior_idx), len(boundary_idx)))
# A = A.tocsr()
# Ahat = Ahat.tocsr()


# Now, et's get creative:
# ********************************
#   DISPLAY AS PIXEL MAP:
# ********************************

# Converts array index to Z^2 point
def grid(i: int) -> int:
    return int(i - n/2 + 0.5)


def make_img(show_bv=True, show_iv=True):
    """
    Makes a cool picture of the boundary
    :param show_bv: Whether to show the boundary function values
    :param show_iv: Whether to show the interior function values
    :return: Image object for saving or showing. (hint: img = make_img(); img.show())
    """
    # We'll layer these to create an RGB image
    red_window = np.zeros((n, n))
    green_window = np.zeros((n, n))
    blue_window = np.zeros((n, n))

    # We need to make a map [-1, 1] --> [0, 255].
    # For enhanced visualization, we first min-max normalize the boundary values,
    # linearly squeezing them into to fit perfectly into [0, 1] (min +-> 0, max +-> 1)
    # Then a simple multiplication by 255 puts them into the 8-bit unsigned int range for RGB
    bv_min = np.min(g_vec)
    bv_max = np.max(g_vec)
    normalized_boundary_values = np.zeros_like(g_vec)
    for i in range(len(g_vec)):
        # In case we just want the un-normalized values
        # normalized_boundary_values[i] = g_vec[i]
        if show_bv:
            normalized_boundary_values[i] = 255*(g_vec[i] - bv_min)/(bv_max - bv_min) if bv_max != bv_min else 0
        # else it will remain zero

    # Make a "layer" for each RGB color.
    # We can be creative or whatever to make it look nice, e.g. green ==> good, red ==> bad
    for bp, idx in boundary_idx.items():
        c = int(bp[0] + (n-1)/2)              # x-axis goes left --> right
        r = (n - 1) - int(bp[1] + (n-1)/2)    # y-axis goes bottom --> top
        red_window[r][c] = 255 - normalized_boundary_values[idx]
        green_window[r][c] = normalized_boundary_values[idx] if show_bv else 255
        blue_window[r][c] = 0 if show_bv else 255

    # Now do the same for the interior.
    # It's going to be a real nice addition to the scrapbook.
    iv_min = np.min(f_vec)
    iv_max = np.max(f_vec)
    normalized_interior_values = np.zeros_like(f_vec)
    for i in range(len(f_vec)):
        if show_iv:
            normalized_interior_values[i] = 128*(f_vec[i] - iv_min)/(iv_max - iv_min) if iv_max != iv_min else 0
        # else it will remain zero and hence be a black pixel

    for ip, idx in interior_idx.items():
        c = int(ip[0] + (n-1)/2)              # x-axis goes left --> right
        r = (n - 1) - int(ip[1] + (n-1)/2)    # y-axis goes bottom --> top
        red_window[r][c] = 16 if show_iv else 0
        green_window[r][c] = 16 if show_iv else 0
        blue_window[r][c] = normalized_interior_values[idx]

    window_stack = np.stack((red_window, green_window, blue_window), axis=2)
    return Image.fromarray(window_stack.astype('uint8'))


# img = make_img()
# img.save('imgs/boundary0.png')
# img.show()


# ********************************
#   DISPLAY IN TERMINAL:
# ********************************
# NOTE: this gets slightly messed up when n >= 200 because "100" is three digits
# and this function displays the grid values from [-(n/2), (n/2) - 1]

# The rows will print backwards so origin is bottom left

def to_string():
    s = ''

    for y in range(n - 1, 0, -1):
        yz = grid(y)

        # Start out the row with the y-value
        row = str(yz) + '\t'

        for x in range(n):
            xz = grid(x)

            # Marking boundary
            if (xz, yz) in boundary_idx:
                row += '[@]'

            # Marking interior
            elif (xz, yz) in interior_idx:
                # Marking origin, if that's where we be
                if xz == yz == 0:
                    row += ' 0 '
                else:
                    row += ' # '

            # If the grid is big enough, maybe some vertical axes will look nice?
            # Or not... it does kind of clutter things up.
            # elif x == int((n-1)/2) and n > 32:
            #     row += '-|-'
            #
            # elif y == int((n-1)/2) and n > 32:
            #     row += '-+-'

            # Marking exterior
            else:
                row += ' + '
                # row += f'({x},{y})'

        s += row + '\n'

    # The bottom row will be x-values
    row0 = str(grid(0)) + '\t'
    for x in range(n):
        # row0 += str(grid(x+1)).center(3)
        row0 += str(grid(x)).center(3)
    s += row0 + '\n'

    return s

print(to_string())










