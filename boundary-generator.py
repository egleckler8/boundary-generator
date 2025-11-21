from typing import List, Tuple
from scipy.sparse import csr_matrix
from PIL import Image
import numpy as np
import noise
import queue
import time
import json


n = 40

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


def g(p: Tuple[int, int]) -> float:

    return p[0]
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
boundary_idx = {}

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
            if (int(x), int(y)) not in boundary_idx:
                boundary_idx[(int(x), int(y))] = len(boundary_idx) - 1
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
            if (int(x), int(y)) not in boundary_idx:
                boundary_idx[(int(x), int(y))] = len(boundary_idx) - 1
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
# FDM doesn't care what ordering we choose for the interior and boundary points.
#


# This is the function inside the boundary, like f on the LHS of the system in FDM
# For now, for us, it can just return zero.
def f(p: Tuple[int, int]) -> float:
    # This can help us visualize the boundary function:
    # return g(p)
    # This can help us check the orientation:
    return p[0]
    # ... and this will surely make a fun picture, too:
    # return np.sin(0.08 * (p[0] * p[1]) / n)


def get_neighbors(p: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the grid neighbors of p in standard coordinates (origin is 0).
    Totally not index-safe! Please don't use this directly on an array...
    :param p: Grid point in standard coordinates
    :return: Grid neighbors of p
    """
    return [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]


t0 = time.perf_counter()  # let's see how long this takes in practice... algo is O(n^2) probably

# Here's the duplicates-removed boundary set, equipped with an ordering.
# Python `dict` data type gives O(1) lookup for membership
# boundary_idx = {}
# for bp in boundary_points:
#     boundary_idx[bp] = len(boundary_idx) - 1

interior_idx = {}
q = queue.Queue()
q.put((0, 0))  # Infect the origin

# Origin infects the whole interior. Boundary quarantines effectively.
# Huge fan of constant time dict lookup
while not q.empty():
    p = q.get()
    if p not in boundary_idx and p not in interior_idx:

        # Setting the dict entry to the size of the set at the time gives
        # us an ordering on the interior. Now we can map each point to
        # its index in a vector. Will come in clutch when we vectorize D
        interior_idx[p] = len(interior_idx) - 1

        for neighbor in get_neighbors(p):
            q.put(neighbor)

tf = time.perf_counter()
dt = tf - t0

print('****************************************************************')
print('\tBOUNDARY GENERATION COMPLETE. INTERIOR GENERATION COMPLETE.')
print('****************************************************************')
print('Number of interior points:', len(interior_idx), f' --> {len(interior_idx) / (n * n):.2%} of the {n}x{n} grid.')
print('Number of boundary points:', len(boundary_idx), f' --> {len(boundary_idx) / (n * n):.2%} of the {n}x{n} grid.')
print('Time to compute interior:', f'~{dt:.4f}s')
print(f'~{(len(interior_idx) / dt):.2f} points per second.')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (3) Vectorize interior & boundary                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# We need all the components of "Au = A'g + f"
#

# (1) f vector - interior values
f_vec = np.zeros(len(interior_idx))
for ip in interior_idx:
    f_vec[interior_idx[ip]] = f(ip)

# (2) g vector - boundary values
g_vec = np.zeros(len(boundary_idx))
for bp in boundary_idx:
    g_vec[boundary_idx[bp]] = g(bp)



# (3) A' matrix. Use CSR format because it's pretty big.
# TODO

# (4) The mighty A matrix. Use CSR format because it's HUGE.
# TODO

# Bam.

# Shall we go so far as to output a json file?
# data = {
#     'boundary_vec': list(boundary_vec),
#     'interior_vec': list(interior_vec),
#     'closure_vec': list(closure_vec)
# }
# filename = 'output/boundary0.json'
# try:
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=4)
#         print('****************************************************************')
#         print(f'JSON DATA SUCCESSFULLY SAVED TO: {filename}.')
#         print('****************************************************************')
# except IOError as e:
#     print('****************************************************************')
#     print('ERROR WRITING JSON DATA TO: {filename}')
#     print('****************************************************************')

# Double bam.


# Let's get creative:
# ********************************
#   DISPLAY AS PIXEL MAP:
# ********************************

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
    normalized_boundary_values[i] = 255*(g_vec[i] - bv_min)/(bv_max - bv_min) if bv_max != bv_min else 0

    # In case we just want the un-normalized values
    # normalized_boundary_values[i] = g_vec[i]

# Make a "layer" for each RGB color.
# We can be creative or whatever to make it look nice, e.g. green ==> good, red ==> bad
for bp, idx in boundary_idx.items():
    c = int(bp[0] + n / 2)              # x-axis goes left --> right
    r = (n - 1) - int(bp[1] + n / 2)    # y-axis goes bottom --> top
    red_window[r][c] = 255 - normalized_boundary_values[idx]
    green_window[r][c] = normalized_boundary_values[idx]
    blue_window[r][c] = 0


# Now do the same for the interior.
# It's going to be a real nice addition to the scrapbook.
iv_min = np.min(f_vec)
iv_max = np.max(f_vec)

normalized_interior_values = np.zeros_like(f_vec)
for i in range(len(f_vec)):
    normalized_interior_values[i] = 128*(f_vec[i] - iv_min)/(iv_max - iv_min) if iv_max != iv_min else 0

for ip, idx in interior_idx.items():
    c = int(ip[0] + n / 2)              # x-axis goes left --> right
    r = (n - 1) - int(ip[1] + n / 2)    # y-axis goes bottom --> top
    red_window[r][c] = 0
    green_window[r][c] = 0
    blue_window[r][c] = normalized_interior_values[idx]

window_stack = np.stack((red_window, green_window, blue_window), axis=2)
img = Image.fromarray(window_stack.astype('uint8'))
img.save('imgs/boundary0.png')
# img.show()


# ********************************
#   DISPLAY IN TERMINAL:
# ********************************
# NOTE: this gets slightly messed up when n >= 200 because "100" is three digits
# and this function displays the grid values from [-(n/2), (n/2) - 1]

# The rows will print backwards so origin is bottom left
for y in range(n - 1, 0, -1):

    # Start out the row with the y-value
    row = str(int(y - n/2)) + '\t'

    for x in range(n):

        # Marking origin
        if x == y == n:
            row += '[O]'

        # Marking boundary
        elif (int(x - n / 2), int(y - n / 2)) in boundary_idx:
            # row += '[@]'
            bv_for_display = str(int(normalized_boundary_values[boundary_idx[int(x - n/2), int(y - n/2)]]))
            row += f'{str(bv_for_display)}   '[:3]

        # Marking interior
        elif (int(x - n / 2), int(y - n / 2)) in interior_idx:
            row += '###'

        # If the grid is big enough, maybe some vertical axes will look nice?
        # Or not... it does kind of clutter things up.
        # elif x == int(n/2) and n > 32:
        #     row += '-|-'
        #
        # elif y == int(n/2) and n > 32:
        #     row += '-+-'

        # Marking exterior
        else:
            row += ' + '
            # row += f'({x},{y})'

    print(row)

# The bottom row will be x-values
row0 = '0\t'
for x in range(n):
    row0 += f'{int(x  - n/2) }   '[:3]
print(row0)










