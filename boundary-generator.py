from typing import List, Tuple
from PIL import Image
import numpy as np
import noise
import queue
import time
import json


n = 512

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (0) Boundary function                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# aka: g(x). A map R^2 --> [-1, 1]
#
# Do inquire if curious. I have spent long hours delving deeply into the dark arts
# of "perlin noise." It's just smooth noise, and the boundary samples from it.
# Hence, a smoothly random boundary function. Should be interesting for testing?
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def boundary_function(p):
    #
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
# works sorts this out. By having "if" statements
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
boundary_points = []
for i in range(len(corners)):

    if corners[i][0] != corners[(i+1) % len(corners)][0]:  # x's different, do line

        # look at the points left-to-right
        if corners[i][0] < corners[(i+1) % len(corners)][0]:
            p1, p2 = corners[i], corners[(i+1) % len(corners)]
        else:
            p2, p1 = corners[i], corners[(i + 1) % len(corners)]

        # x's different, so slope safe
        m1 = (p2[1] - p1[1])/(p2[0] - p1[0])
        b1 = p2[1] - m1 * p2[0]

        # increment x one at a time and plot y
        x = p1[0]
        while x < p2[0]:
            y = m1*x + b1
            boundary_points.append((int(x), int(y)))
            x += 1

    if corners[i][1] != corners[(i+1) % len(corners)][1]:  # y's different, do line

        # look at the points bottom-to-top
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
            boundary_points.append((int(x), int(y)))
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
# is closed! It quarantines the exterior and we end up with all the interior points.
#
# FDM doesn't care what ordering we choose for the interior and boundary points.
#


# This is the function inside the boundary, like f on the LHS of the system in FDM
# For now, for us, it can just return zero.
def f(p):
    # This can help us visualize the boundary function:
    # return boundary_function(p)

    # ... and this will surely make a fun picture, too:
    return np.sin(0.08 * (p[0] * p[1]) / n)

def get_neighbors(p: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the grid neighbors of p in standard coordinates (origin is 0).
    Totally not index-safe! Please don't use this diretly on an array...
    :param p: Grid point in standard coordinates
    :return: Grid neighbors of p
    """
    return [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]


t0 = time.perf_counter()  # let's see how long this takes in practice... algo is O(n^2) probably

# Here's the duplicates-removed boundary set, equipped with boundary values:
# Python `dict` data type gives O(1) lookup for membership
boundary_map = {bp: boundary_function(bp) for bp in boundary_points}
interior_map = {}
q = queue.Queue()
q.put((0, 0))  # Infect the origin

# Origin infects the whole interior. Boundary quarantines effectively.
# Huge fan of constant time dict lookup
while not q.empty():
    p = q.get()
    if p not in boundary_map and p not in interior_map:
        interior_map[p] = f(p)
        for neighbor in get_neighbors(p):
            q.put(neighbor)

tf = time.perf_counter()
dt = tf - t0

print('****************************************************************')
print('\tBOUNDARY GENERATION COMPLETE. INTERIOR GENERATION COMPLETE.')
print('****************************************************************')
print('Number of interior points:', len(interior_map), f' --> {len(interior_map) / (n * n):.2%} of the {n}x{n} grid.')
print('Number of boundary points:', len(boundary_map), f' --> {len(boundary_map) / (n * n):.2%} of the {n}x{n} grid.')
print('Time to compute interior:', f'~{dt:.4f}s')
print(f'~{(len(interior_map) / dt):.2f} points per second.')


# Finally, let's vectorize the boundary and interior:
# Making it a np.array here in case we want to use that,
# converting back to a list to json serialize
boundary_vec = np.array(list(boundary_map.values()))
interior_vec = np.array(list(interior_map.values()))
closure_vec = np.concatenate([interior_vec, boundary_vec])  # boundary, then interior--as the book stipulated

# Bam.

# Shall we go so far as to output a json file?
data = {
    'boundary_vec': list(boundary_vec),
    'interior_vec': list(interior_vec),
    'closure_vec': list(closure_vec)
}
filename = 'output/boundary0.json'
try:
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        print('****************************************************************')
        print(f'JSON DATA SUCCESSFULLY SAVED TO: {filename}.')
        print('****************************************************************')
except IOError as e:
    print('****************************************************************')
    print('ERROR WRITING JSON DATA TO: {filename}')
    print('****************************************************************')

# Double bam.


# Let's get creative:
# ********************************
#   DISPLAY AS PIXEL MAP:
# ********************************

# We'll layer these to create an RGB image
red_window = np.zeros((n, n))
green_window = np.zeros((n, n))
blue_window = np.zeros((n, n))

# rgb normalize the boundary values:
boundary_values = np.array([bv for bv in boundary_map.values()])
bv_min = np.min(boundary_values)
bv_max = np.max(boundary_values)

# We need to make a map [-1, 1] --> [0, 255].
# For enhanced visualization, we first min-max normalize the boundary values,
# linearly squeezing them into to fit perfectly into [0, 1] (min +-> 0, max +-> 1)
# Then a simple multiplication by 255 puts them into the 8-bit unsigned int range for RGB
normalized_boundary_map = {}
for bp, bv in boundary_map.items():
    normalized_bv = 255*(bv - bv_min)/(bv_max - bv_min) if bv_max != bv_min else 0
    normalized_boundary_map[bp] = normalized_bv

# Make a "layer" for each RGB color.
# We can be creative or whatever to make it look nice, e.g. green ==> good, red ==> bad
for bp, bv in normalized_boundary_map.items():
    c = int(bp[0] + n / 2)
    r = int(bp[1] + n / 2)
    red_window[r][c] = 255 - bv
    green_window[r][c] = bv
    blue_window[r][c] = 0


# Now do the same for the interior.
# It's going to be a real nice addition to the scrapbook.
interior_values = np.array([iv for iv in interior_map.values()])
iv_min = np.min(interior_values)
iv_max = np.max(interior_values)

normalized_interior_map = {}
for ip, iv in interior_map.items():
    normalized_iv = 128*(iv - iv_min)/(iv_max - iv_min) if iv_max != iv_min else 0
    normalized_interior_map[ip] = normalized_iv

for ip, iv in normalized_interior_map.items():
    c = int(ip[0] + n / 2)
    r = int(ip[1] + n / 2)
    red_window[r][c] = 0
    green_window[r][c] = 0
    blue_window[r][c] = iv

window_stack = np.stack((red_window, green_window, blue_window), axis=2)
img = Image.fromarray(window_stack.astype('uint8'))
img.save('imgs/boundary0.png')
# img.show()


# ********************************
#   DISPLAY IN TERMINAL:
# ********************************
# grid = np.zeros((n, n))
# for i in range(grid.shape[0]):
#     row = ''
#     for j in range(grid.shape[1]):
#         if i == j == n:
#             row += '[O]'
#         elif (int(i - n / 2), int(j - n / 2)) in boundary_map:
#             row += '[@]'
#         elif (int(i - n / 2), int(j - n / 2)) in interior_map:
#             row += '###'
#         else:
#             row += ' + '
#
#     print(row)










