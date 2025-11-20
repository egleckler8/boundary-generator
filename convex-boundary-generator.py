from PIL import Image
import numpy as np
import noise


n = 500
grid = np.zeros((n, n))


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
boundary = []
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
            boundary.append((int(x), int(y)))
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
            boundary.append((int(x), int(y)))
            y += 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                       (3) Make the boundary + interior                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Here's the duplicates-removed boundary set, equipped with boundary values:
# Python `dict` data type gives O(1) lookup for membership
boundary_map = {bp: boundary_function(bp) for bp in boundary}

# Now let's make the interior set.
#
for i in range(n):
    inside_boundary = False
    for j in range(n):
        if (i, j) in boundary_map:
            inside_boundary = not inside_boundary



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
    normalized_bv = 255*(bv - bv_min)/(bv_max - bv_min)
    normalized_boundary_map[bp] = normalized_bv

# Make a "layer" for each RGB color.
# We can be creative or whatever to make it look nice, e.g. green ==> good, red ==> bad
red_window = np.zeros((n, n))
green_window = np.zeros((n, n))
blue_window = np.zeros((n, n))
for bp, bv in normalized_boundary_map.items():
    c = int(bp[0] + n / 2)
    r = int(bp[1] + n / 2)
    red_window[r][c] = 255 - bv
    green_window[r][c] = bv
    blue_window[r][c] = 0

window_stack = np.stack((red_window, green_window, blue_window), axis=2)
img = Image.fromarray(window_stack.astype('uint8'))
img.show()


# ********************************
#   DISPLAY IN TERMINAL:
# ********************************

# for i in range(grid.shape[0]):
#     row = ''
#     for j in range(grid.shape[1]):
#         if i == j == n:
#             row += '[O]'
#         elif grid[i][j] == 0:
#             row += ' + '
#         else:
#             row += f'[{int(round(grid[i][j]))}]'
#     print(row)









