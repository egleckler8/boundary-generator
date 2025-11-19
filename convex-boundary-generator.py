import numpy as np

n = 40
grid = np.zeros((n, n))
n = int((n - 1)/2)
# origin is (m+1, n+1)


# aka: g(x)
def boundary_function(p):
    return 1


max_theta = np.pi / 16
min_theta = np.pi / 2
max_r = n - 1
min_r = (n - 1)/2

# (1) Place markers on the corners
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




# (2) Interpolate between corners
boundary = []
for i in range(len(corners)):

    side = []

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
            side.append((int(x), int(y)))
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
            side.append((int(x), int(y)))
            y += 1

    boundary.extend(side)

# Might want to remove duplicates from boundary
# (and have set for O(1) lookup!!)
# (could change to dictionary with boundary fn values?)
boundary_set = set(boundary)
boundary = list(boundary_set)

# (3) transform boundary points & mark grid with g(x, y)
# Boundary origin is grid (m+1, n+1)
for i in range(len(boundary)):
    p = boundary[i]
    c = int(p[0]) + n
    r = int(p[1]) + n
    grid[r][c] = boundary_function(p)




# check out what happened:
for i in range(grid.shape[0]):
    row = ''
    for j in range(grid.shape[1]):
        if i == j == n:
            row += '[0]'
        elif grid[i][j] == 0:
            row += ' + '
        else:
            row += f'[{int(round(grid[i][j]))}]'
    print(row)









