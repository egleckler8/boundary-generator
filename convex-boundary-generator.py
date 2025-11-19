import numpy as np

m = 25
n = 25
grid = np.zeros((2*m+1, 2*n+1))
# origin is (m+1, n+1)


# aka: g(x)
def boundary_function(p):
    return 1


k = 13
max_theta = np.pi / 8
min_theta = np.pi / 3
max_r = min(m, n) - 1
min_r = (min(m, n) - 1)/2

# (1) Place markers on the corners
theta = 0
corners = []

while theta < 2*np.pi - 0.001:
    r = np.random.uniform(min_r, max_r)
    x, y = int(r * np.cos(theta)), int(r * np.sin(theta))
    corners.append((x, y))
    theta += np.random.uniform(min_theta, max_theta)
    #theta += (2*np.pi)/k


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


# (3) transform boundary points & mark grid with g(x, y)
# Boundary origin is grid (m+1, n+1)
for i in range(len(boundary)):
    p = boundary[i]
    c = int(p[0]) + n
    r = int(p[1]) + m
    # grid[r][c] = boundary_function(p)
    grid[r][c] = 1


# check out what happened:
for i in range(grid.shape[0]):
    row = ''
    for j in range(grid.shape[1]):
        if i == m and j == n:
            row += '[0]'
        elif grid[i][j] == 0:
            row += ' + '
        else:
            row += f'[{int(round(grid[i][j]))}]'
    print(row)









