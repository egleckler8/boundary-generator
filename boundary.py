from typing import List, Tuple, Set, Callable
from PIL import Image
import numpy as np
import queue
import time
# import json # TODO: Export boundary as json?

STAR_BAR = '********************************************************'


def generate_corners(theta_0: float = 0,
                     theta_f: float = 2*np.pi,
                     max_theta: float = np.pi/2,
                     min_theta: float = np.pi/16,
                     max_r=1.0,
                     min_r=0.5) -> List[Tuple[float, float]]:
    """
    Generates corners of a boundary within the unit disc.
    Start at theta_0 and pick a random radius. Mark a corner. Then turn
    counterclockwise by a random theta and pick another random radius,
    mark a point, etc. until theta_f.

    At each step, the random angle of rotation and radius are chosen
    from a uniform distribution.

    :param theta_0: Starting angle
    :param theta_f: Ending angle
    :param max_theta: Biggest possible random rotation angle
    :param min_theta: Smallest possible random rotation angle
    :param max_r: Biggest possible random radius
    :param min_r: Smallest possible random radius
    :return: List of corners in the unit disc in counterclockwise order
    """
    theta = theta_0
    C = []

    # Never forget about numerical error...
    while theta < theta_f - 0.001:

        # To test regular polygons:
        # r = min(m, n)/2
        # theta += (2*np.pi)/k

        r = np.random.uniform(min_r, max_r)
        x, y = r * np.cos(theta), r * np.sin(theta)
        C.append((x, y))
        theta += np.random.uniform(min_theta, max_theta)

    return C


def discrete_interpolate(a: Tuple[float, float],
                         b: Tuple[float, float]) -> Set[Tuple[int, int]]:
    """
    Takes two points in the plane (R^2) and returns a line between them
    embedded in the grid (Z^2). This line is not guaranteed to be "minimal"
    in the sense that it uses the least amount of points to connect the two lines,
    only that the line will be straight and there will be no gaps.

    :param a: First plane point
    :param b: Second plane point
    :return: Set of the grid points between a and b (includes a and b)
    """

    # Store the points generated in a set so no dupes, quick lookup
    line = set()

    # x's different, do "y = mx + b" between points
    if a[0] != b[0]:

        # Look at the points bottom-to-top
        if a[0] < b[0]:
            p1, p2 = a, b
        else:
            p2, p1 = a, b

        # y's different, so slope safe
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p2[1] - m1 * p2[0]

        x = p1[0]
        while x < p2[0]:
            y = m1*x + b1
            line.add((int(x), int(y)))
            x += 1

    # y's different, do "x = my + b" between points for good measure
    if a[1] != b[1]:

        # Look at the points bottom-to-top
        if a[1] < b[1]:
            p1, p2 = a, b
        else:
            p2, p1 = a, b

        # y's different, so slope safe
        m2 = (p2[0] - p1[0]) / (p2[1] - p1[1])
        b2 = p2[0] - m2 * p2[1]

        y = p1[1]
        while y < p2[1]:
            x = m2*y + b2
            line.add((int(x), int(y)))
            y += 1

    return line


def rgb_normalize(values: np.ndarray) -> np.array:
    """
    Normalizes values in any float range to integers [0, 255].
    Accomplishes this by mapping the min value in the input
    vector to 0 and the max to 255, linearly interpolating in
    between. Truncates to integers for output

    :param values: ``np.array`` of values to normalize
    :return: An ``np.array`` of values normalized to integers [0, 255]
    """
    min_val = np.min(values)
    max_val = np.max(values)
    normalized_values = 255 * np.ones_like(values)

    # Choose to "error out" by returning all 255.
    if min_val == max_val:
        return normalized_values

    for i in range(len(values)):
        normalized_values[i] = int(255*(values[i] - min_val) / (max_val - min_val))
    return normalized_values


class Boundary:
    """
    Class to represent a closed boundary in a grid space.

    Try properties ``boundary_size`` and `interior_size`` to get the number
    of points on and within the boundary, respectively.

    :ivar interior_idx: Dictionary assigning a unique ID to each point
        inside the boundary. ID's range [0, interior_size].
    :ivar boundary_idx: Dictionary assigning a unique ID to each point on the boundary.
        ID's range [0, boundary_size].
    """

    def __init__(self, n: int):
        """
        Constructor

        TODO: make accessible attributes immutable. Would be terrifying if users could
        edit and screw up the boundary/interior index. Use tuples? Vectors for now

        :param n: length of one side of the grid containing boundary
        """

        self.n = n
        self.interior_idx = {}
        self.boundary_idx = {}

        # NOTE: Corners are generated in the unit disc, so we must dilate them to fit the nxn grid
        boundary_set = set()
        corners = generate_corners()
        corners = [(int(corner[0] * (n/2)), int(corner[1] * (n/2))) for corner in corners]
        for i in range(len(corners)):
            a, b = corners[i], corners[(i + 1) % len(corners)]
            boundary_set.update(discrete_interpolate(a, b))

        # Let's see how long this takes in practice... algo is O(n^2) probably
        t0 = time.perf_counter()

        # Origin infects the whole interior. Boundary quarantines effectively.
        # Keep on the relevant boundary points. Huge fan of constant time dict lookup.
        q = queue.Queue()
        q.put((0, 0))
        while not q.empty():
            p = q.get()

            # The boundary is strictly generated so that it will never touch the grid edges.
            # Therefore, the interior will not, either. If this happens, something failed.
            if int(p[0] - (n - 1) / 2) in {0, n} or int(p[1] - (n - 1) / 2) in {0, n}:
                raise Exception('CRITICAL FAILURE: BOUNDARY NOT CLOSED!')

            if p not in boundary_set:
                if p not in self.interior_idx:

                    # Setting the dict entry to the size of the set at the time gives
                    # us an ordering on the interior. Now we can map each point to
                    # its index in a vector. Will come in clutch when we vectorize D
                    self.interior_idx[p] = len(self.interior_idx)

                    neighbors = [(p[0], p[1] + 1), (p[0], p[1] - 1), (p[0] + 1, p[1]), (p[0] - 1, p[1])]
                    for neighbor in neighbors:
                        q.put(neighbor)

            # Here we are constructing a minimal boundary, including on the
            # points that the interior points are touching.
            else:
                # PLEASE!! do not try to double-add. this will TOTALLY screw up ``boundary_idx``
                # because if a duplicate is considered here, it will update, but it won't change
                # the length, and the next unique point added will then have the same index. Bad!!
                if p not in self.boundary_idx:
                    self.boundary_idx[p] = len(self.boundary_idx)

        tf = time.perf_counter()
        dt = tf - t0

        # Now that the boundary/interior_idx's are created, I feel comfortable defining
        # properties for the number of points in each set--see properties far below.
        print(STAR_BAR)
        print('BOUNDARY GENERATION COMPLETE'.center(len(STAR_BAR)))
        print(STAR_BAR)
        print(f'Boundary centered at (0,0) in a {n}x{n} grid')
        print('Number of interior points:', self.interior_size,
              f'\t({(self.interior_size / (n * n)):.2%} of the grid)')
        print('Number of boundary points:', self.boundary_size,
              f'\t({(self.boundary_size / (n * n)):.2%} of the grid)')
        print('Time to compute interior:', f'~{dt:.4f}s')
        print(f'~{(self.interior_size / dt):.2f} points per second.')
        print(STAR_BAR)

    def vectorize_interior(self,
                           interior_function: Callable[[Tuple[float, float]], float] = lambda p: 0.0
                           ) -> np.ndarray:
        """
        :return: Vector of interior values indexed by ``self.interior_idx``
        """
        f_vec = np.zeros(self.interior_size)
        for ip in self.interior_idx:
            f_vec[self.interior_idx[ip]] = interior_function(ip)
        return f_vec

    def vectorize_boundary(self,
                           boundary_function: Callable[[Tuple[float, float]], float] = lambda p: 1.0
                           ) -> np.ndarray:
        """
        :return: Vector of boundary values indexed by ```self.boundary_idx``
        """
        g_vec = np.zeros(self.boundary_size)
        for bp in self.boundary_idx:
            g_vec[self.boundary_idx[bp]] = boundary_function(bp)
        return g_vec

    def make_img(self,
                 interior_function: Callable[[Tuple[float, float]], float] = lambda p: 0.0,
                 boundary_function: Callable[[Tuple[float, float]], float] = lambda p: 1.0) -> Image.Image:
        """
        Makes a cool picture of the boundary
        :param interior_function: Will be evaluated over interior points & rgb normalized
        :param boundary_function: Will be evaluated over boundary points & rgb normalized
        :return: Image object for saving or showing. (hint: img = make_img(...); img.show())
        """
        # We'll layer these to create an RGB image
        red_window = np.zeros((self.n, self.n))
        green_window = np.zeros((self.n, self.n))
        blue_window = np.zeros((self.n, self.n))

        # First vectorize & rgb normalize
        interior_vec = self.vectorize_interior(interior_function)
        boundary_vec = self.vectorize_boundary(boundary_function)
        normalized_ivs = rgb_normalize(interior_vec)
        normalized_bvs = rgb_normalize(boundary_vec)

        # Make a "layer" for each RGB color.
        # We can be creative or whatever to make it look nice, e.g. green ==> good, red ==> bad
        for bp, idx in self.boundary_idx.items():
            c = int(bp[0] + (self.n - 1) / 2)  # x-axis goes left --> right
            r = (self.n - 1) - int(bp[1] + (self.n - 1) / 2)  # y-axis goes bottom --> top
            red_window[r][c] = 255 - normalized_bvs[idx]
            green_window[r][c] = normalized_bvs[idx]
            blue_window[r][c] = 0

        for ip, idx in self.interior_idx.items():
            c = int(ip[0] + (self.n - 1) / 2)  # x-axis goes left --> right
            r = (self.n - 1) - int(ip[1] + (self.n - 1) / 2)  # y-axis goes bottom --> top
            red_window[r][c] = 0
            green_window[r][c] = 0
            blue_window[r][c] = normalized_ivs[idx]

        # Mark the origin
        o = int((self.n - 1) / 2)
        red_window[o][o] = 255
        green_window[o][o] = 255
        blue_window[o][o] = 255

        # & two extra points to mark the directions
        # red_window[o+3][o] = 0
        # green_window[o+3][o] = 255
        # blue_window[o+3][o] = 255
        #
        # red_window[o][o+3] = 255
        # green_window[o][o+3] = 0
        # blue_window[o][o+3] = 255

        window_stack = np.stack((red_window, green_window, blue_window), axis=2)
        return Image.fromarray(window_stack.astype('uint8'))

    def _grid(self, i: int) -> int:
        """
        Helper function:
        Converts array index to point on the integer line based on N
        :param i: array index
        :return: Point on integer line
        """
        return int(i - (self.n - 1) / 2)

    def __str__(self):
        """
        NOTE: this gets slightly messed up when n >= 200 because "100" is three digits
        and this function displays the grid values from [-(n/2), (n/2) - 1].

        NOTE: This also gets fucked up when n is a multiple of two because then
        the origin will be marked by a 2x2 square of zeroes... Therefore the grid
        axis printed is not to be blindly trusted for mathematical purposes.
        TODO: Still working on this one...

        The rows will print backwards so origin is bottom left

        :return: string representation of the boundary
        """
        s = ''

        for y in range(self.n):
            yz = self._grid((self.n - 1) - y)

            # Start out the row with the y-value
            row = str(yz) + '\t'

            for x in range(self.n):
                xz = self._grid(x)

                # Marking boundary
                if (xz, yz) in self.boundary_idx:
                    row += '[@]'
                    # row += f'[{str(round(g((xz, yz)), 2))}]'

                # Marking interior
                elif (xz, yz) in self.interior_idx:
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
        row0 = '\t'
        for x in range(self.n):
            row0 += str(self._grid(x)).center(3)
        s += row0 + '\n'

        return s

    @property
    def interior_size(self):
        return len(self.interior_idx)

    @property
    def boundary_size(self):
        return len(self.boundary_idx)

