# space_carving.py - implements the first part of the reconstruction pipeline
# Usage:
# include the target image as "target.png" in the same directory and run the file

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib import cm


def set_axes_equal(ax: plt.axes, z_scale=1.125) -> None:
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    - Source: Modified from visutils.py - UCI CS 117 S21 C. Fowlkes

    :param ax: a matplotlib axis, e.g., as output from plt.gca().
    :param z_scale: factor to scale z-axis for better viewing
    :return: None
    """

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:2, 1] - limits[:2, 0]))
    z_radius = z_scale * np.abs(limits[2, 1] - limits[2, 0])
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - z_radius, origin[2] + z_radius])


def spinZ(a: [[float, float, float]]) -> np.array:
    """
    Spin direction vectors 90 degrees 3 times, producing a collection of
    4 copies of each vector, each spun into a unique quadrant of the XOY
    plane

    :param a: a collection of [X, Y, Z] vectors Nx3
    :return: a (4N + 1)x3 array: 3 copies of the original vectors, and the Z axis
    """
    q0 = np.array(a)
    r = Rotation.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    q1 = r.apply(q0)
    q2 = r.apply(q1)
    q3 = r.apply(q2)
    return np.vstack((q0, q1, q2, q3, (0., 0., 1)))


def lattice_hemisphere(n=9, u=False):
    """
    Populates a hemisphere (norm=[0,0,1]) of direction vectors to the closest
    neighboring cubic lattice points.

    Approximates lattice points by iteratively expanding radius of hemisphere

    Uses hemispherical volume to approximate number of desired directions

    :param n: number of desired directions
    :param u: boolean value indicating output of normalized direction vectors
    :return: ~Nx3 vector of lattice steps
    """

    # Find desired radius limit of lattice points through
    # hemisphere volume; approximate points = volume
    # n = V = (2/3) pi * r^3
    # r = (3n / (2pi))^(1/3)

    # Generate 1/4 the desired lattice points in one quadrant
    actual_r = np.power(3 * n / (2 * np.pi), 1 / 3)
    r = int(np.ceil(actual_r))
    pos_quad = set()
    ind = []
    c = 0

    # Iteratively expand the lattice coordinates, keeping the unique
    # normalized directions
    for x in range(r):
        for y in range(1, r):
            for z in range(1, r):
                if np.linalg.norm(np.array([x, y, z])) > actual_r:
                    continue
                if x == y == 0:
                    continue
                c += 1
                norm = np.linalg.norm((x, y, z))
                L = tuple((x, y, z) / norm)
                if L not in pos_quad:
                    pos_quad.add(L)
                    ind.append((x, y, z))

    # Spin the generated coordinates into the remaining 3 quadrants
    return (np.rint(spinZ(ind)).astype(int), spinZ(np.array(list(pos_quad)))) if u else \
        np.rint(spinZ(ind)).astype(int)


def static_aperture_est(image: np.array, p=0.8) -> np.array:
    """
    Estimate the aperture at each point in the target image

    :param image: np.array of gray-scale pixels NxM
    :param p: surface albedo 0. -1.
    :return: NxM array
    """

    alb_r = 1 / (1 - p)
    i_norm = image / image.max()  # top percentile (sq)
    interm = 1 - np.sqrt(alb_r * (1 - i_norm))
    interm[interm < 0] = 0
    return 0.5 * (np.sqrt(i_norm) + interm)


def initialize_V(image: np.array, h_shape: int) -> np.array:
    # Z:Y:X:{L}
    """
    Initialize the carving lattice to contain full apertures

    :param image: np.array of gray-scale pixels NxM
    :param h_shape: number of sampled lattice point directions
    :return: 1xNxM np.array of {0,1,...,h_shape}; representing unobstructed directions
    """
    return np.full((1,) + image.shape, set(range(h_shape))), np.full(image.shape, h_shape)


def Bresenham3D(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> np.array:
    """
    Bresenham3D algorithm; returns a list of voxels crossed
    given two endpoints of a line in 3D space

    Source: https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
    - Modified to work with numpy

    :param x1: starting x coordinate
    :param y1: starting y coordinate
    :param z1: starting z coordinate
    :param x2: ending x coordinate
    :param y2: ending y coordinate
    :param z2: ending z coordinate
    :return: np.array of voxels crossed
    """
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            yield np.array((x1, y1, z1))

    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            yield np.array((x1, y1, z1))

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            yield np.array((x1, y1, z1))


def march_V(v_L: np.array, reached: np.array, H: np.array) -> np.array:
    """
    Create a new layer on the lattice, checking for newly obstructed directions
    and updating the aperture estimation by remaining % of visible directions

    :param v_L: NxMxZ voxel lattice
    :param reached: NxM array indicating if each pixel/column in the NxMxZ has reached its
        estimated aperture
    :param H: sampled hemisphere directions
    :return: NxMx(Z+1) voxel lattice with newly appended slice of current apertures
    """

    V_C = np.copy(v_L[-1])
    count = np.zeros(v_L.shape[1:])
    zl = len(v_L)
    z = zl - 1
    yl, xl = V_C.shape

    # Iterate over lattice slice
    for y in range(yl):
        for x in range(xl):

            # ignore if column in lattice has been frozen
            if reached[z][y, x]:
                continue

            # Iterate over available sampling directions
            newV = set()
            for Li in V_C[y, x]:
                zp, yp, xp = np.array([z, y, x]) - H[Li]
                remove_Li = False

                # Validate the hemisphere is still visible along the sampled ray
                if 0 <= zp < zl and 0 <= xp < xl and 0 <= yp < yl:
                    found_node = v_L[zp, yp, xp]

                    # 1 x' is strictly above the surface
                    if reached[zp, yp, xp]:
                        remove_Li = True

                    # 2 Li in V(x')
                    elif Li not in found_node:
                        remove_Li = True
                    else:
                        # 3 x-x' lies above the surface
                        for node_to_check in Bresenham3D(x, y, z, xp, yp, zp):
                            xp, yp, zp = node_to_check
                            if reached[zp, yp, xp]:
                                remove_Li = True
                                break

                if not remove_Li:
                    newV.add(Li)
                    count[y, x] += 1
            V_C[y, x] = newV

    return np.vstack([v_L, [V_C]]), count


def depth_estimation(image: np.array, output="exp_d.png", threshold_p=0.1, show_progress=False) -> None:
    """
    Initializes and runs a depth estimation for the provided orthographic,
    uniformly lit and shaded image

    :param image: NxM image to estimate; color images will be averaged to black and white
    :param output: output file name
    :param threshold_p: percent to clip noisy highlights
    :param show_progress: bool to output progress of carving every 10 steps
    :return: None
    """
    # Flatten image to black & white
    if len(image.shape) > 2:
        image = image.mean(2)

    # Clip firefly noise with brightness threshold
    i_sorted = np.sort(np.unique(image.flatten()))[::-1]
    threshold = i_sorted[int(i_sorted.size * threshold_p)]
    image = np.where(image > threshold, threshold, image)
    plt.imshow(image, cmap='gray')
    plt.show()

    # initialize depth estimate array
    z = np.zeros(image.shape)

    # obtain sampling rays
    H = lattice_hemisphere(160)[:, ::-1]
    H_n = H.shape[0]
    print("Sampled directions:", H_n)

    # initialize carving lattice
    v_lattice, v_count = initialize_V(image, H_n)

    # estimate target aperture
    static_ap_est = static_aperture_est(image, 0.5)
    plt.imshow(static_ap_est, cmap='gray')
    plt.show()

    # initialize lattice tracking flags
    reached = np.zeros((1,) + image.shape, dtype=bool)

    n = 0
    while not reached[-1].all():
        # update viewed_surface
        v_lattice, v_count = march_V(v_lattice, reached, H)

        # update % of remaining directions
        aperture_est = v_count / H_n

        # increment z for A* > A~
        z[aperture_est > static_ap_est] += 1
        temp = static_ap_est - aperture_est
        reached = np.vstack([reached, (np.logical_or(np.isclose(temp, 0), temp > 0)).reshape((1,) + image.shape)])

        if show_progress and not n % 10:
            plt.imshow(v_count, cmap='gray')
            plt.title(n)
            plt.show()
        n += 1
        print(n)
        if n == 21:
            break

    # normalize z [0, 1] and save
    z = z - z.min()
    z = z / z.max()
    plt.imsave(output, z, cmap='gray')

    # display 3D plot
    view_3D(z)


def view_3D(image: np.array) -> None:
    """
    Display image as height-map in 3D space

    :param image: height map
    :return: None
    """
    # Flatten image to black & white
    if len(image.shape) > 2:
        image = image.mean(2)

    # Normalize Z
    image -= image.min()
    image /= image.max()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot surface in 3D space
    x = np.arange(0, image.shape[1], 1)
    y = np.arange(0, image.shape[0], 1)
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, image, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    set_axes_equal(ax)
    plt.show()


if __name__ == '__main__':
    depth_estimation(plt.imread("target.png"))
