import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from papor.utils.visualize import axis_equal


def ellipse_3d(center, size):
    # Generate data points for the ellipse
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + size[0] * np.cos(u)
    y = center[1] + size[1] * np.sin(u)
    z = center[2] + np.zeros_like(u)

    points = np.vstack([x, z, y]).T

    # points = np.swapaxes(points, 0, 2)
    return points


def box_3d(center, dimensions, point_cloud_density=0.1):
    # Extracting dimensions
    width, height, length = dimensions

    # Define the corners of the box
    corners = np.array(
        [
            [-width / 2, -height / 2, -length / 2],  # Corner 1
            [width / 2, -height / 2, -length / 2],  # Corner 2
            [width / 2, height / 2, -length / 2],  # Corner 3
            [-width / 2, height / 2, -length / 2],  # Corner 4
            [-width / 2, -height / 2, length / 2],  # Corner 5
            [width / 2, -height / 2, length / 2],  # Corner 6
            [width / 2, height / 2, length / 2],  # Corner 7
            [-width / 2, height / 2, length / 2],  # Corner 8
        ]
    )

    # Translate corners to the center
    corners += center

    # Define the edges of the box
    edges = [
        [corners[0], corners[1]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[0]],
        [corners[4], corners[5]],
        [corners[5], corners[6]],
        [corners[6], corners[7]],
        [corners[7], corners[4]],
        [corners[0], corners[4]],
        [corners[1], corners[5]],
        [corners[2], corners[6]],
        [corners[3], corners[7]],
    ]

    # Generate point cloud around the surface of the box
    cloud_points = []
    for i in np.arange(-width / 2, width / 2, point_cloud_density):
        for j in np.arange(-height / 2, height / 2, point_cloud_density):
            cloud_points.append([i, j, -length / 2])
            cloud_points.append([i, j, length / 2])
    for i in np.arange(-width / 2, width / 2, point_cloud_density):
        for k in np.arange(-length / 2, length / 2, point_cloud_density):
            cloud_points.append([i, -height / 2, k])
            cloud_points.append([i, height / 2, k])
    for j in np.arange(-height / 2, height / 2, point_cloud_density):
        for k in np.arange(-length / 2, length / 2, point_cloud_density):
            cloud_points.append([-width / 2, j, k])
            cloud_points.append([width / 2, j, k])
    points = np.array(cloud_points) + center

    return points


# --------------------------- Generate point cloud --------------------------- #
# ellipses
center1 = (0, 0, 0)  # Center point of the ellipse
size1 = (3, 2)  # Size of the ellipse along x and y axes respectively
ellipse1 = ellipse_3d(center1, size1)

center2 = (0, 0, 10)  # Center point of the ellipse
size2 = (2, 4)  # Size of the ellipse along x and y axes respectively
ellipse2 = ellipse_3d(center2, size2)

# boxes
center3 = (2, 5, 0)  # Center point of the box
dimensions = (1, 1, 7)  # Width, height, and length of the box
box = box_3d(center3, dimensions, point_cloud_density=0.3)

cloud = np.vstack([ellipse1, ellipse2, box])


# --------------------------------- Visualize -------------------------------- #

ax = plt.figure().add_subplot(111, projection="3d")
ax.scatter(
    cloud[:, 0],
    cloud[:, 1],
    cloud[:, 2],
    c=cloud[:, 2],
    cmap="turbo",
    alpha=0.5,
)


axis_equal(cloud[:, 0], cloud[:, 1], cloud[:, 2], ax=plt.gca())

plt.show()
