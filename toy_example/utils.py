import numpy as np


def add_world_boundaries(occ_cl, planar):

    x_min = min(occ_cl[:, 0])  # -1
    x_max = max(occ_cl[:, 0])  # 11
    y_min = min(occ_cl[:, 1])  # -1
    y_max = max(occ_cl[:, 1])  # 11
    z_min = min(occ_cl[:, 2])  # 0
    z_max = max(occ_cl[:, 2])  # 6
    n_side = 50

    z_side = np.linspace(z_min, z_max, n_side)
    z_side = np.repeat(z_side, n_side)

    x_side1 = np.linspace(x_min, x_max, n_side)
    y_side1 = np.linspace(y_min, y_min, n_side)
    side1 = np.vstack([x_side1, y_side1]).T
    side1 = np.hstack([np.tile(side1.T, n_side).T, z_side[:, None]])

    x_side2 = np.linspace(x_max, x_max, n_side)
    y_side2 = np.linspace(y_min, y_max, n_side)
    side2 = np.vstack([x_side2, y_side2]).T
    side2 = np.hstack([np.tile(side2.T, n_side).T, z_side[:, None]])

    x_side3 = np.linspace(x_max, x_min, n_side)
    y_side3 = np.linspace(y_max, y_max, n_side)
    side3 = np.vstack([x_side3, y_side3]).T
    side3 = np.hstack([np.tile(side3.T, n_side).T, z_side[:, None]])

    x_side4 = np.linspace(x_min, x_min, n_side)
    y_side4 = np.linspace(y_max, y_min, n_side)
    side4 = np.vstack([x_side4, y_side4]).T
    side4 = np.hstack([np.tile(side4.T, n_side).T, z_side[:, None]])

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, n_side), np.linspace(y_min, y_max, n_side)
    )
    side5 = np.vstack(
        [X.flatten(), Y.flatten(), z_min * np.ones((n_side, n_side)).flatten()]
    ).T
    side6 = np.vstack(
        [X.flatten(), Y.flatten(), z_max * np.ones((n_side, n_side)).flatten()]
    ).T

    if planar:
        bnd = np.vstack([side1[:, :2], side2[:, :2], side3[:, :2], side4[:, :2]])
    else:
        bnd = np.vstack([side1, side2, side3, side4, side5, side6])
    occ_cl = np.vstack([occ_cl, bnd])

    return occ_cl
