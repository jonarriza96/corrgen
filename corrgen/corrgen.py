import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from scipy.spatial.distance import cdist

import time


def poly_basis(xi, k, basis, d):
    if basis == "n":  # nominal
        return xi**k
    elif basis == "c":  # chebyshev
        return np.cos(k * np.arccos(2 * xi - 1))
    elif basis == "b":  # bernstein
        return np.math.comb(d, k) * xi**k * (1 - xi) ** (d - k)


def polynomial(xi, coeffs, degree):
    p_basis = "c"
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for k in range(degree + 1):
        a += coeffs["a"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)
        b += coeffs["b"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)
        c += coeffs["c"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)
        d += coeffs["d"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)
        e += coeffs["e"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)

    return a, b, c, d, e


def NLP(poly_deg, occ, ellipse_axis_lims, eps, LP):

    eig_min = 4 / (ellipse_axis_lims[0] ** 2)
    eig_max = 4 / (ellipse_axis_lims[1] ** 2)
    n_sweep = 100

    # define variables X = [[a c], [c b]]
    a = cp.Variable(poly_deg + 1)
    b = cp.Variable(poly_deg + 1)
    c = cp.Variable(poly_deg + 1)
    d = cp.Variable(poly_deg + 1)
    e = cp.Variable(poly_deg + 1)
    coeffs = {"a": a, "b": b, "c": c, "d": d, "e": e}

    # cost
    cost = 0
    # for k in range(poly_deg + 1):
    #     cost += a[k] / (k + 1)
    #     cost += b[k] / (k + 1)
    xi_sweep = np.linspace(0, 1, n_sweep)
    for i in range(n_sweep):
        a_i, b_i, c_i, d_i, e_i = polynomial(
            xi=xi_sweep[i], degree=poly_deg, coeffs=coeffs
        )
        cost += a_i + b_i  # - (d_i + e_i)

    # constraints
    constraints = []

    # 1- sweep path
    xi_sweep = np.linspace(0, 1, n_sweep)
    for i in range(n_sweep):

        # calculate ellipse matrix
        a_i, b_i, c_i, d_i, e_i = polynomial(
            xi=xi_sweep[i], degree=poly_deg, coeffs=coeffs
        )
        Pi = cp.bmat([[a_i, c_i], [c_i, b_i]])

        # constraints
        if not LP:
            # if i == 0:
            #     constraints += [(Pi - eig_min * np.eye(2)) >> 0]
            #     constraints += [d_i == 0, e_i == 0]
            # else:
            constraints += [Pi >> 0]
            # constraints += [(Pi - eig_min * np.eye(2)) >> 0]
            # constraints += [(-Pi + eig_max * np.eye(2)) >> 0]
        else:
            # min_det = (4 / (ellipse_axis_lims[0]) ** 2) * (
            #     4 / (ellipse_axis_lims[0] / 2) ** 2
            # )
            # c_min = 1 / 2 * (min_det / eps - eps)
            # constraints += [
            #     a_i >= c_i + eps,
            #     b_i >= c_i + eps,
            #     c_i >= c_min,
            # ]
            constraints += [a_i >= c_i, b_i >= c_i]

    if not LP:
        # print("CORRGEN --> Ellipse bounds [eig_min]:", eig_min)
        print("CORRGEN --> NO ellipse bounds!")
    else:
        # print("CORRGEN --> Ellipse bounds [eps, c_min]:", eps, c_min)
        print("CORRGEN --> NO ellipse bounds!")

    # 2- loop all occupancy points
    for i in range(occ.shape[0]):
        xi_i = occ[i, 0]
        w_i = occ[i, 1:]

        # calculate ellipse matrix and vector
        a_i, b_i, c_i, d_i, e_i = polynomial(xi=xi_i, degree=poly_deg, coeffs=coeffs)
        Pi = cp.bmat([[a_i, c_i], [c_i, b_i]])
        ppi = cp.bmat([[d_i, e_i]])

        # constraints
        constraints += [w_i.T @ Pi @ w_i + ppi @ w_i >= 1]

    # for k in range(poly_deg + 1):
    #     constraints += [d[k] == 0, e[k] == 0]

    # define problem
    prob = cp.Problem(cp.Minimize(cost), constraints)

    return prob, [a, b, c, d, e]


def project_cloud_to_parametric_path(
    pcl, parametric_path, safety_check=False, prune=True
):

    # do projection
    if pcl.shape[1] == 2:
        parametric_path["p"] = parametric_path["p"][:, :2]
        parametric_path["erf"] = parametric_path["erf"][:, :2, :2]
    t1 = time.time()
    dists_sc = cdist(parametric_path["p"], pcl, "sqeuclidean")
    ind_proj = np.argmin(dists_sc, axis=0)
    t2 = time.time()
    print("\nCORRGEN --> Projection:", 1000 * (t2 - t1), "ms")

    xi_occ = parametric_path["xi"][ind_proj]
    p_occ = parametric_path["p"][ind_proj]
    erf_occ = parametric_path["erf"][ind_proj]
    w_occ = np.zeros((pcl.shape[0], pcl.shape[1]))
    for k in range(pcl.shape[0]):  # TODO: Parallelize
        w_occ[k, :] = (pcl[k] - p_occ[k]) @ erf_occ[k]
    occ_erf = np.hstack([xi_occ[:, None], w_occ[:, 1:]])

    # remove points not properly projected
    if prune:
        if pcl.shape[1] == 3:
            ind_proj = np.abs(w_occ[:, 0]) < 0.01
            occ_erf = occ_erf[ind_proj]
    else:
        ind_proj = np.ones(pcl.shape[0], dtype=bool)
    # get trasnverse distances
    d_tr = np.linalg.norm(occ_erf[:, 1:], axis=1)
    min_d_tr = np.min(d_tr)
    max_d_tr = np.max(d_tr)
    # print("CORRGEN --> Transverse distances [min, max]:", min_d_tr, max_d_tr)

    # safety check
    if safety_check:

        plt.figure()
        ax = plt.subplot(211)
        plt.plot(xi_occ, w_occ[:, 0], ".")
        if pcl.shape[1] == 3:
            plt.plot(xi_occ[~ind_proj], w_occ[~ind_proj, 0], ".")
        plt.ylabel(r"$d_\parallel$")
        ax.set_title(
            r"Projection uniqueness check $d_{\parallel}\approx0$",
        )

        ax = plt.subplot(212)
        ind_ord = np.argsort(occ_erf[:, 0])
        w_norm = np.linalg.norm(occ_erf[ind_ord, 1:], axis=1)
        plt.plot(occ_erf[ind_ord, 0], w_norm, ".")
        plt.plot([0, 1], [min_d_tr, min_d_tr], "k--")
        plt.plot([0, 1], [max_d_tr, max_d_tr], "k--")
        plt.xlabel(r"$\xi$")
        plt.ylabel(r"$||w||$")
        # ax.set_title(r"$||w||=$")

        plt.show()

    return occ_erf, min_d_tr, max_d_tr, ind_proj


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


def add_roof_floor(ref_path, occ_cl, kitti_zmax, kitti_zmin):

    cl_x_min = np.min(occ_cl[:, 0])
    cl_x_max = np.max(occ_cl[:, 0])
    cl_y_min = np.min(occ_cl[:, 1])
    cl_y_max = np.max(occ_cl[:, 1])

    rp_x_min = np.min(ref_path[:, 0])
    rp_x_max = np.max(ref_path[:, 0])
    rp_y_min = np.min(ref_path[:, 1])
    rp_y_max = np.max(ref_path[:, 1])

    x_min = min(cl_x_min, rp_x_min)
    x_max = max(cl_x_max, rp_x_max)
    y_min = min(cl_y_min, rp_y_min)
    y_max = max(cl_y_max, rp_y_max)

    n_side = 30
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, n_side), np.linspace(y_min, y_max, n_side)
    )
    roof = np.vstack(
        [
            X.flatten(),
            Y.flatten(),
            kitti_zmax * np.ones((n_side, n_side)).flatten(),
        ]
    ).T
    floor = np.vstack(
        [
            X.flatten(),
            Y.flatten(),
            kitti_zmin * np.ones((n_side, n_side)).flatten(),
        ]
    ).T
    occ_cl = np.vstack([occ_cl, roof, floor])

    return occ_cl


def get_ellipse_parameters(P, pp, p0=np.array([0, 0])):
    """Calculates parameters for an ellipse given as (x-p0)^T @ P @ (x-p0) + pp @ (x-p0) - 1 = 0

    Args:
        P (np.array): matrix of the ellipse
        pp (np.array): vector of the ellipse (allows for offsets from center point p)
        p0 (np.array): center point (without offset)
    Returns:
        pc: Actual center of the ellipse
        width: width of the ellipse (2*a from x**2/a**2 + y**2/b**2 = 1)
        height: height of the ellipse (2*b from x**2/a**2 + y**2/b**2 = 1)
        theta: angle of the major axis

    NOTE: Equations taken from wikipedia (ellipse, section"General ellipse",
    https://en.wikipedia.org/wiki/Ellipse). Check "visualization.nb" for theoretical
    analysis of the equations.
    """
    a = P[0, 0]
    b = 2 * P[0, 1]
    c = P[1, 1]
    d = pp[0]
    ee = pp[1]
    f = -1

    aell = -np.sqrt(
        2
        * (a * ee**2 + c * d**2 - b * d * ee + (b**2 - 4 * a * c) * f)
        * ((a + c) + np.sqrt((a - c) ** 2 + b**2))
    ) / (b**2 - 4 * a * c)
    bell = -np.sqrt(
        2
        * (a * ee**2 + c * d**2 - b * d * ee + (b**2 - 4 * a * c) * f)
        * ((a + c) - np.sqrt((a - c) ** 2 + b**2))
    ) / (b**2 - 4 * a * c)

    xc = (2 * c * d - b * ee) / (b**2 - 4 * a * c) + p0[0]
    yc = (2 * a * ee - b * d) / (b**2 - 4 * a * c) + p0[1]

    height = 2 * aell
    width = 2 * bell
    pc = np.array([xc, yc])
    theta = 1 / 2 * np.arctan2(-b, c - a) + np.pi / 2

    return pc, width, height, theta


def get_ellipse_points(width, height, angle, theta):
    """Computes the points in the contour of an ellipse x**2/a + y**2/b = 1
    NOTE: Implementation taken from https://math.stackexchange.com/a/4517941
    Args:
        width (float): 2*a
        height (float): 2*b
        angle (float): Angle by which the x-axis (width or a) is rotated (anticlockwise)
        theta (float): Rotation angle by which the point needs to be located

    Returns:
        pt (np.ndarray): Point in the contour of the ellipse
    """

    def r_ellipse(theta, a, b):
        return a * b / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)

    a = width / 2
    b = height / 2

    R_ellipse = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rot = r_ellipse(theta - angle, a, b) * np.array(
        [np.cos(theta - angle), np.sin(theta - angle)]
    )

    pt = R_ellipse @ rot

    return pt


def get_cage(parametric_path, covers):
    l = 10  # side length of the cage
    h = 4  # heigh of the cage
    n_topbottom = (
        5  # 6  # how many points in the top and bottom of the cage per sweep line
    )
    n_sides = 5
    # int(
    #     h / l * n_topbottom
    # )  # how many points in the sides of the cage per sweep line

    # n_sweep_cage = 100  # how many cages to sweep along reference path
    # xi_wrap = np.linspace(0, 1, n_sweep_cage)
    xi_wrap_init = np.linspace(0, 0.1, 10)
    xi_wrap_mid = np.linspace(0.1, 0.9, 30)
    xi_wrap_end = np.linspace(0.9, 1, 10)
    xi_wrap = np.hstack([xi_wrap_init, xi_wrap_mid, xi_wrap_end])

    occ_cage = []
    for i in range(xi_wrap.shape[0]):
        ind_i = np.argmin(np.abs(parametric_path["xi"] - xi_wrap[i]))
        p_i = parametric_path["p"][ind_i]
        e1_i = parametric_path["erf"][ind_i, :, 0]
        i_horizontal = np.cross(e1_i, np.array([0, 0, 1]))

        horizontal_vecs = np.array([[1, 0, 0], [0, 1, 0]])  # pick the most orthongonal
        horizontal_vec = horizontal_vecs[
            1
        ]  # horizontal_vecs[np.argmin(np.dot(e1_i, horizontal_vecs.T))]
        i_vertical = np.cross(e1_i, horizontal_vec)

        i_horizontal = i_horizontal / np.linalg.norm(i_horizontal)
        i_vertical = i_vertical / np.linalg.norm(i_vertical)

        h_min = -1
        h_max = h_min + h
        # top and bottom
        occ_cage += [
            p_i[:, None]
            + (i_vertical[:, None] * h_min)
            + (i_horizontal[:, None] * np.linspace(-l / 2, l / 2, n_topbottom))
        ]

        occ_cage += [
            p_i[:, None]
            + (i_vertical[:, None] * h_max)
            + (i_horizontal[:, None] * np.linspace(-l / 2, l / 2, n_topbottom))
        ]

        # sides

        occ_cage += [
            p_i[:, None]
            + (i_horizontal[:, None] * l / 2)
            + (i_vertical[:, None] * np.linspace(h_min, h_max, n_sides))
        ]

        occ_cage += [
            p_i[:, None]
            + (i_horizontal[:, None] * -l / 2)
            + (i_vertical[:, None] * np.linspace(h_min, h_max, n_sides))
        ]

        if covers:
            if i == 0 or i == xi_wrap.shape[0] - 1:
                for hh in np.linspace(h_min, h_max, 4):
                    occ_cage += [
                        p_i[:, None]
                        + (i_vertical[:, None] * hh)
                        + (
                            i_horizontal[:, None]
                            * np.linspace(-l / 2, l / 2, n_topbottom)
                        )
                    ]

    occ_cage = np.hstack(occ_cage).T
    print("CORRGEN--> Wrapper points: ", occ_cage.shape[0])

    return occ_cage
