# %%
import numpy as np
import casadi as cs
import cvxpy as cp
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


import pyny3d.geoms as pyny
import cdd


def symbolic_decomp_erf(max_points, max_hyperplanes):
    """Generates a casadi function that computes the biggest convex polygon around
    the centroid of a planar occupancy map.

    Args:
        max_points : Maximum number of points in the occupancy map.
        max_hyperplanes : Maximum number of hyperplanes in the polygon.

    Returns:
        f: Casadi function that takes an occupancy map and returns
        the contained biggest convex polygon
    """
    # symbolic variables
    p_cs = cs.MX.zeros(2)  # cs.MX.sym("p", 2) --> in erf frame is always 0
    occ_points_cs = cs.MX.sym("occ", max_points, 2)  # [x, y]
    A_cs = cs.MX.zeros(max_hyperplanes, 2)
    b_cs = cs.MX.zeros(max_hyperplanes)

    # decomposition algorithm
    occ_remaining_cs = occ_points_cs
    n_hp = 0
    finished = 1
    for k in range(max_hyperplanes):
        # find closest point
        dists_sq = cs.sum1((p_cs - occ_remaining_cs.T) ** 2).T
        ind_closest = cs.find(dists_sq == cs.mmin(dists_sq))
        closest_pt_cs = occ_remaining_cs[ind_closest, :]

        # find hyperplane
        C = cs.DM.eye(2) * cs.sqrt(dists_sq[ind_closest])
        d = p_cs
        n = (cs.inv(C) * cs.inv(C).T) @ (closest_pt_cs.T - d)
        n = n / cs.norm_2(n)

        c = cs.dot(closest_pt_cs, n.T)
        cond = (cs.dot(n, d) - c.T) > 0
        n = cs.if_else(cond, -n, n)
        c = cs.if_else(cond, -c, c)

        Ak = n.T
        bk = c

        # remove points outside hyperplane
        ind_out = (n.T @ (occ_remaining_cs.T - closest_pt_cs.T)) < 0
        occ_remaining_cs = cs.if_else(
            ind_out.T, occ_remaining_cs, 1e10 * cs.MX.ones(max_points, 2)
        )

        # add counter if hyperplane active
        all_occ_out = cs.sum1(ind_out.T) == 0
        deactivate_hp = cs.logic_and(all_occ_out, finished)
        n_hp = cs.if_else(deactivate_hp, n_hp, n_hp + 1)
        A_cs[k, :] = cs.if_else(deactivate_hp, cs.MX.zeros(1, 2), Ak)
        b_cs[k, :] = cs.if_else(deactivate_hp, cs.MX.zeros(1, 1), bk)
        finished = cs.if_else(all_occ_out, 1, 0)

    # remove non-active hyperplanes
    n_hp = n_hp + 1

    # create a function
    f = cs.Function(
        "f_decomp", [occ_points_cs], [A_cs, b_cs, n_hp], ["occ"], ["A", "b", "n_hp"]
    )
    return f


def hyperplane_to_vertices(A, b):
    mat = cdd.Matrix(np.hstack([b[:, np.newaxis], -A]))
    mat.rep_type = cdd.RepType.INEQUALITY

    poly = cdd.Polyhedron(mat)
    gen = poly.get_generators()

    vert = np.array(list(gen))[:, 1:]

    return vert


def visualize_hyperplane(A, b):
    vert = hyperplane_to_vertices(A, b)
    polygon = pyny.Polygon(vert)
    ax = pyny.Surface(polygon).plot2d(c_poly="0000FF10", alpha=0.15, ret=True)
    return ax


def ellipse_LP(pcl, p, offset, circle):
    # define variables
    X = cp.Variable((2, 2), symmetric=True)
    if offset:
        d = cp.Variable(2)

    # define constraints
    constraints = []
    for i in range(pcl.shape[0]):
        if offset:
            constraints += [
                (pcl[i] - p).T @ X @ (pcl[i] - p) + d.T @ (pcl[i] - p) - 1 >= 0
            ]
        else:
            constraints += [(pcl[i] - p).T @ X @ (pcl[i] - p) - 1 >= 0]
    if circle:
        constraints += [X[0, 0] == X[1, 1]]
        constraints += [X[0, 1] == 0]
        constraints += [X[1, 0] == 0]

    # define cost function
    prob = cp.Problem(cp.Minimize(X[0, 0] + X[1, 1]), constraints)

    # solve
    t1 = time.time()
    prob.solve(solver=cp.CLARABEL, verbose=True)
    t2 = time.time()
    # print("SDP time: ", 1000 * (t2 - t1), " ms")

    if offset:
        return X.value, d.value, prob
    else:
        return X.value, prob


def get_ellipse_parameters(P, pp, p0):
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
    pc = np.array([xc, yc])

    height = 2 * aell
    width = 2 * bell
    theta = 1 / 2 * np.arctan2(-b, c - a) + np.pi / 2

    return pc, width, height, theta


# ---------------------------------- Ellipse --------------------------------- #
# generate pointcloud and centre point
p = np.array([0, 0])
pcl = np.random.rand(300, 2) - np.array([0.5, 0.5])

# ellipse
e = []
w = []
h = []
a = []
cent = []
Ps = []
for color, offset, circle in zip(
    ["r", "g", "b"], [False, False, True], [True, False, False]
):
    if offset:
        P, pp, prob = ellipse_LP(pcl=pcl, p=p, offset=offset, circle=circle)
    else:
        P, prob = ellipse_LP(pcl=pcl, p=p, offset=offset, circle=circle)
        pp = np.array([0, 0])
    pc, width, height, angle = get_ellipse_parameters(P=P, pp=pp, p0=p)

    w += [width]
    h += [height]
    a += [angle]
    cent += [pc]
    Ps += [P]
    # add ellipse to plot
    e += [
        Ellipse(
            xy=(pc[0], pc[1]),
            width=width,
            height=height,
            angle=np.rad2deg(angle),
            facecolor=color,
            alpha=0.2,
            # edgecolor=color,
            # linewidth=5,
        )
    ]
# ---------------------------------- Polygon --------------------------------- #
# larges linear convex set
f_decomp = symbolic_decomp_erf(max_points=pcl.shape[0], max_hyperplanes=10)
A, b, n_hp = f_decomp(pcl)
A = np.squeeze(A)
b = np.squeeze(b)


# --------------------------------- Visualize -------------------------------- #
# hyperplane
ax = visualize_hyperplane(A, b)

# others
ax.scatter(pcl[:, 0], pcl[:, 1], color="k", marker=".")
ax.add_patch(e[2])
ax.add_patch(e[1])
ax.add_patch(e[0])
ax.set_xlim(-0.35, 0.35)
ax.set_ylim(-0.35, 0.35)
ax.scatter(pc[0], pc[1], color="b", edgecolor="k")
ax.plot([0, pc[0]], [0, pc[1]], "--b", alpha=0.7, linewidth=0.6)
ax.scatter(p[0], p[1], color="m", edgecolor="k")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis("off")
plt.show()

# %%
# --------------------------------- Save data -------------------------------- #
# import pickle
# from corrgen.utils import get_corrgen_path

# path = "/home/jonarriza96/corrgen_v2/data/cross_section6.pkl"
# with open(path, "wb") as f:
#     data = {
#         "w": w,
#         "h": h,
#         "a": a,
#         "cent": cent,
#         "A": A,
#         "b": b,
#         "pcl": pcl,
#         "p": p,
#         "pc": pc,
#     }
#     pickle.dump(data, f)
