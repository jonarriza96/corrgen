# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cvxpy as cp


import pickle
import time

from corrgen.utils import get_corrgen_path

VERBOSE = False


def LP_2D_separate(d, occ, d_max, d_min):
    m = occ.shape[0]  # number of points == constraints
    n = d + 1  # number of variables

    # decision variables
    x = cp.Variable(n)

    # cost function
    cost = 0
    for k in range(n):
        cost += x[k] / (k + 1)

    # constraints
    constraints = []

    # 1- sweep path
    n_sweep = 100
    xi_sweep = np.linspace(0, 1, n_sweep)
    for i in range(n_sweep):
        xi_i = xi_sweep[i]
        x_i = 0
        for k in range(n):
            x_i += x[k] * (xi_i**k)
        constraints += [x_i <= d_max]
        constraints += [x_i >= d_min]

    # # 2- loop all occupancy points
    for i in range(m):
        xi_i = occ[i, 0]
        x_i = 0
        for k in range(n):
            x_i += x[k] * (xi_i**k)
        constraints += [x_i <= occ[i, 1]]

    # linear program
    prob = cp.Problem(cp.Minimize(-cost), constraints)

    return prob, x


def generate_2d_corridor(world, ocp_params):
    ppr = world["ppr"]
    occ_cl = world["occ_cl"]
    poly_deg = ocp_params["poly_deg"]

    # -------------------------------- projection -------------------------------- #
    occ_erf, min_d_tr, max_d_tr = project_cloud_to_parametric_path(
        pcl=occ_cl,
        parametric_path=ppr,  # .parametric_path.copy(),
        safety_check=VERBOSE,
    )

    # ------------------------------- process cloud ------------------------------ #

    occ_p = occ_erf[occ_erf[:, 1] >= 0]
    occ_n = occ_erf[occ_erf[:, 1] < 0]
    occ_n[:, 1] = -occ_n[:, 1]

    # --------------------------------- solve OCP -------------------------------- #

    dp_max = np.max(occ_p[:, 1])
    dp_min = np.min(occ_p[:, 1])
    dn_max = np.max(occ_n[:, 1])
    dn_min = np.min(occ_n[:, 1])

    prob_p, x_p = LP_2D_separate(
        d=poly_deg,
        occ=occ_p,
        d_max=dp_max,
        d_min=dp_min,
    )  # positive side of normal
    prob_n, x_n = LP_2D_separate(
        d=poly_deg,
        occ=occ_n,
        d_max=dn_max,
        d_min=dn_min,
    )  # negative side of normal

    # solve
    prob_p.solve(solver=cp.CLARABEL, verbose=VERBOSE)
    prob_n.solve(solver=cp.CLARABEL, verbose=VERBOSE)

    print("CORRGEN --> OCP status [p, n]:", prob_p.status, prob_n.status)
    print(
        "CORRGEN --> OCP time [p, n]: ",
        1000 * (prob_p.solver_stats.solve_time),
        1000 * (prob_n.solver_stats.solve_time),
        " ms",
    )

    # ---------------------------------- output ---------------------------------- #
    # if separate_ocps:
    corridor = {
        "x_p": x_p,
        "x_n": x_n,
        "occ_p": occ_p,
        "occ_n": occ_n,
    }

    return corridor


def project_cloud_to_parametric_path(
    pcl, parametric_path, safety_check=False, kitti=False, prune=True
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

    # get trasnverse distances
    d_tr = np.linalg.norm(occ_erf[:, 1:], axis=1)
    min_d_tr = np.min(d_tr)
    max_d_tr = np.max(d_tr)
    print("CORRGEN --> Transverse distances [min, max]:", min_d_tr, max_d_tr)

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
    if kitti:
        return occ_erf, min_d_tr, max_d_tr, ind_proj
    return occ_erf, min_d_tr, max_d_tr


def visualize_2d(world, corridor, ocp_params):
    # linear_path = world["linear_path"]
    # A = world["A_hs"]
    # b = world["b_hs"]
    ppr = world["ppr"]
    # occ_cl = world["occ_cl"]
    occ_cl_nbnd = world["occ_cl"]

    poly_deg = ocp_params["poly_deg"]

    # separate_ocps = ocp_params["separate_ocps"]
    # if separate_ocps:
    x_p = corridor["x_p"]
    x_n = corridor["x_n"]
    # else:
    #     x_cross = corridor["x_cross"]
    #     x_center = corridor["x_center"]

    occ_p = corridor["occ_p"]
    occ_n = corridor["occ_n"]

    # evaluate
    n = poly_deg + 1

    n_eval = ppr["p"].shape[0]  # 100
    xi_eval = np.linspace(0, 1, n_eval)
    corridor = np.zeros((n_eval, 2, 2))  # [n_eval, {bnd_p, bnd_n}, {x,y}]
    dists = np.zeros((n_eval, 2))
    cross = np.zeros((n_eval))
    center = np.zeros((n_eval))
    for i in range(n_eval):

        # if separate_ocps:
        for k in range(n):
            dists[i, 0] += x_p.value[k] * (xi_eval[i] ** k)
            dists[i, 1] += x_n.value[k] * (xi_eval[i] ** k)
        # else:
        #     for k in range(n):
        #         cross[i] += x_cross.value[k] * (xi_eval[i] ** k)
        #         center[i] += x_center.value[k] * (xi_eval[i] ** k)

        #     dists[i, 0] = cross[i] / 2 + center[i]
        #     dists[i, 1] = cross[i] / 2 - center[i]

        gamma = ppr["p"][i, :2]
        w1 = ppr["erf"][i, 1, :2]

        corridor[i, 0, :] = gamma + w1 * dists[i, 0]
        corridor[i, 1, :] = gamma - w1 * dists[i, 1]

    # distances
    plt.figure()
    plt.subplot(211)
    plt.scatter(occ_p[:, 0], occ_p[:, 1], color="r")
    plt.plot(
        [xi_eval[0], xi_eval[-1]], [np.max(occ_p[:, 1]), np.max(occ_p[:, 1])], "k--"
    )
    plt.plot(
        [xi_eval[0], xi_eval[-1]], [np.min(occ_p[:, 1]), np.min(occ_p[:, 1])], "k--"
    )
    plt.plot(xi_eval, dists[:, 0], "b-")
    plt.ylabel("d_pos")
    plt.subplot(212)
    plt.scatter(occ_n[:, 0], occ_n[:, 1], color="r")
    plt.plot(
        [xi_eval[0], xi_eval[-1]], [np.max(occ_n[:, 1]), np.max(occ_n[:, 1])], "k--"
    )
    plt.plot(
        [xi_eval[0], xi_eval[-1]], [np.min(occ_n[:, 1]), np.min(occ_n[:, 1])], "k--"
    )
    plt.plot(xi_eval, dists[:, 1], "b-")
    plt.ylabel("d_neg")
    plt.xlabel(r"$\xi$")

    # 2D view
    ax = plt.figure().add_subplot(111)
    # ax = mpl.visualize_environment(Al=A, bl=b, p=path, planar=True, ax=ax)
    ax.scatter(occ_cl_nbnd[:, 0], occ_cl_nbnd[:, 1], color="k", marker=".")
    ax.plot(
        ppr["p"][:, 0],
        ppr["p"][:, 1],
        "b-",
    )
    # ax = plot_frames(
    #     r=ppr.parametric_path["p"],
    #     e1=ppr.parametric_path["erf"][:, :, 0],
    #     e2=ppr.parametric_path["erf"][:, :, 1],
    #     e3=np.tile(np.array([[0, 0, 1]]), (ppr.n_eval * ppr.n_segments, 1)),
    #     interval=1,  # 0.99,
    #     scale=0.25,
    #     ax=ax,
    #     planar=True,
    # )
    ax.plot(corridor[:, 0, 0], corridor[:, 0, 1], "r-")
    ax.plot(corridor[:, 1, 0], corridor[:, 1, 1], "r-")

    plt.show()


ocp_params = {
    "poly_deg": 12,  # degree of polynomial for corridor bound
}

# load point cloud
with open(get_corrgen_path() + "/examples/data/corridor_2d.pkl", "rb") as f:
    world = pickle.load(f)
    # world = data["world"]

# generate corridor
corridor = generate_2d_corridor(world=world, ocp_params=ocp_params)

# visualize
visualize_2d(world=world, corridor=corridor, ocp_params=ocp_params)
