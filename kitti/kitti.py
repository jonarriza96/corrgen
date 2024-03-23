# %%
import numpy as np
import matplotlib.pyplot as plt

import argparse
import time
import pickle

import cvxpy as cp
import pydecomp as pdc

from corrgen.utils import axis_equal

from corrgen.corrgen import (
    polynomial,
    NLP,
    project_cloud_to_parametric_path,
    get_ellipse_parameters,
    get_ellipse_points,
    get_cage,
)

# %matplotlib tk

# ---------------------------------------------------------------------------- #
#                                  User inputs                                 #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_corrgen",
    type=int,
    default=-1,
    help="Polynomial degree for corrgen",
)
parser.add_argument(
    "--n_decomp",
    type=int,
    default=-1,
    help="Number of polygons for decomp",
)
parser.add_argument(
    "--case",
    type=int,
    default=2,
    help="Case study [2 or 3]",
)
parser.add_argument(
    "--lp",
    action="store_true",
    help="Runs LP instead of SDP",
)
parser.add_argument(
    "--no_visualization",
    action="store_true",
    help="Does not show data and animation",
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Save data",
)
args = parser.parse_args()

n_corrgen = args.n_corrgen
n_decomp = args.n_decomp
decomp = False if n_decomp == -1 else True
corrgen = False if n_corrgen == -1 else True
poly_deg = args.n_corrgen
LP = args.lp
visualize = not args.no_visualization
save = args.save

file_name = "case" + str(args.case)
ellipse_axis_max = 20  # SDP and LP ###NOT USED###
ellipse_axis_min = 0.5  # SDP and LP ###NOT USED###
eps = 1e-1  # LP ###NOT USED###

# ---------------------------------------------------------------------------- #
#                               Import case-study                              #
# ---------------------------------------------------------------------------- #

# ------------------------------- Import world ------------------------------- #
print("Importing data...")
path = "/home/jon/corrgen2/kitti/data/" # TODO
with open(path + file_name + ".pkl", "rb") as f:
    data = pickle.load(f)
    world = data["world"]
    occ_clean = data["occ_sensor_clean"]
occ_cl = world["occ_cl"]
path = world["linear_path"]
ppr = world["ppr"]


# ---------------------------- Process point cloud --------------------------- #

occ_cl = occ_clean.copy()

# remove ground
ind = occ_cl[:, 2] > -1.5
occ_cl = occ_cl[ind]

# add cage
occ_cage = get_cage(ppr=ppr, covers=not corrgen)
occ_cl_no_cage = occ_cl.copy()
occ_cl = np.vstack([occ_cl, occ_cage])
# occ_cl = occ_cage.copy()

# project to the path and prune unnecessary points
occ_erf, min_d_tr, max_d_tr, ind_proj = project_cloud_to_parametric_path(
    pcl=occ_cl, parametric_path=ppr.parametric_path, safety_check=False, prune=True
)
occ_cl = occ_cl[ind_proj]

# remove points outside the maximum ellipse size
ind_in = np.linalg.norm(occ_erf[:, 1:], axis=1) <= ellipse_axis_max / 1.5
occ_erf = occ_erf[ind_in]
occ_cl = occ_cl[ind_in]


# %%
# ---------------------------------------------------------------------------- #
#                             Convex decomposition                             #
# ---------------------------------------------------------------------------- #
if decomp:
    box = np.array([[10, 10, 10]])
    occ_cl_decomp = occ_cl.copy()  # add_world_boundaries(occ_cl, planar=False)
    ind_dc = np.linspace(0, ppr.parametric_path["p"].shape[0] - 1, n_decomp, dtype=int)
    path_decomp = ppr.parametric_path["p"][ind_dc]
    # path_decomp[:, 2] = 0
    t0_dc = time.time()
    A_hs, b_hs = pdc.convex_decomposition_3D(occ_cl_decomp, path_decomp, box)
    t1_dc = time.time()
    print("DECOMP --> Time:", (t1_dc - t0_dc) * 1000, "ms")


# ---------------------------------------------------------------------------- #
#                 Differentiable Parametric Corridor Generator                 #
# ---------------------------------------------------------------------------- #
if corrgen:
    # ----------------------------- compute corridor ----------------------------- #
    prob, variables = NLP(
        poly_deg=poly_deg,
        occ=occ_erf,
        ellipse_axis_lims=np.array([ellipse_axis_max, ellipse_axis_min]),
        LP=LP,
        eps=eps,
    )
    a, b, c, d, e = variables

    if not LP:
        prob.solve(solver=cp.CLARABEL, verbose=True)
    else:
        # prob.solve(solver=cp.GUROBI, verbose=True)
        # prob.solve(solver=cp.OSQP, verbose=True)
        prob.solve(solver=cp.PROXQP, backend="dense", verbose=True)
    coeffs = {
        "a": a.value,
        "b": b.value,
        "c": c.value,
        "d": d.value,
        "e": e.value,
    }

    print("CORRGEN --> OCP status:", prob.status)
    print("CORRGEN --> OCP solver time:", 1000 * prob._solve_time, "ms")

    # ----------------------------- evaluate corridor ---------------------------- #
    n_angles = 18

    n_eval = ppr.parametric_path["p"].shape[0]  # 100
    xi_eval = np.linspace(0, 1, n_eval)
    P_eval = np.zeros((n_eval, 2, 2))
    pp_eval = np.zeros((n_eval, 2))
    ellipse_params = np.zeros((n_eval, 5))
    ellipse_pts = np.zeros((n_eval, n_angles, 2))
    angles = np.linspace(0, 2 * np.pi, n_angles)  # [:-1]
    ellipse_pts_world = np.zeros((n_eval, n_angles, 3))
    eigs = np.zeros((n_eval, 2))
    for i in range(n_eval):
        a_i, b_i, c_i, d_i, e_i = polynomial(
            xi=xi_eval[i], degree=poly_deg, coeffs=coeffs
        )
        P_eval[i] = np.array([[a_i, c_i], [c_i, b_i]])
        pp_eval[i] = np.array([d_i, e_i])
        eigs[i] = np.linalg.eigvals(P_eval[i])

        pc, width, height, angle = get_ellipse_parameters(P=P_eval[i], pp=pp_eval[i])
        ellipse_params[i] = np.array([width, height, angle, pc[0], pc[1]])

        for j in range(n_angles):
            ellipse_pts[i, j, :] = get_ellipse_points(
                width=ellipse_params[i, 0],
                height=ellipse_params[i, 1],
                angle=ellipse_params[i, 2],
                theta=angles[j],  # + angle,
            )

            ind = np.argmin(np.abs(ppr.parametric_path["xi"] - xi_eval[i]))
            gamma = ppr.parametric_path["p"][ind]
            erf = ppr.parametric_path["erf"][ind]

            w = ellipse_pts[i, j, :] + ellipse_params[i, -2:]
            ellipse_pts_world[i, j, :] = gamma + w[0] * erf[:, 1] + w[1] * erf[:, 2]

    area = np.pi * ellipse_params[:, 0] / 2 * ellipse_params[:, 1] / 2
    parametric_volume = np.trapz(area, xi_eval)
    print(f"CORRGEN --> Parametric volume: {parametric_volume}")


# ---------------------------------------------------------------------------- #
#                                 Visualization                                #
# ---------------------------------------------------------------------------- #
if visualize:

    if corrgen:
        # ---------------------------- ellipse parameters ---------------------------- #
        eig_min = 4 / (ellipse_axis_max**2)
        eig_max = 4 / (ellipse_axis_min**2)
        plt.figure()
        plt.subplot(411)
        plt.plot(xi_eval, eigs, ".")
        plt.plot([0, 1], [eig_min, eig_min], "k--")
        # plt.plot([0, 1], [2*eig_max, 2*eig_max], "k--")
        plt.ylabel("Eigenvalues")
        plt.subplot(412)
        plt.plot([0, 1], [ellipse_axis_max, ellipse_axis_max], "k--")
        plt.plot([0, 1], [ellipse_axis_min, ellipse_axis_min], "k--")
        plt.plot(xi_eval, ellipse_params[:, :2])
        plt.plot(xi_eval, 2 / np.sqrt(eigs))
        plt.ylabel("Width/Height")
        plt.subplot(413)
        plt.plot(xi_eval, ellipse_params[:, -2:])
        plt.ylabel("Offset")
        plt.subplot(414)
        plt.plot(xi_eval, ellipse_params[:, 2])
        plt.ylabel("Angle")

        ind_view = np.random.randint(0, n_eval)
        width = ellipse_params[ind_view, 0]
        height = ellipse_params[ind_view, 1]
        angle = ellipse_params[ind_view, 2]
        axis1 = width / 2 * np.array([np.cos(angle), np.sin(angle)])
        axis2 = (
            height
            / 2
            * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
        )

        # ----------------------- components of ellipse matrix ----------------------- #
        a_mat = P_eval[:, 0, 0]
        b_mat = P_eval[:, 1, 1]
        c_mat = P_eval[:, 0, 1]
        d_mat = pp_eval[:, 0]
        e_mat = pp_eval[:, 1]

        plt.figure()
        plt.plot(xi_eval, a_mat, label="a")
        plt.plot(xi_eval, b_mat, label="b")
        plt.plot(xi_eval, c_mat, label="c")
        plt.plot(xi_eval, d_mat, label="d")
        plt.plot(xi_eval, e_mat, label="e")

        plt.legend(title="P=[[a,c],[c,b]],pp=[d,e]")
        plt.suptitle("Components of ellipse matrix")

    # ---------------------------------- 3D view --------------------------------- #
    if decomp:
        ax = pdc.visualize_environment(Al=A_hs, bl=b_hs, p=path, planar=False)
    else:
        ax = plt.figure().add_subplot(111, projection="3d")

    occ_v = occ_cl.copy()
    ind = occ_v[:, 2] > -1005  # -1.5
    ax.scatter(
        occ_v[ind, 0],
        occ_v[ind, 1],
        occ_v[ind, 2],
        marker=".",
        c=-(occ_v[ind, 0] ** 2 + occ_v[ind, 1] ** 2),
        cmap="turbo",
    )

    ax.plot(path[:, 0], path[:, 1], path[:, 2], "k-o")

    ax.plot(
        ppr.parametric_path["p"][:, 0],
        ppr.parametric_path["p"][:, 1],
        ppr.parametric_path["p"][:, 2],
        "-b",
    )
    # ax.plot(path[:, 0], path[:, 1], path[:, 2], "-ok")
    if corrgen:
        for j in range(n_angles):
            ax.plot(
                ellipse_pts_world[:, j, 0],
                ellipse_pts_world[:, j, 1],
                ellipse_pts_world[:, j, 2],
                "k-",
                alpha=0.5,
            )
        for i in range(0, n_eval, 10):
            ax.plot(
                ellipse_pts_world[i, :, 0],
                ellipse_pts_world[i, :, 1],
                ellipse_pts_world[i, :, 2],
                "r-",
                alpha=0.5,
            )

    pts_map = np.vstack([occ_cl.copy()])
    axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())

    plt.show()

# ---------------------------------------------------------------------------- #
#                                     Save                                     #
# ---------------------------------------------------------------------------- #

if save:
    data_path = "/home/jonarriza96/corrgen_v2/kitti/data/"

    print("Saving data...")
    if corrgen:
        corrgen_path = data_path + file_name + "/corrgen/time_volumes/" + str(poly_deg)
        if LP:
            corrgen_path += "_LP"
        else:
            corrgen_path += "_SDP"
        with open(corrgen_path + ".pkl", "wb") as f:
            pickle.dump(
                {
                    "occ_cl": occ_cl,
                    "occ_cl_no_cage": occ_cl_no_cage,
                    "occ_cage": occ_cage,
                    "occ_clean": occ_clean,
                    "occ_erf": occ_erf,
                    "path": path,
                    "ppr": ppr,
                    "coeffs": coeffs,
                    "ellipse_params": ellipse_params,
                    "ellipse_pts": ellipse_pts,
                    "ellipse_pts_world": ellipse_pts_world,
                    "corridor_volume": parametric_volume,
                    "solve_time": prob._solve_time,
                },
                f,
            )
    if decomp:
        decomp_path = data_path + file_name + "/decomp/" + str(n_decomp) + ".pkl"
        with open(decomp_path, "wb") as f:
            pickle.dump(
                {
                    "occ_cl": occ_cl,
                    "occ_cl_no_cage": occ_cl_no_cage,
                    "occ_cage": occ_cage,
                    "occ_clean": occ_clean,
                    "occ_erf": occ_erf,
                    "path": path,
                    "ppr": ppr,
                    "A_hs": A_hs,
                    "b_hs": b_hs,
                },
                f,
            )
    print("Done.")
