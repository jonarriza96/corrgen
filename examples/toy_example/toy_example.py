import numpy as np
import matplotlib.pyplot as plt

import pydecomp as pdc
import cvxpy as cp

import argparse
import pickle

from corrgen.corrgen import (
    polynomial,
    NLP,
    project_cloud_to_parametric_path,
    get_ellipse_parameters,
    get_ellipse_points,
    get_cage,
)

from corrgen.utils import axis_equal


parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_corrgen",
    type=int,
    default=-1,
    help="Polynomial degree for corrgen",
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

poly_deg = args.n_corrgen
LP = args.lp
visualize = not args.no_visualization
save = args.save

polynomial_order = 3  # Order of the parametric reference polynomial
path = np.array([[0, -3, 0], [0, 1, 0], [0, 10, 0]])

ellipse_axis_max = 20  # SDP and LP ###NOT USED###
ellipse_axis_min = 0.5  # SDP and LP ###NOT USED###
eps = 1e-1  # LP ###NOT USED###


# ---------------------------------------------------------------------------- #
#                                 Import world                                 #
# ---------------------------------------------------------------------------- #
from corrgen.utils import get_corrgen_path

print("Importing data...")
with open(get_corrgen_path() + "/examples/toy_example/data/toy_example.pkl", "rb") as f:
    data = pickle.load(f)
    parametric_path = data["parametric_path"]
    cloud = data["cloud"]

# ---------------------------------------------------------------------------- #
#                              Corridor generation                             #
# ---------------------------------------------------------------------------- #

# ---------------------------- Process point cloud --------------------------- #
occ_cage = get_cage(parametric_path=parametric_path, case="toy_example", covers=False)
cloud_no_cage = cloud.copy()
cloud = np.vstack([cloud, occ_cage])

cloud_erf, min_d_tr, max_d_tr, ind_proj = project_cloud_to_parametric_path(
    pcl=cloud, parametric_path=parametric_path, safety_check=False, prune=False
)
cloud = cloud[ind_proj]

# ----------------------------- Compute corridor ----------------------------- #
prob, variables = NLP(
    poly_deg=poly_deg,
    occ=cloud_erf,
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

# ----------------------------- Evaluate corridor ---------------------------- #
n_angles = 18

n_eval = parametric_path["p"].shape[0]  # 100
xi_eval = np.linspace(0, 1, n_eval)
P_eval = np.zeros((n_eval, 2, 2))
pp_eval = np.zeros((n_eval, 2))
ellipse_params = np.zeros((n_eval, 5))
ellipse_pts = np.zeros((n_eval, n_angles, 2))
angles = np.linspace(0, 2 * np.pi, n_angles)  # [:-1]
ellipse_pts_world = np.zeros((n_eval, n_angles, 3))
eigs = np.zeros((n_eval, 2))
for i in range(n_eval):
    a_i, b_i, c_i, d_i, e_i = polynomial(xi=xi_eval[i], degree=poly_deg, coeffs=coeffs)
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

        ind = np.argmin(np.abs(parametric_path["xi"] - xi_eval[i]))
        gamma = parametric_path["p"][ind]
        erf = parametric_path["erf"][ind]

        w = ellipse_pts[i, j, :] + ellipse_params[i, -2:]
        ellipse_pts_world[i, j, :] = gamma + w[0] * erf[:, 1] + w[1] * erf[:, 2]

area = np.pi * ellipse_params[:, 0] / 2 * ellipse_params[:, 1] / 2
parametric_volume = np.trapz(area, xi_eval)
print(f"CORRGEN --> Parametric volume: {parametric_volume}")

# ---------------------------------------------------------------------------- #
#                                   Visualize                                  #
# ---------------------------------------------------------------------------- #

if visualize:
    # ---------------------------- Ellipse parameters ---------------------------- #
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
        height / 2 * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
    )

    # ----------------------- Components of ellipse matrix ----------------------- #
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
    # if decomp:
    # ax = pdc.visualize_environment(Al=A_hs, bl=b_hs, p=path, planar=False)
    # else:
    ax = plt.figure().add_subplot(111, projection="3d")

    occ_v = cloud.copy()
    ind = occ_v[:, 2] > -1005  # -1.5
    ax.scatter(
        occ_v[ind, 0],
        occ_v[ind, 1],
        occ_v[ind, 2],
        marker=".",
        c=occ_v[ind, 2],  # -(occ_v[ind, 0] ** 2 + occ_v[ind, 1] ** 2),
        cmap="turbo",
    )

    ax.plot(path[:, 0], path[:, 1], path[:, 2], "k-o")

    ax.plot(
        parametric_path["p"][:, 0],
        parametric_path["p"][:, 1],
        parametric_path["p"][:, 2],
        "-b",
    )
    # ax.plot(path[:, 0], path[:, 1], path[:, 2], "-ok")
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

    pts_map = np.vstack([cloud.copy()])
    axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())

    plt.show()


# --------------------------------- Save data -------------------------------- #
if save:
    path = "/home/jonarriza96/corrgen_v2/toy_example/figures/data/3d/"
    file_name = str(poly_deg)
    if LP:
        file_name += "_LP"
    else:
        file_name += "_SDP"
    file_path = path + file_name + ".pkl"
    with open(file_path, "wb") as f:
        pickle.dump(
            {
                "occ_cl": cloud,
                "occ_cl_no_cage": cloud_no_cage,
                "occ_cage": occ_cage,
                "occ_erf": cloud_erf,
                "path": path,
                "parametric_path": parametric_path,
                "coeffs": coeffs,
                "ellipse_params": ellipse_params,
                "ellipse_pts": ellipse_pts,
                "ellipse_pts_world": ellipse_pts_world,
                "corridor_volume": parametric_volume,
                "solve_time": prob._solve_time,
            },
            f,
        )
