# %%
import numpy as np
import matplotlib.pyplot as plt

import time
import pickle

import cvxpy as cp
import pydecomp as pdc

from papor.utils.visualize import axis_equal, plot_frames

from utils import (
    polynomial,
    SDP_3D,
    project_cloud_to_parametric_path,
    add_roof_floor,
    add_world_boundaries,
    get_ellipse_parameters,
    get_ellipse_points,
)

# %matplotlib tk

# ---------------------------------------------------------------------------- #
#                               Import case-study                              #
# ---------------------------------------------------------------------------- #

# ------------------------------- Import world ------------------------------- #
print("Importing data...")
path = "/home/jonarriza96/corrgen/examples/experiments/data/kitti/"
file_name = "case2.pkl"
with open(path + file_name, "rb") as f:
    data = pickle.load(f)
    world = data["world"]
    occ_clean = data["occ_sensor_clean"]
    # corridor = data["corridor"]
    # corridor_eval = data["evaluation"]
occ_cl = world["occ_cl"]
path = world["linear_path"]
ppr = world["ppr"]


# ---------------------------- Process point cloud --------------------------- #
ellipse_axis_max = 10
ellipse_axis_min = 0.5

occ_cl = occ_clean.copy()

# project to the path and prune unnecessary points
occ_erf, min_d_tr, max_d_tr, ind_proj = project_cloud_to_parametric_path(
    pcl=occ_cl, parametric_path=ppr.parametric_path, safety_check=False, prune=True
)
occ_cl = occ_cl[ind_proj]

# add roof and floor
occ_cl = add_roof_floor(occ_cl=occ_cl, kitti_zmax=1.25)

# # project pruned points with roof and floor
occ_erf, min_d_tr, max_d_tr, _ = project_cloud_to_parametric_path(
    pcl=occ_cl,
    parametric_path=ppr.parametric_path,
    safety_check=False,
    prune=False,
)

# wrapper around path
# n_angles_wrap = 20  # 10
# r_wrap = ellipse_axis_max / 2
# angles = np.linspace(0, 2 * np.pi, n_angles_wrap)
# w_wrap = np.vstack([r_wrap * np.cos(angles), r_wrap * np.sin(angles)]).T
# xi_wrap = np.linspace(0, 1, 200)
# occ_wrap = np.zeros((xi_wrap.shape[0] * n_angles_wrap, 3))
# for i in range(xi_wrap.shape[0]):
#     occ_wrap[i * n_angles_wrap : (i + 1) * n_angles_wrap, 0] = xi_wrap[i]
#     occ_wrap[i * n_angles_wrap : (i + 1) * n_angles_wrap, 1:] = w_wrap

# occ_erf = np.vstack([occ_erf, occ_wrap])

# # remove points outside the maximum ellipse size
# # if not kitti:
# ind_in = np.linalg.norm(occ_erf[:, 1:], axis=1) <= ellipse_axis_max / 2
# occ_erf = occ_erf[ind_in]


###########
# %%
# ---------------------------------------------------------------------------- #
#                             Convex decomposition                             #
# ---------------------------------------------------------------------------- #
box = np.array([[10, 10, 10]])
occ_cl_decomp = add_world_boundaries(occ_cl, planar=False)
A_hs, b_hs = pdc.convex_decomposition_3D(occ_cl_decomp, path, box)

# ---------------------------------------------------------------------------- #
#                 Differentiable Parametric Corridor Generator                 #
# ---------------------------------------------------------------------------- #
poly_deg = 12

# ----------------------------- compute corridor ----------------------------- #
prob, variables = SDP_3D(
    poly_deg=poly_deg,
    occ=occ_erf,
    ellipse_axis_lims=np.array([ellipse_axis_max, ellipse_axis_min]),
)
a, b, c, d, e = variables

prob.solve(solver=cp.CLARABEL, verbose=True)
coeffs = {
    "a": a.value,
    "b": b.value,
    "c": c.value,
    "d": d.value,
    "e": e.value,
}

print("CORRGEN --> OCP status:", prob.status)
print("CORRGEN --> OCP solver time:", 1000 * prob.solver_stats.solve_time, "ms")

# %%
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

        gamma = ppr.parametric_path["p"][i]
        erf = ppr.parametric_path["erf"][i]

        w = ellipse_pts[i, j, :] + ellipse_params[i, -2:]
        ellipse_pts_world[i, j, :] = gamma + w[0] * erf[:, 1] + w[1] * erf[:, 2]

area = np.pi * ellipse_params[:, 0] / 2 * ellipse_params[:, 1] / 2
parametric_volume = np.trapz(area, xi_eval)
print(f"Parametric volume: {parametric_volume}")


# ---------------------------------------------------------------------------- #
#                                 Visualization                                #
# ---------------------------------------------------------------------------- #


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
axis2 = height / 2 * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])

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
ax = pdc.visualize_environment(Al=A_hs, bl=b_hs, p=path, planar=False)
ind = occ_cl[:, 2] > -15  # -1.5
ax.scatter(
    occ_cl[ind, 0],
    occ_cl[ind, 1],
    occ_cl[ind, 2],
    marker=".",
    c=-(occ_cl[ind, 0] ** 2 + occ_cl[ind, 1] ** 2),
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

plt.show()


# ax = plt.figure().add_subplot(111, projection="3d")
# ax.scatter(
#     occ_cl[:, 0],
#     occ_cl[:, 1],
#     occ_cl[:, 2],
#     c=-(occ_cl[:, 0] ** 2 + occ_cl[:, 1] ** 2),
#     marker=".",
#     cmap="turbo",
# )
# pts_map = world["occ_cl"].copy()
# axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())
