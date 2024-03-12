import numpy as np
import matplotlib.pyplot as plt

import pydecomp as pdc
import pymoplit as mpl
import cvxpy as cp

from papor.papor import PAPOR
from papor.utils.visualize import axis_equal

from corrgen2.corrgen import (
    polynomial,
    NLP,
    project_cloud_to_parametric_path,
    add_roof_floor,
    add_world_boundaries,
    get_ellipse_parameters,
    get_ellipse_points,
    # get_cage,
)

from corrgen2.utils import convert_curve_to_casadi_func


def ellipse_3d(center, size, angle=0):
    # Generate data points for the ellipse
    u = np.linspace(0, 2 * np.pi, 100)
    # x = center[0] + size[0] * np.cos(u)
    # y = center[1] + size[1] * np.sin(u)
    x = (
        center[0]
        + size[0] * np.cos(u) * np.cos(angle)
        - size[1] * np.sin(u) * np.sin(angle)
    )
    y = (
        center[1]
        + size[0] * np.cos(u) * np.sin(angle)
        + size[1] * np.sin(u) * np.cos(angle)
    )
    z = center[2] + np.zeros_like(u)

    # if angle == 0:
    #     rotation_matrix = np.eye(2)
    # else:
    # rotation_matrix = np.array(
    #     [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    # )
    # rotated_xz = np.dot(rotation_matrix, np.stack((x, z), axis=0))

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


def get_cage(ppr, covers):
    l = 6  # side length of the cage
    h = -7  # height of the cage
    n_topbottom = (
        5  # 6  # how many points in the top and bottom of the cage per sweep line
    )
    n_sides = 5
    # int(
    #     h / l * n_topbottom
    # )  # how many points in the sides of the cage per sweep line

    # n_sweep_cage = 100  # how many cages to sweep along reference path
    # xi_wrap = np.linspace(0, 1, n_sweep_cage)
    xi_wrap_init = np.linspace(-0.05, 0.1, 20)
    xi_wrap_mid = np.linspace(0.1, 0.9, 30)
    xi_wrap_end = np.linspace(0.9, 1.05, 20)
    xi_wrap = np.hstack([xi_wrap_init, xi_wrap_mid, xi_wrap_end])

    occ_cage = []
    for i in range(xi_wrap.shape[0]):
        ind_i = np.argmin(np.abs(ppr.parametric_path["xi"] - xi_wrap[i]))
        p_i = ppr.parametric_path["p"][ind_i]
        e1_i = ppr.parametric_path["erf"][ind_i, :, 0]
        i_horizontal = np.cross(e1_i, np.array([0, 0, 1]))

        horizontal_vecs = np.array([[1, 0, 0], [0, 1, 0]])  # pick the most orthongonal
        horizontal_vec = horizontal_vecs[
            0
        ]  # horizontal_vecs[np.argmin(np.dot(e1_i, horizontal_vecs.T))]
        i_vertical = np.cross(e1_i, horizontal_vec)

        i_horizontal = i_horizontal / np.linalg.norm(i_horizontal)
        i_vertical = i_vertical / np.linalg.norm(i_vertical)

        h_min = 3  # -1
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


poly_deg = 12  # Degree of the polynomial for corridor
LP = False  # Corridor computation with Linear Programming
polynomial_order = 3  # Order of the parametric reference polynomial
path = np.array([[0, -3, 0], [0, 1, 0], [0, 10, 0]])

ellipse_axis_max = 20  # SDP and LP ###NOT USED###
ellipse_axis_min = 0.5  # SDP and LP ###NOT USED###
eps = 1e-1  # LP ###NOT USED###

# ---------------------------------------------------------------------------- #
#                                   Planning                                   #
# ---------------------------------------------------------------------------- #

# --------------------------- Generate point cloud --------------------------- #

# ellipses
center1 = (0.1, 0.1, path[0, 1] + 0.1)  # Center point of the ellipse
size1 = (2, 3)  # Size of the ellipse along x and y axes respectively
ellipse1 = ellipse_3d(center1, size1, angle=np.pi / 4)

center2 = (0.1, 0.1, 4)  # Center point of the ellipse
size2 = (2, 4)  # Size of the ellipse along x and y axes respectively
ellipse2 = ellipse_3d(center2, size2, angle=-np.pi / 4)

center3 = (0.1, 0.1, path[-1, 1] - 0.1)  # Center point of the ellipse
size3 = (2, 3)  # Size of the ellipse along x and y axes respectively
ellipse3 = ellipse_3d(center3, size3, angle=-np.pi / 4)

# boxes
center_tower = (1, 1.0, 0.1)  # Center point of the box
dimensions = (1, 1, 7)  # Width, height, and length of the box
tower1 = box_3d(center_tower, dimensions, point_cloud_density=0.3)

center_tower = (-1.5, 7.0, 0.1)  # Center point of the box
dimensions = (1, 1, 7)  # Width, height, and length of the box
tower2 = box_3d(center_tower, dimensions, point_cloud_density=0.3)

cloud = np.vstack(
    [ellipse1, ellipse2, ellipse3, tower1, tower2]
)  # , ellipse4])  # , tower1, tower2])


# ax = plt.figure().add_subplot(111, projection="3d")
# ax.scatter(
#     cloud[:, 0],
#     cloud[:, 1],
#     cloud[:, 2],
#     c=cloud[:, 2],
#     cmap="turbo",
#     alpha=0.5,
# )
# plt.show()
# exit()
# ------------------------------ Reference path ------------------------------ #
config_path = "/home/jonarriza96/corrgen/examples/simulated/simworld_config.yaml"

box = np.array([[4, 4, 2]])
A_hs, b_hs = pdc.convex_decomposition_3D(cloud, path, box)


spline = mpl.Spline3d(polynomial_order, config_path)
coeffs, T = spline.get_coefficients(path[0], path[-1], A_hs, b_hs, path)
curve = spline.evaluate_spline(100, coeffs, T)

# ------------------------ Parameterize reference path ----------------------- #
# papor parameters
interp_order = polynomial_order - 1
n_segments = 2**4
n_eval = 30  # curve.shape[0]  # 2 * n_segments

# initialize papor
print("\nPAPOR --> Initializing ...")
ppr = PAPOR(interp_order=interp_order, n_segments=n_segments, n_eval=n_eval)

# parameterize path
curve_cs = convert_curve_to_casadi_func(
    T=T, coeffs=coeffs, order=polynomial_order, n=len(A_hs), dim=3
)
print("PAPOR --> Parameterizing curve ...")
ppr.parameterize_path(nominal_path=curve_cs)

# evaluate parameterization
print("PAPOR --> Evaluating path ...")
ppr.evaluate_parameterization()

print("PAPOR --> Done!")

# ---------------------------------------------------------------------------- #
#                              Corridor generation                             #
# ---------------------------------------------------------------------------- #

# ---------------------------- Process point cloud --------------------------- #
occ_cage = get_cage(ppr=ppr, covers=False)
cloud_no_cage = cloud.copy()
cloud = np.vstack([cloud, occ_cage])

cloud_erf, min_d_tr, max_d_tr, ind_proj = project_cloud_to_parametric_path(
    pcl=cloud, parametric_path=ppr.parametric_path, safety_check=False, prune=False
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

        ind = np.argmin(np.abs(ppr.parametric_path["xi"] - xi_eval[i]))
        gamma = ppr.parametric_path["p"][ind]
        erf = ppr.parametric_path["erf"][ind]

        w = ellipse_pts[i, j, :] + ellipse_params[i, -2:]
        ellipse_pts_world[i, j, :] = gamma + w[0] * erf[:, 1] + w[1] * erf[:, 2]

area = np.pi * ellipse_params[:, 0] / 2 * ellipse_params[:, 1] / 2
parametric_volume = np.trapz(area, xi_eval)
print(f"CORRGEN --> Parametric volume: {parametric_volume}")

# ---------------------------------------------------------------------------- #
#                                   Visualize                                  #
# ---------------------------------------------------------------------------- #

# ax = pdc.visualize_environment(Al=A_hs, bl=b_hs, p=path)
# # ax = plt.figure().add_subplot(111, projection="3d")
# ax.scatter(
#     cloud[:, 0],
#     cloud[:, 1],
#     cloud[:, 2],
#     c=cloud[:, 2],
#     cmap="turbo",
#     alpha=0.5,
# )

# ax.plot(path[:, 0], path[:, 1], path[:, 2], "ko-")
# ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "b-")

# axis_equal(cloud[:, 0], cloud[:, 1], cloud[:, 2], ax=plt.gca())

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
axis2 = height / 2 * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])

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

pts_map = np.vstack([cloud.copy()])
axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())

plt.show()
