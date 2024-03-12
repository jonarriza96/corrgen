# %%
import numpy as np
import matplotlib.pyplot as plt

import pydecomp as pdc
from papor.utils.visualize import axis_equal, plot_frames

import pickle

save_path = "/home/jonarriza96/corrgen_v2/toy_example/figures/figures/"
corrgen_path1 = "/home/jonarriza96/corrgen_v2/toy_example/figures/data/3d/3_SDP.pkl"
corrgen_path2 = "/home/jonarriza96/corrgen_v2/toy_example/figures/data/3d/6_SDP.pkl"
corrgen_path3 = "/home/jonarriza96/corrgen_v2/toy_example/figures/data/3d/12_SDP.pkl"
corrgen_path_no = (
    "/home/jonarriza96/corrgen_v2/toy_example/figures/data/3d/6_SDP_no.pkl"
)

# -------------------------------- Import data ------------------------------- #
print("Importing data...")
with open(corrgen_path1, "rb") as f:
    data = pickle.load(f)
    path = data["path"]
    occ = data["occ_cl_no_cage"]
    cage = data["occ_cage"]
    ellipse_pts_world1 = data["ellipse_pts_world"]
    ppr1 = data["ppr"]

with open(corrgen_path2, "rb") as f:
    data = pickle.load(f)
    ellipse_pts_world2 = data["ellipse_pts_world"]

with open(corrgen_path3, "rb") as f:
    data = pickle.load(f)
    ellipse_pts_world3 = data["ellipse_pts_world"]

with open(corrgen_path_no, "rb") as f:
    data = pickle.load(f)
    ellipse_pts_no = data["ellipse_pts_world"]

ellipse_pts = [ellipse_pts_world1, ellipse_pts_world2, ellipse_pts_world3]
# %%
# ------------------------------ Isometric view ------------------------------ #
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for cont, e_pts in enumerate(ellipse_pts):
    col = colors[cont]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    occ_v = occ
    ax.scatter(
        occ_v[:, 0],
        occ_v[:, 1],
        occ_v[:, 2],
        marker=".",
        c=occ_v[:, 2],
        cmap="turbo",
    )
    # if cont == 0:
    #     n_start = 0
    # else:
    #     n_start = 10
    n_start = 10
    n_end = -1
    n_eval = e_pts.shape[0]
    n_angles = e_pts.shape[1]
    if cont == 0:
        n_step_rings = 10
    else:
        n_step_rings = 5
    for j in range(0, n_angles - 1):
        ax.plot(
            e_pts[n_start:n_end, j, 0],
            e_pts[n_start:n_end, j, 1],
            e_pts[n_start:n_end, j, 2],
            "k-",
            alpha=0.25,
        )
    for i in range(n_start, n_eval, n_step_rings):
        ax.plot(
            e_pts[i, :, 0],
            e_pts[i, :, 1],
            e_pts[i, :, 2],
            "-",
            color=col,
            alpha=0.5,
        )

    pts_map = occ.copy()
    axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())

    # ax.view_init(azim=-66, elev=22)
    ax.view_init(azim=-39, elev=17)
    ax.dist = 6
    plt.tight_layout()
    ax.set_axis_off()

    # fig.savefig(save_path + "overview_isometric_" + str(cont) + ".pdf", dpi=1800)
plt.show()

# %%
# --------------------------------- Top view --------------------------------- #

fig = plt.figure()
ax = fig.add_subplot(111)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for en, corr_ind in enumerate([0, 1, 2]):

    for i in range(n_start, n_eval):
        ax.plot(
            ellipse_pts[en][i, :, 0],
            ellipse_pts[en][i, :, 1],
            "-",
            color=colors[en],
            alpha=0.075,
        )

    ax.axis("equal")
    # ax.legend()
    ax.set_axis_off()

ax.scatter(
    occ[:, 0],
    occ[:, 1],
    c=occ[:, 2],
    cmap="turbo",
    marker=".",
)

# n_angles = ellipse_pts[en].shape[1]
# for j in [0, 9]:
#     ax.plot(
#         ellipse_pts[en][n_start:, j, 0],
#         ellipse_pts[en][n_start:, j, 1],
#         # ellipse_pts_world2[n_start:, j, 2],
#         "k-",
#         alpha=0.25,
#     )

plt.tight_layout()
ax.set_axis_off()
fig.savefig(save_path + "3d_top.pdf", dpi=1800)
plt.show()


# --------------------------------- Side view -------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(111)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for en, corr_ind in enumerate([0, 1, 2]):

    for i in range(n_start, n_eval):
        ax.plot(
            ellipse_pts[en][i, :, 1],
            ellipse_pts[en][i, :, 2],
            "-",
            color=colors[en],
            alpha=0.075,
        )

    ax.axis("equal")
    # ax.legend()
    ax.set_axis_off()

ax.scatter(
    occ[:, 1],
    occ[:, 2],
    c=occ[:, 2],
    cmap="turbo",
    marker=".",
)


plt.tight_layout()
ax.set_axis_off()
fig.savefig(save_path + "3d_side.pdf", dpi=1800)
plt.show()

# %%
# ---------------------------- No offset isometric --------------------------- #

col = "r"
fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
ax = fig.add_subplot(111)

occ_v = occ
ax.scatter(
    occ_v[:, 0],
    occ_v[:, 1],
    # occ_v[:, 2],
    marker=".",
    c=occ_v[:, 2],
    cmap="turbo",
)
# if cont == 0:
#     n_start = 0
# else:
#     n_start = 10
n_start = 10
n_end = -1
n_eval = ellipse_pts_no.shape[0]
n_angles = ellipse_pts_no.shape[1]
n_step_rings = 10
# for j in range(0, n_angles - 1):
#     ax.plot(
#         ellipse_pts_no[n_start:n_end, j, 0],
#         ellipse_pts_no[n_start:n_end, j, 1],
#         # ellipse_pts_no[n_start:n_end, j, 2],
#         "k-",
#         alpha=0.25,
#     )
for i in range(n_start, n_eval):  # , n_step_rings):
    ax.plot(
        ellipse_pts_no[i, :, 0],
        ellipse_pts_no[i, :, 1],
        # ellipse_pts_no[i, :, 2],
        "-",
        color=col,
        alpha=0.075,
    )

ax.plot(
    ppr1.parametric_path["p"][:, 0],
    ppr1.parametric_path["p"][:, 1],
    # ppr1.parametric_path["p"][:, 2],
    "k--",
)

# pts_map = occ.copy()
# axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())

# ax.view_init(azim=-66, elev=22)
# ax.view_init(azim=-39, elev=17)
# ax.dist = 6
ax.set_aspect("equal")
# plt.tight_layout()
ax.set_axis_off()

# fig.savefig(save_path + "3d_top_no" + ".pdf", dpi=1800)
plt.show()


# %%

# ----------------------------------- Cage ----------------------------------- #


def get_cage(ppr, covers):
    l = 6  # side length of the cage
    h = -8  # height of the cage
    n_topbottom = (
        5  # 6  # how many points in the top and bottom of the cage per sweep line
    )
    n_sides = 5
    # int(
    #     h / l * n_topbottom
    # )  # how many points in the sides of the cage per sweep line

    n_sweep_cage = 1000  # how many cages to sweep along reference path
    xi_wrap = np.linspace(0, 1, n_sweep_cage)
    # xi_wrap_init = np.linspace(-0.05, 0.1, 20)
    # xi_wrap_mid = np.linspace(0.1, 0.9, 30)
    # xi_wrap_end = np.linspace(0.9, 1.05, 20)
    # xi_wrap = np.hstack([xi_wrap_init, xi_wrap_mid, xi_wrap_end])

    edge1 = []
    edge2 = []
    edge3 = []
    edge4 = []
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

        h_min = 4  # -1
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

        edge1 += [
            p_i[:, None]
            + (i_vertical[:, None] * h_max)
            + (i_horizontal[:, None] * -l / 2)
        ]

        edge2 += [
            p_i[:, None]
            + (i_vertical[:, None] * h_max)
            + (i_horizontal[:, None] * l / 2)
        ]

        edge3 += [
            p_i[:, None]
            + (i_vertical[:, None] * h_min)
            + (i_horizontal[:, None] * -l / 2)
        ]

        edge4 += [
            p_i[:, None]
            + (i_vertical[:, None] * h_min)
            + (i_horizontal[:, None] * l / 2)
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
    edge1 = np.squeeze(edge1)
    edge2 = np.squeeze(edge2)
    edge3 = np.squeeze(edge3)
    edge4 = np.squeeze(edge4)
    edges = [edge1, edge2, edge3, edge4]
    print("CORRGEN--> Wrapper points: ", occ_cage.shape[0])

    return occ_cage, edges


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

cage, edges = get_cage(ppr1, True)

occ_v = occ
ax.scatter(
    occ_v[:, 0],
    occ_v[:, 1],
    occ_v[:, 2],
    marker=".",
    c=occ_v[:, 2],
    cmap="turbo",
)

ind = cage[:, 2] > 3
ax.plot(cage[ind, 0], cage[ind, 1], cage[ind, 2], color="b", alpha=0.25)

ind = cage[:, 2] < -2
ax.plot(cage[ind, 0], cage[ind, 1], cage[ind, 2], color="b", alpha=0.25)

ind = cage[:, 0] < -3
ax.plot(cage[ind, 0], cage[ind, 1], cage[ind, 2], color="b", alpha=0.25)

ind = cage[:, 0] > 2.7
ax.plot(cage[ind, 0], cage[ind, 1], cage[ind, 2], color="b", alpha=0.25)

# ind = cage[:, 1] < -2
# ax.plot_trisurf(cage[ind, 0], cage[ind, 1], cage[ind, 2], color="b", alpha=0.5)

for i in range(4):
    ax.plot(edges[i][:, 0], edges[i][:, 1], edges[i][:, 2], color="k")

# ax.plot(
#     [edges[0][0, 0], edges[1][0, 0]],
#     [edges[0][0, 1], edges[1][0, 1]],
#     [edges[0][0, 2], edges[1][0, 2]],
#     "-k",
# )

ax.plot(
    ppr1.parametric_path["p"][:, 0],
    ppr1.parametric_path["p"][:, 1],
    # ppr1.parametric_path["p"][:, 2],
    "k--",
)


pts_map = occ.copy()
axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())
ax.view_init(azim=-39, elev=17)
ax.dist = 6
ax.set_axis_off()
# fig.savefig(save_path + "3d_cage" + ".pdf", dpi=1800)
plt.show()
