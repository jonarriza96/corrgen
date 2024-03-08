# %%
import numpy as np
import matplotlib.pyplot as plt

import pydecomp as pdc
from papor.utils.visualize import axis_equal, plot_frames

import pickle

save_path = "/home/jonarriza96/corrgen_v2/kitti/figures/figures/"

decomp_path = "/home/jonarriza96/corrgen_v2/kitti/data/case2/decomp/10.pkl"
corrgen_path1 = "/home/jonarriza96/corrgen_v2/kitti/data/case2/corrgen/9_LP.pkl"
corrgen_path2 = "/home/jonarriza96/corrgen_v2/kitti/data/case3/corrgen/9_LP.pkl"

print("Importing data...")
# with open(decomp_path, "rb") as f:
#     data = pickle.load(f)
#     A_hs = data["A_hs"]
#     b_hs = data["b_hs"]
#     path = data["path"]
#     occ_clean = data["occ_clean"]

with open(corrgen_path1, "rb") as f:
    data = pickle.load(f)
    path = data["path"]
    occ_clean = data["occ_clean"]
    ellipse_pts_world1 = data["ellipse_pts_world"]
    ppr1 = data["ppr"]

with open(corrgen_path2, "rb") as f:
    data = pickle.load(f)
    path = data["path"]
    occ_clean = data["occ_clean"]
    ellipse_pts_world2 = data["ellipse_pts_world"]
    ppr2 = data["ppr"]

# %%
# ax = pdc.visualize_environment(Al=A_hs, bl=b_hs, p=None, planar=False)

step_rings = 9

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

occ_v = occ_clean
ind = occ_v[:, 2] > -1.5
ax.scatter(
    occ_v[ind, 0],
    occ_v[ind, 1],
    occ_v[ind, 2],
    marker=".",
    c=-(occ_v[ind, 0] ** 2 + occ_v[ind, 1] ** 2),
    cmap="turbo",
    # alpha=0.8,
)

n_start = 3
n_angles = ellipse_pts_world1.shape[1]
for j in range(0, n_angles - 1):
    ax.plot(
        ellipse_pts_world1[n_start:-8, j, 0],
        ellipse_pts_world1[n_start:-8, j, 1],
        ellipse_pts_world1[n_start:-8, j, 2],
        "k-",
        alpha=0.25,
    )
n_eval = ellipse_pts_world1.shape[0]
for i in range(n_start, n_eval, step_rings):
    ax.plot(
        ellipse_pts_world1[i, :, 0],
        ellipse_pts_world1[i, :, 1],
        ellipse_pts_world1[i, :, 2],
        "m-",
        alpha=0.5,
    )

ax.plot(
    ellipse_pts_world1[n_start, :, 0],
    ellipse_pts_world1[n_start, :, 1],
    ellipse_pts_world1[n_start, :, 2],
    "-",
    color="k",
    linewidth=1,
    alpha=1,
)


n_angles = ellipse_pts_world2.shape[1]
for j in range(n_angles - 1):
    ax.plot(
        ellipse_pts_world2[:-2, j, 0],
        ellipse_pts_world2[:-2, j, 1],
        ellipse_pts_world2[:-2, j, 2],
        "k-",
        alpha=0.25,
    )
n_eval = ellipse_pts_world2.shape[0]
for i in range(0, n_eval, step_rings):
    ax.plot(
        ellipse_pts_world2[i, :, 0],
        ellipse_pts_world2[i, :, 1],
        ellipse_pts_world2[i, :, 2],
        "g-",
        alpha=0.5,
    )
ax.plot(
    ellipse_pts_world2[0, :, 0],
    ellipse_pts_world2[0, :, 1],
    ellipse_pts_world2[0, :, 2],
    "-",
    color="k",
    linewidth=1,
    alpha=1,
)

# plt.plot(
#     ppr1.parametric_path["p"][:, 0],
#     ppr1.parametric_path["p"][:, 1],
#     ppr1.parametric_path["p"][:, 2],
#     "r-",
#     alpha=0.5,
# )
# plt.plot(
#     ppr2.parametric_path["p"][:, 0],
#     ppr2.parametric_path["p"][:, 1],
#     ppr2.parametric_path["p"][:, 2],
#     "r-",
#     alpha=0.5,
# )

pts_map = occ_clean.copy()
axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())

ax.view_init(azim=156, elev=14)
# ax.view_init(azim=155, elev=31)
ax.dist = 4
plt.tight_layout()
ax.set_axis_off()
# fig.savefig(save_path + 'overview_isometric.pdf',dpi=1800)

# %%
fig = plt.figure()
ax = fig.add_subplot(111)  # , projection="3d")

occ_v = occ_clean
occ_v = occ_v[occ_v[:, 1] > -49]
occ_v = occ_v[occ_v[:, 1] < 0]
occ_v = occ_v[occ_v[:, 0] < 49]
ind = occ_v[:, 2] > -1.5
ax.scatter(
    occ_v[ind, 0],
    occ_v[ind, 1],
    # occ_v[ind, 2],
    marker=".",
    c=-(occ_v[ind, 0] ** 2 + occ_v[ind, 1] ** 2),
    cmap="turbo",
)

n_angles = ellipse_pts_world1.shape[1]
for j in [0, 9]:
    ax.plot(
        ellipse_pts_world1[:, j, 0],
        ellipse_pts_world1[:, j, 1],
        # ellipse_pts_world1[:, j, 2],
        "k-",
        alpha=0.5,
    )
n_eval = ellipse_pts_world1.shape[0]
for i in range(0, n_eval):
    ax.plot(
        ellipse_pts_world1[i, :, 0],
        ellipse_pts_world1[i, :, 1],
        # ellipse_pts_world1[i, :, 2],
        "m-",
        alpha=0.1,
    )

n_angles = ellipse_pts_world2.shape[1]
for j in [0, 9]:
    ax.plot(
        ellipse_pts_world2[:, j, 0],
        ellipse_pts_world2[:, j, 1],
        # ellipse_pts_world2[:, j, 2],
        "k-",
        alpha=0.5,
    )
n_eval = ellipse_pts_world2.shape[0]
for i in range(0, n_eval):
    ax.plot(
        ellipse_pts_world2[i, :, 0],
        ellipse_pts_world2[i, :, 1],
        # ellipse_pts_world2[i, :, 2],
        "g-",
        alpha=0.1,
    )


ax.dist = 3
# plt.tight_layout()
# ax.set_axis_off()
ax.set_ylim([-50, 0])
ax.axis("equal")
plt.show()
# fig.savefig(save_path + 'kitti_cross_section.pdf',dpi=1800)
