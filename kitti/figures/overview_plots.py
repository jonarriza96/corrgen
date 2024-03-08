# %%
import numpy as np
import matplotlib.pyplot as plt

import pydecomp as pdc
from papor.utils.visualize import axis_equal, plot_frames

import pickle


decomp_path = "/home/jonarriza96/corrgen_v2/kitti/data/case2/decomp/10.pkl"
corrgen_path1 = "/home/jonarriza96/corrgen_v2/kitti/data/case2/corrgen/24_LP.pkl"
corrgen_path2 = "/home/jonarriza96/corrgen_v2/kitti/data/case3/corrgen/24_LP.pkl"

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

with open(corrgen_path2, "rb") as f:
    data = pickle.load(f)
    path = data["path"]
    occ_clean = data["occ_clean"]
    ellipse_pts_world2 = data["ellipse_pts_world"]

# %%
# ax = pdc.visualize_environment(Al=A_hs, bl=b_hs, p=None, planar=False)

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
)

n_angles = ellipse_pts_world1.shape[1]
for j in range(n_angles - 1):
    ax.plot(
        ellipse_pts_world1[:, j, 0],
        ellipse_pts_world1[:, j, 1],
        ellipse_pts_world1[:, j, 2],
        "k-",
        alpha=0.5,
    )
n_eval = ellipse_pts_world1.shape[0]
for i in range(0, n_eval, 10):
    ax.plot(
        ellipse_pts_world1[i, :, 0],
        ellipse_pts_world1[i, :, 1],
        ellipse_pts_world1[i, :, 2],
        "m-",
        alpha=0.5,
    )

n_angles = ellipse_pts_world2.shape[1]
for j in range(n_angles - 1):
    ax.plot(
        ellipse_pts_world2[:, j, 0],
        ellipse_pts_world2[:, j, 1],
        ellipse_pts_world2[:, j, 2],
        "k-",
        alpha=0.5,
    )
n_eval = ellipse_pts_world2.shape[0]
for i in range(0, n_eval, 10):
    ax.plot(
        ellipse_pts_world2[i, :, 0],
        ellipse_pts_world2[i, :, 1],
        ellipse_pts_world2[i, :, 2],
        "g-",
        alpha=0.5,
    )

pts_map = occ_clean.copy()
axis_equal(X=pts_map[:, 0], Y=pts_map[:, 1], Z=pts_map[:, 2], ax=plt.gca())
plt.show()
