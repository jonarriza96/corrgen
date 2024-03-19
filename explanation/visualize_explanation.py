# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt

from corrgen.utils import get_corrgen_path, get_ellipse_parameters
from papor.utils.visualize import plot_frames


save_path = "/home/jonarriza96/corrgen/paper/figures/"

# -------------------------------- Import data ------------------------------- #
print("Importing world...")
path = "/home/jonarriza96/corrgen/examples/experiments/data/3d/"
file_name = "world.pkl"
with open(path + file_name, "rb") as f:
    data = pickle.load(f)
    world = data["world"]
print("Done.")


print("Importing corridors ...")
corridors = []
corridor_evals = []
for k in range(14, 15):
    with open(path + f"corridor_pd{k}.pkl", "rb") as f:
        data = pickle.load(f)
    corridors.append(data["corridor"])
    corridor_evals.append(data["evaluation"])
    print(k)

print("Done")

#     pickle.dump(data, f)
# %%

# ------------------------------ Isometric view ------------------------------ #
corr_ind = 0
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
if corr_ind == 0:
    col = "b"  # colors[0]
elif corr_ind == 3:
    col = "g"  # colors[2]


occ_cl_nbnd = world["occ_cl_nbnd"]
linear_path = world["linear_path"]
ppr = world["ppr"]
n_eval = corridor_evals[corr_ind]["xi_eval"].shape[0]
ellipse_pts_world = corridor_evals[corr_ind]["ellipse_pts_world"]

ind = np.linalg.norm(occ_cl_nbnd[:, :2], axis=1) < 8.5
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    occ_cl_nbnd[ind, 0],
    occ_cl_nbnd[ind, 1],
    occ_cl_nbnd[ind, 2],
    c=occ_cl_nbnd[ind, 2],
    cmap="turbo",
    alpha=0.5,
)
ax.plot(
    ppr.parametric_path["p"][20:-160, 0],
    ppr.parametric_path["p"][20:-160, 1],
    ppr.parametric_path["p"][20:-160, 2],
    "--k",
)
# gamma = ppr.parametric_path["p"][20:-160]
# e1 = ppr.parametric_path["erf"][20:-160, :, 0]
# e2 = ppr.parametric_path["erf"][20:-160, :, 1]
# e3 = ppr.parametric_path["erf"][20:-160, :, 2]
# erf = ppr.parametric_path["erf"][20:-160, :, :]
# ax = plot_frames(
#     r=gamma,
#     e1=-e1,
#     e2=e2,
#     e3=-e3,
#     scale=0.25,
#     ax=ax,
#     planar=False,
#     interval=0.9,
# )


# ax = mpl.visualize_environment(Al=A_hs, bl=b_hs, p=linear_path, planar=False, ax=ax)
# ax.plot(linear_path[:, 0], linear_path[:, 1], linear_path[:, 2], "-ok")

for j in range(ellipse_pts_world.shape[1] - 1):
    ax.plot(
        ellipse_pts_world[20:-160, j, 0],
        ellipse_pts_world[20:-160, j, 1],
        ellipse_pts_world[20:-160, j, 2],
        "k-",
        alpha=0.5,
    )
for i in range(20, n_eval - 150, 10):
    ax.plot(
        ellipse_pts_world[i, :, 0],
        ellipse_pts_world[i, :, 1],
        ellipse_pts_world[i, :, 2],
        "-",
        color="m",
        alpha=0.5,
    )
ax.plot(
    ellipse_pts_world[i, :, 0],
    ellipse_pts_world[i, :, 1],
    ellipse_pts_world[i, :, 2],
    "-",
    color="k",
    linewidth=1.5,
    alpha=1,
)


ct = corridor_evals[0]["ellipse_params"][20:-160, -2:]
ct_world = []
for i in range(ct.shape[0]):
    ct_world.append(gamma[i] + (e2[i] * ct[i, 0] + e3[i] * ct[i, 1]))
ct_world = np.squeeze(ct_world)

ax.plot(ct_world[:, 0], ct_world[:, 1], ct_world[:, 2], "r-")


# ax.view_init(azim=54, elev=14)
ax.view_init(azim=54, elev=7)
# ax.dist = 8
plt.tight_layout()
ax.set_axis_off()
# fig.savefig(save_path + 'explanation4.pdf',dpi=1800)

plt.show()
# %%
