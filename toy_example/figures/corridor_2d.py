# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt

from papor.utils.visualize import plot_frames


import pyny3d.geoms as pyny
import cdd
from papor.utils.visualize import axis_equal, plot_frames


# %%
save_path = "/home/jonarriza96/corrgen_v2/toy_example/figures/figures/"
# -------------------------------- Import data ------------------------------- #
print("Importing world...")
path = "/home/jonarriza96/corrgen_v2/toy_example/figures/data/2d/"
file_name = "world.pkl"
with open(path + file_name, "rb") as f:
    data = pickle.load(f)
    world = data["world"]
print("Done.")


print("Importing corridors ...")
corridors = []
corridor_evals = []
degrees = np.arange(3, 20)
for k in degrees:
    try:
        with open(path + f"corridor_pd{k}.pkl", "rb") as f:
            data = pickle.load(f)
        corridors.append(data["corridor"])
        corridor_evals.append(data["evaluation"])
    except Exception:
        pass
    print(k)

print("Done")


# %%
occ_cl_nbnd = world["occ_cl_nbnd"]
ppr = world["ppr"]

# 2D view
fig = plt.figure()
ax = fig.add_subplot(111)
# ax = mpl.visualize_environment(Al=A, bl=b, p=path, planar=True, ax=ax)

ax.plot(occ_cl_nbnd[:, 0], occ_cl_nbnd[:, 1], ".", color="k")
ax.plot(
    ppr.parametric_path["p"][:, 0],
    ppr.parametric_path["p"][:, 1],
    "k--",
    # linewidth=1,
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
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for k, deg in enumerate([3, 6, 15]):
    corr_ind = deg - 3
    corr_ev = corridor_evals[corr_ind]["corridor_eval"]
    ax.plot(corr_ev[:, 0, 0], corr_ev[:, 0, 1], "-", color=colors[k], linewidth=2)
    ax.plot(corr_ev[:, 1, 0], corr_ev[:, 1, 1], "-", color=colors[k], linewidth=2)

    # for i in range(0,corr_ev.shape[0]):
    ax.plot(
        [corr_ev[:, 0, 0], corr_ev[:, 1, 0]],
        [corr_ev[:, 0, 1], corr_ev[:, 1, 1]],
        color=colors[k],
        alpha=0.2,
        linewidth=3,
    )
    # ax.plot(corridor_evals[corr_ind]["corridor_eval"][:, 0, 0], corridor_evals[corr_ind]["corridor_eval"][:, 0, 1], "-", color=colors[k])


plt.tight_layout()
plt.axis("off")
ax.set_aspect("equal")

# ax.get_figure().savefig(save_path + '2d_top.pdf',dpi=1800)
