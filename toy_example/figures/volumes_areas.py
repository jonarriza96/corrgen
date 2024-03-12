# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle


save_path = "/home/jonarriza96/corrgen/paper/figures/"


print("Importing data...")
areas_path = "/home/jonarriza96/corrgen_v2/toy_example/figures/data/2d/areas.pkl"
volumes_path = "/home/jonarriza96/corrgen/examples/experiments/data/3d/"
with open(areas_path, "rb") as f:
    data = pickle.load(f)
    area_degs = data["poly_degs"]
    areas = data["areas"]

volume_degs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
volumes = []
for i in volume_degs:
    with open(volumes_path + str(i) + "_SDP.pkl", "rb") as f:
        data = pickle.load(f)
        volumes += data["corridor_volume"]
print("Done.")


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fs = 14
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(area_degs, areas, ".-", color="k")
ax[0].plot(3, areas[0], marker="o", color=colors[0], alpha=0.95, markersize=8)
ax[0].plot(6, areas[3], marker="o", color=colors[1], alpha=0.95, markersize=8)
ax[0].plot(15, areas[11], marker="o", color=colors[2], alpha=0.95, markersize=8)
ax[0].set_ylabel("Area", fontsize=fs)
ax[0].set_yticks([1, 3, 5], fontsize=fs)
ax[0].grid(axis="x")

ax[0].plot(
    [3, 3], [0.6 * min(areas), 1.1 * max(areas)], "-", color=colors[0], alpha=0.5
)
ax[0].plot(
    [6, 6], [0.6 * min(areas), 1.1 * max(areas)], "-", color=colors[1], alpha=0.5
)
ax[0].plot(
    [15, 15], [0.6 * min(areas), 1.1 * max(areas)], "-", color=colors[2], alpha=0.5
)
ax[0].set_ylim([0.6 * min(areas), 1.1 * max(areas)])

ax[1].plot(volume_degs, volumes, ".-", color="k")
ax[1].plot(3, volumes[0], marker="o", color=colors[0], alpha=0.95, markersize=8)
ax[1].plot(6, volumes[3], marker="o", color=colors[1], alpha=0.95, markersize=8)
ax[1].plot(15, volumes[12], marker="o", color=colors[2], alpha=0.95, markersize=8)

ax[1].plot(
    [3, 3], [0.975 * min(volumes), 1.02 * max(volumes)], "-", color=colors[0], alpha=0.5
)
ax[1].plot(
    [6, 6], [0.975 * min(volumes), 1.02 * max(volumes)], "-", color=colors[1], alpha=0.5
)
ax[1].plot(
    [15, 15],
    [0.975 * min(volumes), 1.02 * max(volumes)],
    "-",
    color=colors[2],
    alpha=0.5,
)
ax[1].set_ylim([0.975 * min(volumes), 1.02 * max(volumes)])
ax[1].set_ylabel("Volume", fontsize=fs)

ax[1].set_xlabel("Polynomial degree", fontsize=fs)
# ax[1].set_xticks([2,8,14,20])
ax[1].set_xticks([3, 6, 9, 12, 15, 18])
ax[1].set_yticks([6, 7, 8])
ax[1].grid(axis="x")

fig.set_size_inches(4.3, 3.5)
# fig.subplots_adjust(hspace=0.2, wspace=0)
plt.tight_layout()

# fig.savefig(save_path + 'volumes_areas.pdf',dpi=1800)

plt.show()
