# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

save_path = "/home/jonarriza96/corrgen_v2/kitti/figures/figures/"
corrgen_path = "/home/jonarriza96/corrgen_v2/kitti/data/case2/corrgen/"


LP_times = []
SDP_times = []
LP_volumes = []
SDP_volumes = []

poly_degs = [3, 6, 9, 12, 16, 24]
print("Importing data...")
for k in poly_degs:
    with open(corrgen_path + str(k) + "_LP.pkl", "rb") as f:
        data = pickle.load(f)
        LP_times += [data["solve_time"]]
        LP_volumes += [data["corridor_volume"]]

# %%
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fs = 12
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(poly_degs, LP_times, ".-", color="k")
ax[0].set_ylabel("Solve-time [s]", fontsize=fs)
# ax[0].set_yticks([1, 3, 5], fontsize=fs)
ax[0].grid(axis="x")
ax[0].set_ylim([0.6 * min(LP_times), 1.1 * max(LP_times)])

ax[1].plot(poly_degs, LP_volumes, ".-", color="k")
ax[1].set_ylim([0.975 * min(LP_volumes), 1.02 * max(LP_volumes)])
ax[1].set_ylabel("Volume", fontsize=fs)

ax[1].set_xlabel("Polynomial degree (n)", fontsize=fs)
# ax[1].set_xticks([2,8,14,20])
# ax[1].set_xticks([3, 6, 9, 12, 15, 18])
# ax[1].set_yticks([6, 7, 8])
ax[1].grid(axis="x")

fig.set_size_inches(3.15, 3.75)
# fig.subplots_adjust(hspace=0.2, wspace=0)
plt.tight_layout()

# fig.savefig(save_path + 'volumes_times.pdf',dpi=1800)

plt.show()
