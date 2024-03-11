# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

save_path = "/home/jonarriza96/corrgen_v2/kitti/figures/figures/"
corrgen_path = "/home/jonarriza96/corrgen_v2/kitti/data/case2/corrgen/time_volumes/"


LP_times = []
SDP_times = []
LP_volumes = []
SDP_volumes = []

poly_degs = [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
]
print("Importing data...")
for k in poly_degs:
    with open(corrgen_path + str(k) + "_LP.pkl", "rb") as f:
        data = pickle.load(f)
        LP_times += [data["solve_time"]]
        LP_volumes += [data["corridor_volume"]]

for k in poly_degs:
    with open(corrgen_path + str(k) + "_SDP.pkl", "rb") as f:
        data = pickle.load(f)
        SDP_times += [data["solve_time"]]
        SDP_volumes += [data["corridor_volume"]]

# %%
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fs = 12
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(poly_degs, SDP_times, ".-", color="r")
ax[0].plot(poly_degs, LP_times, ".-", color="b")
ax[0].set_ylabel("Solve-time [s]", fontsize=fs)
ax[0].set_yticks([0.05, 0.25, 0.5, 0.75, 1], fontsize=fs)
ax[0].grid(axis="x")
# ax[0].set_ylim([0.6 * min(LP_times), 1.1 * max(LP_times)])

ax[1].plot(poly_degs, SDP_volumes, ".-", color="r", label="SDP")
ax[1].plot(poly_degs, LP_volumes, ".-", color="b", label="LP")
# ax[1].set_ylim([0.975 * min(LP_volumes), 1.02 * max(LP_volumes)])
ax[1].set_ylabel("Volume", fontsize=fs)

ax[1].set_xlabel("Polynomial degree (n)", fontsize=fs)
# ax[1].set_xticks([2,8,14,20])
# ax[1].set_xticks([3, 6, 9, 12, 15, 18])
ax[1].set_yticks([10, 15, 20, 25])
ax[1].grid(axis="x")
ax[1].legend()

fig.set_size_inches(3.15, 3.75)
# fig.subplots_adjust(hspace=0.2, wspace=0)
plt.tight_layout()

# fig.savefig(save_path + 'volumes_times.pdf',dpi=1800)

plt.show()
