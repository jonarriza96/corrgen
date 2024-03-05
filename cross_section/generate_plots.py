# %%
import numpy as np
import casadi as cs
import cvxpy as cp
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import pickle

import pyny3d.geoms as pyny
import cdd


def hyperplane_to_vertices(A, b):
    mat = cdd.Matrix(np.hstack([b[:, np.newaxis], -A]))
    mat.rep_type = cdd.RepType.INEQUALITY

    poly = cdd.Polyhedron(mat)
    gen = poly.get_generators()

    vert = np.array(list(gen))[:, 1:]

    return vert


def visualize_hyperplane(A, b):
    vert = hyperplane_to_vertices(A, b)
    polygon = pyny.Polygon(vert)
    ax = pyny.Surface(polygon).plot2d(c_poly="0000FF10", alpha=0.15, ret=True)
    return ax


%matplotlib tk

print("Importing data...")
path = "/home/jonarriza96/corrgen_v2/data/cross_section6.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)
    w = data["w"]
    h = data["h"]
    a = data["a"]
    cent = data["cent"]
    A = data["A"]
    b = data["b"]
    pcl = data["pcl"]
    p = data["p"]
    pc = data["pc"]
print("Done.")

e = []
for width, height, angle, pc, cl in zip(w, h, a, cent, ["r", "g", "b"]):
    e.append(
        Ellipse(
            xy=pc,
            width=width,
            height=height,
            angle=np.rad2deg(angle),
            color=cl,
            alpha=0.1,
            # linewidth=2,
        )
    )

ax = visualize_hyperplane(A, b)

# others
ax.scatter(p[0], p[1], color="r", marker="x")
ax.scatter(pcl[:, 0], pcl[:, 1], color="k", marker=".")
ax.add_patch(e[2])
ax.add_patch(e[1])
ax.add_patch(e[0])
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.25, 0.25)
ax.plot(pc[0], pc[1], "b",marker=".")
plt.plot([0, pc[0]], [0, pc[1]], "--b", alpha=0.6, linewidth=0.6)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis("off")
plt.show()

# %%
save_path = "/home/jonarriza96/corrgen/paper/figures/"
plt.gcf().savefig(save_path + 'cross_section_matplotlib3.pdf',dpi=1800)
