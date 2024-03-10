# %%
import numpy as np
import matplotlib.pyplot as plt

import pydecomp as pdc
from papor.utils.visualize import axis_equal, plot_frames

import pyny3d.geoms as pyny
import cdd


import pickle

save_path = "/home/jonarriza96/corrgen_v2/kitti/figures/figures/"

decomp_path = "/home/jonarriza96/corrgen_v2/kitti/data/case2/decomp/6.pkl"
corrgen_path = "/home/jonarriza96/corrgen_v2/kitti/data/case2/corrgen/16_LP.pkl"

# n_decomp--> 4,5,6,8
# n_corrgen--> 3(6),12,16,24


def visualize_environment(
    Al, bl, p=None, p_interp=None, q=None, ax=None, planar=False, ax_view=True
):
    bl = [np.squeeze([b]) for b in bl]

    if ax is None:
        if not planar:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

    vert = []
    polyhedrons = []
    if Al is not None:
        for A, b in zip(Al, bl):
            if not (A == 0).all():  # only draw if polyhedron is active
                # remove zeros (inactive sides)
                ind_zero = ~np.all(A == 0, axis=1)
                A = A[ind_zero]
                b = b[ind_zero]

                # convert to vertices and arrange by faces
                mat = cdd.Matrix(np.hstack([b[:, np.newaxis], -A]))
                mat.rep_type = cdd.RepType.INEQUALITY

                # visualize
                try:
                    poly = cdd.Polyhedron(mat)
                    gen = poly.get_generators()
                    vert += [np.array(list(gen))[:, 1:]]
                    intersections = [list(x) for x in poly.get_input_incidence()]
                    if planar:
                        polyhedrons += [pyny.Polygon(vert[-1])]
                    else:
                        polygon = []
                        for inter in intersections[:-1]:
                            if inter:
                                polygon += [pyny.Polygon(vert[-1][inter])]
                        polyhedrons += [pyny.Polyhedron(polygon)]
                except:
                    print("Error in plotting polyhedrons")

    if not planar:
        if len(vert) > 0:
            vert = np.concatenate(vert)
            ax = axis_equal(vert[:, 0], vert[:, 1], vert[:, 2], ax=ax)
        else:
            if Al is None and p_interp is not None:
                ax = axis_equal(p_interp[:, 0], p_interp[:, 1], p_interp[:, 2], ax=ax)
            elif p_interp is not None:
                ax = axis_equal(p_interp[:, 0], p_interp[:, 1], p_interp[:, 2])

    for plh in polyhedrons:
        if planar:
            ax = pyny.Surface(polyhedrons).plot2d(c_poly="k", alpha=0.15, ret=True)
            if p_interp is not None:
                ax.plot(
                    p_interp[0, 0], p_interp[0, 1], "o", color="lime", markersize=15
                )
                ax.plot(
                    p_interp[-1, 0], p_interp[-1, 1], "o", color="red", markersize=15
                )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if not ax_view:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

        else:
            ax = plh.plot(color="#0000FF10", ax=ax, ret=True)

            if p_interp is not None:
                ax.plot(
                    p_interp[0, 0],
                    p_interp[0, 1],
                    p_interp[0, 2],
                    "o",
                    color="lime",
                    markersize=15,
                )
                ax.plot(
                    p_interp[-1, 0],
                    p_interp[-1, 1],
                    p_interp[-1, 2],
                    "o",
                    color="red",
                    markersize=15,
                )
            ax.set_zlabel("z")

        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # if q is not None:
    #     e1_i, e2_i, e3_i = quaternion_to_rotation(q[0, :]).T
    #     e1_f, e2_f, e3_f = quaternion_to_rotation(q[1, :]).T

    #     r = np.vstack([p_interp[0, :], p_interp[-1, :]])
    #     e1 = np.vstack([e1_i, e1_f])
    #     e2 = np.vstack([e2_i, e2_f])
    #     e3 = np.vstack([e3_i, e3_f])

    #     if planar:
    #         r = np.hstack([r, np.zeros((2, 1))])
    #     plot_frames(r, e1, e2, e3, interval=1, scale=1.1, ax=ax, planar=planar)

    if p is not None:
        if planar:
            ax.plot(p[:, 0], p[:, 1], "-o", alpha=0.75, color="k")
            ax.plot(p[0, 0], p[0, 1], "-o", alpha=0.75, color="g")
            ax.plot(p[-1, 0], p[-1, 1], "-o", alpha=0.75, color="r")
        else:
            # ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", alpha=0.5, color="k")
            ax.plot(p[0, 0], p[0, 1], p[0, 2], "-o", alpha=0.75, color="g")
            ax.plot(p[-1, 0], p[-1, 1], p[-1, 2], "-o", alpha=0.75, color="r")
    return ax


print("Importing data...")
with open(decomp_path, "rb") as f:
    data = pickle.load(f)
    A_hs = data["A_hs"]
    b_hs = data["b_hs"]
    path = data["path"]
    occ_clean = data["occ_clean"]

with open(corrgen_path, "rb") as f:
    data = pickle.load(f)
    path = data["path"]
    occ_clean = data["occ_clean"]
    ellipse_pts_world = data["ellipse_pts_world"]
    ppr = data["ppr"]


# %%
# fig = plt.figure()
# ax = fig.add_subplot(111)  # , projection="3d")


ax = visualize_environment(
    Al=[Aa[:, :2] for Aa in A_hs],
    bl=b_hs,
    p=None,
    p_interp=None,
    q=None,
    planar=True,
    ax_view=False,
)


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

n_angles = ellipse_pts_world.shape[1]
for j in [0, 8]:
    ax.plot(
        ellipse_pts_world[:, j, 0],
        ellipse_pts_world[:, j, 1],
        # ellipse_pts_world[:, j, 2],
        "k-",
        alpha=0.5,
    )
n_eval = ellipse_pts_world.shape[0]
for i in range(0, n_eval):
    ax.plot(
        ellipse_pts_world[i, :, 0],
        ellipse_pts_world[i, :, 1],
        # ellipse_pts_world[i, :, 2],
        "m-",
        alpha=0.1,
    )

ax.dist = 3
# plt.tight_layout()
# ax.set_axis_off()
ax.set_ylim([-50, 0])
ax.axis("equal")
plt.show()
# plt.gcf().savefig(save_path + 'comparison_3.pdf',dpi=1800)
