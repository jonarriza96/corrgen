import numpy as np
# import casadi as cs

import matplotlib as plt
import subprocess

# def convert_curve_to_casadi_func(T, coeffs, order, n, dim):
#     curve_f = spline_curve(n=n, order=order, dim=dim)
#     xi_grid = np.hstack([0, np.cumsum(T)])
#     xi = cs.MX.sym("xi")
#     coeffs_vec = np.array(coeffs).reshape((order + 1) * n, dim)
#     p_cs = curve_f(xi=xi, xi_grid=xi_grid, coeffs=coeffs_vec)["p"]
#     if dim == 2:
#         p_cs = cs.vertcat(p_cs, 0)
#     v_cs = cs.jacobian(p_cs, xi)
#     a_cs = cs.jacobian(v_cs, xi)
#     j_cs = cs.jacobian(a_cs, xi)
#     s_cs = cs.jacobian(j_cs, xi)
#     curve_cs = {
#         "f_p": cs.Function(
#             "f_p",
#             [xi],
#             [p_cs, v_cs, a_cs, j_cs, s_cs],
#             ["xi"],
#             ["p", "v", "a", "j", "s"],
#         )
#     }

#     return curve_cs


# def spline_curve(n, order, dim):
#     """Generates spline curve of bezier segments

#     Args:
#         n (int): Number of segments in the spline
#         order (int): Order of the spline
#         dim (int): Dimensions of the spline (2D or 3D)

#     Returns:
#         f: casadi function that evaluates the spline and derivatives
#     """

#     xi = cs.MX.sym("xi")
#     coeffs = cs.MX.sym("coeffs", (order + 1) * n, dim)
#     xi_grid = cs.MX.sym("xi_grid", n + 1)

#     # find segment and  path parameter for the section
#     # xi_sat = cs.if_else(xi <= 0.0, 1e-16, xi)
#     # xi_sat = cs.if_else(xi_sat >= xi_grid[-1], xi_grid[-1], xi_sat)
#     XI = xi * xi_grid[-1]
#     ind = cs.find(cs.sign(xi_grid - XI) + 1) - 1
#     ind = cs.if_else(ind < 0, 0, ind)
#     xik = XI - xi_grid[ind]

#     # unify x,y,z dimensions
#     p = []
#     v = []
#     a = []
#     j = []
#     s = []
#     fxD = bezier_curve(order)
#     for d in range(dim):  # loop all dimensions (x,y,z)
#         # get coefficients for dimension d and segment ind
#         coeffsD = []
#         for cf_k in range(order + 1):
#             coeffsD = cs.vertcat(
#                 coeffsD, coeffs[(order + 1) * ind + cf_k, d]
#             )  # coeffs[4 * ind : 4 * (ind + 1), d]

#         # evaluate
#         pD = fxD(xi=xik, coeffs=coeffsD)["p"]

#         p = cs.vertcat(p, pD)

#     fX = cs.Function(
#         "fP", [xi, xi_grid, coeffs], [p], ["xi", "xi_grid", "coeffs"], ["p"]
#     )
#     return fX


# def bezier_curve(order):
#     t = cs.MX.sym("xi")  # parametric variable
#     # T = cs.MX.sym("T")  # max parametric value
#     a = cs.MX.sym("a", order + 1, 1)  # coefficients of polynomial

#     # polynomial
#     if order == 3:
#         x = a[3] * t**3 + a[2] * t**2 + a[1] * t + a[0]
#     elif order == 5:
#         x = a[5] * t**5 + a[4] * t**4 + a[3] * t**3 + a[2] * t**2 + a[1] * t + a[0]

#     # control points
#     # r0 = a[0]
#     # r1 = 1 / 3 * (a[1] * T + 3 * a[0])
#     # r2 = 1 / 3 * (a[2] * T**2 + 2 * a[1] * T + 3 * a[0])
#     # r3 = a[3] * T**3 + a[2] * T**2 + a[1] * T + a[0]
#     # r = cs.vertcat(r0, r1, r2, r3)

#     # casadi functions
#     fxD = cs.Function(
#         "f_spline_with_derivatives", [t, a], [x], ["xi", "coeffs"], ["p"]
#     )  # function that evaluates Bezier polynomial and derivatives

#     # fr = cs.Function(
#     #     "f_spline_controlpoints", [a, T], [r], ["coeffs", "T"], ["cpt"]
#     # )  # function that evaluates a polynomial

#     return fxD

def axis_equal(X, Y, Z, ax=None):
    """
    Sets axis bounds to "equal" according to the limits of X,Y,Z.
    If axes are not given, it generates and labels a 3D figure.

    Args:
        X: Vector of points in coord. x
        Y: Vector of points in coord. y
        Z: Vector of points in coord. z
        ax: Axes to be modified

    Returns:
        ax: Axes with "equal" aspect


    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - 1.2 * max_range, mid_x + 1.2 * max_range)
    ax.set_ylim(mid_y - 1.2 * max_range, mid_y + 1.2 * max_range)
    ax.set_zlim(mid_z - 1.2 * max_range, mid_z + 1.2 * max_range)

    return ax

def get_corrgen_path():
    corrgen_path = subprocess.run(
        "echo $CORRGEN_PATH", shell=True, capture_output=True, text=True
    ).stdout.strip("\n")
    return corrgen_path
