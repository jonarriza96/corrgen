import numpy as np

import matplotlib as plt
import subprocess


def get_corrgen_path():
    corrgen_path = subprocess.run(
        "echo $CORRGEN_PATH", shell=True, capture_output=True, text=True
    ).stdout.strip("\n")
    return corrgen_path


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
