"""
Visual test for create_u_shaped_scalar_profile.

Layout (2x2 figure):
  [0,0]  3D scatter of all grid points, coloured and sized by scalar value
  [0,1]  XY slice at central Z
  [1,0]  XZ slice at central Y
  [1,1]  YZ slice at central X
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Allow running from the workspace root or from inside manual_tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from helper_module import create_u_shaped_scalar_profile  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — edit these to explore different shapes
# ---------------------------------------------------------------------------
ECM_AGENTS_PER_DIR = (21, 21, 21)   # (Nx, Ny, Nz)
COORDS_BOUNDARIES  = (10.0, -10.0, 10.0, -10.0, 10.0, -10.0)  # (X_POS,X_NEG, Y_POS,Y_NEG, Z_POS,Z_NEG)

KWARGS = dict(
    loaded_face="z_pos",
    u_axes=("x","y"),
    max_scalar=1.0,
    side_surface_fraction=0.0,  # 0.0 = no side surface, 1.0 = side surface extends to the center
    center_reach=0.30,
    side_reach=0.15,
    lateral_power=1.3,
    decay_length=0.60,
    decay_power=2.0,
    front_smoothing=0.08,
)
# ---------------------------------------------------------------------------

Nx, Ny, Nz = map(int, ECM_AGENTS_PER_DIR)

scalar_flat = create_u_shaped_scalar_profile(ECM_AGENTS_PER_DIR, COORDS_BOUNDARIES, **KWARGS)
print(f"Generated scalar profile with shape {scalar_flat.shape} and range [{scalar_flat.min():.3f}, {scalar_flat.max():.3f}]")
# Reshape to 3D: index order is [i, j, k] = [x, y, z].
# order='C' is the exact inverse of the ravel(order='C') inside the function.
scalar_3d = scalar_flat.reshape((Nx, Ny, Nz), order="C")

# Physical coordinates derived from COORDS_BOUNDARIES so that negative
# (or otherwise non-unit) domains are represented correctly on the axes.
X_POS, X_NEG, Y_POS, Y_NEG, Z_POS, Z_NEG = map(float, COORDS_BOUNDARIES)
xs = np.linspace(X_NEG, X_POS, Nx)
ys = np.linspace(Y_NEG, Y_POS, Ny)
zs = np.linspace(Z_NEG, Z_POS, Nz)

# Build meshgrid matching linear index: i=x, j=y, k=z
XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")

# Flat arrays for the scatter
x_flat = XX.ravel()
y_flat = YY.ravel()
z_flat = ZZ.ravel()
v_flat = scalar_3d.ravel()

# Normalise scalar to [0, 1] for colour / size mapping
v_min, v_max = v_flat.min(), v_flat.max()
v_norm = (v_flat - v_min) / (v_max - v_min + 1e-12)

cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=v_min, vmax=v_max)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 11))
fig.suptitle("U-shaped scalar profile — create_u_shaped_scalar_profile()", fontsize=13)

# ---- [0,0]  3D scatter -------------------------------------------------------
ax3d = fig.add_subplot(2, 2, 1, projection="3d")

scatter_size = 4 + 40 * v_norm          # min 4, max 44 pt²
scatter_alpha = 0.15 + 0.75 * v_norm    # semi-transparent for low values

sc = ax3d.scatter(
    x_flat, y_flat, z_flat,
    c=v_flat,
    s=scatter_size,
    alpha=None,              # per-point alpha not supported natively; use uniform
    cmap=cmap,
    norm=norm,
    linewidths=0,
    depthshade=True,
)
ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
ax3d.set_title("3D scatter (all points)")
fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1, label="scalar")

# ---- helper: 2D slice plot ---------------------------------------------------
def _slice_plot(ax, data2d, h_vals, v_vals, h_label, v_label, title):
    """Plot a 2D slice as a filled contour / pcolormesh."""
    im = ax.pcolormesh(
        h_vals, v_vals, data2d.T,   # .T because pcolormesh is (x, y) but data is [h, v]
        cmap=cmap, norm=norm, shading="auto",
    )
    fig.colorbar(im, ax=ax, label="scalar")
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    ax.set_aspect("equal")

# ---- [0,1]  XY slice at central Z -------------------------------------------
ax_xy = fig.add_subplot(2, 2, 2)
k_mid = Nz // 2
_slice_plot(
    ax_xy,
    scalar_3d[:, :, k_mid],   # shape (Nx, Ny)
    xs, ys,
    "X", "Y",
    f"XY slice  (Z = {zs[k_mid]:.2f})",
)

# ---- [1,0]  XZ slice at central Y -------------------------------------------
ax_xz = fig.add_subplot(2, 2, 3)
j_mid = Ny // 2
_slice_plot(
    ax_xz,
    scalar_3d[:, j_mid, :],   # shape (Nx, Nz)
    xs, zs,
    "X", "Z",
    f"XZ slice  (Y = {ys[j_mid]:.2f})",
)

# ---- [1,1]  YZ slice at central X -------------------------------------------
ax_yz = fig.add_subplot(2, 2, 4)
i_mid = Nx // 2
_slice_plot(
    ax_yz,
    scalar_3d[i_mid, :, :],   # shape (Ny, Nz)
    ys, zs,
    "Y", "Z",
    f"YZ slice  (X = {xs[i_mid]:.2f})",
)

plt.tight_layout()
plt.show()
