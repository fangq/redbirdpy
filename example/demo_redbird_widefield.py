"""
Redbird - A Diffusion Solver for Diffuse Optical Tomography
Copyright Qianqian Fang, 2018

In this example, we show the usage of Redbird with wide-field
(planar) sources and detectors.

This file is part of Redbird URL: https://github.com/fangq/redbirdpy

INDEX CONVENTION: iso2mesh returns 1-based elem/face indices.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbird as rb
from redbird import forward
from redbird.solver import femsolve
import iso2mesh as i2m
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata

def plot_mesh_slice(ax, cutpos, cutval, facedata, xlabel="x", ylabel="y", **kwargs):
    """Helper to plot mesh slice using tricontourf."""
    # facedata is 1-based from iso2mesh, convert to 0-based
    facedata_0 = np.asarray(facedata, dtype=int) - 1

    # If facedata has 4 columns (quads), convert to triangles
    if facedata_0.ndim == 2 and facedata_0.shape[1] == 4:
        # Split each quad into 2 triangles: [0,1,2] and [0,2,3]
        tri1 = facedata_0[:, [0, 1, 2]]
        tri2 = facedata_0[:, [0, 2, 3]]
        facedata_0 = np.vstack([tri1, tri2])

    # Create triangulation
    triang = mtri.Triangulation(cutpos[:, 0], cutpos[:, 1], facedata_0)
    tc = ax.tricontourf(triang, cutval, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    return tc

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   prepare simulation input
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create regular grid mesh using meshabox instead of meshgrid6
# meshgrid6 has API issues, use meshabox with small element size
node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 2)

# Reorient elements if needed
elem_reoriented = i2m.meshreorient(node[:, :3], elem[:, :4])
if elem_reoriented is not None and not isinstance(elem_reoriented, tuple):
    elem = elem_reoriented

nn = node.shape[0]

# Handle elem shape
elem_4 = elem[:, :4] if elem.ndim == 2 and elem.shape[1] > 4 else elem

cfg = {
    "node": node,
    "elem": elem_4,  # 1-based
    "face": face[:, :3] if face.ndim == 2 and face.shape[1] > 3 else face,
    "seg": np.ones(elem.shape[0], dtype=int),
    # Planar source
    "srctype": "planar",
    "srcpos": [9.5, 9.5, 0],
    "srcparam1": [40, 0, 0, 0],
    "srcparam2": [0, 40, 0, 0],
    "srcdir": [0, 0, 1],
    # Planar detector
    "dettype": "planar",
    "detpos": [10, 10, 30],
    "detparam1": [40, 0, 0, 0],
    "detparam2": [0, 40, 0, 0],
    "detdir": [0, 0, -1],
    # Optical properties
    "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37]]),
    "omega": 0,
    "srcweight": 2,
}

z0 = 1.0 / (cfg["prop"][1, 0] + cfg["prop"][1, 1] * (1 - cfg["prop"][1, 2]))

cfg = rb.meshprep(cfg)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Build LHS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

deldotdel_mat, _ = forward.deldotdel(cfg)
cfg["deldotdel"] = deldotdel_mat
Amat = forward.femlhs(cfg, deldotdel_mat)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Build RHS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Debug: check what getoptodes returns
from redbird.utility import getoptodes
optsrc, optdet, widesrc, widedet = getoptodes(cfg, "")
print(f"optsrc shape: {optsrc.shape if optsrc is not None else None}")
print(f"optdet shape: {optdet.shape if optdet is not None else None}")
print(f"widesrc shape: {widesrc.shape if widesrc is not None else None}")
print(f"widedet shape: {widedet.shape if widedet is not None else None}")

rhs, loc, bary, optode = forward.femrhs(cfg)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Solve for solutions at all freenodes: Afree*sol=rhs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic = time.perf_counter()
print("solving for the solution ...")
phi, flag = femsolve(Amat, rhs)
phi = np.real(phi)
phi[phi < 0] = 0
print(f"solving time: {time.perf_counter() - tic:.4f} seconds")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Extract detector readings from the solutions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

detval = forward.femgetdet(phi, cfg, rhs, loc, bary)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   MCX comparison (optional, requires pmcx)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try:
    import pmcx
    HAS_MCX = True
except ImportError:
    HAS_MCX = False
    print("pmcx not available, skipping MCX comparison")

fcw = None
if HAS_MCX:
    xcfg = {
        "nphoton": int(1e8),
        "vol": np.ones((60, 60, 30), dtype=np.uint8),
        "srcdir": [0, 0, 1, 0],
        "gpuid": 1,
        "autopilot": 1,
        "prop": cfg["prop"].tolist(),
        "tstart": 0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "seed": 99999,
        "issrcfrom0": 0,
        # Uniform planar source outside the volume
        "srctype": "planar",
        "srcpos": [10, 10, 0],
        "srcparam1": [40, 0, 0, 0],
        "srcparam2": [0, 40, 0, 0],
    }

    result = pmcx.run(xcfg)
    fcw = result["flux"] * xcfg["tstep"]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Visualization
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig, axes = plt.subplots(1, 2 if HAS_MCX else 1, figsize=(12 if HAS_MCX else 6, 5))

if HAS_MCX and fcw is not None:
    # MCX solution
    ax1 = axes[0]
    mcx_slice = np.rot90(np.log10(np.abs(cfg["srcweight"] * fcw[:, 29, :].squeeze()) + 1e-20))
    im1 = ax1.imshow(mcx_slice, extent=[0, 60, 0, 30], aspect="auto", origin="lower")
    ax1.set_xlim([0, 60])
    ax1.set_ylim([0, 30])
    ax1.set_title("MCX Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    plt.colorbar(im1, ax=ax1)
    cl = [mcx_slice.min(), mcx_slice.max()]
    ax2 = axes[1]
else:
    ax2 = axes if not HAS_MCX else axes[0]
    cl = None

# Plot Redbird solution on slice x=30
cutpos, cutval, facedata = i2m.qmeshcut(
    cfg["elem"][:, :4], cfg["node"][:, :3],
    np.log10(np.abs(phi[:nn, 0]) + 1e-20), "x=29.5"
)[:3]
# For x=const slice, use y,z coordinates (columns 1,2 of cutpos)
cutpos_2d = cutpos[:, 1:3]  # y, z
tc = plot_mesh_slice(ax2, cutpos_2d, cutval, facedata, xlabel="y", ylabel="z", levels=20, cmap="viridis")
ax2.set_title("Redbird Solution (x=30)")
plt.colorbar(tc, ax=ax2)

plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Contour Comparison
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if HAS_MCX and fcw is not None:
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Contour levels must be increasing
    clines = np.arange(-5.0, 1.0, 0.5)
    xi, yi = np.meshgrid(np.arange(0.5, 60, 1), np.arange(0.5, 30, 1))

    # Interpolate Redbird solution to regular grid
    cutpos, cutvalue, facedata = i2m.qmeshcut(
        cfg["elem"][:, :4], cfg["node"][:, :3], phi[:nn, 0], "x=29.5"
    )[:3]
    facedata_int = np.asarray(facedata, dtype=int)
    vphi = griddata(
        (cutpos[:, 1], cutpos[:, 2]), cutvalue, (xi + 0.5, yi), method="linear"
    )

    # Redbird contours
    cs1 = ax.contour(xi, yi, np.log10(np.abs(vphi) + 1e-20), clines, colors="r", linewidths=2)
    ax.clabel(cs1, inline=True, fontsize=8)

    # MCX contours
    cwf = fcw[29, :, :].squeeze().T
    cs2 = ax.contour(
        xi, yi, np.log10(cfg["srcweight"] * np.abs(cwf) + 1e-20), clines, colors="b", linewidths=2
    )
    ax.clabel(cs2, inline=True, fontsize=8)

    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Contour Comparison: Redbird (red) vs MCX (blue)")

    # Create legend with proxy artists
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', linewidth=2, label='Redbird'),
        Line2D([0], [0], color='b', linewidth=2, label='MCX')
    ]
    ax.legend(handles=legend_elements, loc="upper right")

plt.show()