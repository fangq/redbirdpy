"""
Redbird - A Diffusion Solver for Diffuse Optical Tomography
Copyright Qianqian Fang, 2018

Continuous-Wave (CW) reconstruction of absorption (mua) target
(streamlined version by calling rb.run with recon structure)

This file is part of Redbird URL: https://github.com/fangq/redbirdpy

INDEX CONVENTION: iso2mesh returns 1-based elem/face indices.
tsearchn returns 1-based element indices to match MATLAB convention.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbirdpy as rb
import iso2mesh as i2m
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


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

s0 = [70, 50, 20]

# Create bounding box with spherical inclusion
nobbx, fcbbx, _ = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)  # elem/face are 1-based
nosp, fcsp, _ = i2m.meshasphere(s0, 5, 1)
no, fc = i2m.mergemesh(nobbx, fcbbx, nosp, fcsp[:, :3])

node, elem, _ = i2m.s2m(
    no, fc[:, :3], 1, 40, "tetgen", [[41, 1, 1], s0]
)  # elem is 1-based

xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))

cfg0 = {
    "node": node,
    "elem": elem[:, :4],  # 1-based
    "seg": elem[:, 4].astype(int),
    "srcpos": np.c_[xi.flat, yi.flat, np.zeros(xi.size)],
    "detpos": np.c_[xi.flat, yi.flat, 60 * np.ones(xi.size)],
    "srcdir": [0, 0, 1],
    "detdir": [0, 0, -1],
    "prop": np.array([[0, 0, 1, 1], [0.008, 1, 0, 1.37], [0.016, 1, 0, 1.37]]),
    "omega": 0,
}

cfg = {k: v.copy() if hasattr(v, "copy") else v for k, v in cfg0.items()}
cfg0 = rb.meshprep(cfg0)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Run forward for the heterogeneous domain
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

detphi0 = rb.run(cfg0)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Reset the domain to a homogeneous medium for recon
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create forward mesh for reconstruction
node, face, elem = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)  # 1-based
cfg = rb.setmesh(cfg, node, elem, cfg["prop"], np.ones(node.shape[0], dtype=int))

sd = rb.sdmap(cfg)

# Create coarse reconstruction mesh
recon = {}
recon["node"], _, recon["elem"] = i2m.meshabox(
    [40, 0, 0], [160, 120, 60], 20
)  # 1-based
recon["mapid"], recon["mapweight"] = i2m.tsearchn(
    recon["node"], recon["elem"], cfg["node"]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %  Streamlined reconstruction
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Initialize reconstruction to homogeneous (label=1)
recon["prop"] = np.tile(cfg["prop"][1, :], (recon["node"].shape[0], 1))
cfg["prop"] = np.tile(cfg["prop"][1, :], (cfg["node"].shape[0], 1))
cfg.pop("seg", None)

# Run streamlined image reconstruction
newrecon, resid, newcfg = rb.run(cfg, recon, detphi0, sd, lambda_=1e-4)[:3]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %  Plotting results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot z=20 slice
ax1 = axes[0]
cutpos, cutval, facedata = i2m.qmeshcut(
    newcfg["elem"][:, :4], newcfg["node"][:, :3], newcfg["prop"][:, 0], "z=20"
)[:3]
# For z=const slice, use x,y coordinates (columns 0,1 of cutpos)
cutpos_2d = cutpos[:, :2]  # x, y
tc1 = plot_mesh_slice(
    ax1,
    cutpos_2d,
    cutval,
    facedata,
    xlabel="x (mm)",
    ylabel="y (mm)",
    levels=20,
    cmap="jet",
)
ax1.set_title("Reconstructed mua (z=20)")
plt.colorbar(tc1, ax=ax1, label="mua (1/mm)")

# Plot x=70 slice
ax2 = axes[1]
cutpos, cutval, facedata = i2m.qmeshcut(
    newcfg["elem"][:, :4], newcfg["node"][:, :3], newcfg["prop"][:, 0], "x=70"
)[:3]
# For x=const slice, use y,z coordinates (columns 1,2 of cutpos)
cutpos_2d = cutpos[:, 1:3]  # y, z
tc2 = plot_mesh_slice(
    ax2,
    cutpos_2d,
    cutval,
    facedata,
    xlabel="y (mm)",
    ylabel="z (mm)",
    levels=20,
    cmap="jet",
)
ax2.set_title("Reconstructed mua (x=70)")
plt.colorbar(tc2, ax=ax2, label="mua (1/mm)")

plt.tight_layout()

# Plot residual convergence
fig2, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(np.arange(1, len(resid) + 1), resid, "b-o")
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual")
ax.set_title("Reconstruction Convergence")
ax.grid(True)

plt.show(block=(len(sys.argv) == 1))
