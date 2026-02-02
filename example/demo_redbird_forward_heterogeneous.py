"""
Redbird - A Diffusion Solver for Diffuse Optical Tomography
Copyright Qianqian Fang, 2018

Forward simulation with heterogeneous domain (box inclusion)

This file is part of Redbird URL: https://github.com/fangq/redbirdpy

INDEX CONVENTION: iso2mesh returns 1-based elem/face indices.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbirdpy as rb
from redbirdpy import forward

import iso2mesh as i2m
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_mesh_slice(ax, cutpos, cutval, facedata, xlabel="x", ylabel="y", **kwargs):
    """Helper to plot mesh slice using tricontourf."""
    facedata_0 = np.asarray(facedata, dtype=int) - 1
    if facedata_0.ndim == 2 and facedata_0.shape[1] == 4:
        tri1 = facedata_0[:, [0, 1, 2]]
        tri2 = facedata_0[:, [0, 2, 3]]
        facedata_0 = np.vstack([tri1, tri2])
    triang = mtri.Triangulation(cutpos[:, 0], cutpos[:, 1], facedata_0)
    tc = ax.tricontourf(triang, cutval, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    return tc


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Mesh generation (iso2mesh)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("== mesh generation (iso2mesh) ...")

# Create a box-like homogeneous domain
boxsize = [60, 50, 40]  # domain size
trisize = 4  # max triangle size on the surface
maxvol = 4  # max tet element volume

tic = time.perf_counter()

# Create a big box
nbox1, fbox1, _ = i2m.meshabox([0, 0, 0], boxsize, trisize)
# Create a box inclusion
nbox2, fbox2, _ = i2m.meshabox([10, 10, 10], [30, 30, 30], trisize)

# Clean the surfaces
nbox1, fbox1 = i2m.removeisolatednode(nbox1, fbox1[:, :3])[:2]
nbox2, fbox2 = i2m.removeisolatednode(nbox2, fbox2[:, :3])[:2]

# Combine the two non-intersecting surfaces
no1, fc1 = i2m.mergemesh(nbox1, fbox1[:, :3], nbox2, fbox2[:, :3])[:2]

# Seed points: inside region 1 (large box) and region 2 (small box)
regionseed = [[1, 1, 1], [11, 11, 11]]

# Generate tetrahedral mesh - outer box: label 1, inner box: label 2
node, elem, _ = i2m.s2m(no1, fc1[:, :3], 1, maxvol, "tetgen", regionseed)

print(f"Mesh generation time: {time.perf_counter() - tic:.4f} seconds")

# cfg.seg is similar to cfg.elemprop in mmclab, but also supports node-based labels
seg = elem[:, 4].astype(int)
elem_4 = elem[:, :4]

cfg = {
    "node": node,
    "elem": elem_4,  # 1-based
    "seg": seg,
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Define settings in cfg
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("== define settings in cfg ...")

# Creating forward simulation data structure cfg
# properties: [mua(1/mm), mus(1/mm), g, n]
# if both mus/g are given, mus'=mus*(1-g) will be used for diffusion, usually set g to 0
cfg["prop"] = np.array(
    [
        [0, 0, 1, 1],  # cfg.prop row-1 is for label 0
        [0.006, 0.8, 0, 1.37],  # label 1 (background domain)
        [0.02, 1, 0, 1.37],  # label 2 (inclusion)
    ]
)

# Default redbird source is a pencil beam, same as mcx/mmc
cfg["srcpos"] = np.array([[25, 25, 0]])  # redbird srcpos can have multiple rows
cfg["srcdir"] = [0, 0, 1]  # srcdir determines how sources are sunken into the mesh

# Redbird detector positions are point-like, directly sampling the output fluence
cfg["detpos"] = np.array(
    [[35, 25, np.max(node[:, 2])]]
)  # redbird detpos can have multiple rows
cfg["detdir"] = [0, 0, -1]  # redbird automatically computes adjoint solutions

# redbird cfg does not need cfg.{nphoton,tstart,tend,tstep,elemprop}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Preprocessing domain
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("== preprocessing domain ...")

# calling rb.meshprep() populates other needed mesh data
# such as cfg.{evol,nvol,face,area,reff,deldotdel,cols,idxsum}

tic = time.perf_counter()
cfg, sd = rb.meshprep(cfg)
print(f"Mesh prep time: {time.perf_counter() - tic:.4f} seconds")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Run simulation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("== run simulation ...")

# Run forward simulation (you can also call rb.run())
tic = time.perf_counter()
detphi, phi = forward.runforward(cfg, sd=sd)
print(f"Forward simulation time: {time.perf_counter() - tic:.4f} seconds")

# rbrunforward returned two outputs:
# detphi - the sampled measurements (fluence, not diffuse reflectance!) at all src/det pairs
# phi - the fluence at all nodes for all sources and detectors (column-dimension)

print(f"detphi shape: {detphi.shape}")
print(f"phi shape: {phi.shape}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Plot results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("== plot results ...")

nn = node.shape[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot forward solution from source (first column)
ax1 = axes[0]
cutpos, cutval, facedata = i2m.qmeshcut(
    cfg["elem"][:, :4],
    cfg["node"][:, :3],
    np.log10(np.abs(phi[:nn, 0]) + 1e-20),
    "y=25",
)[:3]
cutpos_2d = cutpos[:, [0, 2]]  # x, z for y=const slice
tc1 = plot_mesh_slice(
    ax1,
    cutpos_2d,
    cutval,
    facedata,
    xlabel="x (mm)",
    ylabel="z (mm)",
    levels=20,
    cmap="jet",
)
ax1.set_title("Forward solution from source")
plt.colorbar(tc1, ax=ax1, label="log10(fluence)")

# Plot forward solution from detector (second column)
ax2 = axes[1]
cutpos, cutval, facedata = i2m.qmeshcut(
    cfg["elem"][:, :4],
    cfg["node"][:, :3],
    np.log10(np.abs(phi[:nn, 1]) + 1e-20),
    "y=25",
)[:3]
cutpos_2d = cutpos[:, [0, 2]]
tc2 = plot_mesh_slice(
    ax2,
    cutpos_2d,
    cutval,
    facedata,
    xlabel="x (mm)",
    ylabel="z (mm)",
    levels=20,
    cmap="jet",
)
ax2.set_title("Forward solution from detector")
plt.colorbar(tc2, ax=ax2, label="log10(fluence)")

plt.tight_layout()
plt.show()
