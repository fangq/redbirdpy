"""
Redbird - A Diffusion Solver for Diffuse Optical Tomography
Copyright Qianqian Fang, 2018

This example shows explicitly the detailed steps of running a forward
simulation. One can call rb.run or forward.runforward as one-liner alternatives

This file is part of Redbird URL: https://github.com/fangq/redbirdpy

INDEX CONVENTION: iso2mesh returns 1-based elem/face indices.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbirdpy as rb
from redbirdpy import forward, analytical
from redbirdpy.solver import femsolve

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

node, face, elem = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)  # elem is 1-based

nn = node.shape[0]

# Ensure elem and face have correct dimensions
elem_4 = elem[:, :4] if elem.ndim == 2 and elem.shape[1] > 4 else elem
face_3 = face[:, :3] if face.ndim == 2 and face.shape[1] > 3 else face

xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))

cfg = {
    "node": node,
    "elem": elem_4,  # 1-based
    "face": face_3,
    "seg": np.ones(elem_4.shape[0], dtype=int),
    "srcpos": np.c_[xi.flat, yi.flat, np.zeros(xi.size)],
    "detpos": np.c_[xi.flat, yi.flat, 60 * np.ones(xi.size)],
    "srcdir": [0, 0, 1],
    "detdir": [0, 0, -1],
    "prop": np.array([[0, 0, 1, 1], [0.008, 1, 0, 1.37], [0.016, 1, 0, 1.37]]),
    "omega": 0,  # CW mode; set to 2*np.pi*70e6 for frequency domain
}

tic = time.perf_counter()
cfg, sd = rb.meshprep(cfg)
print(f"preparing mesh ... \t{time.perf_counter() - tic:.6f} seconds")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Build LHS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic = time.perf_counter()
deldotdel_mat, delphi = forward.deldotdel(cfg)  # returns tuple (result, delphi)
cfg["deldotdel"] = deldotdel_mat  # store in cfg for later use
Amat = forward.femlhs(cfg, deldotdel_mat)  # use native code
print(f"build LHS using native code ... \t{time.perf_counter() - tic:.6f} seconds")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Build RHS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rhs, loc, bary, optode = forward.femrhs(cfg)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Solve for solutions at all freenodes: Afree*sol=rhs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic = time.perf_counter()
print("solving for the solution ...")
# phi, sflag = forward.femsolve(Amat, rhs, method='cg', tol=1e-8, maxiter=200)
phi, flag = femsolve(Amat, rhs)
print(f"solving forward solutions ... \t{time.perf_counter() - tic:.6f} seconds")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Extract detector readings from the solutions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

detval = forward.femgetdet(phi, cfg, rhs, loc, bary)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Analytical solution
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sid = 12  # 0-based index (13 in MATLAB)

srcloc = cfg["srcpos"][sid, :3]
detloc = cfg["node"]

# CW diffusion analytical solution for semi-infinite medium
phicw = analytical.semi_infinite_cw(
    cfg["prop"][1, 0],  # mua
    cfg["prop"][1, 1] * (1 - cfg["prop"][1, 2]),  # musp = mus * (1-g)
    cfg["prop"][1, 3],  # n_in
    1.0,  # n_out (air)
    srcloc,
    detloc,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Visualization
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: FEM solution
ax1 = axes[0, 0]
cutpos, cutval, facedata = i2m.qmeshcut(
    cfg["elem"][:, :4],
    cfg["node"][:, :3],
    np.log10(np.abs(phi[:nn, sid]) + 1e-20),
    "y=30",
)[:3]
# For y=const slice, use x,z coordinates (columns 0,2 of cutpos)
cutpos_2d = cutpos[:, [0, 2]]  # x, z
tc1 = plot_mesh_slice(
    ax1, cutpos_2d, cutval, facedata, xlabel="x", ylabel="z", levels=20
)
ax1.set_xlim([60, 140])
ax1.set_ylim([0, 60])
ax1.set_title("FEM Solution (log10)")

# Plot 2: Analytical solution
ax2 = axes[0, 1]
cutpos, cutval_ana, facedata = i2m.qmeshcut(
    cfg["elem"][:, :4], cfg["node"][:, :3], np.log10(np.abs(phicw) + 1e-20), "y=30"
)[:3]
cutpos_2d = cutpos[:, [0, 2]]
tc2 = plot_mesh_slice(
    ax2, cutpos_2d, cutval_ana, facedata, xlabel="x", ylabel="z", levels=20
)
ax2.set_xlim([60, 140])
ax2.set_ylim([0, 60])
ax2.set_title("Analytical Solution (log10)")

# Plot 3: Difference
ax3 = axes[1, 0]
dd = np.log10(np.abs(phi[:nn, sid]) + 1e-20) - np.log10(np.abs(phicw) + 1e-20)
cutpos, cutval_diff, facedata = i2m.qmeshcut(
    cfg["elem"][:, :4], cfg["node"][:, :3], dd, "y=30"
)[:3]
cutpos_2d = cutpos[:, [0, 2]]
tc3 = plot_mesh_slice(
    ax3, cutpos_2d, cutval_diff, facedata, xlabel="x", ylabel="z", levels=20
)
ax3.set_xlim([60, 140])
ax3.set_ylim([0, 60])
ax3.set_title("Difference (FEM - Analytical)")
plt.colorbar(tc3, ax=ax3)

# Plot 4: Histogram of difference
ax4 = axes[1, 1]
ax4.hist(dd.flatten(), bins=100)
ax4.set_title("Histogram of Difference")
ax4.set_xlabel("log10(FEM) - log10(Analytical)")
ax4.set_ylabel("Count")

plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Test add-noise function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2, ax = plt.subplots(figsize=(10, 6))

dist = rb.getdistance(cfg["srcpos"], cfg["detpos"])
ax.plot(
    dist.flatten(), np.log10(np.abs(detval.flatten()) + 1e-20), "b.", label="Original"
)

# Add noise to detector values
# addnoise signature: addnoise(data, noiselvl, offset) where noiselvl is SNR in dB
newdata = rb.addnoise(detval, 110, 40)
ax.plot(
    dist.flatten(),
    np.log10(np.abs(newdata.flatten()) + 1e-20),
    "r.",
    label="With Noise",
)

ax.set_xlabel("Source-Detector Distance (mm)")
ax.set_ylabel("log10(Detector Value)")
ax.set_title("Detector Readings vs Distance")
ax.legend()

plt.show(block=False)
