"""
Redbird - A Diffusion Solver for Diffuse Optical Tomography
Copyright Qianqian Fang, 2018

In this example, we show the most basic usage of Redbird.

This file is part of Redbird URL: https://github.com/fangq/redbirdpy
"""

import numpy as np
import time
import sys
import os

# For mesh generation and plotting
try:
    from iso2mesh import meshabox, plotmesh
except ImportError:
    raise ImportError("iso2mesh is required. Install with: pip install iso2mesh")

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import redbird
import redbird as rb

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   prepare simulation input
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create mesh
node, face, elem = meshabox([40, 0, 0], [160, 120, 60], 10)

# Create source/detector grid
xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))
xi = xi.flatten()
yi = yi.flatten()

cfg = {
    "node": node,
    "face": face,
    "elem": elem,
    "seg": np.ones(elem.shape[0], dtype=int),
    "srcpos": np.column_stack([xi, yi, np.zeros(len(xi))]),
    "detpos": np.column_stack([xi, yi, 60 * np.ones(len(xi))]),
    "srcdir": [0, 0, 1],
    "detdir": [0, 0, -1],
    "prop": np.array([[0, 0, 1, 1], [0.008, 1, 0, 1.37], [0.016, 1, 0, 1.37]]),
    "omega": 2 * np.pi * 70e6,
}

# Prepare mesh
tic = time.perf_counter()
cfg, sd = rb.meshprep(cfg)
print(f"preparing mesh ... \t{time.perf_counter() - tic:.6f} seconds")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Solve the forward problem
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic = time.perf_counter()
detphi, phi = rb.run(cfg)
print(f"forward solution ... \t{time.perf_counter() - tic:.6f} seconds")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Visualization of fluence
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Source index (MATLAB uses 1-based, Python uses 0-based)
src_idx = 12  # This is source 13 in MATLAB

# Compute log10 of absolute fluence
phi_log = np.log10(np.abs(phi[:, src_idx]) + 1e-20)

# Create node array with fluence as 4th column for plotmesh
node_with_data = np.column_stack([cfg["node"], phi_log])

# plotmesh with slice 'y>60'
plotmesh(node_with_data, cfg["elem"], "y>60")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Print summary
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print(f"\nResults summary:")
print(f'  Nodes: {cfg["node"].shape[0]}')
print(f'  Elements: {cfg["elem"].shape[0]}')
print(f'  Sources: {cfg["srcpos"].shape[0]}')
print(f'  Detectors: {cfg["detpos"].shape[0]}')
print(f"  phi shape: {phi.shape}")
print(f"  detphi shape: {detphi.shape}")
print(
    f"\nDetector amplitude range: {np.abs(detphi).min():.2e} - {np.abs(detphi).max():.2e}"
)
print(
    f"Detector phase range: {np.angle(detphi).min():.2f} - {np.angle(detphi).max():.2f} rad"
)
