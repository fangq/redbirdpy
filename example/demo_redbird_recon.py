"""
Redbird - A Diffusion Solver for Diffuse Optical Tomography
Copyright Qianqian Fang, 2018

Continuous-Wave (CW) reconstruction of absorption (mua) target
(explicit iterative loop to show internal steps of rbrun)

This file is part of Redbird URL: https://github.com/fangq/redbirdpy

INDEX CONVENTION: iso2mesh returns 1-based elem/face indices.
tsearchn returns 1-based element indices to match MATLAB convention.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import iso2mesh as i2m
import redbird as rb
from redbird import forward
from redbird.recon import reginv, matreform

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   prepare simulation input
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s0 = [70, 50, 20]

nobbx, fcbbx, _ = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)  # elem/face are 1-based
nosp, fcsp, _ = i2m.meshasphere(s0, 5, 1)
no, fc = i2m.mergemesh(nobbx, fcbbx, nosp, fcsp[:, :3])

node, elem, _ = i2m.s2m(
    no, fc[:, :3], 1, 10, "tetgen", [[41, 1, 1], s0]
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

node, face, elem = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)  # 1-based
cfg = rb.setmesh(cfg, node, elem, cfg["prop"], np.ones(node.shape[0], dtype=int))

sd = rb.sdmap(cfg)

# create coarse reconstruction mesh
recon = {}
recon["node"], _, recon["elem"] = i2m.meshabox(
    [40, 0, 0], [160, 120, 60], 20
)  # 1-based
recon["mapid"], recon["mapweight"] = i2m.tsearchn(
    recon["node"], recon["elem"], cfg["node"]
)  # mapid is 0-based

# Check for NaN (points outside recon mesh)
nan_count = np.isnan(recon["mapid"]).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} forward nodes outside recon mesh")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Initialize properties for reconstruction
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxiter = 10
resid = np.zeros(maxiter)

# initialize reconstruction to homogeneous (label=1)
recon["prop"] = np.tile(cfg["prop"][1, :], (recon["node"].shape[0], 1))
cfg["prop"] = np.tile(cfg["prop"][1, :], (cfg["node"].shape[0], 1))
cfg.pop("seg", None)

# prepare mesh for forward solver
cfg = rb.meshprep(cfg)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %  Explicit iterative reconstruction
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i in range(maxiter):
    tic = time.perf_counter()

    # run forward on forward mesh
    detphi, phi = forward.runforward(cfg, sd=sd, method='blqmr', rhsblock=4, nthread=4)
    # build Jacobian for mua (cfg["elem"] is 1-based, jac handles conversion internally)
    # Jmua shape: (nsd, nn_forward)
    Jmua, _ = forward.jac(sd, phi, cfg["deldotdel"], cfg["elem"], cfg["evol"])

    # remap Jacobian to coarse recon mesh
    Jmua_recon = i2m.meshremap(
        Jmua.T,
        recon["mapid"],
        recon["mapweight"],
        recon["elem"],
        recon["node"].shape[0],
    ).T

    # create inverse problem (real-valued formulation)
    Jmua_recon, misfit, _ = matreform(
        Jmua_recon, detphi0.flatten(), detphi.flatten(), "real"
    )

    # store residual
    resid[i] = np.sum(np.abs(misfit))

    # normalize Jacobian
    blockscale = 1.0 / np.sqrt(np.sum(Jmua_recon**2))
    Jmua_recon = Jmua_recon * blockscale

    # solve for update on recon mesh
    dmu_recon = reginv(Jmua_recon, misfit, 1e-3)

    # de-normalize
    dmu_recon = dmu_recon * blockscale

    # update recon mesh mua
    recon["prop"][:, 0] = recon["prop"][:, 0] + dmu_recon

    # interpolate update to forward mesh
    # meshinterp: interpolate from recon nodes to forward nodes
    cfg["prop"] = i2m.meshinterp(
        recon["prop"], recon["mapid"], recon["mapweight"], recon["elem"], cfg["prop"]
    )

    elapsed = time.perf_counter() - tic
    print(
        f"iter [{i+1:4d}]: residual={resid[i]:.6e}, relres={resid[i]/resid[0]:.6e} (time={elapsed:.4f} s)"
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %  Plotting results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i2m.plotmesh(np.c_[cfg["node"], cfg["prop"][:, 0]], cfg["elem"], "z==20")
i2m.plotmesh(np.c_[cfg["node"], cfg["prop"][:, 0]], cfg["elem"], "x==70")
