"""
Redbird - Python toolbox for Diffuse Optical Tomography

A Python translation of the Redbird MATLAB toolbox for forward and inverse
modeling of diffuse optical tomography (DOT) and near-infrared spectroscopy (NIRS).

This toolbox provides:
- Forward modeling using Finite Element Method (FEM) for the diffusion equation
- Inverse reconstruction using Gauss-Newton methods with Tikhonov regularization
- Multi-spectral analysis for chromophore concentration estimation
- Support for both continuous-wave (CW) and frequency-domain (FD) measurements

IMPORTANT: This toolbox uses 1-based indexing for mesh elements (node, elem, face)
to maintain compatibility with the MATLAB version and iso2mesh conventions.
When interfacing with numpy arrays (0-based), conversion is handled internally.

Modules:
    forward: Forward modeling functions (FEM solver, Jacobian computation)
    recon: Reconstruction algorithms (Gauss-Newton, regularization)
    utility: Mesh utilities, source/detector handling, data processing
    property: Optical property management, extinction coefficients

Dependencies:
    - numpy, scipy
    - iso2mesh (pyiso2mesh): https://github.com/NeuroJSON/pyiso2mesh

Example:

import redbird as rb
import numpy as np
from iso2mesh import meshabox

# Create mesh using iso2mesh (returns 1-based indices)
node, face, elem = meshabox([0,0,0], [60,60,30], 5)

cfg = {
     'node': node,
     'elem': elem,
     'prop': [[0,0,1,1], [0.01, 1, 0, 1.37]],
     'srcpos': [30, 30, 0],
     'srcdir': [0, 0, 1],
     'detpos': [30, 40, 0],
     'detdir': [0, 0, 1],
     'seg': elem.shape[0],
     'omega': 0
}
cfg, sd = rb.utility.meshprep(cfg)
detval, phi = rb.forward.runforward(cfg)


Author: Translated from Redbird MATLAB toolbox by Qianqian Fang (q.fang <at> neu.edu)
License: GPL version 3
"""

__version__ = "0.3.1"
__author__ = "Qianqian Fang"

import numpy as np

from . import forward
from . import recon
from . import utility
from . import property
from . import solver

# Re-export all public functions from submodules
from .forward import *
from .recon import *
from .utility import *
from .property import *
from .solver import *

# Combine all exports
__all__ = (
    forward.__all__
    + solver.__all__
    + recon.__all__
    + utility.__all__
    + property.__all__
    + ["run", "forward", "recon", "utility", "property", "solver"]
)


# Main entry point (similar to rbrun in MATLAB)
def run(cfg, recon_cfg=None, detphi0=None, sd=None, **kwargs):
    """
    Main entry point for Redbird - runs forward or inverse modeling.

    If only cfg is provided, runs forward simulation.
    If recon_cfg and detphi0 are provided, runs reconstruction.

    Parameters
    ----------
    cfg : dict
        Forward simulation configuration
    recon_cfg : dict, optional
        Reconstruction configuration
    detphi0 : ndarray, optional
        Measured data for reconstruction
    sd : ndarray or dict, optional
        Source-detector mapping
    mode : {'image', 'bulk', 'seg'}, kwarg, default 'image'
        Reconstruction granularity (ignored for forward-only runs).
        - 'image': per-node reconstruction (the default).
        - 'bulk':  fit a single bulk mua/musp by collapsing recon.seg into one
                   segment. Useful as a Stage-1 fast sanity check before the
                   full per-node image recon.
        - 'seg':   per-segment reconstruction driven by recon.seg / cfg.seg.
        Port of the mode switch in redbird-m/matlab/rbrun.m.
    **kwargs : dict
        Additional options passed to runforward or runrecon

    Returns
    -------
    Results from runforward (if forward only) or runrecon (if reconstruction)
    """
    if recon_cfg is None:
        return runforward(cfg, **kwargs)

    if detphi0 is None:
        raise ValueError("detphi0 is required for reconstruction")

    # mode handling is opt-in: when omitted, leave (cfg, recon) untouched so
    # existing callers that pre-populate recon.prop / cfg.seg behave exactly
    # as before. Pass mode='bulk' / 'image' to trigger the rbrun.m-style
    # reseeding flow used by the two-stage MC recon demo.
    mode = kwargs.pop("mode", None)
    if mode is not None:
        cfg, recon_cfg = _apply_run_mode(cfg, recon_cfg, mode)

    return runrecon(cfg, recon_cfg, detphi0, sd, **kwargs)


def _apply_run_mode(cfg: dict, recon: dict, mode: str):
    """Apply 'mode' switch to (cfg, recon) before invoking runrecon.

    Port of the rbrun.m mode block (matlab/rbrun.m lines 129-194). Reseeds
    recon.seg / cfg.seg and recon.prop / cfg.prop from recon.bulk so the
    Gauss-Newton driver in runrecon can start from a uniform initial guess.

    - mode='bulk': collapse all unknowns into a single bulk segment by
      setting recon.seg = ones(Nnode) (or cfg.seg if recon has no mesh).
    - mode='image': drop any pre-existing recon.seg / cfg.seg so each
      node carries its own unknown; rebuild a per-node recon.prop and
      per-forward-node cfg.prop from recon.bulk.
    - mode='seg': per-segment recon driven by cfg.seg (no change to seg).
    """
    import copy as _copy

    if mode not in ("bulk", "seg", "image"):
        return cfg, recon

    cfg = _copy.copy(cfg)
    recon = _copy.copy(recon)

    if mode == "bulk":
        if "node" in recon:
            recon["seg"] = np.ones(np.asarray(recon["node"]).shape[0], dtype=int)
        else:
            cfg["seg"] = np.ones(np.asarray(cfg["node"]).shape[0], dtype=int)
        maxseg = 1
    elif mode == "image":
        recon.pop("seg", None)
        cfg.pop("seg", None)
        if "node" in recon:
            maxseg = int(np.asarray(recon["node"]).shape[0])
        else:
            maxseg = int(np.asarray(cfg["node"]).shape[0])
    else:  # 'seg'
        maxseg = int(np.max(np.asarray(cfg["seg"])))

    bulk = recon.get("bulk")
    has_prop = ("prop" in recon) and (
        recon["prop"] is not None
        and not (isinstance(recon["prop"], np.ndarray) and recon["prop"].size == 0)
    )

    # Reseed recon.prop / cfg.prop from recon.bulk when the user hasn't
    # supplied an initial recon.prop (matches the rbrun.m fallback).
    if isinstance(bulk, dict) and not has_prop and "param" not in recon:
        nref = float(bulk.get("n", 1.37))
        mua_b = float(bulk.get("mua", 0.0))
        if "musp" in bulk:
            musp_b = float(bulk["musp"])
        elif "dcoeff" in bulk:
            musp_b = 1.0 / (3.0 * float(bulk["dcoeff"]))
        else:
            musp_b = 0.0

        prop_row = np.array([[mua_b, musp_b, 0.0, nref]])
        recon["prop"] = np.tile(prop_row, (maxseg, 1))

        if mode == "image":
            n_cfg = int(np.asarray(cfg["node"]).shape[0])
            cfg["prop"] = np.tile(prop_row, (n_cfg, 1))
        else:
            # prepend the air/background row (matches the rbrun.m convention)
            recon["prop"] = np.vstack([[0.0, 0.0, 1.0, 1.0], recon["prop"]])

    return cfg, recon
