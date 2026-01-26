"""
Redbird - Python toolbox for Diffuse Optical Tomography

A Python translation of the Redbird-m MATLAB toolbox for forward and inverse
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


Author: Translated from Redbird-m MATLAB toolbox by Qianqian Fang (q.fang <at> neu.edu)
License: GPL version 3
"""

__version__ = "0.1.0"
__author__ = "Qianqian Fang"

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
    **kwargs : dict
        Additional options passed to runforward or runrecon

    Returns
    -------
    Results from runforward (if forward only) or runrecon (if reconstruction)
    """
    if recon_cfg is None:
        return runforward(cfg, **kwargs)
    else:
        if detphi0 is None:
            raise ValueError("detphi0 is required for reconstruction")
        return runrecon(cfg, recon_cfg, detphi0, sd, **kwargs)
