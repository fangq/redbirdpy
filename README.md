# Redbird-Python - Python Toolbox for Diffuse Optical Imaging

A Python translation of the [Redbird](https://github.com/fangq/redbird) MATLAB toolbox for forward and inverse modeling of diffuse optical tomography (DOT) and near-infrared spectroscopy (NIRS).

## Overview

Redbird provides a complete framework for simulating light propagation in biological tissue using the diffusion approximation and the Finite Element Method (FEM). It supports:

- **Forward Modeling**: Solve the diffusion equation for photon fluence rate
- **Inverse Reconstruction**: Recover optical properties from boundary measurements
- **Multi-spectral Analysis**: Wavelength-dependent simulations for spectroscopy
- **Frequency-Domain**: Support for amplitude-modulated light sources

## Installation

```bash
# Basic installation
pip install numpy scipy

# Optional: for mesh generation
pip install iso2mesh  # or install from https://github.com/NeuroJSON/pyiso2mesh
```

Place the `redbirdpy` folder in your Python path or project directory.

## Quick Start

```python
import numpy as np
import redbirdpy as rb

# Define mesh (nodes and tetrahedral elements)
node = ...  # Nn x 3 array of coordinates
elem = ...  # Ne x 4 array of node indices

# Define optical properties [mua, musp, g, n]
prop = np.array([
    [0.0, 0.0, 1.0, 1.0],     # Label 0 (external)
    [0.01, 1.0, 0.0, 1.37]    # Label 1 (tissue)
])

# Configure simulation
cfg = {
    'node': node,
    'elem': elem,
    'prop': prop,
    'srcpos': np.array([[50, 50, 0]]),    # Source position
    'srcdir': np.array([[0, 0, 1]]),      # Source direction
    'detpos': np.array([[50, 60, 0]]),    # Detector position
    'detdir': np.array([[0, 0, 1]]),      # Detector direction
    'seg': np.ones(elem.shape[0], dtype=int),
    'omega': 0  # CW mode (or 2*pi*freq for FD)
}

# Prepare mesh and run forward simulation
cfg, sd = rb.utility.meshprep(cfg)
detval, phi = rb.forward.runforward(cfg)

print(f"Detector measurement: {detval}")
```

## Module Structure

### `redbirdpy.forward` - Forward Modeling

| Function | Description |
|----------|-------------|
| `runforward(cfg)` | Main forward solver for all sources/wavelengths |
| `femlhs(cfg, deldotdel, wv)` | Build FEM stiffness matrix |
| `femrhs(cfg, sd, wv)` | Build right-hand-side vectors |
| `femsolve(A, b, method)` | Solve linear system |
| `femgetdet(phi, cfg)` | Extract detector values |
| `deldotdel(cfg)` | Compute gradient operators |
| `jac(sd, phi, ...)` | Build Jacobian matrices |
| `jacchrome(Jmua, chromes)` | Chromophore Jacobians |

### `redbirdpy.recon` - Reconstruction

| Function | Description |
|----------|-------------|
| `runrecon(cfg, recon, data, sd)` | Iterative Gauss-Newton reconstruction |
| `reginv(A, b, lambda)` | Regularized inversion (auto-selects method) |
| `reginvover(A, b, lambda)` | Overdetermined solver |
| `reginvunder(A, b, lambda)` | Underdetermined solver |
| `matreform(A, ymeas, ymodel)` | Matrix reformulation |
| `prior(seg, type)` | Structure-prior matrices |

### `redbirdpy.utility` - Utilities

| Function | Description |
|----------|-------------|
| `meshprep(cfg)` | Prepare mesh with derived quantities |
| `sdmap(cfg, maxdist)` | Create source-detector mapping |
| `getoptodes(cfg)` | Get optode positions |
| `getdistance(src, det)` | Source-detector distances |
| `getreff(n_in, n_out)` | Effective reflection coefficient |
| `addnoise(data, snr)` | Add simulated noise |
| `elem2node(elem, val)` | Element to node interpolation |

### `redbirdpy.property` - Optical Properties

| Function | Description |
|----------|-------------|
| `extinction(wavelengths, chromes)` | Molar extinction coefficients |
| `updateprop(cfg)` | Update props from chromophore concentrations |
| `getbulk(cfg)` | Get bulk/background properties |
| `musp2sasp(musp, wavelength)` | Convert μs' to scattering amp/power |

## Configuration Dictionary

The main data structure is a Python dictionary with these fields:

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `node` | (Nn, 3) float | Node coordinates in mm |
| `elem` | (Ne, 4) int | Tetrahedral element connectivity |
| `prop` | (Nseg, 4) or dict | Optical properties [mua, musp, g, n] |
| `srcpos` | (Ns, 3) float | Source positions |
| `srcdir` | (Ns, 3) or (1, 3) | Source directions |
| `detpos` | (Nd, 3) float | Detector positions |
| `detdir` | (Nd, 3) or (1, 3) | Detector directions |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `seg` | (Ne,) int | Element labels for segmentation |
| `omega` | float or dict | Angular frequency (rad/s), 0 for CW |
| `param` | dict | Chromophore concentrations |
| `bulk` | dict | Background property values |
| `face` | (Nf, 3) int | Surface triangles (auto-computed) |
| `evol` | (Ne,) float | Element volumes (auto-computed) |

## Multi-Spectral Simulations

For wavelength-dependent simulations:

```python
# Define properties for each wavelength
prop = {
    '690': np.array([[0, 0, 1, 1], [0.008, 1.2, 0, 1.37]]),
    '830': np.array([[0, 0, 1, 1], [0.012, 1.0, 0, 1.37]])
}

# Or use chromophore concentrations
cfg['param'] = {
    'hbo': 50.0,    # μM
    'hbr': 25.0,    # μM
    'water': 0.7,   # volume fraction
    'scatamp': 10.0,
    'scatpow': 1.5
}
```

## Reconstruction Example

```python
# Measured data (simulated or real)
detphi0 = ...  # (Ndet, Nsrc) array

# Initial guess
recon_cfg = {
    'prop': initial_prop,
    'lambda': 0.1,  # Regularization parameter
}

# Run reconstruction
recon_out, resid, cfg_out = rb.recon.runrecon(
    cfg, recon_cfg, detphi0, sd,
    maxiter=10,
    lambda_=0.1,
    reform='real',
    report=True
)
```

## Units

- **Length**: millimeters (mm)
- **Absorption coefficient (μa)**: 1/mm
- **Scattering coefficient (μs')**: 1/mm
- **Concentration**: micromolar (μM) for hemoglobin
- **Frequency**: Hz (converted to rad/s internally)

## References

1. Fang Q, Boas DA. "Monte Carlo simulation of photon migration in 3D turbid media accelerated by graphics processing units." *Optics Express* 17(22):20178-90 (2009)

2. Arridge SR. "Optical tomography in medical imaging." *Inverse Problems* 15(2):R41-R93 (1999)

3. Prahl S. "Optical Absorption of Hemoglobin." https://omlc.org/spectra/hemoglobin/

## License

GPL version 3 - see LICENSE file for details.

## Author

Original MATLAB toolbox by Qianqian Fang (q.fang@neu.edu)  
Python translation based on Redbird
