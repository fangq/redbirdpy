![Redbird Banner](https://raw.githubusercontent.com/fangq/redbird/refs/heads/master/doc/images/redbird_banner.png)

# RedbirdPy - A Model-Based Diffuse Optical Imaging Toolbox for Python

* **Copyright**: (C) Qianqian Fang (2005–2026) \<q.fang at neu.edu>
* **License**: GNU Public License V3 or later
* **Version**: 0.2.0 (Flamingo)
* **GitHub**: [https://github.com/fangq/redbirdpy](https://github.com/fangq/redbirdpy)
* **Acknowledgement**: This project is supported by the US National Institute of Health (NIH)
  grant [R01-CA204443](https://reporter.nih.gov/project-details/10982160)

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)
- [Module Structure](#module-structure)
- [Data Structures](#data-structures)
  - [Forward Structure `cfg`](#forward-structure-cfg)
  - [Reconstruction Structure `recon`](#reconstruction-structure-recon)
- [Wide-Field Sources and Detectors](#wide-field-sources-and-detectors)
- [Multi-Spectral Simulations](#multi-spectral-simulations)
- [Examples](#examples)
- [Units](#units)
- [Running Tests](#running-tests)
- [How to Cite](#how-to-cite)
- [References](#references)

---

## Introduction

**RedbirdPy** is a Python translation of the [Redbird MATLAB toolbox](https://github.com/fangq/redbird) for diffuse optical imaging (DOI) and diffuse optical tomography (DOT). It provides a fast, experimentally-validated forward solver for the diffusion equation using the finite-element method (FEM), along with advanced non-linear image reconstruction algorithms.

Redbird is the result of over two decades of active research in DOT and image reconstruction. It has been the core data analysis tool in numerous publications related to optical breast imaging, prior-guided reconstruction techniques, multi-modal imaging, and wide-field DOT systems.

### Key Features

- **Forward Modeling**: Solve the diffusion equation for photon fluence using FEM
- **Inverse Reconstruction**: Iterative Gauss-Newton with Tikhonov regularization
- **Wide-Field Sources/Detectors**: Support for planar, pattern, and Fourier-basis illumination
- **Multi-Spectral Analysis**: Wavelength-dependent simulations for chromophore estimation
- **Frequency-Domain**: Support for amplitude-modulated light sources
- **Dual-Mesh Reconstruction**: Use coarse mesh for faster inverse solving
- **Structure Priors**: Laplacian, Helmholtz, and compositional priors

### Validation

The forward solver is carefully validated against Monte Carlo solvers - **MCX** and **MMC**. The diffusion approximation is valid in high-scattering media where the reduced scattering coefficient (μs') is much greater than the absorption coefficient (μa).

---

## Installation

### Requirements

- Python 3.6+
- NumPy
- SciPy
- Iso2Mesh

### Basic Installation

```bash
pip install numpy scipy
pip install iso2mesh  # or from https://github.com/NeuroJSON/pyiso2mesh
```

### Optional Dependencies

```bash
# For accelerated Jacobian computation
pip install numba          # JIT compilation

# For accelerated solvers
pip install blocksolver  # or from https://github.com/fangq/blit

# For other linear solvers
pip install pypardiso      # Intel MKL PARDISO (fastest direct solver)
pip install scikit-umfpack # UMFPACK direct solver
pip install pyamg          # Algebraic multigrid preconditioner
```

### Installation from Source

```bash
git clone https://github.com/fangq/redbirdpy.git
cd redbirdpy
pip install -e .
```

Or simply import `redbirdpy` from inside the repository's top folder.

---

## Quick Start

```python
import numpy as np
import redbirdpy as rb
from iso2mesh import meshabox

# Create mesh (iso2mesh returns 1-based indices)
node, face, elem = meshabox([0, 0, 0], [60, 60, 30], 5)

# Define optical properties [mua, mus, g, n]
prop = np.array([
    [0.0, 0.0, 1.0, 1.0],     # Label 0 (external/air)
    [0.01, 1.0, 0.0, 1.37]    # Label 1 (tissue)
])

# Configure simulation
cfg = {
    'node': node,
    'elem': elem,
    'prop': prop,
    'srcpos': np.array([[30, 30, 0]]),
    'srcdir': np.array([[0, 0, 1]]),
    'detpos': np.array([[30, 40, 0], [40, 30, 0]]),
    'detdir': np.array([[0, 0, 1]]),
    'seg': np.ones(elem.shape[0], dtype=int),
    'omega': 0  # CW mode
}

# Prepare mesh and run forward simulation
cfg, sd = rb.meshprep(cfg)
detval, phi = rb.run(cfg)

print(f"Detector measurements: {detval}")
```

---

## Workflow Overview

Redbird performs two main tasks:

1. **Forward Simulation**: Computes light distribution (fluence, in 1/mm²) across source-detector arrays within a mesh-based medium with known optical properties.

2. **Image Reconstruction**: Iteratively recovers 3D distributions of unknown optical properties by fitting forward simulations to measured data.

### Reconstruction Modes

Redbird supports four types of image reconstructions:

| Mode | Description                                                   |
|------|---------------------------------------------------------------|
| **Bulk Fitting** | Estimate single set of properties for the entire domain       |
| **Segmented** | One property set per labeled tissue segment ("hard-prior")    |
| **Soft-Prior** | Spatial priors as soft constraints (Laplacian, compositional) |
| **Unconstrained** | Independent properties per node with Tikhonov regularization  |

---

## Module Structure

### `redbirdpy.forward` - Forward Modeling

| Function | Description |
|----------|-------------|
| `runforward(cfg)` | Main forward solver for all sources/wavelengths |
| `femlhs(cfg, deldotdel, wv)` | Build FEM stiffness matrix (LHS) |
| `femrhs(cfg, sd, wv)` | Build right-hand-side vectors |
| `femgetdet(phi, cfg, rhs)` | Extract detector values |
| `jac(sd, phi, ...)` | Build Jacobian matrices for mua |
| `jacchrome(Jmua, chromes)` | Build chromophore Jacobians |

### `redbirdpy.recon` - Reconstruction

| Function | Description |
|----------|-------------|
| `runrecon(cfg, recon, data, sd)` | Iterative Gauss-Newton reconstruction |
| `reginv(A, b, lambda)` | Regularized inversion (auto-selects method) |
| `reginvover(A, b, lambda)` | Overdetermined system solver |
| `reginvunder(A, b, lambda)` | Underdetermined system solver |
| `matreform(A, ymeas, ymodel, form)` | Matrix reformulation (real/complex/logphase) |
| `prior(seg, type)` | Structure-prior regularization matrices |
| `syncprop(cfg, recon)` | Synchronize properties between meshes |

### `redbirdpy.utility` - Utilities

| Function | Description |
|----------|-------------|
| `meshprep(cfg)` | Prepare mesh with all derived quantities |
| `sdmap(cfg, maxdist)` | Create source-detector mapping |
| `src2bc(cfg, isdet)` | Convert wide-field sources to boundary conditions |
| `getoptodes(cfg)` | Get displaced optode positions |
| `getdistance(src, det)` | Compute source-detector distances |
| `getreff(n_in, n_out)` | Effective reflection coefficient |
| `getltr(cfg)` | Transport mean-free path |
| `addnoise(data, snr)` | Add simulated shot/thermal noise |
| `elem2node(elem, val)` | Element to node interpolation |
| `meshinterp(...)` | Interpolate between meshes |

### `redbirdpy.property` - Optical Properties

| Function | Description |
|----------|-------------|
| `extinction(wavelengths, chromes)` | Molar extinction coefficients |
| `updateprop(cfg)` | Update props from chromophore concentrations |
| `getbulk(cfg)` | Get bulk/background properties |
| `musp2sasp(musp, wavelength)` | Convert μs' to scattering amplitude/power |
| `setmesh(cfg, node, elem)` | Associate new mesh with configuration |

### `redbirdpy.solver` - Linear Solvers

| Function | Description |
|----------|-------------|
| `femsolve(A, b, method)` | Solve linear system with auto-selection |
| `get_solver_info()` | Query available solver backends |

Supported solvers: `pardiso`, `umfpack`, `cholmod`, `superlu`, `blqmr`, `cg`, `cg+amg`, `gmres`, `bicgstab`

### `redbirdpy.analytical` - Analytical Solutions

| Function | Description |
|----------|-------------|
| `infinite_cw(...)` | CW fluence in infinite medium |
| `semi_infinite_cw(...)` | CW fluence in semi-infinite medium |
| `semi_infinite_cw_flux(...)` | Diffuse reflectance |
| `infinite_td(...)` | Time-domain in infinite medium |
| `semi_infinite_td(...)` | Time-domain in semi-infinite medium |
| `sphere_infinite(...)` | Sphere in infinite medium |
| `sphere_semi_infinite(...)` | Sphere in semi-infinite medium |
| `sphere_slab(...)` | Sphere in slab geometry |

---

## Data Structures

### Forward Structure `cfg`

The forward solver uses a dictionary with the following fields:

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `node` | (Nn, 3) float | Node coordinates in mm |
| `elem` | (Ne, 4+) int | Tetrahedral connectivity (1-based) |
| `prop` | (Nseg, 4) or dict | Optical properties [mua, mus, g, n] |
| `srcpos` | (Ns, 3) float | Source positions |
| `srcdir` | (Ns, 3) or (1, 3) | Source directions |
| `detpos` | (Nd, 3) float | Detector positions |
| `detdir` | (Nd, 3) or (1, 3) | Detector directions |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `seg` | (Ne,) int | Element labels for segmentation |
| `omega` | float or dict | Angular frequency (rad/s), 0 for CW |
| `srctype` | str | Source type: `'pencil'`, `'planar'`, `'pattern'`, `'fourier'` |
| `srcparam1` | (4,) float | Wide-field source parameter 1 |
| `srcparam2` | (4,) float | Wide-field source parameter 2 |
| `srcpattern` | (Nx, Ny) or (Nx, Ny, Np) | Pattern source data |
| `dettype` | str | Detector type (same options as srctype) |
| `detparam1` | (4,) float | Wide-field detector parameter 1 |
| `detparam2` | (4,) float | Wide-field detector parameter 2 |
| `detpattern` | array | Pattern detector data |
| `bulk` | dict | Background property values |
| `param` | dict | Chromophore concentrations |

#### Auto-Computed Fields (via `meshprep`)

| Field | Description |
|-------|-------------|
| `face` | Surface triangles (1-based) |
| `area` | Face areas |
| `evol` | Element volumes |
| `nvol` | Nodal volumes |
| `reff` | Effective reflection coefficient |
| `deldotdel` | Gradient operator matrix |

### Reconstruction Structure `recon`

| Field | Type | Description |
|-------|------|-------------|
| `node` | (Nn_r, 3) float | Reconstruction mesh nodes (optional) |
| `elem` | (Ne_r, 4) int | Reconstruction mesh elements (optional) |
| `prop` | (Nn_r, 4) float | Initial/recovered optical properties |
| `param` | dict | Multi-spectral parameters |
| `lambda` | float | Tikhonov regularization parameter |
| `bulk` | dict | Initial guess values |
| `mapid` | (Nn, ) float | Forward-to-recon mesh mapping (element IDs) |
| `mapweight` | (Nn, 4) float | Barycentric interpolation weights |
| `seg` | array | Segmentation labels for priors |

---

## Wide-Field Sources and Detectors

Redbird supports wide-field illumination patterns for spatially-modulated imaging.

### Source Types

| Type | Description |
|------|-------------|
| `'pencil'` | Point source (default) |
| `'planar'` | Uniform rectangular illumination |
| `'pattern'` | User-defined 2D/3D pattern array |
| `'fourier'` | Fourier-basis spatial frequencies |

### Configuration Example

```python
cfg = {
    # ... mesh and properties ...
    
    # Planar source
    'srctype': 'planar',
    'srcpos': np.array([[10, 10, 0]]),      # Corner position
    'srcparam1': np.array([40, 0, 0, 0]),   # Width in x (mm)
    'srcparam2': np.array([0, 40, 0, 0]),   # Width in y (mm)
    'srcdir': np.array([[0, 0, 1]]),
    
    # Pattern source (multiple patterns)
    'srctype': 'pattern',
    'srcpattern': patterns,  # Shape: (Nx, Ny) or (Nx, Ny, Npatterns)
    
    # Fourier source (kx × ky patterns)
    'srctype': 'fourier',
    'srcparam1': np.array([40, 0, 0, 3]),   # Last value = kx
    'srcparam2': np.array([0, 40, 0, 3]),   # Last value = ky
}
```

### Wide-Field Detectors

Configure similarly using `dettype`, `detparam1`, `detparam2`, `detpattern`.

---

## Multi-Spectral Simulations

### Wavelength-Dependent Properties

```python
cfg['prop'] = {
    '690': [[0, 0, 1, 1], [0.012, 1.1, 0, 1.37]],
    '830': [[0, 0, 1, 1], [0.008, 0.9, 0, 1.37]]
}
```

### Chromophore-Based Properties

```python
cfg['param'] = {
    'hbo': 50.0,       # Oxyhemoglobin (μM)
    'hbr': 25.0,       # Deoxyhemoglobin (μM)
    'water': 0.7,      # Water volume fraction
    'lipids': 0.1,     # Lipid volume fraction
    'scatamp': 10.0,   # Scattering amplitude
    'scatpow': 1.5     # Scattering power
}

# Properties computed via: μs' = scatamp × λ^(-scatpow)
```

### Available Chromophores

| Name | Units | Description |
|------|-------|-------------|
| `hbo` | μM | Oxyhemoglobin |
| `hbr` | μM | Deoxyhemoglobin |
| `water` | fraction | Water content |
| `lipids` | fraction | Lipid content |
| `aa3` | μM | Cytochrome c oxidase |

---

## Examples

### Basic Forward Simulation

```python
import redbirdpy as rb
from iso2mesh import meshabox

node, face, elem = meshabox([0, 0, 0], [60, 60, 30], 5)

cfg = {
    'node': node, 'elem': elem,
    'prop': np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
    'srcpos': np.array([[30, 30, 0]]),
    'srcdir': np.array([[0, 0, 1]]),
    'detpos': np.array([[30, 40, 0]]),
    'detdir': np.array([[0, 0, 1]]),
    'seg': np.ones(elem.shape[0], dtype=int),
    'omega': 0
}

cfg, sd = rb.meshprep(cfg)
detval, phi = rb.run(cfg)
```

### Frequency-Domain Simulation

```python
cfg['omega'] = 2 * np.pi * 100e6  # 100 MHz modulation
detval, phi = rb.run(cfg)

amplitude = np.abs(detval)
phase = np.angle(detval)
```

### Image Reconstruction

```python
# Generate synthetic measurement
detphi0, _ = rb.run(cfg0)  # Heterogeneous ground truth

# Setup reconstruction
recon = {
    'prop': initial_prop,
    'lambda': 0.1,
}

# Run reconstruction
newrecon, resid, newcfg = rb.run(cfg, recon, detphi0, 
                                  lambda_=1e-4, maxiter=10)
```

### Dual-Mesh Reconstruction

```python
import iso2mesh as i2m

# Fine forward mesh
cfg, sd = rb.meshprep(cfg)

# Coarse reconstruction mesh
recon = {}
recon['node'], _, recon['elem'] = i2m.meshabox([0,0,0], [60,60,30], 15)
recon['mapid'], recon['mapweight'] = i2m.tsearchn(
    recon['node'], recon['elem'], cfg['node']
)
recon['prop'] = np.tile(cfg['prop'][1,:], (recon['node'].shape[0], 1))

newrecon, resid = rb.run(cfg, recon, detphi0, lambda_=1e-4)[:2]
```

### Wide-Field Forward and Reconstruction

```python
# Create illumination patterns
srcpattern = np.zeros((16, 16, 32))
# ... define patterns ...

cfg = {
    # ... mesh ...
    'srctype': 'pattern',
    'srcpos': np.array([[10, 10, 0]]),
    'srcparam1': [100, 0, 0, 0],
    'srcparam2': [0, 40, 0, 0],
    'srcdir': np.array([[0, 0, 1]]),
    'srcpattern': srcpattern,
    'dettype': 'pattern',
    'detpattern': srcpattern,
    # ...
}

cfg, sd = rb.meshprep(cfg)
detval, phi = rb.run(cfg)
```

---

## Units

| Quantity | Unit |
|----------|------|
| Length | millimeters (mm) |
| Absorption coefficient (μa) | 1/mm |
| Scattering coefficient (μs') | 1/mm |
| Hemoglobin concentration | micromolar (μM) |
| Water/lipid content | volume fraction (0-1) |
| Frequency | Hz (converted to rad/s internally) |
| Fluence | 1/mm² |

---

## Running Tests

```bash
# Run all tests
python -m unittest discover -v -s test

# Run specific test module
python -m unittest test.test_forward -v

# Run with pytest (if installed)
pytest test/ -v
```

---

## How to Cite

If you use Redbird in your research, please cite:

**Software workflow:**
> Fang Q, et al., "A multi-modality image reconstruction platform for diffuse optical tomography," in Biomed. Opt., BMD24 (2008). https://doi.org/10.1364/BIOMED.2008.BMD24

**Validation and methodology:**
> Fang Q, Carp SA, Selb J, et al., "Combined optical Imaging and mammography of the healthy breast: optical contrast derives from breast structure and compression," IEEE Trans. Medical Imaging, vol. 28, issue 1, pp. 30–42, Jan. 2009.

**Compositional priors:**
> Fang Q, Moore RH, Kopans DB, Boas DA, "Compositional-prior-guided image reconstruction algorithm for multi-modality imaging," Biomedical Optics Express, vol. 1, issue 1, pp. 223-235, 2010.

**Multi-spectral reconstruction:**
> Fang Q, Meaney PM, Paulsen KD, "Microwave image reconstruction of tissue property dispersion characteristics utilizing multiple frequency information," IEEE Trans. Microwave Theory and Techniques, vol. 52, No. 8, pp. 1866-1875, Aug. 2004.

---

## References

1. Fang Q, "Computational methods for microwave medical imaging," Ph.D. dissertation, Dartmouth College, 2004.

2. Arridge SR, "Optical tomography in medical imaging," Inverse Problems 15(2):R41-R93 (1999).

3. Prahl S, "Optical Absorption of Hemoglobin," https://omlc.org/spectra/hemoglobin/

4. Haskell RC, et al., "Boundary conditions for the diffusion equation in radiative transfer," JOSA A 11(10):2727-2741 (1994).

---

## License

GNU General Public License v3.0 or later - see [LICENSE](LICENSE) file for details.

## Author

**Qianqian Fang** (q.fang@neu.edu)
Computational Optics & Translational Imaging (COTI) Lab
Northeastern University

Python translation based on the [Redbird MATLAB toolbox](https://github.com/fangq/redbird).