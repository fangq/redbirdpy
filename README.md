![Redbird Banner](https://raw.githubusercontent.com/fangq/redbird/refs/heads/master/doc/images/redbird_banner.png)

# RedbirdPy - A Model-Based Diffuse Optical Imaging Toolbox for Python

* **Copyright**: (C) Qianqian Fang (2005–2026) \<q.fang at neu.edu>
* **License**: GNU Public License V3 or later
* **Version**: 0.4.1 (Summer Tanagers)
* **GitHub**: [https://github.com/fangq/redbirdpy](https://github.com/fangq/redbirdpy)
* **PyPI**: [https://pypi.org/project/redbirdpy](https://pypi.org/project/redbirdpy)
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
- [Monte Carlo Forward Solvers (MCX/MMC)](#monte-carlo-forward-solvers-mcxmmc)
- [Microwave Tomography (MWT)](#microwave-tomography-mwt)
- [Time-Domain DOT](#time-domain-dot)
- [Ratiometric (fNIRS) Reconstruction](#ratiometric-fnirs-reconstruction)
- [Examples](#examples)
- [Units](#units)
- [Running Tests](#running-tests)
- [How to Cite](#how-to-cite)
- [References](#references)

---

## Introduction

**RedbirdPy** is a Python translation of the [Redbird MATLAB toolbox](https://github.com/fangq/redbird) for diffuse optical imaging (DOI) and diffuse optical tomography (DOT). It provides a fast, experimentally-validated forward solver for the diffusion equation using the finite-element method (FEM), along with advanced non-linear image reconstruction algorithms. Beyond diffuse optics, RedbirdPy now also supports **microwave tomography (MWT)** through a vector/scalar Helmholtz forward solver, and can interface with the **Monte Carlo** photon transport engines **MCX/pmcx** and **MMC/pmmc** as alternative forward/adjoint solvers feeding the same reconstruction pipeline.

Redbird is the result of over two decades of active research in DOT and image reconstruction. It has been the core data analysis tool in numerous publications related to optical breast imaging, prior-guided reconstruction techniques, multi-modal imaging, and wide-field DOT systems.

### Key Features

- **Forward Modeling**: Solve the diffusion equation for photon fluence using FEM
- **Monte Carlo Forward/Adjoint**: Drop-in `pmcx` (voxel-grid) and `pmmc` (mesh) Monte Carlo solvers with replay-based adjoint Jacobians, routed through the same API
- **Microwave Tomography (MWT)**: FEM Helmholtz solver with a first-order Bayliss-Turkel radiating boundary condition for permittivity/conductivity reconstruction
- **Inverse Reconstruction**: Iterative Gauss-Newton with Tikhonov regularization, plus a matrix-free LSQR path for large voxel-grid problems
- **Continuous-Wave, Frequency-Domain, and Time-Domain**: CW, amplitude-modulated (RF), and Crank-Nicolson time-domain (TPSF) forward solvers
- **Ratiometric / fNIRS Data**: Reconstruct from relative (I/I₀) measurements via `recon['isratio']`
- **Wide-Field Sources/Detectors**: Support for planar, pattern, and Fourier-basis illumination
- **Multi-Spectral Analysis**: Wavelength-dependent simulations for chromophore estimation
- **Dual-Mesh Reconstruction**: Use coarse mesh for faster inverse solving
- **Structure Priors**: Laplacian, Helmholtz, and compositional priors
- **Parallel Linear Solvers**: Multiple direct and iterative backends with `multiprocessing`-based multi-RHS solving

### Validation

The FEM forward solver is carefully validated against Monte Carlo solvers - **MCX** and **MMC**. The diffusion approximation is valid in high-scattering media where the reduced scattering coefficient (μs') is much greater than the absorption coefficient (μa). When the diffusion approximation is insufficient (e.g. near sources, in low-scattering regions, or voids), the integrated Monte Carlo forward/adjoint solvers can be used in place of the FEM solver without changing the reconstruction code.

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

# For Monte Carlo forward/adjoint solvers
pip install pmcx           # MCX voxel-grid Monte Carlo (GPU)
pip install pmmc           # MMC mesh-based Monte Carlo (GPU)

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
| `runforward(cfg, return_jacobian=False)` | Main forward solver; auto-dispatches to FEM, Monte Carlo (`cfg['nphoton']`), time-domain (`cfg['tstart/tstep/tend']`), or MWT (`cfg['bulk']['epsilon/sigma']`) |
| `runtd(cfg, theta=0.5, ...)` | Time-domain DOT solver (theta-method / Crank-Nicolson) |
| `femlhs(cfg, deldotdel, wv, mode)` | Build FEM system matrix (diffusion, Helmholtz, or mass matrix) |
| `femrhs(cfg, sd, wv)` | Build right-hand-side vectors |
| `femgetdet(phi, cfg, rhs)` | Extract detector values |
| `jac(sd, phi, ...)` | Build Jacobian matrices for mua |
| `jacchrome(Jmua, chromes)` | Build chromophore Jacobians |
| `jacepssigma(Jmua, omegas)` | Chain mua-Jacobian to permittivity/conductivity Jacobians (MWT) |
| `jacmus / jacscat / jacscatamp / jacscatpow` | Scattering-property Jacobians |
| `jacnode(Jelem, ...)` | Convert element-based Jacobian to node-based |

### `redbirdpy.recon` - Reconstruction

| Function | Description |
|----------|-------------|
| `runrecon(cfg, recon, data, sd)` | Iterative Gauss-Newton reconstruction |
| `reginv(A, b, lambda)` | Regularized inversion (auto-selects method) |
| `reginvover(A, b, lambda)` | Overdetermined system solver |
| `reginvunder(A, b, lambda)` | Underdetermined system solver |
| `reglsqr(J, r, maxit, tol)` | Matrix-free LSQR for large voxel-grid Jacobians |
| `jacop(J)` | Wrap a 4-D voxel-grid Jacobian as a SciPy `LinearOperator` |
| `matreform(A, ymeas, ymodel, form)` | Matrix reformulation (real/complex/logphase) |
| `multispectral(...)` | Assemble multi-wavelength data into a single linear system |
| `createinv(...)` | Reformulate the inverse problem (complex/real/log-phase) |
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
| `getdetdir(cfg)` | Estimate inward detector normals on a surface mesh (MC) |
| `getdetdir_vol(cfg)` | Estimate inward detector normals on a voxel grid (MC) |
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
| `solverinfo()` | Query available solver backends |

Supported solvers: `pardiso`, `umfpack`, `cholmod`, `superlu`, `blqmr`, `cg`, `cg+amg`, `gmres`, `minres`, `minres+amg`, `qmr`, `bicgstab`

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
| `bulk` | dict | Background property values (`mua`/`musp`/`n`, or `epsilon`/`sigma` for MWT) |
| `param` | dict | Chromophore concentrations (or `epsilon`/`sigma` parameters for MWT) |
| `nphoton` | int | Photon count; **its presence routes the run to the Monte Carlo solver** |
| `vol` | (Nx,Ny,Nz) uint8 | Voxel-label volume; selects the `pmcx` voxel-grid MC path |
| `gpuid` | int | GPU device id for MC solvers |
| `tstart`, `tstep`, `tend` | float | Time-domain gate (s); their presence triggers the Crank-Nicolson solver |

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
| `isratio` | bool | Treat `data` as ratiometric (I/I₀) measurements, e.g. fNIRS |

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

## Monte Carlo Forward Solvers (MCX/MMC)

In addition to the built-in FEM diffusion solver, RedbirdPy can use the GPU-accelerated Monte Carlo photon transport engines [**MCX/pmcx**](https://github.com/fangq/mcx) and [**MMC/pmmc**](https://github.com/fangq/mmc) as the forward solver. This is useful where the diffusion approximation breaks down (near sources, low-scattering regions, voids). The MC solver plugs into the same `runforward`/`runrecon` API, and an **adjoint (replay-based) Jacobian** is returned for reconstruction.

### Selecting the Monte Carlo solver

Simply add `cfg['nphoton']` to a configuration. Its presence routes the run through Monte Carlo instead of FEM:

| `cfg` fields present | Solver used | Engine |
|----------------------|-------------|--------|
| `nphoton`, plus `node`/`elem` | Mesh-based MC | `pmmc` (MMC) |
| `nphoton`, plus `vol` | Voxel-grid MC | `pmcx` (MCX) |

```python
# Mesh-based Monte Carlo forward (pmmc)
cfg, sd = rb.meshprep(cfg)        # standard DOT cfg (node/elem/srcpos/detpos/prop)
cfg['nphoton'] = int(1e7)         # presence of nphoton -> MC branch
cfg['gpuid']   = 1
detphi, phi = rb.run(cfg)
```

### Adjoint Jacobians for reconstruction

Pass `return_jacobian=True` to obtain the adjoint Jacobian directly from the MC engine (computed via replay), bypassing the FEM `jac()` build:

```python
detphi, phi, Jext = rb.runforward(cfg, return_jacobian=True)
# pmmc (mesh):  Jext['mua'] has shape (Nnode, Nsrc*Ndet)
# pmcx (voxel): Jext['mua'] has shape (Nx, Ny, Nz, Nsrc*Ndet)
```

`runrecon` automatically detects an MC-supplied Jacobian and consumes it in place of the FEM Jacobian. Detector directions (`cfg['detdir']`) are auto-filled from the surface mesh (`getdetdir`) or voxel grid (`getdetdir_vol`) when missing. In CW mode only the mua Jacobian is available; in RF mode (`omega > 0`) both mua and D Jacobians are computed.

### Matrix-free LSQR for voxel-grid problems

Voxel-grid (`pmcx`) reconstructions can have millions of unknowns, making the normal equations `JᵀJ` intractable. RedbirdPy provides a **matrix-free LSQR** path that wraps the 4-D Jacobian as a SciPy `LinearOperator` and solves with early stopping as implicit regularization:

```python
detphi, phi, Jext = rb.runforward(cfg, return_jacobian=True)   # cfg has 'vol'
misfit = ymeas - np.asarray(detphi).ravel()
delta_mua, info = rb.reglsqr(Jext['mua'], misfit, maxit=100, tol=1e-6)
```

See `example/demo_redbird_forward_mc.py`, `demo_redbird_jacobian_mc.py`, `demo_redbird_recon_mc.py`, and `demo_redbird_recon_mcx.py`.

---

## Microwave Tomography (MWT)

RedbirdPy includes a FEM **Helmholtz** forward solver with a first-order **Bayliss-Turkel radiating boundary condition (RBC)**, enabling microwave (and, more generally, scalar wave) tomography of complex permittivity. The reconstruction recovers relative permittivity (`epsilon`) and conductivity (`sigma`).

### Triggering MWT mode

MWT is detected automatically when `cfg['bulk']` carries `epsilon` and/or `sigma`. The optical-property table is reinterpreted with columns `[epsilon_r, sigma, mu0, n]` (in place of `[mua, musp, g, n]`), and `cfg['omega']` is the angular frequency (rad/s) of the field.

```python
import numpy as np
MU0 = 4 * np.pi * 1e-10   # permeability (H/mm)

cfg = {
    'node': node, 'elem': elem,
    'seg':  seg,                       # element labels
    # bulk with epsilon/sigma -> Helmholtz (MWT) solver
    'bulk': {'epsilon': 78.0, 'sigma': 1e-3, 'n': np.sqrt(78.0)},
    # property columns become [epsilon_r, sigma, mu0, n]
    'prop': {'5e8': np.array([[ 1, 0,      MU0, 1],     # background
                              [40, 0.5e-3, MU0, 2]])},  # target
    # line antennas use 6-column [x0 y0 z0 x1 y1 z1] src/det rows
    'srcpos': np.array([[20, 20, 0, 20, 20, 40]]),
    'detpos': np.array([[10, 10, 0, 10, 10, 40]]),
    'omega':  2 * np.pi * 5e8,         # 500 MHz
}

cfg, sd = rb.meshprep(cfg)             # precomputes RBC geometry
detphi, phi = rb.run(cfg)              # complex-valued field
```

`meshprep` precomputes the RBC geometry (`facecenter`, `facenormal`, `rbcorigin`, `facer`). For reconstruction, `jacepssigma` chains the absorption-style Jacobian into permittivity/conductivity Jacobians; multi-frequency data is supported through the multi-spectral machinery. See `test/test_mwt.py` for end-to-end examples.

> **Note:** MWT and time-domain modes are mutually exclusive, and time-domain requires `omega = 0`.

---

## Time-Domain DOT

A time-resolved (TPSF) forward solver is available using an implicit **theta-method / Crank-Nicolson** time-stepping scheme. It is triggered when `cfg['tstart']`, `cfg['tstep']`, and `cfg['tend']` are all set (and requires `omega = 0`).

```python
cfg = {
    # ... standard DOT mesh + [mua, musp, g, n] properties ...
    'omega':  0,           # required for time-domain
    'tstart': 0.0,
    'tstep':  50e-12,      # 50 ps
    'tend':   2e-9,        # 2 ns
}
cfg, sd = rb.meshprep(cfg)
detphi, phi = rb.run(cfg)
# detphi shape: (Ndet, Nsrc, Nt) - temporal point-spread functions
```

| Option (kwarg) | Default | Description |
|----------------|---------|-------------|
| `theta` | 0.5 | Time-stepping weight (0.5 = Crank-Nicolson, 1.0 = backward Euler) |
| `srctemporal` | impulse | Source time profile: `None` (impulse/TPSF), a callable `s(t)`, or a per-step array |
| `tdsavevol` | False | If `True`, also return the volumetric field `(Nn, Nsrc, Nt)` |

See `test/test_td.py` for impulse, custom-pulse, and validation examples.

---

## Ratiometric (fNIRS) Reconstruction

Many measurement systems (e.g. fNIRS) report **relative** changes `I/I₀` rather than absolute fluence. Setting `recon['isratio'] = True` tells `runrecon` that the supplied `data` are ratios; on the first iteration they are rescaled by the forward-model baseline so the Gauss-Newton misfit is computed consistently in absolute terms.

```python
recon = {
    'node': recon_node, 'elem': recon_elem,
    'lambda': 1e-4,
    'isratio': True,        # 'data' holds I/I0 ratios (e.g. fNIRS)
}
newrecon, resid, newcfg = rb.run(cfg, recon, ratio_data, sd, maxiter=10)
```

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

### Demo Scripts

The `example/` directory contains runnable demos:

| Script | Topic |
|--------|-------|
| `demo_redbird_basic.py` | Minimal forward + reconstruction |
| `demo_redbird_forward.py` | Forward simulation |
| `demo_redbird_forward_heterogeneous.py` | Heterogeneous-medium forward |
| `demo_redbird_forward_layered.py` | Layered-medium forward |
| `demo_redbird_forward_expert.py` | Low-level forward API |
| `demo_redbird_recon.py` / `demo_redbird_recon_expert.py` | FEM reconstruction |
| `demo_redbird_widefield.py` / `demo_redbird_recon_widefield.py` | Wide-field forward / reconstruction |
| `demo_redbird_forward_mc.py` | Monte Carlo (pmmc) forward vs FEM |
| `demo_redbird_jacobian_mc.py` | Mesh-mode MC adjoint Jacobian |
| `demo_redbird_recon_mc.py` | MC-based CW DOT reconstruction (pmmc) |
| `demo_redbird_recon_mcx.py` | Voxel-grid MC reconstruction with matrix-free LSQR (pmcx) |

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
| Time (time-domain gate) | seconds (s) |
| Relative permittivity (εr, MWT) | dimensionless |
| Conductivity (σ, MWT) | S/mm |

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

GNU General Public License v3.0 or later - see [LICENSE](LICENSE.txt) file for details.

## Author

**Qianqian Fang** (q.fang@neu.edu)
Computational Optics & Translational Imaging (COTI) Lab
Northeastern University

Python translation based on the [Redbird MATLAB toolbox](https://github.com/fangq/redbird).
