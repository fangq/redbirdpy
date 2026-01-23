#!/usr/bin/env python
"""
Redbird Basic Example - Most basic usage of Redbird.

Translated from: demo_redbird_basic.m

This example demonstrates:
- Creating a simple box mesh
- Setting up source and detector
- Running forward simulation
- Comparing with analytical solution
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbird as rb
from redbird import forward, utility

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False
    print("Warning: iso2mesh not installed. Using simple mesh.")


def cwdiffusion(mua, musp, reff, srcpos, detpos):
    """
    Analytical CW diffusion solution for semi-infinite medium.

    Parameters
    ----------
    mua : float
        Absorption coefficient (1/mm)
    musp : float
        Reduced scattering coefficient (1/mm)
    reff : float
        Effective reflection coefficient
    srcpos : ndarray
        Source position (1x3)
    detpos : ndarray
        Detector positions (Nx3)

    Returns
    -------
    phi : ndarray
        Fluence at detector positions
    """
    # Diffusion coefficient
    D = 1.0 / (3.0 * (mua + musp))

    # Effective attenuation coefficient
    mu_eff = np.sqrt(mua / D)

    # Extrapolated boundary distance
    A = (1 + reff) / (1 - reff)
    zb = 2 * A * D

    # Transport mean free path
    z0 = 1.0 / musp

    srcpos = np.atleast_2d(srcpos)
    detpos = np.atleast_2d(detpos)

    phi = np.zeros(detpos.shape[0])

    for i, det in enumerate(detpos):
        # Real source
        r1 = np.sqrt(np.sum((det - srcpos[0] - np.array([0, 0, z0])) ** 2))

        # Image source (for boundary condition)
        r2 = np.sqrt(np.sum((det - srcpos[0] + np.array([0, 0, z0 + 2 * zb])) ** 2))

        # Fluence from dipole source
        phi[i] = (1.0 / (4 * np.pi * D)) * (
            np.exp(-mu_eff * r1) / r1 - np.exp(-mu_eff * r2) / r2
        )

    return phi


def run_basic_example():
    """Run the basic forward simulation example."""
    print("=" * 60)
    print("Redbird Basic Example")
    print("=" * 60)

    # ============================================================
    # Prepare simulation input
    # ============================================================
    print("\n>> Creating mesh...")

    if HAS_ISO2MESH:
        node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 1)
    else:
        # Simple manual mesh for testing
        from itertools import product

        nx, ny, nz = 7, 7, 4
        x = np.linspace(0, 60, nx)
        y = np.linspace(0, 60, ny)
        z = np.linspace(0, 30, nz)

        node = np.array(list(product(x, y, z)))
        # Create simple tet elements (this is a placeholder)
        elem = np.array([[1, 2, 3, 4]])  # Minimal mesh
        print("  Using placeholder mesh (install iso2mesh for full mesh)")

    nn = node.shape[0]
    ne = elem.shape[0]

    print(f"  Nodes: {nn}, Elements: {ne}")

    # Configuration
    cfg = {
        "node": node,
        "elem": elem,
        "seg": np.ones(ne, dtype=int),
        "srcpos": np.array([[30, 30, 0]]),
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": np.array([[30, 30, 30]]),
        "detdir": np.array([[0, 0, -1]]),
        "prop": np.array(
            [
                [0, 0, 1, 1],  # Label 0: background
                [0.005, 1, 0, 1.37],  # Label 1: tissue (mua=0.005, mus'=1)
            ]
        ),
        "omega": 0,  # CW mode
    }

    # ============================================================
    # Prepare mesh
    # ============================================================
    print("\n>> Preparing mesh...")

    import time

    t_start = time.time()
    cfg, sd = utility.meshprep(cfg)
    t_prep = time.time() - t_start

    print(f"  Mesh preparation time: {t_prep:.4f} seconds")
    print(f"  Reff: {cfg['reff']:.4f}")

    # Verify 1-based indices
    assert cfg["elem"].min() >= 1, "elem should be 1-based"
    assert cfg["face"].min() >= 1, "face should be 1-based"
    print("  Index convention: 1-based ✓")

    # ============================================================
    # Run forward simulation
    # ============================================================
    print("\n>> Running forward simulation...")

    t_start = time.time()
    detphi, phi = rb.run(cfg)
    t_forward = time.time() - t_start

    print(f"  Forward solution time: {t_forward:.4f} seconds")
    print(f"  Detector value: {detphi[0, 0]:.6e}")
    print(f"  Phi shape: {phi.shape}")
    print(f"  Phi range: [{phi.min():.6e}, {phi.max():.6e}]")

    # ============================================================
    # Analytical solution (semi-infinite)
    # ============================================================
    print("\n>> Computing analytical solution...")

    mua = cfg["prop"][1, 0]
    musp = cfg["prop"][1, 1] * (1 - cfg["prop"][1, 2])
    reff = cfg["reff"]

    srcpos = cfg["srcpos"][0]

    phi_analytical = cwdiffusion(mua, musp, reff, srcpos, cfg["node"])

    print(
        f"  Analytical phi range: [{phi_analytical.min():.6e}, {phi_analytical.max():.6e}]"
    )

    # ============================================================
    # Comparison
    # ============================================================
    print("\n>> Comparing solutions...")

    # Compare at interior nodes (avoid boundary effects)
    interior = (cfg["node"][:, 2] > 5) & (cfg["node"][:, 2] < 25)

    if np.sum(interior) > 0:
        phi_rb = np.abs(phi[interior, 0])
        phi_an = np.abs(phi_analytical[interior])

        # Avoid division by zero
        valid = phi_an > 1e-20
        if np.sum(valid) > 0:
            ratio = phi_rb[valid] / phi_an[valid]

            print(f"  Interior nodes compared: {np.sum(valid)}")
            print(f"  Ratio (Redbird/Analytical): {np.median(ratio):.4f} (median)")
            print(f"  Ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")

    # ============================================================
    # Results summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Mesh: {nn} nodes, {ne} elements")
    print(f"  Source: {cfg['srcpos'][0]}")
    print(f"  Detector: {cfg['detpos'][0]}")
    print(f"  Optical properties: mua={mua}, mus'={musp}")
    print(f"  Detector measurement: {detphi[0, 0]:.6e}")
    print(f"  Computation time: {t_prep + t_forward:.4f} s")

    return cfg, detphi, phi, phi_analytical


def plot_results(cfg, phi, phi_analytical):
    """Plot comparison of Redbird and analytical solutions."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract slice at x=30 (approximate)
    x_slice = np.abs(cfg["node"][:, 0] - 30) < 5

    if np.sum(x_slice) > 10:
        nodes_slice = cfg["node"][x_slice]
        phi_slice = np.log10(np.abs(phi[x_slice, 0]) + 1e-20)
        phi_an_slice = np.log10(np.abs(phi_analytical[x_slice]) + 1e-20)

        # Redbird solution
        sc1 = axes[0].scatter(
            nodes_slice[:, 1], nodes_slice[:, 2], c=phi_slice, cmap="jet", s=20
        )
        axes[0].set_xlabel("Y (mm)")
        axes[0].set_ylabel("Z (mm)")
        axes[0].set_title("Redbird Solution (log10)")
        plt.colorbar(sc1, ax=axes[0])

        # Analytical solution
        sc2 = axes[1].scatter(
            nodes_slice[:, 1], nodes_slice[:, 2], c=phi_an_slice, cmap="jet", s=20
        )
        axes[1].set_xlabel("Y (mm)")
        axes[1].set_ylabel("Z (mm)")
        axes[1].set_title("Analytical Solution (log10)")
        plt.colorbar(sc2, ax=axes[1])

        # Difference
        diff = phi_slice - phi_an_slice
        sc3 = axes[2].scatter(
            nodes_slice[:, 1], nodes_slice[:, 2], c=diff, cmap="RdBu", s=20
        )
        axes[2].set_xlabel("Y (mm)")
        axes[2].set_ylabel("Z (mm)")
        axes[2].set_title("Difference (log10)")
        plt.colorbar(sc3, ax=axes[2])

    plt.tight_layout()
    plt.savefig("example_basic_results.png", dpi=150)
    print("\nPlot saved to: example_basic_results.png")
    plt.show()


if __name__ == "__main__":
    cfg, detphi, phi, phi_analytical = run_basic_example()

    # Optionally plot results
    try:
        plot_results(cfg, phi, phi_analytical)
    except Exception as e:
        print(f"Plotting skipped: {e}")
