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

import redbirdpy as rb
from redbirdpy.analytical import semi_infinite_cw
from redbirdpy.utility import meshprep

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False
    print("Warning: iso2mesh not installed. Using simple mesh.")


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
    cfg, sd = meshprep(cfg)
    t_prep = time.time() - t_start

    print(f"  Mesh preparation time: {t_prep:.4f} seconds")

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

    srcpos = cfg["srcpos"][0]

    phi_analytical = semi_infinite_cw(
        mua, musp, cfg["prop"][1, -1], cfg["prop"][0, -1], srcpos, cfg["node"]
    )

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


if __name__ == "__main__":
    cfg, detphi, phi, phi_analytical = run_basic_example()

    # Optionally plot results
    try:
        cutpos, phival, facedata = i2m.qmeshcut(
            cfg["elem"][:, :4], cfg["node"][:, :3], phi[:, 0], "x = 20"
        )[:3]
        hh = i2m.plotmesh(
            np.c_[cutpos, np.log10(np.abs(phival) + 1e-20)],
            facedata.tolist(),
            subplot=131,
            hold="on",
        )
        cutpos, cutval, facedata = i2m.qmeshcut(
            cfg["elem"][:, :4], cfg["node"][:, :3], phi_analytical, "x = 20"
        )[:3]
        i2m.plotmesh(
            np.c_[cutpos, np.log10(np.abs(cutval) + 1e-20)],
            facedata.tolist(),
            subplot=132,
            parent=hh,
            hold="on",
        )
        i2m.plotmesh(
            np.c_[
                cutpos,
                np.log10(np.abs(cutval) + 1e-20) - np.log10(np.abs(phival) + 1e-20),
            ],
            facedata.tolist(),
            subplot=133,
            parent=hh,
        )
        plt.show(block=False)

    except Exception as e:
        print(f"Plotting skipped: {e}")
