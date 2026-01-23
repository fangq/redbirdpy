#!/usr/bin/env python
"""
Redbird Heterogeneous Example - Forward simulation with inclusion.

Translated from: demo_redbird_forward_heterogeneous.m

This example demonstrates:
- Creating mesh with embedded inclusion
- Multi-region segmentation
- Forward simulation with heterogeneous properties
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbird as rb
from redbird import forward, utility

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False
    raise ImportError("iso2mesh required. Run: pip install iso2mesh")


def create_box_with_inclusion():
    """Create a box mesh with an embedded box inclusion."""
    print(">> Creating mesh with inclusion...")

    boxsize = [60, 50, 40]
    trisize = 4
    maxvol = 4

    t_start = time.time()

    # Create outer box
    nbox1, fbox1, _ = i2m.meshabox([0, 0, 0], boxsize, trisize)

    # Create inner box (inclusion)
    nbox2, fbox2, _ = i2m.meshabox([10, 10, 10], [30, 30, 30], trisize)

    # Clean isolated nodes
    nbox1, fbox1, _ = i2m.removeisolatednode(nbox1, fbox1[:, :3])
    nbox2, fbox2, _ = i2m.removeisolatednode(nbox2, fbox2[:, :3])

    # Merge meshes
    no1, fc1 = i2m.mergemesh(nbox1, fbox1, nbox2, fbox2)

    # Region seeds: one inside large box, one inside small box
    regionseed = np.array(
        [
            [1, 1, 1],  # Inside region 1 (large box)
            [11, 11, 11],  # Inside region 2 (small box)
        ]
    )

    # Generate tetrahedral mesh
    node, elem, _ = i2m.s2m(no1, fc1, 1, maxvol, "tetgen", regionseed)

    print(f"  Mesh creation time: {time.time() - t_start:.4f} s")
    print(f"  Nodes: {node.shape[0]}, Elements: {elem.shape[0]}")

    # Extract segmentation from 5th column
    seg = (
        elem[:, 4].astype(int)
        if elem.shape[1] > 4
        else np.ones(elem.shape[0], dtype=int)
    )
    elem = elem[:, :4]

    # Count elements in each region
    unique, counts = np.unique(seg, return_counts=True)
    print(f"  Regions: {dict(zip(unique, counts))}")

    return node, elem, seg


def run_heterogeneous_example():
    """Run forward simulation on heterogeneous domain."""
    print("=" * 60)
    print("Redbird Heterogeneous Example")
    print("=" * 60)

    # ============================================================
    # Create mesh with inclusion
    # ============================================================
    node, elem, seg = create_box_with_inclusion()

    # ============================================================
    # Configuration
    # ============================================================
    print("\n>> Setting up configuration...")

    cfg = {
        "node": node,
        "elem": elem,
        "seg": seg,
        "srcpos": np.array([[25, 25, 0]]),
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": np.array([[35, 25, 40]]),  # On top surface
        "detdir": np.array([[0, 0, -1]]),
        "prop": np.array(
            [
                [0, 0, 1, 1],  # Label 0: background (outside)
                [0.006, 0.8, 0, 1.37],  # Label 1: background domain
                [0.02, 1, 0, 1.37],  # Label 2: inclusion (higher absorption)
            ]
        ),
        "omega": 0,  # CW mode
    }

    print(f"  Source: {cfg['srcpos'][0]}")
    print(f"  Detector: {cfg['detpos'][0]}")
    print("  Properties:")
    print(f"    Background: mua={cfg['prop'][1, 0]}, mus'={cfg['prop'][1, 1]}")
    print(f"    Inclusion:  mua={cfg['prop'][2, 0]}, mus'={cfg['prop'][2, 1]}")

    # ============================================================
    # Prepare mesh
    # ============================================================
    print("\n>> Preparing mesh...")

    t_start = time.time()
    cfg, sd = utility.meshprep(cfg)
    print(f"  Preparation time: {time.time() - t_start:.4f} s")

    # ============================================================
    # Run forward simulation
    # ============================================================
    print("\n>> Running forward simulation...")

    t_start = time.time()
    detphi, phi = rb.forward.runforward(cfg)
    print(f"  Forward time: {time.time() - t_start:.4f} s")

    print(f"  Phi shape: {phi.shape}")
    print(f"  Phi (from source) range: [{phi[:, 0].min():.6e}, {phi[:, 0].max():.6e}]")
    print(
        f"  Phi (from detector) range: [{phi[:, 1].min():.6e}, {phi[:, 1].max():.6e}]"
    )
    print(f"  Detector value: {detphi[0, 0]:.6e}")

    # ============================================================
    # Compare with homogeneous
    # ============================================================
    print("\n>> Comparing with homogeneous medium...")

    cfg_homo = cfg.copy()
    cfg_homo["seg"] = np.ones(elem.shape[0], dtype=int)
    cfg_homo, sd_homo = utility.meshprep(cfg_homo)

    detphi_homo, phi_homo = rb.forward.runforward(cfg_homo)

    print(f"  Homogeneous detector value: {detphi_homo[0, 0]:.6e}")
    print(f"  Heterogeneous detector value: {detphi[0, 0]:.6e}")
    print(f"  Ratio (het/homo): {detphi[0, 0] / detphi_homo[0, 0]:.4f}")

    # The inclusion has higher absorption, so signal should be lower
    if detphi[0, 0] < detphi_homo[0, 0]:
        print("  ✓ Inclusion reduces signal (as expected)")

    return cfg, detphi, phi


def run_layered_example():
    """
    Run forward simulation on layered domain.

    Translated from: demo_redbird_forward_layered.m
    """
    print("\n" + "=" * 60)
    print("Redbird Layered Example")
    print("=" * 60)

    # ============================================================
    # Create 3-layer mesh
    # ============================================================
    print("\n>> Creating 3-layer mesh...")

    maxvol = 4

    # Create a 3-layer z-lattice
    no1, fc1, regionseeds = i2m.latticegrid([0, 60], [0, 50], [0, 5, 10, 30])
    node, elem, _ = i2m.s2m(no1, fc1, 1, maxvol, "tetgen", regionseeds)

    # Extract segmentation
    seg = (
        elem[:, 4].astype(int)
        if elem.shape[1] > 4
        else np.ones(elem.shape[0], dtype=int)
    )
    elem = elem[:, :4]

    print(f"  Nodes: {node.shape[0]}, Elements: {elem.shape[0]}")

    unique, counts = np.unique(seg, return_counts=True)
    print(f"  Layers: {dict(zip(unique, counts))}")

    # ============================================================
    # Configuration with multiple sources/detectors
    # ============================================================
    xi, yi = np.meshgrid(np.arange(5, 51, 5), np.arange(5, 41, 5))
    srcpos = np.column_stack([xi.flatten(), yi.flatten(), np.ones(xi.size)])

    xi, yi = np.meshgrid(np.arange(7.5, 51, 5), np.arange(7.5, 41, 5))
    detpos = np.column_stack([xi.flatten(), yi.flatten(), np.ones(xi.size)])

    cfg = {
        "node": node,
        "elem": elem,
        "seg": seg,
        "srcpos": srcpos,
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": detpos,
        "detdir": np.array([[0, 0, 1]]),
        "prop": np.array(
            [
                [0, 0, 1, 1],  # Label 0
                [0.006, 0.8, 0, 1.37],  # Layer 1 (0 < z < 5)
                [0.02, 0.4, 0, 1.37],  # Layer 2 (5 < z < 10)
                [0.002, 1, 0, 1.37],  # Layer 3 (10 < z < 30)
            ]
        ),
        "omega": 0,
    }

    print(f"  Sources: {srcpos.shape[0]}, Detectors: {detpos.shape[0]}")

    # ============================================================
    # Run simulation
    # ============================================================
    print("\n>> Running forward simulation...")

    t_start = time.time()
    cfg, sd = utility.meshprep(cfg)
    detphi, phi = rb.forward.runforward(cfg)
    print(f"  Total time: {time.time() - t_start:.4f} s")

    print(f"  Detphi shape: {detphi.shape}")
    print(f"  Measurement range: [{detphi.min():.6e}, {detphi.max():.6e}]")

    return cfg, detphi, phi


if __name__ == "__main__":
    # Run heterogeneous example
    cfg_het, detphi_het, phi_het = run_heterogeneous_example()

    # Run layered example
    try:
        cfg_lay, detphi_lay, phi_lay = run_layered_example()
    except Exception as e:
        print(f"\nLayered example skipped: {e}")
