#!/usr/bin/env python
"""
Redbird Forward Example - Forward simulation with multiple sources/detectors.

Translated from: demo_redbird_forward.m, demo_redbird_forward_expert.m

This example demonstrates:
- Multiple source and detector positions
- Frequency-domain (RF) simulation
- Step-by-step forward solution process
- Noise addition to measurements
"""

import numpy as np
import time

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbird as rb
from redbird import forward, utility, property as prop_module

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False
    raise ImportError("iso2mesh required for this example. Run: pip install iso2mesh")


def run_forward_example():
    """Run forward simulation with multiple sources."""
    print("=" * 60)
    print("Redbird Forward Example - Multiple Sources/Detectors")
    print("=" * 60)

    # ============================================================
    # Create mesh
    # ============================================================
    print("\n>> Creating mesh...")

    node, face, elem = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)

    nn = node.shape[0]
    ne = elem.shape[0]
    print(f"  Nodes: {nn}, Elements: {ne}")

    # ============================================================
    # Define source/detector grid
    # ============================================================
    xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))
    xi = xi.flatten()
    yi = yi.flatten()

    srcpos = np.column_stack([xi, yi, np.zeros(len(xi))])
    detpos = np.column_stack([xi, yi, 60 * np.ones(len(xi))])

    print(f"  Sources: {srcpos.shape[0]}, Detectors: {detpos.shape[0]}")

    # ============================================================
    # Configuration
    # ============================================================
    cfg = {
        "node": node,
        "elem": elem,
        "seg": np.ones(ne, dtype=int),
        "srcpos": srcpos,
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": detpos,
        "detdir": np.array([[0, 0, -1]]),
        "prop": np.array(
            [
                [0, 0, 1, 1],  # Label 0
                [0.008, 1, 0, 1.37],  # Label 1: mua=0.008, mus'=1
                [0.016, 1, 0, 1.37],  # Label 2: inclusion
            ]
        ),
        "omega": 2 * np.pi * 70e6,  # 70 MHz modulation
    }

    print(f"  Modulation frequency: {cfg['omega']/(2*np.pi)/1e6:.1f} MHz")

    # ============================================================
    # Prepare mesh
    # ============================================================
    print("\n>> Preparing mesh...")

    t_start = time.time()
    cfg, sd = utility.meshprep(cfg)
    print(f"  Preparation time: {time.time() - t_start:.4f} s")

    # ============================================================
    # Method 1: One-liner forward solution
    # ============================================================
    print("\n>> Running forward (one-liner)...")

    t_start = time.time()
    detphi, phi = rb.run(cfg)
    t_forward = time.time() - t_start

    print(f"  Forward time: {t_forward:.4f} s")
    print(f"  Phi shape: {phi.shape}")
    print(f"  Detphi shape: {detphi.shape}")

    # ============================================================
    # Method 2: Step-by-step (expert mode)
    # ============================================================
    print("\n>> Running forward (step-by-step)...")

    # Step 1: Build deldotdel
    t_start = time.time()
    deldotdel, delphi = forward.deldotdel(cfg)
    print(f"  deldotdel time: {time.time() - t_start:.4f} s")
    print(f"  deldotdel shape: {deldotdel.shape}")

    # Step 2: Build LHS (stiffness matrix)
    t_start = time.time()
    Amat = forward.femlhs(cfg, deldotdel)
    print(f"  LHS build time: {time.time() - t_start:.4f} s")
    print(f"  Amat shape: {Amat.shape}, nnz: {Amat.nnz}")

    # Step 3: Build RHS
    t_start = time.time()
    rhs, loc, bary, optode = forward.femrhs(cfg, sd)
    print(f"  RHS build time: {time.time() - t_start:.4f} s")
    print(f"  RHS shape: {rhs.shape}")

    # Step 4: Solve
    t_start = time.time()
    phi_expert, flag = forward.femsolve(Amat, rhs)
    print(f"  Solve time: {time.time() - t_start:.4f} s")
    print(f"  Solver flag: {flag}")

    # Step 5: Extract detector values
    detval_expert = forward.femgetdet(phi_expert, cfg, loc, bary)
    print(f"  Detval shape: {detval_expert.shape}")

    # ============================================================
    # Compare methods
    # ============================================================
    print("\n>> Comparing methods...")

    diff = np.abs(detphi - detval_expert)
    rel_diff = diff / (np.abs(detphi) + 1e-20)

    print(f"  Max absolute difference: {diff.max():.6e}")
    print(f"  Max relative difference: {rel_diff.max():.6e}")

    # ============================================================
    # Build Jacobian
    # ============================================================
    print("\n>> Building Jacobian...")

    t_start = time.time()
    Jmua_n, Jmua_e = forward.jac(sd, phi, deldotdel, cfg["elem"], cfg["evol"])
    print(f"  Jacobian time: {time.time() - t_start:.4f} s")
    print(f"  Jmua (node) shape: {Jmua_n.shape}")
    print(f"  Jmua (elem) shape: {Jmua_e.shape}")
    print(f"  Jmua sum: {np.sum(Jmua_n):.6e}")

    # ============================================================
    # Add noise to measurements
    # ============================================================
    print("\n>> Adding noise...")

    snr_shot = 110  # dB
    snr_thermal = 40  # dB

    detphi_noisy = utility.addnoise(detphi, snr_shot, snr_thermal)

    noise_level = np.abs(detphi_noisy - detphi) / (np.abs(detphi) + 1e-20)
    print(f"  SNR (shot): {snr_shot} dB, SNR (thermal): {snr_thermal} dB")
    print(f"  Relative noise level: {np.mean(noise_level)*100:.2f}%")

    # ============================================================
    # Results summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Measurement pairs: {detphi.size}")
    print(
        f"  Amplitude range: [{np.abs(detphi).min():.6e}, {np.abs(detphi).max():.6e}]"
    )

    if np.iscomplexobj(detphi):
        phase = np.angle(detphi)
        print(
            f"  Phase range: [{np.degrees(phase.min()):.2f}°, {np.degrees(phase.max()):.2f}°]"
        )

    return cfg, detphi, phi, Jmua_n


def run_cw_example():
    """Run CW (omega=0) forward simulation for comparison."""
    print("\n" + "=" * 60)
    print("Redbird Forward Example - CW Mode")
    print("=" * 60)

    node, face, elem = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)

    xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))
    srcpos = np.column_stack([xi.flatten(), yi.flatten(), np.zeros(xi.size)])
    detpos = np.column_stack([xi.flatten(), yi.flatten(), 60 * np.ones(xi.size)])

    cfg = {
        "node": node,
        "elem": elem,
        "seg": np.ones(elem.shape[0], dtype=int),
        "srcpos": srcpos,
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": detpos,
        "detdir": np.array([[0, 0, -1]]),
        "prop": np.array([[0, 0, 1, 1], [0.008, 1, 0, 1.37]]),
        "omega": 0,  # CW mode
    }

    cfg, sd = utility.meshprep(cfg)

    t_start = time.time()
    detphi_cw, phi_cw = rb.run(cfg)
    print(f"  CW forward time: {time.time() - t_start:.4f} s")

    print(f"  CW detphi is real: {np.isreal(detphi_cw).all()}")
    print(f"  CW amplitude range: [{detphi_cw.min():.6e}, {detphi_cw.max():.6e}]")

    return cfg, detphi_cw, phi_cw


if __name__ == "__main__":
    # Run RF example
    cfg_rf, detphi_rf, phi_rf, Jmua = run_forward_example()

    # Run CW example
    cfg_cw, detphi_cw, phi_cw = run_cw_example()

    print("\n" + "=" * 60)
    print("RF vs CW Comparison")
    print("=" * 60)
    print(f"  RF amplitude mean: {np.abs(detphi_rf).mean():.6e}")
    print(f"  CW amplitude mean: {np.abs(detphi_cw).mean():.6e}")
