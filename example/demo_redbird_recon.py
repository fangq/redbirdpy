#!/usr/bin/env python
"""
Redbird Reconstruction Example - Image reconstruction with Gauss-Newton.

Translated from: demo_redbird_recon.m, demo_redbird_recon_singlemesh.m

This example demonstrates:
- Generating synthetic measurement data
- Setting up reconstruction with dual mesh
- Running iterative Gauss-Newton reconstruction
- Bulk property fitting
"""

import numpy as np
import time

try:
    import redbird as rb
    from redbird import forward, recon, utility, property as prop_module
except ImportError:
    raise ImportError("redbird not installed")

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    raise ImportError("iso2mesh required. Run: pip install iso2mesh")


def create_target_mesh():
    """Create mesh with spherical inclusion for reconstruction target."""
    print(">> Creating target mesh with inclusion...")

    s0 = [70, 50, 20]  # Inclusion center

    # Create bounding box
    nobbx, fcbbx = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)

    # Create spherical inclusion
    nosp, fcsp = i2m.meshasphere(s0, 5, 1)

    # Merge meshes
    no, fc = i2m.mergemesh(nobbx, fcbbx, nosp, fcsp)

    # Generate tetrahedral mesh with region labels
    regionseed = np.array([[41, 1, 1], s0])  # Inside box  # Inside sphere

    node, elem = i2m.s2m(no, fc[:, :3], 1, 40, "tetgen", regionseed)

    seg = (
        elem[:, 4].astype(int)
        if elem.shape[1] > 4
        else np.ones(elem.shape[0], dtype=int)
    )
    elem_4 = elem[:, :4]

    print(f"  Target mesh: {node.shape[0]} nodes, {elem_4.shape[0]} elements")
    print(f"  Inclusion center: {s0}")

    return node, elem_4, seg, s0


def run_recon_example():
    """Run basic reconstruction example."""
    print("=" * 60)
    print("Redbird Reconstruction Example")
    print("=" * 60)

    # ============================================================
    # Create target (truth) mesh with inclusion
    # ============================================================
    node_target, elem_target, seg_target, inclusion_center = create_target_mesh()

    # Source/detector grid
    xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))
    srcpos = np.column_stack([xi.flatten(), yi.flatten(), np.zeros(xi.size)])
    detpos = np.column_stack([xi.flatten(), yi.flatten(), 60 * np.ones(xi.size)])

    # Target configuration (ground truth)
    cfg0 = {
        "node": node_target,
        "elem": elem_target,
        "seg": seg_target,
        "srcpos": srcpos,
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": detpos,
        "detdir": np.array([[0, 0, -1]]),
        "prop": np.array(
            [
                [0, 0, 1, 1],  # Label 0
                [0.008, 1, 0, 1.37],  # Background: mua=0.008
                [0.016, 1, 0, 1.37],  # Inclusion: mua=0.016 (2x background)
            ]
        ),
        "omega": 0,  # CW mode
    }

    print(f"\n  Sources: {srcpos.shape[0]}, Detectors: {detpos.shape[0]}")
    print(f"  Background mua: {cfg0['prop'][1, 0]}")
    print(f"  Inclusion mua: {cfg0['prop'][2, 0]}")

    # ============================================================
    # Generate synthetic measurement data
    # ============================================================
    print("\n>> Generating synthetic data...")

    cfg0, sd0 = utility.meshprep(cfg0)

    t_start = time.time()
    detphi0, _ = forward.runforward(cfg0)
    print(f"  Forward time: {time.time() - t_start:.4f} s")
    print(f"  Measurement shape: {detphi0.shape}")
    print(f"  Measurement range: [{detphi0.min():.6e}, {detphi0.max():.6e}]")

    # ============================================================
    # Create reconstruction mesh (homogeneous initial guess)
    # ============================================================
    print("\n>> Setting up reconstruction...")

    # Forward mesh for reconstruction
    node_fwd, face_fwd, elem_fwd = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)

    cfg = {
        "node": node_fwd,
        "elem": elem_fwd,
        "seg": np.ones(elem_fwd.shape[0], dtype=int),
        "srcpos": srcpos,
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": detpos,
        "detdir": np.array([[0, 0, -1]]),
        "prop": cfg0["prop"].copy(),
        "omega": 0,
    }

    cfg = prop_module.setmesh(
        cfg, node_fwd, elem_fwd, cfg["prop"], np.ones(elem_fwd.shape[0], dtype=int)
    )

    sd = utility.sdmap(cfg)

    # Coarse reconstruction mesh (dual mesh)
    node_recon, _, elem_recon = i2m.meshabox([40, 0, 0], [160, 120, 60], 20)

    # Compute mapping from recon mesh to forward mesh
    # mapid: which recon element contains each forward node
    # mapweight: barycentric coordinates
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(node_recon)
        mapid = tri.find_simplex(cfg["node"])

        # Compute barycentric weights (simplified)
        mapweight = np.ones((cfg["node"].shape[0], 4)) * 0.25

    except Exception as e:
        print(f"  Warning: tsearchn equivalent failed: {e}")
        mapid = np.zeros(cfg["node"].shape[0])
        mapweight = np.ones((cfg["node"].shape[0], 4)) * 0.25

    print(f"  Forward mesh: {cfg['node'].shape[0]} nodes")
    print(f"  Recon mesh: {node_recon.shape[0]} nodes")

    # ============================================================
    # Initialize reconstruction structure
    # ============================================================
    recon_cfg = {
        "node": node_recon,
        "elem": elem_recon,
        "mapid": mapid.astype(float),
        "mapweight": mapweight,
        "prop": cfg["prop"][np.ones(node_recon.shape[0], dtype=int) + 1, :].copy(),
        "lambda": 1e-4,  # Regularization parameter
    }

    # Set initial guess to homogeneous
    nn_recon = node_recon.shape[0]
    recon_cfg["prop"] = np.tile(cfg["prop"][1, :], (nn_recon, 1))

    # Also set forward mesh to homogeneous
    nn_fwd = cfg["node"].shape[0]
    cfg["prop"] = np.tile(cfg["prop"][1, :], (nn_fwd, 1))
    if "seg" in cfg:
        del cfg["seg"]

    # ============================================================
    # Run reconstruction
    # ============================================================
    print("\n>> Running reconstruction...")

    t_start = time.time()

    newrecon, resid, newcfg, updates, Jmua, detphi_final, phi_final = recon.runrecon(
        cfg, recon_cfg, detphi0, sd, maxiter=10, lambda_=1e-4, report=True
    )

    print(f"\n  Total reconstruction time: {time.time() - t_start:.4f} s")
    print(f"  Initial residual: {resid[0]:.6e}")
    print(f"  Final residual: {resid[-1]:.6e}")
    print(f"  Residual reduction: {resid[0]/resid[-1]:.2f}x")

    # ============================================================
    # Analyze results
    # ============================================================
    print("\n>> Analyzing results...")

    mua_recon = newrecon["prop"][:, 0] if "prop" in newrecon else newcfg["prop"][:, 0]

    print(f"  Reconstructed mua range: [{mua_recon.min():.6f}, {mua_recon.max():.6f}]")
    print(f"  True background mua: {cfg0['prop'][1, 0]:.6f}")
    print(f"  True inclusion mua: {cfg0['prop'][2, 0]:.6f}")

    # Find max mua location
    max_idx = np.argmax(mua_recon)
    if "node" in newrecon:
        max_loc = newrecon["node"][max_idx]
    else:
        max_loc = newcfg["node"][max_idx]

    print(f"  Max mua location: {max_loc}")
    print(f"  True inclusion center: {inclusion_center}")

    dist_error = np.sqrt(np.sum((max_loc - inclusion_center) ** 2))
    print(f"  Localization error: {dist_error:.2f} mm")

    return cfg, recon_cfg, detphi0, newrecon, resid


def run_singlemesh_recon():
    """
    Run reconstruction on single mesh (no dual mesh).

    Translated from: demo_redbird_recon_singlemesh.m
    """
    print("\n" + "=" * 60)
    print("Redbird Single-Mesh Reconstruction")
    print("=" * 60)

    # Create target mesh
    s0 = [70, 50, 20]
    nobbx, fcbbx = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)
    nosp, fcsp = i2m.meshasphere(s0, 5, 1)
    no, fc = i2m.mergemesh(nobbx, fcbbx, nosp, fcsp)

    regionseed = np.array([[41, 1, 1], s0])
    node, elem = i2m.s2m(no, fc[:, :3], 1, 20, "tetgen", regionseed)

    seg = (
        elem[:, 4].astype(int)
        if elem.shape[1] > 4
        else np.ones(elem.shape[0], dtype=int)
    )
    elem = elem[:, :4]

    xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))
    srcpos = np.column_stack([xi.flatten(), yi.flatten(), np.zeros(xi.size)])
    detpos = np.column_stack([xi.flatten(), yi.flatten(), 60 * np.ones(xi.size)])

    # Generate data
    cfg0 = {
        "node": node,
        "elem": elem,
        "seg": seg,
        "srcpos": srcpos,
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": detpos,
        "detdir": np.array([[0, 0, -1]]),
        "prop": np.array([[0, 0, 1, 1], [0.008, 1, 0, 1.37], [0.016, 1, 0, 1.37]]),
        "omega": 2 * np.pi * 70e6,  # 70 MHz
    }

    cfg0, sd = utility.meshprep(cfg0)
    detphi0, _ = forward.runforward(cfg0)

    print(f"  Generated data with RF modulation")
    print(f"  Measurement is complex: {np.iscomplexobj(detphi0)}")

    # Reset to homogeneous for reconstruction
    node_fwd, _, elem_fwd = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)

    cfg = {
        "node": node_fwd,
        "elem": elem_fwd,
        "seg": np.ones(elem_fwd.shape[0], dtype=int),
        "srcpos": srcpos,
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": detpos,
        "detdir": np.array([[0, 0, -1]]),
        "prop": cfg0["prop"].copy(),
        "omega": cfg0["omega"],
    }

    cfg, sd = utility.meshprep(cfg)

    # Single-mesh reconstruction
    nn = cfg["node"].shape[0]
    recon_cfg = {"prop": np.tile(cfg["prop"][1, :], (nn, 1)), "lambda": 1e-3}

    cfg["prop"] = np.tile(cfg["prop"][1, :], (nn, 1))
    del cfg["seg"]

    print("\n>> Running single-mesh reconstruction...")

    newrecon, resid, *_ = recon.runrecon(
        cfg, recon_cfg, detphi0, sd, maxiter=5, lambda_=1e-3, reform="real", report=True
    )

    print(f"\n  Residual reduction: {resid[0]/resid[-1]:.2f}x")

    return cfg, detphi0, newrecon, resid


if __name__ == "__main__":
    # Run dual-mesh reconstruction
    cfg, recon_cfg, detphi0, newrecon, resid = run_recon_example()

    # Run single-mesh reconstruction
    try:
        cfg_sm, detphi0_sm, newrecon_sm, resid_sm = run_singlemesh_recon()
    except Exception as e:
        print(f"\nSingle-mesh example error: {e}")
