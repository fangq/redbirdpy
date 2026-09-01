#!/usr/bin/env python
"""
Redbird - Monte Carlo (pmmc) based CW DOT reconstruction.

Port of redbird/example/demo_redbird_recon_mc.m.

Continuous-Wave (CW) reconstruction of an absorption (mua) inclusion
using pmmc as the forward-and-adjoint engine. Mirrors the streamlined
FEM example demo_redbird_recon.py but adds ``cfg.nphoton`` to route
through pmmc. The mesh-mode adjoint Jacobian returned by pmmc is
consumed by ``runrecon`` directly, bypassing the FEM ``jac()`` build.

Third demo in the pmmc-integration trio:
    demo_redbird_forward_mc.py   - forward simulation via pmmc
    demo_redbird_jacobian_mc.py  - mesh-mode adjoint Jacobian via pmmc
    demo_redbird_recon_mc.py     - this demo (CW DOT reconstruction)

Photon counts of order 1e7 keep the per-iteration Monte Carlo noise
small enough that Gauss-Newton converges in a handful of steps.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbirdpy as rb

try:
    import pmmc  # noqa: F401
except ImportError:
    print("Error: pmmc is not installed; this demo requires the MMC Python binding.")
    sys.exit(1)

try:
    import iso2mesh as i2m
except ImportError:
    print("Error: iso2mesh is required to build the tet mesh.")
    sys.exit(1)


def main():
    # ----------------------------------------------------------------
    # ground-truth heterogeneous domain.  iso2mesh's meshabox returns
    # (node, face, elem); meshasphere returns (node, face); s2m
    # tetrahedralizes a closed surface.  For simplicity we build a
    # background slab and merge in a small spherical absorber.
    # ----------------------------------------------------------------
    s0 = np.array([70, 50, 20], dtype=float)

    nobbx, fcbbx, _ = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)
    nosp, fcsp = i2m.meshasphere(s0, 5.0, 1.0)[:2]
    no, fc = i2m.mergemesh(nobbx, fcbbx[:, :3], nosp, fcsp[:, :3])

    node, elem = i2m.s2m(
        no, fc[:, :3], 1.0, 40.0, "tetgen", np.vstack([[41, 1, 1], s0])
    )[:2]

    cfg0 = {
        "node": node,
        "elem": elem,
        "seg": elem[:, 4] if elem.shape[1] > 4 else np.ones(elem.shape[0], dtype=int),
        "srcdir": np.array([[0, 0, 1]], dtype=float),
    }

    xi, yi = np.meshgrid(np.arange(60, 141, 20), np.arange(20, 101, 20))
    cfg0["srcpos"] = np.column_stack([xi.flatten(), yi.flatten(), np.zeros(xi.size)])
    cfg0["detpos"] = np.column_stack(
        [xi.flatten(), yi.flatten(), 60.0 * np.ones(xi.size)]
    )
    cfg0["detdir"] = np.array([[0, 0, -1]], dtype=float)

    cfg0["prop"] = np.array(
        [
            [0, 0, 1, 1],
            [0.008, 1, 0, 1.37],  # background
            [0.016, 1, 0, 1.37],  # absorbing inclusion (2x background mua)
        ],
        dtype=float,
    )
    cfg0["omega"] = 0

    cfg = {k: v for k, v in cfg0.items()}
    cfg0, _ = rb.utility.meshprep(cfg0)

    # ----------------------------------------------------------------
    # Synthesize measurement data with the FEM forward
    # (acts as a "perfect" measurement; in practice this would be
    # boundary-detector readings or a noisier mmclab forward)
    # ----------------------------------------------------------------
    print("Synthesizing measurements via FEM forward ...")
    detphi0, _ = rb.forward.runforward(cfg0)
    print(f"  detphi0 shape: {np.asarray(detphi0).shape}")

    # ----------------------------------------------------------------
    # Forward mesh + coarse recon mesh for the inversion
    # ----------------------------------------------------------------
    node_fwd, _, elem_fwd = i2m.meshabox([40, 0, 0], [160, 120, 60], 10)
    cfg = rb.utility.meshprep(
        {
            "node": node_fwd,
            "elem": elem_fwd,
            "seg": np.ones(elem_fwd.shape[0], dtype=int),
            "srcpos": cfg["srcpos"],
            "srcdir": cfg["srcdir"],
            "detpos": cfg["detpos"],
            "prop": cfg["prop"],
            "omega": cfg["omega"],
        }
    )[0]

    # MC path: route cfg.nphoton through to pmmc. detdir is auto-filled
    # from the surface mesh by getdetdir inside _runforward_mc when missing.
    cfg["nphoton"] = int(1e7)
    cfg["gpuid"] = 1

    sd = rb.utility.sdmap(cfg)

    # coarse reconstruction mesh
    recon_node, _, recon_elem = i2m.meshabox([40, 0, 0], [160, 120, 60], 20)
    if recon_elem.shape[1] > 4:
        recon_elem = recon_elem[:, :4]
    # mapid + barycentric weights via iso2mesh tsearchn (1-based mapid; the
    # same convention used by demo_redbird_recon.py).
    mapid, mapweight = i2m.tsearchn(recon_node, recon_elem, cfg["node"])
    recon = {
        "node": recon_node,
        "elem": recon_elem,
        "mapid": mapid,
        "mapweight": mapweight,
        "lambda": 1e-4,
    }

    # ----------------------------------------------------------------
    # Stage 1: bulk mua/musp fit (single segment)
    # ----------------------------------------------------------------
    # Start from a deliberately poor initial guess. rb.run(mode='bulk') sets
    # recon.seg = ones(Nnode), collapsing the Jacobian columns into a single
    # per-segment unknown. Each Gauss-Newton step adjusts one global mua so
    # we get a fast sanity check on the MC forward+adjoint before per-node
    # imaging starts. Mirrors demo_redbird_recon_mc.m in redbird.
    cfg.pop("seg", None)
    recon["bulk"] = {"mua": 0.003, "musp": 0.6}

    print("\n=== Stage 1: bulk mua/musp fit ===")
    import time

    t0 = time.time()
    newrecon = rb.run(cfg, recon, detphi0, sd, mode="bulk", lambda_=1e-3, maxiter=5)[0]
    print(f"  done in {time.time() - t0:.1f} s")
    print(
        f"  bulk fit: mua = {newrecon['prop'][1, 0]:g}, "
        f"musp = {newrecon['prop'][1, 1]:g}"
    )

    # ----------------------------------------------------------------
    # Stage 2: per-node image recon seeded with stage-1 bulk
    # ----------------------------------------------------------------
    # Re-seed recon.bulk from the stage-1 fit and let rb.run(mode='image')
    # rebuild per-node recon.prop and cfg.prop from those bulk values.
    # Per-node cfg.prop is auto-detected by _runforward_mc and split into
    # cfg.nodemua/cfg.nodemusp for the mmc per-node kernel; runrecon picks
    # up the Jacobian from runforward's Jext return and skips the FEM jac().
    recon["bulk"] = {
        "mua": float(newrecon["prop"][1, 0]),
        "musp": float(newrecon["prop"][1, 1]),
    }
    recon.pop("prop", None)  # let _apply_run_mode rebuild per-node from new bulk

    print("\n=== Stage 2: per-node image reconstruction ===")
    t0 = time.time()
    out = rb.run(cfg, recon, detphi0, sd, mode="image", lambda_=1e-4, maxiter=5)
    newrecon, resid, newcfg = out[:3]
    print(f"  done in {time.time() - t0:.1f} s")
    print(f"  residual trajectory: {resid}")

    # ----------------------------------------------------------------
    # Plot reconstructed mua slices (optional)
    # ----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping reconstruction plot.")
        return

    mua = newcfg["prop"][:, 0]
    nodes = newcfg["node"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (axis, val, lab) in zip(axes, [(2, 20.0, "z=20"), (0, 70.0, "x=70")]):
        mask = np.abs(nodes[:, axis] - val) < 5
        if axis == 2:
            xs, ys, c = nodes[mask, 0], nodes[mask, 1], mua[mask]
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        else:
            xs, ys, c = nodes[mask, 1], nodes[mask, 2], mua[mask]
            ax.set_xlabel("y (mm)")
            ax.set_ylabel("z (mm)")
        sc = ax.scatter(xs, ys, c=c, cmap="hot", s=5)
        ax.set_title(f"Reconstructed mua, {lab}")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax)

    plt.suptitle(
        "MC-based mua reconstruction (5 Gauss-Newton iterations, 1e7 photons/iter)"
    )
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "demo_redbird_recon_mc.png")
    plt.savefig(out_path, dpi=100)
    print(f"\nReconstruction plot saved to: {out_path}")


if __name__ == "__main__":
    main()
