#!/usr/bin/env python
"""
Redbird - Forward simulation via pmmc (Monte Carlo).

Port of redbird/example/demo_redbird_forward_mc.m.

`runforward` automatically routes to pmmc when ``cfg["nphoton"]`` is set
(MC branch); without ``nphoton`` it uses the diffusion FEM solver. This
demo runs both on the same tet mesh and compares the boundary fluence
at a set of detectors.

The MC branch passes ``cfg.srcpos`` as an Nsrc-by-3 matrix in a single
batched pmmc call. ``flux.data`` is reshaped back into the redbird
``[Nnode x (Nsrc + Ndet)]`` convention (forward slots followed by
detector-as-adjoint slots when ``cfg.detdir`` is set).

Companion demos:
    demo_redbird_jacobian_mc.py   - mesh-mode adjoint Jacobian
    demo_redbird_recon_mc.py      - CW DOT reconstruction via MC

Requires: pmmc on the Python path (``pip install pmmc`` or build from
mmc/pmmc/).
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
    print("Install via 'pip install pmmc' or from mmc/pmmc/.")
    sys.exit(1)

try:
    import iso2mesh as i2m
except ImportError:
    print("Error: iso2mesh is required to build the tet mesh.")
    sys.exit(1)


def main():
    # ----------------------------------------------------------------
    # shared geometry: 60 x 60 x 30 mm box
    # ----------------------------------------------------------------
    node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 4)

    cfg = {
        "node": node,
        "elem": elem,
        "seg": np.ones(elem.shape[0], dtype=int),
        # three source positions spread along the +x face of the slab
        "srcpos": np.array([[20, 30, 0], [30, 30, 0], [40, 30, 0]], dtype=float),
        "srcdir": np.array([[0, 0, 1]], dtype=float),  # broadcast to all three
        # detectors at increasing s-d separation on the SAME (z=0) surface
        "detpos": np.array(
            [[15, 30, 0], [25, 30, 0], [35, 30, 0], [45, 30, 0]], dtype=float
        ),
        "detdir": np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float
        ),
        "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37]], dtype=float),
        "omega": 0,  # CW
    }

    Nsrc = cfg["srcpos"].shape[0]
    Ndet = cfg["detpos"].shape[0]

    # ----------------------------------------------------------------
    # FEM forward (no cfg.nphoton)
    # ----------------------------------------------------------------
    import time

    cfg_fem, _ = rb.utility.meshprep({k: v for k, v in cfg.items()})

    print("FEM forward ...")
    t0 = time.time()
    detphi_fem, phi_fem = rb.forward.runforward(cfg_fem)
    print(f"  done in {time.time() - t0:.2f} s   detphi shape: {detphi_fem.shape}")

    # ----------------------------------------------------------------
    # MC forward via pmmc (cfg.nphoton triggers the MC branch)
    # ----------------------------------------------------------------
    cfg_mc = {k: v for k, v in cfg.items()}
    cfg_mc["nphoton"] = int(1e7)
    cfg_mc["gpuid"] = 1

    print("pmmc forward (1e7 photons) ...")
    t0 = time.time()
    detphi_mc, phi_mc = rb.forward.runforward(cfg_mc)
    print(
        f"  done in {time.time() - t0:.2f} s   "
        f"detphi shape: {detphi_mc.shape}   phi shape: {phi_mc.shape}"
    )

    # phi_mc is (Nnode, Nsrc + Ndet) - forward fluence in columns 0..Nsrc-1,
    # detector-as-adjoint fluence in columns Nsrc..Nsrc+Ndet-1.

    # ----------------------------------------------------------------
    # detector-fluence agreement
    # ----------------------------------------------------------------
    # Reshape FEM detphi to (Ndet, Nsrc); the MC detphi is already
    # (Ndet, Nsrc) from rb.forward.runforward.
    det_fem = np.asarray(detphi_fem).reshape(Ndet, Nsrc)
    det_mc = np.asarray(detphi_mc)

    print("\nDetector fluence agreement (MC vs FEM, log10):")
    for s in range(Nsrc):
        for d in range(Ndet):
            if det_fem[d, s] > 0 and det_mc[d, s] > 0:
                ratio = det_mc[d, s] / det_fem[d, s]
                print(
                    f"  S{s+1}-D{d+1}  FEM {det_fem[d, s]:.3e}   "
                    f"MC {det_mc[d, s]:.3e}   ratio {ratio:.3f}"
                )

    # ----------------------------------------------------------------
    # per-source fluence cross-section on y=30 mid-plane (optional plot)
    # ----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping cross-section plot.")
        return

    fig, axes = plt.subplots(2, Nsrc, figsize=(4 * Nsrc, 8))

    for s in range(Nsrc):
        # FEM phi at all nodes (single-source slice)
        phi_fem_s = (
            phi_fem[:, s]
            if (np.ndim(phi_fem) == 2 and phi_fem.shape[1] >= Nsrc)
            else phi_fem
        )
        phi_mc_s = phi_mc[:, s]

        for row, (lab, vals) in enumerate([("FEM", phi_fem_s), ("MC", phi_mc_s)]):
            ax = axes[row, s] if Nsrc > 1 else axes[row]
            sc = ax.scatter(
                node[:, 0],
                node[:, 2],
                c=np.log10(np.abs(np.asarray(vals).flatten()) + 1e-12),
                cmap="viridis",
                s=2,
            )
            ax.set_title(
                f"{lab} log10|phi|  S{s+1} at "
                f"({cfg['srcpos'][s, 0]:.0f}, {cfg['srcpos'][s, 1]:.0f}, "
                f"{cfg['srcpos'][s, 2]:.0f})"
            )
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")
            ax.set_aspect("equal")
            plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "demo_redbird_forward_mc.png"
    )
    plt.savefig(out_path, dpi=100)
    print(f"\nCross-section plot saved to: {out_path}")


if __name__ == "__main__":
    main()
