#!/usr/bin/env python
"""
Redbird - Mesh-mode adjoint Jacobian via pmmc.

Port of redbird/example/demo_redbird_jacobian_mc.m.

Demonstrates the optional 3rd return value of ``runforward``:

    detphi, phi, Jext = rb.forward.runforward(cfg, return_jacobian=True)

Set when ``cfg["nphoton"]`` is present and ``return_jacobian=True`` is
passed. ``_runforward_mc`` then:
    * auto-fills ``cfg.detdir`` via ``getdetdir`` when absent,
    * sets ``cfg.outputtype = 'adjoint_mua_d'``, ``cfg.basisorder = 1``,
      ``cfg.srcid = -1``,
    * launches one batched pmmc call per wavelength packing all
      ``Ns`` forward + ``Nd`` detector-adjoint slots into one GPU run,
    * returns ``Jext = {"mua": J_mua, "dcoeff": J_D}`` in the
      ``(Nn, Ns*Nd)`` orientation used by mmclab.

Companion demos:
    demo_redbird_forward_mc.py - forward simulation via pmmc
    demo_redbird_recon_mc.py   - CW DOT reconstruction via MC
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
    # homogeneous 60 x 60 x 30 mm slab
    # ----------------------------------------------------------------
    node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 4)

    cfg = {
        "node": node,
        "elem": elem,
        "seg": np.ones(elem.shape[0], dtype=int),
        "srcpos": np.array([[30, 30, 0]], dtype=float),
        "srcdir": np.array([[0, 0, 1]], dtype=float),
        # three detectors at increasing source-detector separation along +x.
        # The disk-source radius (cfg.detpos[:, 3]) is auto-filled by
        # _runforward_mc to avgsize=1.0 when missing.
        "detpos": np.array([[20, 30, 0], [40, 30, 0], [45, 30, 0]], dtype=float),
        "detdir": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float),
        "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37]], dtype=float),
        "omega": 0,
    }

    cfg, _ = rb.utility.meshprep(cfg)

    # ----------------------------------------------------------------
    # MC adjoint Jacobian via runforward (return_jacobian=True)
    # ----------------------------------------------------------------
    import time

    cfg_mc = {k: v for k, v in cfg.items()}
    cfg_mc["nphoton"] = int(3e7)
    cfg_mc["gpuid"] = 1

    print(f"pmmc mesh adjoint ({cfg_mc['nphoton']:.0e} photons) ...")
    t0 = time.time()
    _, phi_mc, Jext = rb.forward.runforward(cfg_mc, return_jacobian=True)
    print(
        f"  done in {time.time() - t0:.2f} s   "
        f"Jext.mua shape: {Jext['mua'].shape}   "
        f"Jext.dcoeff shape: {Jext['dcoeff'].shape}"
    )

    # Jext orientation is mmclab/pmmc-native (Nn, Ns*Nd). Each COLUMN is
    # one source-detector pair's nodal Jacobian.

    # ----------------------------------------------------------------
    # FEM reference via the same mesh
    # ----------------------------------------------------------------
    cfg_fem = {k: v for k, v in cfg.items() if k != "nphoton"}
    _, phi_fem = rb.forward.runforward(cfg_fem)

    sd = rb.utility.sdmap(cfg_fem)
    Jmua_node_fem, _ = rb.forward.jac(
        sd, phi_fem, cfg_fem["deldotdel"], cfg_fem["elem"], cfg_fem["evol"]
    )
    # Jmua_node_fem is (Nsd, Nn); transpose to match Jext's (Nn, Nsd).
    Jfem = np.asarray(Jmua_node_fem).T
    print(f"FEM Jmua_node shape (transposed to (Nn, Nsd)): {Jfem.shape}")

    # ----------------------------------------------------------------
    # quantitative agreement: log10-log10 correlation per pair
    # ----------------------------------------------------------------
    Nsd = Jext["mua"].shape[1]
    print("\nMC vs FEM J_mua agreement (log10-log10 corr coef per s-d pair):")
    for k in range(Nsd):
        Jmc = np.abs(np.asarray(Jext["mua"])[:, k])
        Jfe = np.abs(Jfem[:, k])
        mask = (Jmc > 1e-12) & (Jfe > 1e-12)
        if mask.any():
            cc = np.corrcoef(np.log10(Jmc[mask]), np.log10(Jfe[mask]))[0, 1]
            print(f"  pair {k+1}:  {cc:.4f}  (expected close to 1.0)")

    # ----------------------------------------------------------------
    # plot banana profiles, MC vs FEM, on the y=30 mid-plane (optional)
    # ----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping banana profile plot.")
        return

    fig, axes = plt.subplots(2, Nsd, figsize=(4 * Nsd, 8))

    for k in range(Nsd):
        Jmc = np.abs(np.asarray(Jext["mua"])[:, k])
        Jfe = np.abs(Jfem[:, k])

        for row, (lab, vals) in enumerate([("MC", Jmc), ("FEM", Jfe)]):
            ax = axes[row, k] if Nsd > 1 else axes[row]
            sc = ax.scatter(
                node[:, 0],
                node[:, 2],
                c=np.log10(np.asarray(vals).flatten() + 1e-14),
                cmap="viridis",
                s=2,
            )
            ax.set_title(f"{lab}  log10|J_mua|  pair {k+1}")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")
            ax.set_aspect("equal")
            plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "demo_redbird_jacobian_mc.png")
    plt.savefig(out_path, dpi=100)
    print(f"\nBanana profile plot saved to: {out_path}")


if __name__ == "__main__":
    main()
