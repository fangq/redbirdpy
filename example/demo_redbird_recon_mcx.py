#!/usr/bin/env python
"""
Redbird - Voxel-grid Monte Carlo (mcxlab/pmcx) adjoint + matrix-free LSQR.

Port of redbird/example/demo_redbird_recon_mcx.m.

Demonstrates the mcxlab-grid Jacobian path: when cfg.nphoton AND cfg.vol
are set together, redbirdpy.runforward routes through pmcx, auto-fills
cfg.detdir via getdetdir_vol, sets cfg.srcid = -1, and returns
Jext = {'mua': ...} with .mua of shape (Nx, Ny, Nz, Ns*Nd).

This demo runs CW (omega = 0), in which case mua and the diffusion
coefficient D are inseparable from the measurements, so runforward
requests outputtype='adjoint' (J_mua only).  Jext['dcoeff'] is absent
on this path and only mua is reconstructed.  RF runs (cfg.omega > 0)
would auto-switch to outputtype='adjoint_mua_d' and also populate
Jext['dcoeff'], but the inversion below would still need to be
extended to consume it.

That 4D Jacobian is too large for the normal-equation form
(J.T @ J would be Nv x Nv with Nv ~ 1e6).  reglsqr wraps it via
jacop into a matvec/rmatvec scipy LinearOperator and solves
    min || J @ delta_mu - (y_meas - y_model) ||_2
via scipy.sparse.linalg.lsqr.  Early stopping acts as regularization;
no lambda to tune.

This demo performs ONE Gauss-Newton step (linearization around the
background mua) to keep the runtime small.  For a full iterative
reconstruction loop, wire cfg.muavol back into the next pmcx call's
cfg.vol (per-voxel property encoding) and re-run runforward.

Requires: pmcx on the Python path; an mcx tree that supports
outputtype='adjoint' / 'adjoint_mua_d' and srcid=-1 / -2.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbirdpy as rb

try:
    import pmcx  # noqa: F401
except ImportError:
    print("Error: pmcx is not installed; this demo requires the MCX Python binding.")
    sys.exit(1)


def main():
    # ----------------------------------------------------------------
    # 60 x 60 x 30 voxel slab (1 mm voxels)
    # ----------------------------------------------------------------
    Nx, Ny, Nz = 60, 60, 30
    cfg = {
        "vol": np.ones((Nx, Ny, Nz), dtype=np.uint8),  # single tissue label
        "unitinmm": 1.0,
        "srcpos": np.array([[20, 30, 0], [30, 30, 0], [40, 30, 0]], dtype=float),
        "srcdir": np.array([[0, 0, 1]], dtype=float),
        "srctype": "pencil",
        "detpos": np.array(
            [
                [15, 30, 0, 1.5],
                [25, 30, 0, 1.5],
                [35, 30, 0, 1.5],
                [45, 30, 0, 1.5],
            ],
            dtype=float,
        ),
        "prop": np.array([[0, 0, 1, 1], [0.005, 1, 0, 1.37]], dtype=float),
        "tstart": 0.0,
        "tend": 5e-9,
        "tstep": 5e-9,  # single-gate CW
        "omega": 0.0,  # CW: only J_mua is meaningful
        "gpuid": 1,
        "autopilot": 1,
        "maxdetphoton": int(1e6),
        "isreflect": 1,
    }

    Nsrc = cfg["srcpos"].shape[0]
    Ndet = cfg["detpos"].shape[0]

    # ----------------------------------------------------------------
    # Baseline forward + adjoint Jacobian
    # ----------------------------------------------------------------
    cfg["nphoton"] = int(5e7)

    print(f"pmcx adjoint ({cfg['nphoton']:.0e} photons) ...")
    import time

    t0 = time.time()
    detphi, phi, Jext = rb.runforward(cfg, return_jacobian=True)
    print(f"  done in {time.time() - t0:.2f} s")

    print(f"  detphi shape : {np.asarray(detphi).shape}   (Ndet={Ndet} x Nsrc={Nsrc})")
    print(f"  phi shape    : {np.asarray(phi).shape}   (Nx,Ny,Nz,Ns+Nd)")
    print(f"  Jext.mua shp : {Jext['mua'].shape}   (Nx,Ny,Nz,Ns*Nd)")
    # CW (omega=0): runforward used outputtype='adjoint', so Jext has 'mua'
    # only.  Jext['dcoeff'] would only be present for RF (omega > 0) runs.

    # ----------------------------------------------------------------
    # Synthesize a "measurement" by perturbing two voxel blocks
    # ----------------------------------------------------------------
    # Build linearized synthetic y_meas = y_model + J @ delta_mu_true.
    # This is the cleanest possible test: LSQR should recover something
    # whose forward projection matches r exactly (modulo MC noise +
    # iteration cap).
    delta_mu_true = np.zeros((Nx, Ny, Nz))
    delta_mu_true[10:15, 28:32, 14:16] = 0.005
    delta_mu_true[28:32, 38:42, 14:16] = 0.008

    J2 = Jext["mua"].reshape(Nx * Ny * Nz, Nsrc * Ndet)
    y_meas = np.asarray(detphi).ravel() + J2.T @ delta_mu_true.ravel()

    # ----------------------------------------------------------------
    # Inverse step via reglsqr (matrix-free LSQR)
    # ----------------------------------------------------------------
    r = y_meas - np.asarray(detphi).ravel()

    print("\nreglsqr ...")
    t0 = time.time()
    delta_mu_rec, info = rb.reglsqr(Jext["mua"], r, maxit=100, tol=1e-8)
    print(
        f"  done in {time.time() - t0:.2f} s   LSQR iters: {info['itn']}   "
        f"relres: {info['relres']:.3e}   adjoint err: {info['adjoint_err']:.2e}"
    )

    # ----------------------------------------------------------------
    # Compare reconstruction to ground truth
    # ----------------------------------------------------------------
    true_slice = delta_mu_true[:, :, 15]
    rec_slice = delta_mu_rec[:, :, 15]

    sum_true = float(np.sum(delta_mu_true))
    sum_rec = float(np.sum(delta_mu_rec))
    peak_true = float(np.max(true_slice))
    peak_rec = float(np.max(rec_slice))

    print(
        f"\nGround-truth sum: {sum_true:.3e}   "
        f"Reconstructed sum: {sum_rec:.3e}   "
        f"ratio: {sum_rec / sum_true:.3f}"
    )
    print(
        f"Ground-truth peak: {peak_true:.3e}   "
        f"Recon peak (z=15 slice): {peak_rec:.3e}"
    )

    # ----------------------------------------------------------------
    # Plot truth vs reconstruction on the z = 15 slice (optional)
    # ----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping reconstruction plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(true_slice.T, origin="lower", cmap="viridis")
    axes[0].set_title("ground-truth delta mua  (z=15)")
    axes[0].set_xlabel("x (voxel)")
    axes[0].set_ylabel("y (voxel)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(rec_slice.T, origin="lower", cmap="viridis")
    axes[1].set_title("recovered delta mua via LSQR  (z=15)")
    axes[1].set_xlabel("x (voxel)")
    axes[1].set_ylabel("y (voxel)")
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle("mcxlab voxel-grid mua reconstruction (1 Gauss-Newton step)")
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "demo_redbird_recon_mcx.png")
    plt.savefig(out_path, dpi=100)
    print(f"\nReconstruction plot saved to: {out_path}")


if __name__ == "__main__":
    main()
