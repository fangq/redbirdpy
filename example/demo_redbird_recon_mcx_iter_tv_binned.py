#!/usr/bin/env python
"""
Redbird - Iterative voxel-grid Monte Carlo (pmcx) reconstruction on a
coarse (binned) grid with TV and positivity.

Port of redbird/example/demo_redbird_recon_mcx_iter_tv_binned.m.

Same setup as demo_redbird_recon_mcx_tv.py with a denser optode array,
but each Gauss-Newton step bins the fine 4D Jacobian onto a coarse grid
(redbirdpy.binjac, here 2x2x2 voxel blocks, 8x fewer unknowns), solves the
TV + positivity problem there (redbirdpy.regtv), and maps the update back
to the fine grid by replication (redbirdpy.upsample, the adjoint of the
binning).  The coarser basis regularizes the problem further and speeds
up the FISTA iterations.

Requires: pmcx on the Python path and a CUDA GPU.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import redbirdpy as rb

try:
    import pmcx  # noqa: F401
except ImportError:
    print("Error: pmcx is not installed; this demo requires the MCX Python binding.")
    sys.exit(1)


def main():
    Nx, Ny, Nz = 60, 60, 30
    mua_bg, musp_bg, n_bg = 0.005, 1.0, 1.37

    delta_mu_true = np.zeros((Nx, Ny, Nz))
    delta_mu_true[9:15, 27:32, 13:16] = 0.005
    delta_mu_true[27:32, 37:42, 13:16] = 0.008

    Sx, Sy = np.meshgrid([10, 20, 30, 40, 50], [10, 20, 30, 40, 50], indexing="ij")
    Dx, Dy = np.meshgrid([15, 22, 30, 38, 45], [15, 22, 30, 38, 45], indexing="ij")
    cfg_template = {
        "srcpos": np.column_stack([Sx.ravel(), Sy.ravel(), np.zeros(Sx.size)]),
        "srcdir": np.array([0, 0, 1.0]),
        "srctype": "pencil",
        "detpos": np.column_stack(
            [Dx.ravel(), Dy.ravel(), np.zeros(Dx.size), 1.5 * np.ones(Dx.size)]
        ),
        "prop": np.array([[0, 0, 1, 1], [mua_bg, musp_bg, 0, n_bg]]),
        "unitinmm": 1.0,
        "tstart": 0.0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "omega": 0.0,
        "gpuid": 1,
        "autopilot": 1,
        "maxdetphoton": int(1e6),
        "isreflect": 1,
    }

    def with_mua(mua, nphoton):
        cfg = dict(cfg_template)
        # a (1, Nx, Ny, Nz) float32 volume is read by mcx as per-voxel mua
        cfg["vol"] = np.asarray(mua, dtype=np.float32).reshape(1, Nx, Ny, Nz)
        cfg["nphoton"] = int(nphoton)
        return cfg

    print("Synthesizing target measurements (pmcx, true mua) ...")
    t0 = time.time()
    y_meas = np.asarray(rb.runforward(with_mua(mua_bg + delta_mu_true, 5e8))[0]).ravel()
    print(f"  done in {time.time() - t0:.2f} s   y_meas length: {y_meas.size}")

    mua_est = np.full((Nx, Ny, Nz), mua_bg)
    maxiter = 5
    binsize = 2
    alpha_rel = 1e-4
    resid_history = []
    mua_best, resid_best, iter_best = mua_est, np.inf, 0
    info = None

    for it in range(1, maxiter + 1):
        print(f"\n=== iter {it} ===")
        t0 = time.time()
        detphi, _, Jext = rb.runforward(with_mua(mua_est, 5e8), return_jacobian=True)
        print(f"  forward + Jacobian (pmcx) done in {time.time() - t0:.2f} s")

        r = y_meas - np.asarray(detphi).ravel()
        cur_resid = np.linalg.norm(r) / np.linalg.norm(y_meas)
        resid_history.append(cur_resid)
        print(f"  relative residual: {cur_resid:.4e}")
        # the residual at iteration it measures the update made in step it-1
        if it == 1 or cur_resid < resid_best:
            resid_best, mua_best, iter_best = cur_resid, mua_est, it - 1
        if cur_resid < 1e-4:
            print("  converged.")
            break

        print(f"  bin J to coarse basis (binsize = {binsize}) ...")
        J_c = rb.binjac(Jext["mua"], binsize)
        Nv_c = int(np.prod(J_c.shape[:3]))
        print(
            f"    coarse unknowns: {Nv_c}   (fine: {Nx * Ny * Nz}   "
            f"ratio: {Nx * Ny * Nz / Nv_c:.0f}x)"
        )
        grad0 = rb.jacop(J_c).rmatvec(r)
        alpha = alpha_rel * np.max(np.abs(grad0))
        print(f"  TV-FISTA (alpha = {alpha:.3e}, positive) ...")
        t0 = time.time()
        dmu_c, info = rb.regtv(
            J_c, r, alpha, maxit=200, innerit=10, tol=1e-6, positive=True
        )
        delta_mu_step = rb.upsample(dmu_c, binsize, (Nx, Ny, Nz))
        print(
            f"    done in {time.time() - t0:.2f} s   FISTA iters: {info['iter']}   "
            f"final obj: {info['obj'][-1]:.4e}   adjoint err: {info['adjoint_err']:.2e}"
        )
        mua_est = np.maximum(mua_est + delta_mu_step, 0)

    print(f"\nBest iterate: iter {iter_best} with rel residual {resid_best:.4e}")
    delta_mu_rec = mua_best - mua_bg

    slice_z = 14
    true_slice = delta_mu_true[:, :, slice_z]
    rec_slice = delta_mu_rec[:, :, slice_z]
    print(
        f"\nGround-truth sum (3D): {delta_mu_true.sum():.3e}   "
        f"Reconstructed sum: {delta_mu_rec.sum():.3e}   "
        f"ratio: {delta_mu_rec.sum() / max(delta_mu_true.sum(), 1e-30):.3f}"
    )
    print(
        f"Ground-truth peak (z={slice_z + 1}): {true_slice.max():.3e}   "
        f"Recon peak: {rec_slice.max():.3e}"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping the plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, img, title in [
        (axes[0, 0], true_slice, f"ground-truth delta mua  (z={slice_z + 1})"),
        (
            axes[0, 1],
            rec_slice,
            f"binned TV recon (best: after step {iter_best})  (z={slice_z + 1})",
        ),
    ]:
        im = ax.imshow(img.T, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x (voxel)")
        ax.set_ylabel("y (voxel)")
        plt.colorbar(im, ax=ax)
    axes[1, 0].semilogy(range(1, len(resid_history) + 1), resid_history, "-o")
    axes[1, 0].semilogy(iter_best, resid_best, "rs", markersize=12)
    axes[1, 0].set_xlabel("GN iteration")
    axes[1, 0].set_ylabel("relative residual")
    axes[1, 0].set_title("residual history")
    axes[1, 0].grid(True)
    if info is not None:
        axes[1, 1].semilogy(info["obj"], "-")
    axes[1, 1].set_xlabel("FISTA iteration (last GN step)")
    axes[1, 1].set_ylabel("objective 0.5||Jx-r||^2 + alpha TV(x)")
    axes[1, 1].set_title("FISTA convergence")
    axes[1, 1].grid(True)
    plt.suptitle("pmcx GN voxel mua recon: binned + TV-FISTA + positivity")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "demo_redbird_recon_mcx_iter_tv_binned.png",
        )
    )
    plt.show()


if __name__ == "__main__":
    main()
