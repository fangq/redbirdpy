"""
Redbird Reconstruction Module - Inverse problem solvers for DOT/NIRS.

INDEX CONVENTION: All mesh indices (elem, face) stored in cfg/recon are 1-based
to match MATLAB/iso2mesh. Conversion to 0-based occurs only when indexing numpy
arrays, using local variables named with '_0' suffix.

Functions:
    runrecon: Main reconstruction driver with iterative Gauss-Newton
    reginv: Regularized matrix inversion (auto-selects over/under-determined)
    reginvover: Overdetermined least-squares solver
    reginvunder: Underdetermined least-squares solver
    matreform: Reformat matrix equation for different output forms
    prior: Generate structure-prior regularization matrices
    regtv: Total-variation regularized least squares (FISTA)
    binjac/upsample: Bin a voxel Jacobian onto a coarse grid / its adjoint
    sdjac: Jacobian block for source-detector coupling estimation
"""

__all__ = [
    "runrecon",
    "reginv",
    "reginvover",
    "reginvunder",
    "matreform",
    "matflat",
    "prior",
    "syncprop",
    "multispectral",
    "createinv",
    "regemperical",
    "jacop",
    "reglsqr",
    "regtv",
    "binjac",
    "upsample",
    "sdjac",
]

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Optional, Union, List, Any
import warnings

from .forward import runforward, jac, jacchrome, jacepssigma, jacscat
from .solver import femsolve
from .utility import sdmap, meshinterp
from .property import updateprop, extinction


def runrecon(
    cfg: dict,
    recon: dict,
    detphi0: Union[np.ndarray, dict],
    sd: Union[np.ndarray, dict] = None,
    **kwargs,
) -> tuple:
    """
    Perform iterative Gauss-Newton reconstruction.

    Parameters
    ----------
    cfg : dict
        Forward simulation structure (forward mesh). elem/face are 1-based.
    recon : dict
        Reconstruction structure containing:
        - node, elem: Reconstruction mesh (optional, for dual-mesh). elem is 1-based.
        - param: Initial chromophore concentrations
        - prop: Initial optical properties
        - lambda: Regularization parameter
        - bulk: Background property values
        - mapid: Forward-to-recon mesh mapping (0-based element indices)
        - mapweight: Barycentric weights for mapping
    detphi0 : ndarray or dict
        Measured data to fit
    sd : ndarray or dict
        Source-detector mapping (0-based column indices into phi matrix)
    **kwargs : dict
        Options: maxiter, lambda_, tol, reform, report, prior, and the
        simultaneous source-detector coupling options below (each may also
        be given as a recon key):

        - sdcoupling: 'n' (off, default), 'y' (both roles), 's'
          (source-only) or 'd' (detector-only); True maps to 'y'
        - sdweight: extra weight of the coupling Jacobian columns (1)
        - sdprior: if > 0, Tikhonov pull of the couplings toward 1+0i (0)
        - sdfirst: fit only the couplings, freezing the image, during the
          first N iterations (0)

        Coupling estimation fits one complex coefficient per channel,
        ``recon["sd"]`` (initialized to 1+0i), with
        ``phi_hat(d, s) = c_s * c_d * phi(d, s)``; ``recon["sdindex"]`` maps
        every optode (sources first, then detectors, 0-based) to a 0-based
        channel (default: one channel per optode).  Requires reform 'real',
        'reim' or 'logphase', complex (RF) data and the FEM Jacobian.

    Returns
    -------
    recon : dict
        Updated reconstruction with fitted properties
    resid : ndarray
        Residual at each iteration
    cfg : dict
        Updated forward structure
    """
    import time

    # Parse options
    maxiter = kwargs.get("maxiter", 5)
    lambda_ = kwargs.get("lambda_", recon.get("lambda", 0.05))
    report = kwargs.get("report", True)
    tol = kwargs.get("tol", 0)
    reform = kwargs.get("reform", "real")
    solverflag = kwargs.get("solverflag", {})
    rfcw = kwargs.get("rfcw", [1])
    solvermethod = kwargs.get("method", "auto")
    prior_type = kwargs.get("prior", "")

    if isinstance(rfcw, int):
        rfcw = [rfcw]

    # simultaneous source-detector (SD) coupling coefficient estimation
    # options, see rbrunrecon.m
    sdmode = kwargs.get("sdcoupling", recon.get("sdcoupling", "n"))
    if isinstance(sdmode, (bool, int, np.integer, np.bool_)):
        sdmode = "y" if sdmode else "n"
    sdweight = kwargs.get("sdweight", recon.get("sdweight", 1))
    sdprior = kwargs.get("sdprior", recon.get("sdprior", 0))
    sdfirst = kwargs.get("sdfirst", recon.get("sdfirst", 0))
    dosd = sdmode in ("y", "s", "d")

    if sd is None:
        sd = sdmap(cfg)

    # one complex coupling coefficient per channel, shared across
    # wavelengths; recon["sdindex"] maps every optode (sources first, then
    # detectors, the same 0-based order as the phi columns) to a channel
    if dosd:
        nsrctot = ndettot = 0
        if cfg.get("srcpos") is not None and np.size(cfg["srcpos"]) > 0:
            nsrctot += np.atleast_2d(cfg["srcpos"]).shape[0]
        if cfg.get("detpos") is not None and np.size(cfg["detpos"]) > 0:
            ndettot += np.atleast_2d(cfg["detpos"]).shape[0]
        # widesrc/widedet are stored as (Nn x Npattern) in redbirdpy
        if cfg.get("widesrc") is not None and np.size(cfg["widesrc"]) > 0:
            nsrctot += np.asarray(cfg["widesrc"]).shape[1]
        if cfg.get("widedet") is not None and np.size(cfg["widedet"]) > 0:
            ndettot += np.asarray(cfg["widedet"]).shape[1]
        if recon.get("sdindex") is None or np.size(recon["sdindex"]) == 0:
            recon["sdindex"] = np.arange(nsrctot + ndettot)
        recon["sdindex"] = np.asarray(recon["sdindex"], dtype=int).ravel()
        if recon["sdindex"].size != nsrctot + ndettot:
            raise ValueError(
                f"recon['sdindex'] must have one entry per optode "
                f"({nsrctot} sources + {ndettot} detectors)"
            )
        if recon.get("sd") is None or np.size(recon["sd"]) == 0:
            recon["sd"] = np.ones(int(recon["sdindex"].max()) + 1, dtype=complex)
        recon["sd"] = np.asarray(recon["sd"], dtype=complex).ravel()

    # Normalize recon["prop"] to always be 2D
    if "prop" in recon:
        if isinstance(recon["prop"], np.ndarray):
            if recon["prop"].ndim == 1:
                recon["prop"] = recon["prop"].reshape(1, -1)
        elif isinstance(recon["prop"], dict):
            for key in recon["prop"]:
                if (
                    isinstance(recon["prop"][key], np.ndarray)
                    and recon["prop"][key].ndim == 1
                ):
                    recon["prop"][key] = recon["prop"][key].reshape(1, -1)

    # Determine if this is label-based reconstruction
    # Label-based: recon["prop"] has few rows (matching number of tissue labels)
    # Node-based: recon["prop"] has many rows (matching number of nodes)
    is_label_based = False
    if "prop" in recon and isinstance(recon["prop"], np.ndarray):
        n_prop_rows = recon["prop"].shape[0]
        # If prop has fewer rows than a reasonable mesh would have nodes,
        # it's label-based. Typical meshes have 100+ nodes.
        if n_prop_rows < 50:
            is_label_based = True
            # Create seg array if not present, assuming all elements use label 1
            if "seg" not in recon and "elem" in cfg:
                recon["seg"] = np.ones(cfg["elem"].shape[0], dtype=int)

    resid = np.zeros(maxiter)
    updates = []

    # Build regularization matrix if needed
    Aregu = {}
    if "lmat" in kwargs:
        Aregu["lmat"] = kwargs["lmat"]
    elif "ltl" in kwargs:
        Aregu["ltl"] = kwargs["ltl"]
    elif prior_type and "seg" in recon:
        Aregu["lmat"] = prior(recon["seg"], prior_type, kwargs)

    # Determine if using dual mesh
    dual_mesh = "node" in recon and "elem" in recon and "mapid" in recon

    # Main iteration loop
    for iteration in range(maxiter):
        t_start = time.time()

        # Sync properties between recon and forward mesh
        if "param" in recon or "prop" in recon:
            cfg, recon = syncprop(cfg, recon)

        # Update cfg.prop from cfg.param if multi-spectral
        if "param" in cfg and isinstance(cfg.get("prop"), dict):
            cfg["prop"] = updateprop(cfg)

        # Run forward simulation. When cfg.nphoton is set, runforward routes
        # to the Monte Carlo (pmmc) branch and can return the mesh-mode
        # adjoint Jacobian directly; we ask for it via return_jacobian=True
        # and skip the FEM jac() build when Jext is present. The FEM/TD
        # paths leave Jext = None and fall through to the rbjac equivalent.
        detphi, phi, Jext = runforward(
            cfg,
            solverflag=solverflag,
            sd=sd,
            rfcw=rfcw,
            return_jacobian=True,
            **kwargs,
        )

        # ratiometric (fNIRS) data support: when recon["isratio"] is set, the
        # measured detphi0 holds the ratio I/I0 (= exp(-dOD)); on the first
        # iteration convert it to pseudo-absolute by scaling with the model
        # baseline detphi, so the misfit detphi0 - detphi works in absolute
        # terms (port of redbird/matlab/rbrunrecon.m recon.isratio block).
        if iteration == 0 and isinstance(recon, dict) and recon.get("isratio"):
            if isinstance(detphi0, dict):
                for wv in list(detphi0.keys()):
                    model = detphi[wv] if wv in detphi else detphi[float(wv)]
                    detphi0[wv] = np.asarray(detphi0[wv]) * np.asarray(model)
            else:
                detphi0 = np.asarray(detphi0) * np.asarray(detphi)

        # simultaneous SD coupling estimation: scale the model prediction and
        # the forward/adjoint field columns by the current per-optode coupling
        # factors, so that the misfit AND the property Jacobians (built from
        # phi by jac) describe the model phi_hat(d, s) = c_s * c_d * phi(d, s)
        if dosd:
            if Jext is not None:
                raise ValueError(
                    "sdcoupling is not supported with Monte Carlo (Jext) Jacobians"
                )
            sdfac = recon["sd"][recon["sdindex"]]
            srcfac = sdfac[:nsrctot]
            detfac = sdfac[nsrctot:]
            sdouter = np.outer(detfac, srcfac)

            def _scale_detphi(d):
                d = np.asarray(d)
                if d.shape != sdouter.shape:
                    raise ValueError(
                        "sdcoupling does not yet support multi-modality (rfcw) "
                        "forward outputs"
                    )
                return d * sdouter

            def _scale_phi(p):
                p = np.asarray(p)
                if p.ndim != 2 or p.shape[1] != sdfac.size:
                    raise ValueError(
                        "sdcoupling does not yet support multi-modality (rfcw) "
                        "forward outputs"
                    )
                return p * sdfac[np.newaxis, :]

            if isinstance(detphi, dict):
                detphi = {wv: _scale_detphi(v) for wv, v in detphi.items()}
            else:
                detphi = _scale_detphi(detphi)
            if isinstance(phi, dict):
                phi = {wv: _scale_phi(v) for wv, v in phi.items()}
            else:
                phi = _scale_phi(phi)

        # ---- voxel-grid (mcxlab/pmcx) Jacobian: route through LSQR -------
        # When runforward's mcxlab/pmcx branch returned Jext['mua'] as a 4D
        # array (Nx, Ny, Nz, Nch), Nch = Ns*Nd or the active sd rows, the Jacobian is too large for the
        # normal-equation form (J.T @ J would be Nv x Nv).  Use reglsqr
        # (matrix-free LSQR with optional early stopping) to solve
        # J*delta_mu = (detphi0 - detphi) directly and write the voxel
        # update back to cfg.muavol (or stash as cfg.delta_muavol when the
        # caller's parameterization has no per-voxel field).  Mirrors the
        # rbrunrecon.m mcxlab dispatch.
        Jext_mua = Jext.get("mua") if isinstance(Jext, dict) else None
        if isinstance(Jext_mua, np.ndarray) and Jext_mua.ndim >= 4:
            rvec = np.asarray(detphi0).ravel() - np.asarray(detphi).ravel()
            if "sdidx" in Jext:
                rvec = rvec[Jext["sdidx"]]  # the channels of sd, as Jext["mua"] holds
            lsqr_maxit = int(kwargs.get("lsqrmaxit", 100))
            lsqr_tol = float(kwargs.get("lsqrtol", 1e-6))
            delta_mu_vol, lsqr_info = reglsqr(
                Jext_mua,
                rvec,
                maxit=lsqr_maxit,
                tol=lsqr_tol,
            )

            if (
                isinstance(cfg.get("muavol"), np.ndarray)
                and cfg["muavol"].shape == delta_mu_vol.shape
            ):
                cfg["muavol"] = cfg["muavol"] + delta_mu_vol
            elif (
                isinstance(cfg.get("prop"), np.ndarray)
                and np.asarray(cfg["prop"]).shape[0] == 2
            ):
                # single tissue label + outside row: stash as cfg.muavol
                cfg["muavol"] = delta_mu_vol
            else:
                cfg["delta_muavol"] = delta_mu_vol

            resid[iteration] = float(np.linalg.norm(rvec))
            if report:
                print(
                    f"iter [{iteration + 1:4d}] (mcxlab/LSQR): "
                    f"residual={resid[iteration]:.4e} "
                    f"relres={lsqr_info['relres']:.4e} "
                    f"lsqr_iter={lsqr_info['itn']} "
                    f"(time={time.time() - t_start:.2f}s)"
                )
            continue

        # Build Jacobians
        wavelengths = [""]
        if isinstance(cfg.get("prop"), dict):
            wavelengths = list(cfg["prop"].keys())

        Jmua = {}

        if Jext is not None and "mua" in Jext:
            # Monte Carlo path: Jext["mua"] / Jext["dcoeff"] are in the
            # pmmc-native (Nn, Ns*Nd) orientation. Transpose to (Nsd, Nn) so
            # the downstream pipeline (jacchrome, jacepssigma, matflat,
            # _masksum, _remap_jacobian) keeps the same FEM convention as
            # the rbjac path.
            jmua_src = Jext["mua"]

            if isinstance(jmua_src, dict):
                for wv in wavelengths:
                    key = "mua" if wv == "" else wv
                    Jmua[key] = np.asarray(jmua_src[wv]).T
            else:
                # single-wavelength: jmua_src is a 2D array
                Jmua["mua"] = np.asarray(jmua_src).T

            # Jext["dcoeff"] consumption is RF-only: CW (omega == 0) cannot
            # separate mua and D from a single measurement set, so the FEM
            # branch guards Jd on (any(omegas) > 0) and we mirror that here.
            # Note: even on RF, dcoeff is not currently routed into Jmua's
            # dict shape -- D-coefficient image recon needs a wider refactor.
            omegas_r = cfg.get("omega", 0)
            if isinstance(omegas_r, dict):
                _is_rf = any(float(v) > 0 for v in omegas_r.values())
            else:
                _is_rf = float(omegas_r) > 0 if omegas_r is not None else False
            if _is_rf and "dcoeff" in Jext:
                pass  # placeholder for future D-coefficient recon hook
        else:
            for wv in wavelengths:
                sdwv = sd.get(wv, sd) if isinstance(sd, dict) else sd
                phiwv = phi.get(wv, phi) if isinstance(phi, dict) else phi

                Jmua_n, Jmua_e = jac(
                    sdwv, phiwv, cfg["deldotdel"], cfg["elem"], cfg["evol"]
                )
                # Use "mua" as key for single-wavelength case
                key = "mua" if wv == "" else wv
                Jmua[key] = Jmua_n

        # Build chromophore Jacobians if multi-spectral, OR MWT Jacobians
        # (epsilon/sigma) if cfg.param has those entries.
        if isinstance(cfg.get("prop"), dict) and "param" in cfg:
            has_eps = "epsilon" in cfg["param"]
            has_sigma = "sigma" in cfg["param"]
            if has_eps or has_sigma:
                Jmua = jacepssigma(
                    Jmua, cfg.get("omega", 0), has_eps=has_eps, has_sigma=has_sigma
                )
            else:
                chromophores = [
                    k
                    for k in cfg["param"].keys()
                    if k in ["hbo", "hbr", "water", "lipids", "aa3"]
                ]
                if chromophores:
                    Jmua = jacchrome(Jmua, chromophores)

        # Flatten measurement data
        detphi0_flat = _flatten_detphi(detphi0, sd, wavelengths, rfcw)
        detphi_flat = _flatten_detphi(detphi, sd, wavelengths, rfcw)

        # Get block structure
        if isinstance(Jmua, dict):
            blocks = {k: v.shape for k, v in Jmua.items()}
        else:
            blocks = {"mua": Jmua.shape}

        # Flatten Jacobian
        Jflat = matflat(Jmua)

        # Reformat for real-valued solver if needed
        if reform != "complex":
            Jflat, misfit, nblock = matreform(Jflat, detphi0_flat, detphi_flat, reform)
        else:
            misfit = detphi0_flat - detphi_flat

        # Map Jacobian to recon mesh if dual-mesh
        if dual_mesh:
            Jflat = _remap_jacobian(Jflat, recon, cfg)
            # Update blocks to reflect recon mesh size
            nn_recon = recon["node"].shape[0]
            blocks = {k: (v[0], nn_recon) for k, v in blocks.items()}

        # Compress for segmented reconstruction ONLY if:
        # 1. seg array length matches Jacobian columns (node-based seg for compression)
        # 2. Few unique labels (true label-based reconstruction, not element segmentation)
        if "seg" in recon and np.ndim(recon["seg"]) == 1:
            seg = recon["seg"]
            n_jac_cols = Jflat.shape[1]
            n_labels = len(np.unique(seg))

            # Only compress if seg is node-based (matches Jac columns) with few labels
            if len(seg) == n_jac_cols and n_labels < 50:
                Jflat = _masksum(Jflat, seg)
                # Update blocks to reflect compressed size
                blocks = {k: (v[0], n_labels) for k, v in blocks.items()}

        # Store residual
        resid[iteration] = np.sum(np.abs(misfit))

        # Prepare regularization
        if iteration == 0 and Aregu:
            if "lmat" in Aregu and "ltl" not in Aregu:
                if Jflat.shape[0] >= Jflat.shape[1]:
                    Aregu["ltl"] = Aregu["lmat"].T @ Aregu["lmat"]
                else:
                    from scipy.linalg import qr

                    _, Aregu["lir"] = qr(Aregu["lmat"])
                    Aregu["lir"] = np.linalg.inv(np.triu(Aregu["lir"]))

        blockscale = 1.0 / np.sqrt(np.sum(Jflat**2))
        Jflat = Jflat * blockscale

        # append the SD coupling Jacobian block (see sdjac); its columns are
        # [Re(dc) (or d|c|), Im(dc) (or d angle(c))], Frobenius-normalized and
        # weighted by sdweight; sdprior appends Tikhonov rows pulling the
        # coefficients toward 1+0i
        if dosd:
            if reform == "complex":
                raise ValueError(
                    "sdcoupling requires reform set to 'real', 'reim' or 'logphase'"
                )
            Asd, _, _ = sdjac(
                sd,
                detphi_flat,
                recon["sd"],
                recon["sdindex"],
                reform,
                sdmode,
                rfcw,
                wavelengths,
            )
            if Asd.shape[0] != Jflat.shape[0]:
                raise ValueError(
                    f"sdcoupling: SD block rows ({Asd.shape[0]}) do not match the "
                    f"reformed system rows ({Jflat.shape[0]}); CW-only (real) data "
                    "is not supported"
                )
            nsdhalf = Asd.shape[1] // 2
            if sdprior > 0:
                if reform == "logphase":
                    sdtarget = np.concatenate(
                        [1 - np.abs(recon["sd"]), -np.angle(recon["sd"])]
                    )
                else:
                    sdtarget = np.concatenate([1 - recon["sd"].real, -recon["sd"].imag])
                Jflat = np.vstack([Jflat, np.zeros((2 * nsdhalf, Jflat.shape[1]))])
                Asd = np.vstack([Asd, sdprior * np.eye(2 * nsdhalf)])
                misfit = np.concatenate([misfit, sdprior * sdtarget])
            sdscalefact = np.array(
                [
                    sdweight / np.linalg.norm(Asd[:, :nsdhalf]),
                    sdweight / np.linalg.norm(Asd[:, nsdhalf:]),
                ]
            )
            sdscalefact[~np.isfinite(sdscalefact)] = 1
            if iteration < sdfirst:
                Jflat = Jflat * 0  # coupling-only fit: freeze the image unknowns
            Jflat = np.hstack(
                [
                    Jflat,
                    Asd[:, :nsdhalf] * sdscalefact[0],
                    Asd[:, nsdhalf:] * sdscalefact[1],
                ]
            )

        # Solve inverse problem
        dmu = reginv(
            Jflat, misfit, lambda_, Aregu, blocks, method=solvermethod, **solverflag
        )

        # extract and apply the SD coupling update: additive in the complex
        # form, or in log-magnitude/phase; amplitudes are floored at 0.05 as in
        # Redbird90, then the mean is re-pinned at 1+0i to remove the global
        # scale ambiguity between the couplings and the image
        if dosd:
            dsd = dmu[-2 * nsdhalf :].copy()
            dmu = dmu[: -2 * nsdhalf]
            dsd[:nsdhalf] *= sdscalefact[0]
            dsd[nsdhalf:] *= sdscalefact[1]
            if reform == "logphase":
                sdamp = np.maximum(np.abs(recon["sd"]) + dsd[:nsdhalf], 0.05)
                sdpha = np.angle(recon["sd"]) + dsd[nsdhalf:]
                recon["sd"] = sdamp * np.exp(1j * sdpha)
            else:
                recon["sd"] = recon["sd"] + (dsd[:nsdhalf] + 1j * dsd[nsdhalf:])
                sdneg = recon["sd"].real < 0
                recon["sd"][sdneg] = 0.05 + 1j * recon["sd"][sdneg].imag
            recon["sd"] = recon["sd"] - np.mean(recon["sd"]) + 1

        dmu = dmu * blockscale

        # Parse update and apply to recon structure
        update = {}
        idx = 0
        output_keys = list(blocks.keys())

        for key in output_keys:
            size = blocks[key][1]
            dx = dmu[idx : idx + size]
            update[key] = dx
            idx += size

            # Apply update to recon structure (not cfg!)
            if key in ["mua", "dcoeff"]:
                propidx = 0 if key == "mua" else 1
                if "prop" in recon and isinstance(recon["prop"], np.ndarray):
                    prop = recon["prop"]
                    n_prop_rows = prop.shape[0]

                    # Determine if label-based by comparing prop rows to dx length
                    if n_prop_rows < len(dx) and n_prop_rows < 50:
                        # Label-based: prop has one row per tissue label
                        n_updates = min(n_prop_rows, len(dx))
                        for li in range(n_updates):
                            if key == "dcoeff":
                                old_dcoeff = 1.0 / (3 * prop[li, propidx])
                                new_dcoeff = old_dcoeff + dx[li]
                                recon["prop"][li, propidx] = 1.0 / (3 * new_dcoeff)
                            else:
                                recon["prop"][li, propidx] += dx[li]
                    else:
                        # Node/element based: prop has one row per node
                        if key == "dcoeff":
                            old_dcoeff = 1.0 / (3 * prop[:, propidx])
                            new_dcoeff = old_dcoeff + dx
                            recon["prop"][:, propidx] = 1.0 / (3 * new_dcoeff)
                        else:
                            recon["prop"][:, propidx] += dx

            elif key in ["hbo", "hbr", "water", "lipids", "scatamp", "scatpow"]:
                # Determine target: recon["param"] if present, else cfg["param"]
                if "param" in recon and key in recon["param"]:
                    target = recon
                elif "param" in cfg and key in cfg["param"]:
                    target = cfg
                else:
                    continue

                param_val = target["param"][key]

                # Get length of parameter (scalar vs array)
                if hasattr(param_val, "__len__"):
                    n_param = len(param_val)
                else:
                    n_param = 1

                # Determine if label-based by comparing param length to dx length
                if n_param < len(dx) and n_param < 50:
                    # Label-based: param has one value per tissue label
                    if n_param == 1:
                        # Scalar parameter
                        if hasattr(param_val, "__len__"):
                            target["param"][key][0] += dx[0]
                        else:
                            target["param"][key] += dx[0]
                    else:
                        # Array parameter with few elements (labels)
                        n_updates = min(n_param, len(dx))
                        for li in range(n_updates):
                            target["param"][key][li] += dx[li]
                else:
                    # Node/element based: param has one value per node
                    target["param"][key] = param_val + dx

        if dosd:
            update["sd"] = recon["sd"].copy()
        updates.append(update)

        if report:
            elapsed = time.time() - t_start
            rel_resid = resid[iteration] / resid[0] if iteration > 0 else 1.0
            print(
                f"iter [{iteration + 1:4d}]: residual={resid[iteration]:.6e}, "
                f"relres={rel_resid:.6e} lambda={lambda_:.6e} (time={elapsed:.2f} s)"
            )

        # Check convergence
        if (
            iteration > 0
            and abs(resid[iteration] - resid[iteration - 1]) / resid[0] < tol
        ):
            resid = resid[: iteration + 1]
            break

    recon["lambda"] = lambda_

    return recon, resid, cfg, updates, Jmua, detphi, phi


def jacop(J: np.ndarray):
    """Wrap a 4D voxel-grid Jacobian as a scipy LinearOperator.

    Port of redbird/matlab/rbjacop.m.  The 4D Jacobian shape is
    ``(Nx, Ny, Nz, Nsd)`` where ``Nsd = Ns*Nd`` is the number of
    source-detector pairs and ``J[..., k]`` is the sensitivity volume for the
    k-th pair.  We never form the ``Nv x Nv`` Hessian ``J.T @ J``; LSQR only
    needs the matvec / rmatvec products, so the operator reshapes ``J`` into
    a 2D ``(Nv, Nsd)`` view once and dispatches:

        matvec(x)  : voxel-space ``x`` (length Nv)  ->  data-space (Nsd,)
        rmatvec(y) : data-space ``y`` (length Nsd) ->  voxel-space (Nv,)

    Parameters
    ----------
    J : ndarray, shape (Nx, Ny, Nz, Nsd)
        Voxel-grid adjoint Jacobian returned by mcxlab in
        ``cfg.outputtype = 'adjoint'`` / ``'adjoint_mua_d'`` mode.

    Returns
    -------
    op : scipy.sparse.linalg.LinearOperator
        Shape ``(Nsd, Nv)`` operator with matvec / rmatvec wired to ``J``.
    """
    from scipy.sparse.linalg import LinearOperator

    if J.ndim != 4:
        raise ValueError(f"jacop: J must be a 4D array, got ndim={J.ndim}")

    Nx, Ny, Nz, Nsd = J.shape
    Nv = Nx * Ny * Nz
    # reshape on a C-contiguous array is an O(1) view; both matvec and
    # rmatvec then reduce to a single BLAS GEMV.
    J2 = J.reshape(Nv, Nsd)

    def matvec(x):
        return J2.T @ np.asarray(x).ravel()

    def rmatvec(y):
        return J2 @ np.asarray(y).ravel()

    return LinearOperator(
        shape=(Nsd, Nv), matvec=matvec, rmatvec=rmatvec, dtype=J.dtype
    )


def reglsqr(
    J,
    r: np.ndarray,
    tol: float = 1e-6,
    maxit: int = 200,
    damp: float = 0.0,
    reshape3d: bool = True,
    adjointtest: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Iterative least-squares solve of ``J * delta_mu = r`` via LSQR.

    Port of redbird/matlab/rbreglsqr.m.  Designed for the mcxlab voxel-grid
    Jacobian where ``J`` is too large for the normal-equation form
    ``(J^T J + lambda I)^{-1} J^T r`` (``J^T J`` would be ``Nv x Nv`` with
    ``Nv ~ 1e6 - 1e7`` voxels).  Regularization is by **early stopping** --
    the Krylov subspace captures dominant singular components first
    (semi-convergence), so the iteration count itself acts as the
    regularizer.  Pass ``damp > 0`` for explicit Tikhonov.

    Parameters
    ----------
    J : ndarray or scipy.sparse.linalg.LinearOperator
        Either a 4D Jacobian ``(Nx, Ny, Nz, Nsd)`` (wrapped on the fly via
        `jacop`) or an existing LinearOperator-like with shape ``(Nsd, Nv)``.
    r : ndarray, shape (Nsd,)
        Measurement residual ``y_meas - y_model``.
    tol : float, optional
        LSQR convergence tolerance (default 1e-6).
    maxit : int, optional
        Max LSQR iterations (default 200).
    damp : float, optional
        ``sqrt(lambda)`` for Tikhonov damping (default 0; early stopping
        provides implicit regularization).  scipy's lsqr exposes ``damp``
        directly, so unlike the MATLAB port we don't need to build an
        augmented system manually.
    reshape3d : bool, optional
        When ``J`` is a 4D array and this is True, reshape ``delta_mu`` back
        to ``(Nx, Ny, Nz)`` on return (default True).
    adjointtest : bool, optional
        Run a one-shot dot-product test ``<Jv, u>`` vs ``<v, J'u>`` with
        random ``v``, ``u`` before LSQR.  Catches sign / index-order errors
        in custom operators.  Warns if the relative error exceeds 1e-4.

    Returns
    -------
    delta_mu : ndarray
        Voxel-space update; shape ``(Nx, Ny, Nz)`` for 4D-array ``J`` (when
        ``reshape3d``), otherwise the flat ``Nv``-vector.
    info : dict
        ``flag``, ``itn``, ``relres``, ``adjoint_err``.

    Notes
    -----
    Uses scipy's built-in ``scipy.sparse.linalg.lsqr``.  Hansen, "Discrete
    Inverse Problems" (SIAM 2010) ch.6-8 covers semi-convergence / early
    stopping for ill-posed least-squares; Paige & Saunders (1982) is the
    original LSQR reference.
    """
    from scipy.sparse.linalg import lsqr as _lsqr, LinearOperator

    input_was_4d = isinstance(J, np.ndarray) and J.ndim == 4
    if input_was_4d:
        sz4 = J.shape
        op = jacop(J)
    elif isinstance(J, LinearOperator) or hasattr(J, "matvec"):
        op = J
        sz4 = None
    else:
        raise TypeError(
            "reglsqr: J must be a 4D ndarray or a scipy LinearOperator-like"
        )

    Nsd = int(op.shape[0])
    Nv = int(op.shape[1])

    adjoint_err = float("nan")
    if adjointtest:
        rng = np.random.default_rng()  # don't touch the global RNG state
        v_probe = rng.standard_normal(Nv)
        u_probe = rng.standard_normal(Nsd)
        Jv = op.matvec(v_probe)
        Jtu = op.rmatvec(u_probe)
        lhs = float(u_probe @ Jv)
        rhs = float(Jtu @ v_probe)
        adjoint_err = abs(lhs - rhs) / max(abs(lhs), np.finfo(float).eps)
        if adjoint_err > 1e-4:
            warnings.warn(
                "reglsqr: adjoint dot-product test failed -- "
                f"|<Jv,u> - <v,J'u>| / |<u,Jv>| = {adjoint_err:.3e}. "
                "LSQR may converge to garbage; check the sign / normalization "
                "of your operator.",
                RuntimeWarning,
                stacklevel=2,
            )

    r_flat = np.asarray(r, dtype=float).ravel()
    if r_flat.size != Nsd:
        raise ValueError(f"reglsqr: r has length {r_flat.size}, expected Nsd = {Nsd}")

    result = _lsqr(op, r_flat, atol=tol, btol=tol, iter_lim=maxit, damp=damp)
    x = result[0]
    info = {
        "flag": int(result[1]),
        "itn": int(result[2]),
        "relres": float(result[3]),
        "adjoint_err": adjoint_err,
    }

    if input_was_4d and reshape3d:
        delta_mu = x.reshape(sz4[0], sz4[1], sz4[2])
    else:
        delta_mu = x

    return delta_mu, info


def regtv(
    J,
    r: np.ndarray,
    alpha: float,
    maxit: int = 100,
    innerit: int = 10,
    tol: float = 1e-5,
    L: float = None,
    reshape3d: bool = True,
    adjointtest: bool = True,
    verbose: bool = False,
    positive: bool = False,
    dims3: tuple = None,
) -> Tuple[np.ndarray, dict]:
    """Total-variation regularized least squares via FISTA.

    Port of redbird/matlab/rbregtv.m.  Solves

        min_x  0.5 * ||J x - r||_2^2  +  alpha * TV_iso(x)

    where TV_iso is the isotropic 3D total-variation seminorm, using
    Beck & Teboulle (2009) FISTA: each outer step takes a gradient step on
    the data-fit term, then applies the TV proximal operator, solved by
    ``innerit`` iterations of Chambolle's (2004) projected-gradient method.
    TV favours piecewise-constant images and preserves compact inclusions
    better than the minimum-norm `reglsqr`.

    Parameters
    ----------
    J : ndarray, LinearOperator or dict
        A 4D voxel Jacobian ``(Nx, Ny, Nz, Nsd)``, a LinearOperator of shape
        ``(Nsd, Nv)`` (then ``dims3`` is required), or a dict with key "mua".
    r : ndarray, shape (Nsd,)
        Measurement residual.
    alpha : float
        TV regularization weight (>= 0).
    maxit, innerit, tol : optional
        Outer FISTA iterations (100), inner Chambolle iterations (10), and
        the relative-change stopping tolerance (1e-5).
    L : float, optional
        Lipschitz constant of J^T J; estimated by power iteration if None.
    reshape3d : bool, optional
        Return ``(Nx, Ny, Nz)`` for a 4D/dict J (default True).
    adjointtest : bool, optional
        Run a one-shot dot-product test of the operator (default True).
    verbose : bool, optional
        Print per-iteration progress.
    positive : bool, optional
        Project x >= 0 after each proximal step (default False).
    dims3 : tuple, optional
        Volume shape ``(Nx, Ny, Nz)`` when J is a LinearOperator.

    Returns
    -------
    delta_mu : ndarray
        The voxel update, ``(Nx, Ny, Nz)`` or a flat vector.
    info : dict
        ``iter``, ``L``, ``obj`` (objective history), ``relchg``,
        ``adjoint_err``.
    """
    from scipy.sparse.linalg import LinearOperator

    input_was_4d = False
    if isinstance(J, dict) and "mua" in J:
        J = np.asarray(J["mua"])
    if isinstance(J, np.ndarray) and J.ndim == 4:
        dims3 = J.shape[:3]
        op = jacop(J)
        input_was_4d = True
    elif isinstance(J, LinearOperator) or hasattr(J, "matvec"):
        op = J
        if dims3 is None:
            raise ValueError(
                "regtv: when J is a LinearOperator, pass dims3=(Nx, Ny, Nz) so "
                "the TV operators know the volume shape"
            )
    else:
        raise TypeError(
            "regtv: J must be a 4D ndarray, a LinearOperator, or a dict with key 'mua'"
        )

    dims3 = tuple(int(d) for d in dims3)
    Nsd, Nv = int(op.shape[0]), int(op.shape[1])
    if int(np.prod(dims3)) != Nv:
        raise ValueError(f"regtv: dims3 {dims3} does not match Nv = {Nv}")

    rng = np.random.default_rng()
    adjoint_err = float("nan")
    if adjointtest:
        v_probe = rng.standard_normal(Nv)
        u_probe = rng.standard_normal(Nsd)
        lhs = float(u_probe @ op.matvec(v_probe))
        rhs = float(op.rmatvec(u_probe) @ v_probe)
        adjoint_err = abs(lhs - rhs) / max(abs(lhs), np.finfo(float).eps)
        if adjoint_err > 1e-4:
            warnings.warn(
                f"regtv: adjoint dot-product test failed: rel err = {adjoint_err:.3e}",
                RuntimeWarning,
                stacklevel=2,
            )

    if L is None:
        L = _power_iter_jtj(op, Nv, 10, rng) * 1.05  # small safety factor

    r_col = np.asarray(r, dtype=float).ravel()
    x_prev = np.zeros(Nv)
    x_new = x_prev
    y = np.zeros(Nv)
    t = 1.0
    obj_hist = np.zeros(maxit)
    relchg_hist = np.zeros(maxit)

    k = 0
    for k in range(maxit):
        # gradient step on the data-fit term at y
        Jy = op.matvec(y)
        grad = op.rmatvec(Jy - r_col)
        z = y - grad / L

        # proximal step: 3D TV denoising of z, optionally x >= 0
        x_new3 = _tv_prox_iso3d(z.reshape(dims3), alpha / L, innerit)
        if positive:
            x_new3 = np.maximum(x_new3, 0)
        x_new = x_new3.ravel()

        # FISTA acceleration
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = x_new + ((t - 1) / t_new) * (x_new - x_prev)

        res = Jy - r_col
        obj_hist[k] = 0.5 * float(res @ res) + alpha * _tv_iso3d_value(x_new3)
        relchg_hist[k] = np.linalg.norm(x_new - x_prev) / max(
            np.linalg.norm(x_prev), np.finfo(float).eps
        )

        if verbose and (k < 5 or (k + 1) % 10 == 0 or k == maxit - 1):
            print(
                f"  FISTA iter {k + 1:3d}:  obj={obj_hist[k]:.4e}   relchg={relchg_hist[k]:.3e}"
            )

        if k > 0 and relchg_hist[k] < tol:
            break

        x_prev = x_new
        t = t_new

    niter = k + 1
    info = {
        "iter": niter,
        "L": float(L),
        "obj": obj_hist[:niter],
        "relchg": relchg_hist[:niter],
        "adjoint_err": adjoint_err,
    }

    if input_was_4d and reshape3d:
        return x_new.reshape(dims3), info
    return x_new, info


def _power_iter_jtj(op, Nv: int, niter: int, rng) -> float:
    """Estimate ||J^T J||_2 by power iteration (see rbregtv.m)."""
    v = rng.standard_normal(Nv)
    v = v / np.linalg.norm(v)
    for _ in range(niter):
        v = op.rmatvec(op.matvec(v))
        nrm = np.linalg.norm(v)
        if nrm < np.finfo(float).eps:
            return 0.0
        v = v / nrm
    w = op.matvec(v)
    return float(w @ w)


def _tv_prox_iso3d(z: np.ndarray, lam: float, niter: int) -> np.ndarray:
    """Chambolle's projected gradient for argmin_u 0.5||u-z||^2 + lam*TV_iso(u)."""
    if lam <= 0:  # no TV penalty: the proximal operator is the identity
        return z
    p1 = np.zeros_like(z)
    p2 = np.zeros_like(z)
    p3 = np.zeros_like(z)
    tau = 1.0 / 12.0  # ||D||_2^2 <= 12 for 3D forward differences

    for _ in range(niter):
        g1, g2, g3 = _grad3d(_div3d(p1, p2, p3) - z / lam)
        p1 = p1 + tau * g1
        p2 = p2 + tau * g2
        p3 = p3 + tau * g3
        # project onto the isotropic unit ball per voxel
        nrm = np.maximum(np.sqrt(p1 * p1 + p2 * p2 + p3 * p3), 1.0)
        p1 = p1 / nrm
        p2 = p2 / nrm
        p3 = p3 / nrm

    return z - lam * _div3d(p1, p2, p3)


def _tv_iso3d_value(x: np.ndarray) -> float:
    """Isotropic 3D TV value."""
    g1, g2, g3 = _grad3d(x)
    return float(np.sum(np.sqrt(g1 * g1 + g2 * g2 + g3 * g3)))


def _grad3d(x: np.ndarray):
    """Forward differences with a zero difference at the last voxel of each axis."""
    g1 = np.zeros_like(x)
    g2 = np.zeros_like(x)
    g3 = np.zeros_like(x)
    g1[:-1, :, :] = x[1:, :, :] - x[:-1, :, :]
    g2[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    g3[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    return g1, g2, g3


def _div3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Discrete divergence, the negative adjoint of _grad3d."""
    d1 = np.zeros_like(p1)
    d2 = np.zeros_like(p2)
    d3 = np.zeros_like(p3)
    d1[1:-1, :, :] = p1[1:-1, :, :] - p1[:-2, :, :]
    d1[0, :, :] = p1[0, :, :]
    d1[-1, :, :] = -p1[-2, :, :]
    d2[:, 1:-1, :] = p2[:, 1:-1, :] - p2[:, :-2, :]
    d2[:, 0, :] = p2[:, 0, :]
    d2[:, -1, :] = -p2[:, -2, :]
    d3[:, :, 1:-1] = p3[:, :, 1:-1] - p3[:, :, :-2]
    d3[:, :, 0] = p3[:, :, 0]
    d3[:, :, -1] = -p3[:, :, -2]
    return d1 + d2 + d3


def binjac(J: np.ndarray, binsize) -> np.ndarray:
    """Bin a 4D fine-grid voxel Jacobian onto a coarse grid.

    Port of redbird/matlab/rbbinjac.m.  Fine-voxel columns are SUMMED within
    each ``binsize`` block, i.e. ``J_c = J_fine * U`` with ``U`` the
    replication operator implemented by `upsample`, so a coarse unknown
    ``x_c`` predicts the same data as its upsampled fine field.  The grid is
    zero-padded up to a multiple of ``binsize`` when needed.

    Parameters
    ----------
    J : ndarray, shape (Nx, Ny, Nz, Nsd)
        Fine-grid voxel Jacobian.
    binsize : int or sequence of 3 ints
        Block size along x, y, z.

    Returns
    -------
    J_c : ndarray, shape (ceil(Nx/bx), ceil(Ny/by), ceil(Nz/bz), Nsd)
    """
    J = np.asarray(J)
    if J.ndim != 4:
        raise ValueError("binjac: J must be a 4D array (Nx, Ny, Nz, Nsd)")
    bx, by, bz = _binsize3(binsize, "binjac")
    Nx, Ny, Nz, Nsd = J.shape
    Nxc, Nyc, Nzc = -(-Nx // bx), -(-Ny // by), -(-Nz // bz)

    if (Nxc * bx, Nyc * by, Nzc * bz) != (Nx, Ny, Nz):
        J_pad = np.zeros((Nxc * bx, Nyc * by, Nzc * bz, Nsd), dtype=J.dtype)
        J_pad[:Nx, :Ny, :Nz, :] = J
    else:
        J_pad = J

    return J_pad.reshape(Nxc, bx, Nyc, by, Nzc, bz, Nsd).sum(axis=(1, 3, 5))


def upsample(x_coarse: np.ndarray, binsize, fine_dims=None) -> np.ndarray:
    """Upsample a 3D coarse field to a fine grid by replication.

    Port of redbird/matlab/rbupsample.m, the adjoint of the binning in
    `binjac`: each coarse voxel is copied to the ``binsize`` block it
    covers, then the result is cropped to ``fine_dims`` if given.

    Parameters
    ----------
    x_coarse : ndarray, shape (Nxc, Nyc, Nzc)
    binsize : int or sequence of 3 ints
    fine_dims : sequence of 3 ints, optional
        Fine-grid shape to crop to (e.g. the unpadded Jacobian grid).

    Returns
    -------
    x_fine : ndarray
    """
    bx, by, bz = _binsize3(binsize, "upsample")
    x_fine = np.repeat(np.repeat(np.repeat(np.asarray(x_coarse), bx, 0), by, 1), bz, 2)
    if fine_dims is not None:
        fine_dims = tuple(int(d) for d in fine_dims[:3])
        if any(s < f for s, f in zip(x_fine.shape[:3], fine_dims)):
            raise ValueError(
                "upsample: upsampled shape is smaller than requested fine_dims"
            )
        x_fine = x_fine[: fine_dims[0], : fine_dims[1], : fine_dims[2]]
    return x_fine


def _binsize3(binsize, caller: str) -> Tuple[int, int, int]:
    """Expand a scalar bin size to 3 positive integers."""
    b = np.atleast_1d(np.asarray(binsize)).ravel()
    if b.size == 1:
        b = np.repeat(b, 3)
    if b.size != 3 or np.any(b < 1) or np.any(b != np.round(b)):
        raise ValueError(f"{caller}: binsize must be a positive integer or 3-vector")
    return int(b[0]), int(b[1]), int(b[2])


def sdjac(
    sd: Union[np.ndarray, dict],
    detphiflat: np.ndarray,
    csd: np.ndarray,
    sdindex: np.ndarray,
    reform: str = "real",
    sdmode: str = "y",
    rfcw: Union[int, List[int]] = 1,
    wavelengths: List = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobian block for simultaneous source-detector coupling estimation.

    Port of redbird/matlab/rbsdjac.m.  The measurement model is

        phi_hat(i, j) = csd[srcch(i)] * csd[detch(j)] * phi(i, j)

    so the complex Jacobian entry for coupling channel ``ch`` is
    ``model / csd[ch]`` for each of the two roles (source and detector) of
    every measurement (Fang 2004, Redbird90 rb2_inversion.f90).

    Parameters
    ----------
    sd : ndarray or dict
        Source-detector table(s) as returned by `sdmap`: columns
        ``[src_col, det_col, active, mode]`` with 0-based phi column indices
        (sources first, then detectors).  Its active rows must follow the
        order of ``detphiflat``.
    detphiflat : ndarray
        Flattened complex model prediction (already scaled by the current
        couplings), one entry per active measurement, concatenated over
        ``wavelengths``.
    csd : ndarray
        Current complex coupling coefficients, one per channel.
    sdindex : ndarray of int
        0-based coupling channel of every optode (sources first, then
        detectors, the same indices as sd columns 0-1); give a source and a
        detector the same channel when they are one physical optode.
    reform : str
        'real'/'reim' for the expanded complex form (unknowns Re(dc), Im(dc))
        or 'logphase' for the log-amplitude/phase form (unknowns d|c|,
        d angle(c)).
    sdmode : str
        'y' both roles (default), 's' source-only, 'd' detector-only.
    rfcw : int or list
        Source modality kept from a 4-column sd table (default 1).
    wavelengths : list, optional
        Keys of ``sd`` (when a dict) in the order used to flatten the data.

    Returns
    -------
    Asd : ndarray, shape (2*Nmeas, 2*Nch)
        Real Jacobian block; the first Nch columns are Re(dc) (or d|c|), the
        last Nch columns Im(dc) (or d angle(c)).
    srcch, detch : ndarray
        Per-measurement source and detector coupling channels.
    """
    csd = np.asarray(csd).ravel()
    sdindex = np.asarray(sdindex, dtype=int).ravel()
    nch = int(sdindex.max()) + 1
    modes = list(np.atleast_1d(rfcw)) + [3]

    if isinstance(sd, dict):
        keys = list(sd.keys()) if wavelengths is None else list(wavelengths)
        tables = [sd[k] for k in keys]
    else:
        tables = [sd] * (1 if wavelengths is None else len(wavelengths))

    srcch, detch = [], []
    for sdwv in tables:
        sdwv = np.atleast_2d(np.asarray(sdwv))
        if sdwv.shape[1] > 3:
            sdwv = sdwv[np.isin(sdwv[:, 3], modes)]
        kept = sdwv[:, 2] == 1 if sdwv.shape[1] > 2 else np.ones(len(sdwv), bool)
        srcch.append(sdindex[sdwv[kept, 0].astype(int)])
        detch.append(sdindex[sdwv[kept, 1].astype(int)])
    srcch = np.concatenate(srcch)
    detch = np.concatenate(detch)

    detphiflat = np.asarray(detphiflat).ravel()
    nrows = detphiflat.size
    if srcch.size != nrows:
        raise ValueError(
            f"sd table rows ({srcch.size}) do not match the flattened data length ({nrows})"
        )

    fitsrc = sdmode in ("y", "s")
    fitdet = sdmode in ("y", "d")
    rows = np.arange(nrows)

    if reform == "logphase":
        # d(log|phi_hat|)/d|c| = 1/|c|, d(angle(phi_hat))/d(angle(c)) = 1
        Alog = np.zeros((nrows, nch))
        Apha = np.zeros((nrows, nch))
        if fitsrc:
            Alog[rows, srcch] = 1.0 / np.abs(csd[srcch])
            Apha[rows, srcch] = 1.0
        if fitdet:
            Alog[rows, detch] += 1.0 / np.abs(csd[detch])
            Apha[rows, detch] += 1.0
        Z = np.zeros((nrows, nch))
        return np.block([[Alog, Z], [Z, Apha]]), srcch, detch

    # complex form J(:,ch) = model/csd(ch), expanded as [Re -Im; Im Re]
    Jsd = np.zeros((nrows, nch), dtype=complex)
    if fitsrc:
        Jsd[rows, srcch] = detphiflat / csd[srcch]
    if fitdet:
        Jsd[rows, detch] += detphiflat / csd[detch]
    Asd = np.block([[Jsd.real, -Jsd.imag], [Jsd.imag, Jsd.real]])
    return Asd, srcch, detch


def reginv(
    Amat: np.ndarray,
    rhs: np.ndarray,
    lambda_: float,
    Areg: dict = None,
    blocks: dict = None,
    **kwargs,
) -> np.ndarray:
    """
    Solve regularized linear system, auto-selecting method.

    Automatically chooses overdetermined or underdetermined solver
    based on matrix dimensions.
    """
    if Areg is None:
        Areg = {}

    if Amat.shape[0] >= Amat.shape[1]:
        LTL = Areg.get("ltl", None)
        return reginvover(Amat, rhs, lambda_, LTL, blocks, **kwargs)
    else:
        invR = Areg.get("lir", None)
        return reginvunder(Amat, rhs, lambda_, invR, blocks, **kwargs)


def reginvover(
    Amat: np.ndarray,
    rhs: np.ndarray,
    lambda_: float,
    LTL: np.ndarray = None,
    blocks: dict = None,
    **kwargs,
) -> np.ndarray:
    """
    Solve overdetermined Gauss-Newton normal equation.

    Solves: delta_mu = inv(J'J + lambda*(L'L)) * J' * (y - phi)
    """
    # Remove zero-sensitivity columns
    col_sum = np.sum(np.abs(Amat), axis=0)
    idx0 = np.where(col_sum != 0)[0]
    length0 = Amat.shape[1]

    if len(idx0) < length0:
        Amat = Amat[:, idx0]
        if LTL is not None and LTL.shape[0] > len(idx0):
            Lidx = idx0[idx0 < LTL.shape[0]]
            LTL = LTL[np.ix_(Lidx, Lidx)]

    # Remove zero-data rows
    row_sum = np.sum(np.abs(Amat), axis=1)
    valid_rows = row_sum != 0
    if np.sum(valid_rows) < Amat.shape[0]:
        Amat = Amat[valid_rows, :]
        rhs = rhs[valid_rows]

    # Build normal equation
    rhs_proj = Amat.T @ rhs.flatten()
    Hess = Amat.T @ Amat

    # Add regularization
    if LTL is None:
        Hess[np.diag_indices_from(Hess)] += lambda_
    else:
        if Hess.shape[0] == LTL.shape[0]:
            Hess = Hess + lambda_ * LTL
        else:
            nx = LTL.shape[0]
            for i in range(0, Hess.shape[0], nx):
                end_i = min(i + nx, Hess.shape[0])
                Hess[i:end_i, i:end_i] = (
                    Hess[i:end_i, i:end_i] + lambda_ * LTL[: end_i - i, : end_i - i]
                )

    # Normalize and solve
    Hess_norm, Gdiag = _normalize_diag(Hess)

    res = Gdiag * femsolve(sparse.csc_matrix(Hess_norm), Gdiag * rhs_proj, **kwargs)[0]

    # Restore full-length result
    if len(idx0) < length0:
        res_full = np.zeros(length0)
        res_full[idx0] = res
        res = res_full

    return res


def reginvunder(
    Amat: np.ndarray,
    rhs: np.ndarray,
    lambda_: float,
    invR: np.ndarray = None,
    blocks: dict = None,
    **kwargs,
) -> np.ndarray:
    """
    Solve underdetermined Gauss-Newton equation.

    Solves: delta_mu = inv(L'L)*J'*inv(J*inv(L'L)*J' + lambda*I)*(y-phi)
    """
    Alen = Amat.shape[1]

    # Remove zero columns
    col_sum = np.sum(np.abs(Amat), axis=0)
    idx = np.where(col_sum != 0)[0]
    if len(idx) < Alen:
        Amat = Amat[:, idx]

    # Remove zero rows
    row_sum = np.sum(np.abs(Amat), axis=1)
    valid_rows = row_sum != 0
    if np.sum(valid_rows) < Amat.shape[0]:
        Amat = Amat[valid_rows, :]
        rhs = rhs[valid_rows]

    # Apply regularization transform
    if invR is not None:
        nx = invR.shape[0]
        if nx == Amat.shape[1]:
            Amat = Amat @ invR
        elif blocks is not None:
            block_keys = list(blocks.keys())
            cumlen = np.cumsum([0] + [blocks[k][1] for k in block_keys])
            for i, k in enumerate(block_keys):
                if cumlen[i + 1] - cumlen[i] == nx:
                    Amat[:, cumlen[i] : cumlen[i + 1]] = (
                        Amat[:, cumlen[i] : cumlen[i + 1]] @ invR
                    )

    rhs = rhs.flatten()

    # Build Hessian in dual space
    Hess = Amat @ Amat.T
    Hess[np.diag_indices_from(Hess)] += lambda_

    # Normalize and solve
    Hess_norm, Gdiag = _normalize_diag(Hess)

    y = Gdiag * femsolve(sparse.csc_matrix(Hess_norm), Gdiag * rhs, **kwargs)[0]

    # Transform back to primal space
    if invR is not None:
        nx = invR.shape[0]
        if nx == Amat.shape[1]:
            res = invR @ (Amat.T @ y)
        else:
            res = Amat.T @ y
            if blocks is not None:
                block_keys = list(blocks.keys())
                cumlen = np.cumsum([0] + [blocks[k][1] for k in block_keys])
                for i, k in enumerate(block_keys):
                    if cumlen[i + 1] - cumlen[i] == nx:
                        res[cumlen[i] : cumlen[i + 1]] = (
                            invR @ res[cumlen[i] : cumlen[i + 1]]
                        )
    else:
        res = Amat.T @ y

    # Restore full length
    if len(idx) < Alen:
        res_full = np.zeros(Alen)
        res_full[idx] = res
        res = res_full

    return res


def matreform(
    Amat: np.ndarray, ymeas: np.ndarray, ymodel: np.ndarray, form: str = "complex"
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reformat matrix equation for different output forms.

    Parameters
    ----------
    form : str
        'complex': No transformation
        'real': Real-valued system
        'reim': Expand to [Re(x); Im(x)]
        'logphase': Log-amplitude and phase form
    """
    nblock = 1
    rhs = ymeas - ymodel

    if form == "complex":
        return Amat, rhs, nblock

    if form in ["real", "reim"]:
        newA = np.real(Amat)
        newrhs = np.real(rhs)

        if not np.isreal(rhs).all() and not np.isreal(Amat).all():
            if form == "reim":
                newA = np.block(
                    [[np.real(Amat), -np.imag(Amat)], [np.imag(Amat), np.real(Amat)]]
                )
            else:
                newA = np.vstack([np.real(Amat), np.imag(Amat)])
            newrhs = np.concatenate([np.real(rhs), np.imag(rhs)])
            nblock = 2

        return newA, newrhs, nblock

    if form == "logphase":
        temp = np.conj(ymodel) / np.abs(ymodel * ymodel)
        temp = temp[:, np.newaxis] * Amat if Amat.ndim == 2 else temp * Amat

        if np.isreal(ymodel).all():
            newA = np.real(temp)
            newrhs = np.log(np.abs(ymeas)) - np.log(np.abs(ymodel))
        else:
            newA = np.vstack([np.real(temp), np.imag(temp)])
            newrhs = np.concatenate(
                [
                    np.log(np.abs(ymeas)) - np.log(np.abs(ymodel)),
                    np.angle(ymeas) - np.angle(ymodel),
                ]
            )
            nblock = 2

        return newA, newrhs, nblock

    raise ValueError(f"Unknown form: {form}")


def matflat(Amat: Union[dict, np.ndarray], weight: np.ndarray = None) -> np.ndarray:
    """Flatten dict of matrices into single 2D matrix."""
    if isinstance(Amat, np.ndarray):
        return Amat

    if isinstance(Amat, dict):
        keys = list(Amat.keys())
        if weight is None:
            weight = np.ones(len(keys))

        first_val = Amat[keys[0]]
        if isinstance(first_val, dict):
            # Multi-wavelength: vertically concatenate
            inner_keys = list(first_val.keys())
            Anew = []
            for wv in inner_keys:
                row = np.hstack([Amat[k][wv] * weight[j] for j, k in enumerate(keys)])
                Anew.append(row)
            return np.vstack(Anew)
        else:
            # Single wavelength: horizontally concatenate
            return np.hstack([Amat[k] * weight[i] for i, k in enumerate(keys)])

    return Amat


def prior(seg: np.ndarray, priortype: str, params: dict = None) -> np.ndarray:
    """
    Generate structure-prior regularization matrix.

    Parameters
    ----------
    seg : ndarray
        Segmentation labels (node or element based) or composition matrix
    priortype : str
        'laplace': Laplacian prior within segments
        'helmholtz': Helmholtz-like prior with beta parameter
        'comp': Compositional prior for soft segmentation
    """
    if not priortype:
        return None

    params = params or {}

    if np.ndim(seg) == 1:
        # Label-based segmentation
        labels, inverse = np.unique(seg, return_inverse=True)
        counts = np.bincount(inverse)
        n = len(seg)

        if priortype == "laplace":
            Lmat = np.eye(n)
            for i, label in enumerate(labels):
                idx = np.where(inverse == i)[0]
                if counts[i] > 1:
                    Lmat[np.ix_(idx, idx)] = -1.0 / counts[i]
            np.fill_diagonal(Lmat, 1.0)
            return Lmat

        elif priortype == "helmholtz":
            beta = params.get("beta", 1.0)
            Lmat = np.eye(n)
            for i, label in enumerate(labels):
                idx = np.where(inverse == i)[0]
                if counts[i] > 1:
                    Lmat[np.ix_(idx, idx)] = -1.0 / (counts[i] + beta)
            np.fill_diagonal(Lmat, 1.0)
            return Lmat

    elif priortype == "comp" and seg.ndim == 2:
        # Compositional prior for soft segmentation
        alpha = params.get("alpha", 0.1)
        beta = params.get("beta", 1.0)
        n = seg.shape[0]
        nc = seg.shape[1]

        Lmat = sparse.lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dval = np.sum(np.abs(seg[i, :] - seg[j, :]))
                if dval < alpha * nc:
                    val = -alpha - dval / nc
                    Lmat[i, j] = val
                    Lmat[j, i] = val

        # Normalize rows
        rowsum = np.abs(np.array(Lmat.sum(axis=1)).flatten())
        for i in range(n):
            for j in range(n):
                if Lmat[i, j] != 0 and i != j:
                    Lmat[i, j] /= beta * np.sqrt(rowsum[i] * rowsum[j] + 1e-16)

        Lmat = Lmat + sparse.eye(n)
        return Lmat.tocsr()

    return None


def syncprop(cfg: dict, recon: dict) -> Tuple[dict, dict]:
    """
    Synchronize properties between forward and reconstruction meshes.

    Handles both single-mesh and dual-mesh reconstruction scenarios.

    For dual-mesh reconstruction:
    - recon mesh is typically coarser than forward mesh
    - mapid/mapweight map FORWARD mesh nodes to RECON mesh elements
    - We interpolate from recon mesh to forward mesh

    mapid contains 1-based element indices into recon["elem"].
    """
    # Use iso2mesh's meshinterp for interpolation
    try:
        from iso2mesh import meshinterp
    except ImportError:
        from .utility import meshinterp

    # Determine mesh sizes
    cfg_nn = cfg["node"].shape[0]
    cfg_ne = cfg["elem"].shape[0]

    if "node" in recon and "elem" in recon:
        recon_nn = recon["node"].shape[0]
        recon_ne = recon["elem"].shape[0]
    else:
        recon_nn = cfg_nn
        recon_ne = cfg_ne

    # Threshold to distinguish label-based from node/element-based
    # Use a small number that's clearly less than any reasonable mesh size
    label_threshold = 50

    if "param" in recon:
        # Map recon.param to cfg.param
        allkeys = list(recon["param"].keys())
        first_param = recon["param"][allkeys[0]]
        param_len = len(first_param) if hasattr(first_param, "__len__") else 1

        if param_len < label_threshold:
            # Label-based - direct copy (no interpolation needed)
            cfg["param"] = {
                k: v.copy() if hasattr(v, "copy") else v
                for k, v in recon["param"].items()
            }
        else:
            # Node/element based - need interpolation for dual-mesh
            if "param" not in cfg:
                cfg["param"] = {}

            for key in allkeys:
                if "mapid" in recon and "mapweight" in recon and "elem" in recon:
                    # Interpolate from recon mesh to forward mesh nodes
                    # Result should have cfg_nn rows - pass None for toval
                    cfg["param"][key] = meshinterp(
                        recon["param"][key],
                        recon["mapid"],
                        recon["mapweight"],
                        recon["elem"],  # 1-based, meshinterp converts
                        None,  # Create new array of correct size
                    )
                else:
                    # Same mesh - direct copy
                    cfg["param"][key] = recon["param"][key].copy()

    elif "prop" in recon:
        # Map recon.prop to cfg.prop
        if not isinstance(recon["prop"], dict):
            recon_prop_len = recon["prop"].shape[0]

            if recon_prop_len < label_threshold:
                # Label-based recon prop - direct copy
                cfg["prop"] = recon["prop"].copy()
            elif "mapid" in recon and "mapweight" in recon and "elem" in recon:
                # Node-based recon prop with dual mesh - interpolate to forward mesh
                # The result should be node-based on the FORWARD mesh (cfg_nn rows)
                # Pass None for toval to create new array of correct size
                cfg["prop"] = meshinterp(
                    recon["prop"],
                    recon["mapid"],
                    recon["mapweight"],
                    recon["elem"],
                    None,  # Don't pass cfg["prop"] - it may be label-based
                )
            else:
                # Same mesh or no mapping - direct copy
                cfg["prop"] = recon["prop"].copy()
        else:
            # Multi-wavelength
            allkeys = list(recon["prop"].keys())
            first_prop = recon["prop"][allkeys[0]]
            recon_prop_len = first_prop.shape[0]

            if recon_prop_len < label_threshold:
                # Label-based - direct copy
                cfg["prop"] = {k: v.copy() for k, v in recon["prop"].items()}
            elif "mapid" in recon and "mapweight" in recon and "elem" in recon:
                # Node-based with dual mesh - interpolate
                cfg["prop"] = {}
                for k in allkeys:
                    cfg["prop"][k] = meshinterp(
                        recon["prop"][k],
                        recon["mapid"],
                        recon["mapweight"],
                        recon["elem"],
                        None,  # Create new array of correct size
                    )
            else:
                # Same mesh - direct copy
                cfg["prop"] = {k: v.copy() for k, v in recon["prop"].items()}

    return cfg, recon


def _normalize_diag(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize matrix to have unit diagonal for better conditioning."""
    Adiag = np.diag(A)
    di = 1.0 / np.sqrt(np.abs(Adiag) + 1e-16)
    Anorm = (di[:, np.newaxis] * di[np.newaxis, :]) * A
    return Anorm, di


def _flatten_detphi(
    detphi: Union[np.ndarray, dict],
    sd: Union[np.ndarray, dict],
    wavelengths: List[str],
    rfcw: List[int],
) -> np.ndarray:
    """Flatten detector measurements from nested dict to 1D array.

    When the sd table lists every source-detector pair of detphi (in the
    ``detphi.ravel()`` order that `sdmap` produces), only its active rows are
    kept, as `jac` does for the Jacobian (rbrunrecon.m's sdkeep).
    """
    if isinstance(detphi, np.ndarray):
        flat = detphi.flatten()
        sdarr = _sd_single(sd, wavelengths)
        if sdarr is not None and sdarr.shape[1] > 2 and sdarr.shape[0] == flat.size:
            flat = flat[sdarr[:, 2] == 1]
        return flat

    result = []
    for wv in wavelengths:
        if isinstance(detphi, dict):
            phi_wv = detphi.get(wv, detphi)
        else:
            phi_wv = detphi

        if isinstance(phi_wv, dict):
            for md in rfcw:
                result.extend(phi_wv.get(md, {}).get("detphi", phi_wv).flatten())
        else:
            result.extend(np.asarray(phi_wv).flatten())

    return np.array(result)


def _sd_single(sd, wavelengths: List[str]):
    """the sd table of a single-wavelength run, or None"""
    if isinstance(sd, dict):
        if len(sd) != 1:
            return None
        sd = next(iter(sd.values()))
    return None if sd is None else np.asarray(sd)


def _remap_jacobian(J: np.ndarray, recon: dict, cfg: dict) -> np.ndarray:
    """
    Remap Jacobian from forward mesh nodes to reconstruction mesh nodes.

    Parameters
    ----------
    J : ndarray
        Jacobian on forward mesh (Nsd x Nn_forward)
    recon : dict
        Reconstruction structure with mapid (1-based), mapweight, elem (1-based)
    cfg : dict
        Forward structure

    Returns
    -------
    J_new : ndarray
        Jacobian on reconstruction mesh (Nsd x Nn_recon)
    """
    nn_recon = recon["node"].shape[0]
    nn_forward = J.shape[1]
    nsd = J.shape[0]

    J_new = np.zeros((nsd, nn_recon), dtype=J.dtype)

    mapid = recon["mapid"]  # 1-based element indices into recon mesh
    mapweight = recon["mapweight"]  # Barycentric coordinates (Nn_forward x 4)

    # Convert 1-based elem to 0-based for numpy indexing
    elem_0 = recon["elem"][:, :4].astype(int) - 1
    n_elem = elem_0.shape[0]

    # For each forward mesh node, distribute its Jacobian contribution
    # to the reconstruction mesh nodes of the enclosing element
    for i in range(nn_forward):
        eid_raw = mapid[i]

        # Skip NaN entries (forward node outside recon mesh)
        if np.isnan(eid_raw):
            continue

        eid = int(eid_raw) - 1  # Convert 1-based to 0-based

        # Bounds check on element index
        if eid < 0 or eid >= n_elem:
            continue

        # Get reconstruction mesh node indices for this element
        node_ids = elem_0[eid, :]  # 4 node indices (0-based)

        # Bounds check on node indices
        if np.any(node_ids < 0) or np.any(node_ids >= nn_recon):
            continue

        # Distribute Jacobian contribution using barycentric weights
        for j in range(4):
            node_idx = node_ids[j]
            J_new[:, node_idx] += J[:, i] * mapweight[i, j]

    return J_new


def _masksum(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Sum columns by segmentation mask for label-based reconstruction.

    Compresses node-based Jacobian to label-based by summing all nodes
    with the same label.
    """
    labels = np.unique(mask)
    result = np.zeros((data.shape[0], len(labels)), dtype=data.dtype)

    for i, label in enumerate(labels):
        idx = mask == label
        result[:, i] = np.sum(data[:, idx], axis=1)

    return result


def multispectral(
    sd: Union[np.ndarray, dict],
    cfg: dict,
    Jmua: Union[np.ndarray, dict],
    y0: Union[np.ndarray, dict],
    phi: Union[np.ndarray, dict],
    params: dict,
    rfcw: Union[int, list] = 1,
    Jd: Union[np.ndarray, dict, None] = None,
    prop: Union[np.ndarray, dict, None] = None,
) -> Tuple[Union[dict, np.ndarray], np.ndarray, np.ndarray]:
    """Concatenate multi-spectral forward data into a single linear system.

    Port of redbird/matlab/rbmultispectral.m.  When ``Jmua`` is keyed
    by wavelength, builds either a chromophore-space Jacobian (DOT) or
    an eps/sigma Jacobian (MWT) by chaining through the per-wavelength
    extinction coefficients / Helmholtz physics.  Optionally adds the
    scattering-amplitude and scattering-power Jacobians when ``Jd`` is
    supplied and the parameter struct names ``scatamp`` / ``scatpow``
    (or the 500 nm-normalized variants ``scatamp500`` / ``scatpow500``).

    Parameters
    ----------
    sd : ndarray or dict
        Source-detector mapping table.  ``(Nsd, {3,4})`` or a dict
        keyed by wavelength.
    cfg : dict
        Forward configuration (used for ``cfg.omega`` in the MWT path).
    Jmua : ndarray or dict
        Per-wavelength ``mua`` Jacobian.  ``(Nsd, Nn)`` per wavelength;
        dict keyed by wavelength when multi-spectral.
    y0 : ndarray or dict
        Measurement data (matching shape of detphi).
    phi : ndarray or dict
        Model prediction at detectors.
    params : dict
        Parameter struct (chromophore concentrations OR eps/sigma for
        MWT, optionally with ``scatamp``/``scatpow`` for scattering).
    rfcw : int or list, default 1
        Forward modes to flatten (1 = RF, 2 = CW, [1, 2] = both).
    Jd : ndarray or dict, optional
        Diffusion-coefficient Jacobian.  Triggers the scattering chain
        when ``params`` names ``scatamp``/``scatpow``.
    prop : dict, optional
        Per-wavelength property table.  Required for the scattering
        chain (used to extract dcoeff).

    Returns
    -------
    newJ : dict
        Concatenated chromophore / eps-sigma / scattering Jacobian.
    newy0 : ndarray
        Concatenated measurement vector (sd column 2 == 1 filter).
    newphi : ndarray
        Concatenated model prediction vector.
    """
    if isinstance(rfcw, int):
        rfcw = [rfcw]

    is_helmholtz = isinstance(params, dict) and (
        "epsilon" in params or "sigma" in params
    )

    newJ: Union[dict, np.ndarray] = {}

    if isinstance(Jmua, dict):
        wavelengths = list(Jmua.keys())

        if is_helmholtz:
            # MWT: chain J_mua -> J_eps / J_sigma per frequency
            eps0_mm = 8.854187817e-15
            mu0_mm = 4 * np.pi * 1e-10
            omegas = np.zeros(len(wavelengths))
            for i, wv in enumerate(wavelengths):
                omega = cfg.get("omega", 0)
                if isinstance(omega, dict):
                    omegas[i] = omega.get(wv, 0)
                else:
                    omegas[i] = omega
            weight_eps = -(omegas**2) * mu0_mm * eps0_mm
            weight_sigma = 1j * omegas * mu0_mm

            if "epsilon" in params:
                newJ["epsilon"] = matflat(Jmua, weight_eps)
            if "sigma" in params:
                newJ["sigma"] = matflat(Jmua, weight_sigma)
        else:
            # DOT chromophore + (optional) scattering chain
            paramlist = list(params.keys()) if isinstance(params, dict) else []
            has_norm_scat = "scatamp500" in paramlist and "scatpow500" in paramlist
            has_legacy_scat = "scatamp" in paramlist and "scatpow" in paramlist

            Jscat = None
            if (
                Jd is not None
                and (has_norm_scat or has_legacy_scat)
                and isinstance(Jd, dict)
                and prop is not None
            ):
                # build per-wavelength D from prop
                dcoeff = {}
                for wv in wavelengths:
                    dtemp = (
                        np.asarray(prop[wv])
                        if isinstance(prop, dict)
                        else np.asarray(prop)
                    )
                    # drop the "outside" row (row 0) when prop has Nn+1 rows
                    if dtemp.ndim == 2 and dtemp.shape[0] < Jd[wv].shape[1]:
                        dtemp = dtemp[1:, :]
                    dcoeff[wv] = (1.0 / (3.0 * (dtemp[:, 0] + dtemp[:, 1]))).T

                if has_norm_scat:
                    Jscat = jacscat(
                        Jd,
                        dcoeff,
                        params["scatpow500"],
                        wv=wavelengths,
                        lref=500.0,
                        suffix="500",
                    )
                else:
                    Jscat = jacscat(
                        Jd,
                        dcoeff,
                        params["scatpow"],
                        wv=wavelengths,
                        lref=1e9,
                    )

            chromophores = [
                k for k in paramlist if k in ("hbo", "hbr", "water", "lipids", "aa3")
            ]
            newJ = jacchrome(Jmua, chromophores) if chromophores else {}

            if Jscat is not None:
                for key, val in Jscat.items():
                    newJ[key] = val
    else:
        newJ["mua"] = Jmua
        if Jd is not None and not isinstance(Jd, dict):
            newJ["dcoeff"] = Jd

    if Jd is not None and not isinstance(Jd, dict) and "dcoeff" not in newJ:
        newJ["dcoeff"] = Jd

    # ---- flatten y0 ----
    newy0 = _flatten_msdata(y0, sd, rfcw)
    # ---- flatten phi (model prediction) ----
    newphi = _flatten_msdata(phi, sd, rfcw)

    return newJ, newy0, newphi


def _flatten_msdata(
    data: Union[np.ndarray, dict], sd: Union[np.ndarray, dict], rfcw: list
) -> np.ndarray:
    """Helper for multispectral: stack per-wavelength detphi vectors
    filtered by the sd column-2 active-pair mask, matching the inner
    loops of rbmultispectral.m lines 133-199."""
    if not isinstance(data, dict):
        return np.asarray(data)

    wavelengths = list(data.keys())
    out_blocks = {j: [] for j in rfcw}

    for wv in wavelengths:
        sdwv = sd[wv] if isinstance(sd, dict) else sd
        sd_arr = np.asarray(sdwv)
        if sd_arr.shape[1] == 3:
            sd_arr = np.column_stack([sd_arr, np.full(sd_arr.shape[0], rfcw[0])])

        for j in rfcw:
            mask_pair = (sd_arr[:, 3] == j) | (sd_arr[:, 3] == 3)
            sd_active = sd_arr[mask_pair]
            keep = sd_active[:, 2] == 1
            tempphi = np.asarray(data[wv]).ravel(order="F")[: sd_active.shape[0]]
            tempphi = tempphi[keep]
            out_blocks[j].append(tempphi)

    stacked = {
        j: np.concatenate(out_blocks[j]) if out_blocks[j] else np.array([])
        for j in rfcw
    }

    if len(rfcw) == 1:
        return stacked[rfcw[0]]
    return np.concatenate([stacked[j] for j in rfcw])


def createinv(
    Amat: Union[np.ndarray, dict],
    ymeas: Union[np.ndarray, dict],
    ymodel: Union[np.ndarray, dict],
    params: Optional[dict] = None,
    output: str = "complex",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Reformulate the inverse problem in complex / real / log-phase form.

    Port of redbird/matlab/rbcreateinv.m.  Splits a complex-valued
    linear system into a real-valued one when ``output`` selects
    ``'real'`` (block-diagonal Re/Im) or ``'logphase'`` (log-amplitude
    + unwrapped-phase form for RF measurements).

    Parameters
    ----------
    Amat : ndarray or dict
        LHS (per-wavelength when dict).
    ymeas, ymodel : ndarray or dict
        Complex measurement and model vectors.
    params : dict, optional
        Chromophore parameter struct.  When supplied, the columns of the
        per-wavelength Amat are weighted by the extinction coefficients
        and stacked per chromophore.
    output : {'complex', 'real', 'logphase'}, default 'complex'

    Returns
    -------
    finalAmat : ndarray
        Reformulated LHS matrix.
    finalrhs : ndarray
        Reformulated RHS vector ``ymeas - ymodel`` (with sign + log-phase
        transformation applied as appropriate).
    nblock : int
        ``2`` when the complex-to-real split doubled the system,
        ``1`` otherwise.
    """
    if not isinstance(Amat, dict):
        Amat = {"_": Amat}
        ymeas = {"_": ymeas}
        ymodel = {"_": ymodel}

    wavelengths = list(Amat.keys())
    newA: Dict[str, np.ndarray] = {}
    newrhs: Dict[str, np.ndarray] = {}
    nblock = 1

    if output == "complex":
        newA = {wv: Amat[wv] for wv in wavelengths}
        for wv in wavelengths:
            newrhs[wv] = ymeas[wv] - ymodel[wv]
    else:
        for wv in wavelengths:
            A_wv = np.asarray(Amat[wv])
            rhs = np.asarray(ymeas[wv]) - np.asarray(ymodel[wv])

            if output == "real":
                if np.iscomplexobj(rhs) and np.iscomplexobj(A_wv):
                    newA[wv] = np.block(
                        [[A_wv.real, -A_wv.imag], [A_wv.imag, A_wv.real]]
                    )
                    newrhs[wv] = np.concatenate([rhs.real, rhs.imag])
                    nblock = 2
                else:
                    newA[wv] = A_wv.real
                    newrhs[wv] = rhs.real

            elif output == "logphase":
                ymod = np.asarray(ymodel[wv])
                temp_scalar = np.conj(ymod) / (np.abs(ymod) ** 2)
                # broadcast: each row of A weighted by ymod-scalar at that row
                temp = temp_scalar[:, np.newaxis] * A_wv
                log_diff = np.log(np.abs(np.asarray(ymeas[wv]))) - np.log(np.abs(ymod))

                if np.isrealobj(ymod):
                    newA[wv] = temp.real
                    newrhs[wv] = log_diff.ravel()
                else:
                    phase_diff = np.angle(np.asarray(ymeas[wv])) - np.angle(ymod)
                    newA[wv] = np.vstack([temp.real, temp.imag])
                    newrhs[wv] = np.concatenate([log_diff.ravel(), phase_diff.ravel()])
                    nblock = 2
            else:
                raise ValueError(f"Unknown output mode: {output!r}")

    if isinstance(params, dict):
        chromos = [
            k for k in params.keys() if k in ("hbo", "hbr", "water", "lipids", "aa3")
        ]
        if not chromos:
            raise ValueError("createinv: params must contain at least one chromophore")
        extins, _ = extinction(wavelengths, chromos)
        finalA_rows = []
        finalrhs_rows = []
        for i, wv in enumerate(wavelengths):
            row_blocks = [newA[wv] * extins[i, j] for j in range(len(chromos))]
            finalA_rows.append(np.hstack(row_blocks))
            finalrhs_rows.append(newrhs[wv])
        finalAmat = np.vstack(finalA_rows)
        finalrhs = np.concatenate(finalrhs_rows)
    else:
        finalAmat = np.vstack([newA[wv] for wv in wavelengths])
        finalrhs = np.concatenate([newrhs[wv] for wv in wavelengths])

    return finalAmat, finalrhs, nblock


def regemperical(Hess: np.ndarray, residual: float, alpha: float) -> float:
    """Empirical regularization parameter estimate.

    Port of redbird/matlab/rbregemperical.m.

        lambda = alpha * mean(diag(Hess)) * residual^2

    Parameters
    ----------
    Hess : ndarray  (Np x Np)
        Gauss-Newton Hessian.
    residual : float
        Total data-model misfit residual from the previous iteration.
    alpha : float
        Empirical scaling factor.

    Returns
    -------
    lambda_ : float
        Empirical regularization parameter.
    """
    ggav = np.mean(np.diag(np.asarray(Hess)))
    return alpha * ggav * residual * residual
