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
        Options: maxiter, lambda_, tol, reform, report, prior

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

    if sd is None:
        sd = sdmap(cfg)

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

            # Note: Jext["dcoeff"] is currently not consumed by the inversion
            # loop (D-coefficient recon requires extending Jmua's dict shape;
            # not in scope for the initial MC integration).
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

        # Solve inverse problem
        dmu = reginv(
            Jflat, misfit, lambda_, Aregu, blocks, method=solvermethod, **solverflag
        )
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
    """Flatten detector measurements from nested dict to 1D array."""
    if isinstance(detphi, np.ndarray):
        return detphi.flatten()

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

    Port of redbird-m/matlab/rbmultispectral.m.  When ``Jmua`` is keyed
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

    Port of redbird-m/matlab/rbcreateinv.m.  Splits a complex-valued
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

    Port of redbird-m/matlab/rbregemperical.m.

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
