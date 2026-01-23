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

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Optional, Union, List, Any
import warnings


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
    from . import forward, utility, property as prop_module
    import time

    # Parse options
    maxiter = kwargs.get("maxiter", 5)
    lambda_ = kwargs.get("lambda_", recon.get("lambda", 0.05))
    report = kwargs.get("report", True)
    tol = kwargs.get("tol", 0)
    reform = kwargs.get("reform", "real")
    solverflag = kwargs.get("solverflag", {})
    rfcw = kwargs.get("rfcw", [1])
    prior_type = kwargs.get("prior", "")

    if isinstance(rfcw, int):
        rfcw = [rfcw]

    if sd is None:
        sd = utility.sdmap(cfg)

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

    # Main iteration loop
    for iteration in range(maxiter):
        t_start = time.time()

        # Sync properties between recon and forward mesh
        if "param" in recon or "prop" in recon:
            cfg, recon = syncprop(cfg, recon)

        # Update cfg.prop from cfg.param if multi-spectral
        if "param" in cfg and isinstance(cfg.get("prop"), dict):
            cfg["prop"] = prop_module.updateprop(cfg)

        # Run forward simulation
        detphi, phi = forward.runforward(cfg, solverflag=solverflag, sd=sd, rfcw=rfcw)

        # Build Jacobians
        wavelengths = [""]
        if isinstance(cfg.get("prop"), dict):
            wavelengths = list(cfg["prop"].keys())

        Jmua = {}

        for wv in wavelengths:
            sdwv = sd.get(wv, sd) if isinstance(sd, dict) else sd
            phiwv = phi.get(wv, phi) if isinstance(phi, dict) else phi

            # jac expects 1-based elem, converts internally
            Jmua_n, Jmua_e = forward.jac(
                sdwv, phiwv, cfg["deldotdel"], cfg["elem"], cfg["evol"]
            )
            Jmua[wv] = Jmua_n

        # Build chromophore Jacobians if multi-spectral
        if isinstance(cfg.get("prop"), dict) and "param" in cfg:
            chromophores = [
                k
                for k in cfg["param"].keys()
                if k in ["hbo", "hbr", "water", "lipids", "aa3"]
            ]
            if chromophores:
                Jmua = forward.jacchrome(Jmua, chromophores)

        # Flatten measurement data
        detphi0_flat = _flatten_detphi(detphi0, sd, wavelengths, rfcw)
        detphi_flat = _flatten_detphi(detphi, sd, wavelengths, rfcw)

        # Get block structure
        if isinstance(Jmua, dict):
            blocks = {k: v.shape for k, v in Jmua.items()}
        else:
            blocks = {"mua": Jmua.shape}

        # Flatten Jacobian and reformat
        Jflat = matflat(Jmua)

        if reform != "complex":
            Jflat, misfit, nblock = matreform(Jflat, detphi0_flat, detphi_flat, reform)
        else:
            misfit = detphi0_flat - detphi_flat

        # Map Jacobian to recon mesh if dual-mesh
        if "mapid" in recon and "mapweight" in recon:
            Jflat = _remap_jacobian(Jflat, recon, cfg)

        # Compress for segmented reconstruction
        if "seg" in recon and np.ndim(recon["seg"]) == 1:
            Jflat = _masksum(Jflat, recon["seg"])

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

        # Solve inverse problem
        dmu = reginv(Jflat, misfit, lambda_, Aregu, blocks, **solverflag)

        # Parse update and apply
        update = {}
        idx = 0
        output_keys = list(blocks.keys())
        for key in output_keys:
            size = blocks[key][1]
            dx = dmu[idx : idx + size]
            update[key] = dx
            idx += size

            # Apply update to appropriate structure
            if key in ["mua", "dcoeff"]:
                propidx = 0 if key == "mua" else 1
                if "prop" in recon:
                    if key == "dcoeff":
                        dcoeff = 1.0 / (3 * recon["prop"][:, propidx])
                        dcoeff = dcoeff + dx
                        recon["prop"][:, propidx] = 1.0 / (3 * dcoeff)
                    else:
                        recon["prop"][:, propidx] = recon["prop"][:, propidx] + dx
            elif key in ["hbo", "hbr", "water", "lipids", "scatamp", "scatpow"]:
                if "node" in recon:
                    recon["param"][key] = recon["param"][key] + dx
                else:
                    cfg["param"][key] = cfg["param"][key] + dx

        updates.append(update)

        if report:
            elapsed = time.time() - t_start
            rel_resid = resid[iteration] / resid[0] if iteration > 0 else 1.0
            print(
                f"iter [{iteration+1:4d}]: residual={resid[iteration]:.6e}, "
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

    if sparse.issparse(Hess_norm):
        res = Gdiag * spsolve(Hess_norm, Gdiag * rhs_proj)
    else:
        res = Gdiag * np.linalg.solve(Hess_norm, Gdiag * rhs_proj)

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

    if sparse.issparse(Hess_norm):
        y = Gdiag * spsolve(Hess_norm, Gdiag * rhs)
    else:
        y = Gdiag * np.linalg.solve(Hess_norm, Gdiag * rhs)

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
    mapid contains 0-based element indices from tsearchn.
    """
    from . import utility

    # Determine if we're in label mode or node/element mode
    labelmax = min(cfg["node"].shape[0], cfg["elem"].shape[0])
    if "node" in recon and "elem" in recon:
        labelmax = min(recon["node"].shape[0], recon["elem"].shape[0])

    if "param" in recon:
        # Map recon.param to cfg.param
        allkeys = list(recon["param"].keys())
        first_param = recon["param"][allkeys[0]]

        if len(first_param) < labelmax:
            # Label-based - direct copy
            cfg["param"] = recon["param"].copy()
        else:
            # Node/element based - interpolate using mapping
            if "param" not in cfg:
                cfg["param"] = {}
            for key in allkeys:
                if "mapid" in recon and "mapweight" in recon:
                    # mapid is 0-based, meshinterp handles the indexing
                    cfg["param"][key] = utility.meshinterp(
                        recon["param"][key],
                        recon["mapid"],
                        recon["mapweight"],
                        recon["elem"],  # 1-based, meshinterp converts
                        cfg["param"].get(key),
                    )
                else:
                    cfg["param"][key] = recon["param"][key].copy()

    elif "prop" in recon:
        # Map recon.prop to cfg.prop
        if not isinstance(recon["prop"], dict):
            if recon["prop"].shape[0] < labelmax:
                cfg["prop"] = recon["prop"].copy()
            elif "mapid" in recon:
                cfg["prop"] = utility.meshinterp(
                    recon["prop"],
                    recon["mapid"],
                    recon["mapweight"],
                    recon["elem"],
                    cfg["prop"],
                )
        else:
            # Multi-wavelength
            allkeys = list(recon["prop"].keys())
            if recon["prop"][allkeys[0]].shape[0] < labelmax:
                cfg["prop"] = {k: v.copy() for k, v in recon["prop"].items()}
            elif "mapid" in recon:
                cfg["prop"] = {}
                for k in allkeys:
                    cfg["prop"][k] = utility.meshinterp(
                        recon["prop"][k],
                        recon["mapid"],
                        recon["mapweight"],
                        recon["elem"],
                        cfg.get("prop", {}).get(k),
                    )

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
        Reconstruction structure with mapid (0-based), mapweight, elem (1-based)
    cfg : dict
        Forward structure

    Returns
    -------
    J_new : ndarray
        Jacobian on reconstruction mesh (Nsd x Nn_recon)
    """
    nn_recon = recon["node"].shape[0]
    nsd = J.shape[0]

    J_new = np.zeros((nsd, nn_recon), dtype=J.dtype)

    mapid = recon["mapid"]  # 0-based element indices
    mapweight = recon["mapweight"]  # Barycentric coordinates (Nn_forward x 4)

    # Convert 1-based elem to 0-based for numpy indexing
    elem_0 = recon["elem"][:, :4].astype(int) - 1

    # For each forward mesh node, distribute its Jacobian contribution
    # to the reconstruction mesh nodes of the enclosing element
    for i in range(J.shape[1]):
        if not np.isnan(mapid[i]):
            eid = int(mapid[i])  # 0-based element index
            for j in range(4):
                node_idx = elem_0[eid, j]  # 0-based node index
                J_new[:, node_idx] += J[:, i : i + 1].flatten() * mapweight[i, j]

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
