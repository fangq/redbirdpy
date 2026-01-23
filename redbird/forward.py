"""
Redbird Forward Module - FEM-based forward modeling for diffuse optics.

INDEX CONVENTION: All mesh indices (elem, face) stored in cfg are 1-based
to match MATLAB/iso2mesh. This module converts to 0-based internally when
indexing numpy arrays, using local variables named with '_0' suffix.

Functions:
    runforward: Main forward solver for all sources/wavelengths
    femlhs: Build FEM left-hand-side (stiffness) matrix
    femrhs: Build FEM right-hand-side vector
    femsolve: Solve linear system with various methods
    femgetdet: Extract detector values from forward solution
    deldotdel: Compute gradient dot product operator
    jac: Compute Jacobian matrices using adjoint method
"""
"""
Redbird Forward Module - FEM-based forward modeling for diffuse optics.

OPTIMIZED VERSION - Preserves original algorithm, optimizes implementation:
1. LU factorization reuse for multiple RHS
2. Vectorized Jacobian computation  
3. Bounding box acceleration for point location
4. Pre-allocated arrays for sparse matrix assembly

INDEX CONVENTION: All mesh indices (elem, face) stored in cfg are 1-based
to match MATLAB/iso2mesh. This module converts to 0-based internally.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab, splu
from typing import Dict, Tuple, Optional, Union, List, Any
import warnings

# Speed of light in mm/s
C0 = 299792458000.0
R_C0 = 1.0 / C0


def runforward(cfg: dict, **kwargs) -> Tuple[Any, Any]:
    """
    Perform forward simulations at all sources and all wavelengths.
    """
    from . import utility

    solverflag = kwargs.get("solverflag", {})
    rfcw = kwargs.get("rfcw", [1])
    if isinstance(rfcw, int):
        rfcw = [rfcw]

    if "deldotdel" not in cfg or cfg["deldotdel"] is None:
        cfg["deldotdel"], _ = deldotdel(cfg)

    wavelengths = [""]
    if isinstance(cfg.get("prop"), dict):
        wavelengths = list(cfg["prop"].keys())

    sd = kwargs.get("sd")
    if sd is None:
        sd = utility.sdmap(cfg)
    if not isinstance(sd, dict):
        sd = {wv: sd for wv in wavelengths}

    Amat = {}
    detval_out = {md: {"detphi": {}} for md in rfcw}
    phi_out = {md: {"phi": {}} for md in rfcw}

    for wv in wavelengths:
        for md in rfcw:
            rhs, loc, bary, optode = femrhs(cfg, sd, wv, md)
            Amat[wv] = femlhs(cfg, cfg["deldotdel"], wv, md)
            phi_sol, flag = femsolve(Amat[wv], rhs, **solverflag)
            phi_out[md]["phi"][wv] = phi_sol
            detval = femgetdet(phi_sol, cfg, loc, bary)
            detval_out[md]["detphi"][wv] = detval

    if len(wavelengths) == 1:
        Amat = Amat[wavelengths[0]]
        for md in rfcw:
            phi_out[md]["phi"] = phi_out[md]["phi"][wavelengths[0]]
            detval_out[md]["detphi"] = detval_out[md]["detphi"][wavelengths[0]]

    if len(rfcw) == 1:
        phi_out = phi_out[rfcw[0]]["phi"]
        detval_out = detval_out[rfcw[0]]["detphi"]

    return detval_out, phi_out


def deldotdel(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute del(phi_i) dot del(phi_j) for FEM assembly.

    PRESERVES ORIGINAL ALGORITHM - only optimizes array operations.
    """
    node = cfg["node"]
    evol = cfg["evol"]

    # Convert 1-based elem to 0-based for numpy indexing
    elem_0 = cfg["elem"][:, :4].astype(np.int32) - 1
    ne = elem_0.shape[0]

    # Reshape nodes for vectorized computation: (Ne, 4, 3) -> (3, 4, Ne)
    no = node[elem_0, :].transpose(2, 1, 0)  # Shape: (3, 4, Ne)

    delphi = np.zeros((3, 4, ne))

    # Column indices for cross-product computation (original algorithm)
    col = np.array([[3, 1, 2, 1], [2, 0, 3, 2], [1, 3, 0, 3], [0, 2, 1, 0]])

    # evol needs to be shape (ne,) for broadcasting
    evol_inv = 1.0 / (evol * 6.0)  # Shape: (ne,)

    # Original algorithm preserved exactly
    for coord in range(3):
        idx = [c for c in range(3) if c != coord]
        for i in range(4):
            # Each term is shape (ne,)
            term1 = no[idx[0], col[i, 0], :] - no[idx[0], col[i, 1], :]
            term2 = no[idx[1], col[i, 2], :] - no[idx[1], col[i, 3], :]
            term3 = no[idx[0], col[i, 2], :] - no[idx[0], col[i, 3], :]
            term4 = no[idx[1], col[i, 0], :] - no[idx[1], col[i, 1], :]

            delphi[coord, i, :] = (term1 * term2 - term3 * term4) * evol_inv

    result = np.zeros((ne, 10))
    count = 0
    for i in range(4):
        for j in range(i, 4):
            result[:, count] = np.sum(delphi[:, i, :] * delphi[:, j, :], axis=0)
            count += 1

    result *= evol[:, np.newaxis]

    return result, delphi


def femlhs(
    cfg: dict, deldotdel_mat: np.ndarray, wavelength: str = "", mode: int = 1
) -> sparse.csr_matrix:
    """
    Create FEM stiffness matrix - optimized assembly with original algorithm.
    """
    nn = cfg["node"].shape[0]
    ne = cfg["elem"].shape[0]
    evol = cfg["evol"]
    area = cfg["area"]

    # Convert 1-based to 0-based
    elem_0 = cfg["elem"][:, :4].astype(np.int32) - 1
    face_0 = cfg["face"].astype(np.int32) - 1

    # Get properties for current wavelength
    if isinstance(cfg.get("prop"), dict) and wavelength:
        props = cfg["prop"][wavelength]
        reff = (
            cfg["reff"][wavelength]
            if isinstance(cfg.get("reff"), dict)
            else cfg["reff"]
        )
        omega = (
            cfg["omega"].get(wavelength, 0)
            if isinstance(cfg.get("omega"), dict)
            else cfg.get("omega", 0)
        )
    else:
        props = cfg["prop"]
        reff = cfg.get("reff", 0.493)
        omega = cfg.get("omega", 0)

    if mode == 2:
        omega = 0

    # Extract mua and musp (original logic preserved)
    seg = cfg.get("seg", None)
    if props.shape[0] == nn or props.shape[0] == ne:
        mua = props[:, 0]
        musp = props[:, 1] * (1 - props[:, 2]) if props.shape[1] >= 3 else props[:, 1]
        nref = props[:, 3] if props.shape[1] >= 4 else 1.37
    elif seg is not None:
        seg_idx = np.clip(seg.astype(np.int32), 0, props.shape[0] - 1)
        mua = props[seg_idx, 0]
        musp = (
            props[seg_idx, 1] * (1 - props[seg_idx, 2])
            if props.shape[1] >= 3
            else props[seg_idx, 1]
        )
        nref = props[seg_idx[0], 3] if props.shape[1] >= 4 else 1.37
    else:
        raise ValueError("Property format not recognized")

    dcoeff = 1.0 / (3.0 * (mua + musp))
    Reff = reff

    # Pre-allocate lists (faster than repeated extend)
    rows_list = []
    cols_list = []
    vals_list = []

    offdiag_idx = [1, 2, 3, 5, 6, 8]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Element-based assembly (original algorithm)
    if len(mua) == ne:
        for k, (i, j) in enumerate(pairs):
            rows_list.append(elem_0[:, i])
            cols_list.append(elem_0[:, j])
            val = deldotdel_mat[:, offdiag_idx[k]] * dcoeff + 0.05 * mua * evol
            if omega > 0:
                val = val.astype(complex) + 1j * 0.05 * omega * R_C0 * nref * evol
            vals_list.append(val)

            rows_list.append(elem_0[:, j])
            cols_list.append(elem_0[:, i])
            vals_list.append(val)

        diag_idx = [0, 4, 7, 9]
        for k in range(4):
            rows_list.append(elem_0[:, k])
            cols_list.append(elem_0[:, k])
            val = deldotdel_mat[:, diag_idx[k]] * dcoeff + 0.10 * mua * evol
            if omega > 0:
                val = val.astype(complex) + 1j * 0.10 * omega * R_C0 * nref * evol
            vals_list.append(val)
    else:
        # Node-based properties (original algorithm)
        w1 = (1 / 120) * np.array(
            [
                [2, 2, 1, 1],
                [2, 1, 2, 1],
                [2, 1, 1, 2],
                [1, 2, 2, 1],
                [1, 2, 1, 2],
                [1, 1, 2, 2],
            ]
        ).T
        w2 = (1 / 60) * (np.diag([2, 2, 2, 2]) + 1)

        mua_e = mua[elem_0]
        dcoeff_e = np.mean(dcoeff[elem_0], axis=1)
        nref_e = nref[elem_0] if hasattr(nref, "__len__") and len(nref) == nn else nref

        for k, (i, j) in enumerate(pairs):
            rows_list.append(elem_0[:, i])
            cols_list.append(elem_0[:, j])
            val = (
                deldotdel_mat[:, offdiag_idx[k]] * dcoeff_e + (mua_e @ w1[:, k]) * evol
            )
            if omega > 0:
                if hasattr(nref_e, "__len__"):
                    val = (
                        val.astype(complex)
                        + 1j * omega * R_C0 * (nref_e @ w1[:, k]) * evol
                    )
                else:
                    val = val.astype(complex) + 1j * omega * R_C0 * nref_e * 0.05 * evol
            vals_list.append(val)

            rows_list.append(elem_0[:, j])
            cols_list.append(elem_0[:, i])
            vals_list.append(val)

        diag_idx = [0, 4, 7, 9]
        for k in range(4):
            rows_list.append(elem_0[:, k])
            cols_list.append(elem_0[:, k])
            val = deldotdel_mat[:, diag_idx[k]] * dcoeff_e + (mua_e @ w2[:, k]) * evol
            if omega > 0:
                if hasattr(nref_e, "__len__"):
                    val = (
                        val.astype(complex)
                        + 1j * omega * R_C0 * (nref_e @ w2[:, k]) * evol
                    )
                else:
                    val = val.astype(complex) + 1j * omega * R_C0 * nref_e * 0.10 * evol
            vals_list.append(val)

    # Boundary condition (original algorithm)
    bc_coeff = (1 - Reff) / (12.0 * (1 + Reff))
    Adiagbc = area * bc_coeff
    Aoffdbc = Adiagbc * 0.5

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        rows_list.append(face_0[:, i])
        cols_list.append(face_0[:, j])
        vals_list.append(Aoffdbc)
        rows_list.append(face_0[:, j])
        cols_list.append(face_0[:, i])
        vals_list.append(Aoffdbc)

    for k in range(3):
        rows_list.append(face_0[:, k])
        cols_list.append(face_0[:, k])
        vals_list.append(Adiagbc)

    # Concatenate all arrays at once (faster than repeated extend)
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    dtype = complex if omega > 0 else float
    Amat = sparse.coo_matrix((vals, (rows, cols)), shape=(nn, nn), dtype=dtype).tocsr()

    return Amat


def femrhs(
    cfg: dict, sd: dict = None, wv: str = "", md: int = 1
) -> Tuple[sparse.spmatrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create right-hand-side vectors for FEM system.
    """
    from . import utility

    optsrc, optdet, widesrc, widedet = utility.getoptodes(cfg, wv)

    srcnum = optsrc.shape[0] if optsrc is not None and len(optsrc) > 0 else 0
    detnum = optdet.shape[0] if optdet is not None and len(optdet) > 0 else 0
    wfsrcnum = widesrc.shape[0] if widesrc is not None and len(widesrc) > 0 else 0
    wfdetnum = widedet.shape[0] if widedet is not None and len(widedet) > 0 else 0

    nn = cfg["node"].shape[0]
    total_cols = srcnum + wfsrcnum + detnum + wfdetnum
    rhs = sparse.lil_matrix((nn, total_cols))

    loc = np.full(total_cols, np.nan)
    bary = np.full((total_cols, 4), np.nan)

    elem_0 = cfg["elem"][:, :4].astype(np.int32) - 1

    # Precompute bounding boxes for acceleration
    if srcnum > 0 or detnum > 0:
        tet_nodes = cfg["node"][elem_0, :3]
        bbox_min = np.min(tet_nodes, axis=1) - 1e-6
        bbox_max = np.max(tet_nodes, axis=1) + 1e-6
    else:
        bbox_min = bbox_max = None

    # Process point sources
    if srcnum > 0:
        locsrc, barysrc = _tsearchn(cfg["node"], elem_0, optsrc, bbox_min, bbox_max)

        for i in range(srcnum):
            if not np.isnan(locsrc[i]):
                eid = int(locsrc[i])
                rhs[elem_0[eid, :], i] = barysrc[i, :]

        loc[:srcnum] = locsrc
        bary[:srcnum, :] = barysrc

    # Process widefield sources
    if wfsrcnum > 0:
        rhs[:, srcnum : srcnum + wfsrcnum] = widesrc.T

    # Process point detectors
    if detnum > 0:
        locdet, barydet = _tsearchn(cfg["node"], elem_0, optdet, bbox_min, bbox_max)

        offset = srcnum + wfsrcnum
        for i in range(detnum):
            if not np.isnan(locdet[i]):
                eid = int(locdet[i])
                rhs[elem_0[eid, :], offset + i] = barydet[i, :]

        loc[offset : offset + detnum] = locdet
        bary[offset : offset + detnum, :] = barydet

    # Process widefield detectors
    if wfdetnum > 0:
        offset = srcnum + wfsrcnum + detnum
        rhs[:, offset : offset + wfdetnum] = widedet.T

    # Combine optode positions
    if srcnum > 0 and detnum > 0:
        optode = np.vstack([optsrc, optdet])
    elif srcnum > 0:
        optode = optsrc
    elif detnum > 0:
        optode = optdet
    else:
        optode = np.array([])

    return rhs.tocsr(), loc, bary, optode


def _tsearchn(
    nodes: np.ndarray,
    elem_0: np.ndarray,
    points: np.ndarray,
    bbox_min: np.ndarray = None,
    bbox_max: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find enclosing tetrahedral element for each point.
    Uses bounding box pre-filtering for acceleration.
    """
    npts = points.shape[0]
    loc = np.full(npts, np.nan)
    bary = np.full((npts, 4), np.nan)

    # Compute bounding boxes if not provided
    if bbox_min is None or bbox_max is None:
        tet_nodes = nodes[elem_0, :3]
        bbox_min = np.min(tet_nodes, axis=1) - 1e-6
        bbox_max = np.max(tet_nodes, axis=1) + 1e-6

    for i in range(npts):
        pt = points[i, :3]

        # Quick bounding box test to filter candidates
        in_bbox = np.all((pt >= bbox_min) & (pt <= bbox_max), axis=1)
        candidates = np.where(in_bbox)[0]

        for e in candidates:
            n = nodes[elem_0[e, :], :3]
            b = _compute_bary(n, pt)
            if b is not None and np.all(b >= -1e-10) and np.all(b <= 1 + 1e-10):
                loc[i] = e
                bary[i, :] = b
                break

    return loc, bary


def _compute_bary(tet_nodes: np.ndarray, point: np.ndarray) -> Optional[np.ndarray]:
    """Compute barycentric coordinates of point in tetrahedron."""
    T = np.column_stack(
        [
            tet_nodes[0] - tet_nodes[3],
            tet_nodes[1] - tet_nodes[3],
            tet_nodes[2] - tet_nodes[3],
        ]
    )
    try:
        b = np.linalg.solve(T, point - tet_nodes[3])
        return np.append(b, 1 - np.sum(b))
    except np.linalg.LinAlgError:
        return None


def femsolve(
    Amat: sparse.spmatrix,
    rhs: Union[np.ndarray, sparse.spmatrix],
    method: str = "direct",
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """
    Solve FEM linear system A*x = b.
    Uses LU factorization for multiple RHS columns (major speedup).
    """
    if sparse.issparse(rhs):
        rhs = rhs.toarray()

    if rhs.ndim == 1:
        rhs = rhs.reshape(-1, 1)

    ncol = rhs.shape[1]
    dtype = complex if np.iscomplexobj(Amat) or np.iscomplexobj(rhs) else float
    x = np.zeros((Amat.shape[0], ncol), dtype=dtype)
    flag = 0

    tol = kwargs.get("tol", 1e-10)
    maxiter = kwargs.get("maxiter", 1000)

    if method == "direct" or method == "mldivide":
        # Use LU factorization for efficiency with multiple RHS
        if ncol > 1:
            try:
                lu = splu(Amat.tocsc())
                for i in range(ncol):
                    if np.any(rhs[:, i] != 0):
                        x[:, i] = lu.solve(rhs[:, i])
            except Exception:
                # Fallback to spsolve if LU fails
                for i in range(ncol):
                    if np.any(rhs[:, i] != 0):
                        x[:, i] = spsolve(Amat, rhs[:, i])
        else:
            for i in range(ncol):
                if np.any(rhs[:, i] != 0):
                    x[:, i] = spsolve(Amat, rhs[:, i])
    elif method == "cg":
        for i in range(ncol):
            if np.any(rhs[:, i] != 0):
                try:
                    x[:, i], info = cg(Amat, rhs[:, i], rtol=tol, maxiter=maxiter)
                except TypeError:
                    x[:, i], info = cg(Amat, rhs[:, i], tol=tol, maxiter=maxiter)
                flag = max(flag, info)
    elif method == "gmres":
        for i in range(ncol):
            if np.any(rhs[:, i] != 0):
                try:
                    x[:, i], info = gmres(Amat, rhs[:, i], rtol=tol, maxiter=maxiter)
                except TypeError:
                    x[:, i], info = gmres(Amat, rhs[:, i], tol=tol, maxiter=maxiter)
                flag = max(flag, info)
    elif method == "bicgstab":
        for i in range(ncol):
            if np.any(rhs[:, i] != 0):
                try:
                    x[:, i], info = bicgstab(Amat, rhs[:, i], rtol=tol, maxiter=maxiter)
                except TypeError:
                    x[:, i], info = bicgstab(Amat, rhs[:, i], tol=tol, maxiter=maxiter)
                flag = max(flag, info)
    else:
        raise ValueError(f"Unknown solver method: {method}")

    return x, flag


def femgetdet(
    phi: np.ndarray, cfg: dict, loc: np.ndarray, bary: np.ndarray
) -> np.ndarray:
    """
    Extract detector measurements from forward solution.
    """
    srcnum = (
        cfg["srcpos"].shape[0] if "srcpos" in cfg and cfg["srcpos"] is not None else 0
    )
    detnum = (
        cfg["detpos"].shape[0] if "detpos" in cfg and cfg["detpos"] is not None else 0
    )
    widesrcnum = cfg.get("widesrc", np.array([])).shape[0] if "widesrc" in cfg else 0
    widedetnum = cfg.get("widedet", np.array([])).shape[0] if "widedet" in cfg else 0

    if srcnum + widesrcnum == 0 or detnum + widedetnum == 0:
        return np.array([])

    elem_0 = cfg["elem"][:, :4].astype(np.int32) - 1

    det_offset = srcnum + widesrcnum
    detval = np.zeros((detnum, srcnum), dtype=phi.dtype)

    for i in range(detnum):
        det_idx = det_offset + i
        if not np.isnan(loc[det_idx]):
            eid = int(loc[det_idx])
            node_ids = elem_0[eid, :]
            # Vectorized over sources using matrix multiply
            detval[i, :] = bary[det_idx, :] @ phi[node_ids, :srcnum]

    return detval


def jac(
    sd: np.ndarray,
    phi: np.ndarray,
    deldotdel_mat: np.ndarray,
    elem: np.ndarray,
    evol: np.ndarray,
    iselem: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Jacobian matrices using adjoint method.
    Optimized but preserving original algorithm.
    """
    elem_0 = elem[:, :4].astype(np.int32) - 1
    nelem = elem_0.shape[0]
    nn = phi.shape[0]

    # Get active source-detector pairs
    if sd.shape[1] >= 3:
        active = sd[:, 2] == 1
        sd_active = sd[active, :2].astype(np.int32)
    else:
        sd_active = sd[:, :2].astype(np.int32)

    nsd = sd_active.shape[0]
    src_cols = sd_active[:, 0]
    det_cols = sd_active[:, 1]

    Jmua_node = np.zeros((nsd, nn), dtype=phi.dtype)
    Jmua_elem = np.zeros((nsd, nelem), dtype=phi.dtype)

    # Process elements in batches for memory efficiency
    batch_size = min(1000, nelem)

    for batch_start in range(0, nelem, batch_size):
        batch_end = min(batch_start + batch_size, nelem)
        batch_elems = range(batch_start, batch_end)
        batch_elem_0 = elem_0[batch_start:batch_end, :]
        batch_evol = evol[batch_start:batch_end]

        # Get phi at batch element nodes: (batch_size, 4, ncols)
        phi_at_nodes = phi[batch_elem_0, :]

        # Extract for sources and detectors: (batch_size, 4, nsd)
        phi_src = phi_at_nodes[:, :, src_cols]
        phi_det = phi_at_nodes[:, :, det_cols]

        # Diagonal: sum of phi_src * phi_det over 4 nodes -> (batch_size, nsd)
        phidotphi_diag = np.sum(phi_src * phi_det, axis=1)

        # Off-diagonal terms
        phidotphi_offdiag = (
            phi_src[:, 0, :] * phi_det[:, 1, :]
            + phi_src[:, 1, :] * phi_det[:, 0, :]
            + phi_src[:, 0, :] * phi_det[:, 2, :]
            + phi_src[:, 2, :] * phi_det[:, 0, :]
            + phi_src[:, 0, :] * phi_det[:, 3, :]
            + phi_src[:, 3, :] * phi_det[:, 0, :]
            + phi_src[:, 1, :] * phi_det[:, 2, :]
            + phi_src[:, 2, :] * phi_det[:, 1, :]
            + phi_src[:, 1, :] * phi_det[:, 3, :]
            + phi_src[:, 3, :] * phi_det[:, 1, :]
            + phi_src[:, 2, :] * phi_det[:, 3, :]
            + phi_src[:, 3, :] * phi_det[:, 2, :]
        )

        # Element Jacobian: (batch_size, nsd)
        batch_Jmua_elem = (
            -(phidotphi_diag * 0.1 + phidotphi_offdiag * 0.025)
            * batch_evol[:, np.newaxis]
        )

        # Store in Jmua_elem (transposed)
        Jmua_elem[:, batch_start:batch_end] = batch_Jmua_elem.T

        # Accumulate to nodes
        for j in range(4):
            np.add.at(Jmua_node.T, batch_elem_0[:, j], batch_Jmua_elem)

    Jmua_node *= 0.25

    if iselem:
        return Jmua_elem, Jmua_elem
    return Jmua_node, Jmua_elem


def jacchrome(Jmua: dict, chromophores: List[str]) -> dict:
    """Build Jacobian matrices for chromophores from mua Jacobian."""
    from . import property as prop_module

    if not isinstance(Jmua, dict):
        raise ValueError("Jmua must be a dict with wavelength keys")

    wavelengths = list(Jmua.keys())
    extin, _ = prop_module.extinction(wavelengths, chromophores)

    Jchrome = {}
    for i, ch in enumerate(chromophores):
        Jch = None
        for j, wv in enumerate(wavelengths):
            weighted = Jmua[wv] * extin[j, i]
            Jch = weighted if Jch is None else np.vstack([Jch, weighted])
        Jchrome[ch] = Jch

    return Jchrome
