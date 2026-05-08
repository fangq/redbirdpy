"""
Redbird Forward Module - FEM-based forward modeling for diffuse optics.

INDEX CONVENTION: All mesh indices (elem, face) stored in cfg are 1-based
to match MATLAB/iso2mesh. This module converts to 0-based internally when
indexing numpy arrays, using local variables named with '_0' suffix.

Functions:
    runforward: Main forward solver for all sources/wavelengths
    femlhs: Build FEM left-hand-side (stiffness) matrix
    femrhs: Build FEM right-hand-side vector
    femgetdet: Extract detector values from forward solution
    jac: Compute Jacobian matrices using adjoint method
"""

__all__ = [
    "runforward",
    "femlhs",
    "femrhs",
    "femgetdet",
    "jac",
    "jacchrome",
    "jacepssigma",
    "C0",
]

import numpy as np
from scipy import sparse
from typing import Dict, Tuple, Optional, Union, List, Any

# Import solver functions from solver module
from .solver import femsolve
from .utility import sdmap, getoptodes, deldotdel
from .property import extinction

# Speed of light in mm/s
C0 = 299792458000.0
R_C0 = 1.0 / C0


def runforward(cfg: dict, **kwargs) -> Tuple[Any, Any]:
    """
    Perform forward simulations at all sources and all wavelengths.
    """
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
        sd = sdmap(cfg)
    if not isinstance(sd, dict):
        sd = {wv: sd for wv in wavelengths}

    Amat = {}
    detval_out = {md: {"detphi": {}} for md in rfcw}
    phi_out = {md: {"phi": {}} for md in rfcw}

    for wv in wavelengths:
        for md in rfcw:
            rhs, loc, bary, optode = femrhs(cfg, sd, wv, md)
            Amat[wv] = femlhs(cfg, cfg["deldotdel"], wv, md)
            phi_sol, flag = femsolve(Amat[wv], rhs, **kwargs)
            phi_out[md]["phi"][wv] = phi_sol

            # Pass rhs to femgetdet for wide-field detection
            detval = femgetdet(phi_sol, cfg, rhs, loc, bary)
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


def femlhs(
    cfg: dict, deldotdel_mat: np.ndarray, wavelength: str = "", mode: int = 1
) -> sparse.csr_matrix:
    """
    Create FEM stiffness matrix.

    Builds the FEM left-hand-side (system matrix) for either
    DOT (diffusion equation) or MWT (scalar Helmholtz equation).
    MWT mode is engaged when cfg.bulk.epsilon or cfg.bulk.sigma is
    present (defining the bulk medium for the first-order
    Bayliss-Turkel radiation boundary condition).

    For both PDEs, the per-element volume integral has the form
        A_e = avol * <grad phi_i, grad phi_j>_e + (breal + 1j*bimag) * <phi_i, phi_j>_e
    DOT: avol = D, breal = mua, bimag = omega/c * n
    MWT: avol = 1, breal = -omega^2 * mu * eps0 * eps_r, bimag = +omega * mu * sigma
    """
    nn = cfg["node"].shape[0]
    ne = cfg["elem"].shape[0]
    evol = cfg["evol"]
    area = cfg["area"]

    # Convert 1-based to 0-based
    elem_0 = cfg["elem"][:, :4].astype(np.int32) - 1
    face_0 = cfg["face"].astype(np.int32) - 1

    # MWT detection: cfg.bulk.epsilon or cfg.bulk.sigma defines the bulk medium
    # for the radiation boundary condition.
    ishelmholtz = isinstance(cfg.get("bulk"), dict) and (
        "epsilon" in cfg["bulk"] or "sigma" in cfg["bulk"]
    )

    # Get properties for current wavelength
    if isinstance(cfg.get("prop"), dict) and wavelength:
        props = cfg["prop"][wavelength]
        omega = (
            cfg["omega"].get(wavelength, 0)
            if isinstance(cfg.get("omega"), dict)
            else cfg.get("omega", 0)
        )
        if not ishelmholtz:
            reff = (
                cfg["reff"][wavelength]
                if isinstance(cfg.get("reff"), dict)
                else cfg["reff"]
            )
    else:
        props = cfg["prop"]
        omega = cfg.get("omega", 0)
        if not ishelmholtz:
            reff = cfg.get("reff", 0.493)

    if mode == 2:
        omega = 0

    seg = cfg.get("seg", None)

    # Extract material properties and compute (avol, breal, bimag) per node or per elem
    if ishelmholtz:
        EPS0_MM = 8.854187817e-15  # F/mm
        if props.shape[0] == nn or props.shape[0] == ne:
            eps_r = props[:, 0]
            sigma = props[:, 1]
            permea = props[:, 2] if props.shape[1] >= 3 else 4 * np.pi * 1e-10
            if np.isscalar(permea):
                permea = np.full_like(eps_r, permea)
        elif seg is not None:
            seg_idx = np.clip(seg.astype(np.int32), 0, props.shape[0] - 1)
            eps_r = props[seg_idx, 0]
            sigma = props[seg_idx, 1]
            permea = props[seg_idx, 2] if props.shape[1] >= 3 else np.full_like(eps_r, 4 * np.pi * 1e-10)
        else:
            raise ValueError("Property format not recognized")
        avol = np.ones_like(eps_r)
        breal = -(omega ** 2) * permea * EPS0_MM * eps_r
        bimag = omega * permea * sigma
        is_complex = True
    else:
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
        if np.isscalar(nref):
            nref = np.full_like(mua, nref)
        avol = 1.0 / (3.0 * (mua + musp))
        breal = mua
        bimag = omega * R_C0 * nref
        is_complex = omega > 0

    # Pre-allocate lists (faster than repeated extend)
    rows_list = []
    cols_list = []
    vals_list = []

    offdiag_idx = [1, 2, 3, 5, 6, 8]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Volume assembly: dispatches on per-element vs per-node properties.
    # Coefficient (avol, breal, bimag) is what differs between DOT and MWT.
    if len(breal) == ne:
        for k, (i, j) in enumerate(pairs):
            rows_list.append(elem_0[:, i])
            cols_list.append(elem_0[:, j])
            val = deldotdel_mat[:, offdiag_idx[k]] * avol + 0.05 * breal * evol
            if is_complex:
                val = val.astype(complex) + 1j * 0.05 * bimag * evol
            vals_list.append(val)

            rows_list.append(elem_0[:, j])
            cols_list.append(elem_0[:, i])
            vals_list.append(val)

        diag_idx = [0, 4, 7, 9]
        for k in range(4):
            rows_list.append(elem_0[:, k])
            cols_list.append(elem_0[:, k])
            val = deldotdel_mat[:, diag_idx[k]] * avol + 0.10 * breal * evol
            if is_complex:
                val = val.astype(complex) + 1j * 0.10 * bimag * evol
            vals_list.append(val)
    else:
        # Node-based properties: use consistent-mass weights w1 (off-diag) / w2 (diag)
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

        breal_e = breal[elem_0]
        bimag_e = bimag[elem_0] if hasattr(bimag, "__len__") else np.full(elem_0.shape, bimag)
        avol_e = np.mean(avol[elem_0], axis=1) if hasattr(avol, "__len__") else avol

        for k, (i, j) in enumerate(pairs):
            rows_list.append(elem_0[:, i])
            cols_list.append(elem_0[:, j])
            val = deldotdel_mat[:, offdiag_idx[k]] * avol_e + (breal_e @ w1[:, k]) * evol
            if is_complex:
                val = val.astype(complex) + 1j * (bimag_e @ w1[:, k]) * evol
            vals_list.append(val)

            rows_list.append(elem_0[:, j])
            cols_list.append(elem_0[:, i])
            vals_list.append(val)

        diag_idx = [0, 4, 7, 9]
        for k in range(4):
            rows_list.append(elem_0[:, k])
            cols_list.append(elem_0[:, k])
            val = deldotdel_mat[:, diag_idx[k]] * avol_e + (breal_e @ w2[:, k]) * evol
            if is_complex:
                val = val.astype(complex) + 1j * (bimag_e @ w2[:, k]) * evol
            vals_list.append(val)

    # Boundary condition: Robin (DOT) or first-order Bayliss-Turkel (MWT)
    if ishelmholtz:
        from .property import getbulk
        bk = getbulk(cfg)
        if isinstance(bk, dict):
            bk = bk[wavelength] if wavelength in bk else list(bk.values())[0]
        EPS0_MM = 8.854187817e-15
        k2bg = (omega ** 2) * bk[2] * EPS0_MM * bk[0] - 1j * omega * bk[2] * bk[1]
        kbg = np.sqrt(k2bg)
        rvec = cfg["facecenter"] - cfg["rbcorigin"][np.newaxis, :]
        rdotn = np.sum(rvec * cfg["facenormal"], axis=1) / cfg["facer"]
        bccoef = (1j * kbg - 1.0 / (2.0 * cfg["facer"])) * rdotn
        Adiagbc = (area / 6.0) * bccoef
        Aoffdbc = Adiagbc * 0.5
        is_complex = True
    else:
        Reff = reff
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

    dtype = complex if is_complex else float
    Amat = sparse.coo_matrix((vals, (rows, cols)), shape=(nn, nn), dtype=dtype).tocsr()

    return Amat


def femrhs(
    cfg: dict, sd: dict = None, wv: str = "", md: int = 1
) -> Tuple[sparse.spmatrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create right-hand-side vectors for FEM system.

    Returns
    -------
    rhs : sparse matrix (Nn x Ncols)
        RHS vectors. Column order: [point_src, wide_src, point_det, wide_det]
    loc : ndarray
        Element IDs enclosing each optode (1-based, NaN for wide-field)
    bary : ndarray
        Barycentric coordinates for point optodes
    optode : ndarray
        Combined optode positions
    """
    import iso2mesh as i2m

    optsrc, optdet, widesrc, widedet = getoptodes(cfg, wv)

    # Get counts
    srcnum = optsrc.shape[0] if optsrc is not None and optsrc.size > 0 else 0
    detnum = optdet.shape[0] if optdet is not None and optdet.size > 0 else 0

    # widesrc/widedet are stored as (Nn x Npattern) in cfg
    # But internally we work with (Npattern x Nn) for easier indexing
    wfsrcnum = widesrc.shape[1] if widesrc is not None and widesrc.size > 0 else 0
    wfdetnum = widedet.shape[1] if widedet is not None and widedet.size > 0 else 0

    nn = cfg["node"].shape[0]
    total_cols = srcnum + wfsrcnum + detnum + wfdetnum

    if total_cols == 0:
        return (
            sparse.csr_matrix((nn, 0)),
            np.array([]),
            np.array([]).reshape(0, 4),
            np.array([]),
        )

    # detect complex-valued widesrc/widedet (e.g. MWT line sources scaled by -j*omega*mu0)
    is_complex_rhs = (widesrc is not None and np.iscomplexobj(widesrc)) or (
        widedet is not None and np.iscomplexobj(widedet)
    )
    rhs = sparse.lil_matrix((nn, total_cols), dtype=complex if is_complex_rhs else float)

    # Initialize loc and bary for ALL optodes (including wide-field as NaN)
    total_optodes = srcnum + wfsrcnum + detnum + wfdetnum
    loc = np.full(total_optodes, np.nan)
    bary = np.full((total_optodes, 4), np.nan)

    # elem is 1-based, tsearchn expects 1-based and returns 1-based
    elem = cfg["elem"][:, :4].astype(np.int32)
    elem_0 = elem - 1  # 0-based for indexing

    col_idx = 0

    # Process point sources using iso2mesh.tsearchn
    if srcnum > 0:
        optsrc = np.atleast_2d(optsrc)
        locsrc, barysrc = i2m.tsearchn(cfg["node"], elem, optsrc[:, :3])

        for i in range(srcnum):
            if not np.isnan(locsrc[i]):
                eid = int(locsrc[i]) - 1  # Convert to 0-based
                rhs[elem_0[eid, :], col_idx + i] = barysrc[i, :]

        # Store in loc/bary (keep 1-based for loc)
        loc[:srcnum] = locsrc
        bary[:srcnum, :] = barysrc
        col_idx += srcnum

    # Process widefield sources - widesrc is (Nn x wfsrcnum)
    if wfsrcnum > 0:
        rhs[:, col_idx : col_idx + wfsrcnum] = widesrc
        # loc/bary already NaN for wide-field indices
        col_idx += wfsrcnum

    # Process point detectors using iso2mesh.tsearchn
    if detnum > 0:
        optdet = np.atleast_2d(optdet)
        locdet, barydet = i2m.tsearchn(cfg["node"], elem, optdet[:, :3])

        for i in range(detnum):
            if not np.isnan(locdet[i]):
                eid = int(locdet[i]) - 1  # Convert to 0-based
                rhs[elem_0[eid, :], col_idx + i] = barydet[i, :]

        # Store in loc/bary
        det_start = srcnum + wfsrcnum
        loc[det_start : det_start + detnum] = locdet
        bary[det_start : det_start + detnum, :] = barydet
        col_idx += detnum

    # Process widefield detectors - widedet is (Nn x wfdetnum)
    if wfdetnum > 0:
        rhs[:, col_idx : col_idx + wfdetnum] = widedet

    # Combine optode positions
    optode_list = []
    if srcnum > 0:
        optode_list.append(optsrc)
    if detnum > 0:
        optode_list.append(optdet)
    optode = np.vstack(optode_list) if optode_list else np.array([])

    return rhs.tocsr(), loc, bary, optode


def femgetdet(
    phi: np.ndarray,
    cfg: dict,
    rhs: np.ndarray,
    loc: np.ndarray = None,
    bary: np.ndarray = None,
) -> np.ndarray:
    """
    Extract detector measurements from forward solution.

    Parameters
    ----------
    phi : ndarray
        Forward solution (nn x nsrc_total)
    cfg : dict
        Configuration with srcpos, detpos, widesrc, widedet, etc.
    rhs : ndarray or sparse matrix
        RHS matrix from femrhs (nn x total_cols)
    loc : ndarray, optional
        Element indices for point optodes (1-based)
    bary : ndarray, optional
        Barycentric coordinates for point optodes

    Returns
    -------
    detval : ndarray
        Detector values (ndet x nsrc)
    """
    # Get source/detector counts
    srcnum = 0
    if "srcpos" in cfg and cfg["srcpos"] is not None:
        srcpos = np.atleast_2d(cfg["srcpos"])
        if srcpos.size > 0:
            srcnum = srcpos.shape[0]

    detnum = 0
    if "detpos" in cfg and cfg["detpos"] is not None:
        detpos = np.atleast_2d(cfg["detpos"])
        if detpos.size > 0:
            detnum = detpos.shape[0]

    wfsrcnum = 0
    if "widesrc" in cfg and cfg["widesrc"] is not None and cfg["widesrc"].size > 0:
        wfsrcnum = cfg["widesrc"].shape[1]  # (Nn x Npattern)

    wfdetnum = 0
    if "widedet" in cfg and cfg["widedet"] is not None and cfg["widedet"].size > 0:
        wfdetnum = cfg["widedet"].shape[1]  # (Nn x Npattern)

    total_src = srcnum + wfsrcnum
    total_det = detnum + wfdetnum

    if total_src == 0 or total_det == 0:
        return np.array([])

    # Column indices in rhs/phi:
    # [0:srcnum] = point sources
    # [srcnum:srcnum+wfsrcnum] = wide sources
    # [srcnum+wfsrcnum:srcnum+wfsrcnum+detnum] = point detectors
    # [srcnum+wfsrcnum+detnum:end] = wide detectors

    det_col_start = srcnum + wfsrcnum
    det_col_end = det_col_start + total_det

    # Extract detector RHS columns
    if sparse.issparse(rhs):
        rhs_det = rhs[:, det_col_start:det_col_end].toarray()
    else:
        rhs_det = rhs[:, det_col_start:det_col_end]

    # Extract source phi columns
    phi_src = phi[:, :total_src]

    # Compute detector values using adjoint: detval = rhs_det^T @ phi_src
    # Result shape: (total_det x total_src)
    detval = rhs_det.T @ phi_src

    return detval


try:
    from numba import njit, prange

    HAS_NUMBA = True
    print("Using Numba for Jacobian acceleration")
except ImportError:
    HAS_NUMBA = False
    print("Numba not available")

if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _jac_core(phi, elem_0, evol, src_cols, det_cols):
        """Numba-accelerated Jacobian core computation."""
        nelem = elem_0.shape[0]
        nsd = len(src_cols)
        Jmua_elem = np.zeros((nsd, nelem))

        for isd in prange(nsd):
            src_col = src_cols[isd]
            det_col = det_cols[isd]

            for ie in range(nelem):
                n0, n1, n2, n3 = (
                    elem_0[ie, 0],
                    elem_0[ie, 1],
                    elem_0[ie, 2],
                    elem_0[ie, 3],
                )

                ps0, ps1, ps2, ps3 = (
                    phi[n0, src_col],
                    phi[n1, src_col],
                    phi[n2, src_col],
                    phi[n3, src_col],
                )
                pd0, pd1, pd2, pd3 = (
                    phi[n0, det_col],
                    phi[n1, det_col],
                    phi[n2, det_col],
                    phi[n3, det_col],
                )

                diag_sum = ps0 * pd0 + ps1 * pd1 + ps2 * pd2 + ps3 * pd3
                cross_sum = (
                    ps0 * pd1
                    + ps1 * pd0
                    + ps0 * pd2
                    + ps2 * pd0
                    + ps0 * pd3
                    + ps3 * pd0
                    + ps1 * pd2
                    + ps2 * pd1
                    + ps1 * pd3
                    + ps3 * pd1
                    + ps2 * pd3
                    + ps3 * pd2
                )

                Jmua_elem[isd, ie] = -(diag_sum + cross_sum * 0.5) * 0.1 * evol[ie]

        return Jmua_elem


def jac(sd, phi, deldotdel_mat, elem, evol, iselem=False):
    """Build Jacobian matrices - Numba accelerated if available."""
    elem_0 = elem[:, :4].astype(np.int32) - 1
    nelem = elem_0.shape[0]
    nn = phi.shape[0]

    if sd.shape[1] >= 3:
        active = sd[:, 2] == 1
        sd_active = sd[active, :2].astype(np.int32)
    else:
        sd_active = sd[:, :2].astype(np.int32)

    nsd = sd_active.shape[0]
    src_cols = sd_active[:, 0]
    det_cols = sd_active[:, 1]

    if HAS_NUMBA:
        # Use Numba-accelerated version
        Jmua_elem = _jac_core(
            np.ascontiguousarray(phi), elem_0, evol, src_cols, det_cols
        )
    else:
        # Fallback to numpy loop
        Jmua_elem = np.zeros((nsd, nelem), dtype=phi.dtype)
        evol_scaled = 0.1 * evol

        for isd in range(nsd):
            src_col = src_cols[isd]
            det_col = det_cols[isd]

            phi_src = phi[elem_0, src_col]
            phi_det = phi[elem_0, det_col]

            diag_sum = (phi_src * phi_det).sum(axis=1)
            cross_sum = (
                phi_src[:, 0] * phi_det[:, 1]
                + phi_src[:, 1] * phi_det[:, 0]
                + phi_src[:, 0] * phi_det[:, 2]
                + phi_src[:, 2] * phi_det[:, 0]
                + phi_src[:, 0] * phi_det[:, 3]
                + phi_src[:, 3] * phi_det[:, 0]
                + phi_src[:, 1] * phi_det[:, 2]
                + phi_src[:, 2] * phi_det[:, 1]
                + phi_src[:, 1] * phi_det[:, 3]
                + phi_src[:, 3] * phi_det[:, 1]
                + phi_src[:, 2] * phi_det[:, 3]
                + phi_src[:, 3] * phi_det[:, 2]
            )
            Jmua_elem[isd, :] = -(diag_sum + cross_sum * 0.5) * evol_scaled

    # Accumulate to nodes using sparse matrix
    from scipy import sparse

    rows = elem_0.ravel()
    cols = np.repeat(np.arange(nelem), 4)
    data = np.full(nelem * 4, 0.25)
    P = sparse.csr_matrix((data, (rows, cols)), shape=(nn, nelem))

    Jmua_node = (P @ Jmua_elem.T).T

    return Jmua_node, Jmua_elem


def jacchrome(Jmua: dict, chromophores: List[str]) -> dict:
    """Build Jacobian matrices for chromophores from mua Jacobian."""

    if not isinstance(Jmua, dict):
        raise ValueError("Jmua must be a dict with wavelength keys")

    wavelengths = list(Jmua.keys())
    extin, _ = extinction(wavelengths, chromophores)

    Jchrome = {}
    for i, ch in enumerate(chromophores):
        Jch = None
        for j, wv in enumerate(wavelengths):
            weighted = Jmua[wv] * extin[j, i]
            Jch = weighted if Jch is None else np.vstack([Jch, weighted])
        Jchrome[ch] = Jch

    return Jchrome


def jacepssigma(Jmua: dict, omegas, has_eps: bool = True, has_sigma: bool = True) -> dict:
    """
    Build Jacobian matrices for MWT permittivity (eps_r) and conductivity (sigma)
    from the absorption-Jacobian kernel.

    The DOT-style jac() returns Jmua = -<E_s, E_r>_M (per-element mass kernel).
    For Helmholtz (A has -k^2*M), the chain rule gives:
        Jk^2   = -Jmua_returned
        J_eps  = (omega^2 * mu0 * eps0) * Jk^2 = -(omega^2 * mu0 * eps0) * Jmua
        J_sig  = (-j * omega * mu0)     * Jk^2 = +(j * omega * mu0)     * Jmua

    Per-frequency Jacobians are stacked vertically (one block per frequency).

    Parameters
    ----------
    Jmua : dict
        Per-frequency mua Jacobians (dict keyed by frequency string)
    omegas : float or dict
        Angular frequency (rad/s); dict keyed by the same frequency strings
        as Jmua, or a scalar for single-frequency runs.
    has_eps, has_sigma : bool
        If True, include the corresponding parameter in the output.

    Returns
    -------
    Jchain : dict
        {'epsilon': Jeps, 'sigma': Jsigma} (only the requested keys).
    """
    if not isinstance(Jmua, dict):
        raise ValueError("Jmua must be a dict with frequency keys")

    EPS0_MM = 8.854187817e-15  # F/mm
    MU0_MM = 4.0 * np.pi * 1e-10  # H/mm

    wavelengths = list(Jmua.keys())
    Jeps = None
    Jsigma = None

    for wv in wavelengths:
        if isinstance(omegas, dict):
            omega = omegas[wv]
        else:
            omega = omegas
        we = -(omega ** 2) * MU0_MM * EPS0_MM
        ws = 1j * omega * MU0_MM
        if has_eps:
            block_e = Jmua[wv] * we
            Jeps = block_e if Jeps is None else np.vstack([Jeps, block_e])
        if has_sigma:
            block_s = Jmua[wv] * ws
            Jsigma = block_s if Jsigma is None else np.vstack([Jsigma, block_s])

    Jchain = {}
    if has_eps:
        Jchain["epsilon"] = Jeps
    if has_sigma:
        Jchain["sigma"] = Jsigma
    return Jchain
