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
    "runtd",
    "femlhs",
    "femrhs",
    "femgetdet",
    "jac",
    "jacchrome",
    "jacepssigma",
    "jacmuafast",
    "jacmus",
    "jacscatamp",
    "jacscatpow",
    "jacscat",
    "jacnode",
    "C0",
]

import copy
import warnings

import numpy as np
from scipy import sparse
from typing import Dict, Tuple, Optional, Union, List, Any

# Import solver functions from solver module
from .solver import femsolve
from .utility import sdmap, getoptodes, deldotdel, getdetdir
from .property import extinction

# Optional Monte Carlo backend (mmc); the MC branch in `runforward` falls back
# to the FEM solver with a warning when this isn't importable.
try:
    import pmmc as _pmmc

    _HAS_PMMC = True
except ImportError:
    _pmmc = None
    _HAS_PMMC = False

try:
    import pmcx as _pmcx

    _HAS_PMCX = True
except ImportError:
    _pmcx = None
    _HAS_PMCX = False

# Speed of light in mm/s
C0 = 299792458000.0
R_C0 = 1.0 / C0


def runforward(cfg: dict, **kwargs) -> Tuple[Any, ...]:
    """
    Perform forward simulations at all sources and all wavelengths.

    Three execution paths are selected automatically from the cfg fields:

    * **Monte Carlo (mmclab/pmmc)** -- ``cfg["nphoton"]`` is set. Requires
      ``cfg["node"] + cfg["elem"]`` (tetrahedral mesh; ``cfg["vol"]`` /
      mcxlab path is not yet supported here). MWT/Helmholtz is forbidden
      on this path. The optional kwarg ``return_jacobian=True`` requests
      the mesh-mode adjoint Jacobian as a third return; mmc auto-runs the
      adjoint kernel and returns ``Jext = {"mua": ..., "dcoeff": ...}``.
    * **Time-domain FEM (Crank-Nicolson)** -- when ``cfg.tstart``,
      ``cfg.tstep``, and ``cfg.tend`` are all defined and no nphoton.
      Delegated to `runtd` (3D arrays Nn x Nsrc x Nt).
    * **CW / FD FEM** -- default.

    Parameters
    ----------
    return_jacobian : bool, kwarg, default False
        When True, return a 3-tuple (detval, phi, Jext). Jext is the
        mesh-mode adjoint Jacobian on the MC path with shape
        (Nn, Ns*Nd) per wavelength (matching the pmmc/mmclab-native
        orientation); None on FEM/TD paths.

    Notes
    -----
    When ``cfg.nphoton`` is set but ``pmmc`` is not importable, this
    function emits a warning and falls back to the FEM solver.
    """
    return_jacobian = kwargs.pop("return_jacobian", False)

    # Monte Carlo dispatch (must precede time-domain check; mc handles its
    # own time grid).  Three sub-branches:
    #   - cfg.vol            -> mcxlab/pmcx voxel-grid path (_runforward_mcx)
    #   - cfg.node + cfg.elem -> mmclab/pmmc mesh-mode path (_runforward_mc)
    #   - neither             -> fall back to FEM with a warning
    if cfg.get("nphoton") is not None:
        if cfg.get("vol") is not None:
            return _runforward_mcx(cfg, return_jacobian=return_jacobian, **kwargs)
        if not _HAS_PMMC:
            warnings.warn(
                "cfg.nphoton is set but pmmc is not importable; falling back "
                "to the FEM forward. Install pmmc to enable Monte Carlo.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            return _runforward_mc(cfg, return_jacobian=return_jacobian, **kwargs)

    # Time-domain dispatch (Crank-Nicolson)
    if (
        cfg.get("tstart") is not None
        and cfg.get("tstep") is not None
        and cfg.get("tend") is not None
    ):
        td_result = runtd(cfg, **kwargs)
        if return_jacobian:
            return (*td_result, None)
        return td_result

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

    if return_jacobian:
        return detval_out, phi_out, None
    return detval_out, phi_out


def _runforward_mc(
    cfg: dict, return_jacobian: bool = False, **kwargs
) -> Tuple[Any, ...]:
    """Monte Carlo forward branch of ``runforward``.

    Port of the mmclab path in redbird-m/matlab/rbrunforward.m (lines
    98-280). Routes the cfg through ``pmmc`` once per wavelength,
    rebuilds detector readings via barycentric interpolation, and packs
    the mesh-mode adjoint Jacobian into the optional 3rd return when
    requested.
    """
    if cfg.get("helmholtz") or cfg.get("bulkprop") is not None:
        raise ValueError(
            "MWT/Helmholtz forward (cfg.helmholtz / cfg.bulkprop) cannot "
            "be combined with the Monte Carlo path. Remove cfg.nphoton."
        )

    if "node" not in cfg or "elem" not in cfg:
        raise ValueError(
            "MC forward requires cfg.node and cfg.elem (tetrahedral mesh)."
        )

    # work on a shallow copy so per-wavelength edits don't leak back to caller
    cfg = copy.copy(cfg)
    # ensure cfg.prop isn't shared with caller across our pop/restore
    if "prop" in cfg:
        cfg["prop"] = copy.copy(cfg["prop"])

    # default single-gate CW time grid
    if cfg.get("tstart") is None:
        cfg["tstart"] = 0.0
    if cfg.get("tend") is None:
        cfg["tend"] = 5e-9
    if cfg.get("tstep") is None:
        cfg["tstep"] = cfg["tend"]

    avgsize = kwargs.get("avgsize", 1.0)

    srcpos = np.atleast_2d(np.asarray(cfg["srcpos"], dtype=float))
    detpos = np.atleast_2d(np.asarray(cfg["detpos"], dtype=float))
    srcnum = srcpos.shape[0]
    detnum = detpos.shape[0]

    # ensure detector radius (column 3) is set for the disk-source adjoint
    if detpos.shape[1] == 3:
        radius_col = np.full((detnum, 1), avgsize)
        detpos = np.hstack([detpos, radius_col])
        cfg["detpos"] = detpos

    cfg["method"] = "elem"
    cfg["basisorder"] = 1
    if "seg" in cfg and "elemprop" not in cfg:
        # Per-element seg: use directly. Per-node seg (or any other size) would
        # corrupt mmclab's mesh.ne = numel(elemprop) heuristic, so fall back to
        # a single tissue label; the per-node DOT property mode below routes
        # per-node mua/musp via cfg.nodemua / cfg.nodemusp instead.
        seg = np.asarray(cfg["seg"]).ravel()
        n_elem = int(np.asarray(cfg["elem"]).shape[0])
        if seg.size == n_elem:
            cfg["elemprop"] = seg.astype(np.int32)
        else:
            cfg["elemprop"] = np.ones(n_elem, dtype=np.int32)

    need_jacobian = bool(return_jacobian)

    # CW (omega == 0) cannot separate mua and D from a single measurement set,
    # so request the single-output 'adjoint' kernel (J_mua only); RF (omega>0)
    # uses 'adjoint_mua_d' to also compute J_D. Mirrors the FEM/rbjac branch
    # which only builds Jd when any(omegas) > 0.
    omega_val = cfg.get("omega", 0)
    if isinstance(omega_val, dict):
        is_rf_mc = any(float(v) > 0 for v in omega_val.values())
    else:
        is_rf_mc = float(omega_val) > 0 if omega_val is not None else False

    if need_jacobian:
        # require cfg.detdir; build it if missing
        detdir = cfg.get("detdir")
        if detdir is None or len(np.asarray(detdir)) == 0:
            cfg["detdir"] = getdetdir(cfg)
        cfg["detdir"] = np.atleast_2d(np.asarray(cfg["detdir"], dtype=float))
        if cfg["detdir"].shape[1] < 4:
            pad = np.zeros((cfg["detdir"].shape[0], 4 - cfg["detdir"].shape[1]))
            cfg["detdir"] = np.hstack([cfg["detdir"], pad])
        cfg["srcid"] = -1
        cfg["outputtype"] = "adjoint_mua_d" if is_rf_mc else "adjoint"
    elif cfg.get("detdir") is not None and len(np.asarray(cfg["detdir"])) > 0:
        # forward-only but user wants detector slots too
        cfg["detdir"] = np.atleast_2d(np.asarray(cfg["detdir"], dtype=float))
        if cfg["detdir"].shape[1] < 4:
            pad = np.zeros((cfg["detdir"].shape[0], 4 - cfg["detdir"].shape[1]))
            cfg["detdir"] = np.hstack([cfg["detdir"], pad])
        cfg["srcid"] = -2
        cfg["outputtype"] = "fluence"
    else:
        cfg["outputtype"] = "fluence"

    # multi-wavelength dispatch (mmc/pmmc is single-wavelength per call)
    is_multi_wv = isinstance(cfg.get("prop"), dict)
    if is_multi_wv:
        wavelengths = list(cfg["prop"].keys())
        prop_all = cfg["prop"]
        phi_out: Any = {}
        detval_out: Any = {}
        Jmua_map: Dict[str, np.ndarray] = {}
        Jd_map: Dict[str, np.ndarray] = {}
    else:
        wavelengths = [""]
        prop_all = None

    # precompute (optode_loc, optode_bary) for detector interpolation
    optode_loc, optode_bary = _tsearchn_bary(
        np.asarray(cfg["node"], dtype=float),
        np.asarray(cfg["elem"], dtype=int)[:, :4],
        detpos[:, :3],
    )

    # broadcast srcdir to match srcnum rows so pmmc's multi-source parser is
    # happy (it requires matching row counts)
    srcdir = np.atleast_2d(np.asarray(cfg["srcdir"], dtype=float))
    if srcdir.shape[0] == 1 and srcnum > 1:
        srcdir = np.tile(srcdir, (srcnum, 1))
    cfg["srcdir"] = srcdir

    # mmclab.cpp populates srcdata via calloc (zero-initialized) then copies
    # arraydim[1] columns from cfg.srcpos. With only 3 columns the per-photon
    # weight srcdata.srcpos.w stays at 0 and every photon launches with weight
    # zero ("total simulated energy: 0.00"). Pad the 4th column with weight=1
    # so multi-source MC launches actually carry energy.
    if srcpos.shape[1] < 4:
        weight_col = np.ones((srcpos.shape[0], 4 - srcpos.shape[1]))
        srcpos = np.hstack([srcpos, weight_col])
    cfg["srcpos"] = srcpos

    for wv in wavelengths:
        if is_multi_wv:
            cfg["prop"] = prop_all[wv]

        prop_arr = np.asarray(cfg["prop"], dtype=float)
        n_node = int(np.asarray(cfg["node"]).shape[0])

        # Per-node optical-property mode for DOT reconstruction: when
        # cfg.prop has one row per forward-mesh node, route mua (and musp
        # for RF) into cfg.nodemua / cfg.nodemusp so mmc's per-node global-
        # memory path is engaged.
        if prop_arr.ndim == 2 and prop_arr.shape[0] == n_node:
            cfg["nodemua"] = prop_arr[:, 0].astype(np.float32, copy=True)
            cfg["isnodalmua"] = 1
            if cfg.get("omega", 0) > 0:
                cfg["nodemusp"] = prop_arr[:, 1].astype(np.float32, copy=True)
                cfg["isnodalmusp"] = 1
            bulk = np.mean(prop_arr, axis=0)
            if bulk[1] < 1e-3:
                bulk[1] = 1e-3
            cfg["prop"] = np.array(
                [[0.0, 0.0, 1.0, 1.0], [bulk[0], bulk[1], 0.0, bulk[3]]],
                dtype=float,
            )
            cfg["elemprop"] = np.ones(np.asarray(cfg["elem"]).shape[0], dtype=np.int32)

        # let pmmc's multi-source parser rebuild srcdata fresh from srcpos
        cfg.pop("srcdata", None)
        cfg.pop("extrasrclen", None)

        out = _pmmc.run(cfg)

        # mmc flux.data layouts:
        #   single source, single gate            : (nn,)
        #   single source, maxgate > 1            : (nn, maxgate)
        #   multi-source / detector-adjoint mode  : (nn, maxgate, Ns+Nd)
        # Normalize to (nn, Ns_total) by dropping the gate axis and ensuring
        # a second source-slot axis is always present.
        raw_phi = np.asarray(out["flux"])
        if raw_phi.ndim == 3:
            # (nn, maxgate, slots) - take gate 0 (CW)
            phi_wv = raw_phi[:, 0, :]
        elif raw_phi.ndim == 2:
            # (nn, maxgate) - single source, multiple gates
            phi_wv = raw_phi[:, :1]
        else:
            # (nn,) - single source, single gate
            phi_wv = raw_phi.reshape(-1, 1)
        # transpose if mmc returned (slots, nn) instead of (nn, slots)
        if phi_wv.ndim == 2 and phi_wv.shape[0] != n_node and phi_wv.shape[1] == n_node:
            phi_wv = phi_wv.T

        # detphi[d, s] = forward fluence from source s evaluated at detector d
        detphi_wv = np.full((detnum, srcnum), np.nan)
        elem = np.asarray(cfg["elem"], dtype=int)
        for d in range(detnum):
            eid = optode_loc[d]
            if eid >= 0:
                nodes_d = elem[eid, :4] - 1  # to 0-based
                detphi_wv[d, :] = optode_bary[d] @ phi_wv[nodes_d, :srcnum]

        if is_multi_wv:
            phi_out[wv] = phi_wv
            detval_out[wv] = detphi_wv
            if need_jacobian:
                # mmclab/pmmc-native orientation: (Nn, Ns*Nd). Downstream
                # consumers (e.g. runrecon's MC branch) transpose only when
                # they need the (Nsd, Nn) layout used by the FEM jac() path.
                # CW mode (outputtype='adjoint') skips J_D, so out['jd'] is
                # absent; only RF (outputtype='adjoint_mua_d') populates it.
                Jmua_map[wv] = np.asarray(out["jmua"], dtype=float)
                if "jd" in out and out["jd"] is not None:
                    Jd_map[wv] = np.asarray(out["jd"], dtype=float)
        else:
            phi_out = phi_wv
            detval_out = detphi_wv
            if need_jacobian:
                Jext_single = {"mua": np.asarray(out["jmua"], dtype=float)}
                if "jd" in out and out["jd"] is not None:
                    Jext_single["dcoeff"] = np.asarray(out["jd"], dtype=float)

    if is_multi_wv and need_jacobian:
        Jext = {"mua": Jmua_map}
        if Jd_map:
            Jext["dcoeff"] = Jd_map
    elif need_jacobian:
        Jext = Jext_single
    else:
        Jext = None

    if return_jacobian:
        return detval_out, phi_out, Jext
    return detval_out, phi_out


def _runforward_mcx(
    cfg: dict, return_jacobian: bool = False, **kwargs
) -> Tuple[Any, ...]:
    """Monte Carlo voxel-grid forward branch (mcxlab/pmcx).

    Port of the cfg.vol mcxlab branch in redbird-m/matlab/rbrunforward.m
    (~ lines 290-345).  Routes the cfg through ``pmcx`` once.  Returns the
    voxel-averaged detector readings and, when ``return_jacobian`` is True,
    the 4D voxel Jacobian ``Jext.mua`` (shape ``(Nx, Ny, Nz, Ns*Nd)``).
    Downstream consumers (e.g. `runrecon`) detect the 4D shape and route
    through `reglsqr` instead of the normal-equation Gauss-Newton step.
    """
    if not _HAS_PMCX:
        raise ImportError(
            "cfg.vol is set but pmcx is not importable; install pmcx to enable "
            "the voxel-grid Monte Carlo path."
        )

    if cfg.get("helmholtz") or cfg.get("bulkprop") is not None:
        raise ValueError(
            "MWT/Helmholtz forward cannot be combined with the Monte Carlo path."
        )

    cfg = copy.copy(cfg)
    if "prop" in cfg:
        cfg["prop"] = copy.copy(cfg["prop"])

    if cfg.get("tstart") is None:
        cfg["tstart"] = 0.0
    if cfg.get("tend") is None:
        cfg["tend"] = 5e-9
    if cfg.get("tstep") is None:
        cfg["tstep"] = cfg["tend"]

    avgsize = float(kwargs.get("avgsize", 1.0))

    srcpos = np.atleast_2d(np.asarray(cfg["srcpos"], dtype=float))
    detpos = np.atleast_2d(np.asarray(cfg["detpos"], dtype=float))
    srcnum = srcpos.shape[0]
    detnum = detpos.shape[0]

    # broadcast srcdir to Nsrc rows (mcxlab/pmcx multi-source parser requires
    # matching row counts, same convention as mmclab)
    srcdir = np.atleast_2d(np.asarray(cfg["srcdir"], dtype=float))
    if srcdir.shape[0] == 1 and srcnum > 1:
        srcdir = np.tile(srcdir, (srcnum, 1))
    cfg["srcdir"] = srcdir
    if srcpos.shape[1] < 4:
        srcpos = np.hstack([srcpos, np.ones((srcpos.shape[0], 4 - srcpos.shape[1]))])
    cfg["srcpos"] = srcpos

    need_jacobian = bool(return_jacobian)

    # CW (omega == 0) cannot separate mua and D; ask for the single-output
    # 'adjoint' kernel (J_mua only), RF uses 'adjoint_mua_d'.
    omega_val = cfg.get("omega", 0)
    if isinstance(omega_val, dict):
        is_rf_mc = any(float(v) > 0 for v in omega_val.values())
    else:
        is_rf_mc = float(omega_val) > 0 if omega_val is not None else False

    if need_jacobian:
        detdir = cfg.get("detdir")
        if detdir is None or len(np.asarray(detdir)) == 0:
            from .utility import getdetdir_vol

            cfg["detdir"] = getdetdir_vol(cfg)
        cfg["detdir"] = np.atleast_2d(np.asarray(cfg["detdir"], dtype=float))
        if cfg["detdir"].shape[1] < 4:
            pad = np.zeros((cfg["detdir"].shape[0], 4 - cfg["detdir"].shape[1]))
            cfg["detdir"] = np.hstack([cfg["detdir"], pad])
        cfg["srcid"] = -1
        cfg["outputtype"] = "adjoint_mua_d" if is_rf_mc else "adjoint"
    elif cfg.get("detdir") is not None and len(np.asarray(cfg["detdir"])) > 0:
        cfg["detdir"] = np.atleast_2d(np.asarray(cfg["detdir"], dtype=float))
        if cfg["detdir"].shape[1] < 4:
            pad = np.zeros((cfg["detdir"].shape[0], 4 - cfg["detdir"].shape[1]))
            cfg["detdir"] = np.hstack([cfg["detdir"], pad])
        cfg["srcid"] = -2
        cfg["outputtype"] = "fluence"
    else:
        cfg["srcid"] = -1
        cfg["outputtype"] = "fluence"

    cfg.pop("srcdata", None)
    cfg.pop("extrasrclen", None)

    out = _pmcx.run(cfg)

    # mcxlab raw flux layout: (Nx, Ny, Nz, Nt, Nsrc+Ndet) or with a singleton
    # time axis squeezed.  Take time-bin 0 (CW).
    phi = np.asarray(out["flux"])
    if phi.ndim == 5:
        phi = phi[:, :, :, 0, :]
    elif phi.ndim == 4 and phi.shape[3] == 1:
        phi = phi[:, :, :, 0]
        phi = phi[..., None]

    # detphi[d, s] = average forward fluence at detector d for source s
    # (centered voxel block of half-width `avgsize` mm).
    detphi = np.full((detnum, srcnum), np.nan)
    for s in range(srcnum):
        for d in range(detnum):
            detphi[d, s] = _voxelmean(phi[..., s], detpos[d, :3], avgsize)

    Jext = None
    if need_jacobian:
        Jext = {"mua": np.asarray(out["jmua"], dtype=float)}
        if "jd" in out and out["jd"] is not None:
            Jext["dcoeff"] = np.asarray(out["jd"], dtype=float)

    if return_jacobian:
        return detphi, phi, Jext
    return detphi, phi


def _voxelmean(vol: np.ndarray, pos: np.ndarray, avgsize: float) -> float:
    """Mean of a voxel-grid scalar field over a (2*r+1)^3 block centered at pos.

    Port of redbird-m/matlab/rbvoxelmean.m: clamp the block to the grid bounds
    and return the arithmetic mean.  Used to convert a 3D fluence volume into
    a single detector reading.
    """
    Nx, Ny, Nz = vol.shape
    r = max(int(round(avgsize)), 0)
    cx = min(max(int(round(pos[0])), 1), Nx)
    cy = min(max(int(round(pos[1])), 1), Ny)
    cz = min(max(int(round(pos[2])), 1), Nz)
    x0, x1 = max(cx - r, 1), min(cx + r, Nx)
    y0, y1 = max(cy - r, 1), min(cy + r, Ny)
    z0, z1 = max(cz - r, 1), min(cz + r, Nz)
    block = vol[x0 - 1 : x1, y0 - 1 : y1, z0 - 1 : z1]
    if block.size == 0:
        return float("nan")
    return float(np.mean(block))


def _tsearchn_bary(
    node: np.ndarray, elem: np.ndarray, pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Locate each point in ``pts`` inside a tet mesh and compute its
    barycentric coordinates.

    Mirrors ``[loc, bary] = tsearchn(node, elem, pts)`` from MATLAB.

    Parameters
    ----------
    node : ndarray  (Nn, 3)
    elem : ndarray  (Ne, 4)   -- 1-based vertex indices
    pts  : ndarray  (Np, 3)

    Returns
    -------
    loc  : ndarray  (Np,)     -- 0-based element index, -1 when not inside any
    bary : ndarray  (Np, 4)   -- barycentric weights (sum to 1 when loc >= 0)
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:  # pragma: no cover -- scipy is a hard dep elsewhere
        raise ImportError("scipy.spatial.Delaunay is required for _tsearchn_bary")

    np_pts = pts.shape[0]
    loc = np.full(np_pts, -1, dtype=np.int64)
    bary = np.zeros((np_pts, 4), dtype=float)

    # Build a Delaunay tessellation just for the point-location query, then
    # match each found simplex back to the user's elem table by vertex set.
    # For a forward mesh whose elem already IS a Delaunay tessellation this
    # is a wash; for user meshes it's a robust fallback at the cost of one
    # rebuild.
    elem0 = elem.astype(int) - 1
    dt = Delaunay(node[:, :3])

    found = dt.find_simplex(pts[:, :3])

    # Map Delaunay simplex -> element index by sorted vertex tuple
    elem_keys = {tuple(sorted(row)): i for i, row in enumerate(elem0)}

    for p in range(np_pts):
        s = found[p]
        if s < 0:
            continue
        vidx = dt.simplices[s]
        key = tuple(sorted(vidx.tolist()))
        eid = elem_keys.get(key)
        if eid is None:
            continue
        loc[p] = eid
        # barycentric weights against the user's vertex order: solve
        # A @ bary = b where A rows are [v0 v1 v2 v3; 1 1 1 1] (4x4) and
        # b = [pt; 1]
        ee = elem0[eid]
        A = np.vstack([node[ee, :3].T, np.ones(4)])
        b = np.append(pts[p, :3], 1.0)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        bary[p] = w

    return loc, bary


def runtd(cfg: dict, **kwargs) -> Tuple[Any, Any]:
    """
    Time-domain DOT forward solver using implicit Crank-Nicolson.

    Solves the time-dependent diffusion equation
        -div(D grad Phi) + mua * Phi + (1/c) dPhi/dt = S(r, t)
    using the theta-method (default theta=0.5 = Crank-Nicolson). Engaged
    automatically by `runforward` when cfg.tstart, cfg.tstep, and cfg.tend
    are all defined.

    Per-wavelength time-step operators (constant Delta_t -> single
    factorization reused across all time steps and all sources):
        A_TD = M / (c * dt) + theta * A_cw                  (LHS)
        B_TD = M / (c * dt) - (1 - theta) * A_cw            (RHS update)
    where A_cw is the CW spatial operator (femlhs mode=2) and M is the
    consistent mass matrix (femlhs mode=3). The time-step recurrence is
        Phi_{n+1} = A_TD^-1 ( B_TD * Phi_n + 0.5 * (b_n + b_{n+1}) )

    Parameters
    ----------
    cfg : dict
        Standard forward configuration plus the time-domain triplet
        cfg.tstart, cfg.tstep, cfg.tend (all in seconds).
    **kwargs : dict
        theta : float, default 0.5
            Theta-method weight (0.5 = Crank-Nicolson, 1.0 = backward Euler).
        srctemporal : array or callable, optional
            Temporal modulation of the source. Defaults to an impulse at
            t = tstart (TPSF). If a vector of length Nt, gives s(t_n) per
            step. If a callable f(t), evaluated at the time grid.
        phi0 : ndarray, optional
            Initial condition Nn x Nsrc. Defaults to zeros, except for
            the impulse default which uses Phi(t_start) = c * M\\S_spatial.
        tdsavevol : bool, default False
            If True, also return the full volumetric phi (Nn x Nsrc x Nt).
            Defaults to False to manage memory.

    Returns
    -------
    detphi : ndarray or dict
        Detector readings, Ndet x Nsrc x Nt (single-wavelength) or a dict
        keyed by wavelength.
    phi : ndarray or dict or None
        Volumetric forward solution Nn x Nsrc x Nt when tdsavevol is True;
        otherwise None.
    """
    from scipy.sparse.linalg import splu
    from .property import getbulk

    theta = kwargs.get("theta", 0.5)
    tdsavevol = kwargs.get("tdsavevol", False)
    phi0 = kwargs.get("phi0", None)
    srctemporal = kwargs.get("srctemporal", None)

    if cfg["tend"] <= cfg["tstart"] or cfg["tstep"] <= 0:
        raise ValueError("require cfg.tend > cfg.tstart and cfg.tstep > 0")

    # time grid (inclusive endpoint)
    tvec = np.arange(cfg["tstart"], cfg["tend"] + 0.5 * cfg["tstep"], cfg["tstep"])
    nt = len(tvec)

    # bulk refractive index for the 1/c prefactor
    bk = getbulk(cfg)
    if isinstance(bk, dict):
        bk = list(bk.values())[0]
    nref_bulk = bk[3]
    c_mm_per_s = C0 / nref_bulk

    # wavelength keys
    wavelengths = [""]
    if isinstance(cfg.get("prop"), dict):
        wavelengths = list(cfg["prop"].keys())

    # source-detector mapping
    sd = kwargs.get("sd")
    if sd is None:
        sd = sdmap(cfg)
    if not isinstance(sd, dict):
        sd = {wv: sd for wv in wavelengths}

    if "deldotdel" not in cfg or cfg["deldotdel"] is None:
        cfg["deldotdel"], _ = deldotdel(cfg)

    nn = cfg["node"].shape[0]

    # column counts per femrhs ordering: [point_src, wide_src, point_det, wide_det]
    srcnum = 0
    if cfg.get("srcpos") is not None:
        sp = np.atleast_2d(cfg["srcpos"])
        if sp.size > 0 and sp.shape[1] >= 3 and sp.shape[1] < 6:
            srcnum = sp.shape[0]
    wfsrcnum = 0
    if cfg.get("widesrc") is not None and np.size(cfg["widesrc"]) > 0:
        wfsrcnum = cfg["widesrc"].shape[1]
    detnum = 0
    if cfg.get("detpos") is not None:
        dp = np.atleast_2d(cfg["detpos"])
        if dp.size > 0 and dp.shape[1] >= 3 and dp.shape[1] < 6:
            detnum = dp.shape[0]
    wfdetnum = 0
    if cfg.get("widedet") is not None and np.size(cfg["widedet"]) > 0:
        wfdetnum = cfg["widedet"].shape[1]
    total_src = srcnum + wfsrcnum
    total_det = detnum + wfdetnum

    # consistent mass matrix is wavelength-independent (geometry only),
    # build it once outside the wavelength loop
    M_mass = femlhs(cfg, cfg["deldotdel"], wavelengths[0] if wavelengths[0] else "", 3)

    detphi_out = {}
    phi_out = {}

    for wv in wavelengths:
        # CW spatial operator (omega = 0)
        A_cw = femlhs(cfg, cfg["deldotdel"], wv, 2)

        M_T = M_mass / (c_mm_per_s * cfg["tstep"])
        A_TD = (M_T + theta * A_cw).tocsc()
        B_TD = (M_T - (1 - theta) * A_cw).tocsr()

        # factor A_TD once per wavelength (constant Delta_t -> constant LHS)
        dA_TD = splu(A_TD)

        # spatial source/detector vectors (Nn x (total_src + total_det))
        rhs_spatial, _loc, _bary, _opt = femrhs(cfg, sd, wv, 1)
        if sparse.issparse(rhs_spatial):
            rhs_dense = np.asarray(rhs_spatial.toarray())
        else:
            rhs_dense = np.asarray(rhs_spatial)
        S_spatial = rhs_dense[:, :total_src]
        D_spatial = rhs_dense[:, total_src : total_src + total_det]

        # temporal modulation
        if srctemporal is None:
            srct = np.zeros(nt, dtype=float)  # impulse handled via IC
            is_impulse = True
        elif callable(srctemporal):
            srct = np.asarray(srctemporal(tvec)).flatten()
            is_impulse = False
        else:
            srct = np.asarray(srctemporal).flatten()
            if len(srct) != nt:
                raise ValueError(
                    f"cfg.srctemporal must have {nt} entries (one per time step)"
                )
            is_impulse = False

        # initial condition: for impulse default, Phi(t_start) = c * M\S_spatial
        # (the unique solution to the impulse-jump equation across the delta).
        if phi0 is not None:
            phi_prev = np.asarray(phi0, dtype=float).copy()
            if phi_prev.shape != (nn, total_src):
                raise ValueError(f"phi0 must have shape (Nn={nn}, Nsrc={total_src})")
        elif is_impulse and total_src > 0:
            from scipy.sparse.linalg import spsolve

            phi_prev = c_mm_per_s * spsolve(M_mass.tocsc(), S_spatial)
            if phi_prev.ndim == 1:
                phi_prev = phi_prev[:, np.newaxis]
        else:
            phi_prev = np.zeros((nn, total_src), dtype=float)

        # output allocation
        detphi_t = np.zeros((total_det, total_src, nt), dtype=float)
        phi_t = np.zeros((nn, total_src, nt), dtype=float) if tdsavevol else None
        if tdsavevol:
            phi_t[:, :, 0] = phi_prev
        if total_det > 0 and total_src > 0:
            detphi_t[:, :, 0] = D_spatial.T @ phi_prev

        # time loop
        for n in range(1, nt):
            if is_impulse:
                rhs_step = B_TD @ phi_prev
            else:
                rhs_step = B_TD @ phi_prev + 0.5 * (srct[n - 1] + srct[n]) * S_spatial
            # multi-RHS solve via cached factorization
            if total_src > 0:
                phi_new = dA_TD.solve(rhs_step)
            else:
                phi_new = phi_prev
            if tdsavevol:
                phi_t[:, :, n] = phi_new
            if total_det > 0 and total_src > 0:
                detphi_t[:, :, n] = D_spatial.T @ phi_new
            phi_prev = phi_new

        detphi_out[wv] = detphi_t
        if tdsavevol:
            phi_out[wv] = phi_t

    if len(wavelengths) == 1:
        return (
            detphi_out[wavelengths[0]],
            phi_out.get(wavelengths[0]) if tdsavevol else None,
        )
    return detphi_out, (phi_out if tdsavevol else None)


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

    The optional `mode` argument selects the operator returned:
        mode = 1 (default) : full FD/CW LHS (uses cfg.omega)
        mode = 2           : CW spatial operator (omega=0)
        mode = 3           : pure consistent mass matrix M (no stiffness, no
                             absorption, no BC; used by the time-domain
                             Crank-Nicolson solver in runtd).
    """
    nn = cfg["node"].shape[0]
    ne = cfg["elem"].shape[0]
    evol = cfg["evol"]
    area = cfg["area"]

    # Convert 1-based to 0-based
    elem_0 = cfg["elem"][:, :4].astype(np.int32) - 1
    face_0 = cfg["face"].astype(np.int32) - 1

    # pure mass-matrix mode: skip property extraction and BC; used by runtd.
    ismassonly = mode == 3

    # MWT detection: cfg.bulk.epsilon or cfg.bulk.sigma defines the bulk medium
    # for the radiation boundary condition.
    ishelmholtz = (
        not ismassonly
        and isinstance(cfg.get("bulk"), dict)
        and ("epsilon" in cfg["bulk"] or "sigma" in cfg["bulk"])
    )

    if ismassonly:
        # mass matrix only: avol=0, breal=1 (uniform), bimag=0; skip BC.
        # Property extraction and reff lookup are skipped entirely.
        avol = np.zeros(ne)
        breal = np.ones(ne)
        bimag = np.zeros(ne)
        is_complex = False
        # jump straight to the volume assembly
        rows_list = []
        cols_list = []
        vals_list = []
        offdiag_idx = [1, 2, 3, 5, 6, 8]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for k, (i, j) in enumerate(pairs):
            rows_list.append(elem_0[:, i])
            cols_list.append(elem_0[:, j])
            val = 0.05 * breal * evol  # avol*deldotdel = 0
            vals_list.append(val)
            rows_list.append(elem_0[:, j])
            cols_list.append(elem_0[:, i])
            vals_list.append(val)
        diag_idx = [0, 4, 7, 9]
        for k in range(4):
            rows_list.append(elem_0[:, k])
            cols_list.append(elem_0[:, k])
            val = 0.10 * breal * evol
            vals_list.append(val)
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
        return sparse.coo_matrix(
            (vals, (rows, cols)), shape=(nn, nn), dtype=float
        ).tocsr()

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
            permea = (
                props[seg_idx, 2]
                if props.shape[1] >= 3
                else np.full_like(eps_r, 4 * np.pi * 1e-10)
            )
        else:
            raise ValueError("Property format not recognized")
        avol = np.ones_like(eps_r)
        breal = -(omega**2) * permea * EPS0_MM * eps_r
        bimag = omega * permea * sigma
        is_complex = True
    else:
        if props.shape[0] == nn or props.shape[0] == ne:
            mua = props[:, 0]
            musp = (
                props[:, 1] * (1 - props[:, 2]) if props.shape[1] >= 3 else props[:, 1]
            )
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
        bimag_e = (
            bimag[elem_0] if hasattr(bimag, "__len__") else np.full(elem_0.shape, bimag)
        )
        avol_e = np.mean(avol[elem_0], axis=1) if hasattr(avol, "__len__") else avol

        for k, (i, j) in enumerate(pairs):
            rows_list.append(elem_0[:, i])
            cols_list.append(elem_0[:, j])
            val = (
                deldotdel_mat[:, offdiag_idx[k]] * avol_e + (breal_e @ w1[:, k]) * evol
            )
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
        k2bg = (omega**2) * bk[2] * EPS0_MM * bk[0] - 1j * omega * bk[2] * bk[1]
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
    rhs = sparse.lil_matrix(
        (nn, total_cols), dtype=complex if is_complex_rhs else float
    )

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


def jacepssigma(
    Jmua: dict, omegas, has_eps: bool = True, has_sigma: bool = True
) -> dict:
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
        we = -(omega**2) * MU0_MM * EPS0_MM
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


def jacmuafast(sd, phi, nvol, elem=None):
    """Approximated nodal-adjoint Jacobian for ``mua``.

    Port of redbird-m/matlab/rbjacmuafast.m.  Implements the nodal-adjoint
    closed-form J_mua(n) = -V_n * phi_s(n) * phi_r(n) derived in
    Q. Fang's PhD thesis (Chap. 6 sssec:3d3d:nodal, eq.
    3d3d:adjoint:nodal).  Cheaper than the full FEM ``jac`` build; valid
    when the forward mesh is much finer than the parameter mesh.

    Parameters
    ----------
    sd : ndarray  (Nsd x {3,4})
        Source-detector mapping table (1-based source/detector columns).
    phi : ndarray or dict
        Forward nodal fluence.  ``(Nn, Nsrc+Ndet)`` for a single
        wavelength, or a dict keyed by wavelength.
    nvol : ndarray
        Nodal Voronoi volumes (length Nn).  When ``len(nvol) == Ne``
        instead, ``nvol`` is interpreted as element volumes and the
        result is scattered to nodes via the 0.25 weight from
        ``elem2node``.
    elem : ndarray, optional
        Element list (Ne x 4, 1-based).  Required when ``nvol`` is
        per-element rather than per-node.

    Returns
    -------
    Jmua : ndarray or dict
        ``(Nsd, Nn)`` per wavelength; dict keyed by wavelength when
        ``phi`` is multi-spectral.
    """
    if sd is None or phi is None or nvol is None:
        raise ValueError("jacmuafast requires sd, phi, and nvol")

    is_multi = isinstance(phi, dict)
    if is_multi:
        wavelengths = list(phi.keys())
    else:
        wavelengths = [""]
        phi = {"": phi}

    Jmua_out = {}
    nvol = np.asarray(nvol).ravel()

    for wv in wavelengths:
        phiwv = np.asarray(phi[wv])
        sdwv = sd[wv] if isinstance(sd, dict) else sd
        sd_arr = np.asarray(sdwv)
        nsd = sd_arr.shape[0]
        # sd columns 0 and 1 are 1-based src/det indices into phi columns
        src_cols = sd_arr[:, 0].astype(int) - 1
        det_cols = sd_arr[:, 1].astype(int) - 1

        if phiwv.shape[0] == nvol.size:
            # nvol is per-node: direct elementwise build
            Ja = np.empty((nsd, phiwv.shape[0]), dtype=phiwv.dtype)
            for i in range(nsd):
                Ja[i, :] = phiwv[:, src_cols[i]] * phiwv[:, det_cols[i]] * nvol
        elif elem is not None and nvol.size == np.asarray(elem).shape[0]:
            # nvol is per-element: scatter and convert to nodes via 0.25 sum
            elem_arr = np.asarray(elem)[:, :4].astype(int) - 1
            Ne = elem_arr.shape[0]
            Ja_elem = np.zeros((Ne, nsd), dtype=phiwv.dtype)
            for i in range(nsd):
                for j in range(4):
                    Ja_elem[:, i] += (
                        phiwv[elem_arr[:, j], src_cols[i]]
                        * phiwv[elem_arr[:, j], det_cols[i]]
                        * nvol
                    )
            Ja = Ja_elem.T * 0.25
        else:
            raise ValueError(
                "jacmuafast: phi rows must match len(nvol) (nodal) "
                "or nvol must match elem rows (per-element)"
            )

        # Increasing mua decreases phi -> sign flip
        Jmua_out[wv] = -Ja

    if not is_multi:
        return Jmua_out[""]
    return Jmua_out


def jacmus(Jd, musp, g=0.0):
    """Convert the diffusion-coefficient Jacobian into a ``mus'`` Jacobian.

    Port of redbird-m/matlab/rbjacmus.m.

        J_mus' = -J_D / (3 * mus'^2 * (1 - g))

    Parameters
    ----------
    Jd : ndarray
        Jacobian of the diffusion coefficient D.
    musp : float or ndarray
        Reduced scattering coefficient (per-node or scalar).
    g : float, default 0
        Anisotropy.
    """
    factor = 1.0 / (3.0 * np.asarray(musp) * np.asarray(musp) * (1.0 - g))
    return -np.asarray(Jd) * factor


def jacscatamp(Jd, dcoeff, wavelen, scatpow, lref=1e9):
    """Jacobian of the scattering amplitude from the diffusion-coeff Jacobian.

    Port of redbird-m/matlab/rbjacscatamp.m.  Power-law model
    ``musp = scatamp * (wavelen / lref)^(-scatpow)`` gives
    ``dD/dscatamp = -3 * D^2 * (wavelen/lref)^(-scatpow)``.

    Parameters
    ----------
    Jd : ndarray
        Jacobian of the diffusion coefficient.
    dcoeff : ndarray
        Diffusion coefficient values at each node.
    wavelen : float
        Wavelength in nm.
    scatpow : float or ndarray
        Current scattering-power estimate (scalar or per-node).
    lref : float, default 1e9
        Reference wavelength in nm.  Use ``500`` for the
        500 nm-normalized convention.

    Returns
    -------
    Jscatamp : ndarray
        Jacobian of the scattering amplitude.
    """
    dDdscatamp = (
        -3.0
        * np.asarray(dcoeff)
        * np.asarray(dcoeff)
        * (np.asarray(wavelen) / lref) ** (-np.asarray(scatpow))
    )
    return np.asarray(Jd) * dDdscatamp


def jacscatpow(Jd, dcoeff, wavelen, lref=1e9):
    """Jacobian of the scattering power from the diffusion-coeff Jacobian.

    Port of redbird-m/matlab/rbjacscatpow.m.
        dD/dscatpow = D * log(wavelen / lref)

    Parameters
    ----------
    Jd : ndarray
        Jacobian of the diffusion coefficient.
    dcoeff : ndarray
        Diffusion coefficient values at each node.
    wavelen : float
        Wavelength in nm.
    lref : float, default 1e9
        Reference wavelength in nm.

    Returns
    -------
    Jscatpow : ndarray
        Jacobian of the scattering-power parameter.
    """
    dDdscatpow = np.asarray(dcoeff) * np.log(np.asarray(wavelen) / lref)
    return np.asarray(Jd) * dDdscatpow


def jacscat(Jd, dcoeff, scatpow, wv=None, lref=1e9, suffix=""):
    """Build scattering-amplitude and scattering-power Jacobians from J_D.

    Port of redbird-m/matlab/rbjacscat.m.  Wraps :func:`jacscatamp` and
    :func:`jacscatpow` over multiple wavelengths and packages the
    results in a dict keyed by ``scatamp<suffix>`` and ``scatpow<suffix>``.

    Parameters
    ----------
    Jd : dict
        Jacobian of the diffusion coefficient keyed by wavelength (str).
    dcoeff : dict
        Diffusion coefficient at each node keyed by wavelength.
    scatpow : float or ndarray
        Current scattering-power estimate.
    wv : iterable of str, optional
        Wavelength list.  Defaults to ``Jd.keys()``.
    lref : float, default 1e9
        Reference wavelength in nm.  Pass ``500`` for the 500 nm-
        normalized convention.
    suffix : str, default ''
        Suffix appended to the output keys (e.g. ``'500'`` yields
        ``scatamp500`` / ``scatpow500``).

    Returns
    -------
    Jscat : dict
        ``{f'scatamp{suffix}': vstacked_J, f'scatpow{suffix}': vstacked_J}``.
        Each value is the per-wavelength Jacobian rows vertically
        stacked, matching rbjacscat.m's behaviour.
    """
    if not isinstance(Jd, dict):
        raise TypeError("jacscat requires Jd as a dict keyed by wavelength")

    if wv is None:
        wv = list(Jd.keys())

    ampname = f"scatamp{suffix}"
    powname = f"scatpow{suffix}"

    amp_blocks = []
    pow_blocks = []
    for wv_str in wv:
        wv_num = float(wv_str)
        amp_blocks.append(jacscatamp(Jd[wv_str], dcoeff[wv_str], wv_num, scatpow, lref))
        pow_blocks.append(jacscatpow(Jd[wv_str], dcoeff[wv_str], wv_num, lref))

    return {
        ampname: np.vstack(amp_blocks),
        powname: np.vstack(pow_blocks),
    }


def jacnode(Jmua_elem, Jd_elem=None, elem=None, nodelen=None):
    """Convert element-based Jacobians to node-based Jacobians.

    Port of redbird-m/matlab/rbjacnode.m.  Uses the same elem-to-node
    0.25-weighted scatter as :func:`redbirdpy.utility.elem2node`.

    Parameters
    ----------
    Jmua_elem : ndarray  (Nsd x Ne)
        Element-wise Jacobian of ``mua``.
    Jd_elem : ndarray, optional  (Nsd x Ne)
        Element-wise Jacobian of the diffusion coefficient.
    elem : ndarray  (Ne x 4, 1-based)
        Mesh element list.
    nodelen : int
        Total node count.

    Returns
    -------
    Jmua_node : ndarray  (Nsd x Nn)
    Jd_node : ndarray  (Nsd x Nn), only when ``Jd_elem`` is provided
    """
    if elem is None or nodelen is None:
        raise ValueError("jacnode requires elem and nodelen")

    from .utility import elem2node

    Jmua_node = elem2node(elem, Jmua_elem, nodelen)

    if Jd_elem is None:
        return Jmua_node

    Jd_node = elem2node(elem, Jd_elem, nodelen)
    return Jmua_node, Jd_node
