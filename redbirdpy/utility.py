"""
Redbird Utility Module - Mesh and data utilities for DOT/NIRS.

INDEX CONVENTION: All mesh indices (elem, face) are 1-based to match
MATLAB/iso2mesh. Conversion to 0-based occurs only when indexing numpy arrays.

This module provides utility functions for mesh preparation, source/detector
handling, data manipulation, and visualization support.
"""

__all__ = [
    "meshprep",
    "deldotdel",
    "sdmap",
    "getoptodes",
    "getdistance",
    "getltr",
    "getreff",
    "elem2node",
    "addnoise",
    "meshinterp",
    "src2bc",
    "HAS_ISO2MESH",
    "forcearray",
]

import numpy as np
from scipy import sparse
from typing import Dict, Tuple, Optional, Union, List, Any
import warnings


# Use iso2mesh for mesh operations (maintains 1-based convention)
try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False
    warnings.warn(
        "iso2mesh not found. Some mesh functions will use fallback implementations."
    )

# Try to import matplotlib Path for fast inpolygon
try:
    from matplotlib.path import Path as MplPath

    HAS_MPL_PATH = True
except ImportError:
    HAS_MPL_PATH = False


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


def meshprep(cfg: dict) -> Tuple[dict, np.ndarray]:
    """
    Prepare mesh structure with all derived quantities.

    All mesh indices (elem, face) remain 1-based in the returned cfg.
    """
    from .property import updateprop, getbulk

    # Convert list inputs to numpy arrays
    cfg = forcearray(
        cfg,
        [
            "node",
            "elem",
            "face",
            "srcpos",
            "srcdir",
            "detpos",
            "detdir",
            "prop",
            "seg",
            "widesrc",
            "widedet",
        ],
    )

    if "node" not in cfg or "elem" not in cfg:
        raise ValueError("cfg must contain 'node' and 'elem'")

    node = cfg["node"]
    elem = cfg["elem"]  # 1-based indices

    # Extract segmentation if present in elem column 5
    if elem.shape[1] > 4 and ("seg" not in cfg or cfg["seg"] is None):
        cfg["seg"] = elem[:, 4].astype(int)

    # Reorient elements to have positive volume (outward normals)
    if not cfg.get("isreoriented", False):
        if HAS_ISO2MESH:
            # iso2mesh.meshreorient handles 1-based indices and returns volumes
            elem_reoriented, evol, _ = i2m.meshreorient(node, elem[:, :4])
            cfg["evol"] = evol
        else:
            elem_reoriented = _meshreorient_fallback(node, elem[:, :4])

        if elem.shape[1] > 4:
            cfg["elem"] = np.column_stack([elem_reoriented, elem[:, 4:]])
        else:
            cfg["elem"] = elem_reoriented
        cfg["isreoriented"] = True

    # Compute surface faces (1-based indices)
    if "face" not in cfg or cfg["face"] is None:
        if HAS_ISO2MESH:
            face_result = i2m.volface(cfg["elem"][:, :4])
            # volface may return (face, faceid) tuple or just face
            if isinstance(face_result, tuple):
                cfg["face"] = face_result[0]
            else:
                cfg["face"] = face_result
        else:
            cfg["face"] = _volface_fallback(cfg["elem"][:, :4])

    # Compute face areas
    if "area" not in cfg or cfg["area"] is None:
        if HAS_ISO2MESH:
            cfg["area"] = i2m.elemvolume(node, cfg["face"])
        else:
            cfg["area"] = _elemvolume_fallback(node, cfg["face"])

    # Compute element volumes
    if "evol" not in cfg or cfg["evol"] is None:
        if HAS_ISO2MESH:
            cfg["evol"] = i2m.elemvolume(node, cfg["elem"][:, :4])
        else:
            cfg["evol"] = _elemvolume_fallback(node, cfg["elem"][:, :4])

    # Check for degenerate elements
    if np.any(cfg["evol"] == 0):
        bad_elem = np.where(cfg["evol"] == 0)[0]
        raise ValueError(f"Degenerate elements detected at indices: {bad_elem}")

    # Compute nodal volumes
    if "nvol" not in cfg or cfg["nvol"] is None:
        cfg["nvol"] = _nodevolume(node, cfg["elem"][:, :4], cfg["evol"])

    # Validate sources and detectors
    if "srcpos" not in cfg:
        raise ValueError("cfg.srcpos is required")
    src_is_line = (
        cfg.get("srcpos") is not None and np.atleast_2d(cfg["srcpos"]).shape[1] >= 6
    )
    if "srcdir" not in cfg and not src_is_line:
        raise ValueError("cfg.srcdir is required")

    # Update properties if multi-spectral
    if isinstance(cfg.get("prop"), dict) and "param" in cfg:
        cfg["prop"] = updateprop(cfg)

    # MWT (Helmholtz) detection: cfg.bulk.epsilon or cfg.bulk.sigma
    ishelmholtz = isinstance(cfg.get("bulk"), dict) and (
        "epsilon" in cfg["bulk"] or "sigma" in cfg["bulk"]
    )

    # face-adjacency table (4 face-neighbors per tet, 0 = boundary face).
    # used by line-source tracing for both DOT and MWT.
    if HAS_ISO2MESH and ("facenb" not in cfg or cfg.get("facenb") is None):
        cfg["facenb"] = i2m.faceneighbors(cfg["elem"][:, :4])

    if not ishelmholtz:
        # compute R_eff - effective reflection coeff, and musp0 - background mus'
        if "reff" not in cfg or cfg["reff"] is None:
            bkprop = getbulk(cfg)
            if isinstance(bkprop, dict):
                cfg["reff"] = {}
                cfg["musp0"] = {}
                for wv, prop in bkprop.items():
                    cfg["reff"][wv] = getreff(prop[3], 1.0)
                    cfg["musp0"][wv] = prop[1] * (1 - prop[2])
            else:
                cfg["reff"] = getreff(bkprop[3], 1.0)
                cfg["musp0"] = bkprop[1] * (1 - bkprop[2])
    else:
        # MWT: precompute Bayliss-Turkel RBC geometry on cfg.face
        face_0 = cfg["face"][:, :3].astype(int) - 1  # 1-based -> 0-based
        node3 = cfg["node"][:, :3]
        if "facecenter" not in cfg or cfg.get("facecenter") is None:
            cfg["facecenter"] = (
                node3[face_0[:, 0], :] + node3[face_0[:, 1], :] + node3[face_0[:, 2], :]
            ) / 3.0
        if "facenormal" not in cfg or cfg.get("facenormal") is None:
            ab = node3[face_0[:, 1], :] - node3[face_0[:, 0], :]
            ac = node3[face_0[:, 2], :] - node3[face_0[:, 0], :]
            nrm = np.cross(ab, ac)
            nlen = np.sqrt(np.sum(nrm * nrm, axis=1, keepdims=True))
            cfg["facenormal"] = nrm / np.maximum(nlen, 1e-30)
        if "rbcorigin" not in cfg or cfg.get("rbcorigin") is None:
            optodes = []
            if "srcpos" in cfg and cfg["srcpos"] is not None:
                sp = np.atleast_2d(cfg["srcpos"])
                optodes.append(sp[:, :3])
                if sp.shape[1] >= 6:
                    optodes.append(sp[:, 3:6])
            if "detpos" in cfg and cfg["detpos"] is not None:
                dp = np.atleast_2d(cfg["detpos"])
                optodes.append(dp[:, :3])
                if dp.shape[1] >= 6:
                    optodes.append(dp[:, 3:6])
            if optodes:
                cfg["rbcorigin"] = np.vstack(optodes).mean(axis=0)
            else:
                cfg["rbcorigin"] = node3.mean(axis=0)
        if "facer" not in cfg or cfg.get("facer") is None:
            rvec = cfg["facecenter"] - cfg["rbcorigin"][np.newaxis, :]
            cfg["facer"] = np.sqrt(np.sum(rvec * rvec, axis=1))

    # Detect 6-column line-segment srcpos / detpos
    src_is_line = (
        cfg.get("srcpos") is not None and np.atleast_2d(cfg["srcpos"]).shape[1] >= 6
    )
    det_is_line = (
        cfg.get("detpos") is not None and np.atleast_2d(cfg["detpos"]).shape[1] >= 6
    )

    # Process wide-field sources if present
    srctype = cfg.get("srctype", "pencil")
    if (
        srctype not in ["pencil", "isotropic"] or "widesrcid" in cfg or src_is_line
    ) and "widesrc" not in cfg:
        if not src_is_line:
            cfg["srcpos0"] = cfg["srcpos"].copy()
        cfg = src2bc(cfg, isdet=False)

    # Process wide-field detectors if present
    dettype = cfg.get("dettype", "pencil")
    if (
        dettype not in ["pencil", "isotropic"] or "widedetid" in cfg or det_is_line
    ) and "widedet" not in cfg:
        if not det_is_line:
            cfg["detpos0"] = cfg["detpos"].copy()
        cfg = src2bc(cfg, isdet=True)

    # Compute sparse matrix structure
    if "cols" not in cfg or cfg["cols"] is None:
        cfg["rows"], cfg["cols"], cfg["idxcount"] = _femnz(
            cfg["elem"][:, :4], node.shape[0]
        )

    if "idxsum" not in cfg or cfg["idxsum"] is None:
        cfg["idxsum"] = np.cumsum(cfg["idxcount"])

    # Compute gradient operator
    if "deldotdel" not in cfg or cfg["deldotdel"] is None:
        cfg["deldotdel"], _ = deldotdel(cfg)

    # Set default modulation frequency
    if "omega" not in cfg:
        cfg["omega"] = 0

    # Time-domain DOT (Crank-Nicolson) is engaged when cfg.tstart, cfg.tstep,
    # and cfg.tend are all defined. Validate and reject mixed configurations.
    istd = (
        cfg.get("tstart") is not None
        and cfg.get("tstep") is not None
        and cfg.get("tend") is not None
    )
    if istd:
        if cfg["tend"] <= cfg["tstart"]:
            raise ValueError(
                "cfg.tend must be greater than cfg.tstart for time-domain DOT"
            )
        if cfg["tstep"] <= 0:
            raise ValueError("cfg.tstep must be positive for time-domain DOT")
        omega_check = cfg.get("omega", 0)
        if isinstance(omega_check, dict):
            omega_vals = list(omega_check.values())
        else:
            omega_vals = [omega_check]
        if any(np.asarray(v).any() for v in omega_vals):
            raise ValueError(
                "time-domain DOT (cfg.tstart/tstep/tend) cannot be used with "
                "frequency-domain modulation (cfg.omega>0); pick one"
            )
        if ishelmholtz:
            raise ValueError(
                "time-domain Crank-Nicolson is only valid for the parabolic "
                "diffusion equation; remove cfg.bulk.epsilon/sigma to use TD"
            )

    # Create source-detector mapping
    sd = sdmap(cfg)

    return cfg, sd


# ============== Wide-field Source/Detector Functions ==============


def src2bc(cfg: dict, isdet: bool = False) -> dict:
    """
    Convert wide-field source/detector forms into boundary conditions.

    This function computes the inward flux on mesh surface triangles for
    wide-field illumination patterns (planar, pattern, fourier sources).

    Also supports 6-column cfg.srcpos / cfg.detpos for line-segment sources
    (used by both DOT and MWT): columns 1-3 are the start point, columns
    4-6 are the end point. The line is traced through the tet mesh and
    contributes a sparse RHS column built from per-tet line integrals of
    the linear basis functions.

    Parameters
    ----------
    cfg : dict
        Simulation configuration containing mesh and source/detector info
    isdet : bool
        If False (default), process sources. If True, process detectors.

    Returns
    -------
    cfg : dict
        Updated configuration with widesrc/widedet fields added
    """
    if not HAS_ISO2MESH:
        raise ImportError("iso2mesh is required for wide-field source processing")

    # Line-source/detector path: 6-column position array means line endpoints
    if not isdet:
        pos_key, out_key, posbk_key = "srcpos", "widesrc", "srcpos0"
    else:
        pos_key, out_key, posbk_key = "detpos", "widedet", "detpos0"

    if pos_key in cfg and cfg[pos_key] is not None:
        positions = np.atleast_2d(cfg[pos_key])
        if positions.shape[1] >= 6:
            cfg[posbk_key] = positions
            cfg[out_key] = _lineseg2rhs(positions, cfg)
            cfg[pos_key] = np.zeros((0, 3))
            return cfg

    # Determine field names based on source vs detector
    if not isdet:
        type_key, pos_key, dir_key = "srctype", "srcpos", "srcdir"
        param1_key, param2_key = "srcparam1", "srcparam2"
        pattern_key, weight_key = "srcpattern", "srcweight"
        id_key, wideid_key = "srcid", "widesrcid"
        out_key, mapping_key = "widesrc", "wfsrcmapping"
    else:
        type_key, pos_key, dir_key = "dettype", "detpos", "detdir"
        param1_key, param2_key = "detparam1", "detparam2"
        pattern_key, weight_key = "detpattern", "detweight"
        id_key, wideid_key = "detid", "widedetid"
        out_key, mapping_key = "widedet", "wfdetmapping"

    # Check if wide-field processing is needed
    srctype = cfg.get(type_key, "pencil")
    if srctype in ["pencil", "isotropic"] and wideid_key not in cfg:
        return cfg

    srcdir = np.atleast_2d(cfg[dir_key])
    sources = np.atleast_2d(cfg[pos_key])

    # Build wide-field source parameter structure
    if wideid_key in cfg:
        widesrcid = cfg[wideid_key]
        if not isinstance(widesrcid, dict):
            widesrcid = {"": widesrcid}
    else:
        tempwf = {
            "srctype": [srctype],
            "srcid": [cfg.get(id_key, 0)],
            "srcparam1": [np.atleast_1d(cfg[param1_key])],
            "srcparam2": [np.atleast_1d(cfg[param2_key])],
        }
        if pattern_key in cfg:
            tempwf["srcpattern"] = [cfg[pattern_key]]
        if weight_key in cfg:
            tempwf["srcweight"] = [cfg[weight_key]]
        widesrcid = {"": tempwf}

    # Handle optical properties
    if isinstance(cfg.get("prop"), dict):
        prop = cfg["prop"]
    else:
        prop = {"": cfg["prop"]}

    # Ensure widesrcid keys match prop keys
    if set(widesrcid.keys()) != set(prop.keys()):
        if "" in widesrcid:
            temp = widesrcid[""]
            widesrcid = {wv: temp for wv in prop.keys()}

    widesrc_list = []
    wavelengths = list(prop.keys())
    wfsrcmapping = {}
    all_wide_ids = []

    for wv in wavelengths:
        wideparam = widesrcid[wv]
        op = prop[wv]

        # Compute 1/mu_s' for sinking collimated sources
        if op.ndim == 1:
            z0 = 1.0 / (op[0] + op[1] * (1 - op[2]))
        else:
            z0 = 1.0 / (op[1, 0] + op[1, 1] * (1 - op[1, 2]))

        srcmapping = []

        for wideidx in range(len(wideparam["srcid"])):
            srcid = wideparam["srcid"][wideidx]
            all_wide_ids.append(srcid)
            srctype_i = wideparam["srctype"][wideidx]
            srcparam1 = wideparam["srcparam1"][wideidx]
            srcparam2 = wideparam["srcparam2"][wideidx]

            srcpattern = None
            if srctype_i == "pattern" and "srcpattern" in wideparam:
                srcpattern = wideparam["srcpattern"][wideidx]

            srcweight = None
            if "srcweight" in wideparam:
                srcweight = wideparam["srcweight"][wideidx]

            srcpos = sources[srcid, :3]
            srcdir_i = srcdir[0, :3] if srcdir.shape[0] == 1 else srcdir[srcid, :3]

            # Process based on source type
            if srctype_i in ["planar", "pattern", "fourier"]:
                (
                    srcbc,
                    patsize,
                    pface,
                    parea,
                    pnode,
                    nodeid,
                    used_sinkplane,
                ) = _process_planar_source(
                    cfg,
                    srcpos,
                    srcdir_i,
                    srcparam1,
                    srcparam2,
                    srctype_i,
                    srcpattern,
                    z0,
                )
            else:
                raise ValueError(f"Source type '{srctype_i}' is not supported")

            # Apply boundary condition weighting
            reff = cfg["reff"]
            Reff = reff[wv] if isinstance(reff, dict) else reff

            rhs = _apply_bc_weighting(
                cfg, srcbc, Reff, srcweight, pface, parea, pnode, nodeid, used_sinkplane
            )

            # Record mapping
            indices = [len(widesrc_list), len(widesrc_list) + patsize - 1]
            widesrc_list.append(rhs)
            srcmapping.append([srcid, indices[0], indices[1]])

        wfsrcmapping[wv] = np.array(srcmapping) if srcmapping else np.array([])

    # Stack all wide-field sources
    if widesrc_list:
        widesrc = np.vstack(widesrc_list)
    else:
        widesrc = np.array([])

    # Remove wide-field source positions from point sources
    unique_wide_ids = list(set(all_wide_ids))
    mask = np.ones(sources.shape[0], dtype=bool)
    mask[unique_wide_ids] = False
    sources = sources[mask]

    # Simplify mapping if single wavelength
    if len(wfsrcmapping) == 1:
        wfsrcmapping = wfsrcmapping[wavelengths[0]]

    # Update cfg - transpose to (Nn x Npattern)
    cfg[out_key] = widesrc.T if widesrc.size > 0 else np.array([])
    cfg[mapping_key] = wfsrcmapping
    cfg[pos_key] = sources

    return cfg


def _process_planar_source(
    cfg: dict,
    srcpos: np.ndarray,
    srcdir: np.ndarray,
    srcparam1: np.ndarray,
    srcparam2: np.ndarray,
    srctype: str,
    srcpattern: Optional[np.ndarray],
    z0: float,
) -> Tuple[
    np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], bool
]:
    """Process planar/pattern/fourier source geometry."""

    # Define rectangular source polygon (5 points, closed)
    ps = np.array(
        [
            srcpos,
            srcpos + srcparam1[:3],
            srcpos + srcparam1[:3] + srcparam2[:3],
            srcpos + srcparam2[:3],
            srcpos,
        ]
    )

    node = cfg["node"]
    face = cfg["face"]  # 1-based

    iscolimated = cfg.get("iscolimated", True)
    nodeid = None

    if iscolimated:
        # Create sunk plane using qmeshcut
        sinkplane = np.zeros(4)
        sinkplane[:3] = srcdir
        sinkplane[3] = -np.dot(srcdir, srcpos + srcdir * z0)

        elem = cfg["elem"][:, :4].astype(int)  # 1-based
        nodevals = np.zeros(node.shape[0])

        cutpos, cutvalue, facedata, elemid, nodeid = i2m.qmeshcut(
            elem, node, nodevals, sinkplane
        )

        pnode = cutpos
        # facedata: when col[2] == col[3], it's a triangle; otherwise quad
        tri_mask = facedata[:, 2] == facedata[:, 3]
        pface = facedata[tri_mask, :3].astype(int)  # 1-based

        quad_idx = np.where(~tri_mask)[0]
        if len(quad_idx) > 0:
            quad_tri1 = facedata[quad_idx][:, [0, 1, 2]].astype(int)
            quad_tri2 = facedata[quad_idx][:, [0, 2, 3]].astype(int)
            pface = np.vstack([pface, quad_tri1, quad_tri2])

        parea = i2m.elemvolume(pnode, pface)  # 1-based face
        used_sinkplane = True
    else:
        pnode = node
        pface = face
        parea = cfg["area"]
        used_sinkplane = False

    # Compute face centroids using iso2mesh (1-based)
    c0 = i2m.meshcentroid(pnode, pface)

    # Rotate to align srcdir with z-axis using iso2mesh
    all_pts = np.vstack([c0, ps])
    newnode = i2m.rotatevec3d(all_pts, srcdir[:3])

    srcpoly = newnode[-5:, :2]  # Last 5 points are polygon
    centroids_2d = newnode[:-5, :2]

    # Test which centroids are inside the source polygon
    isin = _inpolygon(
        centroids_2d[:, 0], centroids_2d[:, 1], srcpoly[:, 0], srcpoly[:, 1]
    )
    idx = np.where(isin)[0]

    if len(idx) == 0:
        raise ValueError("Source direction does not intersect with the domain")

    # Check face orientations - convert to 0-based for indexing
    pface_0 = pface - 1
    AB = pnode[pface_0[idx, 1], :] - pnode[pface_0[idx, 0], :]
    AC = pnode[pface_0[idx, 2], :] - pnode[pface_0[idx, 0], :]
    N = np.cross(AB, AC)

    dir_dot = np.sum(N * srcdir, axis=1)

    if used_sinkplane:
        dir_dot[dir_dot > 0] = -dir_dot[dir_dot > 0]

    if np.all(dir_dot >= 0):
        raise ValueError("Please reorient the surface triangles")

    valid_mask = dir_dot < 0
    valid_idx = idx[valid_mask]

    # Initialize boundary condition array
    srcbc = np.zeros((1, len(pface)))
    srcbc[0, valid_idx] = 1.0

    # Compute normalized coordinates for pattern lookup
    pbc = centroids_2d[valid_idx, :]
    dp = pbc - srcpoly[0, :]
    dx = srcpoly[1, :] - srcpoly[0, :]
    dy = srcpoly[3, :] - srcpoly[0, :]
    nx = dx / np.linalg.norm(dx)
    ny = dy / np.linalg.norm(dy)

    bary = np.column_stack(
        [
            np.sum(dp * nx, axis=1) / np.linalg.norm(dx),
            np.sum(dp * ny, axis=1) / np.linalg.norm(dy),
        ]
    )
    bary = np.clip(bary, 0, 1 - 1e-6)

    patsize = 1

    if srcpattern is not None and srctype == "pattern":
        if srcpattern.ndim == 2:
            srcpattern = srcpattern[:, :, np.newaxis]

        pdim = srcpattern.shape
        patsize = pdim[2] if len(pdim) > 2 else 1

        srcbc = np.zeros((patsize, len(pface)))

        for i in range(patsize):
            pat = srcpattern[:, :, i] if patsize > 1 else srcpattern[:, :, 0]
            ix = np.clip((bary[:, 0] * pdim[1]).astype(int), 0, pdim[1] - 1)
            iy = np.clip((bary[:, 1] * pdim[0]).astype(int), 0, pdim[0] - 1)
            srcbc[i, valid_idx] = pat[iy, ix]

    elif srctype == "fourier":
        kx = int(srcparam1[3])
        ky = int(srcparam2[3])
        phi0 = (srcparam1[3] - kx) * 2 * np.pi
        M = 1 - (srcparam2[3] - ky)

        patsize = kx * ky
        srcbc = np.zeros((patsize, len(pface)))

        for i in range(kx):
            for j in range(ky):
                pattern_idx = i * ky + j
                srcbc[pattern_idx, valid_idx] = 0.5 * (
                    1 + M * np.cos((i * bary[:, 0] + j * bary[:, 1]) * 2 * np.pi + phi0)
                )

    return srcbc, patsize, pface, parea, pnode, nodeid, used_sinkplane


def _apply_bc_weighting(
    cfg: dict,
    srcbc: np.ndarray,
    Reff: float,
    srcweight: Optional[np.ndarray],
    pface: np.ndarray,
    parea: np.ndarray,
    pnode: np.ndarray,
    nodeid: Optional[np.ndarray],
    used_sinkplane: bool,
) -> np.ndarray:
    """Apply boundary condition weighting to convert flux to nodal values."""

    nn = cfg["node"].shape[0]
    npattern = srcbc.shape[0]

    # Boundary condition coefficient: 1/18 = 1/2 * 1/9
    Adiagbc = parea * ((1 - Reff) / (18 * (1 + Reff)))
    Adiagbc_weighted = Adiagbc[:, np.newaxis] * srcbc.T  # (Nface, Npattern)

    rhs = np.zeros((nn, npattern))
    pface_0 = pface - 1  # Convert to 0-based

    if nodeid is not None and used_sinkplane:
        # nodeid from qmeshcut: [node1_idx, node2_idx, weight]
        # node indices are 1-based, weight SHOULD be in [0,1]
        # but qmeshcut bug adds +1, so weights are in [1,2] - subtract 1 to fix
        nodeweight = nodeid[:, 2] - 1.0 if nodeid.shape[1] > 2 else np.ones(len(nodeid))
        node_ids = nodeid[:, :2].astype(int) - 1  # Convert to 0-based

        for i in range(npattern):
            for j in range(3):
                face_node_idx = pface_0[:, j]
                face_nodes_1 = node_ids[face_node_idx, 0]
                face_nodes_2 = node_ids[face_node_idx, 1]
                weights_1 = nodeweight[face_node_idx]
                weights_2 = 1 - weights_1

                np.add.at(rhs[:, i], face_nodes_1, Adiagbc_weighted[:, i] * weights_1)
                np.add.at(rhs[:, i], face_nodes_2, Adiagbc_weighted[:, i] * weights_2)
    else:
        for i in range(npattern):
            for j in range(3):
                np.add.at(rhs[:, i], pface_0[:, j], Adiagbc_weighted[:, i])

    # Normalize each pattern
    for i in range(npattern):
        wsrc = 1.0
        if srcweight is not None:
            if isinstance(srcweight, (int, float)):
                wsrc = srcweight
            elif len(srcweight) == npattern:
                wsrc = srcweight[i]

        norm = np.sum(np.abs(rhs[:, i]))
        if norm > 0:
            rhs[:, i] = rhs[:, i] * (wsrc / norm)

    return rhs.T  # Return (Npattern, Nn)


def _lineseg2rhs(linesegs: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Build per-source RHS rows by tracing each line segment through the tet mesh.

    Each row is the per-node line integral of the linear basis along the
    segment. For Helmholtz (MWT), scales by -j*omega*mu0 to account for the
    -j*omega*mu*J source term in the wave equation.

    Returns an (Ns x Nn) ndarray (one row per line segment, complex if MWT).
    """
    if not HAS_ISO2MESH:
        raise ImportError("iso2mesh is required for line-source tracing")

    ns = linesegs.shape[0]
    nn = cfg["node"].shape[0]
    ishelmholtz = isinstance(cfg.get("bulk"), dict) and (
        "epsilon" in cfg["bulk"] or "sigma" in cfg["bulk"]
    )
    dtype = complex if ishelmholtz else float

    widesrc = np.zeros((ns, nn), dtype=dtype)
    for s in range(ns):
        widesrc[s, :] = _traceline(linesegs[s, :3], linesegs[s, 3:6], cfg)

    if "srcweight" in cfg and np.atleast_1d(cfg["srcweight"]).size == ns:
        widesrc = widesrc * np.atleast_1d(cfg["srcweight"]).reshape(-1, 1)

    if ishelmholtz:
        if isinstance(cfg.get("omega"), dict):
            omega = list(cfg["omega"].values())[0]
        else:
            omega = cfg.get("omega", 0)
        mu0_mm = 4.0 * np.pi * 1e-10
        widesrc = -1j * omega * mu0_mm * widesrc

    return widesrc.T  # store as (Nn x Ns) per redbirdpy convention


def _traceline(p1: np.ndarray, p2: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Trace a single line segment through the tet mesh and accumulate
    (bary_in + bary_out)/2 * dl per tet onto a per-node line integral.
    """
    nn = cfg["node"].shape[0]
    rowvec = np.zeros(nn, dtype=float)

    p1 = np.asarray(p1, dtype=float).flatten()
    p2 = np.asarray(p2, dtype=float).flatten()
    seglen = np.linalg.norm(p2 - p1)
    if seglen < 1e-12:
        return rowvec
    dirvec = (p2 - p1) / seglen

    # find starting tet via tsearchn (1-based loc)
    elem = cfg["elem"][:, :4].astype(int)
    estart, barystart = i2m.tsearchn(cfg["node"][:, :3], elem, p1.reshape(1, 3))
    if np.isnan(estart[0]):
        return rowvec

    facenb = cfg["facenb"]  # (Ne x 4), 0 = boundary
    face_local = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])  # 1-based local

    e = int(estart[0])  # 1-based
    pcur = p1.copy()
    barycur = np.atleast_2d(barystart)[0].copy()
    lrem = seglen

    max_iter = 100 * elem.shape[0]
    for _ in range(max_iter):
        if lrem <= 1e-12 or e <= 0:
            break
        elemnodes = elem[e - 1, :4]  # 1-based global node indices
        face_global = elemnodes[face_local - 1]  # (4 x 3) of 1-based node ids

        tt, uu, vv, _ = i2m.raytrace(pcur, dirvec, cfg["node"][:, :3], face_global)
        finite = np.isfinite(tt)
        valid = (
            finite
            & (tt > 1e-10)
            & (uu >= -1e-10)
            & (vv >= -1e-10)
            & (uu + vv <= 1.0 + 1e-10)
        )
        if not np.any(valid):
            break

        valid_idx = np.where(valid)[0]
        kmin = valid_idx[np.argmin(tt[valid_idx])]
        tmin = tt[kmin]

        if tmin >= lrem - 1e-12:
            # segment ends inside this tet
            pend = pcur + lrem * dirvec
            verts = cfg["node"][elemnodes - 1, :3].T  # (3 x 4)
            mat = np.vstack([verts, np.ones((1, 4))])  # (4 x 4)
            rhs_solve = np.append(pend, 1.0)
            baryend = np.linalg.solve(mat, rhs_solve)
            contrib = 0.5 * (barycur + baryend) * lrem
            rowvec[elemnodes - 1] += contrib
            break

        # exit through face kmin
        u_e = uu[kmin]
        v_e = vv[kmin]
        fnodes_local = face_local[kmin, :] - 1  # 0..3 within the tet
        baryexit = np.zeros(4, dtype=float)
        baryexit[fnodes_local[0]] = 1.0 - u_e - v_e
        baryexit[fnodes_local[1]] = u_e
        baryexit[fnodes_local[2]] = v_e

        contrib = 0.5 * (barycur + baryexit) * tmin
        rowvec[elemnodes - 1] += contrib

        e_next = int(facenb[e - 1, kmin])
        pcur = pcur + tmin * dirvec
        barycur = baryexit
        lrem = lrem - tmin
        e = e_next

    return rowvec


def _inpolygon(
    x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    """
    Test if points (x, y) are inside polygon defined by (px, py).

    Uses matplotlib Path if available (faster), otherwise ray casting.
    """
    points = np.column_stack([x, y])
    polygon = np.column_stack([px, py])

    if HAS_MPL_PATH:
        # Use matplotlib's optimized path contains
        path = MplPath(polygon)
        return path.contains_points(points)

    # Fallback: vectorized ray casting
    n = len(px) - 1  # Last point same as first
    inside = np.zeros(len(x), dtype=bool)

    for i in range(n):
        x1, y1 = px[i], py[i]
        x2, y2 = px[i + 1], py[i + 1]

        cond1 = ((y1 <= y) & (y < y2)) | ((y2 <= y) & (y < y1))

        if np.any(cond1):
            x_intersect = (x2 - x1) * (y[cond1] - y1) / (y2 - y1 + 1e-30) + x1
            inside[cond1] ^= x[cond1] < x_intersect

    return inside


# ============== Source/Detector Mapping Functions ==============


def sdmap(cfg: dict, maxdist: float = np.inf, **kwargs) -> Union[np.ndarray, dict]:
    """
    Create source-detector mapping table.

    Returns
    -------
    sd : ndarray or dict
        Mapping table with columns [src_col, det_col, active_flag, mode]
        - src_col: Column index in phi matrix for this source (0-based)
        - det_col: Column index in phi matrix for this detector (0-based)
        - active_flag: 1 if pair is active, 0 otherwise
        - mode: Measurement mode

    Column layout in RHS/phi:
        [0:srcnum] = point sources
        [srcnum:srcnum+wfsrcnum] = wide-field sources
        [srcnum+wfsrcnum:srcnum+wfsrcnum+detnum] = point detectors
        [srcnum+wfsrcnum+detnum:end] = wide-field detectors
    """
    # Get counts
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

    # widesrc/widedet stored as (Nn x Npattern)
    wfsrcnum = 0
    if "widesrc" in cfg and cfg["widesrc"] is not None and cfg["widesrc"].size > 0:
        wfsrcnum = cfg["widesrc"].shape[1]

    wfdetnum = 0
    if "widedet" in cfg and cfg["widedet"] is not None and cfg["widedet"].size > 0:
        wfdetnum = cfg["widedet"].shape[1]

    if (srcnum + wfsrcnum) == 0 or (detnum + wfdetnum) == 0:
        raise ValueError("Must define at least one source and detector")

    badsrc = kwargs.get("excludesrc", cfg.get("excludesrc", []))
    baddet = kwargs.get("excludedet", cfg.get("excludedet", []))

    # Good point sources/detectors (0-based indices)
    goodsrc = sorted(set(range(srcnum)) - set(badsrc))
    gooddet = sorted(set(range(detnum)) - set(baddet))

    # Wide-field source/detector indices (offset by point source/detector count)
    goodwfsrc = list(range(srcnum, srcnum + wfsrcnum))
    goodwfdet = list(range(detnum, detnum + wfdetnum))

    # Detector column offset in RHS matrix
    det_offset = srcnum + wfsrcnum

    if isinstance(cfg.get("prop"), dict):
        wavelengths = list(cfg["prop"].keys())
        sd = {}

        for wv in wavelengths:
            # All source indices (point + wide)
            all_src = goodsrc + goodwfsrc
            # All detector indices (point + wide), offset for RHS column
            all_det = [det_offset + d for d in gooddet] + [
                det_offset + d for d in goodwfdet
            ]

            ss, dd = np.meshgrid(all_src, all_det)
            sdwv = np.column_stack([ss.flatten(), dd.flatten()])

            active = np.ones(len(sdwv))

            # Mark bad pairs as inactive (only for point sources/detectors)
            for i in range(len(sdwv)):
                si = int(sdwv[i, 0])
                di = int(sdwv[i, 1]) - det_offset
                if si < srcnum and di < detnum:
                    if si in badsrc or di in baddet:
                        active[i] = 0

            # Filter by max distance if specified (only for point sources/detectors)
            if maxdist < np.inf and srcnum > 0 and detnum > 0:
                dist = getdistance(cfg["srcpos"], cfg["detpos"], badsrc, baddet)
                for i in range(len(sdwv)):
                    si = int(sdwv[i, 0])
                    di = int(sdwv[i, 1]) - det_offset
                    if si < srcnum and di < detnum:
                        if dist[di, si] >= maxdist:
                            active[i] = 0

            sdwv = np.column_stack([sdwv, active, np.ones(len(sdwv))])
            sd[wv] = sdwv

        return sd
    else:
        # Single wavelength
        all_src = goodsrc + goodwfsrc
        all_det = [det_offset + d for d in gooddet] + [
            det_offset + d for d in goodwfdet
        ]

        ss, dd = np.meshgrid(all_src, all_det)
        sd = np.column_stack([ss.flatten(), dd.flatten()])

        if maxdist < np.inf and srcnum > 0 and detnum > 0:
            dist = getdistance(cfg["srcpos"], cfg["detpos"], badsrc, baddet)
            active = []
            for s, d in sd:
                si = int(s)
                di = int(d) - det_offset
                if si < srcnum and di < detnum:
                    active.append(1.0 if dist[di, si] < maxdist else 0.0)
                else:
                    active.append(1.0)  # Wide-field always active
            sd = np.column_stack([sd, np.array(active)])
        else:
            sd = np.column_stack([sd, np.ones(len(sd))])

        return sd


def getoptodes(
    cfg: dict, wv: str = ""
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get combined optode positions with inward displacement by 1/mu_tr."""
    ltr = getltr(cfg, wv)

    pointsrc = None
    pointdet = None
    widesrc = cfg.get("widesrc", None)
    widedet = cfg.get("widedet", None)

    if "srcpos" in cfg and cfg["srcpos"] is not None and cfg["srcpos"].size > 0:
        srcdir = cfg["srcdir"]
        if srcdir.shape[0] == 1:
            srcdir = np.tile(srcdir, (cfg["srcpos"].shape[0], 1))
        pointsrc = cfg["srcpos"] + srcdir * ltr

    if "detpos" in cfg and cfg["detpos"] is not None and cfg["detpos"].size > 0:
        detdir = cfg.get("detdir", cfg["srcdir"])
        if detdir.shape[0] == 1:
            detdir = np.tile(detdir, (cfg["detpos"].shape[0], 1))
        pointdet = cfg["detpos"] + detdir * ltr

    return pointsrc, pointdet, widesrc, widedet


def getdistance(
    srcpos: np.ndarray,
    detpos: np.ndarray,
    badsrc: List[int] = None,
    baddet: List[int] = None,
    widesrc: np.ndarray = None,
    widedet: np.ndarray = None,
) -> np.ndarray:
    """Calculate source-detector distances. Returns (Ndet x Nsrc) matrix."""
    badsrc = badsrc or []
    baddet = baddet or []

    srcnum = srcpos.shape[0]
    detnum = detpos.shape[0]
    widesrcnum = widesrc.shape[0] if widesrc is not None else 0
    widedetnum = widedet.shape[0] if widedet is not None else 0

    total_src = srcnum + widesrcnum
    total_det = detnum + widedetnum

    dist = np.full((total_det, total_src), np.inf)

    goodsrc = sorted(set(range(srcnum)) - set(badsrc))
    gooddet = sorted(set(range(detnum)) - set(baddet))

    if len(goodsrc) > 0 and len(gooddet) > 0:
        src_good = srcpos[goodsrc, :3]
        det_good = detpos[gooddet, :3]

        diff = src_good[np.newaxis, :, :] - det_good[:, np.newaxis, :]
        d = np.sqrt(np.sum(diff**2, axis=2))

        dist[np.ix_(gooddet, goodsrc)] = d

    return dist


def getdetdir(cfg: dict) -> np.ndarray:
    """Estimate inward surface-normal directions at each detector position.

    Port of redbird-m/matlab/rbgetdetdir.m. Used by the Monte Carlo (mmclab)
    forward path in `runforward` when a Jacobian is requested and the user
    didn't supply ``cfg["detdir"]``: mmc requires a (Nd, 4) inward-normal +
    focal-length array to build detector-as-adjoint sources.

    For each detector position, the nearest surface-face center is located
    and the face normal is oriented toward the mesh centroid (negated to
    produce the inward direction). This works for convex or quasi-convex
    domains; for highly non-convex shapes the user should supply
    ``cfg["detdir"]`` explicitly.

    Parameters
    ----------
    cfg : dict
        Must contain ``cfg["node"]`` (Nn x 3) and ``cfg["elem"]`` (Ne x 4 or
        5, 1-based) and ``cfg["detpos"]`` (Nd x {3,4}). If ``cfg["face"]``
        is absent, it is built via ``iso2mesh.volface`` (or the local
        fallback when iso2mesh isn't available).

    Returns
    -------
    detdir : ndarray
        (Nd, 4) array. Columns 0-2 are unit inward-normal vectors; column 3
        is the focal-length placeholder (0).
    """
    for key in ("node", "elem", "detpos"):
        if key not in cfg or cfg[key] is None:
            raise ValueError(f"getdetdir requires cfg['{key}']")

    node = np.asarray(cfg["node"], dtype=float)
    elem = np.asarray(cfg["elem"], dtype=int)
    detpos = np.asarray(cfg["detpos"], dtype=float)

    face = cfg.get("face")

    if face is None or len(face) == 0:
        try:
            face_result = i2m.volface(elem[:, :4])
            face = face_result[0] if isinstance(face_result, tuple) else face_result
        except (NameError, AttributeError):
            face = _volface_fallback(elem[:, :4])

    face = np.asarray(face, dtype=int)
    # face is 1-based; convert to 0-based for indexing
    face0 = face[:, :3] - 1

    ab = node[face0[:, 1], :3] - node[face0[:, 0], :3]
    ac = node[face0[:, 2], :3] - node[face0[:, 0], :3]
    nrm = np.cross(ab, ac, axis=1)
    nlen = np.sqrt(np.sum(nrm * nrm, axis=1))
    # avoid div-by-zero on degenerate faces; those normals stay zero
    nlen_safe = np.where(nlen > 0, nlen, 1.0)
    facenrm = nrm / nlen_safe[:, None]

    facectr = (
        node[face0[:, 0], :3] + node[face0[:, 1], :3] + node[face0[:, 2], :3]
    ) / 3.0

    # orient each face normal OUTWARD (away from the mesh centroid), then
    # negate to get the inward direction used by mmc as the adjoint source
    # emission direction
    meshctr = np.mean(node[:, :3], axis=0)
    out_proj = np.sum(facenrm * (facectr - meshctr), axis=1)
    flipsign = np.sign(out_proj)
    flipsign[flipsign == 0] = 1.0
    facenrm = facenrm * flipsign[:, None]

    detnum = detpos.shape[0]
    detdir = np.zeros((detnum, 4), dtype=float)

    for d in range(detnum):
        delta = facectr - detpos[d, :3]
        d2 = np.sum(delta * delta, axis=1)
        fid = int(np.argmin(d2))
        detdir[d, :3] = -facenrm[fid]

    return detdir


def getcfg(cfgs: dict, key) -> dict:
    """Extract a single-wavelength cfg from a multi-spectral cfg dict.

    Port of redbird-m/matlab/rbgetcfg.m.  Walks the top-level cfg keys
    and, for any value that is itself a dict keyed by wavelength,
    returns the value for ``key``.  Non-dict values pass through
    unchanged.  When the input cfg contains no multi-spectral fields,
    the return is structurally identical to the input.

    Parameters
    ----------
    cfgs : dict
        Multi-spectral simulation configuration.  Any field may be
        either a plain value or a dict keyed by wavelength.
    key : str or numeric
        Wavelength selector.  Converted to ``str`` for dict lookup.

    Returns
    -------
    cfgkey : dict
        Single-wavelength view of ``cfgs``.
    """
    key_str = str(key)
    out = {}
    for k, v in cfgs.items():
        if isinstance(v, dict) and key_str in v:
            out[k] = v[key_str]
        else:
            out[k] = v
    return out


def getltr(cfg: dict, wv: str = "") -> float:
    """Calculate transport mean free path l_tr = 1/(mua + musp)."""
    from . import property as prop_module

    bkprop = prop_module.getbulk(cfg)

    if isinstance(bkprop, dict):
        if not wv:
            wv = list(bkprop.keys())[0]
        bkprop = bkprop[wv]

    mua = bkprop[0]
    musp = bkprop[1] * (1 - bkprop[2])

    return 1.0 / (mua + musp)


def getreff(n_in: float, n_out: float = 1.0) -> float:
    """Calculate effective reflection coefficient (Haskell 1994)."""
    if n_in <= n_out:
        return 0.0

    oc = np.arcsin(n_out / n_in)
    ostep = np.pi / 2000

    o = np.arange(0, oc, ostep)

    cosop = np.sqrt(1 - (n_in * np.sin(o)) ** 2)
    coso = np.cos(o)

    r_fres = 0.5 * ((n_in * cosop - n_out * coso) / (n_in * cosop + n_out * coso)) ** 2
    r_fres += 0.5 * ((n_in * coso - n_out * cosop) / (n_in * coso + n_out * cosop)) ** 2

    o_full = np.arange(0, np.pi / 2, ostep)
    r_fres_full = np.ones(len(o_full))
    r_fres_full[: len(r_fres)] = r_fres

    coso_full = np.cos(o_full)

    r_phi = 2 * np.sum(np.sin(o_full) * coso_full * r_fres_full) * ostep
    r_j = 3 * np.sum(np.sin(o_full) * coso_full**2 * r_fres_full) * ostep

    return (r_phi + r_j) / (2 - r_phi + r_j)


# ============== Data Manipulation Functions ==============


def elem2node(elem: np.ndarray, elemval: np.ndarray, nodelen: int = None) -> np.ndarray:
    """Interpolate element-based values to nodes."""
    if isinstance(elem, dict):
        nodelen = elem["node"].shape[0]
        elem = elem["elem"]

    elem_0 = elem[:, :4].astype(int) - 1
    nval = elemval.shape[1] if elemval.ndim > 1 else 1

    if elemval.ndim == 1:
        elemval = elemval[:, np.newaxis]

    nodeval = np.zeros((nodelen, nval))

    for j in range(4):
        np.add.at(nodeval, elem_0[:, j], elemval)

    nodeval *= 0.25

    return nodeval.squeeze()


def addnoise(
    data: np.ndarray,
    snrshot: float,
    snrthermal: float = np.inf,
    randseed: int = 123456789,
) -> np.ndarray:
    """Add simulated shot and thermal noise to data."""
    np.random.seed(randseed)

    if np.isinf(snrshot) and np.isinf(snrthermal):
        warnings.warn("No noise added")
        return data.copy()

    datanorm = np.abs(data)
    max_amp = np.max(datanorm)

    sigma_shot = 10 ** (-np.real(snrshot) / 20)
    sigma_thermal = max_amp * 10 ** (-np.real(snrthermal) / 20)

    if np.isreal(data).all():
        newdata = (
            data + np.sqrt(np.abs(data)) * np.random.randn(*data.shape) * sigma_shot
        )
        newdata += np.random.randn(*data.shape) * sigma_thermal
    else:
        sigma_shot_phase = 10 ** (-np.imag(snrshot) / 20)
        sigma_thermal_phase = 10 ** (-np.imag(snrthermal) / 20)

        amp_shot = np.random.randn(*data.shape) * sigma_shot
        phase_shot = np.random.randn(*data.shape) * sigma_shot_phase * 2 * np.pi
        amp_thermal = np.random.randn(*data.shape) * sigma_thermal
        phase_thermal = np.random.randn(*data.shape) * sigma_thermal_phase * 2 * np.pi

        shot_noise = np.sqrt(np.abs(data)) * (amp_shot * np.exp(1j * phase_shot))
        thermal_noise = amp_thermal * np.exp(1j * phase_thermal)

        newdata = data + shot_noise + thermal_noise

    return newdata


def meshinterp(fromval, elemid, elembary, fromelem, toval=None):
    """Interpolate nodal values from source mesh to target mesh."""

    if fromval.ndim == 1:
        fromval = fromval[:, np.newaxis]

    elem_0 = fromelem[:, :4].astype(int) - 1
    npts = len(elemid)
    ncol = fromval.shape[1]

    if toval is None:
        newval = np.zeros((npts, ncol))
    else:
        newval = toval.copy() if toval.ndim > 1 else toval[:, np.newaxis].copy()

    valid = ~np.isnan(elemid)
    valid_idx = np.where(valid)[0]
    valid_eid = elemid[valid].astype(int) - 1
    valid_bary = elembary[valid]

    node_ids = elem_0[valid_eid]
    vals_at_nodes = fromval[node_ids]
    interp_vals = np.sum(vals_at_nodes * valid_bary[:, :, np.newaxis], axis=1)

    newval[valid_idx] = interp_vals

    return newval if newval.shape[1] > 1 else newval.squeeze()


# ============== Private Helper Functions ==============


def _nodevolume(node: np.ndarray, elem: np.ndarray, evol: np.ndarray) -> np.ndarray:
    """Compute nodal volumes (1/4 of connected element volumes)."""
    elem_0 = elem[:, :4].astype(int) - 1
    nn = node.shape[0]

    nvol = np.zeros(nn)

    for j in range(4):
        np.add.at(nvol, elem_0[:, j], evol)

    nvol *= 0.25

    return nvol


def _femnz(elem: np.ndarray, nn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get sparse matrix non-zero indices for FEM assembly."""
    elem_0 = elem[:, :4].astype(int) - 1

    conn = [set() for _ in range(nn)]

    for e in elem_0:
        for i in range(4):
            for j in range(4):
                if i != j:
                    conn[e[i]].add(e[j])

    connnum = np.array([len(c) for c in conn])

    rows = []
    cols = []
    for i in range(nn):
        for j in conn[i]:
            rows.append(i)
            cols.append(j)

    return np.array(rows), np.array(cols), connnum


# ============== Fallback implementations ==============


def _meshreorient_fallback(node: np.ndarray, elem: np.ndarray) -> np.ndarray:
    """Reorient elements to have positive volume. elem is 1-based."""
    elem = elem.copy()
    elem_0 = elem[:, :4].astype(int) - 1

    for i in range(elem.shape[0]):
        n = node[elem_0[i, :], :]

        v1 = n[1] - n[0]
        v2 = n[2] - n[0]
        v3 = n[3] - n[0]
        vol = np.dot(np.cross(v1, v2), v3)

        if vol < 0:
            elem[i, [0, 1]] = elem[i, [1, 0]]

    return elem


def _volface_fallback(elem: np.ndarray) -> np.ndarray:
    """Extract surface triangles from tetrahedral mesh. elem is 1-based."""
    elem_0 = elem[:, :4].astype(int) - 1

    faces_0 = np.vstack(
        [
            elem_0[:, [0, 2, 1]],
            elem_0[:, [0, 1, 3]],
            elem_0[:, [0, 3, 2]],
            elem_0[:, [1, 2, 3]],
        ]
    )

    faces_sorted = np.sort(faces_0, axis=1)

    _, indices, counts = np.unique(
        faces_sorted, axis=0, return_index=True, return_counts=True
    )

    boundary_idx = indices[counts == 1]

    return faces_0[boundary_idx] + 1


def _elemvolume_fallback(node: np.ndarray, elem: np.ndarray) -> np.ndarray:
    """Compute element volumes or areas. elem is 1-based."""
    elem_0 = elem.astype(int) - 1

    if elem.shape[1] >= 4:
        n0 = node[elem_0[:, 0], :]
        v1 = node[elem_0[:, 1], :] - n0
        v2 = node[elem_0[:, 2], :] - n0
        v3 = node[elem_0[:, 3], :] - n0

        vol = np.abs(np.sum(np.cross(v1, v2) * v3, axis=1)) / 6.0
    elif elem.shape[1] == 3:
        v1 = node[elem_0[:, 1], :] - node[elem_0[:, 0], :]
        v2 = node[elem_0[:, 2], :] - node[elem_0[:, 0], :]

        vol = 0.5 * np.sqrt(np.sum(np.cross(v1, v2) ** 2, axis=1))
    else:
        raise ValueError(f"Unsupported element type with {elem.shape[1]} nodes")

    return vol


def forcearray(cfg: dict, keys: List[str]) -> dict:
    """Convert list-valued cfg entries to numpy arrays."""
    for key in keys:
        if key in cfg and isinstance(cfg[key], list):
            cfg[key] = np.array(cfg[key])
    return cfg
