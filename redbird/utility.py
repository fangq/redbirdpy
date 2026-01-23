"""
Redbird Utility Module - Mesh and data utilities for DOT/NIRS.

INDEX CONVENTION: All mesh indices (elem, face) are 1-based to match
MATLAB/iso2mesh. Conversion to 0-based occurs only when indexing numpy arrays.

This module provides utility functions for mesh preparation, source/detector
handling, data manipulation, and visualization support.
"""

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


def meshprep(cfg: dict) -> Tuple[dict, np.ndarray]:
    """
    Prepare mesh structure with all derived quantities.

    All mesh indices (elem, face) remain 1-based in the returned cfg.
    """
    from . import forward, property as prop_module

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
    if "srcdir" not in cfg:
        raise ValueError("cfg.srcdir is required")

    # Update properties if multi-spectral
    if isinstance(cfg.get("prop"), dict) and "param" in cfg:
        cfg["prop"] = prop_module.updateprop(cfg)

    # Compute effective reflection coefficient
    if "reff" not in cfg or cfg["reff"] is None:
        bkprop = prop_module.getbulk(cfg)
        if isinstance(bkprop, dict):
            cfg["reff"] = {}
            cfg["musp0"] = {}
            for wv, prop in bkprop.items():
                cfg["reff"][wv] = getreff(prop[3], 1.0)
                cfg["musp0"][wv] = prop[1] * (1 - prop[2])
        else:
            cfg["reff"] = getreff(bkprop[3], 1.0)
            cfg["musp0"] = bkprop[1] * (1 - bkprop[2])

    # Compute sparse matrix structure
    if "cols" not in cfg or cfg["cols"] is None:
        cfg["rows"], cfg["cols"], cfg["idxcount"] = _femnz(
            cfg["elem"][:, :4], node.shape[0]
        )

    if "idxsum" not in cfg or cfg["idxsum"] is None:
        cfg["idxsum"] = np.cumsum(cfg["idxcount"])

    # Compute gradient operator
    if "deldotdel" not in cfg or cfg["deldotdel"] is None:
        cfg["deldotdel"], _ = forward.deldotdel(cfg)

    # Set default modulation frequency
    if "omega" not in cfg:
        cfg["omega"] = 0

    # Create source-detector mapping
    sd = sdmap(cfg)

    return cfg, sd


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
    """
    if ("srcpos" not in cfg and "widesrc" not in cfg) or (
        "detpos" not in cfg and "widedet" not in cfg
    ):
        raise ValueError("Must define at least one source and detector")

    srcnum = cfg["srcpos"].shape[0] if "srcpos" in cfg else 0
    detnum = cfg["detpos"].shape[0] if "detpos" in cfg else 0
    widesrcnum = cfg.get("widesrc", np.array([])).shape[0] if "widesrc" in cfg else 0
    widedetnum = cfg.get("widedet", np.array([])).shape[0] if "widedet" in cfg else 0

    badsrc = kwargs.get("excludesrc", cfg.get("excludesrc", []))
    baddet = kwargs.get("excludedet", cfg.get("excludedet", []))

    # 0-based indices for good sources/detectors
    goodsrc = sorted(set(range(srcnum)) - set(badsrc))
    gooddet = sorted(set(range(detnum)) - set(baddet))

    # Multi-wavelength handling
    if isinstance(cfg.get("prop"), dict):
        wavelengths = list(cfg["prop"].keys())
        sd = {}

        for wv in wavelengths:
            # Create mesh grid of src/det pairs (0-based column indices)
            # phi matrix layout: [src0, src1, ..., widesrc0, ..., det0, det1, ..., widedet0, ...]
            all_src = list(range(srcnum + widesrcnum))
            det_offset = srcnum + widesrcnum
            all_det = [det_offset + i for i in range(detnum + widedetnum)]

            ss, dd = np.meshgrid(all_src, all_det)
            sdwv = np.column_stack([ss.flatten(), dd.flatten()])

            # Add active flag column
            active = np.ones(len(sdwv))

            # Mark bad pairs as inactive
            for i in range(len(sdwv)):
                si = int(sdwv[i, 0])
                di = int(sdwv[i, 1]) - det_offset
                if si < srcnum and di < detnum:
                    if si in badsrc or di in baddet:
                        active[i] = 0

            # Filter by max distance if specified
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
        det_offset = srcnum + widesrcnum
        ss, dd = np.meshgrid(goodsrc, [det_offset + d for d in gooddet])
        sd = np.column_stack([ss.flatten(), dd.flatten()])

        if maxdist < np.inf:
            dist = getdistance(cfg["srcpos"], cfg["detpos"], badsrc, baddet)
            active = np.array(
                [dist[d - det_offset, s] < maxdist for s, d in sd]
            ).astype(float)
            sd = np.column_stack([sd, active])
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

    if "srcpos" in cfg and len(cfg["srcpos"]) > 0:
        srcdir = cfg["srcdir"]
        if srcdir.shape[0] == 1:
            srcdir = np.tile(srcdir, (cfg["srcpos"].shape[0], 1))
        pointsrc = cfg["srcpos"] + srcdir * ltr

    if "detpos" in cfg and len(cfg["detpos"]) > 0:
        detdir = cfg.get("detdir", cfg["srcdir"])  # Default to srcdir if not specified
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

        # Vectorized pairwise distance
        diff = src_good[np.newaxis, :, :] - det_good[:, np.newaxis, :]
        d = np.sqrt(np.sum(diff**2, axis=2))

        dist[np.ix_(gooddet, goodsrc)] = d

    return dist


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


def elem2node(elem: np.ndarray, elemval: np.ndarray, nodelen: int = None) -> np.ndarray:
    """Interpolate element-based values to nodes."""
    if isinstance(elem, dict):
        nodelen = elem["node"].shape[0]
        elem = elem["elem"]

    # Convert 1-based elem to 0-based for numpy indexing
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


def meshinterp(
    values: np.ndarray,
    mapid: np.ndarray,
    mapweight: np.ndarray,
    elem: np.ndarray,
    default: np.ndarray = None,
) -> np.ndarray:
    """
    Interpolate values between meshes using barycentric coordinates.

    Parameters
    ----------
    values : ndarray
        Values on source mesh nodes (Nn_source x Nval) or (Nn_source,)
    mapid : ndarray
        Element indices in source mesh for each target point (0-based)
    mapweight : ndarray
        Barycentric coordinates (Ntarget x 4)
    elem : ndarray
        Element connectivity of SOURCE mesh (1-based!)
    default : ndarray, optional
        Default values for points outside mesh

    Returns
    -------
    result : ndarray
        Interpolated values at target points
    """
    # elem is 1-based, convert for numpy indexing
    elem_0 = elem[:, :4].astype(int) - 1
    n_target = len(mapid)

    # Handle 1D vs 2D values
    values_2d = values[:, np.newaxis] if values.ndim == 1 else values
    nval = values_2d.shape[1]

    if default is None:
        result = np.zeros((n_target, nval))
    else:
        default_2d = default[:, np.newaxis] if default.ndim == 1 else default
        result = (
            default_2d.copy()
            if default_2d.shape[0] == n_target
            else np.zeros((n_target, nval))
        )

    for i in range(n_target):
        if not np.isnan(mapid[i]):
            eid = int(mapid[i])  # mapid is 0-based element index

            # Check bounds
            if eid < 0 or eid >= elem_0.shape[0]:
                continue

            # Get node indices for this element
            node_ids = elem_0[eid, :]  # 4 node indices (0-based)

            # Check node indices are valid
            if np.any(node_ids < 0) or np.any(node_ids >= values_2d.shape[0]):
                continue

            # Interpolate using barycentric coordinates
            # values_at_nodes: (4, nval), mapweight[i, :]: (4,)
            result[i, :] = mapweight[i, :] @ values_2d[node_ids, :]

    return result.squeeze() if values.ndim == 1 else result


# ============== Private Helper Functions ==============


def _nodevolume(node: np.ndarray, elem: np.ndarray, evol: np.ndarray) -> np.ndarray:
    """Compute nodal volumes (1/4 of connected element volumes)."""
    # elem is 1-based, convert for numpy
    elem_0 = elem[:, :4].astype(int) - 1
    nn = node.shape[0]

    nvol = np.zeros(nn)

    for j in range(4):
        np.add.at(nvol, elem_0[:, j], evol)

    nvol *= 0.25

    return nvol


def _femnz(elem: np.ndarray, nn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get sparse matrix non-zero indices for FEM assembly."""
    # elem is 1-based, convert for numpy
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


# ============== Fallback implementations (when iso2mesh not available) ==============


def _meshreorient_fallback(node: np.ndarray, elem: np.ndarray) -> np.ndarray:
    """Reorient elements to have positive volume. elem is 1-based."""
    elem = elem.copy()
    # Convert to 0-based for computation
    elem_0 = elem[:, :4].astype(int) - 1

    for i in range(elem.shape[0]):
        n = node[elem_0[i, :], :]

        v1 = n[1] - n[0]
        v2 = n[2] - n[0]
        v3 = n[3] - n[0]
        vol = np.dot(np.cross(v1, v2), v3)

        if vol < 0:
            # Swap in 1-based array
            elem[i, [0, 1]] = elem[i, [1, 0]]

    return elem


def _volface_fallback(elem: np.ndarray) -> np.ndarray:
    """Extract surface triangles from tetrahedral mesh. elem is 1-based, returns 1-based."""
    # Convert to 0-based for computation
    elem_0 = elem[:, :4].astype(int) - 1

    # All faces (using consistent winding)
    faces_0 = np.vstack(
        [
            elem_0[:, [0, 2, 1]],
            elem_0[:, [0, 1, 3]],
            elem_0[:, [0, 3, 2]],
            elem_0[:, [1, 2, 3]],
        ]
    )

    # Sort each face for comparison
    faces_sorted = np.sort(faces_0, axis=1)

    # Find unique faces (appearing only once = boundary)
    _, indices, counts = np.unique(
        faces_sorted, axis=0, return_index=True, return_counts=True
    )

    boundary_idx = indices[counts == 1]

    # Return 1-based
    return faces_0[boundary_idx] + 1


def _elemvolume_fallback(node: np.ndarray, elem: np.ndarray) -> np.ndarray:
    """Compute element volumes (tetrahedra) or areas (triangles). elem is 1-based."""
    # Convert to 0-based for numpy indexing
    elem_0 = elem.astype(int) - 1

    if elem.shape[1] >= 4:
        # Tetrahedron volume
        n0 = node[elem_0[:, 0], :]
        v1 = node[elem_0[:, 1], :] - n0
        v2 = node[elem_0[:, 2], :] - n0
        v3 = node[elem_0[:, 3], :] - n0

        vol = np.abs(np.sum(np.cross(v1, v2) * v3, axis=1)) / 6.0
    elif elem.shape[1] == 3:
        # Triangle area
        v1 = node[elem_0[:, 1], :] - node[elem_0[:, 0], :]
        v2 = node[elem_0[:, 2], :] - node[elem_0[:, 0], :]

        vol = 0.5 * np.sqrt(np.sum(np.cross(v1, v2) ** 2, axis=1))
    else:
        raise ValueError(f"Unsupported element type with {elem.shape[1]} nodes")

    return vol
