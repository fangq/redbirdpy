"""
Redbird Property Module - Optical property management for DOT/NIRS.

INDEX CONVENTION: All mesh indices (elem, face) stored in cfg are 1-based
to match MATLAB/iso2mesh. This module converts to 0-based only when
indexing numpy arrays.

Functions:
    extinction: Get molar extinction coefficients for chromophores
    updateprop: Update optical properties from physiological parameters
    getbulk: Get bulk/background optical properties
    musp2sasp: Convert mus' to scattering amplitude/power
    setmesh: Associate new mesh with simulation structure
"""

import numpy as np
from scipy import interpolate
from typing import Dict, Tuple, Optional, Union, List, Any


def extinction(
    wavelengths: Union[List[str], List[float], np.ndarray],
    chromophores: Union[str, List[str]],
    **interp_opts,
) -> Tuple[np.ndarray, dict]:
    """
    Get molar extinction coefficients for chromophores.

    Data compiled by Scott Prahl from https://omlc.org/spectra/hemoglobin/

    Parameters
    ----------
    wavelengths : list or ndarray
        Wavelengths in nm (as strings or numbers)
    chromophores : str or list
        Chromophore names: 'hbo', 'hbr', 'water', 'lipids', 'aa3'
    **interp_opts : dict
        Options passed to scipy.interpolate.interp1d

    Returns
    -------
    extin : ndarray
        Extinction coefficients (Nwv x Nchrome)
        Units: 1/(mm*uM) for hemoglobin, 1/mm for water/lipids
    chrome : dict
        Full lookup tables for each chromophore
    """
    chrome = _get_chromophore_data()

    # Convert wavelengths to float array
    if isinstance(wavelengths, (list, tuple)):
        wavelengths = [float(w) if isinstance(w, str) else w for w in wavelengths]
    wavelengths = np.atleast_1d(wavelengths).astype(float)

    # Handle single chromophore
    if isinstance(chromophores, str):
        chromophores = [chromophores]

    extin = np.zeros((len(wavelengths), len(chromophores)))

    for j, chrom in enumerate(chromophores):
        chrom_lower = chrom.lower()
        if chrom_lower not in chrome:
            raise ValueError(
                f"Unknown chromophore: {chrom}. " f"Available: {list(chrome.keys())}"
            )

        spectrum = chrome[chrom_lower]

        # Interpolate to requested wavelengths
        f = interpolate.interp1d(
            spectrum[:, 0],
            spectrum[:, 1],
            kind="linear",
            fill_value="extrapolate",
            **interp_opts,
        )
        extin[:, j] = f(wavelengths)

    return extin, chrome


def updateprop(cfg: dict, wv: str = None) -> Union[np.ndarray, dict]:
    """
    Update optical properties from physiological parameters.

    Converts chromophore concentrations and scattering parameters to
    wavelength-dependent mua and musp.

    Parameters
    ----------
    cfg : dict
        Configuration with:
        - param: dict with 'hbo', 'hbr', 'water', 'lipids', 'scatamp', 'scatpow'
        - prop: template for output format
        - node, elem: mesh data (1-based elem)
    wv : str, optional
        Specific wavelength to update (if None, update all)

    Returns
    -------
    prop : ndarray or dict
        Updated optical properties [mua, musp, g, n]
        Dict keyed by wavelength if multi-wavelength

    Notes
    -----
    mua = sum_i(extin_i * C_i) where C_i is concentration
    musp = scatamp * (lambda_nm)^(-scatpow)
    """
    if "param" not in cfg or not isinstance(cfg.get("prop"), dict):
        return cfg.get("prop")

    wavelengths = list(cfg["prop"].keys()) if wv is None else [wv]
    params = cfg["param"]

    prop_out = {}

    for wavelen in wavelengths:
        # Default tissue composition values
        if "water" not in params:
            params["water"] = 0.23
        if "lipids" not in params:
            params["lipids"] = 0.58

        # Get chromophore types present in params
        types = [t for t in ["hbo", "hbr", "water", "lipids", "aa3"] if t in params]

        if not types:
            raise ValueError(
                "No recognized chromophores in cfg.param. "
                "Expected one or more of: hbo, hbr, water, lipids, aa3"
            )

        # Get extinction coefficients at this wavelength
        extin, _ = extinction(float(wavelen), types)

        # Compute mua as sum of extinction * concentration
        first_param = params[types[0]]
        if np.isscalar(first_param):
            mua = 0.0
        else:
            mua = np.zeros_like(first_param, dtype=float)

        for j, t in enumerate(types):
            mua = mua + extin[0, j] * params[t]

        # Compute musp from scattering amplitude and power
        # musp = scatamp * (wavelength_nm)^(-scatpow)
        if "scatamp" in params and "scatpow" in params:
            musp = params["scatamp"] * (float(wavelen)) ** (-params["scatpow"])
        else:
            musp = None

        # Build property array based on mesh size
        segprop = cfg["prop"][wavelen]
        nn = (
            cfg["node"].shape[0]
            if "node" in cfg
            else (len(mua) if hasattr(mua, "__len__") else 1)
        )
        ne = (
            cfg["elem"].shape[0]
            if "elem" in cfg
            else (len(mua) if hasattr(mua, "__len__") else 1)
        )

        mua_len = len(mua) if hasattr(mua, "__len__") else 1

        if mua_len < min(nn, ne):
            # Label-based properties: mua/musp are per-label
            # segprop[0] is background, segprop[1:] are tissue labels
            new_prop = segprop.copy()
            if hasattr(mua, "__len__"):
                new_prop[1 : mua_len + 1, 0] = mua
                if musp is not None:
                    new_prop[1 : mua_len + 1, 1] = musp
                    new_prop[1 : mua_len + 1, 2] = 0  # g=0 when using musp directly
            else:
                new_prop[1, 0] = mua
                if musp is not None:
                    new_prop[1, 1] = musp
                    new_prop[1, 2] = 0
        else:
            # Node/element based properties
            if musp is not None:
                n_ref = segprop[1, 3] if segprop.shape[0] > 1 else 1.37
                if hasattr(mua, "__len__"):
                    new_prop = np.column_stack(
                        [mua, musp, np.zeros_like(musp), np.full_like(musp, n_ref)]
                    )
                else:
                    new_prop = np.array([[mua, musp, 0, n_ref]])
            else:
                # Keep existing musp, g, n from template
                if hasattr(mua, "__len__"):
                    tile_prop = np.tile(segprop[1, 1:], (len(mua), 1))
                    new_prop = np.column_stack([mua, tile_prop])
                else:
                    new_prop = np.array([[mua] + list(segprop[1, 1:])])

        prop_out[wavelen] = new_prop

    return prop_out if len(wavelengths) > 1 else prop_out[wavelengths[0]]


def getbulk(cfg: dict) -> Union[np.ndarray, dict]:
    """
    Get bulk/background optical properties.

    Returns the optical properties of the outer-most layer that interfaces
    with air (used for boundary condition calculation).

    Parameters
    ----------
    cfg : dict
        Configuration with 'prop', optionally 'bulk', 'seg', 'face'
        elem and face are 1-based indices

    Returns
    -------
    bkprop : ndarray or dict
        [mua, mus, g, n] for the bulk medium
        Dict keyed by wavelength if multi-wavelength

    Notes
    -----
    Priority: cfg.bulk > cfg.prop at surface node > default [0, 0, 0, 1.37]
    """
    bkprop_default = np.array([0.0, 0.0, 0.0, 1.37])

    # If explicit bulk properties provided, use those
    if "bulk" in cfg:
        bulk = cfg["bulk"]
        bkprop = bkprop_default.copy()
        if "mua" in bulk:
            bkprop[0] = bulk["mua"]
        if "dcoeff" in bulk:
            # Convert diffusion coefficient to mus
            bkprop[1] = 1.0 / (3 * bulk["dcoeff"])
            bkprop[2] = 0
        if "musp" in bulk:
            bkprop[1] = bulk["musp"]
            bkprop[2] = 0
        if "g" in bulk:
            bkprop[2] = bulk["g"]
        if "n" in bulk:
            bkprop[3] = bulk["n"]
        return bkprop

    if "prop" not in cfg or cfg["prop"] is None:
        return bkprop_default

    prop = cfg["prop"]

    # Multi-wavelength handling
    if isinstance(prop, dict):
        bkprop = {}
        for wv, p in prop.items():
            bkprop[wv] = _extract_bulk_from_prop(p, cfg)
        return bkprop

    return _extract_bulk_from_prop(prop, cfg)


def _extract_bulk_from_prop(prop: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Extract bulk property from property array.

    Determines property format (label-based vs node/element-based) and
    extracts the appropriate surface property.
    """
    bkprop_default = np.array([0.0, 0.0, 0.0, 1.37])

    nn = cfg["node"].shape[0] if "node" in cfg else prop.shape[0]
    ne = cfg["elem"].shape[0] if "elem" in cfg else prop.shape[0]

    if prop.shape[0] < min(nn, ne):
        # Label-based properties
        if "seg" in cfg:
            seg = cfg["seg"]
            if "face" in cfg and len(seg) == nn:
                # Node-based segmentation, get label at first surface node
                # face is 1-based, convert for indexing
                face_node_0 = cfg["face"][0, 0] - 1  # Convert to 0-based
                label = seg[face_node_0]
            elif "face" in cfg and len(seg) == ne:
                # Element-based segmentation, find element containing face node
                elem_0 = cfg["elem"][:, :4].astype(int) - 1  # Convert to 0-based
                face_node_0 = cfg["face"][0, 0] - 1
                eid = np.where(np.any(elem_0 == face_node_0, axis=1))[0]
                label = seg[eid[0]] if len(eid) > 0 else 0
            else:
                label = seg[0]

            # prop row 0 is background, row label+1 is the tissue
            prop_idx = int(label) + 1
            if prop.shape[0] > prop_idx:
                return prop[prop_idx, :]
            else:
                return prop[1, :] if prop.shape[0] > 1 else bkprop_default
        else:
            # No segmentation, use first tissue label
            return prop[1, :] if prop.shape[0] > 1 else bkprop_default

    elif prop.shape[0] == nn:
        # Node-based properties
        if "face" in cfg:
            face_node_0 = cfg["face"][0, 0] - 1  # Convert to 0-based
            return prop[face_node_0, :]
        return prop[0, :]

    elif prop.shape[0] == ne:
        # Element-based properties
        if "face" in cfg:
            elem_0 = cfg["elem"][:, :4].astype(int) - 1
            face_node_0 = cfg["face"][0, 0] - 1
            eid = np.where(np.any(elem_0 == face_node_0, axis=1))[0]
            if len(eid) > 0:
                return prop[eid[0], :]
        return prop[0, :]

    return bkprop_default


def musp2sasp(musp: np.ndarray, wavelength: np.ndarray) -> Tuple[float, float]:
    """
    Convert mus' at two wavelengths to scattering amplitude and power.

    Uses the relation: musp = sa * (lambda/500nm)^(-sp)

    Parameters
    ----------
    musp : ndarray
        Reduced scattering coefficients at two wavelengths (1/mm)
    wavelength : ndarray
        Wavelengths in nm (length 2)

    Returns
    -------
    sa : float
        Scattering amplitude (musp at 500nm)
    sp : float
        Scattering power (wavelength exponent)
    """
    if len(musp) < 2 or len(wavelength) < 2:
        raise ValueError("Need at least 2 wavelengths to fit scattering parameters")

    lam = wavelength[:2] / 500.0

    # sp = log(musp1/musp2) / log(lam2/lam1)
    sp = np.log(musp[0] / musp[1]) / np.log(lam[1] / lam[0])

    # sa = average of musp / lam^(-sp) at both wavelengths
    sa = 0.5 * (musp[0] / lam[0] ** (-sp) + musp[1] / lam[1] ** (-sp))

    return sa, sp


def setmesh(
    cfg0: dict,
    node: np.ndarray,
    elem: np.ndarray,
    prop: np.ndarray = None,
    propidx: np.ndarray = None,
) -> dict:
    """
    Associate a new mesh with simulation structure.

    Clears derived quantities that need recomputation with the new mesh.

    Parameters
    ----------
    cfg0 : dict
        Original configuration
    node : ndarray
        New node coordinates (Nn x 3)
    elem : ndarray
        New element connectivity (Ne x 4+), 1-based indices
    prop : ndarray, optional
        New optical properties
    propidx : ndarray, optional
        Segmentation labels

    Returns
    -------
    cfg : dict
        Updated configuration with new mesh
    """
    from . import utility

    # Fields that depend on mesh geometry and need recomputation
    clear_fields = [
        "face",
        "area",
        "evol",
        "deldotdel",
        "isreoriented",
        "nvol",
        "cols",
        "rows",
        "idxsum",
        "idxcount",
        "musp0",
        "reff",
    ]

    cfg = {k: v for k, v in cfg0.items() if k not in clear_fields}

    cfg["node"] = node
    cfg["elem"] = elem[:, :4] if elem.shape[1] > 4 else elem

    if prop is not None:
        cfg["prop"] = prop

    if propidx is not None:
        cfg["seg"] = propidx
    elif elem.shape[1] > 4:
        cfg["seg"] = elem[:, 4].astype(int)

    # Prepare mesh (computes face, area, evol, deldotdel, etc.)
    cfg, _ = utility.meshprep(cfg)

    return cfg


# ============== Chromophore Data ==============


def _get_chromophore_data() -> dict:
    """
    Get built-in chromophore extinction coefficient tables.

    Returns dict with keys: 'hbo', 'hbr', 'water', 'lipids', 'aa3'
    Each value is Nx2 array: [wavelength_nm, extinction_coeff]

    Units:
    - HbO2, Hb: 1/(mm*uM) - multiply by concentration in uM to get 1/mm
    - Water, lipids: 1/mm - multiply by volume fraction
    """
    chrome = {}

    # Hemoglobin data (HbO2 and Hb) from Scott Prahl / OMLC
    # Original units: cm-1/M, converted to 1/(mm*uM) via 2.303e-7
    # Wavelength (nm), HbO2 (cm-1/M), Hb (cm-1/M)
    hb_raw = np.array(
        [
            [250, 106112, 112736],
            [260, 116376, 116296],
            [270, 136068, 122880],
            [280, 131936, 118872],
            [290, 104752, 98364],
            [300, 65972, 64440],
            [310, 63352, 59156],
            [320, 78752, 74508],
            [330, 97512, 90856],
            [340, 107884, 108472],
            [350, 106576, 122092],
            [360, 94744, 134940],
            [370, 88176, 139968],
            [380, 109564, 145232],
            [390, 167748, 167780],
            [400, 266232, 223296],
            [410, 466840, 303956],
            [420, 480360, 407560],
            [430, 246072, 528600],
            [440, 102580, 413280],
            [450, 62816, 103292],
            [460, 44480, 23388.8],
            [470, 33209.2, 16156.4],
            [480, 26629.2, 14550],
            [490, 23684.4, 16684],
            [500, 20932.8, 20862],
            [510, 20035.2, 25773.6],
            [520, 24202.4, 31589.6],
            [530, 39956.8, 39036.4],
            [540, 53236, 46592],
            [550, 43016, 53412],
            [560, 32613.2, 53788],
            [570, 44496, 45072],
            [580, 50104, 37020],
            [590, 14400.8, 28324.4],
            [600, 3200, 14677.2],
            [610, 1506, 9443.6],
            [620, 942, 6509.6],
            [630, 610, 5148.8],
            [640, 442, 4345.2],
            [650, 368, 3750.12],
            [660, 319.6, 3226.56],
            [670, 294, 2795.12],
            [680, 277.6, 2407.92],
            [690, 276, 2334.68],
            [700, 290, 1794.28],
            [710, 314, 1540.48],
            [720, 348, 1325.88],
            [730, 390, 1102.2],
            [740, 446, 1115.88],
            [750, 518, 1405.24],
            [760, 586, 1548.52],
            [770, 650, 1311.88],
            [780, 710, 1075.44],
            [790, 756, 890.8],
            [800, 816, 761.72],
            [810, 864, 717.08],
            [820, 916, 693.76],
            [830, 974, 693.04],
            [840, 1022, 692.36],
            [850, 1058, 691.32],
            [860, 1092, 694.32],
            [870, 1128, 705.84],
            [880, 1154, 726.44],
            [890, 1178, 743.6],
            [900, 1198, 761.84],
            [910, 1214, 774.56],
            [920, 1224, 777.36],
            [930, 1222, 763.84],
            [940, 1214, 693.44],
            [950, 1204, 602.24],
            [960, 1186, 525.56],
            [970, 1162, 429.32],
            [980, 1128, 359.656],
            [990, 1080, 283.22],
            [1000, 1024, 206.784],
        ]
    )

    # Convert units: 2.303 (ln to log10) * 1e-4 (cm to mm) * 1e-3 (M to mM) * 1e-3 (mM to uM)
    # = 2.303e-7 converts from 1/(cm*M) to 1/(mm*uM)
    conversion = 2.303e-7
    chrome["hbo"] = np.column_stack([hb_raw[:, 0], hb_raw[:, 1] * conversion])
    chrome["hbr"] = np.column_stack([hb_raw[:, 0], hb_raw[:, 2] * conversion])

    # Water absorption coefficient (1/mm) - multiply by water fraction
    # Data from Hale & Querry 1973, simplified for NIR range
    water_wv = np.array([400, 500, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    water_mua = (
        np.array(
            [
                0.00058,
                0.00025,
                0.0023,
                0.0032,
                0.006,
                0.026,
                0.02,
                0.043,
                0.068,
                0.39,
                0.36,
            ]
        )
        * 0.1
    )  # Convert cm-1 to mm-1
    chrome["water"] = np.column_stack([water_wv, water_mua])

    # Lipids absorption (1/mm) - multiply by lipid fraction
    # Simplified approximation for NIR range
    lipid_wv = np.arange(650, 1000, 10)
    lipid_mua = 0.0005 * np.ones_like(lipid_wv, dtype=float)
    chrome["lipids"] = np.column_stack([lipid_wv, lipid_mua])

    # Cytochrome c oxidase (aa3) - difference spectrum
    # Gaussian-like peak around 830nm (oxidized-reduced difference)
    aa3_wv = np.arange(650, 950, 5)
    aa3_mua = 0.5 * np.exp(-((aa3_wv - 830) ** 2) / (2 * 50**2)) + 0.4
    chrome["aa3"] = np.column_stack([aa3_wv, aa3_mua])

    return chrome


def get_chromophore_table(name: str) -> np.ndarray:
    """
    Get full chromophore lookup table.

    Parameters
    ----------
    name : str
        Chromophore name ('hbo', 'hbr', 'water', 'lipids', 'aa3')

    Returns
    -------
    table : ndarray
        Nx2 array of [wavelength_nm, extinction_coefficient]
    """
    chrome = _get_chromophore_data()

    name = name.lower()
    if name not in chrome:
        raise ValueError(
            f"Unknown chromophore: {name}. " f"Available: {list(chrome.keys())}"
        )

    return chrome[name]
