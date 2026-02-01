"""
Redbird Analytical Diffusion Models

INDEX CONVENTION: All mesh indices (elem, face) stored in cfg/recon are 1-based
to match MATLAB/iso2mesh. Conversion to 0-based occurs only when indexing numpy
arrays, using local variables named with '_0' suffix.

Functions:
    infinite_cw: CW fluence for infinite homogeneous medium
    semi_infinite_cw: CW fluence for semi-infinite medium
    semi_infinite_cw_flux: CW surface flux (diffuse reflectance) for semi-infinite medium
    infinite_td: Time-domain fluence for infinite medium
    semi_infinite_td: Time-domain fluence for semi-infinite medium
    sphere_infinite: CW/FD (cfg['omega']) fluence for sphere in infinite medium
    sphere_semi_infinite: CW/FD fluence for sphere in semi-infinite medium
    sphere_slab: CW/FD fluence for sphere in slab medium

Special Functions:
    spbesselj: Spherical Bessel function of the first kind
    spbessely: Spherical Bessel function of the second kind (Neumann)
    spbesselh: Spherical Hankel function
    spbesseljprime: Derivative of spherical Bessel function (first kind)
    spbesselyprime: Derivative of spherical Bessel function (second kind)
    spbesselhprime: Derivative of spherical Hankel function
    spharmonic: Spherical harmonic function

References:
    [Fang2010] Fang, "Mesh-based Monte Carlo method using fast ray-tracing"
    [Boas2002] Boas et al., "Scattering of diffuse photon density waves"
    [Haskell1994] Haskell et al., "Boundary conditions for diffusion equation"
    [Kienle1997] Kienle & Patterson, "Improved solutions of diffusion equation"
"""

__all__ = [
    # CW solutions
    "infinite_cw",
    "semi_infinite_cw",
    "semi_infinite_cw_flux",
    # Time-domain solutions
    "infinite_td",
    "semi_infinite_td",
    # Sphere solutions
    "sphere_infinite",
    "sphere_semi_infinite",
    "sphere_slab",
    # Special functions
    "spbesselj",
    "spbessely",
    "spbesselh",
    "spbesseljprime",
    "spbesselyprime",
    "spbesselhprime",
    "spharmonic",
]

import numpy as np
from math import factorial
from .utility import getdistance, getreff


# =============================================================================
# Lazy scipy import
# =============================================================================

_scipy_special = None


def _get_scipy_special():
    """Lazy import of scipy.special."""
    global _scipy_special
    if _scipy_special is None:
        from scipy import special

        _scipy_special = special
    return _scipy_special


# =============================================================================
# Spherical Bessel/Hankel Functions
# =============================================================================


def spbesselj(n, z):
    """
    Spherical Bessel function of the first kind.

    Wrapper around scipy.special.spherical_jn.

    Parameters
    ----------
    n : int
        Order of the function
    z : float or ndarray
        Argument

    Returns
    -------
    jn : float or ndarray
        Spherical Bessel function value(s)

    Example
    -------
    >>> spbesselj(0, 1.0)
    0.8414709848078965
    """
    return _get_scipy_special().spherical_jn(n, z)


def spbessely(n, z):
    """
    Spherical Bessel function of the second kind (Neumann function).

    Wrapper around scipy.special.spherical_yn.

    Parameters
    ----------
    n : int
        Order of the function
    z : float or ndarray
        Argument

    Returns
    -------
    yn : float or ndarray
        Spherical Neumann function value(s)

    Example
    -------
    >>> spbessely(0, 1.0)
    -0.5403023058681398
    """
    return _get_scipy_special().spherical_yn(n, z)


def spbesselh(n, k, z):
    """
    Spherical Hankel function.

    h_n^(1)(z) = j_n(z) + i*y_n(z)  (first kind, k=1)
    h_n^(2)(z) = j_n(z) - i*y_n(z)  (second kind, k=2)

    Parameters
    ----------
    n : int
        Order of the function
    k : int
        Kind of Hankel function (1 or 2)
    z : float or ndarray
        Argument

    Returns
    -------
    hn : complex or ndarray
        Spherical Hankel function value(s)

    Example
    -------
    >>> spbesselh(0, 1, 1.0)
    (0.8414709848078965+0.5403023058681398j)
    """
    sp = _get_scipy_special()
    jn = sp.spherical_jn(n, z)
    yn = sp.spherical_yn(n, z)
    if k == 1:
        return jn + 1j * yn
    elif k == 2:
        return jn - 1j * yn
    else:
        raise ValueError("k must be 1 or 2")


def spbesseljprime(n, z):
    """
    Derivative of spherical Bessel function of the first kind.

    Wrapper around scipy.special.spherical_jn with derivative=True.

    Parameters
    ----------
    n : int
        Order of the function
    z : float or ndarray
        Argument

    Returns
    -------
    jp : float or ndarray
        Derivative value(s)

    Example
    -------
    >>> spbesseljprime(0, 1.0)
    -0.30116867893975674
    """
    return _get_scipy_special().spherical_jn(n, z, derivative=True)


def spbesselyprime(n, z):
    """
    Derivative of spherical Bessel function of the second kind (Neumann).

    Wrapper around scipy.special.spherical_yn with derivative=True.

    Parameters
    ----------
    n : int
        Order of the function
    z : float or ndarray
        Argument

    Returns
    -------
    yp : float or ndarray
        Derivative value(s)

    Example
    -------
    >>> spbesselyprime(0, 1.0)
    0.8414709848078965
    """
    return _get_scipy_special().spherical_yn(n, z, derivative=True)


def spbesselhprime(n, k, z):
    """
    Derivative of spherical Hankel function.

    Parameters
    ----------
    n : int
        Order of the function
    k : int
        Kind of Hankel function (1 or 2)
    z : float or ndarray
        Argument

    Returns
    -------
    hp : complex or ndarray
        Derivative value(s)

    Example
    -------
    >>> spbesselhprime(0, 1, 1.0)
    (-0.30116867893975674-0.8414709848078965j)
    """
    sp = _get_scipy_special()
    jp = sp.spherical_jn(n, z, derivative=True)
    yp = sp.spherical_yn(n, z, derivative=True)
    if k == 1:
        return jp + 1j * yp
    elif k == 2:
        return jp - 1j * yp
    else:
        raise ValueError("k must be 1 or 2")


# =============================================================================
# Spherical Harmonics
# =============================================================================


def spharmonic(l, m, theta, phi):
    """
    Spherical harmonic function Y_l^m(theta, phi).

    Uses the convention where theta is the polar angle (0 to pi) and
    phi is the azimuthal angle (0 to 2*pi). This matches the MATLAB
    convention used in MMC/MCX.

    Note: scipy.special.sph_harm uses opposite convention (phi, theta),
    so we provide our own implementation for consistency.

    Parameters
    ----------
    l : int
        Degree (order), l >= 0
    m : int
        Angular index, -l <= m <= l
    theta : float or ndarray
        Polar angle (0 to pi)
    phi : float or ndarray
        Azimuthal angle (0 to 2*pi)

    Returns
    -------
    Y : complex or ndarray
        Spherical harmonic values

    Example
    -------
    >>> spharmonic(1, 0, np.pi/4, 0)
    (0.3454941494713355+0j)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))

    # Handle negative m using symmetry relation
    coeff = 1.0
    absm = abs(m)
    if m < 0:
        coeff = ((-1.0) ** m) * factorial(l + m) / factorial(l - m)

    # Associated Legendre polynomial P_l^|m|(cos(theta))
    Plm = _get_scipy_special().lpmv(absm, l, np.cos(theta))

    # Normalization factor
    norm = np.sqrt((2 * l + 1) * factorial(l - m) / (4 * np.pi * factorial(l + m)))

    result = coeff * norm * Plm * np.exp(1j * m * phi)

    # Return scalar if inputs were scalar
    return result.item() if result.size == 1 else result


# =============================================================================
# CW Solutions for Homogeneous Media
# =============================================================================


def infinite_cw(mua, musp, srcpos, detpos):
    """
    Analytical CW diffusion solution for infinite homogeneous medium.

    Parameters
    ----------
    mua : float
        Absorption coefficient (1/mm)
    musp : float
        Reduced scattering coefficient (1/mm)
    srcpos : ndarray
        Source position (1x3)
    detpos : ndarray
        Detector positions (Nx3)

    Returns
    -------
    phi : ndarray
        Fluence at detector positions
    """
    D = 1.0 / (3.0 * (mua + musp))
    mu_eff = np.sqrt(mua / D)
    srcpos, detpos = np.atleast_2d(srcpos), np.atleast_2d(detpos)
    r = getdistance(srcpos, detpos)
    return (1.0 / (4 * np.pi * D)) * np.exp(-mu_eff * r) / r


def semi_infinite_cw(mua, musp, n_in, n_out, srcpos, detpos):
    """
    Analytical CW diffusion solution for semi-infinite medium.

    Uses extrapolated boundary condition with image source method.
    See [Haskell1994], [Boas2002].

    Parameters
    ----------
    mua : float
        Absorption coefficient (1/mm)
    musp : float
        Reduced scattering coefficient (1/mm)
    n_in, n_out : float
        Refractive indices (inside medium, outside)
    srcpos : ndarray
        Source position (Mx3), z=0 is the boundary
    detpos : ndarray
        Detector positions (Nx3)

    Returns
    -------
    phi : ndarray
        Fluence at detector positions (MxN if M sources, else N)
    """
    D = 1.0 / (3.0 * (mua + musp))
    Reff = getreff(n_in, n_out)
    mu_eff = np.sqrt(mua / D)
    zb = 2 * D * (1 + Reff) / (1 - Reff)
    z0 = 1.0 / (mua + musp)

    srcpos, detpos = np.atleast_2d(srcpos), np.atleast_2d(detpos)

    # Real source at z0 below surface, image source at -(z0 + 2*zb)
    src_real = srcpos.copy()
    src_real[:, 2] = srcpos[:, 2] + z0
    src_image = srcpos.copy()
    src_image[:, 2] = srcpos[:, 2] - z0 - 2 * zb

    r1 = getdistance(src_real, detpos)
    r2 = getdistance(src_image, detpos)

    phi = (1.0 / (4 * np.pi * D)) * (
        np.exp(-mu_eff * r1) / r1 - np.exp(-mu_eff * r2) / r2
    )
    return phi.squeeze()


def semi_infinite_cw_flux(mua, musp, n_in, n_out, srcpos, detpos):
    """
    Compute surface flux (diffuse reflectance) for semi-infinite medium.

    Implements Eq. 6 of [Kienle1997].

    Parameters
    ----------
    mua : float
        Absorption coefficient (1/mm)
    musp : float
        Reduced scattering coefficient (1/mm)
    n_in, n_out : float
        Refractive indices (inside medium, outside)
    srcpos : ndarray
        Source positions (Mx3)
    detpos : ndarray
        Detector positions (Nx3)

    Returns
    -------
    flux : ndarray
        Diffuse reflectance at detector positions (1/(mm^2))
    """
    D = 1.0 / (3.0 * (mua + musp))
    Reff = getreff(n_in, n_out)
    z0 = 1.0 / (mua + musp)
    zb = 2 * D * (1 + Reff) / (1 - Reff)
    mu_eff = np.sqrt(3 * mua * (mua + musp))

    srcpos, detpos = np.atleast_2d(srcpos), np.atleast_2d(detpos)

    src_real = srcpos.copy()
    src_real[:, 2] = srcpos[:, 2] + z0
    src_image = srcpos.copy()
    src_image[:, 2] = srcpos[:, 2] + z0 + 2 * zb

    r1 = getdistance(src_real, detpos)
    r2 = getdistance(src_image, detpos)

    # Eq. 6 of Kienle1997
    flux = (1.0 / (4 * np.pi)) * (
        z0 * (mu_eff + 1.0 / r1) * np.exp(-mu_eff * r1) / r1**2
        + (z0 + 2 * zb) * (mu_eff + 1.0 / r2) * np.exp(-mu_eff * r2) / r2**2
    )
    return flux.squeeze()


# =============================================================================
# Time-Domain Solutions
# =============================================================================


def infinite_td(mua, musp, n, srcpos, detpos, t):
    """
    Time-domain diffusion solution for semi-infinite medium.

    See [Boas2002].

    Parameters
    ----------
    mua : float
        Absorption coefficient (1/mm)
    musp : float
        Reduced scattering coefficient (1/mm)
     : float

    n : float
        Refractive indices (inside medium)
    srcpos : ndarray
        Source positions (Mx3)
    detpos : ndarray
        Detector positions (Nx3)
    t : ndarray
        Time points (s)

    Returns
    -------
    phi : ndarray
        Fluence at detector positions for each time point (shape: len(t) x N)
        Units: 1/(mm^2*s)
    """
    D = 1.0 / (3.0 * (mua + musp))

    C0 = 299792458000.0  # Speed of light in vacuum (mm/s)
    v = C0 / n  # Speed of light in medium (mm/s)

    srcpos, detpos = np.atleast_2d(srcpos), np.atleast_2d(detpos)
    t = np.atleast_1d(t)

    r1 = getdistance(srcpos, detpos)

    # Broadcast for time: result shape (len(t), n_det)
    t = t[:, np.newaxis]
    s = 4 * D * v * t  # (len(t), 1)

    # Unit of phi: 1/(mm^2*s)
    phi = (v / (s * np.pi) ** 1.5) * np.exp(-mua * v * t) * np.exp(-(r1**2) / s)
    return phi.squeeze()


def semi_infinite_td(mua, musp, n_in, n_out, srcpos, detpos, t):
    """
    Time-domain diffusion solution for semi-infinite medium.

    See [Boas2002].

    Parameters
    ----------
    mua : float
        Absorption coefficient (1/mm)
    musp : float
        Reduced scattering coefficient (1/mm)
    n_in, n_out : float
        Refractive indices (inside medium, outside)
    srcpos : ndarray
        Source positions (Mx3)
    detpos : ndarray
        Detector positions (Nx3)
    t : ndarray
        Time points (s)

    Returns
    -------
    phi : ndarray
        Fluence at detector positions for each time point (shape: len(t) x N)
        Units: 1/(mm^2*s)
    """
    D = 1.0 / (3.0 * (mua + musp))

    C0 = 299792458000.0  # Speed of light in vacuum (mm/s)
    v = C0 / n_in  # Speed of light in medium (mm/s)

    Reff = getreff(n_in, n_out)
    zb = 2 * D * (1 + Reff) / (1 - Reff)
    z0 = 1.0 / (mua + musp)

    srcpos, detpos = np.atleast_2d(srcpos), np.atleast_2d(detpos)
    t = np.atleast_1d(t)

    src_real = srcpos.copy()
    src_real[:, 2] = srcpos[:, 2] + z0
    src_image = srcpos.copy()
    src_image[:, 2] = srcpos[:, 2] - z0 - 2 * zb

    r1 = getdistance(src_real, detpos)
    r2 = getdistance(src_image, detpos)

    # Broadcast for time: result shape (len(t), n_det)
    t = t[:, np.newaxis]
    s = 4 * D * v * t  # (len(t), 1)

    # Unit of phi: 1/(mm^2*s)
    phi = (
        (v / (s * np.pi) ** 1.5)
        * np.exp(-mua * v * t)
        * (np.exp(-(r1**2) / s) - np.exp(-(r2**2) / s))
    )
    return phi.squeeze()


# =============================================================================
# Sphere Diffusion Coefficients (Internal)
# =============================================================================


def _sphere_coeff_A(m, l, cfg):
    """Sphere exterior solution A coefficient."""
    if (cfg["src"][1] in (0, np.pi)) and m != 0:
        return 0.0

    x, y = cfg["kout"] * cfg["a"], cfg["kin"] * cfg["a"]
    Dout, Din = cfg["Dout"], cfg["Din"]

    hl_src = spbesselh(l, 1, cfg["kout"] * cfg["src"][0])
    Ylm_src = np.conj(spharmonic(l, m, cfg["src"][1], cfg["src"][2]))

    jl_x, jl_y = spbesselj(l, x), spbesselj(l, y)
    jlp_x, jlp_y = spbesseljprime(l, x), spbesseljprime(l, y)
    hlp_x, hl_x = spbesselhprime(l, 1, x), spbesselh(l, 1, x)

    numer = Dout * x * jlp_x * jl_y - Din * y * jl_x * jlp_y
    denom = Dout * x * hlp_x * jl_y - Din * y * hl_x * jlp_y

    return -1j * cfg["v"] * cfg["kout"] / Dout * hl_src * Ylm_src * numer / denom


def _sphere_coeff_C(m, l, cfg):
    """Sphere interior solution C coefficient."""
    if (cfg["src"][1] in (0, np.pi)) and m != 0:
        return 0.0

    x, y = cfg["kout"] * cfg["a"], cfg["kin"] * cfg["a"]
    Dout, Din = cfg["Dout"], cfg["Din"]

    hl_src = spbesselh(l, 1, cfg["kout"] * cfg["src"][0])
    Ylm_src = np.conj(spharmonic(l, m, cfg["src"][1], cfg["src"][2]))

    jl_x, jlp_x = spbesselj(l, x), spbesseljprime(l, x)
    jl_y, jlp_y = spbesselj(l, y), spbesseljprime(l, y)
    hl_x, hlp_x = spbesselh(l, 1, x), spbesselhprime(l, 1, x)

    # Wronskian-like numerator
    numer = Dout * x * (hl_x * jlp_x - hlp_x * jl_x)
    denom = Dout * x * hlp_x * jl_y - Din * y * hl_x * jlp_y

    return -1j * cfg["v"] * cfg["kout"] / Dout * hl_src * Ylm_src * numer / denom


# =============================================================================
# Sphere Field Components (Internal)
# =============================================================================


def _sphere_incident(r, theta, phi, cfg):
    """Incident field from point source in infinite medium."""
    # Convert spherical to Cartesian for source
    st, ct = np.sin(cfg["src"][1]), np.cos(cfg["src"][1])
    xs = cfg["src"][0] * st * np.cos(cfg["src"][2])
    ys = cfg["src"][0] * st * np.sin(cfg["src"][2])
    zs = cfg["src"][0] * ct

    # Field points
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    dist = np.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2)
    return cfg["v"] / (4 * np.pi * cfg["Dout"] * dist) * np.exp(1j * cfg["kout"] * dist)


def _sphere_scatter(r, theta, phi, cfg):
    """Scattered field outside sphere (series expansion)."""
    res = np.zeros_like(r, dtype=complex)
    kout_r = cfg["kout"] * r
    for l in range(cfg["maxl"] + 1):
        jl, yl = spbesselj(l, kout_r), spbessely(l, kout_r)
        for m in range(-l, l + 1):
            A = _sphere_coeff_A(m, l, cfg)
            if A == 0:
                continue
            B = 1j * A  # B = i*A
            Ylm = spharmonic(l, m, theta, phi)
            res += (A * jl + B * yl) * Ylm
    return res


def _sphere_interior(r, theta, phi, cfg):
    """Field inside sphere (series expansion)."""
    res = np.zeros_like(r, dtype=complex)
    kin_r = cfg["kin"] * r
    for l in range(cfg["maxl"] + 1):
        jl = spbesselj(l, kin_r)
        for m in range(-l, l + 1):
            C = _sphere_coeff_C(m, l, cfg)
            if C == 0:
                continue
            Ylm = spharmonic(l, m, theta, phi)
            res += C * jl * Ylm
    return res


def _sphere_exterior(r, theta, phi, cfg):
    """Total exterior field = incident + scattered."""
    return _sphere_incident(r, theta, phi, cfg) + _sphere_scatter(r, theta, phi, cfg)


# =============================================================================
# Sphere Configuration Helpers (Internal)
# =============================================================================


def _init_sphere_cfg(cfg):
    """Initialize derived parameters for sphere diffusion."""
    cfg = cfg.copy()
    cfg["Din"] = cfg["v"] / (3 * cfg["imusp"])
    cfg["Dout"] = cfg["v"] / (3 * cfg["omusp"])
    omega = cfg.get("omega", 0)
    cfg["kin"] = np.sqrt((-cfg["v"] * cfg["imua"] + 1j * omega) / cfg["Din"])
    cfg["kout"] = np.sqrt((-cfg["v"] * cfg["omua"] + 1j * omega) / cfg["Dout"])
    return cfg


def _cart2sph_grid(xi, yi, zi):
    """Convert Cartesian meshgrid to spherical coordinates (R, theta, phi)."""
    R = np.sqrt(xi**2 + yi**2 + zi**2).ravel()
    T = np.arccos(np.clip(zi.ravel() / (R + 1e-30), -1, 1))  # theta (polar)
    P = np.arctan2(yi.ravel(), xi.ravel())  # phi (azimuthal)
    return R, T, P


def _compute_field(R, T, P, cfg):
    """Compute field for interior and exterior regions."""
    res = np.zeros(len(R), dtype=complex)
    idx_ext, idx_int = R > cfg["a"], R <= cfg["a"]
    if np.any(idx_ext):
        res[idx_ext] = _sphere_exterior(R[idx_ext], T[idx_ext], P[idx_ext], cfg)
    if np.any(idx_int):
        res[idx_int] = _sphere_interior(R[idx_int], T[idx_int], P[idx_int], cfg)
    return res


# =============================================================================
# Main Sphere Diffusion Functions
# =============================================================================


def sphere_infinite(xrange, yrange, zrange, cfg):
    """
    CW diffusion solution for a sphere in infinite homogeneous medium.

    See [Fang2010].

    Parameters
    ----------
    xrange, yrange, zrange : ndarray
        1D arrays defining the evaluation grid
    cfg : dict
        Problem configuration:
        - v: speed of light (mm/s)
        - a: sphere radius (mm)
        - omua, omusp: outside (background) mua, mus' (1/mm)
        - imua, imusp: inside (sphere) mua, mus' (1/mm)
        - src: source position in spherical coords (R, theta, phi)
        - maxl: maximum order for series expansion (default 20)
        - omega: modulation frequency (default 0 for CW)

    Returns
    -------
    phi : ndarray
        Fluence on the grid (squeezed to remove singleton dims)
    xi, yi, zi : ndarray
        Meshgrid coordinates
    """
    cfg.setdefault("maxl", 20)
    cfg.setdefault("omega", 0)
    cfg = _init_sphere_cfg(cfg)

    xi, yi, zi = np.meshgrid(xrange, yrange, zrange, indexing="ij")
    shape = xi.shape
    R, T, P = _cart2sph_grid(xi, yi, zi)

    res = _compute_field(R, T, P, cfg)

    return (
        np.squeeze(res.reshape(shape)),
        np.squeeze(xi),
        np.squeeze(yi),
        np.squeeze(zi),
    )


def sphere_semi_infinite(xrange, yrange, zrange, cfg, n0=1.0, n1=None):
    """
    CW diffusion solution for a sphere in semi-infinite medium.

    Uses image source method. First-order approximation; accurate when
    sphere is far from boundary. See [Fang2010].

    Parameters
    ----------
    xrange, yrange, zrange : ndarray
        1D arrays defining the evaluation grid
    cfg : dict
        Problem configuration (see sphere_infinite)
    n0 : float
        Refractive index of upper space (above boundary, default 1.0)
    n1 : float
        Refractive index of lower space/medium (default 1.37)

    Returns
    -------
    phi : ndarray
        Fluence on the grid
    xi, yi, zi : ndarray
        Meshgrid coordinates
    """
    cfg.setdefault("maxl", 20)
    cfg.setdefault("omega", 0)
    cfg = _init_sphere_cfg(cfg)

    if n1 is None:
        n1 = 1.37  # typical tissue

    Reff = getreff(n1, n0)
    D = 1.0 / (3.0 * (cfg["omua"] + cfg["omusp"]))
    zb = 2 * D * (1 + Reff) / (1 - Reff)
    z0 = 1.0 / (cfg["omusp"] + cfg["omua"])

    xi, yi, zi = np.meshgrid(xrange, yrange, zrange, indexing="ij")
    shape = xi.shape
    R, T, P = _cart2sph_grid(xi, yi, zi)

    src0 = list(cfg["src"])

    # Real source field for real sphere
    cfg_real = cfg.copy()
    cfg_real["src"] = [src0[0] - z0, src0[1], src0[2]]
    res = _compute_field(R, T, P, cfg_real)

    # Image source field for real sphere (subtract)
    cfg_img = cfg.copy()
    cfg_img["src"] = [src0[0] + z0 + 2 * zb, np.pi, src0[2]]
    res -= _compute_field(R, T, P, cfg_img)

    # Scattered field contributions from mirrored sphere
    idx_ext = R > cfg["a"]
    if np.any(idx_ext):
        zi_m = zi.ravel() + 2 * (src0[0] + zb)
        R_m, T_m, P_m = _cart2sph_grid(xi, yi, zi_m.reshape(shape))

        # Real source scattered by mirrored sphere
        cfg_s1 = cfg.copy()
        cfg_s1["src"] = [src0[0] + z0 + 2 * zb, src0[1], src0[2]]
        res[idx_ext] += _sphere_scatter(
            R_m[idx_ext], T_m[idx_ext], P_m[idx_ext], cfg_s1
        )

        # Image source scattered by mirrored sphere
        cfg_s2 = cfg.copy()
        cfg_s2["src"] = [src0[0] - z0, src0[1], src0[2]]
        res[idx_ext] -= _sphere_scatter(
            R_m[idx_ext], T_m[idx_ext], P_m[idx_ext], cfg_s2
        )

    return (
        np.squeeze(res.reshape(shape)),
        np.squeeze(xi),
        np.squeeze(yi),
        np.squeeze(zi),
    )


def sphere_slab(xrange, yrange, zrange, cfg, h, n0=1.0, n1=None):
    """
    CW diffusion solution for a sphere in infinite slab.

    Uses image source method for both boundaries. First-order approximation.
    See [Fang2010].

    Parameters
    ----------
    xrange, yrange, zrange : ndarray
        1D arrays defining the evaluation grid
    cfg : dict
        Problem configuration (see sphere_infinite)
    h : float
        Slab thickness (mm)
    n0 : float
        Refractive index of upper space (above slab, default 1.0)
    n1 : float
        Refractive index of slab medium (default 1.37)

    Returns
    -------
    phi : ndarray
        Fluence on the grid
    xi, yi, zi : ndarray
        Meshgrid coordinates
    """
    cfg.setdefault("maxl", 20)
    cfg.setdefault("omega", 0)
    cfg = _init_sphere_cfg(cfg)

    if n1 is None:
        n1 = 1.37

    # Reff for both boundaries (medium to air)
    Reff1 = getreff(n1, n0)  # lower boundary
    Reff2 = getreff(n1, n0)  # upper boundary

    D = 1.0 / (3.0 * (cfg["omua"] + cfg["omusp"]))
    zb1 = 2 * D * (1 + Reff1) / (1 - Reff1)
    zb2 = 2 * D * (1 + Reff2) / (1 - Reff2)
    z0 = 1.0 / (cfg["omusp"] + cfg["omua"])

    xi, yi, zi = np.meshgrid(xrange, yrange, zrange, indexing="ij")
    shape = xi.shape

    # Start with semi-infinite solution (lower boundary)
    res, _, _, _ = sphere_semi_infinite(xrange, yrange, zrange, cfg, n0, n1)
    res = res.ravel()

    R, T, P = _cart2sph_grid(xi, yi, zi)
    idx_ext = R > cfg["a"]

    src0 = list(cfg["src"])

    # Image source at upper boundary (subtract)
    cfg_upper = cfg.copy()
    cfg_upper["src"] = [2 * h - src0[0] + 2 * zb2 - z0, np.pi - src0[1], src0[2]]
    res -= _compute_field(R, T, P, cfg_upper)

    # Second image source (add back)
    cfg_upper2 = cfg.copy()
    cfg_upper2["src"] = [
        2 * h - src0[0] + 2 * zb2 + z0 + 2 * zb1,
        np.pi - src0[1],
        src0[2],
    ]
    res += _compute_field(R, T, P, cfg_upper2)

    # Scattered field from mirrored sphere at upper boundary
    if np.any(idx_ext):
        zi_m = zi.ravel() - 2 * (h - src0[0] + zb2)
        R_m, T_m, P_m = _cart2sph_grid(xi, yi, zi_m.reshape(shape))

        scatter_configs = [
            ([2 * h - src0[0] - z0, src0[1], src0[2]], 1),
            ([2 * h - src0[0] + 2 * zb2 + z0, src0[1], src0[2]], -1),
            ([src0[0] - z0, np.pi - src0[1], src0[2]], -1),
            ([src0[0] + 2 * zb1 + z0, np.pi - src0[1], src0[2]], 1),
        ]

        for src_pos, sign in scatter_configs:
            cfg_s = cfg.copy()
            cfg_s["src"] = src_pos
            res[idx_ext] += sign * _sphere_scatter(
                R_m[idx_ext], T_m[idx_ext], P_m[idx_ext], cfg_s
            )

    return (
        np.squeeze(res.reshape(shape)),
        np.squeeze(xi),
        np.squeeze(yi),
        np.squeeze(zi),
    )
