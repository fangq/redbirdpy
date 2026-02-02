"""
Unit tests for wide-field source/detector support in Redbird.

Tests src2bc, planar/pattern/fourier sources, and wide-field detection.

Run with: python -m unittest test_widefield -v
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbirdpy import forward, utility, recon

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False

# Module-level cache for mesh data
_CACHED_MESH = None


def setUpModule():
    """Create and cache the mesh once for all tests."""
    global _CACHED_MESH

    if not HAS_ISO2MESH:
        raise unittest.SkipTest("iso2mesh required for wide-field tests")

    if HAS_ISO2MESH:
        node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 8)
    else:
        # Manual simple tetrahedron mesh (1-based indices)
        node = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        elem = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
            dtype=int,
        )
        face = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]], dtype=int)
    _CACHED_MESH = (node.copy(), face.copy(), elem.copy())


def create_simple_mesh():
    """Return a copy of the cached mesh."""
    global _CACHED_MESH
    if _CACHED_MESH is None:
        setUpModule()
    node, face, elem = _CACHED_MESH
    return node.copy(), face.copy(), elem.copy()


def create_widefield_cfg(srctype="planar"):
    """Create a configuration with wide-field source."""
    if not HAS_ISO2MESH:
        raise unittest.SkipTest("iso2mesh required for wide-field tests")

    node, face, elem = create_simple_mesh()

    cfg = {
        "node": node,
        "elem": elem,
        "face": face,
        "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
        "srcpos": np.array([[30, 30, 0]]),
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": np.array([[30, 40, 0], [40, 30, 0]]),
        "detdir": np.array([[0, 0, 1]]),
        "seg": np.ones(elem.shape[0], dtype=int),
        "omega": 0,
        "srctype": srctype,
        "srcparam1": np.array([20, 0, 0, 0]),  # 20mm in x direction
        "srcparam2": np.array([0, 20, 0, 0]),  # 20mm in y direction
    }
    return cfg


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestSrc2bc(unittest.TestCase):
    """Test utility.src2bc function."""

    def test_src2bc_planar_returns_widesrc(self):
        """src2bc with planar source should add widesrc to cfg."""
        cfg = create_widefield_cfg("planar")
        cfg, _ = utility.meshprep(cfg)

        self.assertIn("widesrc", cfg)
        self.assertIsNotNone(cfg["widesrc"])

    def test_src2bc_widesrc_shape(self):
        """widesrc should have shape (Nn, Npattern)."""
        cfg = create_widefield_cfg("planar")
        cfg, _ = utility.meshprep(cfg)

        nn = cfg["node"].shape[0]
        widesrc = cfg["widesrc"]

        self.assertEqual(widesrc.shape[0], nn)
        self.assertGreaterEqual(widesrc.shape[1], 1)

    def test_src2bc_widesrc_normalized(self):
        """widesrc columns should be normalized."""
        cfg = create_widefield_cfg("planar")
        cfg, _ = utility.meshprep(cfg)

        widesrc = cfg["widesrc"]

        # Each pattern should sum to approximately 1 (normalized)
        for i in range(widesrc.shape[1]):
            col_sum = np.sum(np.abs(widesrc[:, i]))
            self.assertAlmostEqual(
                col_sum, 1.0, places=5, msg=f"Pattern {i} should be normalized to sum=1"
            )

    def test_src2bc_widesrc_nonnegative(self):
        """widesrc values should be non-negative for planar source."""
        cfg = create_widefield_cfg("planar")
        cfg, _ = utility.meshprep(cfg)

        widesrc = cfg["widesrc"]

        self.assertTrue(
            np.all(widesrc >= -1e-10), "Planar source BC should be non-negative"
        )

    def test_src2bc_removes_point_source(self):
        """src2bc should remove wide-field source from point sources."""
        cfg = create_widefield_cfg("planar")
        original_srcnum = cfg["srcpos"].shape[0]

        cfg, _ = utility.meshprep(cfg)

        # Point source should be removed (converted to wide-field)
        self.assertEqual(cfg["srcpos"].shape[0], original_srcnum - 1)

    def test_src2bc_stores_mapping(self):
        """src2bc should store wide-field source mapping."""
        cfg = create_widefield_cfg("planar")
        cfg, _ = utility.meshprep(cfg)

        self.assertIn("wfsrcmapping", cfg)

    def test_src2bc_preserves_srcpos0(self):
        """src2bc should preserve original srcpos in srcpos0."""
        cfg = create_widefield_cfg("planar")
        original_srcpos = cfg["srcpos"].copy()

        cfg, _ = utility.meshprep(cfg)

        self.assertIn("srcpos0", cfg)
        assert_array_almost_equal(cfg["srcpos0"], original_srcpos)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestPatternSource(unittest.TestCase):
    """Test pattern-based wide-field sources."""

    def test_pattern_source_basic(self):
        """Pattern source should create widesrc from srcpattern."""
        cfg = create_widefield_cfg("pattern")

        # Create a simple 4x4 pattern
        pattern = np.ones((4, 4))
        pattern[0:2, 0:2] = 0.5  # Quadrant with different intensity
        cfg["srcpattern"] = pattern

        cfg, _ = utility.meshprep(cfg)

        self.assertIn("widesrc", cfg)
        self.assertEqual(cfg["widesrc"].shape[1], 1)  # Single pattern

    def test_pattern_source_multipattern(self):
        """Pattern source with 3D srcpattern should create multiple patterns."""
        cfg = create_widefield_cfg("pattern")

        # Create 3 patterns (4x4x3)
        patterns = np.zeros((4, 4, 3))
        patterns[:, :, 0] = 1.0  # Uniform
        patterns[0:2, :, 1] = 1.0  # Top half
        patterns[:, 0:2, 2] = 1.0  # Left half
        cfg["srcpattern"] = patterns

        cfg, _ = utility.meshprep(cfg)

        self.assertIn("widesrc", cfg)
        self.assertEqual(cfg["widesrc"].shape[1], 3)

    def test_pattern_normalized_per_pattern(self):
        """Each pattern should be individually normalized."""
        cfg = create_widefield_cfg("pattern")

        patterns = np.zeros((4, 4, 2))
        patterns[:, :, 0] = 1.0
        patterns[:, :, 1] = 0.5  # Different intensity
        cfg["srcpattern"] = patterns

        cfg, _ = utility.meshprep(cfg)

        widesrc = cfg["widesrc"]
        for i in range(widesrc.shape[1]):
            col_sum = np.sum(np.abs(widesrc[:, i]))
            self.assertAlmostEqual(col_sum, 1.0, places=5)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestFourierSource(unittest.TestCase):
    """Test Fourier-basis wide-field sources."""

    def test_fourier_source_basic(self):
        """Fourier source should create spatial frequency patterns."""
        cfg = create_widefield_cfg("fourier")

        # srcparam1[3] = kx, srcparam2[3] = ky
        cfg["srcparam1"] = np.array([20, 0, 0, 2])  # kx=2
        cfg["srcparam2"] = np.array([0, 20, 0, 2])  # ky=2

        cfg, _ = utility.meshprep(cfg)

        self.assertIn("widesrc", cfg)
        # Should have kx * ky = 4 patterns
        self.assertEqual(cfg["widesrc"].shape[1], 4)

    def test_fourier_source_pattern_count(self):
        """Fourier source should create kx*ky patterns."""
        cfg = create_widefield_cfg("fourier")

        kx, ky = 3, 4
        cfg["srcparam1"] = np.array([20, 0, 0, kx])
        cfg["srcparam2"] = np.array([0, 20, 0, ky])

        cfg, _ = utility.meshprep(cfg)

        self.assertEqual(cfg["widesrc"].shape[1], kx * ky)

    def test_fourier_patterns_vary(self):
        """Different Fourier patterns should have different distributions."""
        cfg = create_widefield_cfg("fourier")

        cfg["srcparam1"] = np.array([20, 0, 0, 2])
        cfg["srcparam2"] = np.array([0, 20, 0, 2])

        cfg, _ = utility.meshprep(cfg)

        widesrc = cfg["widesrc"]

        # Patterns should not all be identical
        patterns_differ = False
        for i in range(1, widesrc.shape[1]):
            if not np.allclose(widesrc[:, 0], widesrc[:, i]):
                patterns_differ = True
                break

        self.assertTrue(patterns_differ, "Fourier patterns should vary")


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldSdmap(unittest.TestCase):
    """Test sdmap handling of wide-field sources."""

    def test_sdmap_includes_widesrc(self):
        """sdmap should include wide-field source indices."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        # sd should have entries for wide-field sources
        self.assertGreater(sd.shape[0], 0)

    def test_sdmap_widesrc_column_offset(self):
        """Wide-field source indices should be offset by point source count."""
        cfg = create_widefield_cfg("planar")
        # Add a point source that won't be converted
        cfg["srcpos"] = np.array([[30, 30, 0], [10, 10, 0]])
        cfg["srcid"] = 0  # Only first source is wide-field

        cfg, sd = utility.meshprep(cfg)

        srcnum = cfg["srcpos"].shape[0]
        wfsrcnum = cfg["widesrc"].shape[1] if cfg["widesrc"].size > 0 else 0

        # Source indices should include both point (0 to srcnum-1)
        # and wide-field (srcnum to srcnum+wfsrcnum-1)
        src_indices = sd[:, 0]
        self.assertLessEqual(src_indices.max(), srcnum + wfsrcnum - 1)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldFemrhs(unittest.TestCase):
    """Test femrhs handling of wide-field sources."""

    def test_femrhs_includes_widesrc_columns(self):
        """femrhs should include columns for wide-field sources."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        rhs, loc, bary, optode = forward.femrhs(cfg, sd)

        nn = cfg["node"].shape[0]
        srcnum = cfg["srcpos"].shape[0]
        wfsrcnum = cfg["widesrc"].shape[1] if cfg["widesrc"].size > 0 else 0
        detnum = cfg["detpos"].shape[0]

        expected_cols = srcnum + wfsrcnum + detnum
        self.assertEqual(rhs.shape[1], expected_cols)

    def test_femrhs_widesrc_columns_match(self):
        """femrhs wide-field columns should match widesrc."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        rhs, _, _, _ = forward.femrhs(cfg, sd)

        srcnum = cfg["srcpos"].shape[0]
        wfsrcnum = cfg["widesrc"].shape[1]

        # Extract wide-field source columns from RHS
        wf_cols = rhs[:, srcnum : srcnum + wfsrcnum].toarray()

        # Should match widesrc (transposed for comparison)
        assert_allclose(wf_cols, cfg["widesrc"], rtol=1e-10)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldForward(unittest.TestCase):
    """Test forward simulation with wide-field sources."""

    def test_runforward_with_planar_source(self):
        """runforward should work with planar wide-field source."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        self.assertTrue(np.all(np.isfinite(detval)))
        self.assertTrue(np.all(np.isfinite(phi)))

    def test_runforward_detval_shape_with_widesrc(self):
        """detval shape should account for wide-field sources."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        detnum = cfg["detpos"].shape[0]
        srcnum = cfg["srcpos"].shape[0]
        wfsrcnum = cfg["widesrc"].shape[1] if cfg["widesrc"].size > 0 else 0

        total_src = srcnum + wfsrcnum
        self.assertEqual(detval.shape, (detnum, total_src))

    def test_runforward_phi_shape_with_widesrc(self):
        """phi should have columns for all sources and detectors."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        nn = cfg["node"].shape[0]
        srcnum = cfg["srcpos"].shape[0]
        wfsrcnum = cfg["widesrc"].shape[1] if cfg["widesrc"].size > 0 else 0
        detnum = cfg["detpos"].shape[0]

        expected_cols = srcnum + wfsrcnum + detnum
        self.assertEqual(phi.shape, (nn, expected_cols))

    def test_runforward_widesrc_gives_different_result(self):
        """Wide-field source should give different results than point source."""
        # Point source config
        cfg_point = create_widefield_cfg("pencil")
        cfg_point["srctype"] = "pencil"
        del cfg_point["srcparam1"]
        del cfg_point["srcparam2"]
        cfg_point, sd_point = utility.meshprep(cfg_point)

        detval_point, _ = forward.runforward(cfg_point, sd=sd_point)

        # Wide-field config
        cfg_wide = create_widefield_cfg("planar")
        cfg_wide, sd_wide = utility.meshprep(cfg_wide)

        detval_wide, _ = forward.runforward(cfg_wide, sd=sd_wide)

        # Results should differ
        self.assertFalse(np.allclose(detval_point, detval_wide[:, :1]))

    def test_runforward_pattern_multipattern(self):
        """Forward with multi-pattern source should produce multi-column detval."""
        cfg = create_widefield_cfg("pattern")

        # 3 patterns
        patterns = np.zeros((4, 4, 3))
        patterns[:, :, 0] = 1.0
        patterns[0:2, :, 1] = 1.0
        patterns[:, 0:2, 2] = 1.0
        cfg["srcpattern"] = patterns

        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        detnum = cfg["detpos"].shape[0]
        wfsrcnum = cfg["widesrc"].shape[1]

        self.assertEqual(detval.shape[1], wfsrcnum)
        self.assertEqual(wfsrcnum, 3)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldDetector(unittest.TestCase):
    """Test wide-field detector support."""

    def test_widedet_created(self):
        """Wide-field detector should create widedet in cfg."""
        cfg = create_widefield_cfg("pencil")
        cfg["srctype"] = "pencil"
        del cfg["srcparam1"]
        del cfg["srcparam2"]

        # Configure wide-field detector
        cfg["dettype"] = "planar"
        cfg["detparam1"] = np.array([10, 0, 0, 0])
        cfg["detparam2"] = np.array([0, 10, 0, 0])

        cfg, sd = utility.meshprep(cfg)

        self.assertIn("widedet", cfg)
        self.assertIsNotNone(cfg["widedet"])

    def test_widedet_shape(self):
        """widedet should have shape (Nn, Npattern)."""
        cfg = create_widefield_cfg("pencil")
        cfg["srctype"] = "pencil"
        del cfg["srcparam1"]
        del cfg["srcparam2"]

        cfg["dettype"] = "planar"
        cfg["detparam1"] = np.array([10, 0, 0, 0])
        cfg["detparam2"] = np.array([0, 10, 0, 0])

        cfg, sd = utility.meshprep(cfg)

        nn = cfg["node"].shape[0]
        widedet = cfg["widedet"]

        self.assertEqual(widedet.shape[0], nn)

    def test_runforward_with_widedet(self):
        """runforward should work with wide-field detector."""
        cfg = create_widefield_cfg("pencil")
        cfg["srctype"] = "pencil"
        del cfg["srcparam1"]
        del cfg["srcparam2"]

        cfg["dettype"] = "planar"
        cfg["detparam1"] = np.array([10, 0, 0, 0])
        cfg["detparam2"] = np.array([0, 10, 0, 0])

        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        self.assertTrue(np.all(np.isfinite(detval)))


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldJacobian(unittest.TestCase):
    """Test Jacobian computation with wide-field sources."""

    def test_jac_with_widesrc(self):
        """Jacobian should be computed for wide-field sources."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        Jmua_n, Jmua_e = forward.jac(
            sd, phi, cfg["deldotdel"], cfg["elem"], cfg["evol"]
        )

        # Should have rows for each active source-detector pair
        nsd = np.sum(sd[:, 2] == 1) if sd.shape[1] >= 3 else sd.shape[0]
        self.assertEqual(Jmua_n.shape[0], nsd)

    def test_jac_widesrc_finite(self):
        """Jacobian for wide-field sources should be finite."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        Jmua_n, Jmua_e = forward.jac(
            sd, phi, cfg["deldotdel"], cfg["elem"], cfg["evol"]
        )

        self.assertTrue(np.all(np.isfinite(Jmua_n)))
        self.assertTrue(np.all(np.isfinite(Jmua_e)))

    def test_jac_widesrc_negative(self):
        """Jacobian for mua should be non-positive for wide-field."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        Jmua_n, Jmua_e = forward.jac(
            sd, phi, cfg["deldotdel"], cfg["elem"], cfg["evol"]
        )

        self.assertTrue(np.all(Jmua_n <= 1e-10))


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldReconstruction(unittest.TestCase):
    """Test reconstruction with wide-field sources."""

    def test_runrecon_with_widesrc(self):
        """runrecon should work with wide-field sources."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        # Generate synthetic data
        detphi0, _ = forward.runforward(cfg, sd=sd)
        detphi0 = detphi0 * 0.95  # Perturb

        recon_cfg = {
            "prop": cfg["prop"].copy(),
            "lambda": 0.1,
        }

        recon_out, resid, cfg_out, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=2, report=False
        )

        self.assertIsInstance(recon_out, dict)
        self.assertGreater(len(resid), 0)

    def test_runrecon_widesrc_residual_finite(self):
        """Reconstruction residual should be finite with wide-field sources."""
        cfg = create_widefield_cfg("planar")
        cfg, sd = utility.meshprep(cfg)

        detphi0, _ = forward.runforward(cfg, sd=sd)
        detphi0 = detphi0 * 0.95

        recon_cfg = {
            "prop": cfg["prop"].copy(),
            "lambda": 0.1,
        }

        _, resid, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=2, report=False
        )

        self.assertTrue(np.all(np.isfinite(resid)))


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldMultiwavelength(unittest.TestCase):
    """Test wide-field sources with multi-wavelength."""

    def test_widesrc_multiwavelength(self):
        """Wide-field source should work with multiple wavelengths."""
        cfg = create_widefield_cfg("planar")
        cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.012, 1.1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.008, 0.9, 0, 1.37]]),
        }

        cfg, sd = utility.meshprep(cfg)

        self.assertIn("widesrc", cfg)
        self.assertIsInstance(sd, dict)
        self.assertIn("690", sd)
        self.assertIn("830", sd)

    def test_runforward_widesrc_multiwavelength(self):
        """runforward should handle wide-field with multi-wavelength."""
        cfg = create_widefield_cfg("planar")
        cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.012, 1.1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.008, 0.9, 0, 1.37]]),
        }

        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        self.assertIsInstance(detval, dict)
        self.assertIn("690", detval)
        self.assertIn("830", detval)

        # Different wavelengths should give different results
        self.assertFalse(np.allclose(detval["690"], detval["830"]))


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestInpolygon(unittest.TestCase):
    """Test _inpolygon helper function."""

    def test_inpolygon_square(self):
        """_inpolygon should correctly identify points in square."""
        # Square from (0,0) to (1,1)
        px = np.array([0, 1, 1, 0, 0])
        py = np.array([0, 0, 1, 1, 0])

        # Test points
        x = np.array([0.5, 0.5, 1.5, -0.5])
        y = np.array([0.5, 1.5, 0.5, 0.5])

        result = utility._inpolygon(x, y, px, py)

        expected = np.array([True, False, False, False])
        assert_array_almost_equal(result, expected)

    def test_inpolygon_triangle(self):
        """_inpolygon should correctly identify points in triangle."""
        # Triangle with vertices at (0,0), (1,0), (0.5,1)
        px = np.array([0, 1, 0.5, 0])
        py = np.array([0, 0, 1, 0])

        # Test points:
        # (0.5, 0.3) - inside (center area)
        # (0.5, 0.8) - inside (near top vertex)
        # (0.1, 0.5) - outside (left of left edge)
        x = np.array([0.5, 0.5, 0.1])
        y = np.array([0.3, 0.8, 0.5])

        result = utility._inpolygon(x, y, px, py)

        # (0.5, 0.3) inside, (0.5, 0.8) inside, (0.1, 0.5) outside
        expected = np.array([True, True, False])
        assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
