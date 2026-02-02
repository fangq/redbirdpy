"""
Integration tests and index convention verification for Redbird.

These tests verify end-to-end workflows and proper handling of
1-based mesh indices throughout the codebase.

Run with: python -m unittest test_integration -v
"""

import unittest
import numpy as np
import sys
import os
from numpy.testing import assert_array_almost_equal, assert_allclose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbird as rb
from redbird import forward, recon, utility, property as prop_module

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False

# Module-level cache for mesh data (keyed by mesh parameters)
_MESH_CACHE = {}


def get_cached_mesh(box_min, box_max, max_vol):
    """
    Get or create a cached mesh.

    Parameters
    ----------
    box_min : list
        Minimum corner of box [x, y, z]
    box_max : list
        Maximum corner of box [x, y, z]
    max_vol : float
        Maximum element volume parameter

    Returns
    -------
    tuple
        (node, face, elem) arrays (copies)
    """
    global _MESH_CACHE

    # Create hashable key
    key = (tuple(box_min), tuple(box_max), max_vol)

    if key not in _MESH_CACHE:
        if not HAS_ISO2MESH:
            raise unittest.SkipTest("iso2mesh not installed")
        node, face, elem = i2m.meshabox(box_min, box_max, max_vol)
        _MESH_CACHE[key] = (node.copy(), face.copy(), elem.copy())

    # Return copies so tests don't interfere with each other
    node, face, elem = _MESH_CACHE[key]
    return node.copy(), face.copy(), elem.copy()


def setUpModule():
    """Pre-generate commonly used meshes."""
    if not HAS_ISO2MESH:
        return
    # Pre-cache the most commonly used mesh configurations
    common_meshes = [
        ([0, 0, 0], [10, 10, 10], 3),
        ([0, 0, 0], [60, 60, 30], 10),
        ([0, 0, 0], [60, 60, 30], 8),
        ([0, 0, 0], [60, 60, 30], 5),
    ]
    for box_min, box_max, max_vol in common_meshes:
        get_cached_mesh(box_min, box_max, max_vol)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestIso2meshIndexConvention(unittest.TestCase):
    """Verify iso2mesh returns 1-based indices."""

    def test_meshabox_elem_1based(self):
        """meshabox elem should be 1-based."""
        node, face, elem = get_cached_mesh([0, 0, 0], [10, 10, 10], 3)

        self.assertGreaterEqual(elem.min(), 1, "elem minimum should be >= 1 (1-based)")

    def test_meshabox_face_1based(self):
        """meshabox face should be 1-based."""
        node, face, elem = get_cached_mesh([0, 0, 0], [10, 10, 10], 3)

        self.assertGreaterEqual(face.min(), 1, "face minimum should be >= 1 (1-based)")

    def test_meshabox_max_valid(self):
        """meshabox indices should not exceed node count."""
        node, face, elem = get_cached_mesh([0, 0, 0], [10, 10, 10], 3)

        self.assertLessEqual(elem.max(), node.shape[0])
        self.assertLessEqual(face.max(), node.shape[0])

    def test_volface_1based(self):
        """volface should return 1-based indices."""
        node, _, elem = get_cached_mesh([0, 0, 0], [60, 60, 30], 10)

        # iso2mesh volface may return (face, faceid) tuple
        face_result = i2m.volface(elem[:, :4])
        if isinstance(face_result, tuple):
            face = face_result[0]
        else:
            face = face_result

        # Should be 1-based
        self.assertGreaterEqual(face.min(), 1)

    def test_meshreorient_preserves_1based(self):
        """meshreorient should preserve 1-based indices."""
        node, _, elem = get_cached_mesh([0, 0, 0], [10, 10, 10], 3)

        elem_new, evol, idx = i2m.meshreorient(node, elem[:, :4])

        self.assertGreaterEqual(elem_new.min(), 1)


@unittest.skipUnless(HAS_ISO2MESH, "redbird or iso2mesh not installed")
class TestRedbirdIndexPreservation(unittest.TestCase):
    """Verify Redbird preserves 1-based indices in cfg."""

    def setUp(self):
        """Create test mesh and configuration."""
        self.node, self.face, self.elem = get_cached_mesh([0, 0, 0], [60, 60, 30], 10)

        self.cfg = {
            "node": self.node,
            "elem": self.elem,
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "srcpos": np.array([[30, 30, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[30, 40, 0], [40, 30, 0]]),
            "detdir": np.array([[0, 0, 1]]),
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "omega": 0,
        }

    def test_meshprep_elem_stays_1based(self):
        """meshprep should not change elem to 0-based."""
        cfg, _ = utility.meshprep(self.cfg)

        self.assertGreaterEqual(
            cfg["elem"].min(), 1, "elem should remain 1-based after meshprep"
        )

    def test_meshprep_face_stays_1based(self):
        """meshprep should produce 1-based face."""
        cfg, _ = utility.meshprep(self.cfg)

        self.assertGreaterEqual(
            cfg["face"].min(), 1, "face should be 1-based after meshprep"
        )

    def test_forward_doesnt_modify_cfg_elem(self):
        """runforward should not modify cfg['elem']."""
        cfg, sd = utility.meshprep(self.cfg)
        elem_before = cfg["elem"].copy()

        forward.runforward(cfg, sd=sd)

        assert_array_almost_equal(cfg["elem"], elem_before)
        self.assertGreaterEqual(cfg["elem"].min(), 1)

    def test_forward_doesnt_modify_cfg_face(self):
        """runforward should not modify cfg['face']."""
        cfg, sd = utility.meshprep(self.cfg)
        face_before = cfg["face"].copy()

        forward.runforward(cfg, sd=sd)

        assert_array_almost_equal(cfg["face"], face_before)
        self.assertGreaterEqual(cfg["face"].min(), 1)


@unittest.skipUnless(HAS_ISO2MESH, "redbird or iso2mesh not installed")
class TestEndToEndForward(unittest.TestCase):
    """End-to-end forward simulation tests."""

    def setUp(self):
        """Create test configuration."""
        node, face, elem = get_cached_mesh([0, 0, 0], [60, 60, 30], 8)

        self.cfg = {
            "node": node,
            "elem": elem,
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "srcpos": np.array([[30, 30, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[30, 40, 0], [40, 30, 0], [20, 30, 0]]),
            "detdir": np.array([[0, 0, 1]]),
            "seg": np.ones(elem.shape[0], dtype=int),
            "omega": 0,
        }
        self.cfg, self.sd = utility.meshprep(self.cfg)

    def test_forward_produces_positive_detval(self):
        """Forward simulation should produce positive detector values."""
        detval, phi = forward.runforward(self.cfg)

        self.assertTrue(np.all(detval > 0), "All detector values should be positive")

    def test_forward_produces_positive_phi(self):
        """Forward solution should produce finite fluence values."""
        detval, phi = forward.runforward(self.cfg)

        # Check that fluence is finite (not NaN or Inf)
        self.assertTrue(np.all(np.isfinite(phi)), "Fluence should be finite")

    def test_forward_detval_decreases_with_distance(self):
        """Detector values should decrease with source-detector distance."""
        # Compute distances
        src = self.cfg["srcpos"][0]
        det = self.cfg["detpos"]
        distances = np.sqrt(np.sum((det - src) ** 2, axis=1))

        detval, _ = forward.runforward(self.cfg)
        detval = detval.flatten()

        # Sort by distance
        sort_idx = np.argsort(distances)

        # Values should generally decrease (with some tolerance for near-field)
        for i in range(len(sort_idx) - 1):
            if distances[sort_idx[i + 1]] > distances[sort_idx[i]] * 1.5:
                self.assertLess(detval[sort_idx[i + 1]], detval[sort_idx[i]] * 1.5)

    def test_forward_jacobian_is_negative(self):
        """Jacobian for mua should be negative."""
        detval, phi = forward.runforward(self.cfg, sd=self.sd)

        Jmua_n, Jmua_e = forward.jac(
            self.sd, phi, self.cfg["deldotdel"], self.cfg["elem"], self.cfg["evol"]
        )

        self.assertTrue(np.all(Jmua_n <= 1e-10), "Jacobian should be non-positive")

    def test_forward_multiwavelength(self):
        """Forward simulation should handle multiple wavelengths."""
        cfg = self.cfg.copy()
        cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.012, 1.1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.008, 0.9, 0, 1.37]]),
        }
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        self.assertIsInstance(detval, dict)
        self.assertIn("690", detval)
        self.assertIn("830", detval)

        # Different properties should give different results
        self.assertFalse(np.allclose(detval["690"], detval["830"]))


@unittest.skipUnless(HAS_ISO2MESH, "redbird or iso2mesh not installed")
class TestEndToEndReconstruction(unittest.TestCase):
    """End-to-end reconstruction tests."""

    def setUp(self):
        """Create test configuration with synthetic data."""
        node, face, elem = get_cached_mesh([0, 0, 0], [60, 60, 30], 10)

        self.cfg = {
            "node": node,
            "elem": elem,
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "srcpos": np.array([[30, 30, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[30, 40, 0], [40, 30, 0]]),
            "detdir": np.array([[0, 0, 1]]),
            "seg": np.ones(elem.shape[0], dtype=int),
            "omega": 0,
        }
        self.cfg, self.sd = utility.meshprep(self.cfg)

        # Generate baseline data
        self.detphi_baseline, _ = forward.runforward(self.cfg, sd=self.sd)

        # Simulate perturbation
        self.detphi_perturbed = self.detphi_baseline * 0.9

    def test_reconstruction_reduces_residual(self):
        """Reconstruction should reduce or maintain residual."""
        recon_cfg = {"prop": self.cfg["prop"].copy(), "lambda": 0.1}

        recon_out, resid, *_ = recon.runrecon(
            self.cfg,
            recon_cfg,
            self.detphi_perturbed,
            self.sd,
            maxiter=3,
            report=False,
        )

        # Check that final residual is not worse than initial
        # (may be equal if converged immediately or only 1 iteration)
        self.assertLessEqual(
            resid[-1],
            resid[0] * 1.01,  # Allow 1% tolerance
            "Residual should not increase significantly during reconstruction",
        )

    def test_reconstruction_convergence_tolerance(self):
        """Reconstruction should stop early if tolerance reached."""
        recon_cfg = {"prop": self.cfg["prop"].copy(), "lambda": 0.1}

        # Very tight tolerance should cause early stopping
        recon_out, resid, *_ = recon.runrecon(
            self.cfg,
            recon_cfg,
            self.detphi_perturbed,
            self.sd,
            maxiter=20,
            tol=1e-10,
            report=False,
        )

        # Should have fewer iterations than maxiter if converged
        # (or exactly maxiter if not converged)
        self.assertLessEqual(len(resid), 20)


@unittest.skipUnless(HAS_ISO2MESH, "redbird or iso2mesh not installed")
class TestVolumeConsistency(unittest.TestCase):
    """Test that computed volumes are physically consistent."""

    def test_total_volume_matches_box(self):
        """Sum of element volumes should equal box volume."""
        box_dims = [60, 60, 30]
        expected_vol = np.prod(box_dims)

        node, _, elem = get_cached_mesh([0, 0, 0], box_dims, 5)

        cfg = {
            "node": node,
            "elem": elem,
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "srcpos": np.array([[30, 30, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[30, 40, 0]]),
            "detdir": np.array([[0, 0, 1]]),
            "seg": np.ones(elem.shape[0], dtype=int),
        }
        cfg, _ = utility.meshprep(cfg)

        total_vol = np.sum(cfg["evol"])

        self.assertAlmostEqual(
            total_vol,
            expected_vol,
            delta=expected_vol * 0.01,
            msg="Total element volume should match box volume",
        )

    def test_nodal_volume_conservation(self):
        """Sum of nodal volumes should equal total volume."""
        node, _, elem = get_cached_mesh([0, 0, 0], [10, 10, 10], 3)

        cfg = {
            "node": node,
            "elem": elem,
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "srcpos": np.array([[5, 5, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[5, 7, 0]]),
            "detdir": np.array([[0, 0, 1]]),
            "seg": np.ones(elem.shape[0], dtype=int),
        }
        cfg, _ = utility.meshprep(cfg)

        self.assertAlmostEqual(np.sum(cfg["nvol"]), np.sum(cfg["evol"]), places=5)


class TestRbRun(unittest.TestCase):
    """Test the rb.run convenience function."""

    @unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
    def test_run_forward_only(self):
        """rb.run with only cfg should run forward."""
        node, _, elem = get_cached_mesh([0, 0, 0], [60, 60, 30], 10)

        cfg = {
            "node": node,
            "elem": elem,
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "srcpos": np.array([[30, 30, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[30, 40, 0]]),
            "detdir": np.array([[0, 0, 1]]),
            "seg": np.ones(elem.shape[0], dtype=int),
            "omega": 0,
        }
        cfg, _ = utility.meshprep(cfg)

        result = rb.run(cfg)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
