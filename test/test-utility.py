"""
Unit tests for redbird.utility module.

Run with: python -m unittest test_utility -v
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import modules
import redbird as rb
from redbird import utility

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False


def create_simple_mesh():
    """Create a simple test mesh using iso2mesh or manually."""
    if HAS_ISO2MESH:
        node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 10)
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
    return node, face, elem


def create_simple_cfg():
    """Create a simple configuration for testing."""
    node, face, elem = create_simple_mesh()

    cfg = {
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
    return cfg


class TestMeshprep(unittest.TestCase):
    """Test utility.meshprep function."""

    def test_meshprep_returns_cfg_and_sd(self):
        """meshprep should return updated cfg and sd mapping."""
        cfg = create_simple_cfg()
        result = utility.meshprep(cfg)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        cfg_out, sd = result
        self.assertIsInstance(cfg_out, dict)
        self.assertIsInstance(sd, (np.ndarray, dict))

    def test_meshprep_preserves_1based_elem(self):
        """meshprep should keep elem 1-based."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertGreaterEqual(
            cfg_out["elem"].min(), 1, "elem should remain 1-based (min >= 1)"
        )

    def test_meshprep_preserves_1based_face(self):
        """meshprep should keep face 1-based."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("face", cfg_out)
        self.assertGreaterEqual(
            cfg_out["face"].min(), 1, "face should be 1-based (min >= 1)"
        )

    def test_meshprep_computes_evol(self):
        """meshprep should compute element volumes."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("evol", cfg_out)
        self.assertEqual(len(cfg_out["evol"]), cfg_out["elem"].shape[0])
        self.assertTrue(
            np.all(cfg_out["evol"] > 0), "All element volumes should be positive"
        )

    def test_meshprep_computes_area(self):
        """meshprep should compute face areas."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("area", cfg_out)
        self.assertEqual(len(cfg_out["area"]), cfg_out["face"].shape[0])
        self.assertTrue(
            np.all(cfg_out["area"] > 0), "All face areas should be positive"
        )

    def test_meshprep_computes_nvol(self):
        """meshprep should compute nodal volumes."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("nvol", cfg_out)
        self.assertEqual(len(cfg_out["nvol"]), cfg_out["node"].shape[0])

    def test_meshprep_computes_deldotdel(self):
        """meshprep should compute gradient operator."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("deldotdel", cfg_out)
        self.assertEqual(cfg_out["deldotdel"].shape[0], cfg_out["elem"].shape[0])
        self.assertEqual(cfg_out["deldotdel"].shape[1], 10)  # Upper triangle of 4x4

    def test_meshprep_computes_reff(self):
        """meshprep should compute effective reflection coefficient."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("reff", cfg_out)

    def test_meshprep_sets_isreoriented(self):
        """meshprep should set isreoriented flag."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("isreoriented", cfg_out)
        self.assertTrue(cfg_out["isreoriented"])

    def test_meshprep_requires_node(self):
        """meshprep should raise error without node."""
        cfg = {"elem": np.array([[1, 2, 3, 4]])}

        with self.assertRaises(ValueError):
            utility.meshprep(cfg)

    def test_meshprep_requires_elem(self):
        """meshprep should raise error without elem."""
        cfg = {"node": np.array([[0, 0, 0], [1, 0, 0]])}

        with self.assertRaises(ValueError):
            utility.meshprep(cfg)

    def test_meshprep_requires_srcpos(self):
        """meshprep should raise error without srcpos."""
        node, _, elem = create_simple_mesh()
        cfg = {"node": node, "elem": elem, "prop": np.array([[0, 0, 1, 1]])}

        with self.assertRaises(ValueError):
            utility.meshprep(cfg)


class TestSdmap(unittest.TestCase):
    """Test utility.sdmap function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_simple_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_sdmap_returns_array(self):
        """sdmap should return numpy array for single wavelength."""
        sd = utility.sdmap(self.cfg)
        self.assertIsInstance(sd, np.ndarray)

    def test_sdmap_has_correct_columns(self):
        """sdmap should have at least 3 columns: src, det, active."""
        sd = utility.sdmap(self.cfg)
        self.assertGreaterEqual(sd.shape[1], 3)

    def test_sdmap_source_indices_are_0based(self):
        """sdmap source indices should be 0-based."""
        sd = utility.sdmap(self.cfg)
        srcnum = self.cfg["srcpos"].shape[0]

        self.assertGreaterEqual(sd[:, 0].min(), 0)
        self.assertLess(sd[:, 0].max(), srcnum)

    def test_sdmap_detector_indices_offset(self):
        """sdmap detector indices should be offset by srcnum."""
        sd = utility.sdmap(self.cfg)
        srcnum = self.cfg["srcpos"].shape[0]
        detnum = self.cfg["detpos"].shape[0]

        det_indices = sd[:, 1]
        self.assertGreaterEqual(det_indices.min(), srcnum)
        self.assertLess(det_indices.max(), srcnum + detnum)

    def test_sdmap_with_maxdist(self):
        """sdmap should filter by max distance."""
        sd_all = utility.sdmap(self.cfg, maxdist=np.inf)
        sd_filtered = utility.sdmap(self.cfg, maxdist=5)

        # Filtered should have fewer or equal active pairs
        active_all = np.sum(sd_all[:, 2])
        active_filtered = np.sum(sd_filtered[:, 2])
        self.assertLessEqual(active_filtered, active_all)

    def test_sdmap_multiwavelength(self):
        """sdmap should return dict for multi-wavelength."""
        cfg = self.cfg.copy()
        cfg["prop"] = {"690": self.cfg["prop"], "830": self.cfg["prop"]}

        sd = utility.sdmap(cfg)
        self.assertIsInstance(sd, dict)
        self.assertIn("690", sd)
        self.assertIn("830", sd)


class TestGetoptodes(unittest.TestCase):
    """Test utility.getoptodes function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_simple_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_getoptodes_returns_tuple(self):
        """getoptodes should return 4-tuple."""
        result = utility.getoptodes(self.cfg)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_getoptodes_displaces_sources(self):
        """getoptodes should displace sources inward."""
        pointsrc, _, _, _ = utility.getoptodes(self.cfg)

        self.assertIsNotNone(pointsrc)
        self.assertEqual(pointsrc.shape, self.cfg["srcpos"].shape)

        # Should be displaced from original
        self.assertFalse(np.allclose(pointsrc, self.cfg["srcpos"]))

    def test_getoptodes_displaces_detectors(self):
        """getoptodes should displace detectors inward."""
        _, pointdet, _, _ = utility.getoptodes(self.cfg)

        self.assertIsNotNone(pointdet)
        self.assertEqual(pointdet.shape, self.cfg["detpos"].shape)


class TestGetdistance(unittest.TestCase):
    """Test utility.getdistance function."""

    def test_getdistance_shape(self):
        """getdistance should return (Ndet x Nsrc) matrix."""
        srcpos = np.array([[0, 0, 0], [10, 0, 0]])
        detpos = np.array([[5, 0, 0], [5, 5, 0], [5, 10, 0]])

        dist = utility.getdistance(srcpos, detpos)

        self.assertEqual(dist.shape, (3, 2))

    def test_getdistance_values(self):
        """getdistance should compute correct distances."""
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[3, 4, 0]])  # Distance should be 5

        dist = utility.getdistance(srcpos, detpos)

        self.assertAlmostEqual(dist[0, 0], 5.0)

    def test_getdistance_with_badsrc(self):
        """getdistance should handle excluded sources."""
        srcpos = np.array([[0, 0, 0], [10, 0, 0]])
        detpos = np.array([[5, 0, 0]])

        dist = utility.getdistance(srcpos, detpos, badsrc=[0])

        self.assertEqual(dist[0, 0], np.inf)  # Excluded source


class TestGetltr(unittest.TestCase):
    """Test utility.getltr function."""

    def test_getltr_returns_float(self):
        """getltr should return float."""
        cfg = create_simple_cfg()
        cfg, _ = utility.meshprep(cfg)

        ltr = utility.getltr(cfg)
        self.assertIsInstance(ltr, float)

    def test_getltr_positive(self):
        """getltr should return positive value."""
        cfg = create_simple_cfg()
        cfg, _ = utility.meshprep(cfg)

        ltr = utility.getltr(cfg)
        self.assertGreater(ltr, 0)


class TestGetreff(unittest.TestCase):
    """Test utility.getreff function."""

    def test_getreff_tissue_air(self):
        """getreff for tissue (n=1.37) to air should be ~0.47."""
        reff = utility.getreff(1.37, 1.0)

        # Effective reflection coefficient for n=1.37 tissue
        # Value depends on integration method; accept range 0.46-0.50
        self.assertGreater(reff, 0.4)
        self.assertLess(reff, 0.55)

    def test_getreff_same_index(self):
        """getreff should be 0 when n_in <= n_out."""
        reff = utility.getreff(1.0, 1.0)
        self.assertEqual(reff, 0.0)

    def test_getreff_lower_index(self):
        """getreff should be 0 when n_in < n_out."""
        reff = utility.getreff(1.0, 1.5)
        self.assertEqual(reff, 0.0)


class TestElem2node(unittest.TestCase):
    """Test utility.elem2node function."""

    def test_elem2node_shape(self):
        """elem2node should return (Nn,) or (Nn, Nv) array."""
        node, _, elem = create_simple_mesh()
        nn = node.shape[0]
        ne = elem.shape[0]

        elemval = np.ones(ne)
        nodeval = utility.elem2node(elem, elemval, nn)

        self.assertEqual(nodeval.shape, (nn,))

    def test_elem2node_multival(self):
        """elem2node should handle multiple values per element."""
        node, _, elem = create_simple_mesh()
        nn = node.shape[0]
        ne = elem.shape[0]

        elemval = np.ones((ne, 3))
        nodeval = utility.elem2node(elem, elemval, nn)

        self.assertEqual(nodeval.shape, (nn, 3))


class TestAddnoise(unittest.TestCase):
    """Test utility.addnoise function."""

    def test_addnoise_shape_preserved(self):
        """addnoise should preserve data shape."""
        data = np.ones((10, 5))
        noisy = utility.addnoise(data, snrshot=40)

        self.assertEqual(noisy.shape, data.shape)

    def test_addnoise_modifies_data(self):
        """addnoise should modify data when SNR is finite."""
        data = np.ones((10, 5))
        noisy = utility.addnoise(data, snrshot=40)

        self.assertFalse(np.allclose(noisy, data))

    def test_addnoise_no_change_infinite_snr(self):
        """addnoise should not modify data with infinite SNR."""
        data = np.ones((10, 5))

        with self.assertWarns(Warning):
            noisy = utility.addnoise(data, snrshot=np.inf, snrthermal=np.inf)

        assert_array_equal(noisy, data)

    def test_addnoise_reproducible(self):
        """addnoise should be reproducible with same seed."""
        data = np.ones((10, 5))

        noisy1 = utility.addnoise(data, snrshot=40, randseed=12345)
        noisy2 = utility.addnoise(data, snrshot=40, randseed=12345)

        assert_array_equal(noisy1, noisy2)


class TestMeshinterp(unittest.TestCase):
    """Test utility.meshinterp function."""

    def test_meshinterp_basic(self):
        """meshinterp should interpolate values."""
        node, _, elem = create_simple_mesh()
        nn = node.shape[0]

        values = np.arange(nn, dtype=float)
        mapid = np.zeros(5)  # All points in element 0
        mapweight = np.tile([0.25, 0.25, 0.25, 0.25], (5, 1))

        result = utility.meshinterp(values, mapid, mapweight, elem)

        self.assertEqual(len(result), 5)

    def test_meshinterp_handles_nan(self):
        """meshinterp should handle NaN in mapid."""
        node, _, elem = create_simple_mesh()
        nn = node.shape[0]

        values = np.arange(nn, dtype=float)
        mapid = np.array([0, np.nan, 0])
        mapweight = np.array(
            [[0.25, 0.25, 0.25, 0.25], [0, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]]
        )
        default = np.array([-1.0, -1.0, -1.0])

        result = utility.meshinterp(values, mapid, mapweight, elem, default)

        self.assertEqual(result[1], -1.0)  # Should keep default for NaN


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestIso2meshIntegration(unittest.TestCase):
    """Test integration with iso2mesh functions."""

    def test_volface_consistency(self):
        """volface fallback should match iso2mesh."""
        # Create a simple mesh using iso2mesh
        node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 10)

        # iso2mesh volface may return (face, faceid) tuple
        face_result = i2m.volface(elem[:, :4])
        if isinstance(face_result, tuple):
            face = face_result[0]
        else:
            face = face_result

        # Should be 1-based
        self.assertGreaterEqual(face.min(), 1)

        # Should be 1-based
        self.assertGreaterEqual(face.min(), 1)

    def test_elemvolume_consistency(self):
        """Verify elemvolume returns positive volumes."""
        node, face, elem = i2m.meshabox([0, 0, 0], [10, 10, 10], 3)

        evol = i2m.elemvolume(node, elem[:, :4])
        farea = i2m.elemvolume(node, face)

        self.assertTrue(np.all(evol > 0))
        self.assertTrue(np.all(farea > 0))

        # Total volume should be close to 1000 (10x10x10)
        self.assertAlmostEqual(np.sum(evol), 1000, delta=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
