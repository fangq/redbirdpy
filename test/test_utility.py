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
from redbirdpy import utility

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
    _CACHED_MESH = (node.copy(), face.copy(), elem.copy())


def create_simple_mesh():
    """Return a copy of the cached mesh."""
    global _CACHED_MESH
    if _CACHED_MESH is None:
        setUpModule()
    node, face, elem = _CACHED_MESH
    return node.copy(), face.copy(), elem.copy()


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

    def test_meshprep_per_label_boundary(self):
        """Two exposed media must get their own Reff and sinking depth."""
        import iso2mesh as i2m

        no, fc, seeds = i2m.latticegrid([0, 30, 60], [0, 60], [0, 30])
        node, elem, _ = i2m.s2m(no, fc, 1, 20.0, "tetgen", seeds)
        seg = elem[:, 4].astype(int)
        n1, n2, musp1, musp2, mua = 1.33, 1.55, 0.5, 3.0, 0.006
        cfg = {
            "node": node,
            "elem": elem[:, :4],
            "seg": seg,
            "prop": np.array([[0, 0, 1, 1], [mua, musp1, 0, n1], [mua, musp2, 0, n2]]),
            "srcpos": np.array([[10.0, 30.0, 0.0], [50.0, 30.0, 0.0]]),
            "srcdir": [0, 0, 1],
            "detpos": np.array([[20.0, 30.0, 0.0]]),
            "detdir": [0, 0, 1],
            "omega": 0,
        }
        cfg, _ = utility.meshprep(cfg)

        # Reff is per boundary face, matching the medium behind each face
        fseg = utility.faceseg(cfg)
        reff = np.asarray(cfg["reff"])
        self.assertEqual(reff.size, cfg["face"].shape[0])
        for label, n in ((1, n1), (2, n2)):
            np.testing.assert_allclose(
                reff[fseg == label], utility.getreff(n, 1.0), rtol=1e-12
            )

        # each source is sunk by the l_tr of the medium it sits on
        src = utility.getoptodes(cfg)[0]
        np.testing.assert_allclose(
            src[:, 2], [1.0 / (mua + musp1), 1.0 / (mua + musp2)], rtol=1e-12
        )

    def test_meshprep_single_medium_keeps_scalar_reff(self):
        """One exposed refractive index must keep the scalar Reff."""
        cfg = create_simple_cfg()
        cfg_out, _ = utility.meshprep(cfg)

        self.assertEqual(np.ndim(cfg_out["reff"]), 0)
        self.assertIsInstance(utility.getltr(cfg_out), float)

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
        node, face, elem = create_simple_mesh()

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
        # This test needs a specific mesh size, create it separately
        node, face, elem = i2m.meshabox([0, 0, 0], [10, 10, 10], 3)

        evol = i2m.elemvolume(node, elem[:, :4])
        farea = i2m.elemvolume(node, face)

        self.assertTrue(np.all(evol > 0))
        self.assertTrue(np.all(farea > 0))

        # Total volume should be close to 1000 (10x10x10)
        self.assertAlmostEqual(np.sum(evol), 1000, delta=10)


class TestDeldotdelExtended(unittest.TestCase):
    """Extended tests for deldotdel function."""

    def setUp(self):
        self.cfg = create_simple_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_deldotdel_delphi_values(self):
        """deldotdel delphi should represent gradient basis functions."""
        ddd, delphi = utility.deldotdel(self.cfg)

        # delphi should be (3, 4, Ne)
        self.assertEqual(delphi.shape[0], 3)  # 3D coordinates
        self.assertEqual(delphi.shape[1], 4)  # 4 nodes per tet

        # Values should be finite
        self.assertTrue(np.all(np.isfinite(delphi)))

    def test_deldotdel_consistency(self):
        """deldotdel results should be consistent with element volumes."""
        ddd, delphi = utility.deldotdel(self.cfg)

        # The diagonal entries of deldotdel relate to gradient magnitudes
        # scaled by element volumes
        ne = self.cfg["elem"].shape[0]
        self.assertEqual(ddd.shape[0], ne)


class TestMeshprepExtended(unittest.TestCase):
    """Extended tests for meshprep function."""

    def test_meshprep_with_elem_5th_column(self):
        """meshprep should extract seg from elem 5th column."""
        cfg = create_simple_cfg()

        # Add segmentation as 5th column
        seg_col = np.ones(cfg["elem"].shape[0], dtype=int) * 2
        cfg["elem"] = np.column_stack([cfg["elem"][:, :4], seg_col])
        del cfg["seg"]

        cfg_out, _ = utility.meshprep(cfg)

        self.assertIn("seg", cfg_out)
        assert_array_almost_equal(cfg_out["seg"], seg_col)

    def test_meshprep_with_list_inputs(self):
        """meshprep should convert list inputs to arrays."""
        cfg = create_simple_cfg()
        cfg["srcpos"] = [[30, 30, 0]]
        cfg["srcdir"] = [[0, 0, 1]]
        cfg["detpos"] = [[30, 40, 0]]
        cfg["detdir"] = [[0, 0, 1]]

        cfg_out, _ = utility.meshprep(cfg)

        self.assertIsInstance(cfg_out["srcpos"], np.ndarray)
        self.assertIsInstance(cfg_out["detpos"], np.ndarray)

    def test_meshprep_multiwavelength_reff(self):
        """meshprep should compute reff for each wavelength."""
        cfg = create_simple_cfg()
        cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.40]]),  # Different n
        }

        cfg_out, _ = utility.meshprep(cfg)

        self.assertIsInstance(cfg_out["reff"], dict)
        self.assertIn("690", cfg_out["reff"])
        self.assertIn("830", cfg_out["reff"])


class TestSdmapExtended(unittest.TestCase):
    """Extended tests for sdmap function."""

    def setUp(self):
        self.cfg = create_simple_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_sdmap_exclude_sources(self):
        """sdmap should handle excluded sources."""
        sd = utility.sdmap(self.cfg, excludesrc=[0])

        # First source should be excluded
        self.assertTrue(np.all(sd[sd[:, 0] == 0, 2] == 0))

    def test_sdmap_exclude_detectors(self):
        """sdmap should handle excluded detectors."""
        srcnum = self.cfg["srcpos"].shape[0]
        det_offset = srcnum

        sd = utility.sdmap(self.cfg, excludedet=[0])

        # First detector should be excluded
        det_col = det_offset  # Detector 0's column index
        self.assertTrue(np.all(sd[sd[:, 1] == det_col, 2] == 0))

    def test_sdmap_no_sources_raises(self):
        """sdmap should raise error with no sources."""
        cfg = self.cfg.copy()
        cfg["srcpos"] = np.array([]).reshape(0, 3)
        # Don't call meshprep - directly test sdmap
        cfg["widesrc"] = None
        cfg["widedet"] = None

        with self.assertRaises(ValueError):
            utility.sdmap(cfg)


class TestGetoptdesExtended(unittest.TestCase):
    """Extended tests for getoptodes function."""

    def setUp(self):
        self.cfg = create_simple_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_getoptodes_with_widesrc(self):
        """getoptodes should return widesrc if present."""
        nn = self.cfg["node"].shape[0]
        self.cfg["widesrc"] = np.random.rand(nn, 2)

        pointsrc, pointdet, widesrc, widedet = utility.getoptodes(self.cfg)

        self.assertIsNotNone(widesrc)
        self.assertEqual(widesrc.shape, (nn, 2))

    def test_getoptodes_with_widedet(self):
        """getoptodes should return widedet if present."""
        nn = self.cfg["node"].shape[0]
        self.cfg["widedet"] = np.random.rand(nn, 3)

        pointsrc, pointdet, widesrc, widedet = utility.getoptodes(self.cfg)

        self.assertIsNotNone(widedet)
        self.assertEqual(widedet.shape, (nn, 3))

    def test_getoptodes_multiwavelength(self):
        """getoptodes should handle multi-wavelength."""
        cfg = self.cfg.copy()
        cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
        }
        cfg, _ = utility.meshprep(cfg)

        pointsrc, pointdet, widesrc, widedet = utility.getoptodes(cfg, wv="690")

        self.assertIsNotNone(pointsrc)


class TestGetdistanceExtended(unittest.TestCase):
    """Extended tests for getdistance function."""

    def test_getdistance_with_widesrc(self):
        """getdistance should handle wide-field sources."""
        srcpos = np.array([[0, 0, 0], [10, 0, 0]])
        detpos = np.array([[5, 0, 0]])
        widesrc = np.array([[0, 0, 5], [10, 0, 5]])  # 2 wide-field sources

        dist = utility.getdistance(srcpos, detpos, widesrc=widesrc)

        # Should have columns for point + wide sources
        self.assertEqual(dist.shape, (1, 4))

    def test_getdistance_symmetric(self):
        """getdistance should give symmetric-like results for symmetric config."""
        srcpos = np.array([[0, 0, 0], [10, 10, 0]])
        detpos = np.array([[10, 10, 0], [0, 0, 0]])

        dist = utility.getdistance(srcpos, detpos)

        # d[0,1] should equal d[1,0] (src1-det0 vs src0-det1)
        self.assertAlmostEqual(dist[0, 1], dist[1, 0])


class TestGetreffExtended(unittest.TestCase):
    """Extended tests for getreff function."""

    def test_getreff_high_index(self):
        """getreff for high refractive index ratio."""
        reff = utility.getreff(2.0, 1.0)

        self.assertGreater(reff, 0.5)

    def test_getreff_slightly_higher(self):
        """getreff for slightly higher internal index."""
        reff = utility.getreff(1.01, 1.0)

        self.assertGreater(reff, 0)
        self.assertLess(reff, 0.1)


class TestElem2nodeExtended(unittest.TestCase):
    """Extended tests for elem2node function."""

    def test_elem2node_with_cfg_dict(self):
        """elem2node should accept cfg dict as first argument."""
        cfg = create_simple_cfg()
        cfg, _ = utility.meshprep(cfg)

        ne = cfg["elem"].shape[0]
        elemval = np.ones(ne)

        nodeval = utility.elem2node(cfg, elemval)

        self.assertEqual(len(nodeval), cfg["node"].shape[0])

    def test_elem2node_preserves_dtype(self):
        """elem2node should handle different dtypes."""
        cfg = create_simple_cfg()
        cfg, _ = utility.meshprep(cfg)

        ne = cfg["elem"].shape[0]
        elemval = np.ones(ne, dtype=np.float32)
        nn = cfg["node"].shape[0]

        nodeval = utility.elem2node(cfg["elem"], elemval, nn)

        # Should work regardless of input dtype
        self.assertTrue(np.all(np.isfinite(nodeval)))


class TestAddnoiseExtended(unittest.TestCase):
    """Extended tests for addnoise function."""

    def test_addnoise_complex_data(self):
        """addnoise should handle complex data."""
        data = np.ones((5, 3)) + 1j * np.ones((5, 3))

        noisy = utility.addnoise(data, snrshot=30 + 20j, randseed=42)

        self.assertTrue(np.iscomplexobj(noisy))
        self.assertFalse(np.allclose(noisy, data))

    def test_addnoise_thermal_only(self):
        """addnoise with only thermal noise."""
        data = np.ones((5, 3)) * 100

        noisy = utility.addnoise(data, snrshot=np.inf, snrthermal=40, randseed=42)

        self.assertFalse(np.allclose(noisy, data))

    def test_addnoise_snr_matches_request(self):
        """Requested snrshot is the SNR of the strongest channel, and is
        invariant to the scale/units of the data."""
        amp = np.logspace(0, -2, 3)[:, None]

        for scale in (1.0, 1e-6):
            data = np.tile(scale * amp, (1, 4000))
            noisy = utility.addnoise(data, snrshot=60, randseed=11)

            for row, expected in enumerate((60.0, 50.0, 40.0)):
                sig = data[row, 0]
                snr = 20 * np.log10(sig / np.std(noisy[row] - data[row]))
                self.assertLess(abs(snr - expected), 1.5)

    def test_addnoise_thermal_floor_level(self):
        """Thermal noise sigma is max|data| * 10**(-snrthermal/20)."""
        data = np.full((1, 8000), 1e-6)

        noisy = utility.addnoise(data, snrshot=np.inf, snrthermal=40, randseed=13)

        self.assertAlmostEqual(np.std(noisy - data) / 1e-6, 0.01, places=3)

    def test_addnoise_complex_real_valued_snr(self):
        """A real-valued snr must work for complex data: amplitude noise
        matches the CW case and phase error follows sigma/|data|."""
        dcw = np.full(8000, 1e-6)
        dfd = np.full(8000, 1e-6 * np.exp(1j * 0.7))

        ncw = utility.addnoise(dcw, snrshot=50, randseed=21)
        nfd = utility.addnoise(dfd, snrshot=50, randseed=21)

        self.assertTrue(np.iscomplexobj(nfd))

        ratio = np.std(np.abs(nfd) - 1e-6) / np.std(ncw - dcw)
        self.assertLess(abs(ratio - 1), 0.1)

        sigma_phase = np.std(np.angle(nfd) - 0.7)
        self.assertLess(abs(sigma_phase / 10 ** (-50 / 20) - 1), 0.1)

    def test_addnoise_complex_noise_is_circular(self):
        """Noise must be isotropic about the signal phasor, not pinned to
        the real axis."""
        dfd = np.full(8000, 1e-6 * np.exp(1j * 0.7))

        nfd = utility.addnoise(dfd, snrshot=50, randseed=21)
        err = (nfd - dfd) * np.exp(-1j * 0.7)

        self.assertLess(abs(np.std(err.imag) / np.std(err.real) - 1), 0.15)

    def test_addnoise_imag_snr_adds_phase_jitter(self):
        """The imaginary part of the snr sets extra phase jitter, in radian."""
        dfd = np.full(8000, 1e-6 * np.exp(1j * 0.7))

        noisy = utility.addnoise(dfd, snrshot=complex(np.inf, 40), randseed=22)

        self.assertLess(abs(np.std(np.angle(noisy) - 0.7) / 0.01 - 1), 0.1)

    def test_addnoise_leaves_global_rng_alone(self):
        """addnoise must not reseed the process-wide numpy RNG."""
        np.random.seed(7)
        before = np.random.rand()

        np.random.seed(7)
        utility.addnoise(np.full(10, 1e-6), snrshot=40)
        after = np.random.rand()

        self.assertEqual(before, after)


class TestMeshinterpExtended(unittest.TestCase):
    """Extended tests for meshinterp function."""

    def test_meshinterp_2d_values(self):
        """meshinterp should handle 2D value arrays."""
        cfg = create_simple_cfg()
        cfg, _ = utility.meshprep(cfg)

        nn = cfg["node"].shape[0]
        values = np.column_stack([np.arange(nn), np.arange(nn) * 2])

        # Simple identity mapping for testing
        mapid = np.ones(5)  # All in element 1
        mapweight = np.tile([0.25, 0.25, 0.25, 0.25], (5, 1))

        result = utility.meshinterp(values, mapid, mapweight, cfg["elem"])

        self.assertEqual(result.shape, (5, 2))

    def test_meshinterp_with_toval(self):
        """meshinterp should use toval as default."""
        cfg = create_simple_cfg()
        cfg, _ = utility.meshprep(cfg)

        nn = cfg["node"].shape[0]
        values = np.arange(nn, dtype=float)

        mapid = np.array([1, np.nan, 1])
        mapweight = np.array(
            [[0.25, 0.25, 0.25, 0.25], [0, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]]
        )
        toval = np.array([-999.0, -999.0, -999.0])

        result = utility.meshinterp(values, mapid, mapweight, cfg["elem"], toval)

        # NaN location should keep default value
        self.assertEqual(result[1], -999.0)


class TestForcearrayExtended(unittest.TestCase):
    """Extended tests for forcearray function."""

    def test_forcearray_multiple_keys(self):
        """forcearray should convert multiple keys."""
        cfg = {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": np.array([7, 8, 9]),  # Already array
        }

        result = utility.forcearray(cfg, ["a", "b", "c"])

        self.assertIsInstance(result["a"], np.ndarray)
        self.assertIsInstance(result["b"], np.ndarray)
        self.assertIsInstance(result["c"], np.ndarray)

    def test_forcearray_missing_keys(self):
        """forcearray should ignore missing keys."""
        cfg = {"a": [1, 2, 3]}

        result = utility.forcearray(cfg, ["a", "nonexistent"])

        self.assertIsInstance(result["a"], np.ndarray)
        self.assertNotIn("nonexistent", result)


class TestFallbackImplementations(unittest.TestCase):
    """Test fallback implementations when iso2mesh is not available."""

    def test_meshreorient_fallback(self):
        """_meshreorient_fallback should reorient negative volume elements."""
        node = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        # Element with wrong orientation (negative volume)
        elem = np.array([[1, 3, 2, 4]], dtype=int)  # 1-based

        elem_new = utility._meshreorient_fallback(node, elem)

        # Check that volume is now positive
        elem_0 = elem_new[0, :4] - 1
        n = node[elem_0, :]
        v1, v2, v3 = n[1] - n[0], n[2] - n[0], n[3] - n[0]
        vol = np.dot(np.cross(v1, v2), v3)

        self.assertGreater(vol, 0)

    def test_volface_fallback(self):
        """_volface_fallback should extract boundary faces."""
        # Simple tetrahedron
        elem = np.array([[1, 2, 3, 4]], dtype=int)  # 1-based

        face = utility._volface_fallback(elem)

        # Single tet should have 4 boundary faces
        self.assertEqual(face.shape[0], 4)
        self.assertEqual(face.shape[1], 3)
        self.assertGreaterEqual(face.min(), 1)  # 1-based

    def test_elemvolume_fallback_tet(self):
        """_elemvolume_fallback should compute tet volumes."""
        node = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        elem = np.array([[1, 2, 3, 4]], dtype=int)  # 1-based

        vol = utility._elemvolume_fallback(node, elem)

        # Volume of unit tet is 1/6
        self.assertAlmostEqual(vol[0], 1 / 6, places=10)

    def test_elemvolume_fallback_tri(self):
        """_elemvolume_fallback should compute triangle areas."""
        node = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=float,
        )
        face = np.array([[1, 2, 3]], dtype=int)  # 1-based

        area = utility._elemvolume_fallback(node, face)

        # Area of right triangle with legs 1,1 is 0.5
        self.assertAlmostEqual(area[0], 0.5, places=10)


class TestNodevolume(unittest.TestCase):
    """Test _nodevolume function."""

    def test_nodevolume_conservation(self):
        """Sum of nodal volumes should equal sum of element volumes."""
        cfg = create_simple_cfg()
        cfg, _ = utility.meshprep(cfg)

        total_evol = np.sum(cfg["evol"])
        total_nvol = np.sum(cfg["nvol"])

        self.assertAlmostEqual(total_nvol, total_evol, places=5)


class TestFemnz(unittest.TestCase):
    """Test _femnz function."""

    def test_femnz_connectivity(self):
        """_femnz should return correct connectivity structure."""
        elem = np.array(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
            ],
            dtype=int,
        )  # 1-based
        nn = 5

        rows, cols, connnum = utility._femnz(elem, nn)

        # Node 1 connects to 2, 3, 4 (3 connections)
        # Node 2 connects to 1, 3, 4, 5 (4 connections)
        # etc.
        self.assertEqual(len(connnum), nn)
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), len(cols))


if __name__ == "__main__":
    unittest.main(verbosity=2)
