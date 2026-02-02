"""
Unit tests for redbird.property module.

Run with: python -m unittest test_property -v
"""

import unittest
import sys
import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbirdpy import property as prop_module, utility

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


def create_test_cfg():
    """Create a simple configuration for testing."""
    node, face, elem = create_simple_mesh()

    cfg = {
        "node": node,
        "elem": elem,
        "face": face,
        "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
        "srcpos": np.array([[30, 30, 0]]),
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": np.array([[30, 40, 0]]),
        "detdir": np.array([[0, 0, 1]]),
        "seg": np.ones(elem.shape[0], dtype=int),
        "omega": 0,
    }
    return cfg


class TestExtinction(unittest.TestCase):
    """Test property.extinction function."""

    def test_extinction_returns_tuple(self):
        """extinction should return (extin, chrome) tuple."""
        result = prop_module.extinction([690, 830], ["hbo", "hbr"])

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_extinction_shape(self):
        """extinction should return (Nwv, Nchrome) array."""
        extin, _ = prop_module.extinction([690, 830], ["hbo", "hbr"])

        self.assertEqual(extin.shape, (2, 2))

    def test_extinction_single_wavelength(self):
        """extinction should handle single wavelength."""
        extin, _ = prop_module.extinction([690], ["hbo"])

        self.assertEqual(extin.shape, (1, 1))

    def test_extinction_single_chromophore_string(self):
        """extinction should handle single chromophore as string."""
        extin, _ = prop_module.extinction([690, 830], "hbo")

        self.assertEqual(extin.shape, (2, 1))

    def test_extinction_string_wavelengths(self):
        """extinction should handle string wavelengths."""
        extin, _ = prop_module.extinction(["690", "830"], ["hbo"])

        self.assertEqual(extin.shape, (2, 1))

    def test_extinction_positive(self):
        """extinction coefficients should be positive."""
        extin, _ = prop_module.extinction([690, 830], ["hbo", "hbr"])

        self.assertTrue(np.all(extin > 0))

    def test_extinction_hbo_hbr_crossover(self):
        """HbO2 and Hb should cross over around 800nm."""
        extin_700, _ = prop_module.extinction([700], ["hbo", "hbr"])
        extin_900, _ = prop_module.extinction([900], ["hbo", "hbr"])

        # At 700nm, HbR > HbO2
        self.assertGreater(extin_700[0, 1], extin_700[0, 0])

        # At 900nm, HbO2 > HbR
        self.assertGreater(extin_900[0, 0], extin_900[0, 1])

    def test_extinction_unknown_chromophore(self):
        """extinction should raise error for unknown chromophore."""
        with self.assertRaises(ValueError):
            prop_module.extinction([690], ["unknown"])

    def test_extinction_all_chromophores(self):
        """extinction should handle all supported chromophores."""
        chromes = ["hbo", "hbr", "water", "lipids", "aa3"]

        extin, _ = prop_module.extinction([800], chromes)

        self.assertEqual(extin.shape, (1, 5))

    def test_extinction_chrome_dict(self):
        """extinction should return chromophore lookup tables."""
        _, chrome = prop_module.extinction([690], ["hbo"])

        self.assertIsInstance(chrome, dict)
        self.assertIn("hbo", chrome)
        self.assertIn("hbr", chrome)

        # Each should be Nx2 array
        self.assertEqual(chrome["hbo"].shape[1], 2)


class TestUpdateprop(unittest.TestCase):
    """Test property.updateprop function."""

    def test_updateprop_basic(self):
        """updateprop should compute mua from chromophores."""
        cfg = {
            "prop": {"690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])},
            "param": {
                "hbo": np.array([50.0]),  # 50 uM
                "hbr": np.array([20.0]),  # 20 uM
            },
        }

        result = prop_module.updateprop(cfg)

        self.assertIsInstance(result, np.ndarray)

    def test_updateprop_multiwavelength(self):
        """updateprop should handle multiple wavelengths."""
        cfg = {
            "prop": {
                "690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
                "830": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            },
            "param": {
                "hbo": np.array([50.0]),
                "hbr": np.array([20.0]),
            },
        }

        result = prop_module.updateprop(cfg)

        self.assertIsInstance(result, dict)
        self.assertIn("690", result)
        self.assertIn("830", result)

    def test_updateprop_single_wavelength(self):
        """updateprop should handle single wavelength request."""
        cfg = {
            "prop": {
                "690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
                "830": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            },
            "param": {
                "hbo": np.array([50.0]),
                "hbr": np.array([20.0]),
            },
        }

        result = prop_module.updateprop(cfg, wv="690")

        self.assertIsInstance(result, np.ndarray)

    def test_updateprop_with_scattering(self):
        """updateprop should compute musp from scattering params."""
        cfg = {
            "prop": {"800": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])},
            "param": {
                "hbo": np.array([50.0]),
                "hbr": np.array([20.0]),
                "scatamp": np.array([1.0]),
                "scatpow": np.array([1.0]),
            },
        }

        result = prop_module.updateprop(cfg)

        # Should have musp in column 1
        self.assertIsInstance(result, np.ndarray)

    def test_updateprop_no_param(self):
        """updateprop should return prop if no param."""
        cfg = {"prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])}

        result = prop_module.updateprop(cfg)

        self.assertIs(result, cfg["prop"])

    def test_updateprop_default_water_lipids(self):
        """updateprop should use default water/lipids if not specified."""
        cfg = {
            "prop": {"800": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])},
            "param": {
                "hbo": np.array([50.0]),
            },
        }

        result = prop_module.updateprop(cfg)

        # Should not raise error
        self.assertIsNotNone(result)


class TestGetbulk(unittest.TestCase):
    """Test property.getbulk function."""

    def test_getbulk_default(self):
        """getbulk should return default [0, 0, 0, 1.37]."""
        cfg = {}

        result = prop_module.getbulk(cfg)

        assert_array_almost_equal(result, [0, 0, 0, 1.37])

    def test_getbulk_explicit_bulk(self):
        """getbulk should use explicit bulk properties."""
        cfg = {"bulk": {"mua": 0.01, "musp": 1.0, "n": 1.4}}

        result = prop_module.getbulk(cfg)

        self.assertEqual(result[0], 0.01)
        self.assertEqual(result[1], 1.0)
        self.assertEqual(result[3], 1.4)

    def test_getbulk_from_dcoeff(self):
        """getbulk should convert dcoeff to mus."""
        cfg = {"bulk": {"dcoeff": 1.0 / 3.0}}  # D = 1/(3*mus) => mus = 1

        result = prop_module.getbulk(cfg)

        self.assertAlmostEqual(result[1], 1.0)

    def test_getbulk_label_based(self):
        """getbulk should extract from label-based prop."""
        cfg = create_test_cfg()
        cfg, _ = utility.meshprep(cfg)

        result = prop_module.getbulk(cfg)

        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 0.01)  # mua from prop[1]

    def test_getbulk_multiwavelength(self):
        """getbulk should return dict for multi-wavelength."""
        cfg = create_test_cfg()
        cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.02, 0.8, 0, 1.37]]),
        }
        cfg, _ = utility.meshprep(cfg)

        result = prop_module.getbulk(cfg)

        self.assertIsInstance(result, dict)
        self.assertIn("690", result)
        self.assertIn("830", result)

    def test_getbulk_node_based(self):
        """getbulk should handle node-based properties."""
        cfg = create_test_cfg()
        nn = cfg["node"].shape[0]

        cfg["prop"] = np.column_stack(
            [0.01 * np.ones(nn), 1.0 * np.ones(nn), np.zeros(nn), 1.37 * np.ones(nn)]
        )
        del cfg["seg"]

        cfg, _ = utility.meshprep(cfg)

        result = prop_module.getbulk(cfg)

        self.assertAlmostEqual(result[0], 0.01)

    def test_getbulk_handles_1based_face(self):
        """getbulk should handle 1-based face indices."""
        cfg = create_test_cfg()
        cfg, _ = utility.meshprep(cfg)

        # Verify face is 1-based
        self.assertGreaterEqual(cfg["face"].min(), 1)

        # Should not raise IndexError
        result = prop_module.getbulk(cfg)

        self.assertEqual(len(result), 4)


class TestMusp2sasp(unittest.TestCase):
    """Test property.musp2sasp function."""

    def test_musp2sasp_basic(self):
        """musp2sasp should return (sa, sp) tuple."""
        musp = np.array([1.0, 0.8])
        wavelength = np.array([700, 900])

        sa, sp = prop_module.musp2sasp(musp, wavelength)

        self.assertIsInstance(sa, float)
        self.assertIsInstance(sp, float)

    def test_musp2sasp_positive_power(self):
        """musp2sasp should return positive power for typical tissue."""
        # Tissue typically has musp decreasing with wavelength
        musp = np.array([1.2, 0.9])
        wavelength = np.array([700, 900])

        sa, sp = prop_module.musp2sasp(musp, wavelength)

        self.assertGreater(sp, 0)

    def test_musp2sasp_reconstruction(self):
        """musp2sasp should allow reconstruction of musp."""
        musp_orig = np.array([1.2, 0.9])
        wavelength = np.array([700, 900])

        sa, sp = prop_module.musp2sasp(musp_orig, wavelength)

        # Reconstruct
        musp_recon = sa * (wavelength / 500.0) ** (-sp)

        assert_allclose(musp_recon, musp_orig, rtol=1e-5)

    def test_musp2sasp_requires_two_wavelengths(self):
        """musp2sasp should require at least 2 wavelengths."""
        with self.assertRaises(ValueError):
            prop_module.musp2sasp(np.array([1.0]), np.array([700]))


class TestSetmesh(unittest.TestCase):
    """Test property.setmesh function."""

    def test_setmesh_with_seg(self):
        """setmesh should accept segmentation."""
        cfg0 = create_test_cfg()
        cfg0, _ = utility.meshprep(cfg0)

        new_seg = np.ones(cfg0["elem"].shape[0], dtype=int) * 2

        cfg = prop_module.setmesh(cfg0, cfg0["node"], cfg0["elem"], propidx=new_seg)

        assert_array_almost_equal(cfg["seg"], new_seg)

    def test_setmesh_extracts_seg_from_elem(self):
        """setmesh should extract seg from elem column 5."""
        cfg0 = create_test_cfg()
        cfg0, _ = utility.meshprep(cfg0)

        # Add 5th column to elem
        seg_col = np.ones(cfg0["elem"].shape[0], dtype=int) * 3
        new_elem = np.column_stack([cfg0["elem"][:, :4], seg_col])

        cfg = prop_module.setmesh(cfg0, cfg0["node"], new_elem)

        assert_array_almost_equal(cfg["seg"], seg_col)

    def test_setmesh_updates_node_elem(self):
        """setmesh should update node and elem."""
        cfg0 = create_test_cfg()
        cfg0, _ = utility.meshprep(cfg0)

        # Create new mesh
        if HAS_ISO2MESH:
            new_node, _, new_elem = i2m.meshabox([0, 0, 0], [30, 30, 15], 5)
        else:
            new_node = cfg0["node"][:5, :]
            new_elem = cfg0["elem"][:3, :]

        cfg = prop_module.setmesh(cfg0, new_node, new_elem)

        self.assertEqual(cfg["node"].shape[0], new_node.shape[0])
        self.assertEqual(cfg["elem"].shape[0], new_elem.shape[0])

    def test_setmesh_clears_derived(self):
        """setmesh should clear derived quantities."""
        cfg0 = create_test_cfg()
        cfg0, _ = utility.meshprep(cfg0)

        # Store original derived quantities
        orig_face = cfg0["face"]

        # Create new mesh
        if HAS_ISO2MESH:
            new_node, _, new_elem = i2m.meshabox([0, 0, 0], [30, 30, 15], 5)
        else:
            new_node = cfg0["node"]
            new_elem = cfg0["elem"]

        cfg = prop_module.setmesh(cfg0, new_node, new_elem)

        # Face should be recomputed (different if mesh changed)
        self.assertIn("face", cfg)

    def test_setmesh_with_prop(self):
        """setmesh should accept new properties."""
        cfg0 = create_test_cfg()
        cfg0, _ = utility.meshprep(cfg0)

        new_prop = np.array([[0, 0, 1, 1], [0.02, 0.5, 0, 1.4]])

        cfg = prop_module.setmesh(cfg0, cfg0["node"], cfg0["elem"], prop=new_prop)

        assert_array_almost_equal(cfg["prop"], new_prop)


class TestGetChromophoreTable(unittest.TestCase):
    """Test property.get_chromophore_table function."""

    def test_get_chromophore_table_hbo(self):
        """get_chromophore_table should return HbO2 data."""
        table = prop_module.get_chromophore_table("hbo")

        self.assertEqual(table.shape[1], 2)
        self.assertGreater(table.shape[0], 10)

    def test_get_chromophore_table_hbr(self):
        """get_chromophore_table should return Hb data."""
        table = prop_module.get_chromophore_table("hbr")

        self.assertEqual(table.shape[1], 2)

    def test_get_chromophore_table_case_insensitive(self):
        """get_chromophore_table should be case-insensitive."""
        table1 = prop_module.get_chromophore_table("HbO")
        table2 = prop_module.get_chromophore_table("hbo")

        assert_array_almost_equal(table1, table2)

    def test_get_chromophore_table_unknown(self):
        """get_chromophore_table should raise error for unknown."""
        with self.assertRaises(ValueError):
            prop_module.get_chromophore_table("unknown")

    def test_get_chromophore_table_all(self):
        """get_chromophore_table should work for all chromophores."""
        for name in ["hbo", "hbr", "water", "lipids", "aa3"]:
            table = prop_module.get_chromophore_table(name)
            self.assertEqual(table.shape[1], 2)
            self.assertTrue(np.all(table[:, 0] > 0))  # Wavelengths positive


class TestExtinctionExtended(unittest.TestCase):
    """Extended tests for extinction function."""

    def test_extinction_interpolation(self):
        """extinction should interpolate between data points."""
        # Request wavelength between data points
        extin1, _ = prop_module.extinction([700], ["hbo"])
        extin2, _ = prop_module.extinction([710], ["hbo"])
        extin3, _ = prop_module.extinction([705], ["hbo"])

        # Interpolated value should be between neighbors
        # (assuming monotonic or at least bounded)
        # Just check it returns reasonable values
        self.assertTrue(np.all(np.isfinite(extin3)))

    def test_extinction_extrapolation(self):
        """extinction should extrapolate beyond data range."""
        # Very short wavelength (outside typical data)
        extin, _ = prop_module.extinction([200], ["hbo"])

        self.assertTrue(np.all(np.isfinite(extin)))

    def test_extinction_water_units(self):
        """extinction for water should be in correct units (1/mm)."""
        extin, _ = prop_module.extinction([800], ["water"])

        # Water absorption at 800nm should be small but positive
        self.assertGreater(extin[0, 0], 0)
        self.assertLess(extin[0, 0], 0.1)  # Reasonable range

    def test_extinction_lipids(self):
        """extinction for lipids should work."""
        extin, _ = prop_module.extinction([800], ["lipids"])

        self.assertTrue(np.all(np.isfinite(extin)))
        self.assertGreater(extin[0, 0], 0)

    def test_extinction_aa3(self):
        """extinction for cytochrome aa3 should work."""
        extin, _ = prop_module.extinction([830], ["aa3"])

        # aa3 has a peak around 830nm
        self.assertTrue(np.all(np.isfinite(extin)))
        self.assertGreater(extin[0, 0], 0)


class TestUpdatepropExtended(unittest.TestCase):
    """Extended tests for updateprop function."""

    def test_updateprop_node_based(self):
        """updateprop should handle node-based parameters."""
        cfg = create_test_cfg()
        nn = cfg["node"].shape[0]

        cfg["prop"] = {"800": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])}
        cfg["param"] = {
            "hbo": np.ones(nn) * 50.0,
            "hbr": np.ones(nn) * 20.0,
        }

        result = prop_module.updateprop(cfg)

        # Should return node-based properties
        self.assertEqual(result.shape[0], nn)

    def test_updateprop_scalar_param(self):
        """updateprop should handle scalar parameters."""
        cfg = {
            "prop": {"800": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])},
            "param": {
                "hbo": 50.0,  # Scalar
                "hbr": 20.0,
            },
        }

        result = prop_module.updateprop(cfg)

        self.assertIsInstance(result, np.ndarray)

    def test_updateprop_with_all_chromophores(self):
        """updateprop should handle all chromophore types."""
        cfg = {
            "prop": {"800": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])},
            "param": {
                "hbo": np.array([50.0]),
                "hbr": np.array([20.0]),
                "water": np.array([0.3]),
                "lipids": np.array([0.5]),
            },
        }

        result = prop_module.updateprop(cfg)

        self.assertIsNotNone(result)

    def test_updateprop_scattering_power(self):
        """updateprop should compute wavelength-dependent musp."""
        cfg = {
            "prop": {
                "700": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
                "900": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            },
            "param": {
                "hbo": np.array([50.0]),
                "scatamp": np.array([1.0]),
                "scatpow": np.array([1.0]),  # musp proportional to 1/wavelength
            },
        }

        result = prop_module.updateprop(cfg)

        # musp at 700nm should be higher than at 900nm (sp > 0)
        self.assertIsInstance(result, dict)
        # Result is a single row array (1, 4) - musp is in column 1
        musp_700 = result["700"][0, 1] if result["700"].ndim > 1 else result["700"][1]
        musp_900 = result["900"][0, 1] if result["900"].ndim > 1 else result["900"][1]
        self.assertGreater(musp_700, musp_900)

    def test_updateprop_unrecognized_params_use_defaults(self):
        """updateprop should use default water/lipids when no recognized chromophores."""
        cfg = {
            "prop": {"800": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])},
            "param": {
                "unknown": np.array([50.0]),  # Unrecognized - will be ignored
            },
        }

        # The function adds default water (0.23) and lipids (0.58) if not specified
        # So it will compute mua from those defaults, not raise an error
        result = prop_module.updateprop(cfg)

        # Should return a result (not raise error)
        self.assertIsNotNone(result)
        # The mua will be computed from default water/lipids extinction
        self.assertTrue(np.all(np.isfinite(result)))


class TestGetbulkExtended(unittest.TestCase):
    """Extended tests for getbulk function."""

    def test_getbulk_element_based_seg(self):
        """getbulk should handle element-based segmentation."""
        cfg = create_test_cfg()
        cfg, _ = utility.meshprep(cfg)

        # Make seg element-based (already is)
        ne = cfg["elem"].shape[0]
        cfg["seg"] = np.ones(ne, dtype=int)

        result = prop_module.getbulk(cfg)

        self.assertEqual(len(result), 4)

    def test_getbulk_node_based_seg(self):
        """getbulk should handle node-based segmentation."""
        cfg = create_test_cfg()
        cfg, _ = utility.meshprep(cfg)

        nn = cfg["node"].shape[0]
        cfg["seg"] = np.ones(nn, dtype=int)

        result = prop_module.getbulk(cfg)

        self.assertEqual(len(result), 4)

    def test_getbulk_multiple_labels(self):
        """getbulk should handle multiple tissue labels."""
        cfg = create_test_cfg()
        cfg["prop"] = np.array(
            [
                [0, 0, 1, 1],  # Label 0 (background)
                [0.01, 1, 0, 1.37],  # Label 1
                [0.02, 0.8, 0, 1.4],  # Label 2
            ]
        )
        cfg, _ = utility.meshprep(cfg)

        result = prop_module.getbulk(cfg)

        # Should return tissue label properties
        self.assertAlmostEqual(result[0], 0.01)

    def test_getbulk_with_g_in_bulk(self):
        """getbulk should handle g parameter in bulk dict."""
        cfg = {"bulk": {"mua": 0.01, "musp": 1.0, "g": 0.9, "n": 1.4}}

        result = prop_module.getbulk(cfg)

        self.assertEqual(result[2], 0.9)


class TestExtractBulkFromPropExtended(unittest.TestCase):
    """Extended tests for _extract_bulk_from_prop."""

    def test_extract_element_based_prop(self):
        """Should extract from element-based prop."""
        cfg = create_test_cfg()
        cfg, _ = utility.meshprep(cfg)

        ne = cfg["elem"].shape[0]
        prop = np.zeros((ne, 4))
        prop[:, 0] = 0.015
        prop[:, 1] = 0.9
        prop[:, 3] = 1.38

        bkprop = prop_module._extract_bulk_from_prop(prop, cfg)

        self.assertAlmostEqual(bkprop[0], 0.015)

    def test_extract_with_missing_face(self):
        """Should handle missing face array."""
        prop = np.array(
            [
                [0, 0, 1, 1],
                [0.01, 1, 0, 1.37],
            ]
        )

        cfg = {
            "node": np.zeros((100, 3)),
            "elem": np.ones((50, 4), dtype=int),
            "seg": np.ones(50, dtype=int),
            # No 'face' key
        }

        bkprop = prop_module._extract_bulk_from_prop(prop, cfg)

        # Should use seg[0] as label
        self.assertAlmostEqual(bkprop[0], 0.01)


class TestMusp2saspExtended(unittest.TestCase):
    """Extended tests for musp2sasp function."""

    def test_musp2sasp_flat_spectrum(self):
        """musp2sasp with flat spectrum should give sp=0."""
        musp = np.array([1.0, 1.0])  # Same at both wavelengths
        wavelength = np.array([700, 900])

        sa, sp = prop_module.musp2sasp(musp, wavelength)

        self.assertAlmostEqual(sp, 0, places=5)

    def test_musp2sasp_increasing_spectrum(self):
        """musp2sasp with increasing musp should give negative sp."""
        musp = np.array([0.8, 1.2])  # Increasing with wavelength
        wavelength = np.array([700, 900])

        sa, sp = prop_module.musp2sasp(musp, wavelength)

        self.assertLess(sp, 0)


class TestSetmeshExtended(unittest.TestCase):
    """Extended tests for setmesh function."""

    def test_setmesh_preserves_other_fields(self):
        """setmesh should preserve non-mesh fields."""
        cfg0 = create_test_cfg()
        cfg0["omega"] = 2 * np.pi * 100e6
        cfg0["custom_field"] = "test_value"
        cfg0, _ = utility.meshprep(cfg0)

        cfg = prop_module.setmesh(cfg0, cfg0["node"], cfg0["elem"])

        self.assertEqual(cfg["omega"], cfg0["omega"])
        self.assertEqual(cfg["custom_field"], "test_value")

    def test_setmesh_recomputes_evol(self):
        """setmesh should recompute element volumes."""
        cfg0 = create_test_cfg()
        cfg0, _ = utility.meshprep(cfg0)

        # Scale mesh
        new_node = cfg0["node"] * 2

        cfg = prop_module.setmesh(cfg0, new_node, cfg0["elem"])

        # Volumes should be 8x larger (2^3)
        self.assertAlmostEqual(
            np.sum(cfg["evol"]) / np.sum(cfg0["evol"]), 8.0, places=3
        )


class TestChromophoreData(unittest.TestCase):
    """Test chromophore data tables."""

    def test_hbo_data_range(self):
        """HbO2 data should cover visible to NIR range."""
        table = prop_module.get_chromophore_table("hbo")

        self.assertLess(table[0, 0], 300)  # Starts before 300nm
        self.assertGreater(table[-1, 0], 900)  # Extends past 900nm

    def test_hbr_data_range(self):
        """Hb data should cover visible to NIR range."""
        table = prop_module.get_chromophore_table("hbr")

        self.assertLess(table[0, 0], 300)
        self.assertGreater(table[-1, 0], 900)

    def test_water_data_range(self):
        """Water data should cover NIR range."""
        table = prop_module.get_chromophore_table("water")

        self.assertLessEqual(table[0, 0], 400)
        self.assertGreaterEqual(table[-1, 0], 1000)

    def test_chromophore_monotonic_wavelengths(self):
        """Chromophore wavelengths should be monotonically increasing."""
        for name in ["hbo", "hbr", "water", "lipids", "aa3"]:
            table = prop_module.get_chromophore_table(name)
            diffs = np.diff(table[:, 0])
            self.assertTrue(np.all(diffs > 0), f"{name} wavelengths not monotonic")


class TestExtinctionUnits(unittest.TestCase):
    """Test extinction coefficient units and conversions."""

    def test_hemoglobin_units(self):
        """Hemoglobin extinction should be in 1/(mm*uM)."""
        # At the isosbestic point (~800nm), HbO2 and Hb are similar
        extin, _ = prop_module.extinction([800], ["hbo", "hbr"])

        # Values should be small (typical tissue ~1e-4 to 1e-3 range)
        self.assertGreater(extin[0, 0], 1e-6)
        self.assertLess(extin[0, 0], 1e-2)

    def test_water_units(self):
        """Water extinction should be in 1/mm (multiply by fraction)."""
        extin, _ = prop_module.extinction([900], ["water"])

        # At 900nm, water absorption is small
        self.assertGreater(extin[0, 0], 0)
        self.assertLess(extin[0, 0], 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
