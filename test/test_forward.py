"""
Unit tests for redbird.forward module.

Run with: python -m unittest test_forward -v
"""

import unittest
import sys
import os
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbird as rb
from redbird import forward, utility

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False


def create_test_cfg():
    """Create a test configuration."""
    if HAS_ISO2MESH:
        node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 10)
    else:
        # Simple manual mesh
        node = np.array(
            [
                [0, 0, 0],
                [60, 0, 0],
                [60, 60, 0],
                [0, 60, 0],
                [0, 0, 30],
                [60, 0, 30],
                [60, 60, 30],
                [0, 60, 30],
                [30, 30, 15],
            ],
            dtype=float,
        )
        elem = np.array(
            [
                [1, 2, 3, 9],
                [1, 3, 4, 9],
                [1, 2, 5, 9],
                [2, 5, 6, 9],
                [2, 3, 6, 9],
                [3, 6, 7, 9],
                [3, 4, 7, 9],
                [4, 7, 8, 9],
                [1, 4, 5, 9],
                [4, 5, 8, 9],
                [5, 6, 7, 9],
                [5, 7, 8, 9],
            ],
            dtype=int,
        )
        face = np.array([[1, 2, 3], [1, 3, 4]], dtype=int)

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


class TestDeldotdel(unittest.TestCase):
    """Test forward.deldotdel function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_deldotdel_returns_tuple(self):
        """deldotdel should return (deldotdel, delphi) tuple."""
        result = forward.deldotdel(self.cfg)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_deldotdel_shape(self):
        """deldotdel should have shape (Ne, 10)."""
        ddd, delphi = forward.deldotdel(self.cfg)

        ne = self.cfg["elem"].shape[0]
        self.assertEqual(ddd.shape, (ne, 10))

    def test_delphi_shape(self):
        """delphi should have shape (3, 4, Ne)."""
        ddd, delphi = forward.deldotdel(self.cfg)

        ne = self.cfg["elem"].shape[0]
        self.assertEqual(delphi.shape, (3, 4, ne))

    def test_deldotdel_finite(self):
        """deldotdel values should be finite."""
        ddd, _ = forward.deldotdel(self.cfg)

        self.assertTrue(np.all(np.isfinite(ddd)))

    def test_deldotdel_handles_1based_elem(self):
        """deldotdel should handle 1-based elem correctly."""
        # Verify elem is 1-based
        self.assertGreaterEqual(self.cfg["elem"].min(), 1)

        # Should not raise IndexError
        ddd, _ = forward.deldotdel(self.cfg)

        self.assertIsNotNone(ddd)


class TestFemlhs(unittest.TestCase):
    """Test forward.femlhs function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_femlhs_returns_sparse(self):
        """femlhs should return sparse matrix."""
        Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"])

        self.assertTrue(sparse.issparse(Amat))

    def test_femlhs_shape(self):
        """femlhs should return (Nn, Nn) matrix."""
        Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"])

        nn = self.cfg["node"].shape[0]
        self.assertEqual(Amat.shape, (nn, nn))

    def test_femlhs_symmetric(self):
        """femlhs should return symmetric matrix for CW."""
        Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"], mode=2)

        diff = Amat - Amat.T
        self.assertLess(sparse.linalg.norm(diff), 1e-10)

    def test_femlhs_positive_diagonal(self):
        """femlhs diagonal should be positive."""
        Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"])

        diag = Amat.diagonal()
        self.assertTrue(np.all(diag > 0))

    def test_femlhs_handles_1based(self):
        """femlhs should handle 1-based elem/face correctly."""
        self.assertGreaterEqual(self.cfg["elem"].min(), 1)
        self.assertGreaterEqual(self.cfg["face"].min(), 1)

        # Should not raise IndexError
        Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"])

        self.assertIsNotNone(Amat)

    def test_femlhs_frequency_domain(self):
        """femlhs should return complex matrix for FD."""
        cfg = self.cfg.copy()
        cfg["omega"] = 2 * np.pi * 100e6  # 100 MHz

        Amat = forward.femlhs(cfg, cfg["deldotdel"], mode=1)

        self.assertTrue(np.iscomplexobj(Amat.data))


class TestFemrhs(unittest.TestCase):
    """Test forward.femrhs function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_cfg()
        self.cfg, self.sd = utility.meshprep(self.cfg)

    def test_femrhs_returns_tuple(self):
        """femrhs should return (rhs, loc, bary, optode) tuple."""
        result = forward.femrhs(self.cfg, self.sd)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_femrhs_rhs_shape(self):
        """femrhs RHS should have shape (Nn, Nsrc+Ndet)."""
        rhs, _, _, _ = forward.femrhs(self.cfg, self.sd)

        nn = self.cfg["node"].shape[0]
        srcnum = self.cfg["srcpos"].shape[0]
        detnum = self.cfg["detpos"].shape[0]

        # Include wide-field sources if present
        wfsrcnum = 0
        if (
            "widesrc" in self.cfg
            and self.cfg["widesrc"] is not None
            and self.cfg["widesrc"].size > 0
        ):
            wfsrcnum = self.cfg["widesrc"].shape[1]
        wfdetnum = 0
        if (
            "widedet" in self.cfg
            and self.cfg["widedet"] is not None
            and self.cfg["widedet"].size > 0
        ):
            wfdetnum = self.cfg["widedet"].shape[1]

        self.assertEqual(rhs.shape[0], nn)
        self.assertEqual(rhs.shape[1], srcnum + wfsrcnum + detnum + wfdetnum)

    def test_femrhs_loc_1based(self):
        """femrhs loc should be 1-based element indices from tsearchn."""
        _, loc, _, _ = forward.femrhs(self.cfg, self.sd)

        ne = self.cfg["elem"].shape[0]
        valid_loc = loc[~np.isnan(loc)]

        if len(valid_loc) > 0:
            # tsearchn returns 1-based indices
            self.assertGreaterEqual(valid_loc.min(), 1)
            self.assertLessEqual(valid_loc.max(), ne)

    def test_femrhs_bary_sum_to_one(self):
        """femrhs barycentric coordinates should sum to 1."""
        _, loc, bary, _ = forward.femrhs(self.cfg, self.sd)

        for i in range(len(loc)):
            if not np.isnan(loc[i]):
                self.assertAlmostEqual(np.sum(bary[i, :]), 1.0, places=5)

    def test_femrhs_bary_nonnegative(self):
        """femrhs barycentric coordinates should be non-negative."""
        _, loc, bary, _ = forward.femrhs(self.cfg, self.sd)

        for i in range(len(loc)):
            if not np.isnan(loc[i]):
                self.assertTrue(np.all(bary[i, :] >= -1e-10))


class TestFemsolve(unittest.TestCase):
    """Test forward.femsolve function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_cfg()
        self.cfg, self.sd = utility.meshprep(self.cfg)
        self.Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"])
        self.rhs, _, _, _ = forward.femrhs(self.cfg, self.sd)

    def test_femsolve_direct(self):
        """femsolve with direct method should solve system."""
        x, flag = forward.femsolve(self.Amat, self.rhs, method="direct")

        self.assertEqual(flag, 0)
        self.assertEqual(x.shape[0], self.Amat.shape[0])

    def test_femsolve_solution_finite(self):
        """femsolve solution should be finite."""
        x, _ = forward.femsolve(self.Amat, self.rhs, method="direct")

        # Check that solution is finite (not NaN or Inf)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_femsolve_cg(self):
        """femsolve with CG method should work for symmetric matrix."""
        # CW mode gives symmetric matrix
        Amat_cw = forward.femlhs(self.cfg, self.cfg["deldotdel"], mode=2)

        x, flag = forward.femsolve(Amat_cw, self.rhs, method="cg", tol=1e-6)

        # CG should converge for SPD system
        self.assertIn(flag, [0, 1, 2, 3])  # Various convergence states


class TestFemgetdet(unittest.TestCase):
    """Test forward.femgetdet function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_cfg()
        self.cfg, self.sd = utility.meshprep(self.cfg)

        Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"])
        self.rhs, self.loc, self.bary, _ = forward.femrhs(self.cfg, self.sd)
        self.phi, _ = forward.femsolve(Amat, self.rhs)

    def test_femgetdet_shape(self):
        """femgetdet should return (Ndet, Nsrc) array."""
        detval = forward.femgetdet(self.phi, self.cfg, self.rhs, self.loc, self.bary)

        srcnum = self.cfg["srcpos"].shape[0]
        detnum = self.cfg["detpos"].shape[0]

        # Include wide-field sources if present
        wfsrcnum = 0
        if (
            "widesrc" in self.cfg
            and self.cfg["widesrc"] is not None
            and self.cfg["widesrc"].size > 0
        ):
            wfsrcnum = self.cfg["widesrc"].shape[1]

        self.assertEqual(detval.shape, (detnum, srcnum + wfsrcnum))

    def test_femgetdet_finite(self):
        """femgetdet values should be finite."""
        detval = forward.femgetdet(self.phi, self.cfg, self.rhs, self.loc, self.bary)

        self.assertTrue(np.all(np.isfinite(detval)))

    def test_femgetdet_handles_1based_loc(self):
        """femgetdet should handle 1-based loc correctly."""
        # loc from tsearchn is 1-based
        valid_loc = self.loc[~np.isnan(self.loc)]
        if len(valid_loc) > 0:
            self.assertGreaterEqual(valid_loc.min(), 1)

        # Should not raise IndexError
        detval = forward.femgetdet(self.phi, self.cfg, self.rhs, self.loc, self.bary)

        self.assertIsNotNone(detval)


class TestJac(unittest.TestCase):
    """Test forward.jac function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_cfg()
        self.cfg, self.sd = utility.meshprep(self.cfg)

        Amat = forward.femlhs(self.cfg, self.cfg["deldotdel"])
        rhs, loc, bary, _ = forward.femrhs(self.cfg, self.sd)
        self.phi, _ = forward.femsolve(Amat, rhs)

    def test_jac_returns_tuple(self):
        """jac should return (Jmua_n, Jmua_e) tuple."""
        result = forward.jac(
            self.sd, self.phi, self.cfg["deldotdel"], self.cfg["elem"], self.cfg["evol"]
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_jac_node_shape(self):
        """jac node-based should have shape (Nsd, Nn)."""
        Jmua_n, _ = forward.jac(
            self.sd, self.phi, self.cfg["deldotdel"], self.cfg["elem"], self.cfg["evol"]
        )

        nn = self.cfg["node"].shape[0]
        nsd = np.sum(self.sd[:, 2] == 1) if self.sd.shape[1] >= 3 else self.sd.shape[0]

        self.assertEqual(Jmua_n.shape, (nsd, nn))

    def test_jac_elem_shape(self):
        """jac element-based should have shape (Nsd, Ne)."""
        _, Jmua_e = forward.jac(
            self.sd, self.phi, self.cfg["deldotdel"], self.cfg["elem"], self.cfg["evol"]
        )

        ne = self.cfg["elem"].shape[0]
        nsd = np.sum(self.sd[:, 2] == 1) if self.sd.shape[1] >= 3 else self.sd.shape[0]

        self.assertEqual(Jmua_e.shape, (nsd, ne))

    def test_jac_finite(self):
        """jac values should be finite."""
        Jmua_n, Jmua_e = forward.jac(
            self.sd, self.phi, self.cfg["deldotdel"], self.cfg["elem"], self.cfg["evol"]
        )

        self.assertTrue(np.all(np.isfinite(Jmua_n)))
        self.assertTrue(np.all(np.isfinite(Jmua_e)))

    def test_jac_handles_1based_elem(self):
        """jac should handle 1-based elem correctly."""
        self.assertGreaterEqual(self.cfg["elem"].min(), 1)

        # Should not raise IndexError
        Jmua_n, Jmua_e = forward.jac(
            self.sd, self.phi, self.cfg["deldotdel"], self.cfg["elem"], self.cfg["evol"]
        )

        self.assertIsNotNone(Jmua_n)


class TestRunforward(unittest.TestCase):
    """Test forward.runforward function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_cfg()
        self.cfg, self.sd = utility.meshprep(self.cfg)

    def test_runforward_returns_tuple(self):
        """runforward should return (detval, phi) tuple."""
        result = forward.runforward(self.cfg)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_runforward_detval_shape(self):
        """runforward detval should have shape (Ndet, Nsrc)."""
        detval, _ = forward.runforward(self.cfg)

        srcnum = self.cfg["srcpos"].shape[0]
        detnum = self.cfg["detpos"].shape[0]

        self.assertEqual(detval.shape, (detnum, srcnum))

    def test_runforward_phi_shape(self):
        """runforward phi should have shape (Nn, Nsrc+Ndet)."""
        _, phi = forward.runforward(self.cfg)

        nn = self.cfg["node"].shape[0]

        self.assertEqual(phi.shape[0], nn)

    def test_runforward_detval_finite(self):
        """runforward detval should be finite."""
        detval, _ = forward.runforward(self.cfg)

        self.assertTrue(np.all(np.isfinite(detval)))

    def test_runforward_phi_finite(self):
        """runforward phi should be finite."""
        _, phi = forward.runforward(self.cfg)

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_runforward_multiwavelength(self):
        """runforward should handle multi-wavelength."""
        cfg = self.cfg.copy()
        cfg["prop"] = {"690": self.cfg["prop"], "830": self.cfg["prop"]}
        cfg, sd = utility.meshprep(cfg)

        detval, phi = forward.runforward(cfg, sd=sd)

        # Should return dicts for multi-wavelength
        self.assertIsInstance(detval, dict)
        self.assertIn("690", detval)
        self.assertIn("830", detval)


class TestJacchrome(unittest.TestCase):
    """Test forward.jacchrome function."""

    def setUp(self):
        """Set up multi-wavelength config."""
        self.cfg = create_test_cfg()
        self.cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
        }
        self.cfg, self.sd = utility.meshprep(self.cfg)

    def test_jacchrome_returns_dict(self):
        """jacchrome should return dict keyed by chromophore."""
        # Create mock Jmua dict
        nn = self.cfg["node"].shape[0]
        Jmua = {"690": np.random.randn(2, nn), "830": np.random.randn(2, nn)}

        Jchrome = forward.jacchrome(Jmua, ["hbo", "hbr"])

        self.assertIsInstance(Jchrome, dict)
        self.assertIn("hbo", Jchrome)
        self.assertIn("hbr", Jchrome)

    def test_jacchrome_shape(self):
        """jacchrome should stack wavelengths vertically."""
        nn = self.cfg["node"].shape[0]
        nsd = 2
        Jmua = {"690": np.random.randn(nsd, nn), "830": np.random.randn(nsd, nn)}

        Jchrome = forward.jacchrome(Jmua, ["hbo"])

        # Should have 2*nsd rows (stacked wavelengths)
        self.assertEqual(Jchrome["hbo"].shape[0], 2 * nsd)
        self.assertEqual(Jchrome["hbo"].shape[1], nn)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldFemrhs(unittest.TestCase):
    """Test femrhs with wide-field sources."""

    def setUp(self):
        """Set up wide-field test configuration."""
        node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 8)

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
            "srctype": "planar",
            "srcparam1": np.array([20, 0, 0, 0]),
            "srcparam2": np.array([0, 20, 0, 0]),
        }
        self.cfg, self.sd = utility.meshprep(self.cfg)

    def test_femrhs_widesrc_columns(self):
        """femrhs should include wide-field source columns."""
        rhs, loc, bary, optode = forward.femrhs(self.cfg, self.sd)

        wfsrcnum = self.cfg["widesrc"].shape[1] if self.cfg["widesrc"].size > 0 else 0
        srcnum = self.cfg["srcpos"].shape[0]
        detnum = self.cfg["detpos"].shape[0]

        expected_cols = srcnum + wfsrcnum + detnum
        self.assertEqual(rhs.shape[1], expected_cols)

    def test_femrhs_widesrc_nonzero(self):
        """femrhs wide-field columns should have nonzero entries."""
        rhs, _, _, _ = forward.femrhs(self.cfg, self.sd)

        srcnum = self.cfg["srcpos"].shape[0]
        wfsrcnum = self.cfg["widesrc"].shape[1]

        # Wide-field columns should have nonzero entries
        wf_cols = rhs[:, srcnum : srcnum + wfsrcnum]
        self.assertGreater(wf_cols.nnz, 0)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
class TestWidefieldRunforward(unittest.TestCase):
    """Test runforward with wide-field sources."""

    def setUp(self):
        """Set up wide-field test configuration."""
        node, face, elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 8)

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
            "srctype": "planar",
            "srcparam1": np.array([20, 0, 0, 0]),
            "srcparam2": np.array([0, 20, 0, 0]),
        }
        self.cfg, self.sd = utility.meshprep(self.cfg)

    def test_runforward_widesrc_detval_finite(self):
        """runforward with wide-field should produce finite detval."""
        detval, phi = forward.runforward(self.cfg, sd=self.sd)

        self.assertTrue(np.all(np.isfinite(detval)))

    def test_runforward_widesrc_phi_finite(self):
        """runforward with wide-field should produce finite phi."""
        detval, phi = forward.runforward(self.cfg, sd=self.sd)

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_runforward_widesrc_detval_positive(self):
        """runforward with wide-field should produce positive detval."""
        detval, phi = forward.runforward(self.cfg, sd=self.sd)

        self.assertTrue(np.all(detval > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
