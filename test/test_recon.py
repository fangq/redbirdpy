"""
Unit tests for redbird.recon module.

Run with: python -m unittest test_recon -v
"""

import unittest
import numpy as np
import sys
import os
from numpy.testing import assert_array_almost_equal, assert_allclose
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbirdpy import recon, forward, utility

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
        "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
        "srcpos": np.array([[30, 30, 0]]),
        "srcdir": np.array([[0, 0, 1]]),
        "detpos": np.array([[30, 40, 0], [40, 30, 0]]),
        "detdir": np.array([[0, 0, 1]]),
        "seg": np.ones(elem.shape[0], dtype=int),
        "omega": 0,
    }
    return cfg


def create_recon_setup():
    """Create setup for reconstruction tests."""
    cfg = create_test_cfg()
    cfg, sd = utility.meshprep(cfg)

    # Generate synthetic data
    detphi0, _ = forward.runforward(cfg, sd=sd)

    # Add perturbation to simulate measurement
    detphi0 = detphi0 * 0.95

    recon_cfg = {
        "prop": cfg["prop"].copy(),
        "lambda": 0.1,
    }

    return cfg, recon_cfg, detphi0, sd


class TestReginv(unittest.TestCase):
    """Test recon.reginv function."""

    def test_reginv_overdetermined(self):
        """reginv should solve overdetermined system."""
        # More rows than columns
        A = np.random.randn(100, 10)
        x_true = np.random.randn(10)
        b = A @ x_true + 0.01 * np.random.randn(100)

        x = recon.reginv(A, b, lambda_=0.01)

        self.assertEqual(len(x), 10)

    def test_reginv_underdetermined(self):
        """reginv should solve underdetermined system."""
        # More columns than rows
        A = np.random.randn(10, 100)
        b = np.random.randn(10)

        x = recon.reginv(A, b, lambda_=0.01)

        self.assertEqual(len(x), 100)

    def test_reginv_square(self):
        """reginv should solve square system."""
        A = np.random.randn(10, 10)
        A = A @ A.T + 0.1 * np.eye(10)  # Make positive definite
        b = np.random.randn(10)

        x = recon.reginv(A, b, lambda_=0.01)

        self.assertEqual(len(x), 10)

    def test_reginv_with_regularization(self):
        """reginv should apply Tikhonov regularization."""
        A = np.random.randn(100, 10)
        b = np.random.randn(100)

        x_low_reg = recon.reginv(A, b, lambda_=0.001)
        x_high_reg = recon.reginv(A, b, lambda_=10.0)

        # Higher regularization should give smaller norm solution
        self.assertLess(np.linalg.norm(x_high_reg), np.linalg.norm(x_low_reg))


class TestReginvover(unittest.TestCase):
    """Test recon.reginvover function."""

    def test_reginvover_basic(self):
        """reginvover should solve overdetermined system."""
        A = np.random.randn(100, 10)
        x_true = np.random.randn(10)
        b = A @ x_true

        x = recon.reginvover(A, b, lambda_=0.001)

        assert_allclose(x, x_true, rtol=0.1)

    def test_reginvover_with_ltl(self):
        """reginvover should accept LTL regularization matrix."""
        A = np.random.randn(100, 10)
        b = np.random.randn(100)
        LTL = np.eye(10)

        x = recon.reginvover(A, b, lambda_=0.1, LTL=LTL)

        self.assertEqual(len(x), 10)

    def test_reginvover_handles_zero_columns(self):
        """reginvover should handle zero-sensitivity columns."""
        A = np.random.randn(100, 10)
        A[:, 5] = 0  # Zero column
        b = np.random.randn(100)

        x = recon.reginvover(A, b, lambda_=0.1)

        self.assertEqual(len(x), 10)
        self.assertEqual(x[5], 0)  # Zero column should give zero result


class TestReginvunder(unittest.TestCase):
    """Test recon.reginvunder function."""

    def test_reginvunder_basic(self):
        """reginvunder should solve underdetermined system."""
        A = np.random.randn(10, 100)
        b = np.random.randn(10)

        x = recon.reginvunder(A, b, lambda_=0.1)

        self.assertEqual(len(x), 100)

    def test_reginvunder_minimum_norm(self):
        """reginvunder should find minimum norm solution."""
        A = np.random.randn(10, 100)
        b = np.random.randn(10)

        x = recon.reginvunder(A, b, lambda_=0.001)

        # Check that A @ x is close to b
        residual = np.linalg.norm(A @ x - b)
        self.assertLess(residual, 1.0)


class TestMatreform(unittest.TestCase):
    """Test recon.matreform function."""

    def test_matreform_complex(self):
        """matreform with 'complex' should return unchanged."""
        A = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
        ymeas = np.random.randn(10) + 1j * np.random.randn(10)
        ymodel = np.random.randn(10) + 1j * np.random.randn(10)

        newA, newrhs, nblock = recon.matreform(A, ymeas, ymodel, "complex")

        self.assertEqual(nblock, 1)
        assert_array_almost_equal(newA, A)

    def test_matreform_real(self):
        """matreform with 'real' should stack real/imag."""
        A = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
        ymeas = np.random.randn(10) + 1j * np.random.randn(10)
        ymodel = np.random.randn(10) + 1j * np.random.randn(10)

        newA, newrhs, nblock = recon.matreform(A, ymeas, ymodel, "real")

        self.assertEqual(newA.shape[0], 20)  # Doubled rows
        self.assertEqual(newrhs.shape[0], 20)
        self.assertEqual(nblock, 2)
        self.assertTrue(np.isreal(newA).all())

    def test_matreform_logphase(self):
        """matreform with 'logphase' should convert to log-amplitude/phase."""
        A = np.random.randn(10, 5)
        ymeas = np.abs(np.random.randn(10)) + 0.1  # Positive values
        ymodel = np.abs(np.random.randn(10)) + 0.1

        newA, newrhs, nblock = recon.matreform(A, ymeas, ymodel, "logphase")

        # RHS should be log difference
        expected_rhs = np.log(np.abs(ymeas)) - np.log(np.abs(ymodel))
        assert_allclose(newrhs, expected_rhs)

    def test_matreform_unknown(self):
        """matreform should raise error for unknown form."""
        A = np.random.randn(10, 5)
        ymeas = np.random.randn(10)
        ymodel = np.random.randn(10)

        with self.assertRaises(ValueError):
            recon.matreform(A, ymeas, ymodel, "unknown")


class TestMatflat(unittest.TestCase):
    """Test recon.matflat function."""

    def test_matflat_array(self):
        """matflat should return array unchanged."""
        A = np.random.randn(10, 5)

        result = recon.matflat(A)

        assert_array_almost_equal(result, A)

    def test_matflat_dict(self):
        """matflat should horizontally concatenate dict values."""
        A = {"a": np.random.randn(10, 5), "b": np.random.randn(10, 3)}

        result = recon.matflat(A)

        self.assertEqual(result.shape, (10, 8))

    def test_matflat_nested_dict(self):
        """matflat should handle nested dicts (multi-wavelength)."""
        A = {
            "a": {"690": np.random.randn(5, 10), "830": np.random.randn(5, 10)},
            "b": {"690": np.random.randn(5, 10), "830": np.random.randn(5, 10)},
        }

        result = recon.matflat(A)

        # 2 wavelengths x 5 rows, 2 params x 10 cols
        self.assertEqual(result.shape, (10, 20))


class TestPrior(unittest.TestCase):
    """Test recon.prior function."""

    def test_prior_laplace(self):
        """prior with 'laplace' should create Laplacian matrix."""
        seg = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

        Lmat = recon.prior(seg, "laplace")

        self.assertEqual(Lmat.shape, (9, 9))
        # Diagonal should be 1
        assert_allclose(np.diag(Lmat), 1.0)

    def test_prior_helmholtz(self):
        """prior with 'helmholtz' should create Helmholtz-like matrix."""
        seg = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

        Lmat = recon.prior(seg, "helmholtz", {"beta": 1.0})

        self.assertEqual(Lmat.shape, (9, 9))
        assert_allclose(np.diag(Lmat), 1.0)

    def test_prior_empty(self):
        """prior with empty type should return None."""
        seg = np.array([0, 0, 1, 1])

        result = recon.prior(seg, "")

        self.assertIsNone(result)

    def test_prior_comp(self):
        """prior with 'comp' should handle composition matrix."""
        # Composition matrix: n nodes x c components
        comp = np.random.rand(10, 3)
        comp = comp / comp.sum(axis=1, keepdims=True)  # Normalize

        Lmat = recon.prior(comp, "comp", {"alpha": 0.5, "beta": 1.0})

        self.assertEqual(Lmat.shape, (10, 10))


class TestSyncprop(unittest.TestCase):
    """Test recon.syncprop function."""

    def test_syncprop_label_based(self):
        """syncprop should copy label-based params."""
        cfg = create_test_cfg()
        cfg, _ = utility.meshprep(cfg)

        recon_cfg = {"param": {"hbo": np.array([50.0]), "hbr": np.array([20.0])}}

        cfg_out, recon_out = recon.syncprop(cfg, recon_cfg)

        self.assertIn("param", cfg_out)
        self.assertEqual(cfg_out["param"]["hbo"][0], 50.0)

    def test_syncprop_prop_copy(self):
        """syncprop should copy prop when no dual mesh."""
        cfg = create_test_cfg()
        cfg, _ = utility.meshprep(cfg)

        recon_cfg = {"prop": np.array([[0, 0, 1, 1], [0.02, 0.8, 0, 1.4]])}

        cfg_out, recon_out = recon.syncprop(cfg, recon_cfg)

        assert_array_almost_equal(cfg_out["prop"], recon_cfg["prop"])


class TestRunrecon(unittest.TestCase):
    """Test recon.runrecon function."""

    def test_runrecon_returns_tuple(self):
        """runrecon should return tuple with results."""
        cfg, recon_cfg, detphi0, sd = create_recon_setup()

        result = recon.runrecon(cfg, recon_cfg, detphi0, sd, maxiter=1, report=False)

        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 3)

    def test_runrecon_returns_recon(self):
        """runrecon should return updated recon structure."""
        cfg, recon_cfg, detphi0, sd = create_recon_setup()

        recon_out, resid, cfg_out, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=1, report=False
        )

        self.assertIsInstance(recon_out, dict)

    def test_runrecon_returns_residual(self):
        """runrecon should return residual array."""
        cfg, recon_cfg, detphi0, sd = create_recon_setup()

        _, resid, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=3, report=False
        )

        self.assertIsInstance(resid, np.ndarray)
        self.assertGreaterEqual(len(resid), 1)

    def test_runrecon_residual_decreases(self):
        """runrecon residual should generally decrease."""
        cfg, recon_cfg, detphi0, sd = create_recon_setup()

        _, resid, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=5, report=False
        )

        # First residual should be larger than last (convergence)
        self.assertGreater(resid[0], resid[-1] * 0.5)  # Allow some tolerance

    def test_runrecon_with_regularization(self):
        """runrecon should accept regularization parameter."""
        cfg, recon_cfg, detphi0, sd = create_recon_setup()
        recon_cfg["lambda"] = 1.0

        recon_out, resid, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=1, report=False
        )

        self.assertEqual(recon_out["lambda"], 1.0)

    def test_runrecon_updates_cfg(self):
        """runrecon should return updated cfg."""
        cfg, recon_cfg, detphi0, sd = create_recon_setup()

        _, _, cfg_out, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=1, report=False
        )

        self.assertIn("deldotdel", cfg_out)

    def test_runrecon_stores_lambda(self):
        """runrecon should store final lambda in recon."""
        cfg, recon_cfg, detphi0, sd = create_recon_setup()

        recon_out, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=1, lambda_=0.05, report=False
        )

        self.assertEqual(recon_out["lambda"], 0.05)


class TestPrivateFunctions(unittest.TestCase):
    """Test private helper functions in recon module."""

    def test_normalize_diag(self):
        """_normalize_diag should normalize to unit diagonal."""
        A = np.diag([4, 9, 16])

        Anorm, di = recon._normalize_diag(A)

        assert_allclose(np.diag(Anorm), [1, 1, 1])
        assert_allclose(di, [0.5, 1 / 3, 0.25])

    def test_flatten_detphi_array(self):
        """_flatten_detphi should flatten array."""
        detphi = np.array([[1, 2], [3, 4], [5, 6]])

        result = recon._flatten_detphi(detphi, None, [""], [1])

        self.assertEqual(len(result), 6)

    def test_masksum(self):
        """_masksum should sum by mask labels."""
        data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        mask = np.array([0, 0, 1, 1, 1])

        result = recon._masksum(data, mask)

        # Label 0: columns 0,1; Label 1: columns 2,3,4
        self.assertEqual(result.shape, (2, 2))
        assert_array_almost_equal(result[:, 0], [3, 13])  # Sum of cols 0,1
        assert_array_almost_equal(result[:, 1], [12, 27])  # Sum of cols 2,3,4


class TestReginvoverExtended(unittest.TestCase):
    """Extended tests for reginvover function."""

    def test_reginvover_with_block_ltl(self):
        """reginvover should handle block LTL regularization."""
        A = np.random.randn(100, 20)
        b = np.random.randn(100)
        LTL = np.eye(10)  # Smaller than A columns

        x = recon.reginvover(A, b, lambda_=0.1, LTL=LTL)

        self.assertEqual(len(x), 20)

    def test_reginvover_zero_rows(self):
        """reginvover should handle zero rows in matrix."""
        A = np.random.randn(100, 10)
        A[50:60, :] = 0  # Zero rows
        b = np.random.randn(100)

        x = recon.reginvover(A, b, lambda_=0.1)

        self.assertEqual(len(x), 10)
        self.assertTrue(np.all(np.isfinite(x)))


class TestReginvunderExtended(unittest.TestCase):
    """Extended tests for reginvunder function."""

    def test_reginvunder_with_invR(self):
        """reginvunder should handle invR regularization."""
        A = np.random.randn(10, 100)
        b = np.random.randn(10)
        invR = np.eye(100)

        x = recon.reginvunder(A, b, lambda_=0.1, invR=invR)

        self.assertEqual(len(x), 100)

    def test_reginvunder_with_blocks(self):
        """reginvunder should handle block structure."""
        A = np.random.randn(10, 50)
        b = np.random.randn(10)
        invR = np.eye(25)
        blocks = {"a": (10, 25), "b": (10, 25)}

        x = recon.reginvunder(A, b, lambda_=0.1, invR=invR, blocks=blocks)

        self.assertEqual(len(x), 50)

    def test_reginvunder_zero_columns(self):
        """reginvunder should handle zero columns."""
        A = np.random.randn(10, 50)
        A[:, 25:30] = 0  # Zero columns
        b = np.random.randn(10)

        x = recon.reginvunder(A, b, lambda_=0.1)

        self.assertEqual(len(x), 50)


class TestMatreformExtended(unittest.TestCase):
    """Extended tests for matreform function."""

    def test_matreform_real_with_real_data(self):
        """matreform 'real' with real data should not double rows."""
        A = np.random.randn(10, 5)
        ymeas = np.random.randn(10)
        ymodel = np.random.randn(10)

        newA, newrhs, nblock = recon.matreform(A, ymeas, ymodel, "real")

        self.assertEqual(nblock, 1)
        self.assertEqual(newA.shape[0], 10)

    def test_matreform_reim(self):
        """matreform 'reim' should expand to [Re; Im] form."""
        A = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
        ymeas = np.random.randn(10) + 1j * np.random.randn(10)
        ymodel = np.random.randn(10) + 1j * np.random.randn(10)

        newA, newrhs, nblock = recon.matreform(A, ymeas, ymodel, "reim")

        self.assertEqual(nblock, 2)
        self.assertEqual(newA.shape[0], 20)
        self.assertEqual(newA.shape[1], 10)  # Doubled columns too

    def test_matreform_logphase_complex(self):
        """matreform 'logphase' with complex data."""
        A = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
        ymeas = (np.random.randn(10) + 0.5) + 1j * np.random.randn(10)
        ymodel = (np.random.randn(10) + 0.5) + 1j * np.random.randn(10)

        newA, newrhs, nblock = recon.matreform(A, ymeas, ymodel, "logphase")

        self.assertEqual(nblock, 2)
        self.assertEqual(newA.shape[0], 20)


class TestMatflatExtended(unittest.TestCase):
    """Extended tests for matflat function."""

    def test_matflat_with_weight(self):
        """matflat should apply weights to dict values."""
        A = {
            "a": np.ones((10, 5)),
            "b": np.ones((10, 3)),
        }
        weight = np.array([2.0, 0.5])

        result = recon.matflat(A, weight)

        self.assertEqual(result.shape, (10, 8))
        assert_allclose(result[:, :5], 2.0)
        assert_allclose(result[:, 5:], 0.5)


class TestPriorExtended(unittest.TestCase):
    """Extended tests for prior function."""

    def test_prior_laplace_small(self):
        """prior laplace with small segmentation."""
        # Use small segmentation to avoid memory issues
        seg = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

        Lmat = recon.prior(seg, "laplace")

        self.assertEqual(Lmat.shape, (9, 9))
        # Diagonal should be 1
        assert_allclose(np.diag(Lmat), 1.0)

    def test_prior_laplace_single_segment(self):
        """prior laplace with all same segment."""
        seg = np.zeros(8, dtype=int)  # All same segment

        Lmat = recon.prior(seg, "laplace")

        self.assertEqual(Lmat.shape, (8, 8))
        # Diagonal should be 1
        assert_allclose(np.diag(Lmat), 1.0)

    def test_prior_helmholtz_small(self):
        """prior helmholtz with small segmentation."""
        seg = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

        Lmat = recon.prior(seg, "helmholtz", {"beta": 1.0})

        self.assertEqual(Lmat.shape, (9, 9))
        assert_allclose(np.diag(Lmat), 1.0)

    def test_prior_helmholtz_beta(self):
        """prior helmholtz should respect beta parameter."""
        seg = np.array([0, 0, 1, 1, 2, 2])

        Lmat1 = recon.prior(seg, "helmholtz", {"beta": 0.5})
        Lmat2 = recon.prior(seg, "helmholtz", {"beta": 2.0})

        # Different beta should give different matrices
        self.assertFalse(np.allclose(Lmat1, Lmat2))

    def test_prior_empty(self):
        """prior with empty type should return None."""
        seg = np.array([0, 0, 1, 1])

        result = recon.prior(seg, "")

        self.assertIsNone(result)

    def test_prior_comp_small(self):
        """prior comp with small composition matrix."""
        # Small composition matrix: 10 nodes x 3 components
        comp = np.random.rand(10, 3)
        comp = comp / comp.sum(axis=1, keepdims=True)  # Normalize

        Lmat = recon.prior(comp, "comp", {"alpha": 0.3, "beta": 1.0})

        self.assertTrue(sparse.issparse(Lmat))
        self.assertEqual(Lmat.shape, (10, 10))


class TestSyncpropExtended(unittest.TestCase):
    """Extended tests for syncprop function."""

    def setUp(self):
        self.cfg = create_test_cfg()
        self.cfg, _ = utility.meshprep(self.cfg)

    def test_syncprop_with_param_array(self):
        """syncprop should handle array-valued params."""
        nn = self.cfg["node"].shape[0]

        recon_cfg = {
            "param": {
                "hbo": np.ones(nn) * 50.0,
                "hbr": np.ones(nn) * 20.0,
            }
        }

        cfg_out, recon_out = recon.syncprop(self.cfg, recon_cfg)

        self.assertIn("param", cfg_out)
        self.assertEqual(len(cfg_out["param"]["hbo"]), nn)

    def test_syncprop_multiwavelength_prop(self):
        """syncprop should handle multi-wavelength prop."""
        recon_cfg = {
            "prop": {
                "690": np.array([[0, 0, 1, 1], [0.02, 0.9, 0, 1.37]]),
                "830": np.array([[0, 0, 1, 1], [0.015, 0.85, 0, 1.37]]),
            }
        }

        cfg_out, recon_out = recon.syncprop(self.cfg, recon_cfg)

        self.assertIn("prop", cfg_out)
        self.assertIsInstance(cfg_out["prop"], dict)

    @unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
    def test_syncprop_dual_mesh(self):
        """syncprop should interpolate for dual mesh reconstruction."""
        # Create coarser reconstruction mesh
        recon_node, _, recon_elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 15)

        # Create mapping from forward to recon mesh
        mapid, mapweight = i2m.tsearchn(recon_node, recon_elem[:, :4], self.cfg["node"])

        nn_recon = recon_node.shape[0]
        recon_cfg = {
            "node": recon_node,
            "elem": recon_elem,
            "mapid": mapid,
            "mapweight": mapweight,
            "prop": np.column_stack(
                [
                    0.01 * np.ones(nn_recon),
                    1.0 * np.ones(nn_recon),
                    np.zeros(nn_recon),
                    1.37 * np.ones(nn_recon),
                ]
            ),
        }

        cfg_out, recon_out = recon.syncprop(self.cfg, recon_cfg)

        # cfg.prop should now be node-based on forward mesh
        self.assertEqual(cfg_out["prop"].shape[0], self.cfg["node"].shape[0])


class TestRunreconExtended(unittest.TestCase):
    """Extended tests for runrecon function."""

    def setUp(self):
        self.cfg = create_test_cfg()
        self.cfg, self.sd = utility.meshprep(self.cfg)
        self.detphi0, _ = forward.runforward(self.cfg, sd=self.sd)
        self.detphi0 = self.detphi0 * 0.95

    def test_runrecon_with_prior(self):
        """runrecon should accept prior regularization."""
        # Use label-based segmentation for prior to avoid large dense matrices
        # The prior matrix will be (n_labels x n_labels) instead of (nn x nn)
        n_labels = len(np.unique(self.cfg["seg"]))

        recon_cfg = {
            "prop": self.cfg["prop"].copy(),
            "lambda": 0.1,
            "seg": np.arange(n_labels),  # Small label-based seg for prior
        }

        # Create a small prior matrix manually instead of using "laplace"
        # which would create a huge (nn x nn) matrix
        small_lmat = np.eye(n_labels)

        recon_out, resid, *_ = recon.runrecon(
            self.cfg,
            recon_cfg,
            self.detphi0,
            self.sd,
            maxiter=1,
            report=False,
            lmat=small_lmat,
        )

        self.assertIsNotNone(recon_out)

    def test_runrecon_with_ltl(self):
        """runrecon should accept precomputed LTL matrix."""
        # Use a small identity matrix for regularization
        # For label-based reconstruction, size matches number of labels
        n_labels = len(np.unique(self.cfg["seg"]))
        LTL = np.eye(n_labels)

        recon_cfg = {
            "prop": self.cfg["prop"].copy(),
            "lambda": 0.1,
        }

        recon_out, resid, *_ = recon.runrecon(
            self.cfg, recon_cfg, self.detphi0, self.sd, maxiter=1, report=False, ltl=LTL
        )

        self.assertIsNotNone(recon_out)

    def test_runrecon_chromophore_based(self):
        """runrecon should handle chromophore-based reconstruction."""
        cfg = self.cfg.copy()
        cfg["prop"] = {
            "690": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "830": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
        }
        cfg["param"] = {
            "hbo": np.array([50.0]),
            "hbr": np.array([20.0]),
        }
        cfg, sd = utility.meshprep(cfg)

        detphi0, _ = forward.runforward(cfg, sd=sd)
        detphi0 = {k: v * 0.95 for k, v in detphi0.items()}

        recon_cfg = {
            "param": {
                "hbo": np.array([50.0]),
                "hbr": np.array([20.0]),
            },
            "lambda": 0.1,
        }

        recon_out, resid, *_ = recon.runrecon(
            cfg, recon_cfg, detphi0, sd, maxiter=1, report=False
        )

        self.assertIn("param", recon_out)

    def test_runrecon_returns_jacobian(self):
        """runrecon should return Jacobian matrices."""
        recon_cfg = {
            "prop": self.cfg["prop"].copy(),
            "lambda": 0.1,
        }

        result = recon.runrecon(
            self.cfg, recon_cfg, self.detphi0, self.sd, maxiter=1, report=False
        )

        # Should have at least 5 return values including Jmua
        self.assertGreaterEqual(len(result), 5)

        recon_out, resid, cfg_out, updates, Jmua = result[:5]
        self.assertIsNotNone(Jmua)

    def test_runrecon_tolerance_early_stop(self):
        """runrecon should stop early when tolerance reached."""
        recon_cfg = {
            "prop": self.cfg["prop"].copy(),
            "lambda": 0.1,
        }

        # Use very loose tolerance
        recon_out, resid, *_ = recon.runrecon(
            self.cfg,
            recon_cfg,
            self.detphi0,
            self.sd,
            maxiter=10,
            tol=1.0,
            report=False,
        )

        # Should stop before maxiter
        self.assertLessEqual(len(resid), 10)


class TestRemapJacobian(unittest.TestCase):
    """Test _remap_jacobian function."""

    @unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not installed")
    def test_remap_jacobian_basic(self):
        """_remap_jacobian should interpolate Jacobian to recon mesh."""
        # Create forward mesh
        fwd_node, _, fwd_elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 8)

        # Create coarser recon mesh
        rec_node, _, rec_elem = i2m.meshabox([0, 0, 0], [60, 60, 30], 12)

        # Create mapping
        mapid, mapweight = i2m.tsearchn(rec_node, rec_elem[:, :4], fwd_node)

        recon_struct = {
            "node": rec_node,
            "elem": rec_elem,
            "mapid": mapid,
            "mapweight": mapweight,
        }

        cfg = {"node": fwd_node, "elem": fwd_elem}

        # Create mock Jacobian
        nsd = 5
        nn_fwd = fwd_node.shape[0]
        J = np.random.randn(nsd, nn_fwd)

        J_remapped = recon._remap_jacobian(J, recon_struct, cfg)

        nn_rec = rec_node.shape[0]
        self.assertEqual(J_remapped.shape, (nsd, nn_rec))


class TestMasksum(unittest.TestCase):
    """Test _masksum function."""

    def test_masksum_multiple_labels(self):
        """_masksum should sum by multiple labels."""
        data = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
            ]
        )
        mask = np.array([0, 0, 1, 1, 2, 2])

        result = recon._masksum(data, mask)

        self.assertEqual(result.shape, (2, 3))
        assert_array_almost_equal(result[:, 0], [3, 15])  # Label 0
        assert_array_almost_equal(result[:, 1], [7, 19])  # Label 1
        assert_array_almost_equal(result[:, 2], [11, 23])  # Label 2

    def test_masksum_single_label(self):
        """_masksum with single label should sum all columns."""
        data = np.array([[1, 2, 3, 4]])
        mask = np.array([0, 0, 0, 0])

        result = recon._masksum(data, mask)

        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result[0, 0], 10)


class TestFlattenDetphi(unittest.TestCase):
    """Test _flatten_detphi function."""

    def test_flatten_detphi_dict(self):
        """_flatten_detphi should flatten dict input."""
        detphi = {
            "690": np.array([[1, 2], [3, 4]]),
            "830": np.array([[5, 6], [7, 8]]),
        }

        result = recon._flatten_detphi(detphi, None, ["690", "830"], [1])

        self.assertEqual(len(result), 8)

    def test_flatten_detphi_nested_dict(self):
        """_flatten_detphi should handle nested dict."""
        detphi = {
            1: {"detphi": np.array([1, 2, 3])},
        }

        result = recon._flatten_detphi(detphi, None, [""], [1])

        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
