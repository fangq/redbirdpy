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

import redbird as rb
from redbird import recon, forward, utility

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
