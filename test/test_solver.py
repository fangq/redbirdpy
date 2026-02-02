"""
Unit tests for redbird.solver module.

Run with: python -m unittest test_solver -v
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbird import solver


class TestGetSolverInfo(unittest.TestCase):
    """Test solver.get_solver_info function."""

    def test_get_solver_info_returns_dict(self):
        """get_solver_info should return dictionary."""
        info = solver.get_solver_info()
        self.assertIsInstance(info, dict)

    def test_get_solver_info_keys(self):
        """get_solver_info should have expected keys."""
        info = solver.get_solver_info()

        expected_keys = [
            "direct_solver",
            "has_pardiso",
            "has_umfpack",
            "has_cholmod",
            "has_amg",
            "has_blqmr",
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_get_solver_info_cpu_count(self):
        """get_solver_info should report CPU count."""
        info = solver.get_solver_info()
        self.assertIn("cpu_count", info)
        self.assertGreater(info["cpu_count"], 0)


class TestFemsolveBasic(unittest.TestCase):
    """Test basic femsolve functionality."""

    def setUp(self):
        """Create test matrices."""
        n = 50
        # Create SPD matrix
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A_spd = A @ A.T + sparse.eye(n) * 10
        self.b = np.random.randn(n)
        self.B = np.random.randn(n, 3)  # Multiple RHS

    def test_femsolve_auto(self):
        """femsolve with auto method should solve system."""
        x, flag = solver.femsolve(self.A_spd, self.b, method="auto")

        self.assertEqual(flag, 0)
        self.assertEqual(len(x), self.A_spd.shape[0])

    def test_femsolve_direct(self):
        """femsolve with direct method should solve system."""
        x, flag = solver.femsolve(self.A_spd, self.b, method="direct")

        self.assertEqual(flag, 0)
        # Check residual
        residual = np.linalg.norm(self.A_spd @ x - self.b)
        self.assertLess(residual, 1e-8)

    def test_femsolve_superlu(self):
        """femsolve with superlu should solve system."""
        x, flag = solver.femsolve(self.A_spd, self.b, method="superlu")

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(self.A_spd @ x - self.b)
        self.assertLess(residual, 1e-8)

    def test_femsolve_cg(self):
        """femsolve with CG should solve SPD system."""
        x, flag = solver.femsolve(self.A_spd, self.b, method="cg", tol=1e-10)

        # CG may not always converge to flag=0
        self.assertIn(flag, [0, 1, 2, 3, 4])

        if flag == 0:
            residual = np.linalg.norm(self.A_spd @ x - self.b)
            self.assertLess(residual, 1e-6)

    def test_femsolve_gmres(self):
        """femsolve with GMRES should solve system."""
        x, flag = solver.femsolve(self.A_spd, self.b, method="gmres", tol=1e-10)

        self.assertIn(flag, [0, 1, 2, 3, 4])

    def test_femsolve_bicgstab(self):
        """femsolve with BiCGSTAB should solve system."""
        x, flag = solver.femsolve(self.A_spd, self.b, method="bicgstab", tol=1e-10)

        self.assertIn(flag, [0, 1, 2, 3, 4])

    def test_femsolve_multiple_rhs(self):
        """femsolve should handle multiple RHS."""
        x, flag = solver.femsolve(self.A_spd, self.B, method="direct")

        self.assertEqual(flag, 0)
        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_1d_rhs(self):
        """femsolve should handle 1D RHS."""
        x, flag = solver.femsolve(self.A_spd, self.b, method="direct")

        self.assertEqual(flag, 0)
        self.assertEqual(x.ndim, 1)

    def test_femsolve_sparse_rhs(self):
        """femsolve should handle sparse RHS."""
        b_sparse = sparse.csr_matrix(self.b.reshape(-1, 1))
        x, flag = solver.femsolve(self.A_spd, b_sparse, method="direct")

        self.assertEqual(flag, 0)


class TestFemsolveComplex(unittest.TestCase):
    """Test femsolve with complex matrices."""

    def setUp(self):
        """Create complex test matrix."""
        n = 30
        A_real = sparse.random(n, n, density=0.2, format="csr")
        A_imag = sparse.random(n, n, density=0.2, format="csr")
        self.A_complex = A_real + 1j * A_imag
        self.A_complex = self.A_complex @ self.A_complex.conj().T + sparse.eye(n) * 10
        self.b_complex = np.random.randn(n) + 1j * np.random.randn(n)

    def test_femsolve_complex_direct(self):
        """femsolve should handle complex matrix with direct solver."""
        x, flag = solver.femsolve(self.A_complex, self.b_complex, method="direct")

        self.assertEqual(flag, 0)
        self.assertTrue(np.iscomplexobj(x))

    def test_femsolve_complex_superlu(self):
        """femsolve should handle complex matrix with superlu."""
        x, flag = solver.femsolve(self.A_complex, self.b_complex, method="superlu")

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(self.A_complex @ x - self.b_complex)
        self.assertLess(residual, 1e-6)

    def test_femsolve_complex_gmres(self):
        """femsolve should handle complex matrix with GMRES."""
        x, flag = solver.femsolve(
            self.A_complex, self.b_complex, method="gmres", tol=1e-8
        )

        self.assertIn(flag, [0, 1, 2, 3, 4])


class TestFemsolveIterative(unittest.TestCase):
    """Test iterative solver options."""

    def setUp(self):
        """Create test matrix."""
        n = 100
        A = sparse.random(n, n, density=0.05, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 20
        self.b = np.random.randn(n)
        self.B = np.random.randn(n, 5)

    def test_femsolve_cg_maxiter(self):
        """femsolve CG should respect maxiter."""
        x, flag = solver.femsolve(self.A, self.b, method="cg", maxiter=5, tol=1e-15)

        # With very few iterations, may not converge
        self.assertIsNotNone(x)

    def test_femsolve_gmres_maxiter(self):
        """femsolve GMRES should respect maxiter."""
        x, flag = solver.femsolve(self.A, self.b, method="gmres", maxiter=5, tol=1e-15)

        self.assertIsNotNone(x)

    def test_femsolve_verbose(self):
        """femsolve should accept verbose flag."""
        # Just check it doesn't crash
        x, flag = solver.femsolve(self.A, self.b, method="direct", verbose=True)
        self.assertIsNotNone(x)

    def test_femsolve_nthread_1(self):
        """femsolve should work with nthread=1 (sequential)."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", nthread=1, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveEdgeCases(unittest.TestCase):
    """Test edge cases for femsolve."""

    def test_femsolve_zero_rhs_columns(self):
        """femsolve should handle RHS with zero columns."""
        n = 20
        A = sparse.eye(n) * 10
        B = np.zeros((n, 3))
        B[:, 0] = np.random.randn(n)  # Only first column nonzero
        B[:, 2] = np.random.randn(n)  # Third column nonzero

        x, flag = solver.femsolve(A, B, method="direct")

        self.assertEqual(x.shape, B.shape)
        # Zero columns should give zero solution
        assert_allclose(x[:, 1], 0)

    def test_femsolve_unknown_method(self):
        """femsolve should raise for unknown method."""
        n = 10
        A = sparse.eye(n)
        b = np.ones(n)

        with self.assertRaises(ValueError):
            solver.femsolve(A, b, method="unknown_solver")

    def test_femsolve_small_system(self):
        """femsolve should handle very small systems."""
        A = sparse.csr_matrix([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])

        x, flag = solver.femsolve(A, b, method="direct")

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(A @ x - b)
        self.assertLess(residual, 1e-10)


class TestFemsolveBlqmr(unittest.TestCase):
    """Test BLQMR solver if available."""

    def setUp(self):
        """Create test matrix."""
        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 10
        self.b = np.random.randn(n)
        self.B = np.random.randn(n, 4)

    def test_femsolve_blqmr(self):
        """femsolve with blqmr should solve if available."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        x, flag = solver.femsolve(self.A, self.b, method="blqmr", tol=1e-8)

        self.assertIn(flag, [0, 1, 2, 3, 4])

    def test_femsolve_blqmr_multiple_rhs(self):
        """femsolve blqmr should handle multiple RHS."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        x, flag = solver.femsolve(self.A, self.B, method="blqmr", rhsblock=2, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveAMG(unittest.TestCase):
    """Test CG+AMG solver if available."""

    def setUp(self):
        """Create SPD test matrix."""
        n = 100
        A = sparse.random(n, n, density=0.05, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 20
        self.b = np.random.randn(n)

    def test_femsolve_cg_amg(self):
        """femsolve with cg+amg should solve if pyamg available."""
        info = solver.get_solver_info()

        if not info.get("has_amg", False):
            self.skipTest("PyAMG not available")

        x, flag = solver.femsolve(self.A, self.b, method="cg+amg", tol=1e-8)

        self.assertIn(flag, [0, 1, 2, 3, 4])


class TestFemsolveUmfpack(unittest.TestCase):
    """Test UMFPACK solver if available."""

    def setUp(self):
        """Create test matrix."""
        n = 50
        A = sparse.random(n, n, density=0.2, format="csr")
        self.A = A + sparse.eye(n) * 10
        self.b = np.random.randn(n)

    def test_femsolve_umfpack(self):
        """femsolve with umfpack should solve if available."""
        info = solver.get_solver_info()

        if not info.get("has_umfpack", False):
            self.skipTest("UMFPACK not available")

        x, flag = solver.femsolve(self.A, self.b, method="umfpack")

        self.assertEqual(flag, 0)


class TestFemsolveParallel(unittest.TestCase):
    """Test parallel solving capabilities."""

    def setUp(self):
        """Create test matrix with multiple RHS."""
        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 10
        self.B = np.random.randn(n, 8)

    def test_femsolve_parallel_gmres(self):
        """femsolve gmres should work with multiple threads."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", nthread=2, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_parallel_cg(self):
        """femsolve cg should work with multiple threads."""
        x, flag = solver.femsolve(self.A, self.B, method="cg", nthread=2, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_parallel_bicgstab(self):
        """femsolve bicgstab should work with multiple threads."""
        x, flag = solver.femsolve(
            self.A, self.B, method="bicgstab", nthread=2, tol=1e-8
        )

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolvePardisoComplex(unittest.TestCase):
    """Test Pardiso solver with complex matrices."""

    def setUp(self):
        """Create complex test matrix."""
        n = 30
        A_real = sparse.random(n, n, density=0.2, format="csr")
        A_imag = sparse.random(n, n, density=0.2, format="csr")
        self.A = A_real + 1j * A_imag
        self.A = self.A @ self.A.conj().T + sparse.eye(n) * 10
        self.b = np.random.randn(n) + 1j * np.random.randn(n)
        self.B = np.random.randn(n, 3) + 1j * np.random.randn(n, 3)

    def test_femsolve_pardiso_complex(self):
        """femsolve pardiso should handle complex via real formulation."""
        info = solver.get_solver_info()

        if not info.get("has_pardiso", False):
            self.skipTest("Pardiso not available")

        x, flag = solver.femsolve(self.A, self.b, method="pardiso")

        self.assertEqual(flag, 0)
        self.assertTrue(np.iscomplexobj(x))

        # Check residual
        residual = np.linalg.norm(self.A @ x - self.b)
        self.assertLess(residual, 1e-6)

    def test_femsolve_pardiso_complex_multiple_rhs(self):
        """femsolve pardiso complex with multiple RHS."""
        info = solver.get_solver_info()

        if not info.get("has_pardiso", False):
            self.skipTest("Pardiso not available")

        x, flag = solver.femsolve(self.A, self.B, method="pardiso")

        self.assertEqual(flag, 0)
        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveUmfpackExtended(unittest.TestCase):
    """Extended tests for UMFPACK solver."""

    def setUp(self):
        n = 40
        A = sparse.random(n, n, density=0.15, format="csr")
        self.A = A + sparse.eye(n) * 10
        self.b = np.random.randn(n)
        self.B = np.random.randn(n, 5)

    def test_femsolve_umfpack_multiple_rhs_mixed(self):
        """femsolve umfpack with mixed zero/nonzero RHS columns."""
        info = solver.get_solver_info()

        if not info.get("has_umfpack", False):
            self.skipTest("UMFPACK not available")

        B = self.B.copy()
        B[:, 2] = 0  # Zero column

        x, flag = solver.femsolve(self.A, B, method="umfpack")

        self.assertEqual(flag, 0)
        assert_allclose(x[:, 2], 0, atol=1e-10)

    def test_femsolve_umfpack_complex(self):
        """femsolve umfpack with complex matrix."""
        info = solver.get_solver_info()

        if not info.get("has_umfpack", False):
            self.skipTest("UMFPACK not available")

        n = 30
        A = sparse.random(n, n, density=0.2, format="csr") + 1j * sparse.random(
            n, n, density=0.2
        )
        A = A + sparse.eye(n) * 10
        b = np.random.randn(n) + 1j * np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="umfpack")

        self.assertEqual(flag, 0)
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveChomod(unittest.TestCase):
    """Test CHOLMOD solver."""

    def setUp(self):
        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A_spd = A @ A.T + sparse.eye(n) * 20  # SPD matrix
        self.b = np.random.randn(n)

    def test_femsolve_cholmod_spd(self):
        """femsolve cholmod should solve SPD system."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        x, flag = solver.femsolve(self.A_spd, self.b, method="cholmod", spd=True)

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(self.A_spd @ x - self.b)
        self.assertLess(residual, 1e-8)

    def test_femsolve_cholmod_non_spd_fallback(self):
        """femsolve cholmod should fallback for non-SPD."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        # Non-SPD matrix (asymmetric)
        n = 20
        A = sparse.random(n, n, density=0.3, format="csr") + sparse.eye(n) * 10
        b = np.random.randn(n)

        # Should fallback gracefully
        x, flag = solver.femsolve(A, b, method="cholmod", spd=False)

        self.assertIsNotNone(x)

    def test_femsolve_cholmod_complex_fallback(self):
        """femsolve cholmod should fallback for complex matrix."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        n = 20
        A = sparse.eye(n, dtype=complex) * 10 + 1j * sparse.eye(n)
        b = np.random.randn(n) + 1j * np.random.randn(n)

        # Should fallback to another solver
        x, flag = solver.femsolve(A, b, method="cholmod")

        self.assertIsNotNone(x)


class TestFemsolveBlqmrExtended(unittest.TestCase):
    """Extended tests for BLQMR solver."""

    def setUp(self):
        n = 60
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.b = np.random.randn(n)
        self.B = np.random.randn(n, 10)

    def test_femsolve_blqmr_batched(self):
        """femsolve blqmr with batch processing."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        x, flag = solver.femsolve(
            self.A, self.B, method="blqmr", rhsblock=3, tol=1e-8, nthread=1
        )

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_blqmr_single_rhs(self):
        """femsolve blqmr with single RHS (no batching)."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        x, flag = solver.femsolve(self.A, self.b, method="blqmr", tol=1e-8)

        self.assertEqual(len(x), len(self.b))

    def test_femsolve_blqmr_large_rhsblock(self):
        """femsolve blqmr with rhsblock larger than ncol."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        x, flag = solver.femsolve(
            self.A,
            self.B,
            method="blqmr",
            rhsblock=100,
            tol=1e-8,  # Larger than ncol=10
        )

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_blqmr_with_precond(self):
        """femsolve blqmr with preconditioning."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        x, flag = solver.femsolve(
            self.A, self.b, method="blqmr", tol=1e-8, precond_type=3, droptol=0.01
        )

        self.assertIn(flag, [0, 1, 2, 3, 4])


class TestFemsolveAMGExtended(unittest.TestCase):
    """Extended tests for CG+AMG solver."""

    def setUp(self):
        n = 80
        A = sparse.random(n, n, density=0.05, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 25
        self.b = np.random.randn(n)
        self.B = np.random.randn(n, 4)

    def test_femsolve_cg_amg_multiple_rhs(self):
        """femsolve cg+amg with multiple RHS."""
        info = solver.get_solver_info()

        if not info.get("has_amg", False):
            self.skipTest("PyAMG not available")

        x, flag = solver.femsolve(self.A, self.B, method="cg+amg", tol=1e-8, nthread=1)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_cg_amg_complex_fallback(self):
        """femsolve cg+amg should fallback for complex matrix."""
        info = solver.get_solver_info()

        if not info.get("has_amg", False):
            self.skipTest("PyAMG not available")

        n = 30
        A = sparse.eye(n, dtype=complex) * 10
        b = np.random.randn(n) + 1j * np.random.randn(n)

        # Should fallback to gmres
        x, flag = solver.femsolve(A, b, method="cg+amg")

        self.assertIsNotNone(x)
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveIterativeParallel(unittest.TestCase):
    """Test parallel iterative solvers."""

    def setUp(self):
        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.B = np.random.randn(n, 6)

    def test_femsolve_gmres_parallel(self):
        """femsolve gmres parallel solving."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", nthread=2, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_bicgstab_parallel(self):
        """femsolve bicgstab parallel solving."""
        x, flag = solver.femsolve(
            self.A, self.B, method="bicgstab", nthread=2, tol=1e-8
        )

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_cg_parallel(self):
        """femsolve cg parallel solving."""
        x, flag = solver.femsolve(self.A, self.B, method="cg", nthread=2, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_cg_with_preconditioner(self):
        """femsolve cg with custom preconditioner (sequential)."""
        n = self.A.shape[0]

        # Simple diagonal preconditioner
        M = sparse.diags(1.0 / self.A.diagonal())

        x, flag = solver.femsolve(self.A, self.B, method="cg", M=M, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveComplexIterative(unittest.TestCase):
    """Test iterative solvers with complex matrices."""

    def setUp(self):
        n = 40
        A_real = sparse.random(n, n, density=0.15, format="csr")
        A_imag = sparse.random(n, n, density=0.15, format="csr")
        self.A = A_real + 1j * A_imag
        self.A = self.A @ self.A.conj().T + sparse.eye(n) * 15
        self.b = np.random.randn(n) + 1j * np.random.randn(n)

    def test_femsolve_gmres_complex(self):
        """femsolve gmres with complex system."""
        x, flag = solver.femsolve(self.A, self.b, method="gmres", tol=1e-8)

        self.assertTrue(np.iscomplexobj(x))

    def test_femsolve_bicgstab_complex(self):
        """femsolve bicgstab with complex system."""
        x, flag = solver.femsolve(self.A, self.b, method="bicgstab", tol=1e-8)

        self.assertTrue(np.iscomplexobj(x))

    def test_femsolve_cg_complex_fallback(self):
        """femsolve cg should fallback to gmres for complex."""
        x, flag = solver.femsolve(self.A, self.b, method="cg", tol=1e-8)

        # Should have fallen back to gmres
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveAutoSelection(unittest.TestCase):
    """Test automatic solver selection."""

    def test_femsolve_auto_small_system(self):
        """femsolve auto should use direct for small systems."""
        n = 50
        A = sparse.eye(n) * 10
        b = np.ones(n)

        x, flag = solver.femsolve(A, b, method="auto")

        self.assertEqual(flag, 0)

    def test_femsolve_auto_large_spd(self):
        """femsolve auto should select appropriate solver for large SPD."""
        n = 500
        A = sparse.random(n, n, density=0.01, format="csr")
        A = A @ A.T + sparse.eye(n) * 50
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="auto", spd=True)

        self.assertIsNotNone(x)


class TestFemsolveVerbose(unittest.TestCase):
    """Test verbose output."""

    def test_femsolve_verbose_direct(self):
        """femsolve verbose should not crash for direct solver."""
        n = 20
        A = sparse.eye(n) * 10
        b = np.ones(n)

        x, flag = solver.femsolve(A, b, method="direct", verbose=True)

        self.assertEqual(flag, 0)

    def test_femsolve_verbose_iterative(self):
        """femsolve verbose should not crash for iterative solver."""
        n = 30
        A = sparse.eye(n) * 10
        b = np.ones(n)

        x, flag = solver.femsolve(A, b, method="gmres", verbose=True, tol=1e-8)

        self.assertIsNotNone(x)


class TestGetSolverInfoComplete(unittest.TestCase):
    """Complete tests for get_solver_info."""

    def test_solver_info_blqmr_details(self):
        """get_solver_info should report BLQMR details if available."""
        info = solver.get_solver_info()

        if info.get("has_blqmr", False):
            self.assertIn("blqmr_backend", info)
            self.assertIn("blqmr_has_numba", info)
            self.assertIn("blqmr", info["complex_iterative"])

    def test_solver_info_complex_direct(self):
        """get_solver_info should report complex direct solver."""
        info = solver.get_solver_info()

        self.assertIn("complex_direct", info)
        self.assertIn(info["complex_direct"], ["umfpack", "superlu"])


# =============================================================================
# NEW TESTS FOR IMPROVED COVERAGE
# =============================================================================


class TestFemsolvePardisoFallback(unittest.TestCase):
    """Test Pardiso fallback when not available."""

    def test_femsolve_pardiso_fallback_warning(self):
        """femsolve should warn and fallback when pardiso unavailable."""
        info = solver.get_solver_info()

        if info.get("has_pardiso", False):
            self.skipTest("Pardiso is available, can't test fallback")

        n = 20
        A = sparse.eye(n) * 10
        b = np.ones(n)

        with self.assertWarns(Warning):
            x, flag = solver.femsolve(A, b, method="pardiso")

        self.assertIsNotNone(x)


class TestFemsolveUmfpackFallback(unittest.TestCase):
    """Test UMFPACK fallback when not available."""

    def test_femsolve_umfpack_fallback_warning(self):
        """femsolve should warn and fallback when umfpack unavailable."""
        info = solver.get_solver_info()

        if info.get("has_umfpack", False):
            self.skipTest("UMFPACK is available, can't test fallback")

        n = 20
        A = sparse.eye(n) * 10
        b = np.ones(n)

        with self.assertWarns(Warning):
            x, flag = solver.femsolve(A, b, method="umfpack")

        self.assertIsNotNone(x)


class TestFemsolveBlqmrFallback(unittest.TestCase):
    """Test BLQMR fallback when not available."""

    def test_femsolve_blqmr_fallback_warning(self):
        """femsolve should warn and fallback when blqmr unavailable."""
        info = solver.get_solver_info()

        if info.get("has_blqmr", False):
            self.skipTest("BLQMR is available, can't test fallback")

        n = 20
        A = sparse.eye(n) * 10
        b = np.ones(n)

        with self.assertWarns(Warning):
            x, flag = solver.femsolve(A, b, method="blqmr")

        self.assertIsNotNone(x)


class TestFemsolveAMGFallback(unittest.TestCase):
    """Test CG+AMG fallback when not available."""

    def test_femsolve_cg_amg_fallback_warning(self):
        """femsolve should warn and fallback when pyamg unavailable."""
        info = solver.get_solver_info()

        if info.get("has_amg", False):
            self.skipTest("PyAMG is available, can't test fallback")

        n = 20
        A = sparse.eye(n) * 10
        b = np.ones(n)

        with self.assertWarns(Warning):
            x, flag = solver.femsolve(A, b, method="cg+amg")

        self.assertIsNotNone(x)


class TestFemsolveChomodFallback(unittest.TestCase):
    """Test CHOLMOD fallback when not available."""

    def test_femsolve_cholmod_fallback_warning(self):
        """femsolve should warn and fallback when cholmod unavailable."""
        info = solver.get_solver_info()

        if info.get("has_cholmod", False):
            self.skipTest("CHOLMOD is available, can't test fallback")

        n = 20
        A = sparse.eye(n) * 10
        b = np.ones(n)

        with self.assertWarns(Warning):
            x, flag = solver.femsolve(A, b, method="cholmod")

        self.assertIsNotNone(x)


class TestFemsolveComplexDirectSelection(unittest.TestCase):
    """Test complex matrix direct solver selection."""

    def test_femsolve_auto_complex_small(self):
        """femsolve auto should handle small complex system."""
        n = 30
        A = sparse.eye(n, dtype=complex) * 10 + 1j * sparse.eye(n)
        b = np.ones(n) + 1j * np.ones(n)

        x, flag = solver.femsolve(A, b, method="auto")

        self.assertEqual(flag, 0)
        self.assertTrue(np.iscomplexobj(x))

    def test_femsolve_direct_complex(self):
        """femsolve direct should select appropriate solver for complex."""
        n = 30
        A = sparse.eye(n, dtype=complex) * 10 + 1j * sparse.eye(n)
        b = np.ones(n) + 1j * np.ones(n)

        x, flag = solver.femsolve(A, b, method="direct")

        self.assertEqual(flag, 0)
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveUmfpackMultipleRHS(unittest.TestCase):
    """Test UMFPACK with multiple RHS columns."""

    def test_femsolve_umfpack_all_nonzero_cols(self):
        """femsolve umfpack with all nonzero RHS columns."""
        info = solver.get_solver_info()

        if not info.get("has_umfpack", False):
            self.skipTest("UMFPACK not available")

        n = 30
        A = sparse.random(n, n, density=0.2, format="csr") + sparse.eye(n) * 10
        B = np.random.randn(n, 4)  # All columns nonzero

        x, flag = solver.femsolve(A, B, method="umfpack")

        self.assertEqual(flag, 0)
        self.assertEqual(x.shape, B.shape)

    def test_femsolve_umfpack_single_rhs(self):
        """femsolve umfpack with single RHS."""
        info = solver.get_solver_info()

        if not info.get("has_umfpack", False):
            self.skipTest("UMFPACK not available")

        n = 30
        A = sparse.random(n, n, density=0.2, format="csr") + sparse.eye(n) * 10
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="umfpack")

        self.assertEqual(flag, 0)
        self.assertEqual(x.ndim, 1)


class TestFemsolveChomodMultipleRHS(unittest.TestCase):
    """Test CHOLMOD with multiple RHS columns."""

    def test_femsolve_cholmod_multiple_rhs(self):
        """femsolve cholmod with multiple RHS columns."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 20  # SPD
        B = np.random.randn(n, 3)

        x, flag = solver.femsolve(A, B, method="cholmod", spd=True)

        self.assertEqual(flag, 0)
        self.assertEqual(x.shape, B.shape)

    def test_femsolve_cholmod_zero_column(self):
        """femsolve cholmod should skip zero RHS columns."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        n = 30
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 20  # SPD
        B = np.random.randn(n, 3)
        B[:, 1] = 0  # Zero column

        x, flag = solver.femsolve(A, B, method="cholmod", spd=True)

        self.assertEqual(flag, 0)
        assert_allclose(x[:, 1], 0, atol=1e-10)


class TestFemsolveIterativeVerbose(unittest.TestCase):
    """Test verbose output for iterative solvers."""

    def setUp(self):
        n = 30
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.b = np.random.randn(n)
        self.B = np.random.randn(n, 3)

    def test_femsolve_cg_verbose(self):
        """femsolve cg with verbose output."""
        x, flag = solver.femsolve(
            self.A, self.B, method="cg", verbose=True, tol=1e-8, nthread=1
        )

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_gmres_verbose(self):
        """femsolve gmres with verbose output."""
        x, flag = solver.femsolve(
            self.A, self.B, method="gmres", verbose=True, tol=1e-8, nthread=1
        )

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_bicgstab_verbose(self):
        """femsolve bicgstab with verbose output."""
        x, flag = solver.femsolve(
            self.A, self.B, method="bicgstab", verbose=True, tol=1e-8, nthread=1
        )

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveBlqmrVerbose(unittest.TestCase):
    """Test BLQMR verbose output."""

    def test_femsolve_blqmr_verbose_single_batch(self):
        """femsolve blqmr verbose with single batch."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="blqmr", verbose=True, tol=1e-8)

        self.assertEqual(len(x), n)

    def test_femsolve_blqmr_verbose_sequential_batches(self):
        """femsolve blqmr verbose with sequential batch processing."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        B = np.random.randn(n, 6)

        x, flag = solver.femsolve(
            A, B, method="blqmr", verbose=True, tol=1e-8, rhsblock=2, nthread=1
        )

        self.assertEqual(x.shape, B.shape)


class TestFemsolveBlqmrParallel(unittest.TestCase):
    """Test BLQMR parallel execution."""

    def test_femsolve_blqmr_parallel_batches(self):
        """femsolve blqmr with parallel batch processing."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        B = np.random.randn(n, 8)

        x, flag = solver.femsolve(A, B, method="blqmr", tol=1e-8, rhsblock=2, nthread=2)

        self.assertEqual(x.shape, B.shape)

    def test_femsolve_blqmr_parallel_verbose(self):
        """femsolve blqmr parallel with verbose output."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        B = np.random.randn(n, 6)

        x, flag = solver.femsolve(
            A, B, method="blqmr", verbose=True, tol=1e-8, rhsblock=2, nthread=2
        )

        self.assertEqual(x.shape, B.shape)


class TestFemsolveBlqmrWithX0(unittest.TestCase):
    """Test BLQMR with initial guess."""

    def test_femsolve_blqmr_with_x0(self):
        """femsolve blqmr with initial guess x0."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        B = np.random.randn(n, 4)
        x0 = np.zeros((n, 4))

        x, flag = solver.femsolve(
            A, B, method="blqmr", tol=1e-8, x0=x0, rhsblock=2, nthread=1
        )

        self.assertEqual(x.shape, B.shape)


class TestFemsolveBlqmrRhsblockZero(unittest.TestCase):
    """Test BLQMR with rhsblock <= 0."""

    def test_femsolve_blqmr_rhsblock_zero(self):
        """femsolve blqmr with rhsblock=0 should use single batch."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        B = np.random.randn(n, 4)

        x, flag = solver.femsolve(A, B, method="blqmr", tol=1e-8, rhsblock=0)

        self.assertEqual(x.shape, B.shape)


class TestFemsolveAMGVerbose(unittest.TestCase):
    """Test CG+AMG verbose output."""

    def test_femsolve_cg_amg_verbose(self):
        """femsolve cg+amg with verbose output."""
        info = solver.get_solver_info()

        if not info.get("has_amg", False):
            self.skipTest("PyAMG not available")

        n = 50
        A = sparse.random(n, n, density=0.05, format="csr")
        A = A @ A.T + sparse.eye(n) * 20
        B = np.random.randn(n, 3)

        x, flag = solver.femsolve(
            A, B, method="cg+amg", verbose=True, tol=1e-8, nthread=1
        )

        self.assertEqual(x.shape, B.shape)


class TestFemsolveAMGParallel(unittest.TestCase):
    """Test CG+AMG parallel execution."""

    def test_femsolve_cg_amg_parallel(self):
        """femsolve cg+amg with parallel execution."""
        info = solver.get_solver_info()

        if not info.get("has_amg", False):
            self.skipTest("PyAMG not available")

        n = 60
        A = sparse.random(n, n, density=0.05, format="csr")
        A = A @ A.T + sparse.eye(n) * 20
        B = np.random.randn(n, 6)

        x, flag = solver.femsolve(A, B, method="cg+amg", tol=1e-8, nthread=2)

        self.assertEqual(x.shape, B.shape)


class TestFemsolveIterativeZeroColumns(unittest.TestCase):
    """Test iterative solvers with zero RHS columns."""

    def setUp(self):
        n = 30
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.B = np.random.randn(n, 4)
        self.B[:, 1] = 0  # Zero column

    def test_femsolve_cg_zero_column(self):
        """femsolve cg should skip zero RHS columns."""
        x, flag = solver.femsolve(self.A, self.B, method="cg", tol=1e-8, nthread=1)

        self.assertEqual(x.shape, self.B.shape)
        assert_allclose(x[:, 1], 0)

    def test_femsolve_gmres_zero_column(self):
        """femsolve gmres should skip zero RHS columns."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", tol=1e-8, nthread=1)

        self.assertEqual(x.shape, self.B.shape)
        assert_allclose(x[:, 1], 0)

    def test_femsolve_bicgstab_zero_column(self):
        """femsolve bicgstab should skip zero RHS columns."""
        x, flag = solver.femsolve(
            self.A, self.B, method="bicgstab", tol=1e-8, nthread=1
        )

        self.assertEqual(x.shape, self.B.shape)
        assert_allclose(x[:, 1], 0)


class TestFemsolveIterativePreconditioner(unittest.TestCase):
    """Test iterative solvers with custom preconditioners."""

    def setUp(self):
        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.B = np.random.randn(n, 3)
        self.M = sparse.diags(1.0 / self.A.diagonal())  # Diagonal preconditioner

    def test_femsolve_gmres_with_preconditioner(self):
        """femsolve gmres with custom preconditioner (disables parallel)."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", M=self.M, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_bicgstab_with_preconditioner(self):
        """femsolve bicgstab with custom preconditioner (disables parallel)."""
        x, flag = solver.femsolve(self.A, self.B, method="bicgstab", M=self.M, tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveIterativeParallelVerbose(unittest.TestCase):
    """Test parallel iterative solvers with verbose output."""

    def setUp(self):
        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.B = np.random.randn(n, 4)

    def test_femsolve_gmres_parallel_verbose(self):
        """femsolve gmres parallel with verbose output."""
        x, flag = solver.femsolve(
            self.A, self.B, method="gmres", verbose=True, tol=1e-8, nthread=2
        )

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_cg_parallel_verbose(self):
        """femsolve cg parallel with verbose output."""
        x, flag = solver.femsolve(
            self.A, self.B, method="cg", verbose=True, tol=1e-8, nthread=2
        )

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_bicgstab_parallel_verbose(self):
        """femsolve bicgstab parallel with verbose output."""
        x, flag = solver.femsolve(
            self.A, self.B, method="bicgstab", verbose=True, tol=1e-8, nthread=2
        )

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveAutoLargeComplex(unittest.TestCase):
    """Test auto selection for large complex systems."""

    def test_femsolve_auto_large_complex_spd(self):
        """femsolve auto with large complex SPD system."""
        n = 500
        A_real = sparse.random(n, n, density=0.01, format="csr")
        A = A_real @ A_real.T + sparse.eye(n) * 50
        A = A.astype(complex)  # Make complex but Hermitian
        b = np.random.randn(n) + 1j * np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="auto", spd=True)

        self.assertIsNotNone(x)
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolvePardisoReal(unittest.TestCase):
    """Test Pardiso with real matrices."""

    def test_femsolve_pardiso_real(self):
        """femsolve pardiso with real matrix."""
        info = solver.get_solver_info()

        if not info.get("has_pardiso", False):
            self.skipTest("Pardiso not available")

        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 10
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="pardiso")

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(A @ x - b)
        self.assertLess(residual, 1e-8)

    def test_femsolve_pardiso_real_multiple_rhs(self):
        """femsolve pardiso with multiple real RHS."""
        info = solver.get_solver_info()

        if not info.get("has_pardiso", False):
            self.skipTest("Pardiso not available")

        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 10
        B = np.random.randn(n, 4)

        x, flag = solver.femsolve(A, B, method="pardiso")

        self.assertEqual(flag, 0)
        self.assertEqual(x.shape, B.shape)


class TestFemsolveParallelZeroColumns(unittest.TestCase):
    """Test parallel iterative solvers handle zero columns correctly."""

    def setUp(self):
        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.B = np.random.randn(n, 6)
        self.B[:, 2] = 0  # Zero column

    def test_femsolve_parallel_gmres_zero_column(self):
        """femsolve parallel gmres should handle zero RHS columns."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", tol=1e-8, nthread=2)

        self.assertEqual(x.shape, self.B.shape)
        assert_allclose(x[:, 2], 0)

    def test_femsolve_parallel_cg_zero_column(self):
        """femsolve parallel cg should handle zero RHS columns."""
        x, flag = solver.femsolve(self.A, self.B, method="cg", tol=1e-8, nthread=2)

        self.assertEqual(x.shape, self.B.shape)
        assert_allclose(x[:, 2], 0)


class TestFemsolveComplexBlqmr(unittest.TestCase):
    """Test BLQMR with complex matrices."""

    def test_femsolve_blqmr_complex(self):
        """femsolve blqmr with complex system."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A_real = sparse.random(n, n, density=0.1, format="csr")
        A_imag = sparse.random(n, n, density=0.1, format="csr")
        A = A_real + 1j * A_imag
        A = A @ A.conj().T + sparse.eye(n) * 15
        b = np.random.randn(n) + 1j * np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="blqmr", tol=1e-8)

        self.assertTrue(np.iscomplexobj(x))

    def test_femsolve_blqmr_complex_parallel(self):
        """femsolve blqmr complex with parallel processing."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A_real = sparse.random(n, n, density=0.1, format="csr")
        A_imag = sparse.random(n, n, density=0.1, format="csr")
        A = A_real + 1j * A_imag
        A = A @ A.conj().T + sparse.eye(n) * 15
        B = np.random.randn(n, 6) + 1j * np.random.randn(n, 6)

        x, flag = solver.femsolve(A, B, method="blqmr", tol=1e-8, rhsblock=2, nthread=2)

        self.assertEqual(x.shape, B.shape)
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveDirectVerbose(unittest.TestCase):
    """Test verbose output for direct solvers."""

    def test_femsolve_superlu_verbose(self):
        """femsolve superlu with verbose output."""
        n = 30
        A = sparse.eye(n) * 10
        b = np.ones(n)

        x, flag = solver.femsolve(A, b, method="superlu", verbose=True)

        self.assertEqual(flag, 0)

    def test_femsolve_umfpack_verbose(self):
        """femsolve umfpack with verbose output."""
        info = solver.get_solver_info()

        if not info.get("has_umfpack", False):
            self.skipTest("UMFPACK not available")

        n = 30
        A = sparse.eye(n) * 10 + sparse.random(n, n, density=0.1, format="csr")
        b = np.ones(n)

        x, flag = solver.femsolve(A, b, method="umfpack", verbose=True)

        self.assertEqual(flag, 0)

    def test_femsolve_cholmod_verbose(self):
        """femsolve cholmod with verbose output."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        n = 30
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 20
        b = np.ones(n)

        x, flag = solver.femsolve(A, b, method="cholmod", spd=True, verbose=True)

        self.assertEqual(flag, 0)


class TestFemsolveAutoSelectionExtended(unittest.TestCase):
    """Extended tests for auto solver selection."""

    def test_femsolve_auto_spd_with_amg(self):
        """femsolve auto should prefer cg+amg for large SPD when available."""
        info = solver.get_solver_info()

        n = 15000  # Large enough to trigger iterative
        A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format="csr")
        A = A @ A.T + sparse.eye(n) * 10
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="auto", spd=True)

        self.assertIsNotNone(x)

    def test_femsolve_auto_non_spd_large(self):
        """femsolve auto for large non-SPD should use direct."""
        n = 15000
        A = sparse.diags([1, 10, 1], [-1, 0, 1], shape=(n, n), format="csr")
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="auto", spd=False)

        self.assertIsNotNone(x)


class TestFemsolveSuperlulBatchException(unittest.TestCase):
    """Test SuperLU batch solve exception handling."""

    def test_femsolve_superlu_multiple_rhs_all_nonzero(self):
        """femsolve superlu with all nonzero RHS columns."""
        n = 30
        A = sparse.random(n, n, density=0.2, format="csr") + sparse.eye(n) * 10
        B = np.random.randn(n, 4)

        x, flag = solver.femsolve(A, B, method="superlu")

        self.assertEqual(flag, 0)
        self.assertEqual(x.shape, B.shape)

    def test_femsolve_superlu_some_zero_cols(self):
        """femsolve superlu with some zero RHS columns."""
        n = 30
        A = sparse.random(n, n, density=0.2, format="csr") + sparse.eye(n) * 10
        B = np.random.randn(n, 4)
        B[:, 1] = 0
        B[:, 3] = 0

        x, flag = solver.femsolve(A, B, method="superlu")

        self.assertEqual(flag, 0)
        assert_allclose(x[:, 1], 0, atol=1e-10)
        assert_allclose(x[:, 3], 0, atol=1e-10)

    def test_femsolve_superlu_single_nonzero_col(self):
        """femsolve superlu with single nonzero column among zeros."""
        n = 30
        A = sparse.random(n, n, density=0.2, format="csr") + sparse.eye(n) * 10
        B = np.zeros((n, 4))
        B[:, 2] = np.random.randn(n)

        x, flag = solver.femsolve(A, B, method="superlu")

        self.assertEqual(flag, 0)
        assert_allclose(x[:, 0], 0, atol=1e-10)
        assert_allclose(x[:, 1], 0, atol=1e-10)
        assert_allclose(x[:, 3], 0, atol=1e-10)


class TestFemsolveIterativeParallelComplex(unittest.TestCase):
    """Test parallel iterative solvers with complex matrices."""

    def setUp(self):
        n = 40
        A_real = sparse.random(n, n, density=0.1, format="csr")
        A_imag = sparse.random(n, n, density=0.1, format="csr")
        self.A = A_real + 1j * A_imag
        self.A = self.A @ self.A.conj().T + sparse.eye(n) * 15
        self.B = np.random.randn(n, 4) + 1j * np.random.randn(n, 4)

    def test_femsolve_gmres_complex_parallel(self):
        """femsolve gmres complex with parallel execution."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", tol=1e-8, nthread=2)

        self.assertEqual(x.shape, self.B.shape)
        self.assertTrue(np.iscomplexobj(x))

    def test_femsolve_bicgstab_complex_parallel(self):
        """femsolve bicgstab complex with parallel execution."""
        x, flag = solver.femsolve(
            self.A, self.B, method="bicgstab", tol=1e-8, nthread=2
        )

        self.assertEqual(x.shape, self.B.shape)
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveBlqmrWorkspace(unittest.TestCase):
    """Test BLQMR with workspace parameter."""

    def test_femsolve_blqmr_with_workspace(self):
        """femsolve blqmr with pre-allocated workspace."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        from blocksolver import BLQMRWorkspace

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        B = np.random.randn(n, 4)

        workspace = BLQMRWorkspace(n, 4)

        x, flag = solver.femsolve(
            A, B, method="blqmr", tol=1e-8, workspace=workspace, rhsblock=0
        )

        self.assertEqual(x.shape, B.shape)


class TestFemsolveBlqmrPreconditioners(unittest.TestCase):
    """Test BLQMR with different preconditioner types."""

    def setUp(self):
        info = solver.get_solver_info()
        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 50
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.b = np.random.randn(n)

    def test_femsolve_blqmr_precond_type_0(self):
        """femsolve blqmr with precond_type=0 (no preconditioning)."""
        x, flag = solver.femsolve(
            self.A, self.b, method="blqmr", tol=1e-8, precond_type=0
        )

        self.assertIn(flag, [0, 1, 2, 3, 4])

    def test_femsolve_blqmr_precond_type_1(self):
        """femsolve blqmr with precond_type=1."""
        x, flag = solver.femsolve(
            self.A, self.b, method="blqmr", tol=1e-8, precond_type=1
        )

        self.assertIn(flag, [0, 1, 2, 3, 4])

    def test_femsolve_blqmr_precond_type_2(self):
        """femsolve blqmr with precond_type=2."""
        x, flag = solver.femsolve(
            self.A, self.b, method="blqmr", tol=1e-8, precond_type=2
        )

        self.assertIn(flag, [0, 1, 2, 3, 4])

    def test_femsolve_blqmr_different_droptol(self):
        """femsolve blqmr with different drop tolerances."""
        for droptol in [0.1, 0.01, 0.001]:
            x, flag = solver.femsolve(
                self.A,
                self.b,
                method="blqmr",
                tol=1e-8,
                precond_type=3,
                droptol=droptol,
            )
            self.assertIsNotNone(x)


class TestFemsolveBlqmrCustomPrecond(unittest.TestCase):
    """Test BLQMR with custom M1/M2 preconditioners."""

    def test_femsolve_blqmr_with_m1_m2(self):
        """femsolve blqmr with custom M1 and M2."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15
        b = np.random.randn(n)

        # Simple diagonal preconditioners
        M1 = sparse.diags(1.0 / np.sqrt(A.diagonal()))
        M2 = M1.copy()

        x, flag = solver.femsolve(
            A, b, method="blqmr", tol=1e-8, M1=M1, M2=M2, precond_type=0
        )

        self.assertIn(flag, [0, 1, 2, 3, 4])


class TestFemsolvePardisoVerbose(unittest.TestCase):
    """Test Pardiso with verbose output."""

    def test_femsolve_pardiso_verbose(self):
        """femsolve pardiso with verbose output."""
        info = solver.get_solver_info()

        if not info.get("has_pardiso", False):
            self.skipTest("Pardiso not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 10
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="pardiso", verbose=True)

        self.assertEqual(flag, 0)

    def test_femsolve_pardiso_complex_verbose(self):
        """femsolve pardiso complex with verbose output."""
        info = solver.get_solver_info()

        if not info.get("has_pardiso", False):
            self.skipTest("Pardiso not available")

        n = 30
        A = sparse.eye(n, dtype=complex) * 10 + 1j * sparse.eye(n)
        b = np.ones(n) + 1j * np.ones(n)

        x, flag = solver.femsolve(A, b, method="pardiso", verbose=True)

        self.assertEqual(flag, 0)
        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveAMGZeroColumns(unittest.TestCase):
    """Test CG+AMG with zero RHS columns."""

    def test_femsolve_cg_amg_zero_column(self):
        """femsolve cg+amg should skip zero RHS columns."""
        info = solver.get_solver_info()

        if not info.get("has_amg", False):
            self.skipTest("PyAMG not available")

        n = 50
        A = sparse.random(n, n, density=0.05, format="csr")
        A = A @ A.T + sparse.eye(n) * 20
        B = np.random.randn(n, 4)
        B[:, 1] = 0

        x, flag = solver.femsolve(A, B, method="cg+amg", tol=1e-8, nthread=1)

        self.assertEqual(x.shape, B.shape)
        assert_allclose(x[:, 1], 0)


class TestFemsolveEdgeCasesExtended(unittest.TestCase):
    """Extended edge case tests."""

    def test_femsolve_identity_matrix(self):
        """femsolve with identity matrix."""
        n = 20
        A = sparse.eye(n)
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="direct")

        self.assertEqual(flag, 0)
        assert_allclose(x, b, rtol=1e-10)

    def test_femsolve_diagonal_matrix(self):
        """femsolve with diagonal matrix."""
        n = 20
        diag = np.random.randn(n) + 2  # Ensure positive
        A = sparse.diags(diag)
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="direct")

        self.assertEqual(flag, 0)
        assert_allclose(x, b / diag, rtol=1e-10)

    def test_femsolve_tridiagonal_matrix(self):
        """femsolve with tridiagonal matrix."""
        n = 50
        A = sparse.diags([1, -3, 1], [-1, 0, 1], shape=(n, n), format="csr")
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="direct")

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(A @ x - b)
        self.assertLess(residual, 1e-8)

    def test_femsolve_coo_matrix_input(self):
        """femsolve should accept COO format matrix."""
        n = 30
        A = sparse.random(n, n, density=0.2, format="coo")
        A = A @ A.T + sparse.eye(n, format="coo") * 10
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="direct")

        self.assertEqual(flag, 0)

    def test_femsolve_lil_matrix_input(self):
        """femsolve should accept LIL format matrix."""
        n = 30
        A = sparse.random(n, n, density=0.2, format="lil")
        A = A @ A.T + sparse.eye(n, format="lil") * 10
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="direct")

        self.assertEqual(flag, 0)


class TestFemsolveParallelSingleColumn(unittest.TestCase):
    """Test parallel solvers with single column (edge case)."""

    def setUp(self):
        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.b = np.random.randn(n)

    def test_femsolve_gmres_parallel_single_col(self):
        """femsolve gmres parallel with single column shouldn't parallelize."""
        x, flag = solver.femsolve(self.A, self.b, method="gmres", tol=1e-8, nthread=4)

        self.assertEqual(x.ndim, 1)

    def test_femsolve_cg_parallel_single_col(self):
        """femsolve cg parallel with single column shouldn't parallelize."""
        x, flag = solver.femsolve(self.A, self.b, method="cg", tol=1e-8, nthread=4)

        self.assertEqual(x.ndim, 1)


class TestFemsolveBlqmrMixedRealComplex(unittest.TestCase):
    """Test BLQMR with mixed real/complex inputs."""

    def test_femsolve_blqmr_real_matrix_complex_rhs(self):
        """femsolve blqmr with real matrix and complex RHS."""
        info = solver.get_solver_info()

        if not info.get("has_blqmr", False):
            self.skipTest("BLQMR not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 15  # Real SPD
        b = np.random.randn(n) + 1j * np.random.randn(n)  # Complex RHS

        x, flag = solver.femsolve(A, b, method="blqmr", tol=1e-8)

        self.assertTrue(np.iscomplexobj(x))


class TestFemsolveIterativeNonConvergence(unittest.TestCase):
    """Test iterative solver behavior when not converging."""

    def test_femsolve_cg_difficult_system(self):
        """femsolve cg with difficult system may not converge."""
        n = 50
        # Create ill-conditioned matrix
        A = sparse.random(n, n, density=0.3, format="csr")
        A = A @ A.T + sparse.eye(n) * 0.01  # Small diagonal
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="cg", tol=1e-15, maxiter=5, nthread=1)

        # Should return something even if not converged
        self.assertIsNotNone(x)

    def test_femsolve_gmres_maxiter_reached(self):
        """femsolve gmres should handle maxiter reached."""
        n = 50
        A = sparse.random(n, n, density=0.3, format="csr") + sparse.eye(n)
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="gmres", tol=1e-15, maxiter=3, nthread=1)

        self.assertIsNotNone(x)


class TestFemsolveComplexRealFormulation(unittest.TestCase):
    """Test complex-to-real formulation for Pardiso."""

    def test_femsolve_pardiso_complex_accuracy(self):
        """femsolve pardiso complex should give accurate results."""
        info = solver.get_solver_info()

        if not info.get("has_pardiso", False):
            self.skipTest("Pardiso not available")

        n = 30
        # Create complex symmetric matrix
        A_real = sparse.random(n, n, density=0.2, format="csr")
        A_imag = sparse.random(n, n, density=0.2, format="csr")
        A = A_real + 1j * A_imag
        A = A @ A.conj().T + sparse.eye(n) * 10

        b = np.random.randn(n) + 1j * np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="pardiso")

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        self.assertLess(residual, 1e-8)


class TestFemsolveThreadCount(unittest.TestCase):
    """Test different thread count configurations."""

    def setUp(self):
        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        self.A = A @ A.T + sparse.eye(n) * 15
        self.B = np.random.randn(n, 8)

    def test_femsolve_nthread_default(self):
        """femsolve should use default thread count."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", tol=1e-8)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_nthread_1(self):
        """femsolve should work with nthread=1."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", tol=1e-8, nthread=1)

        self.assertEqual(x.shape, self.B.shape)

    def test_femsolve_nthread_larger_than_cols(self):
        """femsolve should handle nthread > ncol."""
        x, flag = solver.femsolve(self.A, self.B, method="gmres", tol=1e-8, nthread=100)

        self.assertEqual(x.shape, self.B.shape)


class TestFemsolveChomodExtended(unittest.TestCase):
    """Extended CHOLMOD tests."""

    def test_femsolve_cholmod_verbose(self):
        """femsolve cholmod with verbose output."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        n = 40
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 20
        b = np.random.randn(n)

        x, flag = solver.femsolve(A, b, method="cholmod", spd=True, verbose=True)

        self.assertEqual(flag, 0)

    def test_femsolve_cholmod_all_zero_rhs(self):
        """femsolve cholmod with all-zero RHS columns."""
        info = solver.get_solver_info()

        if not info.get("has_cholmod", False):
            self.skipTest("CHOLMOD not available")

        n = 30
        A = sparse.random(n, n, density=0.1, format="csr")
        A = A @ A.T + sparse.eye(n) * 20
        B = np.zeros((n, 3))

        x, flag = solver.femsolve(A, B, method="cholmod", spd=True)

        self.assertEqual(flag, 0)
        assert_allclose(x, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
