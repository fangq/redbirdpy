"""
Extended unit tests for redbird __init__ module - improving coverage.

Run with: python -m unittest test_init_extended -v
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbird as rb
from redbird import forward, recon, utility, property as prop_module, solver

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


class TestModuleExports(unittest.TestCase):
    """Test that all expected functions are exported."""

    def test_forward_exports(self):
        """Check forward module exports."""
        self.assertTrue(hasattr(rb, "runforward"))
        self.assertTrue(hasattr(rb, "femlhs"))
        self.assertTrue(hasattr(rb, "femrhs"))
        self.assertTrue(hasattr(rb, "femgetdet"))
        self.assertTrue(hasattr(rb, "jac"))
        self.assertTrue(hasattr(rb, "jacchrome"))

    def test_recon_exports(self):
        """Check recon module exports."""
        self.assertTrue(hasattr(rb, "runrecon"))
        self.assertTrue(hasattr(rb, "reginv"))
        self.assertTrue(hasattr(rb, "reginvover"))
        self.assertTrue(hasattr(rb, "reginvunder"))
        self.assertTrue(hasattr(rb, "matreform"))
        self.assertTrue(hasattr(rb, "matflat"))
        self.assertTrue(hasattr(rb, "prior"))
        self.assertTrue(hasattr(rb, "syncprop"))

    def test_utility_exports(self):
        """Check utility module exports."""
        self.assertTrue(hasattr(rb, "meshprep"))
        self.assertTrue(hasattr(rb, "deldotdel"))
        self.assertTrue(hasattr(rb, "sdmap"))
        self.assertTrue(hasattr(rb, "getoptodes"))
        self.assertTrue(hasattr(rb, "getdistance"))
        self.assertTrue(hasattr(rb, "getltr"))
        self.assertTrue(hasattr(rb, "getreff"))
        self.assertTrue(hasattr(rb, "elem2node"))
        self.assertTrue(hasattr(rb, "addnoise"))
        self.assertTrue(hasattr(rb, "meshinterp"))

    def test_property_exports(self):
        """Check property module exports."""
        self.assertTrue(hasattr(rb, "extinction"))
        self.assertTrue(hasattr(rb, "updateprop"))
        self.assertTrue(hasattr(rb, "getbulk"))
        self.assertTrue(hasattr(rb, "musp2sasp"))
        self.assertTrue(hasattr(rb, "setmesh"))

    def test_solver_exports(self):
        """Check solver module exports."""
        self.assertTrue(hasattr(rb, "femsolve"))
        self.assertTrue(hasattr(rb, "get_solver_info"))


class TestRunFunction(unittest.TestCase):
    """Test the rb.run convenience function."""

    def setUp(self):
        self.cfg = create_test_cfg()
        self.cfg, self.sd = utility.meshprep(self.cfg)

    def test_run_forward_only(self):
        """rb.run with only cfg should run forward simulation."""
        result = rb.run(self.cfg)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        detval, phi = result
        self.assertTrue(np.all(np.isfinite(detval)))

    def test_run_reconstruction(self):
        """rb.run with recon_cfg and detphi0 should run reconstruction."""
        # Generate baseline data
        detphi0, _ = forward.runforward(self.cfg, sd=self.sd)
        detphi0 = detphi0 * 0.95  # Perturb

        recon_cfg = {
            "prop": self.cfg["prop"].copy(),
            "lambda": 0.1,
        }

        result = rb.run(
            self.cfg,
            recon_cfg=recon_cfg,
            detphi0=detphi0,
            sd=self.sd,
            maxiter=1,
            report=False,
        )

        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 3)

    def test_run_recon_requires_detphi0(self):
        """rb.run should raise error if recon_cfg given without detphi0."""
        recon_cfg = {"prop": self.cfg["prop"].copy()}

        with self.assertRaises(ValueError):
            rb.run(self.cfg, recon_cfg=recon_cfg)


class TestVersionInfo(unittest.TestCase):
    """Test version and metadata."""

    def test_version_exists(self):
        """Package should have version info."""
        self.assertTrue(hasattr(rb, "__version__"))
        self.assertIsInstance(rb.__version__, str)

    def test_author_exists(self):
        """Package should have author info."""
        self.assertTrue(hasattr(rb, "__author__"))


class TestSubmoduleAccess(unittest.TestCase):
    """Test submodule accessibility."""

    def test_forward_submodule(self):
        """forward submodule should be accessible."""
        self.assertIsNotNone(rb.forward)
        self.assertTrue(hasattr(rb.forward, "runforward"))

    def test_recon_submodule(self):
        """recon submodule should be accessible."""
        self.assertIsNotNone(rb.recon)
        self.assertTrue(hasattr(rb.recon, "runrecon"))

    def test_utility_submodule(self):
        """utility submodule should be accessible."""
        self.assertIsNotNone(rb.utility)
        self.assertTrue(hasattr(rb.utility, "meshprep"))

    def test_property_submodule(self):
        """property submodule should be accessible."""
        self.assertIsNotNone(rb.property)
        self.assertTrue(hasattr(rb.property, "extinction"))

    def test_solver_submodule(self):
        """solver submodule should be accessible."""
        self.assertIsNotNone(rb.solver)
        self.assertTrue(hasattr(rb.solver, "femsolve"))


class TestAllExports(unittest.TestCase):
    """Test __all__ exports."""

    def test_all_contains_run(self):
        """__all__ should contain 'run'."""
        self.assertIn("run", rb.__all__)

    def test_all_contains_submodules(self):
        """__all__ should contain submodule names."""
        for name in ["forward", "recon", "utility", "property", "solver"]:
            self.assertIn(name, rb.__all__)

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        for name in rb.__all__:
            self.assertTrue(hasattr(rb, name), f"'{name}' not found in redbird")


if __name__ == "__main__":
    unittest.main(verbosity=2)
