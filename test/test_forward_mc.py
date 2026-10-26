"""
Tests of the Monte Carlo forward path of redbird.forward.runforward.

Setting cfg["nphoton"] routes runforward to pmmc (mesh-based MC). These tests
run pmmc on the CPU (compute="sse", no GPU needed) and are skipped when pmmc
or iso2mesh is not installed (pip install redbirdpy[mc]).

Run with: python -m unittest test_forward_mc -v
"""

import os
import sys
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbirdpy import forward, utility  # noqa: E402

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False

try:
    import pmmc  # noqa: F401

    HAS_PMMC = True
except ImportError:
    HAS_PMMC = False


def base_cfg():
    node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 20], 4)
    return {
        "node": node,
        "elem": elem,
        "seg": np.ones(elem.shape[0], dtype=int),
        "srcpos": np.array([[20, 20, 0]], dtype=float),
        "srcdir": np.array([[0, 0, 1]], dtype=float),
        "detpos": np.array([[30, 20, 0], [35, 20, 0]], dtype=float),
        "detdir": np.array([[0, 0, 1], [0, 0, 1]], dtype=float),
        "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]], dtype=float),
        "omega": 0,
    }


@unittest.skipUnless(HAS_ISO2MESH and HAS_PMMC, "requires iso2mesh and pmmc")
class TestForwardMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, _ = utility.meshprep(base_cfg())
        cls.det_fem = np.asarray(forward.runforward(cfg)[0]).ravel()

        cfgmc = base_cfg()
        cfgmc["nphoton"] = int(2e5)
        cfgmc["compute"] = "sse"
        cfgmc["seed"] = 12345
        cls.det_mc, cls.phi_mc = forward.runforward(cfgmc)
        cls.det_mc = np.asarray(cls.det_mc).ravel()

    def test_detphi_finite_positive(self):
        self.assertEqual(self.det_mc.size, 2)
        self.assertTrue(np.all(np.isfinite(self.det_mc)))
        self.assertTrue(np.all(self.det_mc > 0))

    def test_decay_matches_fem(self):
        # MC and the diffusion FEM solver must agree on how the detected
        # fluence falls off with the source-detector separation
        ratio = (self.det_mc[1] / self.det_mc[0]) / (self.det_fem[1] / self.det_fem[0])
        self.assertLess(abs(ratio - 1), 0.3)

    def test_phi_covers_nodes(self):
        self.assertEqual(np.asarray(self.phi_mc).shape[0], base_cfg()["node"].shape[0])


def has_opencl():
    """True if pmmc can run on its OpenCL backend (device 1), needed by the adjoint Jacobian"""
    if not (HAS_ISO2MESH and HAS_PMMC):
        return False
    try:
        cfg = base_cfg()
        cfg.update({"nphoton": 100, "compute": "opencl", "gpuid": 1})
        forward.runforward(cfg)
        return True
    except Exception:
        return False


@unittest.skipUnless(has_opencl(), "requires iso2mesh and pmmc with an OpenCL device")
class TestForwardMCJacobianSD(unittest.TestCase):
    def test_sd_selects_jacobian_rows(self):
        # with an sd table, the MC Jacobian holds the active sd rows only, in sd
        # order, as jac() does; without it, all pairs in detphi.ravel() order
        cfg = base_cfg()
        cfg["srcpos"] = np.array([[20, 20, 0], [15, 20, 0]], dtype=float)
        cfg["srcdir"] = np.array([[0, 0, 1], [0, 0, 1]], dtype=float)
        cfg.update({"nphoton": int(3e5), "compute": "opencl", "gpuid": 1, "seed": 1})
        sd = utility.sdmap(cfg)
        sd[[1, 2], 2] = 0  # deactivate two of the four channels
        full = forward.runforward(cfg, return_jacobian=True)[2]["mua"]
        sub = forward.runforward(cfg, return_jacobian=True, sd=sd)[2]["mua"]
        rows = np.flatnonzero(sd[:, 2] == 1)
        self.assertEqual(full.shape[-1], 4)
        self.assertEqual(sub.shape[-1], rows.size)
        for c, k in enumerate(rows):
            # the two runs differ only by GPU run-to-run noise
            err = np.linalg.norm(full[:, k] - sub[:, c]) / np.linalg.norm(full[:, k])
            self.assertLess(err, 0.05)


if __name__ == "__main__":
    unittest.main()
