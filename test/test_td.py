"""
Unit tests for time-domain DOT (Crank-Nicolson) features in redbirdpy.

Tests mirror the matlab redbird TD test suite:
- forward.femlhs mode=3 returns the pure consistent mass matrix
- TD forward smoke (impulse + custom srctemporal)
- conflict guards: TD + omega>0 -> error, TD + MWT -> error
- integration invariant: int_0^T phi_TD(t) dt ~ phi_CW for impulse IC
- peak in interior of time window

Run with: python -m pytest test/test_td.py -v
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbirdpy import forward, utility

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False

MU0_MM = 4.0 * np.pi * 1e-10


class TestFemlhsMassOnly(unittest.TestCase):
    """femlhs mode=3 returns the pure consistent mass matrix M."""

    @classmethod
    def setUpClass(cls):
        cls.node = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cls.elem = np.array([[1, 2, 3, 4]])
        cls.face = np.array([[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]])

    def _make_cfg(self):
        cfg = {
            "node": self.node,
            "elem": self.elem,
            "face": self.face,
            "seg": np.array([1]),
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
        }
        if HAS_ISO2MESH:
            cfg["evol"] = i2m.elemvolume(self.node, self.elem)
            cfg["area"] = i2m.elemvolume(self.node, self.face)
        else:
            v1 = self.node[1] - self.node[0]
            v2 = self.node[2] - self.node[0]
            v3 = self.node[3] - self.node[0]
            cfg["evol"] = np.array([abs(np.dot(np.cross(v1, v2), v3)) / 6.0])
            # area triangles
            cfg["area"] = np.array([0.5] * 4)
        cfg["deldotdel"], _ = utility.deldotdel(cfg)
        return cfg

    def test_mass_diag_is_V_over_10(self):
        cfg = self._make_cfg()
        M = forward.femlhs(cfg, cfg["deldotdel"], "", 3)
        v = cfg["evol"][0]
        self.assertAlmostEqual(M[0, 0], v / 10, delta=1e-15)

    def test_mass_offdiag_is_V_over_20(self):
        cfg = self._make_cfg()
        M = forward.femlhs(cfg, cfg["deldotdel"], "", 3)
        v = cfg["evol"][0]
        self.assertAlmostEqual(M[0, 1], v / 20, delta=1e-15)

    def test_mass_symmetric(self):
        cfg = self._make_cfg()
        M = forward.femlhs(cfg, cfg["deldotdel"], "", 3).toarray()
        self.assertTrue(np.allclose(M, M.T, atol=1e-15))

    def test_mass_row_sum_is_V_over_4(self):
        cfg = self._make_cfg()
        M = forward.femlhs(cfg, cfg["deldotdel"], "", 3).toarray()
        v = cfg["evol"][0]
        self.assertTrue(np.allclose(M.sum(axis=1), v / 4, atol=1e-15))


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
class TestTDForwardImpulse(unittest.TestCase):
    """End-to-end TD forward solve with the default impulse source."""

    @classmethod
    def setUpClass(cls):
        node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 40], 8)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def _make_cfg(self):
        return {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[20, 20, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[20, 20, 40]]),
            "detdir": np.array([[0, 0, -1]]),
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "omega": 0,
            "tstart": 0,
            "tstep": 50e-12,
            "tend": 2e-9,
        }

    def test_detphi_3d_shape(self):
        cfg = self._make_cfg()
        cfg, _ = utility.meshprep(cfg)
        detphi, phi = forward.runforward(cfg)
        self.assertEqual(detphi.shape, (1, 1, 41))
        self.assertIsNone(phi)  # tdsavevol=False default

    def test_detphi_finite_and_nonzero(self):
        cfg = self._make_cfg()
        cfg, _ = utility.meshprep(cfg)
        detphi, _ = forward.runforward(cfg)
        self.assertTrue(np.all(np.isfinite(detphi)))
        self.assertGreater(np.max(np.abs(detphi)), 0)

    def test_detphi_tail_nonneg(self):
        # past the peak, the diffusing tail is monotonically non-negative;
        # early-time CN ringing on a coarse mesh with impulse IC is a known
        # numerical artifact (use a smooth srctemporal or finer mesh in
        # production).
        cfg = self._make_cfg()
        cfg, _ = utility.meshprep(cfg)
        detphi, _ = forward.runforward(cfg)
        peak_t = np.argmax(np.abs(detphi[0, 0, :]))
        self.assertTrue(np.all(detphi[0, 0, peak_t:] >= 0))

    def test_detphi_peak_interior(self):
        cfg = self._make_cfg()
        cfg, _ = utility.meshprep(cfg)
        detphi, _ = forward.runforward(cfg)
        peak_t = np.argmax(np.abs(detphi[0, 0, :]))
        self.assertGreater(peak_t, 0)
        self.assertLess(peak_t, detphi.shape[2] - 1)

    def test_tdsavevol_returns_volumetric(self):
        cfg = self._make_cfg()
        cfg, _ = utility.meshprep(cfg)
        detphi, phi = forward.runforward(cfg, tdsavevol=True)
        self.assertEqual(phi.shape, (cfg["node"].shape[0], 1, 41))


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
class TestTDForwardCustomTemporal(unittest.TestCase):
    """TD forward with custom temporal modulation."""

    @classmethod
    def setUpClass(cls):
        node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 40], 8)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def _make_cfg(self):
        return {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[20, 20, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[20, 20, 40]]),
            "detdir": np.array([[0, 0, -1]]),
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "omega": 0,
            "tstart": 0,
            "tstep": 50e-12,
            "tend": 2e-9,
        }

    def test_constant_temporal_grows(self):
        # constant unit source from t=0 starting from phi=0 grows monotonically
        cfg = self._make_cfg()
        cfg, _ = utility.meshprep(cfg)
        nt = len(
            np.arange(cfg["tstart"], cfg["tend"] + 0.5 * cfg["tstep"], cfg["tstep"])
        )
        detphi, _ = forward.runforward(cfg, srctemporal=np.ones(nt))
        # detphi at last step > detphi at t=tstep+ (early)
        self.assertGreater(detphi[0, 0, -1], detphi[0, 0, 1])

    def test_callable_temporal(self):
        # callable temporal modulation: gaussian pulse
        cfg = self._make_cfg()
        cfg, _ = utility.meshprep(cfg)
        gauss = lambda t: np.exp(-(((t - 5e-10) / 1e-10) ** 2))
        detphi, _ = forward.runforward(cfg, srctemporal=gauss)
        self.assertTrue(np.all(np.isfinite(detphi)))


class TestTDConflictGuards(unittest.TestCase):
    """meshprep rejects mixed TD/FD and TD/MWT configurations."""

    @classmethod
    def setUpClass(cls):
        if not HAS_ISO2MESH:
            return
        node, face, elem = i2m.meshabox([0, 0, 0], [20, 20, 20], 5)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def _base_cfg(self):
        return {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[10, 10, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[10, 10, 20]]),
            "detdir": np.array([[0, 0, -1]]),
            "tstart": 0,
            "tstep": 50e-12,
            "tend": 2e-9,
        }

    @unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
    def test_td_with_omega_raises(self):
        cfg = self._base_cfg()
        cfg["prop"] = np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]])
        cfg["omega"] = 2 * np.pi * 1e8  # 100 MHz - conflicts with TD
        with self.assertRaises(ValueError):
            utility.meshprep(cfg)

    @unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
    def test_td_with_mwt_raises(self):
        cfg = self._base_cfg()
        cfg["prop"] = {"5e8": np.array([[1, 0, MU0_MM, 1], [4, 1e-3, MU0_MM, 2]])}
        cfg["bulk"] = {"epsilon": 4.0, "sigma": 1e-3, "n": 2.0}
        cfg["omega"] = 0
        with self.assertRaises(ValueError):
            utility.meshprep(cfg)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
class TestTDIntegrationInvariant(unittest.TestCase):
    """The impulse response integrated over time equals the CW solution."""

    @classmethod
    def setUpClass(cls):
        node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 40], 8)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def test_integration_invariant(self):
        # impulse default; the time integral of detphi should equal CW detphi
        # to within ~5% on a coarse mesh
        cfg = {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[20, 20, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[20, 20, 40]]),
            "detdir": np.array([[0, 0, -1]]),
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "omega": 0,
            "tstart": 0,
            "tstep": 50e-12,
            "tend": 1e-8,  # 10 ns
        }
        cfg, _ = utility.meshprep(cfg)
        detphi_imp, _ = forward.runforward(cfg)

        # trapezoidal integral over time
        integ = (
            np.sum(detphi_imp, axis=2)
            - 0.5 * (detphi_imp[:, :, 0] + detphi_imp[:, :, -1])
        ) * cfg["tstep"]

        # CW solution on the same geometry
        cfg_cw = {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[20, 20, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[20, 20, 40]]),
            "detdir": np.array([[0, 0, -1]]),
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "omega": 0,
        }
        cfg_cw, _ = utility.meshprep(cfg_cw)
        detphi_cw, _ = forward.runforward(cfg_cw)

        relerr = np.abs(integ.flatten()[0] - detphi_cw.flatten()[0]) / np.abs(
            detphi_cw.flatten()[0]
        )
        self.assertLess(relerr, 0.05)


if __name__ == "__main__":
    unittest.main()
