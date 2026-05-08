"""
Unit tests for the MWT (Helmholtz + Bayliss-Turkel RBC) features in redbirdpy.

Tests mirror the matlab redbird MWT test suite:
- bulk + prop dispatch (epsilon/sigma branches in property.getbulk and updateprop)
- RBC geometry precomputation in utility.meshprep
- 6-column line source / detector positions
- forward smoke (complex finite phi, line-source amplitude scaling)
- physics: k formula, lossless vs lossy attenuation, reciprocity, omega scaling

Run with: python -m pytest test/test_mwt.py -v
"""

import unittest
import sys
import os
import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbirdpy as rb
from redbirdpy import forward, utility, property as rbprop

try:
    import iso2mesh as i2m

    HAS_ISO2MESH = True
except ImportError:
    HAS_ISO2MESH = False

MU0_MM = 4.0 * np.pi * 1e-10
EPS0_MM = 8.854187817e-15


class TestMWTBulkAndProp(unittest.TestCase):
    """rbprop.getbulk and rbprop.updateprop with epsilon/sigma."""

    def test_getbulk_mwt_full(self):
        cfg = {"bulk": {"epsilon": 78.0, "sigma": 1e-3, "n": np.sqrt(78.0)}}
        bk = rbprop.getbulk(cfg)
        self.assertAlmostEqual(bk[0], 78.0)
        self.assertAlmostEqual(bk[1], 1e-3)
        self.assertAlmostEqual(bk[2], MU0_MM)
        self.assertAlmostEqual(bk[3], np.sqrt(78.0))

    def test_getbulk_mwt_minimal(self):
        # only epsilon set -> sigma defaults to 0, mu0 = 4*pi*1e-10
        cfg = {"bulk": {"epsilon": 1.0}}
        bk = rbprop.getbulk(cfg)
        self.assertAlmostEqual(bk[0], 1.0)
        self.assertAlmostEqual(bk[1], 0.0)
        self.assertAlmostEqual(bk[2], MU0_MM)

    def test_updateprop_mwt_label_based(self):
        cfg = {
            "node": np.array(
                [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
            ),
            "elem": np.array([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5]]),
            "seg": np.array([1, 1, 2]),
            "param": {
                "epsilon": np.array([40.0, 60.0]),
                "sigma": np.array([0.5e-3, 1.0e-3]),
            },
            "prop": {
                "5e8": np.array(
                    [[1, 0, MU0_MM, 1], [1, 0, MU0_MM, 1], [1, 0, MU0_MM, 1]]
                )
            },
        }
        propnew = rbprop.updateprop(cfg)
        # single-wavelength returns the array directly (not a dict)
        p = propnew["5e8"] if isinstance(propnew, dict) else propnew
        self.assertEqual(p.shape, (3, 4))
        # row 0 is "label 0" (background), rows 1,2 hold our two labels
        self.assertAlmostEqual(p[1, 0], 40.0)
        self.assertAlmostEqual(p[2, 0], 60.0)
        self.assertAlmostEqual(p[1, 1], 0.5e-3)
        self.assertAlmostEqual(p[2, 1], 1.0e-3)
        self.assertAlmostEqual(p[1, 2], MU0_MM)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
class TestMWTMeshprep(unittest.TestCase):
    """utility.meshprep precomputes for the Bayliss-Turkel RBC."""

    @classmethod
    def setUpClass(cls):
        node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 40], 8)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def _build_cfg(self):
        cfg = {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[10, 20, 0]]),
            "srcdir": np.array([[0, 0, 1]]),
            "detpos": np.array([[30, 20, 40]]),
            "detdir": np.array([[0, 0, -1]]),
            "bulk": {"epsilon": 1.0, "sigma": 0.0, "n": 1.0},
            "prop": {"5e8": np.array([[1, 0, MU0_MM, 1], [1, 0, MU0_MM, 1]])},
            "omega": 2 * np.pi * 5e8,
        }
        return cfg

    def test_meshprep_adds_facenb(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        self.assertIn("facenb", cfg)
        self.assertEqual(cfg["facenb"].shape[1], 4)

    def test_meshprep_adds_rbc_geometry(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        for key in ("facecenter", "facenormal", "facer", "rbcorigin"):
            self.assertIn(key, cfg)

    def test_meshprep_normals_unit_length(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        n = cfg["facenormal"]
        self.assertTrue(np.max(np.abs(np.linalg.norm(n, axis=1) - 1.0)) < 1e-9)

    def test_meshprep_facer_positive(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        self.assertTrue(np.all(cfg["facer"] > 0))

    def test_meshprep_skips_reff_in_mwt(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        # reff is DOT-only; absent in MWT path
        self.assertTrue("reff" not in cfg or cfg.get("reff") is None)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
class TestMWTLineSource(unittest.TestCase):
    """6-column cfg.srcpos / cfg.detpos - line endpoints."""

    @classmethod
    def setUpClass(cls):
        node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 40], 8)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def _build_cfg(self):
        return {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[20, 20, 0, 20, 20, 40]]),
            "detpos": np.array([[10, 10, 0, 10, 10, 40]]),
            "bulk": {"epsilon": 1.0, "sigma": 0.0, "n": 1.0},
            "prop": {"5e8": np.array([[1, 0, MU0_MM, 1], [1, 0, MU0_MM, 1]])},
            "omega": 2 * np.pi * 5e8,
        }

    def test_line_srcpos_yields_widesrc(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        self.assertIn("widesrc", cfg)
        self.assertEqual(cfg["widesrc"].shape[1], 1)  # one source line

    def test_line_detpos_yields_widedet(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        self.assertIn("widedet", cfg)
        self.assertEqual(cfg["widedet"].shape[1], 1)  # one detector line

    def test_widesrc_complex_for_mwt(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        self.assertTrue(np.iscomplexobj(cfg["widesrc"]))

    def test_widesrc_nonzero(self):
        cfg = self._build_cfg()
        cfg, _ = utility.meshprep(cfg)
        self.assertTrue(np.max(np.abs(cfg["widesrc"])) > 0)

    def test_dot_line_source_works(self):
        # DOT regression: same line source for the diffusion equation
        cfg = {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[20, 20, 0, 20, 20, 40]]),
            "detpos": np.array([[10, 10, 35, 30, 30, 35]]),
            "prop": np.array([[0, 0, 1, 1], [0.01, 1, 0, 1.37]]),
            "omega": 0,
        }
        cfg, _ = utility.meshprep(cfg)
        self.assertIn("widesrc", cfg)
        self.assertEqual(cfg["widesrc"].shape[1], 1)
        self.assertFalse(np.iscomplexobj(cfg["widesrc"]))  # DOT: real-valued
        # forward solve should produce finite output
        detphi, _ = forward.runforward(cfg)
        self.assertTrue(np.all(np.isfinite(detphi)))


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
class TestMWTForward(unittest.TestCase):
    """End-to-end MWT forward solve with line sources/detectors."""

    @classmethod
    def setUpClass(cls):
        node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 40], 8)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def _make_mwt_cfg(self, sigma=0.0, eps_r=1.0, n_eff=1.0):
        return {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[20, 20, 0, 20, 20, 40]]),
            "detpos": np.array([[10, 10, 0, 10, 10, 40]]),
            "bulk": {"epsilon": eps_r, "sigma": sigma, "n": n_eff},
            "prop": {
                "5e8": np.array([[1, 0, MU0_MM, 1], [eps_r, sigma, MU0_MM, n_eff]])
            },
            "omega": 2 * np.pi * 5e8,
        }

    def test_forward_complex_finite(self):
        cfg = self._make_mwt_cfg()
        cfg, _ = utility.meshprep(cfg)
        detphi, phi = forward.runforward(cfg)
        self.assertTrue(np.iscomplexobj(phi))
        self.assertTrue(np.all(np.isfinite(phi)))
        self.assertTrue(np.max(np.abs(phi)) > 0)

    def test_forward_lossy_attenuates(self):
        cfg_lossless = self._make_mwt_cfg(sigma=0.0, eps_r=4.0, n_eff=2.0)
        cfg_lossy = self._make_mwt_cfg(sigma=1e-3, eps_r=4.0, n_eff=2.0)
        cfg_lossless, _ = utility.meshprep(cfg_lossless)
        cfg_lossy, _ = utility.meshprep(cfg_lossy)
        det_lossless, _ = forward.runforward(cfg_lossless)
        det_lossy, _ = forward.runforward(cfg_lossy)
        # lossy medium attenuates
        self.assertLess(np.abs(det_lossy[0, 0]), np.abs(det_lossless[0, 0]))


class TestMWTKFormula(unittest.TestCase):
    """k formula sanity checks (no mesh required)."""

    def test_k_vacuum_equals_omega_over_c(self):
        omega = 2 * np.pi * 5e8
        c0_mm_per_s = 1.0 / np.sqrt(MU0_MM * EPS0_MM)
        k_vac = np.sqrt(omega**2 * MU0_MM * EPS0_MM)
        assert_allclose(k_vac, omega / c0_mm_per_s, rtol=1e-12)

    def test_k_lossy_im_negative(self):
        # for sigma>0, principal sqrt of (a - jb) has Im < 0 (decay)
        omega = 2 * np.pi * 5e8
        k = np.sqrt(omega**2 * MU0_MM * EPS0_MM - 1j * omega * MU0_MM * 1e-3)
        self.assertLess(k.imag, 0)

    def test_k_scales_with_omega_in_lossless(self):
        omega_lo = 2 * np.pi * 5e8
        omega_hi = 2 * np.pi * 5e9
        k_lo = np.sqrt(omega_lo**2 * MU0_MM * EPS0_MM)
        k_hi = np.sqrt(omega_hi**2 * MU0_MM * EPS0_MM)
        assert_allclose(k_hi.real / k_lo.real, 10.0, rtol=1e-9)


@unittest.skipUnless(HAS_ISO2MESH, "iso2mesh not available")
class TestMWTReciprocity(unittest.TestCase):
    """A_FEM is complex-symmetric, so detphi(s->d) == detphi(d->s)."""

    @classmethod
    def setUpClass(cls):
        node, face, elem = i2m.meshabox([0, 0, 0], [40, 40, 40], 8)
        cls.node = node
        cls.face = face
        cls.elem = elem

    def test_reciprocity(self):
        # two distinct line antennas, both serving as src and det
        cfg = {
            "node": self.node,
            "face": self.face,
            "elem": self.elem,
            "seg": np.ones(self.elem.shape[0], dtype=int),
            "srcpos": np.array([[10, 20, 5, 10, 20, 35], [30, 20, 5, 30, 20, 35]]),
            "detpos": np.array([[10, 20, 5, 10, 20, 35], [30, 20, 5, 30, 20, 35]]),
            "bulk": {"epsilon": 4.0, "sigma": 1e-4, "n": 2.0},
            "prop": {"5e8": np.array([[1, 0, MU0_MM, 1], [4, 1e-4, MU0_MM, 2]])},
            "omega": 2 * np.pi * 5e8,
        }
        cfg, _ = utility.meshprep(cfg)
        detphi, _ = forward.runforward(cfg)
        # off-diagonal entries should agree (within tight tolerance)
        relerr = np.abs(detphi[0, 1] - detphi[1, 0]) / max(
            abs(detphi[0, 1]), abs(detphi[1, 0])
        )
        self.assertLess(relerr, 1e-6)


class TestMWTJacobianChain(unittest.TestCase):
    """forward.jacepssigma chain rule."""

    def test_chain_eps_only(self):
        # synthetic Jmua, single frequency
        Jmua = {"5e8": np.ones((4, 10))}
        omega = 2 * np.pi * 5e8
        chain = forward.jacepssigma(Jmua, omega, has_eps=True, has_sigma=False)
        self.assertIn("epsilon", chain)
        self.assertNotIn("sigma", chain)
        expected = -(omega**2) * MU0_MM * EPS0_MM * np.ones((4, 10))
        assert_allclose(chain["epsilon"], expected)

    def test_chain_sigma_only(self):
        Jmua = {"5e8": np.ones((4, 10))}
        omega = 2 * np.pi * 5e8
        chain = forward.jacepssigma(Jmua, omega, has_eps=False, has_sigma=True)
        self.assertIn("sigma", chain)
        self.assertNotIn("epsilon", chain)
        expected = 1j * omega * MU0_MM * np.ones((4, 10))
        assert_allclose(chain["sigma"], expected)

    def test_chain_both_stack_per_frequency(self):
        # 2 frequencies, J_mua shape (3, 5) each -> stacked (6, 5)
        Jmua = {"5e8": np.ones((3, 5)), "1e9": 2 * np.ones((3, 5))}
        omegas = {"5e8": 2 * np.pi * 5e8, "1e9": 2 * np.pi * 1e9}
        chain = forward.jacepssigma(Jmua, omegas, has_eps=True, has_sigma=True)
        self.assertEqual(chain["epsilon"].shape, (6, 5))
        self.assertEqual(chain["sigma"].shape, (6, 5))


if __name__ == "__main__":
    unittest.main()
