"""
Unit tests for redbird.analytical module.

Run with: python -m unittest test_analytical -v
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbird import analytical


class TestSphericalBesselFunctions(unittest.TestCase):
    """Test spherical Bessel functions."""

    def test_spbesselj_order0(self):
        """spbesselj(0, z) = sin(z)/z."""
        z = 1.0
        result = analytical.spbesselj(0, z)
        expected = np.sin(z) / z
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselj_order1(self):
        """Test spbesselj order 1."""
        z = 2.0
        result = analytical.spbesselj(1, z)
        expected = np.sin(z) / z**2 - np.cos(z) / z
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselj_array(self):
        """spbesselj should handle array input."""
        z = np.array([1.0, 2.0, 3.0])
        result = analytical.spbesselj(0, z)
        self.assertEqual(len(result), 3)

    def test_spbessely_order0(self):
        """spbessely(0, z) = -cos(z)/z."""
        z = 1.0
        result = analytical.spbessely(0, z)
        expected = -np.cos(z) / z
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbessely_order1(self):
        """Test spbessely order 1."""
        z = 2.0
        result = analytical.spbessely(1, z)
        expected = -np.cos(z) / z**2 - np.sin(z) / z
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselh_kind1(self):
        """spbesselh kind 1 = j + i*y."""
        z = 1.5
        n = 0
        result = analytical.spbesselh(n, 1, z)
        expected = analytical.spbesselj(n, z) + 1j * analytical.spbessely(n, z)
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselh_kind2(self):
        """spbesselh kind 2 = j - i*y."""
        z = 1.5
        n = 0
        result = analytical.spbesselh(n, 2, z)
        expected = analytical.spbesselj(n, z) - 1j * analytical.spbessely(n, z)
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselh_invalid_kind(self):
        """spbesselh should raise for invalid kind."""
        with self.assertRaises(ValueError):
            analytical.spbesselh(0, 3, 1.0)

    def test_spbesseljprime(self):
        """Test derivative of spherical Bessel j."""
        z = 2.0
        result = analytical.spbesseljprime(0, z)
        # j'_0(z) = -j_1(z)
        expected = -analytical.spbesselj(1, z)
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselyprime(self):
        """Test derivative of spherical Bessel y."""
        z = 2.0
        result = analytical.spbesselyprime(0, z)
        # y'_0(z) = -y_1(z)
        expected = -analytical.spbessely(1, z)
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselhprime_kind1(self):
        """Test derivative of spherical Hankel kind 1."""
        z = 2.0
        n = 0
        result = analytical.spbesselhprime(n, 1, z)
        expected = analytical.spbesseljprime(n, z) + 1j * analytical.spbesselyprime(
            n, z
        )
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselhprime_kind2(self):
        """Test derivative of spherical Hankel kind 2."""
        z = 2.0
        n = 0
        result = analytical.spbesselhprime(n, 2, z)
        expected = analytical.spbesseljprime(n, z) - 1j * analytical.spbesselyprime(
            n, z
        )
        self.assertAlmostEqual(result, expected, places=10)

    def test_spbesselhprime_invalid_kind(self):
        """spbesselhprime should raise for invalid kind."""
        with self.assertRaises(ValueError):
            analytical.spbesselhprime(0, 3, 1.0)


class TestSphericalHarmonics(unittest.TestCase):
    """Test spherical harmonic functions."""

    def test_spharmonic_l0_m0(self):
        """Y_0^0 = 1/sqrt(4*pi)."""
        theta, phi = np.pi / 4, np.pi / 3
        result = analytical.spharmonic(0, 0, theta, phi)
        expected = 1.0 / np.sqrt(4 * np.pi)
        self.assertAlmostEqual(np.abs(result), np.abs(expected), places=10)

    def test_spharmonic_l1_m0(self):
        """Test Y_1^0."""
        theta, phi = np.pi / 4, 0
        result = analytical.spharmonic(1, 0, theta, phi)
        # Y_1^0 = sqrt(3/(4*pi)) * cos(theta)
        expected = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
        self.assertAlmostEqual(np.real(result), expected, places=10)

    def test_spharmonic_negative_m(self):
        """Test spherical harmonic with negative m."""
        theta, phi = np.pi / 3, np.pi / 4
        result = analytical.spharmonic(2, -1, theta, phi)
        self.assertTrue(np.isfinite(result))

    def test_spharmonic_array(self):
        """spharmonic should handle array input."""
        theta = np.array([0, np.pi / 4, np.pi / 2])
        phi = np.array([0, np.pi / 4, np.pi / 2])
        result = analytical.spharmonic(1, 0, theta, phi)
        self.assertEqual(len(result), 3)


class TestInfiniteCW(unittest.TestCase):
    """Test CW solution for infinite medium."""

    def test_infinite_cw_basic(self):
        """infinite_cw should return positive fluence."""
        mua = 0.01
        musp = 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 0]])

        phi = analytical.infinite_cw(mua, musp, srcpos, detpos)

        self.assertTrue(np.all(phi > 0))

    def test_infinite_cw_decay(self):
        """Fluence should decay with distance."""
        mua = 0.01
        musp = 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[5, 0, 0], [10, 0, 0], [20, 0, 0]])

        phi = analytical.infinite_cw(mua, musp, srcpos, detpos)

        # Fluence should decrease with distance
        self.assertGreater(phi[0, 0], phi[1, 0])
        self.assertGreater(phi[1, 0], phi[2, 0])

    def test_infinite_cw_symmetry(self):
        """Fluence should be symmetric."""
        mua = 0.01
        musp = 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0]])

        phi = analytical.infinite_cw(mua, musp, srcpos, detpos)

        # All detectors are at same distance
        assert_allclose(phi[0], phi[1], rtol=1e-10)
        assert_allclose(phi[0], phi[2], rtol=1e-10)


class TestSemiInfiniteCW(unittest.TestCase):
    """Test CW solution for semi-infinite medium."""

    def test_semi_infinite_cw_basic(self):
        """semi_infinite_cw should return positive fluence inside medium."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 5]])  # Inside medium (z > 0)

        phi = analytical.semi_infinite_cw(mua, musp, n_in, n_out, srcpos, detpos)

        self.assertTrue(np.all(phi > 0))

    def test_semi_infinite_cw_boundary(self):
        """Fluence should be lower near boundary than infinite medium."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 5]])

        phi_semi = analytical.semi_infinite_cw(mua, musp, n_in, n_out, srcpos, detpos)
        phi_inf = analytical.infinite_cw(mua, musp, srcpos, detpos)

        # Semi-infinite should have lower fluence due to boundary loss
        self.assertLess(phi_semi.flatten()[0], phi_inf.flatten()[0])


class TestSemiInfiniteCWFlux(unittest.TestCase):
    """Test diffuse reflectance for semi-infinite medium."""

    def test_semi_infinite_cw_flux_positive(self):
        """Diffuse reflectance should be positive."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 0], [20, 0, 0]])

        flux = analytical.semi_infinite_cw_flux(mua, musp, n_in, n_out, srcpos, detpos)

        self.assertTrue(np.all(flux > 0))

    def test_semi_infinite_cw_flux_decay(self):
        """Reflectance should decay with distance."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[5, 0, 0], [10, 0, 0], [20, 0, 0]])

        flux = analytical.semi_infinite_cw_flux(mua, musp, n_in, n_out, srcpos, detpos)

        # flux is squeezed to 1D when single source, so use 1D indexing
        flux = np.atleast_1d(flux)  # Ensure it's at least 1D

        self.assertGreater(flux[0], flux[1])
        self.assertGreater(flux[1], flux[2])


class TestTimeDomain(unittest.TestCase):
    """Test time-domain solutions."""

    def test_infinite_td_basic(self):
        """infinite_td should return positive fluence."""
        mua = 0.01
        musp = 1.0
        n = 1.37
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 0]])
        t = np.array([1e-9, 2e-9, 5e-9])

        phi = analytical.infinite_td(mua, musp, n, srcpos, detpos, t)

        self.assertTrue(np.all(phi >= 0))

    def test_infinite_td_shape(self):
        """infinite_td should return (len(t), Ndet) array."""
        mua = 0.01
        musp = 1.0
        n = 1.37
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 0], [20, 0, 0]])
        t = np.array([1e-9, 2e-9, 5e-9])

        phi = analytical.infinite_td(mua, musp, n, srcpos, detpos, t)

        self.assertEqual(phi.shape, (3, 2))

    def test_semi_infinite_td_basic(self):
        """semi_infinite_td should return non-negative fluence."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 5]])
        t = np.array([1e-9, 2e-9, 5e-9])

        phi = analytical.semi_infinite_td(mua, musp, n_in, n_out, srcpos, detpos, t)

        self.assertTrue(np.all(phi >= 0))

    def test_semi_infinite_td_shape(self):
        """semi_infinite_td should return (len(t), Ndet) array."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 5], [20, 0, 5]])
        t = np.array([1e-9, 2e-9])

        phi = analytical.semi_infinite_td(mua, musp, n_in, n_out, srcpos, detpos, t)

        self.assertEqual(phi.shape, (2, 2))


class TestSphereInfinite(unittest.TestCase):
    """Test sphere in infinite medium."""

    def test_sphere_infinite_basic(self):
        """sphere_infinite should return finite fluence."""
        cfg = {
            "v": 299792458000 / 1.37,  # Speed of light in medium
            "a": 5.0,  # Sphere radius
            "omua": 0.01,  # Outside mua
            "omusp": 1.0,  # Outside musp
            "imua": 0.05,  # Inside mua
            "imusp": 1.0,  # Inside musp
            "src": [20.0, 0, 0],  # Source in spherical coords (r, theta, phi)
            "maxl": 5,  # Max order
            "omega": 0,  # CW
        }

        xrange = np.array([0, 10, 20])
        yrange = np.array([0])
        zrange = np.array([0])

        phi, xi, yi, zi = analytical.sphere_infinite(xrange, yrange, zrange, cfg)

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_sphere_infinite_shape(self):
        """sphere_infinite should return correct shape."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [20.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.array([-10, 0, 10])
        yrange = np.array([-5, 0, 5])
        zrange = np.array([0])

        phi, xi, yi, zi = analytical.sphere_infinite(xrange, yrange, zrange, cfg)

        self.assertEqual(phi.shape, (3, 3))


class TestSphereSemiInfinite(unittest.TestCase):
    """Test sphere in semi-infinite medium."""

    def test_sphere_semi_infinite_basic(self):
        """sphere_semi_infinite should return finite fluence."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [20.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.array([0, 10])
        yrange = np.array([0])
        zrange = np.array([0, 10])

        phi, xi, yi, zi = analytical.sphere_semi_infinite(xrange, yrange, zrange, cfg)

        self.assertTrue(np.all(np.isfinite(phi)))


class TestSphereSlab(unittest.TestCase):
    """Test sphere in slab geometry."""

    def test_sphere_slab_basic(self):
        """sphere_slab should return finite fluence."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [20.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.array([0, 10])
        yrange = np.array([0])
        zrange = np.array([5, 15])
        h = 40.0  # Slab thickness

        phi, xi, yi, zi = analytical.sphere_slab(xrange, yrange, zrange, cfg, h)

        self.assertTrue(np.all(np.isfinite(phi)))


class TestFrequencyDomain(unittest.TestCase):
    """Test frequency-domain solutions."""

    def test_sphere_infinite_fd(self):
        """sphere_infinite with omega > 0 should return complex values."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [20.0, 0, 0],
            "maxl": 3,
            "omega": 2 * np.pi * 100e6,  # 100 MHz
        }

        xrange = np.array([0, 10, 20])
        yrange = np.array([0])
        zrange = np.array([0])

        phi, xi, yi, zi = analytical.sphere_infinite(xrange, yrange, zrange, cfg)

        # FD solution should be complex
        self.assertTrue(np.iscomplexobj(phi))


class TestSphericalBesselExtended(unittest.TestCase):
    """Extended tests for spherical Bessel functions."""

    def test_spbesselj_higher_order(self):
        """spbesselj should work for higher orders."""
        for n in range(5):
            result = analytical.spbesselj(n, 2.0)
            self.assertTrue(np.isfinite(result))

    def test_spbessely_higher_order(self):
        """spbessely should work for higher orders."""
        for n in range(5):
            result = analytical.spbessely(n, 2.0)
            self.assertTrue(np.isfinite(result))

    def test_spbesselh_higher_order(self):
        """spbesselh should work for higher orders."""
        for n in range(5):
            for k in [1, 2]:
                result = analytical.spbesselh(n, k, 2.0)
                self.assertTrue(np.isfinite(result))

    def test_spbesseljprime_higher_order(self):
        """spbesseljprime should work for higher orders."""
        for n in range(5):
            result = analytical.spbesseljprime(n, 2.0)
            self.assertTrue(np.isfinite(result))

    def test_spbesselyprime_higher_order(self):
        """spbesselyprime should work for higher orders."""
        for n in range(5):
            result = analytical.spbesselyprime(n, 2.0)
            self.assertTrue(np.isfinite(result))


class TestSphericalHarmonicsExtended(unittest.TestCase):
    """Extended tests for spherical harmonics."""

    def test_spharmonic_l2_m_positive(self):
        """Test Y_2^m for positive m."""
        theta, phi = np.pi / 3, np.pi / 4
        for m in range(3):  # m = 0, 1, 2
            result = analytical.spharmonic(2, m, theta, phi)
            self.assertTrue(np.isfinite(result))

    def test_spharmonic_l2_m_negative(self):
        """Test Y_2^m for negative m."""
        theta, phi = np.pi / 3, np.pi / 4
        for m in [-2, -1]:
            result = analytical.spharmonic(2, m, theta, phi)
            self.assertTrue(np.isfinite(result))

    def test_spharmonic_theta_0(self):
        """spharmonic at theta=0 (north pole)."""
        result = analytical.spharmonic(1, 0, 0, 0)
        self.assertTrue(np.isfinite(result))

    def test_spharmonic_theta_pi(self):
        """spharmonic at theta=pi (south pole)."""
        result = analytical.spharmonic(1, 0, np.pi, 0)
        self.assertTrue(np.isfinite(result))


class TestInfiniteCWExtended(unittest.TestCase):
    """Extended tests for infinite CW solution."""

    def test_infinite_cw_multiple_sources(self):
        """infinite_cw should handle multiple sources."""
        mua = 0.01
        musp = 1.0
        srcpos = np.array([[0, 0, 0], [20, 0, 0]])
        detpos = np.array([[10, 0, 0]])

        phi = analytical.infinite_cw(mua, musp, srcpos, detpos)

        self.assertEqual(phi.shape, (1, 2))  # (num_detectors, num_sources)
        self.assertTrue(np.all(phi > 0))

    def test_infinite_cw_close_distance(self):
        """infinite_cw should handle close source-detector distance."""
        mua = 0.01
        musp = 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[1, 0, 0]])  # 1mm distance

        phi = analytical.infinite_cw(mua, musp, srcpos, detpos)

        self.assertTrue(np.all(phi > 0))
        self.assertTrue(np.all(np.isfinite(phi)))


class TestSemiInfiniteCWExtended(unittest.TestCase):
    """Extended tests for semi-infinite CW solution."""

    def test_semi_infinite_cw_multiple_sources(self):
        """semi_infinite_cw should handle multiple sources."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0], [20, 0, 0]])
        detpos = np.array([[10, 0, 5]])

        phi = analytical.semi_infinite_cw(mua, musp, n_in, n_out, srcpos, detpos)

        self.assertEqual(phi.shape, (2,))  # Returns 1D array for single detector
        self.assertTrue(np.all(phi > 0))

    def test_semi_infinite_cw_same_index(self):
        """semi_infinite_cw should handle matched refractive indices."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.37  # Matched
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 5]])

        phi = analytical.semi_infinite_cw(mua, musp, n_in, n_out, srcpos, detpos)

        self.assertTrue(np.all(np.isfinite(phi)))


class TestTimeDomainExtended(unittest.TestCase):
    """Extended tests for time-domain solutions."""

    def test_infinite_td_early_time(self):
        """infinite_td should handle very early times."""
        mua = 0.01
        musp = 1.0
        n = 1.37
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 0]])
        t = np.array([1e-12])  # Very early

        phi = analytical.infinite_td(mua, musp, n, srcpos, detpos, t)

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_infinite_td_late_time(self):
        """infinite_td should handle late times."""
        mua = 0.01
        musp = 1.0
        n = 1.37
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 0]])
        t = np.array([1e-6])  # Late time

        phi = analytical.infinite_td(mua, musp, n, srcpos, detpos, t)

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_semi_infinite_td_multiple_detectors(self):
        """semi_infinite_td should handle multiple detectors."""
        mua = 0.01
        musp = 1.0
        n_in, n_out = 1.37, 1.0
        srcpos = np.array([[0, 0, 0]])
        detpos = np.array([[10, 0, 5], [20, 0, 5], [30, 0, 5]])
        t = np.array([1e-9, 2e-9])

        phi = analytical.semi_infinite_td(mua, musp, n_in, n_out, srcpos, detpos, t)

        self.assertEqual(phi.shape, (2, 3))


class TestSphereInfiniteExtended(unittest.TestCase):
    """Extended tests for sphere in infinite medium."""

    def test_sphere_infinite_2d_grid(self):
        """sphere_infinite should handle 2D grids."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [15.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.linspace(-10, 10, 5)
        yrange = np.linspace(-10, 10, 5)
        zrange = np.array([0])

        phi, xi, yi, zi = analytical.sphere_infinite(xrange, yrange, zrange, cfg)

        self.assertEqual(phi.shape, (5, 5))

    def test_sphere_infinite_inside_sphere(self):
        """sphere_infinite should compute field inside sphere."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 10.0,  # Large sphere
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [20.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        # Points inside the sphere
        xrange = np.array([0, 5])
        yrange = np.array([0])
        zrange = np.array([0])

        phi, xi, yi, zi = analytical.sphere_infinite(xrange, yrange, zrange, cfg)

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_sphere_infinite_default_maxl(self):
        """sphere_infinite should use default maxl."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [15.0, 0, 0],
            # 'maxl' not specified - should default to 20
        }

        xrange = np.array([0, 10])
        yrange = np.array([0])
        zrange = np.array([0])

        phi, xi, yi, zi = analytical.sphere_infinite(xrange, yrange, zrange, cfg)

        self.assertTrue(np.all(np.isfinite(phi)))


class TestSphereSemiInfiniteExtended(unittest.TestCase):
    """Extended tests for sphere in semi-infinite medium."""

    def test_sphere_semi_infinite_custom_n(self):
        """sphere_semi_infinite should accept custom refractive indices."""
        cfg = {
            "v": 299792458000 / 1.4,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [15.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.array([0, 10])
        yrange = np.array([0])
        zrange = np.array([5, 10])

        phi, xi, yi, zi = analytical.sphere_semi_infinite(
            xrange, yrange, zrange, cfg, n0=1.0, n1=1.4
        )

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_sphere_semi_infinite_default_n(self):
        """sphere_semi_infinite should use default n1=1.37."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [15.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.array([0, 10])
        yrange = np.array([0])
        zrange = np.array([5])

        phi, xi, yi, zi = analytical.sphere_semi_infinite(xrange, yrange, zrange, cfg)

        self.assertTrue(np.all(np.isfinite(phi)))


class TestSphereSlabExtended(unittest.TestCase):
    """Extended tests for sphere in slab geometry."""

    def test_sphere_slab_thin_slab(self):
        """sphere_slab should handle thin slabs."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 3.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [10.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.array([5, 10])
        yrange = np.array([0])
        zrange = np.array([5, 10])
        h = 20.0  # Thin slab

        phi, xi, yi, zi = analytical.sphere_slab(xrange, yrange, zrange, cfg, h)

        self.assertTrue(np.all(np.isfinite(phi)))

    def test_sphere_slab_thick_slab(self):
        """sphere_slab should handle thick slabs."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [30.0, 0, 0],
            "maxl": 3,
            "omega": 0,
        }

        xrange = np.array([10, 20])
        yrange = np.array([0])
        zrange = np.array([20, 30])
        h = 100.0  # Thick slab

        phi, xi, yi, zi = analytical.sphere_slab(xrange, yrange, zrange, cfg, h)

        self.assertTrue(np.all(np.isfinite(phi)))


class TestFrequencyDomainExtended(unittest.TestCase):
    """Extended tests for frequency-domain solutions."""

    def test_sphere_semi_infinite_fd(self):
        """sphere_semi_infinite with omega > 0."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [15.0, 0, 0],
            "maxl": 3,
            "omega": 2 * np.pi * 100e6,
        }

        xrange = np.array([0, 10])
        yrange = np.array([0])
        zrange = np.array([5])

        phi, xi, yi, zi = analytical.sphere_semi_infinite(xrange, yrange, zrange, cfg)

        self.assertTrue(np.iscomplexobj(phi))

    def test_sphere_slab_fd(self):
        """sphere_slab with omega > 0."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [20.0, 0, 0],
            "maxl": 3,
            "omega": 2 * np.pi * 50e6,
        }

        xrange = np.array([5, 15])
        yrange = np.array([0])
        zrange = np.array([10, 20])
        h = 50.0

        phi, xi, yi, zi = analytical.sphere_slab(xrange, yrange, zrange, cfg, h)

        self.assertTrue(np.iscomplexobj(phi))


class TestSphereCoefficients(unittest.TestCase):
    """Test internal sphere coefficient functions."""

    def test_sphere_coeff_A_nonzero_m(self):
        """_sphere_coeff_A should return 0 for m!=0 at poles."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [15.0, 0, 0],  # theta=0 (pole)
            "maxl": 3,
            "omega": 0,
        }
        cfg = analytical._init_sphere_cfg(cfg)

        # m != 0 at pole should give 0
        A = analytical._sphere_coeff_A(1, 1, cfg)
        self.assertEqual(A, 0.0)

    def test_sphere_coeff_C_nonzero_m(self):
        """_sphere_coeff_C should return 0 for m!=0 at poles."""
        cfg = {
            "v": 299792458000 / 1.37,
            "a": 5.0,
            "omua": 0.01,
            "omusp": 1.0,
            "imua": 0.05,
            "imusp": 1.0,
            "src": [15.0, np.pi, 0],  # theta=pi (south pole)
            "maxl": 3,
            "omega": 0,
        }
        cfg = analytical._init_sphere_cfg(cfg)

        # m != 0 at pole should give 0
        C = analytical._sphere_coeff_C(1, 1, cfg)
        self.assertEqual(C, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
