"""
Regression tests for meshprep's element-volume sign.

cfg['evol'] scales the FEM volume integrals and cfg['deldotdel'], so a negative
evol negates the whole stiffness matrix and the forward solve returns a
sign-flipped, oscillatory field instead of a fluence -- with nothing raising an
error. rbmeshprep.m computes cfg.evol = elemvolume(node, elem), i.e. UNSIGNED, so
meshprep must too.

The trap: iso2mesh's meshreorient returns the SIGNED volume in iso2mesh's own
convention (negative for a right-handed tet), measured BEFORE its reorienting
swap -- so it neither matches the standard convention nor describes the elements
meshreorient returns. Both the MATLAB and Python iso2mesh ports agree on this, so
it is a convention to absorb here, not an iso2mesh bug.

Run with: python -m unittest test_meshprep_evol -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbirdpy.utility import meshprep


def _unit_tet():
    """A single canonical right-handed tet: det([b-a; c-a; d-a])/6 = +1/6."""
    node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elem = np.array([[1, 2, 3, 4, 1]], dtype=int)  # 1-based, with a label column
    return node, elem


def _lattice(nx, ny, nz, h=2.0):
    """Regular grid split into 6 tets per cube (Kuhn/Freudenthal).

    Every path 000 -> 111 through the cube gives one tet, which makes the
    decomposition exact and conforming across neighbouring cubes.
    """
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]
    node = np.c_[X.ravel(), Y.ravel(), Z.ravel()].astype(float) * h

    def idx(i, j, k):
        return (i * ny + j) * nz + k + 1  # 1-based

    tets = [[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7],
            [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]]
    els = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [idx(i + (o >> 2 & 1), j + (o >> 1 & 1), k + (o & 1)) for o in range(8)]
                for t in tets:
                    els.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]], 1])
    elem = np.array(els, dtype=int)
    # The six corner-to-corner paths tile the cube exactly, but their handedness
    # alternates with permutation parity, so normalize it here: the fixture must hand
    # meshprep a consistently right-handed mesh for the assertions below to mean anything.
    flip = _signed_volumes(node, elem) < 0
    elem[np.ix_(flip, [2, 3])] = elem[np.ix_(flip, [3, 2])]
    return node, elem


def _signed_volumes(node, elem):
    """det([b-a; c-a; d-a])/6 per element (standard convention)."""
    e0 = np.asarray(elem)[:, :4].astype(int) - 1
    a, b, c, d = (node[e0[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def _prep(node, elem):
    cfg = {
        "node": node,
        "elem": elem,
        "prop": np.array([[0, 0, 1, 1], [0.02, 9, 0.89, 1.37]], dtype=float),
        "srcpos": np.array([[node[:, 0].mean(), node[:, 1].mean(), 0.0]]),
        "srcdir": np.array([[0, 0, 1.0]]),
        "detpos": np.array([node.mean(axis=0)]),
        "detdir": np.array([[0, 0, -1.0]]),
        "omega": 0.0,
    }
    return meshprep(cfg)[0]


class TestMeshprepEvolSign(unittest.TestCase):
    def test_single_tet_evol_positive(self):
        node, elem = _unit_tet()
        cfg = _prep(node, elem)
        evol = np.asarray(cfg["evol"]).ravel()
        self.assertTrue(np.all(evol > 0), "evol must be positive, got %s" % evol)
        np.testing.assert_allclose(evol, [1.0 / 6.0], rtol=1e-12)

    def test_mirrored_tet_evol_positive(self):
        """An inverted input tet must still come out with positive volume."""
        node, elem = _unit_tet()
        elem = elem[:, [0, 1, 3, 2, 4]]  # swap two vertices -> left-handed
        self.assertLess(_signed_volumes(node, elem)[0], 0)  # precondition
        cfg = _prep(node, elem)
        evol = np.asarray(cfg["evol"]).ravel()
        self.assertTrue(np.all(evol > 0), "evol must be positive, got %s" % evol)
        np.testing.assert_allclose(evol, [1.0 / 6.0], rtol=1e-12)

    def test_lattice_evol_positive_and_matches_geometry(self):
        """A regular lattice: uniform cells, so every |evol| is h**3/6."""
        node, elem = _lattice(6, 6, 4, h=2.0)
        self.assertTrue(np.all(_signed_volumes(node, elem) > 0))  # precondition
        cfg = _prep(node, elem)
        evol = np.asarray(cfg["evol"]).ravel()
        self.assertEqual(evol.size, elem.shape[0])
        self.assertTrue(np.all(evol > 0),
                        "%d of %d evol entries are negative" % (int((evol < 0).sum()), evol.size))
        np.testing.assert_allclose(evol, 8.0 / 6.0, rtol=1e-10)

    def test_deldotdel_sign_follows_evol(self):
        """deldotdel is scaled by evol, so it must not come out negated either."""
        node, elem = _lattice(6, 6, 4, h=2.0)
        cfg = _prep(node, elem)
        dd = np.asarray(cfg["deldotdel"])
        # the diagonal terms of grad(phi_i).grad(phi_i) are strictly positive
        self.assertGreater(dd.sum(), 0,
                           "deldotdel sums to %.6g; a negative total means the "
                           "stiffness matrix is sign-flipped" % dd.sum())


class TestForwardSolutionIsPhysical(unittest.TestCase):
    """End-to-end guard: a CW diffusion fluence is non-negative and peaks at the source."""

    def test_lattice_forward_is_a_fluence(self):
        import redbirdpy as rb

        node, elem = _lattice(11, 11, 6, h=2.0)
        cfg = _prep(node, elem)
        phi = np.asarray(rb.runforward(cfg)[1])
        f = phi[:, 0] if phi.ndim > 1 else phi
        fmax = f.max()
        self.assertGreater(fmax, 0)
        negfrac = float(np.mean(f < -1e-6 * fmax))
        self.assertLess(negfrac, 0.01,
                        "%.1f%% of nodes are negative: the solve is not a fluence" % (100 * negfrac))
        self.assertGreater(f.min(), -1e-2 * fmax)
        # the peak must sit near the source, not at a far boundary/corner
        src = np.asarray(cfg["srcpos"])[0]
        peak = np.asarray(cfg["node"])[int(np.argmax(f))]
        self.assertLess(np.linalg.norm(peak - src), 5.0,
                        "fluence peaks at %s, %.1f mm from the source at %s"
                        % (peak, np.linalg.norm(peak - src), src))


if __name__ == "__main__":
    unittest.main(verbosity=2)
