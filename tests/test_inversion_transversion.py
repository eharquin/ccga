"""
Inversion and transversion — the remaining conformal (Möbius) maps.

These are **CGA round-family** versors (circles / round points / lines), NOT
general-conic versors: a versor preserves grade-1 = conics, but inversion sends
an ellipse to a quartic.  Tests verify the positive (round-family) action and
document the negative (ellipse → not a conic) result.
"""
import numpy as np

from ccga.point import point
from ccga import cga
from ccga.algebra import eo, einf, e1, e2, Iod
from ccga.transform import apply_versor, inversion, transversion
from ccga.objects import make_ellipse
from ccga.classify import ipns_to_coeffs, _conic_vector

TOL = 1e-7


def _euclid_round(R):
    """Euclidean (x, y) of a CGA round point/object via the CGA blade Iod|R."""
    b = Iod | R
    w = -float((b | einf).e)
    return float((b | e1).e) / w, float((b | e2).e) / w


def test_inversion_unit_circle_round_point():
    R = cga.round_point(point(3, 4))            # |z|² = 25
    Ri = apply_versor(inversion(0, 0, 1), R)
    assert np.allclose(_euclid_round(Ri), (3/25, 4/25), atol=TOL)


def test_inversion_general_circle():
    R = cga.round_point(point(2, 0))
    Ri = apply_versor(inversion(1, 0, 2), R)    # invert in circle centre (1,0) r=2
    # inversion in circle C(c,r): p -> c + r²(p-c)/|p-c|²
    c = np.array([1.0, 0.0]); p = np.array([2.0, 0.0])
    expect = c + 4*(p - c)/np.sum((p - c)**2)
    assert np.allclose(_euclid_round(Ri), expect, atol=TOL)


def test_inversion_is_involution_on_round_points():
    V = inversion(0, 0, 1)
    R = cga.round_point(point(3, 4))
    assert np.allclose(_euclid_round(apply_versor(V, apply_versor(V, R))),
                       (3, 4), atol=TOL)


def test_inversion_maps_circle_to_circle():
    # a circle not through the inversion centre stays a circle: 6 points stay concyclic
    pts = [point(3 + np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 6, endpoint=False)]
    xy = [_euclid_round(apply_versor(inversion(0, 0, 1), cga.round_point(p))) for p in pts]
    M = np.array([[x*x + y*y, x, y, 1] for x, y in xy])
    assert np.linalg.svd(M)[1][-1] < 1e-9       # all 6 concyclic ⇒ one circle eqn


def test_transversion_preserves_round_points():
    V = transversion(0.1, 0.0)
    R = cga.round_point(point(2, 0))
    out = _euclid_round(apply_versor(V, R))
    assert np.all(np.isfinite(out))             # stays a finite round point


def test_inversion_of_ellipse_is_not_a_conic():
    # the decisive negative result: inversion is not a CCGA conic versor
    pts = [(3*np.cos(t), 2*np.sin(t)) for t in np.linspace(0, 2*np.pi, 24, endpoint=False)]
    inv = [(x/(x*x + y*y), y/(x*x + y*y)) for x, y in pts]
    # conic fit (6 coeffs, 24 pts): residual large ⇒ no conic through them
    conic_fit = np.linalg.svd(np.array([[x*x, y*y, x*y, x, y, 1] for x, y in inv]))[1][-1]
    # quartic fit (15 coeffs, 24 pts): a null vector exists ⇒ the points lie on a quartic
    quartic_fit = np.linalg.svd(np.array(
        [[x**4, y**4, x**3*y, x*x*y*y, x*y**3, x**3, y**3, x*x*y, x*y*y,
          x*x, y*y, x*y, x, y, 1] for x, y in inv]))[1][-1]
    assert conic_fit > 1e-3                      # NOT a conic
    assert quartic_fit < 1e-9                    # IS a quartic
