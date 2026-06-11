"""
Tests for the grade-sorted CCGA object taxonomy (paper Table tab:zoo / OBJECTS.md
"Objects by grade" index).

Every taxonomy form is asserted to have its claimed OPNS grade, organised into the
three construction ladders:

  1. bare multipole ladder ......... points only
  2. Iod-gauged conic ladder ....... every full CCGA conic ends in ^Iod (grade 7)
  3. Iinfd CGA-embedded sub-array .. CGA round/flat objects, grade = CGA grade + 2

Plus the pencil-rung grades (p1^…^pn^Iod = grade n+2, n≤4) and the gauged-dipole
E^F^Iod incidence.
"""
import pytest

from ccga.algebra import Iod, I
from ccga.objects import (
    make_point_ccga as P, twopole, make_gauged_dipole,
    make_conic_tripole, make_conic_quadpole, make_conic_pentapole,
    make_conic_opns, make_hyperbola_3points, make_parabola_3points,
    make_ellipse_3points, make_flat_point, make_line_at_infinity,
    make_conic_at_infinity, make_round_point,
)
from ccga.point import point_at_infinity
from ccga import cga

TOL = 1e-9

# concrete sample points (no accidental degeneracies)
P1, P2, P3, P4, P5 = P(1.0, 0.0), P(0.0, 1.0), P(-1.0, 0.0), P(0.0, -1.0), P(2.0, 3.0)


def gset(mv, tol=TOL):
    """Sorted set of grades present in mv, after chopping float noise."""
    return tuple(sorted({bin(k).count("1")
                         for k, v in mv.items() if abs(complex(v)) > tol}))


# ── 1. bare multipole ladder (points only) ───────────────────────────────────

@pytest.mark.parametrize("mv,g", [
    (P1,                                         (1,)),   # point
    (twopole(P1, P2)[0],                         (2,)),   # twopole  p1^p2
    (make_conic_tripole(P1, P2, P3)[0],          (3,)),   # tripole
    (make_conic_quadpole(P1, P2, P3, P4)[0],     (4,)),   # quadpole
    (make_conic_pentapole(P1, P2, P3, P4, P5)[0],(5,)),   # pentapole
])
def test_bare_multipole_ladder_grades(mv, g):
    assert gset(mv) == g


# ── 2. Iod-gauged ladder: pencils (n≤4) → conic (n=5) ────────────────────────

@pytest.mark.parametrize("n,g", [(2, 4), (3, 5), (4, 6)])
def test_pencil_rung_grades(n, g):
    """p1^…^pn^Iod (n≤4) is a pencil of grade n+2."""
    pts = [P1, P2, P3, P4][:n]
    blade = pts[0]
    for p in pts[1:]:
        blade = blade ^ p
    blade = blade ^ Iod
    assert gset(blade) == (g,)


def test_gauged_dipole_is_n2_pencil():
    assert gset(make_gauged_dipole(P1, P2)) == (4,)


def test_conic_ladder_grade7():
    general   = make_conic_opns([P1, P2, P3, P4, P5])          # p1^…^p5^Iod
    hyperbola = make_hyperbola_3points(P1, P2, P3, (1, 0), (0, 1))[0]
    parabola  = make_parabola_3points(P1, P2, P3, (1, 0))[0]
    ellipse   = make_ellipse_3points(P1, P2, P3, 2.0, 1.0)[0]
    assert gset(general)   == (7,)
    assert gset(hyperbola) == (7,)   # 3 pts + 2 real ideal pts + Iod
    assert gset(parabola)  == (7,)   # 3 pts + 1 double ideal pt + Iod
    assert gset(ellipse)   == (7,)   # 3 pts + 2 imaginary ideal pts + Iod


def test_ideal_point_grade():
    assert gset(point_at_infinity(1.0, 2.0)) == (1,)


# ── 3. CGA embedded inside CCGA (^Iinfd, grade = CGA grade + 2) ───────────────

@pytest.mark.parametrize("mv,g", [
    (make_round_point(1.0, 2.0),     (3,)),   # CGA round point  p^Iinfd     (CGA gr1)
    (cga.point_pair(P1, P2),         (4,)),   # CGA point pair   p1^p2^Iinfd (CGA gr2)
    (cga.circle(P1, P2, P3),         (5,)),   # CGA circle       p1^p2^p3^Iinfd (CGA gr3)
    (cga.flat_point(P1),             (4,)),   # CGA flat point   p^einf^Iinfd  (CGA gr2)
    (cga.line(P1, P2),               (5,)),   # CGA line         p1^p2^einf^Iinfd (CGA gr3)
])
def test_cga_embedded_grades(mv, g):
    assert gset(mv) == g


# ── flats / ideal / pseudoscalar ─────────────────────────────────────────────

@pytest.mark.parametrize("mv,g", [
    (make_flat_point(1.0, 2.0),  (4,)),   # native flat point  p^Iinf
    (make_line_at_infinity(),    (3,)),   # line at infinity   Iinf
    (make_conic_at_infinity(),   (5,)),   # conic at infinity  Iod^Iinf
    (I,                          (8,)),   # pseudoscalar
])
def test_flat_and_ideal_grades(mv, g):
    assert gset(mv) == g


# ── gauged dipole E^F^Iod: incident with exactly E and F ─────────────────────

def test_gauged_dipole_incidence():
    D = make_gauged_dipole(P1, P2)

    def wedge_norm(q):
        return max((abs(complex(v)) for v in (q ^ D).values()), default=0.0)

    assert wedge_norm(P1) < TOL                  # E on it
    assert wedge_norm(P2) < TOL                  # F on it
    assert wedge_norm(P3) > TOL                  # third point not on it
    assert wedge_norm(P(0.5, 0.5)) > TOL         # point on line E-F not on it
