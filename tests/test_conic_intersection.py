"""
Conic ∨ conic intersection as the grade-6 object  I4 = C1 ∨ C2 = Q ∧ Iod.

Two conics meet in 4 points (Bézout).  The regressive product is a grade-6 blade
equal (up to scale) to p1∧p2∧p3∧p4∧Iod for the 4 intersection points, which may
be real, imaginary (a conjugate pair) or ideal (at infinity).  Covered:

  - meet is grade 6 and ∝ p1∧p2∧p3∧p4∧Iod on a pencil through 4 known points;
  - incidence q ∧ I4 = 0; quadpole recovery (einf3 ∧ einfbar) | I4;
  - intersection_points recovers the real finite points;
  - intersection_reality gives the {real, imaginary, ideal} Bézout split (sum 4).
"""
import numpy as np
import pytest

from ccga.point import point, point_at_infinity
from ccga.algebra import Iod
from ccga.objects import make_ellipse
from ccga.operations import grades, is_zero
from ccga.extract import (
    conic_intersection, intersection_quadpole, intersection_points,
    intersection_reality, extract_quadpole,
)

TOL = 1e-9


def _wedge(*mvs):
    R = mvs[0]
    for m in mvs[1:]:
        R = R ^ m
    return R


def _prop(a, b):
    A = {k: float(v) for k, v in a.items()}
    B = {k: float(v) for k, v in b.items()}
    rs = [A[k] / B[k] for k in set(A) | set(B) if abs(B.get(k, 0)) > 1e-9]
    return bool(rs) and is_zero(a - float(np.median(rs)) * b)


# four shared points + two distinct 5th points → a pencil meeting exactly there
P1, P2, P3, P4 = point(0, 0), point(3, 0), point(0, 2), point(2, 3)
C1 = _wedge(P1, P2, P3, P4, point(-1, 1), Iod)
C2 = _wedge(P1, P2, P3, P4, point(4, 1), Iod)


def test_meet_is_grade6_quadpole_wedge_iod():
    M = conic_intersection(C1, C2)
    assert grades(M) == [6]
    I4 = _wedge(P1, P2, P3, P4, Iod)
    assert _prop(M, I4)                       # ∝ p1∧p2∧p3∧p4∧Iod


def test_incidence_and_quadpole_recovery():
    M = conic_intersection(C1, C2)
    # incidence: each intersection point q satisfies q ∧ I4 = 0
    assert all(is_zero(p ^ M) for p in (P1, P2, P3, P4))
    assert not is_zero(point(1, 1) ^ M)
    # recover the grade-4 quadpole and extract the 4 points
    Q = intersection_quadpole(M)
    assert grades(Q) == [4]
    got = sorted((round(x, 3), round(y, 3)) for x, y in extract_quadpole(Q))
    assert got == [(0.0, 0.0), (0.0, 2.0), (2.0, 3.0), (3.0, 0.0)]


def test_intersection_points_four_real():
    pts = intersection_points(make_ellipse(3, 2)[0], make_ellipse(2, 3)[0])
    got = sorted((round(x, 3), round(y, 3)) for x, y in pts)
    assert got == [(-1.664, -1.664), (-1.664, 1.664),
                   (1.664, -1.664), (1.664, 1.664)]


@pytest.mark.parametrize("name,A,B,expected", [
    ("4 real",
     make_ellipse(3, 2)[0], make_ellipse(2, 3)[0],
     {'real': 4, 'imaginary': 0, 'ideal': 0}),
    ("2 real + 2 ideal (overlapping circles)",
     make_ellipse(2, 2, 0, 0)[0], make_ellipse(2, 2, 2, 0)[0],
     {'real': 2, 'imaginary': 0, 'ideal': 2}),
    ("0 real (disjoint circles: 2 imaginary + 2 ideal)",
     make_ellipse(1, 1, 0, 0)[0], make_ellipse(1, 1, 5, 0)[0],
     {'real': 0, 'imaginary': 2, 'ideal': 2}),
])
def test_intersection_reality(name, A, B, expected):
    r = intersection_reality(A, B)
    assert sum(r.values()) == 4                # Bézout
    assert r == expected


def test_intersection_reality_with_ideal_point():
    # two hyperbolas sharing the real asymptotic direction (1,0) → 1 ideal point
    f = (point(0, 0), point(2, 1), point(1, 3))
    H1 = _wedge(*f, point_at_infinity(1, 0), point_at_infinity(0, 1), Iod)
    H2 = _wedge(*f, point_at_infinity(1, 0), point_at_infinity(1, 1), Iod)
    r = intersection_reality(H1, H2)
    assert sum(r.values()) == 4
    assert r['real'] == 3 and r['ideal'] == 1
    # the shared ideal point lies on the grade-6 intersection object
    assert is_zero(point_at_infinity(1, 0) ^ conic_intersection(H1, H2))
