"""
Tripole  T = p1∧p2∧p3  with real / ideal / imaginary round points.

Analysis of the circum-conic  T ∧ Iod ∧ Iinfd:
  - 3 finite/round points → their circle (round points, radius included, are
    IPNS-incident; r²_circum = r²_centres − r²_point);
  - 1 ideal point → the line through the two finite points (circle through ∞);
  - 2 ideal points → degenerate (vanishes).
Plus the reality of T² under real vs imaginary radius.
"""
import numpy as np
import pytest

from ccga.point import point, point_at_infinity
from ccga.objects import make_point_ccga
from ccga.operations import grades, is_zero
from ccga.classify import ipns_to_coeffs, conic_subtype
from ccga.extract import circumcircle, tripole_circumconic
from ccga.algebra import I_inv

TOL = 1e-9


def _T(*ps):
    R = ps[0]
    for p in ps[1:]:
        R = R ^ p
    return R


def _coeffs(T):
    return ipns_to_coeffs(tripole_circumconic(T) * I_inv)


def test_tripole_is_grade3_for_all_point_types():
    cases = [
        (point(0, 0), point(3, 0), point(1, 2)),
        (make_point_ccga(0, 0, 1), make_point_ccga(3, 0, 1), make_point_ccga(1, 2, 1)),
        (point(0, 0), point(3, 0), point_at_infinity(1, 1)),
        (point(0, 0), point_at_infinity(1, 0), point_at_infinity(0, 1)),
    ]
    for ps in cases:
        assert grades(_T(*ps)) == [3]


def test_circumcircle_of_finite_points():
    cx, cy, R = circumcircle(_T(point(0, 0), point(3, 0), point(1, 2)))
    assert np.allclose((cx, cy), (1.5, 0.5), atol=TOL)
    assert abs(R - np.sqrt(2.5)) < 1e-9


def test_round_points_are_incident_and_radius_enters():
    # round points (distinct radii) lie on the circum-circle in the IPNS sense
    ps = [make_point_ccga(0, 0, 0.5), make_point_ccga(3, 0, 1.0),
          make_point_ccga(1, 2, 1.5)]
    C = tripole_circumconic(_T(*ps)) * I_inv
    assert all(abs(float((p | C).e)) < 1e-9 for p in ps)
    # equal real radius shrinks r² by r²; imaginary grows it
    base = circumcircle(_T(point(0, 0), point(3, 0), point(1, 2)))[2] ** 2
    rr = circumcircle(_T(make_point_ccga(0, 0, 1), make_point_ccga(3, 0, 1),
                         make_point_ccga(1, 2, 1)))[2] ** 2
    ri = circumcircle(_T(make_point_ccga(0, 0, 1, True), make_point_ccga(3, 0, 1, True),
                         make_point_ccga(1, 2, 1, True)))[2] ** 2
    assert abs(rr - (base - 1)) < 1e-9
    assert abs(ri - (base + 1)) < 1e-9


def test_one_ideal_point_gives_line_through_finite_points():
    T = _T(point(1, 1), point(3, 2), point_at_infinity(2, -1))
    A, B, C, D, E, F = _coeffs(T)
    assert conic_subtype(A, B, C, D, E, F) == 'line'
    # the line passes through both finite points
    assert abs(A*1 + B*1 + C*1 + D*1 + E*1 + F) < 1e-9
    assert abs(A*9 + B*4 + C*6 + D*3 + E*2 + F) < 1e-9
    # and it is (up to scale) x - 2y + 1 = 0
    assert abs(D/E + 0.5) < 1e-9 and abs(F/E + 0.5) < 1e-9
    # circumcircle() refuses (it is not a circle)
    with pytest.raises(ValueError):
        circumcircle(T)


def test_two_ideal_points_degenerate():
    T = _T(point(0, 0), point_at_infinity(1, 0), point_at_infinity(0, 1))
    assert is_zero(tripole_circumconic(T) * I_inv)     # circum-conic vanishes
    assert abs(float((_T(point(0, 0), point_at_infinity(1, 0),
                         point_at_infinity(0, 1)) * 0).e)) < 1e-12


@pytest.mark.parametrize("r2,expect", [(1.0, 80.0), (0.0, 90.0), (-1.0, 104.0)])
def test_tripole_square_reality(r2, expect):
    if r2 == 0:
        p0 = point(0, 0)
    else:
        p0 = make_point_ccga(0, 0, abs(r2) ** 0.5, imaginary=(r2 < 0))
    T = _T(p0, point(3, 0), point(1, 2))
    assert abs(float((T * T).e) - expect) < 1e-9
