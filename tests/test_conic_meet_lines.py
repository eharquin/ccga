"""
Discriminants and center as the meet of three dual lines (paper §7,
"Computing discriminants in QC2GA"):

  l1 = dual( A·e1 + (C/2)·e2 - (D/2)·einf )
  l2 = dual( (C/2)·e1 + B·e2 - (E/2)·einf )
  l3 = dual( (D/2)·e1 + (E/2)·e2 - F·einf )

  l1 & l2       == -1/2 * (Delta_2·eo + x_c·e1 + y_c·e2) ^ Iod ^ Iinf
  l1 & l2 & l3  == -1/2 * Delta_3 * Iod ^ Iinf

This replaces the np.linalg.det/solve fallback in conic_is_degenerate /
conic_center_point with a pure-GA construction.
"""
import numpy as np
import pytest

from ccga.objects import make_conic_ipns, make_line_ipns, make_line_pair, make_ellipse
from ccga.classify import (
    conic_center, conic_center_meet, conic_center_point,
    conic_discriminant, conic_discriminant2, conic_discriminant3,
    conic_is_degenerate, ipns_to_coeffs, _conic_lines,
)
from ccga.operations import grades, is_zero
from ccga.point import point_at_infinity

TOL = 1e-9


def _det_M3(A, B, C, D, E, F):
    M3 = np.array([[A, C/2, D/2], [C/2, B, E/2], [D/2, E/2, F]])
    return np.linalg.det(M3)


CONICS = [
    (0.7, -1.3, 0.4, 2.0, -0.5, -3.0),    # generic hyperbola-ish
    (0.5, 0.5, -0.3, -1.0, 2.0, -2.0),    # generic ellipse-ish
    (1.0, 1.0, 0.0, -4.0, 6.0, 4.0),      # circle
    (2.0, 5.0, 0.0, 0.0, 0.0, -10.0),     # axis-aligned ellipse
]


def test_conic_lines_are_grade7():
    A, B, C, D, E, F = CONICS[0]
    l1, l2, l3 = _conic_lines(A, B, C, D, E, F)
    assert grades(l1) == [7]
    assert grades(l2) == [7]
    assert grades(l3) == [7]


@pytest.mark.parametrize("ABCDEF", CONICS)
def test_discriminant2_matches_AB_minus_C2_4(ABCDEF):
    A, B, C, D, E, F = ABCDEF
    s = make_conic_ipns(A, B, C, D, E, F)
    assert abs(conic_discriminant2(s) - (A*B - C*C/4)) < TOL
    # relation to the classic discriminant Delta = C^2 - 4AB
    assert abs(conic_discriminant2(s) + conic_discriminant(A, B, C) / 4) < TOL


@pytest.mark.parametrize("ABCDEF", CONICS)
def test_discriminant3_matches_det_M3(ABCDEF):
    A, B, C, D, E, F = ABCDEF
    s = make_conic_ipns(A, B, C, D, E, F)
    assert abs(conic_discriminant3(s) - _det_M3(A, B, C, D, E, F)) < TOL


@pytest.mark.parametrize("ABCDEF", CONICS)
def test_center_meet_matches_conic_center(ABCDEF):
    A, B, C, D, E, F = ABCDEF
    s = make_conic_ipns(A, B, C, D, E, F)
    w, x, y = conic_center_meet(s)
    assert abs(w) > TOL                       # central conic: Delta_2 != 0
    cx, cy = conic_center(s)
    assert abs(x/w - cx) < TOL
    assert abs(y/w - cy) < TOL


def test_parabola_center_meet_is_ideal():
    # y = x^2  ->  x^2 - y = 0  =>  A=1,B=0,C=0,D=0,E=-1,F=0
    A, B, C, D, E, F = 1.0, 0.0, 0.0, 0.0, -1.0, 0.0
    s = make_conic_ipns(A, B, C, D, E, F)
    w, x, y = conic_center_meet(s)
    assert abs(w) < TOL                       # Delta_2 == 0: no finite center
    assert abs(conic_discriminant3(s)) > TOL  # genuine (non-degenerate) parabola
    assert not conic_is_degenerate(s)

    pinf = conic_center_point(s)
    n = (x*x + y*y) ** 0.5
    assert is_zero(pinf - point_at_infinity(x/n, y/n))


def test_line_pair_discriminant3_zero():
    l1 = make_line_ipns(1, -1, 0)[1]      # x - y = 0
    l2 = make_line_ipns(1, 1, -2)[1]      # x + y - 2 = 0
    _, ipns = make_line_pair(l1, l2)
    A, B, C, D, E, F = ipns_to_coeffs(ipns)
    assert abs(conic_discriminant3(ipns) - _det_M3(A, B, C, D, E, F)) < TOL
    assert abs(conic_discriminant3(ipns)) < 1e-7
    assert conic_is_degenerate(ipns)
    with pytest.raises(ValueError):
        conic_center_point(ipns)


def test_center_point_finite_matches_conic_center():
    Ce = make_ellipse(5, 2, cx=1.0, cy=-2.0)[0]
    from ccga.classify import _conic_vector
    s = _conic_vector(Ce)
    cx, cy = conic_center(s)
    w, x, y = conic_center_meet(s)
    assert abs(x/w - cx) < TOL and abs(y/w - cy) < TOL
