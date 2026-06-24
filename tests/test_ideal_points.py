"""
GA-native ideal-point extraction via the conic-at-infinity meet.

The conic at infinity is  C_∞ = Iod ∧ Iinf  (grade 5).  For a conic C (grade-7
OPNS):

  - C ∨ C_∞ is grade 7+5−8 = 4 (the conic's incidence with the Veronese cone at
    infinity);
  - T = (C ∨ C_∞) | Iinfd is the grade-2 *asymptotic dipole*, the twopole of the
    conic's two ideal points, T ∝ point_at_infinity(d1) ∧ point_at_infinity(d2);
  - the one-step meet with the line at infinity C ∨ Iinf gives the same twopole;
  - ideal_points pulls the directions out of T by its dual within the infinity
    3-space, agreeing with classify.asymptotic_directions.
"""
import numpy as np

from ccga.point import point_at_infinity
from ccga.algebra import Iod, Iinf, Iinfd
from ccga.objects import make_hyperbola, make_ellipse, make_parabola, make_tilted_ellipse
from ccga.operations import grades, meet, proportional
from ccga.classify import asymptotic_directions, _conic_vector
from ccga.extract import asymptotic_dipole, ideal_points

TOL = 1e-9


def _conics():
    return {
        'hyperbola': make_hyperbola(1, 1),
        'ellipse':   make_ellipse(3, 2),
        'parabola':  make_parabola(1, 'x'),
        'tilted':    make_tilted_ellipse(3, 1, 0.5236),
    }


def test_meet_with_conic_at_infinity_is_grade_4():
    Cinf = Iod ^ Iinf
    assert grades(Cinf) == [5]
    for opns, _ in _conics().values():
        assert grades(meet(opns, Cinf)) == [4]


def test_asymptotic_dipole_is_grade_2():
    for opns, _ in _conics().values():
        assert grades(asymptotic_dipole(opns)) == [2]


def test_both_routes_agree():
    # (C ∨ C_∞) | Iinfd  ∝  C ∨ Iinf
    for opns, _ in _conics().values():
        ok, _ = proportional(asymptotic_dipole(opns), meet(opns, Iinf))
        assert ok


def test_twopole_is_wedge_of_ideal_points():
    opns, _ = make_hyperbola(1, 1)               # asymptotes y = ±x
    V = point_at_infinity(1, 1) ^ point_at_infinity(1, -1)
    ok, _ = proportional(asymptotic_dipole(opns), V)
    assert ok


def test_ideal_points_match_asymptotic_directions():
    for opns, ipns in _conics().values():
        ips = ideal_points(opns)
        # one ideal point per real asymptotic direction
        assert len(ips) == len(asymptotic_directions(ipns))
        # and each lifted ideal point lies on the conic (q · s = 0)
        s = _conic_vector(ipns)
        for q in ips:
            assert abs(float((q | s).e)) < 1e-7


def test_reality_counts():
    counts = {name: len(ideal_points(opns)) for name, (opns, _) in _conics().items()}
    assert counts['hyperbola'] == 2     # two real asymptotes
    assert counts['parabola'] == 1      # one double (axis) direction
    assert counts['ellipse'] == 0       # ideal pair is imaginary
    assert counts['tilted'] == 0
