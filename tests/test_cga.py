"""
Tests for ccga/cga.py — the CGA "round" object family recovered via Iinfd.

For each object: correct grade, OPNS incidence (builder points lie on it,
extra cocircular/collinear point lies on it, off-object point does not),
reality (sign of (Iod|O)²), finite/ideal, and classification.
"""
import numpy as np
import pytest

from ccga import cga
from ccga.point import point
from ccga.objects import (make_point_ccga, make_ideal_point, make_flat_point,
                          make_round_point, make_line_at_infinity,
                          make_conic_at_infinity)
from ccga.operations import grades, is_zero
from ccga.classify import classify

TOL = 1e-9


def _on(Q, O):
    """Point Q lies on OPNS object O iff Q ^ O ≈ 0."""
    return is_zero(Q ^ O, TOL)


# ── grades ────────────────────────────────────────────────────────────────────

def test_grades():
    A, B, C = point(2, 0), point(-2, 0), point(0, 2)
    assert grades(cga.round_point(point(3, 4))) == [3]
    assert grades(cga.point_pair(A, B)) == [4]
    assert grades(cga.flat_point(point(3, 4))) == [4]
    assert grades(cga.circle(A, B, C)) == [5]
    assert grades(cga.line(point(0, 0), point(2, 0))) == [5]


# ── incidence ──────────────────────────────────────────────────────────────────

def test_round_point_incidence():
    R = cga.round_point(point(3, 4))
    assert _on(point(3, 4), R)
    assert not _on(point(0, 0), R)


def test_point_pair_incidence():
    A, B = point(1, 0), point(-1, 0)
    PP = cga.point_pair(A, B)
    assert _on(A, PP) and _on(B, PP)
    assert not _on(point(0, 1), PP)


def test_circle_incidence_and_cocircular():
    c, r = (0.0, 0.0), 2.0
    ang = [0.3, 1.7, 3.0, 4.5]
    pts = [point(c[0] + r*np.cos(a), c[1] + r*np.sin(a)) for a in ang]
    C = cga.circle(pts[0], pts[1], pts[2])
    for p in pts[:3]:
        assert _on(p, C)
    assert _on(pts[3], C), "4th cocircular point should lie on the circle"
    assert not _on(point(5, 5), C)


def test_line_incidence_collinear():
    L = cga.line(point(0, 0), point(2, 0))
    assert _on(point(1, 0), L) and _on(point(5, 0), L)
    assert not _on(point(0, 1), L)


def test_flat_point_incidence_and_equivalence():
    P = point(3, 4)
    FP = cga.flat_point(P)
    assert _on(P, FP)
    assert not _on(point(0, 0), FP)
    # flat_point(p) = p ^ einf ^ Iinfd == −(p ^ Iinf) == −make_flat_point
    assert is_zero(FP + make_flat_point(3, 4), TOL)


# ── reality (sign of (Iod|O)²) ─────────────────────────────────────────────────

def test_reality_real_pair():
    A, B = point(2, 0), point(-2, 0)
    assert cga.reality(cga.point_pair(A, B)) == 'real'


def test_reality_imaginary_sphere():
    sphere = cga.round_point(make_point_ccga(0, 0, 2.0, imaginary=True))
    assert cga.reality(sphere) == 'imaginary'


def test_reality_real_sphere():
    sphere = cga.round_point(make_point_ccga(1, 1, 2.0))
    assert cga.reality(sphere) == 'real'


def test_reality_zero_radius_degenerate():
    """A null (r=0) round point has (Iod|O)² = 0 → degenerate."""
    assert cga.reality(cga.round_point(point(3, 4))) == 'degenerate'


# ── finite vs ideal ─────────────────────────────────────────────────────────────

def test_finite_vs_ideal():
    assert cga.is_finite(cga.round_point(point(3, 4))) is True
    assert cga.is_finite(cga.round_point(make_ideal_point(3, 4))) is False


# ── cga_blade extraction grade ─────────────────────────────────────────────────

def test_cga_blade_grade_drops_by_two():
    A, B, C = point(2, 0), point(-2, 0), point(0, 2)
    assert grades(cga.cga_blade(cga.round_point(point(3, 4)))) == [1]
    assert grades(cga.cga_blade(cga.point_pair(A, B))) == [2]
    assert grades(cga.cga_blade(cga.circle(A, B, C))) == [3]


# ── classify_cga ────────────────────────────────────────────────────────────────

def test_classify_cga_types():
    A, B, C = point(2, 0), point(-2, 0), point(0, 2)
    assert cga.classify_cga(cga.round_point(point(3, 4)))['type'] == 'cga_round_point'
    assert cga.classify_cga(cga.point_pair(A, B))['type'] == 'cga_point_pair'
    assert cga.classify_cga(cga.flat_point(point(3, 4)))['type'] == 'cga_flat_point'
    assert cga.classify_cga(cga.circle(A, B, C))['type'] == 'cga_circle'
    assert cga.classify_cga(cga.line(point(0, 0), point(2, 0)))['type'] == 'cga_line'


# ── classify() integration + at-infinity disambiguation ────────────────────────

def test_classify_integration_cga():
    A, B, C = point(2, 0), point(-2, 0), point(0, 2)
    assert classify(cga.round_point(point(3, 4)))['type'] == 'cga_round_point'
    assert classify(cga.point_pair(A, B))['type'] == 'cga_point_pair'
    assert classify(cga.circle(A, B, C))['type'] == 'cga_circle'
    assert classify(cga.line(point(0, 0), point(2, 0)))['type'] == 'cga_line'
    # origin-centred sphere (no e1/e2) is still recognised
    assert classify(cga.round_point(make_point_ccga(0, 0, 2.0)))['type'] == 'cga_round_point'


def test_classify_keeps_at_infinity_labels():
    assert classify(make_line_at_infinity())['type'] == 'line_at_infinity'
    assert classify(make_conic_at_infinity())['type'] == 'conic_at_infinity'


def test_make_round_point_delegates():
    """objects.make_round_point(x,y,r) builds the same grade-3 CGA round point."""
    R = make_round_point(3, 4)
    assert grades(R) == [3]
    assert is_zero(R - cga.round_point(make_point_ccga(3, 4)), TOL)
