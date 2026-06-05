"""Tests for ccga/point.py — §2 anchors (symbolic + numeric)."""
import numpy as np
import pytest
import sympy as sp
from ccga.point import point, normalize, inner_product
from ccga.algebra import einf, eo


def test_null_numeric():
    """p(x,y)^2 = 0 numerically for several points."""
    for x, y in [(0, 0), (1, 0), (0, 1), (3, -4), (1.5, 2.7)]:
        p = point(x, y)
        assert abs(float((p * p).e)) < 1e-12, f"p({x},{y})^2 ≠ 0"


def test_null_symbolic():
    """p(x,y)^2 = 0 symbolically (SymPy expansion)."""
    from ccga.algebra import e1, e2, eo, einf1, einf2, einf3
    x, y = sp.symbols('x y', real=True)
    p_sym = eo + x*e1 + y*e2 + (x*x/2)*einf1 + (y*y/2)*einf2 + x*y*einf3
    p2 = (p_sym * p_sym).e   # extract scalar component first
    assert sp.simplify(p2) == 0


def test_normalization():
    """p · einf = -1 for all embedded points."""
    for x, y in [(0, 0), (1, 2), (-3, 0.5), (10, -7)]:
        p = point(x, y)
        norm_val = float((p | einf).e)
        assert abs(norm_val + 1.0) < 1e-12, f"p({x},{y}) · einf = {norm_val}"


def test_normalization_symbolic():
    """p(x,y) · einf = -1 symbolically."""
    from ccga.algebra import e1, e2, eo, einf1, einf2, einf3
    x, y = sp.symbols('x y', real=True)
    p_sym = eo + x*e1 + y*e2 + (x*x/2)*einf1 + (y*y/2)*einf2 + x*y*einf3
    norm_val = (p_sym | einf).e
    assert sp.simplify(norm_val + 1) == 0


def test_distance():
    """p(x,y) · p(x',y') = -½[(x-x')²+(y-y')²]."""
    cases = [((0,0),(1,0)), ((1,2),(4,6)), ((0,0),(3,4)), ((-1,1),(2,-3))]
    for (x1,y1), (x2,y2) in cases:
        p1 = point(x1, y1)
        p2 = point(x2, y2)
        got  = inner_product(p1, p2)
        want = -0.5 * ((x1-x2)**2 + (y1-y2)**2)
        assert abs(got - want) < 1e-12, f"distance({(x1,y1)},{(x2,y2)}): {got} ≠ {want}"


def test_distance_symbolic():
    """p·q = -½[(x-x')²+(y-y')²] symbolically."""
    from ccga.algebra import e1, e2, eo, einf1, einf2, einf3
    x, y, xp, yp = sp.symbols('x y xp yp', real=True)
    def _pt(cx, cy):
        return eo + cx*e1 + cy*e2 + (cx*cx/2)*einf1 + (cy*cy/2)*einf2 + cx*cy*einf3
    p_sym = _pt(x, y)
    q_sym = _pt(xp, yp)
    ip = (p_sym | q_sym).e
    expected = sp.Rational(-1, 2) * ((x - xp)**2 + (y - yp)**2)
    diff = sp.expand(sp.simplify(ip - expected))
    assert diff == 0


def test_self_distance_zero():
    """p · p = 0 (self-distance is zero = null vector)."""
    for x, y in [(1, 2), (0, 5)]:
        p = point(x, y)
        assert abs(inner_product(p, p)) < 1e-12


def test_normalize_function():
    """normalize() scales a round object to p·einf = -1."""
    p = point(3, 4)
    scaled = p * 2.5
    p_norm = normalize(scaled)
    assert abs(float((p_norm | einf).e) + 1.0) < 1e-12


def test_verify_all():
    """Meta-test: verify_point_properties passes."""
    from ccga.point import verify_point_properties
    verify_point_properties()
