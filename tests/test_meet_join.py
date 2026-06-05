"""
Tests for meet/join interaction table (§5 intersections + §6 verification protocol).

Covers:
  - line ∧ line → point (via dual: IPNS lines meet at a point)
  - conic ∨ line → point pair (grade 6 meet, incidence)
  - conic ∨ conic → 4 points (Bézout, grade 6 meet, incidence of 4 shared pts)
  - pencil: λC1 + μC2 passes through base points
  - meet grade arithmetic verification
"""
import numpy as np
import pytest
from ccga.algebra import I, I_inv, einf, Iod
from ccga.point import point
from ccga.objects import (
    make_conic_opns, make_conic_ipns, conic_from_5points,
    make_circle, make_ellipse, make_hyperbola, make_parabola, make_line_ipns,
)
from ccga.operations import dual, undual, meet, join, is_zero, grades, pure_grade

TOL = 1e-8


def _on_opns(p, C, tol=TOL):
    wedge = p ^ C
    return all(abs(float(v)) < tol for v in wedge.values())


def _on_ipns(p, s, tol=TOL):
    return abs(float((p | s).e)) < tol


# ── Meet grade arithmetic ────────────────────────────────────────────────────

def test_meet_grade_7_7():
    """Two grade-7 OPNS conics meet in grade 6 (= 7+7-8)."""
    pts1 = [point(np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 6)[:5]]
    pts2 = [point(2*np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 6)[:5]]
    C1 = make_conic_opns(pts1)
    C2 = make_conic_opns(pts2)
    m = C1 & C2
    assert grades(m) == [6], f"Expected grade-6 meet, got {grades(m)}"


def test_meet_grade_7_1():
    """Grade-7 OPNS & grade-1 IPNS = grade 0 (scalar)."""
    pts = [point(np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 6)[:5]]
    C_opns, C_ipns = conic_from_5points(pts)
    m = C_opns & C_ipns
    assert grades(m, tol=1e-6) == [0], f"Expected scalar, got {grades(m)}"


# ── Line ∨ Line → Point ───────────────────────────────────────────────────────

def test_line_meet_line_gives_point():
    """
    Two IPNS lines meet at their intersection point.

    Meet of two IPNS grade-1 vectors gives grade 1+1-8 = -6 (zero) directly.
    Correct approach: dualize lines to OPNS grade-7, meet, then dualize back.
    """
    # Line 1: x = 2  (nx=1,ny=0,d=-2)
    # Line 2: y = 1  (nx=0,ny=1,d=-1)
    # Intersection: (2, 1)
    L1_opns, L1_ipns = make_line_ipns(1, 0, -2)
    L2_opns, L2_ipns = make_line_ipns(0, 1, -1)

    # Meet of two grade-7 OPNS lines → grade 6 (= join of "all line points")
    m = L1_opns & L2_opns
    assert grades(m) == [6]

    # The known intersection point (2,1) must lie in the meet's OPNS
    p_int = point(2, 1)
    assert _on_opns(p_int, m), "Intersection point not in line∨line meet"

    # A non-intersection point must not lie in the meet
    p_off = point(2, 2)
    assert not _on_opns(p_off, m, tol=0.05), "Off-line point wrongly in meet"


def test_line_meet_line_pencil():
    """Three concurrent lines all share the same meeting point."""
    # Lines through (1, 2): ax+by = a+2b for various a,b
    lines = [(1, 0, -1), (0, 1, -2), (1, 1, -3)]   # x=1, y=2, x+y=3
    p_shared = point(1, 2)
    for nx, ny, d in lines:
        L_opns, _ = make_line_ipns(nx, ny, d)
        assert _on_opns(p_shared, L_opns), f"Line ({nx},{ny},{d}) does not pass through (1,2)"


# ── Conic ∨ Line → Point Pair ─────────────────────────────────────────────────

def test_conic_meet_line_incidence():
    """
    Unit circle ∩ horizontal line y=0 gives ±1 on x-axis.

    The grade-6 meet contains the 2 intersection points.
    """
    circ_opns, _ = make_circle(0, 0, 1)
    line_opns, _ = make_line_ipns(0, 1, 0)   # y = 0

    m = circ_opns & line_opns
    assert grades(m) == [6]

    p_plus  = point( 1, 0)
    p_minus = point(-1, 0)
    assert _on_opns(p_plus,  m), "(+1,0) not in conic∨line meet"
    assert _on_opns(p_minus, m), "(-1,0) not in conic∨line meet"

    # A point on the circle but not on y=0 should NOT be in the meet
    p_off = point(0, 1)
    assert not _on_opns(p_off, m, tol=0.05), "(0,1) wrongly in meet"


def test_ellipse_meet_line():
    """Ellipse (x/2)²+y²=1 ∩ x=0 gives (0, ±1)."""
    ell_opns, _ = make_ellipse(2, 1)
    line_opns, _ = make_line_ipns(1, 0, 0)   # x = 0

    m = ell_opns & line_opns
    assert grades(m) == [6]
    for p in [point(0, 1), point(0, -1)]:
        assert _on_opns(p, m), f"{p} not in ellipse∨line meet"


def test_hyperbola_meet_asymptote_like_line():
    """Hyperbola x²-y²=1 ∩ y=0 gives (±1, 0)."""
    hyp_opns, _ = make_hyperbola(1, 1)
    line_opns, _ = make_line_ipns(0, 1, 0)   # y = 0

    m = hyp_opns & line_opns
    assert grades(m) == [6]
    for p in [point(1, 0), point(-1, 0)]:
        assert _on_opns(p, m), f"(±1,0) not in hyperbola∨y=0 meet"


# ── Conic ∨ Conic → 4 Points (Bézout) ────────────────────────────────────────

def test_circle_circle_4intersection_points():
    """
    Two circles with 4 real intersection points.

    Circle 1: unit circle  x²+y² = 1
    Circle 2: x²+y²-x = 0  (passes through (0,±1) and (1,0) — actually
              centred at (0.5, 0) with radius 0.5)

    Simpler: two circles whose intersection is known exactly.
    Use: C1: x²+y²=4, C2: (x-2)²+y²=4 (two circles radius 2 centred 2 apart)
    Intersection: x=1, y=±√3.
    """
    C1_opns, _ = make_circle(0, 0, 2)
    C2_opns, _ = make_circle(2, 0, 2)

    m = C1_opns & C2_opns
    assert grades(m) == [6]

    # Known intersection points: (1, √3) and (1, -√3)
    sq3 = 3**0.5
    p1 = point(1, sq3)
    p2 = point(1, -sq3)
    assert _on_opns(p1, m), "(1,√3) not in circle∨circle meet"
    assert _on_opns(p2, m), "(1,-√3) not in circle∨circle meet"

    # Off points
    p_off = point(2, 0)
    assert not _on_opns(p_off, m, tol=0.05)


def test_conic_conic_4points_bezout():
    """
    Circle ∩ Ellipse = 4 real points (Bézout count verified).

    C1: x²+y² = 4
    C2: x²/4 + y² = 1

    Solve: x²+y²=4 and x²/4+y²=1 → subtract: 3x²/4=3 → x=±2, y=0
    Wait, x=±2, y²=4-4=0, so (±2,0) — only 2 points!

    Better: C1: x²+y²=2, C2: x²/2+y²=1
    Subtract: x²/2=1 → x=±√2, y=0. Still 2.

    Better: C1: (x-1)²+y²=2  and  C2: x²/4+y²=1
    C1: x²-2x+1+y²=2 → x²+y²=1+2x
    C2: x²/4+y²=1 → y²=1-x²/4
    Substitute: x² + 1-x²/4 = 1+2x → 3x²/4 - 2x = 0 → x(3x/4-2)=0
    → x=0 (y=±1) or x=8/3 (y²=1-64/36=1-16/9=-7/9 < 0). So 2 real points.

    Let's use: C1: x²+y²=5,  C2: (x-1)²+y²=2
    C1-C2: x²-(x-1)²=3 → 2x-1=3 → x=2 → y²=1 → (2,±1). Just 2 real.

    For 4 real intersections, use:
    C1: unit circle x²+y²=1
    C2: ellipse (x/a)²+(y/b)²=1 with a>1, b<1 so it crosses the unit circle 4 times.
    e.g. a=1.5, b=0.5: at y=0: x=1.5 > 1 (outside) and x=−1.5 (outside).
    At x=0: y=0.5 < 1 (inside). Crosses 4 times.

    Intersection of x²+y²=1 and x²/1.5²+y²/0.5²=1:
    Let u=x², v=y². u+v=1 and u/2.25+v/0.25=1.
    From first: v=1-u. Sub: u/2.25+(1-u)/0.25=1 → u/2.25+4(1-u)=1
    → u/2.25+4-4u=1 → u(1/2.25-4)=−3 → u*(4/9-4)=−3 → u*(−32/9)=−3 → u=27/32
    v=1-27/32=5/32. y=±√(5/32)≈±0.395. x=±√(27/32)≈±0.919.
    4 real solutions! ✓
    """
    a, b = 1.5, 0.5
    C1_opns, _ = make_circle(0, 0, 1)
    C2_opns, _ = make_ellipse(a, b)

    m = C1_opns & C2_opns
    assert grades(m) == [6]

    u = 27.0/32.0;  v = 5.0/32.0
    x_pos = u**0.5;  y_pos = v**0.5

    pts_int = [
        point( x_pos,  y_pos), point( x_pos, -y_pos),
        point(-x_pos,  y_pos), point(-x_pos, -y_pos),
    ]
    for i, p in enumerate(pts_int):
        assert _on_opns(p, m, tol=TOL), \
            f"Intersection point {i} not in conic∨conic meet"

    # Verify these 4 points lie on both conics separately
    for p in pts_int:
        assert _on_opns(p, C1_opns), "Point not on C1"
        assert _on_opns(p, C2_opns), "Point not on C2"


# ── Pencil of conics ─────────────────────────────────────────────────────────

def test_pencil_base_points():
    """
    λC1 + μC2 (IPNS pencil) passes through the base points of C1 ∩ C2.

    We use the 4-intersection case above: unit circle ∩ ellipse.
    """
    a, b = 1.5, 0.5
    _, C1_ipns = make_circle(0, 0, 1)
    _, C2_ipns = make_ellipse(a, b)

    u = 27.0/32.0;  v = 5.0/32.0
    x_pos = u**0.5;  y_pos = v**0.5
    base_pts = [
        point( x_pos,  y_pos), point( x_pos, -y_pos),
        point(-x_pos,  y_pos), point(-x_pos, -y_pos),
    ]

    # Several pencil members λC1 + μC2
    pencil_members = [
        C1_ipns + C2_ipns,          # λ=1, μ=1
        C1_ipns - C2_ipns,          # λ=1, μ=-1
        2*C1_ipns + 0.5*C2_ipns,    # λ=2, μ=0.5
    ]
    for i, s_pencil in enumerate(pencil_members):
        for j, p in enumerate(base_pts):
            val = abs(float((p | s_pencil).e))
            assert val < TOL * 10, \
                f"Pencil member {i}, base point {j}: q·s = {val}"


# ── Join ─────────────────────────────────────────────────────────────────────

def test_join_point_point():
    """p1 ^ p2 has grade 2 (point pair)."""
    p1, p2 = point(1, 0), point(-1, 0)
    pp = p1 ^ p2
    assert grades(pp) == [2]


def test_join_point_pair_point():
    """(p1^p2) ^ p3 has grade 3."""
    p1, p2, p3 = point(1, 0), point(-1, 0), point(0, 1)
    blade = p1 ^ p2 ^ p3
    assert grades(blade) == [3]


def test_Iod_join_5points_is_grade7():
    """Iod ^ p1 ^ p2 ^ p3 ^ p4 ^ p5 has grade 7."""
    pts = [point(np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 6)[:5]]
    C = Iod ^ pts[0] ^ pts[1] ^ pts[2] ^ pts[3] ^ pts[4]
    assert grades(C) == [7]


# ── Round-trip OPNS ↔ IPNS for conic ─────────────────────────────────────────

def test_opns_ipns_roundtrip_all_types():
    """dual(opns) is grade-1 and has same incidence for each conic type."""
    objects = [
        ('circle',    make_circle(0, 0, 1)),
        ('ellipse',   make_ellipse(2, 1)),
        ('hyperbola', make_hyperbola(1, 1)),
        ('parabola',  make_parabola(1)),
        ('line',      make_line_ipns(1, 0, -2)),
    ]
    for name, (C_opns, C_ipns) in objects:
        C_dual = dual(C_opns)
        assert grades(C_dual) == [1], f"{name}: dual(OPNS) not grade 1"
        # Shared incidence: points on C_opns also satisfy p | dual(C_opns) ≈ 0
        test_points = {
            'circle':    [point(1,0), point(0,1), point(-1,0)],
            'ellipse':   [point(2,0), point(0,1), point(-2,0)],
            'hyperbola': [point(1,0), point(-1,0), point(np.cosh(1), np.sinh(1))],
            'parabola':  [point(t*t, 2*t) for t in [0, 1, 2]],
            'line':      [point(2, y) for y in [0, 1, -1]],
        }
        for p in test_points[name]:
            assert _on_opns(p, C_opns), f"{name}: point not on OPNS"
            assert _on_ipns(p, C_dual), f"{name}: point not on dual(OPNS)"
