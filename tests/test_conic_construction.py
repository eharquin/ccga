"""
Conic construction theory — regression tests for the point→conic ladder.

Covers the findings established in notebook/conic_construction.ipynb:

  1. Same locus: q ∧ P5 = 0 ⟺ q ∧ C7 = 0 ⟺ q on the conic (the bare grade-5
     pentapole already IS the conic as a point-set).
  2. Role of Iod (the dual): dual(C7) = s (grade 1, clean coefficients);
     dual(P5) = −½ (s ∧ Iinfd) (grade 3, smeared by the infinity gauge).
  3/4. Conic type = incidence with the line at infinity; constructively set by
     how many true Veronese ideal points enter the build:
       0 → ellipse, 2 → hyperbola (asymptotes = the directions used),
       1 double (merging limit) → parabola (Δ → 0).
  5. point_at_infinity(v)  ≠  make_ideal_point(v): both on the conic, but only
     the former controls type/asymptotes.
"""
import numpy as np
import pytest

from ccga.algebra import Iod, Iinfd, I_inv
from ccga.point import point, point_at_infinity
from ccga.objects import (
    make_conic_pentapole, pentapole_to_conic, conic_dual_grade1,
    make_conic_opns, make_ideal_point,
    make_hyperbola_3points, make_parabola_3points,
)
from ccga.operations import dual, is_zero, grades
from ccga.classify import (
    ipns_to_coeffs, conic_subtype, conic_discriminant, conic_type,
    asymptotic_directions, classify, conic_center,
)
from ccga.objects import make_conic_ipns, make_ellipse, make_hyperbola

TOL = 1e-9


def _wedge(pts):
    R = pts[0]
    for p in pts[1:]:
        R = R ^ p
    return R


def _coeffs(pts):
    """(A,B,C,D,E,F) of the conic through the 5 given points/ideal points."""
    return ipns_to_coeffs(dual(_wedge(pts) ^ Iod))


FIVE = [point(0, 0), point(3, 0), point(0, 2), point(2, 2), point(-1, 1)]


# ══ grades ════════════════════════════════════════════════════════════════════

def test_pentapole_grades():
    P5, ipns3 = make_conic_pentapole(*FIVE)
    assert grades(P5) == [5]
    assert grades(ipns3) == [3]
    C7 = pentapole_to_conic(P5)
    assert grades(C7) == [7]
    assert grades(conic_dual_grade1(C7)) == [1]
    # pentapole_to_conic matches the direct 5-point OPNS constructor
    assert is_zero(C7 - make_conic_opns(FIVE))


# ══ finding 1 — same locus (incidence equivalence) ════════════════════════════

def test_incidence_equivalence_grade5_grade7():
    P5 = _wedge(FIVE)
    C7 = P5 ^ Iod
    # the 5 builder points lie on both
    assert all(is_zero(p ^ P5) for p in FIVE)
    assert all(is_zero(p ^ C7) for p in FIVE)

    A, B, C, D, E, F = ipns_to_coeffs(dual(C7))

    def on_conic_point(x):
        roots = np.roots([B, C*x + E, A*x*x + D*x + F])
        return [point(x, float(t.real)) for t in roots if abs(t.imag) < 1e-9]

    on = [q for x in np.linspace(-2, 4, 9) for q in on_conic_point(x)]
    assert len(on) >= 8
    for q in on:                                  # every on-conic point: both vanish
        assert is_zero(q ^ P5)
        assert is_zero(q ^ C7)
    for q in (point(1.0, 1.0), point(0.5, 0.5), point(5, 5)):   # off-conic: neither
        assert not is_zero(q ^ P5)
        assert not is_zero(q ^ C7)


# ══ finding 2 — the dual is what Iod changes ══════════════════════════════════

@pytest.mark.parametrize("seed", range(6))
def test_dual_pentapole_is_half_s_wedge_iinfd(seed):
    rng = np.random.default_rng(seed)
    pts = [point(*(rng.standard_normal(2) * 2)) for _ in range(5)]
    P5 = _wedge(pts)
    C7 = P5 ^ Iod
    s = dual(C7)                       # grade-1 clean conic vector
    assert grades(s) == [1]
    assert grades(dual(P5)) == [3]
    # dual(P5) == -1/2 (s ∧ Iinfd)
    assert is_zero(dual(P5) - (-0.5) * (s ^ Iinfd))


# ══ findings 3 & 4 — constructive conic-type ladder ═══════════════════════════

def test_zero_ideal_points_is_ellipse():
    A, B, C, D, E, F = _coeffs(FIVE)
    assert conic_discriminant(A, B, C) < -TOL
    assert conic_subtype(A, B, C, D, E, F) in ('ellipse', 'circle')


def test_two_ideal_points_is_hyperbola():
    for dirs in [((1, 0), (0, 1)), ((2, 1), (1, -1)), ((1, 2), (-1, 1))]:
        v1, v2 = dirs
        pts = [point(1, 0), point(2, 1), point(0, 3),
               point_at_infinity(*v1), point_at_infinity(*v2)]
        A, B, C, D, E, F = _coeffs(pts)
        assert conic_discriminant(A, B, C) > TOL
        assert conic_subtype(A, B, C, D, E, F) == 'hyperbola'


def test_asymptotic_directions_are_the_ideal_points_used():
    v1, v2 = (2, 1), (1, -1)
    pts = [point(1, 0), point(2, 1), point(0, 3),
           point_at_infinity(*v1), point_at_infinity(*v2)]
    A, B, C, D, E, F = _coeffs(pts)
    # the used directions are null directions of the quadratic form
    for (vx, vy) in (v1, v2):
        assert abs(A*vx*vx + C*vx*vy + B*vy*vy) < 1e-7
    # and asymptotic_directions recovers them (up to scale/sign)
    got = asymptotic_directions(make_conic_pentapole(*pts)[0])
    assert len(got) == 2

    def matches(v):
        vx, vy = v
        n = (vx*vx + vy*vy) ** 0.5
        return any(abs(abs(gx*vx/n + gy*vy/n) - 1) < 1e-6 for gx, gy in got)
    assert matches(v1) and matches(v2)


def test_parabola_as_merging_ideal_points_limit():
    fin = [point(0, 0), point(2, 0.5), point(1, 3)]
    deltas = []
    for eps in (0.3, 0.05, 0.005):
        A, B, C, _, _, _ = _coeffs(fin + [point_at_infinity(1, eps),
                                          point_at_infinity(1, -eps)])
        deltas.append(abs(conic_discriminant(A, B, C)))
    # Δ → 0 monotonically (parabola in the limit), scaling ~ eps²
    assert deltas[0] > deltas[1] > deltas[2]
    assert deltas[2] < 1e-5


# ══ finding 5 — point_at_infinity ≠ make_ideal_point ══════════════════════════

def test_vinf_vs_make_ideal_point_distinction():
    v1, v2 = (1, 1), (1, -1)
    fin = [point(0, 0), point(2, 0), point(0, 3)]

    s_vinf = dual(_wedge(fin + [point_at_infinity(*v1),
                                point_at_infinity(*v2)]) ^ Iod)
    s_ideal = dual(_wedge(fin + [make_ideal_point(*v1),
                                 make_ideal_point(*v2)]) ^ Iod)

    # both ideal points lie on their respective conic
    assert is_zero(point_at_infinity(*v1) | s_vinf)
    assert is_zero(make_ideal_point(*v1) | s_ideal)

    A1, B1, C1, *_ = ipns_to_coeffs(s_vinf)
    A2, B2, C2, *_ = ipns_to_coeffs(s_ideal)
    # different quadratic part → genuinely different conics
    assert abs(A1 - A2) + abs(B1 - B2) + abs(C1 - C2) > 1.0
    # only the true infinity point gives v1, v2 as asymptotes
    for (vx, vy) in (v1, v2):
        assert abs(A1*vx*vx + C1*vx*vy + B1*vy*vy) < 1e-7
    assert abs(A2*1*1 + C2*1*1 + B2*1*1) > 1e-3


# ══ 3-point conic constructions (ADVANCEMENT "General Forms from wedge") ═══════

def _on(p, C):
    return max((abs(float(v)) for v in (p ^ C).values()), default=0.0) < 1e-8


@pytest.mark.parametrize("d1,d2", [((1, 0), (0, 1)), ((1, 1), (1, -1)),
                                   ((2, 1), (1, -2)), ((3, 1), (-1, 2))])
def test_hyperbola_from_3points_and_directions(d1, d2):
    p1, p2, p3 = point(0, 0), point(2, 1), point(1, 3)
    H, _ = make_hyperbola_3points(p1, p2, p3, d1, d2)
    assert grades(H) == [7]
    assert conic_type(H) == 'hyperbola'
    assert all(_on(p, H) for p in (p1, p2, p3))          # passes through the 3 points
    # the two directions are recovered as asymptotes
    got = asymptotic_directions(H)
    assert len(got) == 2

    def matches(v):
        n = (v[0]**2 + v[1]**2) ** 0.5
        return any(abs(abs(g[0]*v[0]/n + g[1]*v[1]/n) - 1) < 1e-6 for g in got)
    assert matches(d1) and matches(d2)


@pytest.mark.parametrize("axis", [(1, 0), (0, 1), (1, 1), (2, 1), (3, -2)])
def test_tilted_parabola_from_3points(axis):
    p1, p2, p3 = point(0, 0), point(2, 1), point(1, 3)
    P, _ = make_parabola_3points(p1, p2, p3, axis)
    assert grades(P) == [7]
    A, B, C, D, E, F = ipns_to_coeffs(dual(P))
    assert abs(conic_discriminant(A, B, C)) < 1e-6      # Δ = 0 (parabola)
    assert conic_type(P) == 'parabola'
    assert all(_on(p, P) for p in (p1, p2, p3))
    # single (double) asymptote == the axis direction
    got = asymptotic_directions(P)
    assert len(got) == 1
    n = (axis[0]**2 + axis[1]**2) ** 0.5
    assert abs(abs(got[0][0]*axis[0]/n + got[0][1]*axis[1]/n) - 1) < 1e-6


# ══ conic_center (ADVANCEMENT "Conics properties") ════════════════════════════

@pytest.mark.parametrize("seed", range(6))
def test_conic_center_matches_algebra(seed):
    rng = np.random.default_rng(seed)
    A, B, C = rng.standard_normal(3)
    D, E, F = rng.standard_normal(3)
    if abs(A*B - C*C/4) < 1e-3:        # skip near-parabolic (non-central)
        return
    s = make_conic_ipns(A, B, C, D, E, F)
    cx_ga, cy_ga = conic_center(s)
    # algebraic center solves [2A C; C 2B][x;y] = [-D;-E]
    cx, cy = np.linalg.solve([[2*A, C], [C, 2*B]], [-D, -E])
    assert abs(cx_ga - cx) < 1e-9 and abs(cy_ga - cy) < 1e-9


def test_conic_center_on_named_conics():
    _, e = make_ellipse(3, 2, cx=1.5, cy=-0.5)
    assert np.allclose(conic_center(e), (1.5, -0.5), atol=1e-9)
    _, h = make_hyperbola(2, 1, cx=-2.0, cy=3.0)
    assert np.allclose(conic_center(h), (-2.0, 3.0), atol=1e-9)


# ══ the grade-7 conic as a versor:  C p ~C  ═══════════════════════════════════

def test_conic_versor_sandwich():
    """Sandwiching by the grade-7 conic C reduces to its grade-1 dual s and is
    the R^{5,3} reflection in the hyperplane ⟂ s:  C p ~C = s p s = 2(p·s)s − s²p.
    Points ON the conic (p·s=0) are fixed eigenvectors (eigenvalue −s²) — an
    alternative incidence test; off-conic points are pushed off the point variety.
    """
    C = make_conic_opns(FIVE)
    s = dual(C)
    s2 = float((s * s).e)

    # C p ~C == s p s == 2(p·s)s − s²p
    for q in (point(1.5, 0.5), point(5, 5), FIVE[0]):
        lhs = C * q * (~C)
        assert is_zero(lhs - s * q * s)
        ps = float((q | s).e)
        assert is_zero(lhs - (2*ps*s - s2*q))

    # incidence: on-conic points are fixed (∝ p, eigenvalue −s²); off-conic not
    for p in FIVE:
        assert is_zero(C * p * (~C) + s2 * p)        # fixed, eigenvalue −s²
    assert not is_zero(C * point(1.5, 0.5) * (~C) + s2 * point(1.5, 0.5))


# ══ classify integration ══════════════════════════════════════════════════════

def test_classify_routes_grade5_pentapole():
    assert classify(make_conic_pentapole(*FIVE)[0])['type'] == 'ellipse'
    H = make_conic_pentapole(point(0, 0), point(2, 0), point(0, 2),
                             point_at_infinity(1, 0), point_at_infinity(0, 1))[0]
    assert classify(H)['type'] == 'hyperbola'
    assert conic_type(H) == 'hyperbola'
    # grade-1 / grade-5 / grade-7 all agree
    assert conic_type(H) == conic_type(pentapole_to_conic(H)) == conic_type(dual(pentapole_to_conic(H)))


def test_ellipse_from_3points_imaginary_infinity_pair():
    # ellipse = 3 pts ∧ B ∧ Iod with B an IMAGINARY ideal-point pair (Δ<0)
    from ccga.objects import make_ellipse_3points
    from ccga.algebra import Iinfd, einf1, einf2, einf3
    from ccga.operations import is_zero
    p1, p2, p3 = point(0, 0), point(4, 0), point(2, 3)
    # a≠b → general ellipse
    opns, ipns = make_ellipse_3points(p1, p2, p3, a=3.0, b=2.0)
    assert grades(opns) == [7]
    assert conic_type(ipns) == 'ellipse'
    A, B, C, D, E, F = ipns_to_coeffs(ipns)
    assert conic_discriminant(A, B, C) < 0
    assert all(is_zero(p ^ opns) for p in (p1, p2, p3))   # passes through the 3 pts
    # a=b → the infinity pair is Iinfd (circular points) → a circle
    o2, i2 = make_ellipse_3points(p1, p2, p3, a=1.0, b=1.0)
    assert conic_type(i2) == 'circle'
    assert is_zero(((einf1 - einf2) ^ einf3) - Iinfd)        # B(1,1) = Iinfd
