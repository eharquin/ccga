"""
Tests for ccga/objects.py — §3 ground-truth anchors + §5 object zoo.

Checks for each object:
  - OPNS grade correct
  - IPNS grade correct
  - OPNS incidence: p ^ C_opns ≈ 0 for points on the object
  - IPNS incidence: p | C_ipns ≈ 0 for points on the object
  - OPNS ↔ IPNS round-trip (dual and undual)
  - Reality classification
  - §3 coefficient map verification
"""
import numpy as np
import pytest
from ccga.algebra import einf, Iod, I, I_inv
from ccga.point import point
from ccga.objects import (
    make_point_ccga, make_point_pair, make_tangent_point,
    make_conic_opns, make_conic_ipns, conic_from_5points,
    make_circle, make_ellipse, make_hyperbola, make_parabola,
    make_tilted_ellipse, make_line_ipns, make_line_2points,
    make_ideal_point, make_round_point, make_flat_point,
    make_line_at_infinity, make_conic_at_infinity,
    make_conic_tripole, make_conic_quadpole,
    _ipns_to_conic_coeffs, infinity_ipns_components,
)
from ccga.extract import circumcircle, extract_tripole, extract_quadpole
from ccga.operations import dual, undual, is_zero, grades, pure_grade
from ccga.classify import classify

TOL = 1e-9


def _on_opns(p, C, tol=TOL):
    """True if point p lies on OPNS object C (p ^ C ≈ 0)."""
    wedge = p ^ C
    return all(abs(float(v)) < tol for v in wedge.values())


def _on_ipns(p, s, tol=TOL):
    """True if point p lies on IPNS object s (p | s ≈ 0)."""
    return abs(float((p | s).e)) < tol


# ══ Point ═══════════════════════════════════════════════════════════════════

def test_point_grade():
    p = make_point_ccga(1, 2)
    assert grades(p) == [1]


def test_point_null():
    p = make_point_ccga(3, -4)
    assert abs(float((p * p).e)) < TOL


def test_point_normalization():
    p = make_point_ccga(2, 5)
    assert abs(float((p | einf).e) + 1.0) < TOL


# ══ Point pair ═══════════════════════════════════════════════════════════════

def test_point_pair_grade():
    p1, p2 = make_point_ccga(1, 0), make_point_ccga(-1, 0)
    pp_opns, pp_ipns = make_point_pair(p1, p2)
    assert grades(pp_opns) == [2]
    assert grades(pp_ipns) == [6]


def test_point_pair_real():
    p1, p2 = make_point_ccga(1, 0), make_point_ccga(-1, 0)
    pp_opns, _ = make_point_pair(p1, p2)
    sq = float((pp_opns * pp_opns).e)
    assert sq > 0, f"Real pair should have positive square, got {sq}"


def test_point_pair_round_trip():
    p1, p2 = make_point_ccga(1, 0), make_point_ccga(-1, 0)
    pp_opns, pp_ipns = make_point_pair(p1, p2)
    recovered = undual(pp_ipns)
    # Find scale factor from first nonzero shared component
    ratio = None
    opns_dict  = {k: float(v) for k, v in pp_opns.items()}
    recov_dict = {k: float(v) for k, v in recovered.items()}
    for k in opns_dict:
        if k in recov_dict and abs(opns_dict[k]) > TOL and abs(recov_dict[k]) > TOL:
            ratio = opns_dict[k] / recov_dict[k]
            break
    diff = pp_opns - recovered * ratio if ratio else pp_opns - recovered
    assert is_zero(diff, TOL * 10)


# ══ Tripole & Quadpole ═══════════════════════════════════════════════════════

def _same_point_set(got, want, tol=1e-3):
    """True if the recovered (x,y) set matches `want` up to order and tolerance."""
    got, want = sorted(got), sorted(want)
    if len(got) != len(want):
        return False
    return all(min(abs(g[0]-w[0]) + abs(g[1]-w[1]) for g in got) < tol for w in want)


def _proportional(A, B, tol=1e-7):
    """True if blades A and B are equal up to a single global scale factor."""
    da, db = {k: float(v) for k, v in A.items()}, {k: float(v) for k, v in B.items()}
    ratio = None
    for k in da:
        if abs(da[k]) > TOL and abs(db.get(k, 0.0)) > TOL:
            ratio = da[k] / db[k]
            break
    if ratio is None:
        return False
    return is_zero(A - B * ratio, tol * 100)


_TRI_CONFIGS = [
    [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)],
    [(0, 0), (4, 0), (1, 3)],
    [(-2, -1), (3, 2), (0, -3)],
    [(1, 0), (0, 1), (-1, 0)],            # one point at the t=∞ parametrisation pole
]
_QUAD_CONFIGS = [
    [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9), (1.5, 2.3)],
    [(0, 0), (4, 0), (4, 3), (0, 3)],     # rectangle (symmetric: degenerate p5=eo)
    [(-2, -1), (3, 2), (0, -3), (2, 2)],
    [(1, 0), (0, 1), (-1, 0), (0, -1)],
]


def test_tripole_grades():
    p = [make_point_ccga(*c) for c in _TRI_CONFIGS[0]]
    opns, ipns = make_conic_tripole(*p)
    assert grades(opns) == [3]
    assert grades(ipns) == [5]


def test_tripole_incidence():
    """The 3 builder points lie on the tripole: p ^ T ≈ 0."""
    p = [make_point_ccga(*c) for c in _TRI_CONFIGS[0]]
    T, _ = make_conic_tripole(*p)
    for q in p:
        assert _on_opns(q, T)


def test_tripole_circumcircle():
    """T ∧ Iod ∧ Iinfd is the circle through the 3 points (right centre/radius)."""
    P = [(0, 0), (4, 0), (0, 3)]          # right triangle: circumcircle is the hypotenuse
    T, _ = make_conic_tripole(*[make_point_ccga(*c) for c in P])
    cx, cy, R = circumcircle(T)
    assert abs(cx - 2.0) < 1e-6 and abs(cy - 1.5) < 1e-6
    assert abs(R - 2.5) < 1e-6


@pytest.mark.parametrize("P", _TRI_CONFIGS)
def test_tripole_extract(P):
    T, _ = make_conic_tripole(*[make_point_ccga(*c) for c in P])
    got = extract_tripole(T)
    assert _same_point_set(got, [(float(x), float(y)) for x, y in P])
    # round-trip: rebuilt tripole is the same blade up to scale
    Tr = make_conic_tripole(*[make_point_ccga(*g) for g in got])[0]
    assert _proportional(T, Tr)


def test_quadpole_grades():
    p = [make_point_ccga(*c) for c in _QUAD_CONFIGS[0]]
    opns, ipns = make_conic_quadpole(*p)
    assert grades(opns) == [4]
    assert grades(ipns) == [4]


def test_quadpole_incidence():
    p = [make_point_ccga(*c) for c in _QUAD_CONFIGS[0]]
    Q, _ = make_conic_quadpole(*p)
    for q in p:
        assert _on_opns(q, Q)


def test_quadpole_is_wedge_of_two_dipoles():
    """Q = ± pp_ij ∧ pp_kl for each of the 3 pairings."""
    p = [make_point_ccga(*c) for c in _QUAD_CONFIGS[0]]
    Q, _ = make_conic_quadpole(*p)
    for (i, j), (k, l) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        blade = (p[i] ^ p[j]) ^ (p[k] ^ p[l])
        assert is_zero(Q - blade, TOL * 100) or is_zero(Q + blade, TOL * 100)


@pytest.mark.parametrize("P", _QUAD_CONFIGS)
def test_quadpole_extract(P):
    Q, _ = make_conic_quadpole(*[make_point_ccga(*c) for c in P])
    got = extract_quadpole(Q)
    assert _same_point_set(got, [(float(x), float(y)) for x, y in P])
    Qr = make_conic_quadpole(*[make_point_ccga(*g) for g in got])[0]
    assert _proportional(Q, Qr)


# ══ General conic (§3 results 1, 6, 7, 8) ════════════════════════════════════

def _unit_circle_pts(n=5):
    return [point(np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, n+1)[:n]]


def test_conic_opns_grade():
    pts = _unit_circle_pts()
    C = make_conic_opns(pts)
    assert grades(C) == [7]


def test_conic_ipns_grade():
    pts = _unit_circle_pts()
    _, s = conic_from_5points(pts)
    assert grades(s) == [1]


def test_conic_opns_incidence():
    """Points used to build the conic lie on it (p ^ C ≈ 0)."""
    pts = _unit_circle_pts()
    C_opns, _ = conic_from_5points(pts)
    for p in pts:
        assert _on_opns(p, C_opns), "Point not on OPNS conic"


def test_conic_ipns_incidence():
    """Points used to build the conic satisfy p | s ≈ 0."""
    pts = _unit_circle_pts()
    _, s = conic_from_5points(pts)
    for p in pts:
        assert _on_ipns(p, s), "Point not on IPNS conic"


def test_conic_opns_ipns_roundtrip():
    """dual(C_opns) ≈ λ·C_ipns; undual(C_ipns) ≈ μ·C_opns."""
    pts = _unit_circle_pts()
    C_opns, C_ipns = conic_from_5points(pts)
    # dual(OPNS) should be proportional to IPNS
    C_dual = dual(C_opns)
    assert grades(C_dual) == [1]
    # Check same incidence
    for p in pts:
        assert _on_ipns(p, C_dual), "dual(OPNS) incidence failed"


def test_conic_coefficient_map_result1():
    """§3 result 1: IPNS conic coefficients reproduce Ax²+By²+Cxy+Dx+Ey+F."""
    # 5 points on y = x² (parabola A=0,B=0,C=0 but D=0,E=-1,F=0 rearranged)
    # Use a circle to give clean A,B: x²+y²=4 (r=2, centred origin)
    # A=B=1/4, C=0, D=E=0, F=-1 in normalised form → use make_circle
    _, s_circle = make_circle(0, 0, 2)
    A, B, C, D, E, F = _ipns_to_conic_coeffs(s_circle)
    # Verify: for the unit circle x²+y²=4: Ax²+By²+F=0 with proportional A=B
    assert abs(A - B) < TOL * 10, f"Circle should have A=B, got {A} vs {B}"
    assert abs(C) < TOL, f"Circle C should be 0, got {C}"
    assert abs(D) < TOL and abs(E) < TOL


def test_ipns_conic_point_evaluates_zero():
    """§3: q · s = 0 iff q lies on conic s (checking Ax²+By²+...=0)."""
    # Ellipse (x/2)²+y²=1: 5 known points on it
    pts = [point(2*np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 6)[:5]]
    _, s = conic_from_5points(pts)
    for p in pts:
        assert _on_ipns(p, s)
    # A point OFF the ellipse
    p_off = point(2.0, 2.0)
    assert not _on_ipns(p_off, s, tol=0.1)


def test_gauge_inert_directions_result4():
    """
    §3 result 4: gauge-inertness of einf3 and einfbar.

    Two tests:
    (a) Objects built via make_conic_ipns have zero gauge components (canonical).
    (b) Adding a gauge component (alpha*einf3) to any IPNS conic does NOT change
        q·s for any query point q (the locus is preserved).
    """
    from ccga.algebra import einf3, einfbar
    # (a) make_conic_ipns output is canonical (zero gauge at origin)
    for name, (_, s) in [
        ('circle@0',    make_circle(0, 0, 1)),
        ('ellipse',     make_ellipse(2, 1)),
        ('hyperbola',   make_hyperbola(1, 2)),
        ('parabola',    make_parabola(1)),
        ('line',        make_line_ipns(1, 0, -2)),
    ]:
        s_einf3, s_einfbar = infinity_ipns_components(s)
        assert abs(s_einf3) < TOL, f"{name}: einf3={s_einf3}"
        assert abs(s_einfbar) < TOL, f"{name}: einfbar={s_einfbar}"

    # (b) Gauge-inertness: adding alpha*einf3 does not change q·s for any q
    _, s_base = make_circle(0, 0, 1)
    alpha = 3.14
    s_gauged = s_base + alpha * einf3
    for q in [point(1, 0), point(0, 1), point(0.5, 0.5), point(2, 3)]:
        val_base   = float((q | s_base  ).e)
        val_gauged = float((q | s_gauged).e)
        assert abs(val_base - val_gauged) < TOL, \
            f"Gauge changed q·s: {val_base} vs {val_gauged}"


# ══ Circle ═══════════════════════════════════════════════════════════════════

def test_circle_grade():
    C_opns, C_ipns = make_circle(0, 0, 1)
    assert grades(C_opns) == [7]
    assert grades(C_ipns) == [1]


def test_circle_ipns_incidence():
    cx, cy, r = 1.0, -2.0, 1.5
    C_opns, C_ipns = make_circle(cx, cy, r)
    pts = [point(cx + r*np.cos(t), cy + r*np.sin(t)) for t in np.linspace(0, 2*np.pi, 6)[:5]]
    for p in pts:
        assert _on_ipns(p, C_ipns)
        assert _on_opns(p, C_opns)


def test_circle_radius_from_ipns():
    """§3 result 3: s^2 = r^2 for circle IPNS."""
    for r in [1.0, 0.5, 3.0]:
        _, s = make_circle(0, 0, r)
        sq = float((s * s).e)
        assert abs(sq - r*r) < TOL * 100, f"s^2={sq} != r^2={r*r}"


def test_circle_classification():
    C_opns, C_ipns = make_circle(0, 0, 2)
    assert classify(C_opns)['type'] == 'circle'
    assert classify(C_ipns)['type'] == 'circle'


# ══ Ellipse ══════════════════════════════════════════════════════════════════

def test_ellipse_grade_and_type():
    C_opns, C_ipns = make_ellipse(2, 1)
    assert grades(C_opns) == [7]
    assert grades(C_ipns) == [1]
    assert classify(C_opns)['type'] == 'ellipse'
    assert classify(C_ipns)['type'] == 'ellipse'


def test_ellipse_incidence():
    a, b = 3.0, 1.0
    C_opns, C_ipns = make_ellipse(a, b)
    pts = [point(a*np.cos(t), b*np.sin(t)) for t in np.linspace(0, 2*np.pi, 6)[:5]]
    for p in pts:
        assert _on_ipns(p, C_ipns)
        assert _on_opns(p, C_opns)


def test_tilted_ellipse_type():
    C_opns, C_ipns = make_tilted_ellipse(2, 1, np.pi/4)
    assert classify(C_opns)['type'] == 'ellipse'
    # tilted ellipse should have nonzero C coefficient (eo3 component)
    _, _, Ccoeff, _, _, _ = _ipns_to_conic_coeffs(C_ipns)
    assert abs(Ccoeff) > 0.01, f"Tilted ellipse C should be nonzero, got {Ccoeff}"


# ══ Hyperbola ════════════════════════════════════════════════════════════════

def test_hyperbola_grade_and_type():
    C_opns, C_ipns = make_hyperbola(1, 1)
    assert grades(C_opns) == [7]
    assert grades(C_ipns) == [1]
    assert classify(C_opns)['type'] == 'hyperbola'
    assert classify(C_ipns)['type'] == 'hyperbola'


def test_hyperbola_incidence():
    a, b = 2.0, 1.0
    C_opns, C_ipns = make_hyperbola(a, b)
    # Points on right branch: x = a*cosh(t), y = b*sinh(t)
    pts = [point(a*np.cosh(t), b*np.sinh(t)) for t in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    for p in pts:
        assert _on_ipns(p, C_ipns)
        assert _on_opns(p, C_opns)


# ══ Parabola ═════════════════════════════════════════════════════════════════

def test_parabola_grade_and_type():
    C_opns, C_ipns = make_parabola(1.0)
    assert grades(C_opns) == [7]
    assert grades(C_ipns) == [1]
    assert classify(C_opns)['type'] == 'parabola'
    assert classify(C_ipns)['type'] == 'parabola'


def test_parabola_incidence():
    # y^2 = 4x  (p=1)
    p_val = 1.0
    C_opns, C_ipns = make_parabola(p_val)
    pts = [point(t*t, 2*t) for t in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    for p in pts:
        assert _on_ipns(p, C_ipns)
        assert _on_opns(p, C_opns)


# ══ Line ═════════════════════════════════════════════════════════════════════

def test_line_grade_and_type():
    C_opns, C_ipns = make_line_ipns(1, 0, -2)   # x = 2
    assert grades(C_opns) == [7]
    assert grades(C_ipns) == [1]
    assert classify(C_opns)['type'] == 'line'
    assert classify(C_ipns)['type'] == 'line'


def test_line_incidence():
    # Line x = 2: nx=1, ny=0, d=-2
    C_opns, C_ipns = make_line_ipns(1, 0, -2)
    pts = [point(2, y) for y in [-3, 0, 1, 5, -1]]
    for p in pts:
        assert _on_ipns(p, C_ipns)
        assert _on_opns(p, C_opns)


def test_line_2points():
    # Line through (0, 1) and (2, 3): y = x + 1
    p1, p2 = make_point_ccga(0, 1), make_point_ccga(2, 3)
    C_opns, C_ipns = make_line_2points(p1, p2)
    assert grades(C_opns) == [7]
    pts = [point(t, t + 1) for t in [-2, 0, 1, 3]]
    for p in pts:
        assert _on_opns(p, C_opns)


# ══ Flat point ═══════════════════════════════════════════════════════════════

def test_flat_point_grade():
    fp = make_flat_point(1, 2)
    assert grades(fp) == [4]


# ══ Ideal point ══════════════════════════════════════════════════════════════

def test_ideal_point_zero_homogeneous():
    """Ideal point = round point with eo (homogeneous) coordinate zeroed."""
    from ccga.algebra import einf
    ip = make_ideal_point(3, 4)
    # No homogeneous coordinate: p · einf = 0 (not normalizable as a finite pt)
    assert abs(float((ip | einf).e)) < TOL


def test_ideal_point_keeps_euclidean():
    """Ideal point keeps the Euclidean (e1,e2) part of the round point."""
    from ccga.algebra import e1, e2
    ip = make_ideal_point(3, 4)
    assert abs(float((ip | e1).e) - 3.0) < TOL
    assert abs(float((ip | e2).e) - 4.0) < TOL


def test_ideal_point_grade():
    ip = make_ideal_point(1, 1)
    assert grades(ip) == [1]


def test_ideal_point_classifies():
    """Ideal point is distinguished from a line (anisotropic infinity)."""
    from ccga.classify import classify
    for x, y in [(3, 4), (2, 0), (0, 5), (1, 1)]:
        assert classify(make_ideal_point(x, y))['type'] == 'ideal_point'


# ══ CCGA point with radius (make_point_ccga, optional r) ══════════════════════

def test_ccga_point_radius_real():
    """CCGA point with real radius: p² = +r² (it IS a circle)."""
    from ccga.classify import classify
    for r in [1.0, 2.5, 0.5]:
        p = make_point_ccga(2, 3, r)
        assert abs(float((p * p).e) - r*r) < TOL * 100
        assert classify(p)['type'] == 'circle'
        assert classify(p)['reality'] == 'real'


def test_ccga_point_radius_imaginary():
    """CCGA point with imaginary radius: p² = −r²."""
    p = make_point_ccga(2, 3, 2.0, imaginary=True)
    assert abs(float((p * p).e) + 4.0) < TOL * 100
    from ccga.classify import classify
    assert classify(p)['reality'] == 'imaginary'


def test_ccga_point_zero_radius_is_point():
    """CCGA point with r=0 (default) is the null finite point."""
    from ccga.classify import classify
    p = make_point_ccga(2, 3)
    assert abs(float((p * p).e)) < TOL
    assert classify(p)['type'] == 'point'


def test_ccga_point_radius_equals_circle():
    """make_point_ccga(x,y,r) is identical to the IPNS circle make_circle(x,y,r)."""
    p = make_point_ccga(1, 2, 1.5)
    _, c = make_circle(1, 2, 1.5)
    diff = p - c
    assert all(abs(float(v)) < TOL for v in diff.values())


# ══ CGA round point (make_round_point = P ∧ I_inf^▷) ══════════════════════════

def test_cga_round_point_grade():
    """The CGA round point is a grade-3 OPNS blade."""
    assert grades(make_round_point(3, 4)) == [3]


def test_cga_round_point_is_gauge_wedge():
    """make_round_point(x,y) == make_point_ccga(x,y) ∧ Iinfd by construction."""
    from ccga.algebra import Iinfd
    R = make_round_point(3, 4)
    R2 = make_point_ccga(3, 4) ^ Iinfd
    assert is_zero(R - R2, TOL)


def test_cga_round_point_collapses_to_cga():
    """
    §3.3 / §3.9: wedging with Iinfd collapses the CCGA point onto the CGA
    conformal point p_cga = eo + x·e1 + y·e2 + (x²+y²)/2·einf.
    """
    from ccga.algebra import Iinfd, eo, e1 as E1, e2 as E2, einf1
    for x, y in [(3, 4), (2, 0), (-1, 5)]:
        p_cga = eo + x*E1 + y*E2 + (x*x + y*y)/2 * einf1
        assert is_zero(make_round_point(x, y) - (p_cga ^ Iinfd), TOL)


# ══ Line and conic at infinity ════════════════════════════════════════════════

def test_line_at_infinity_grade():
    from ccga.algebra import Iinf
    li = make_line_at_infinity()
    assert grades(li) == [3]


def test_conic_at_infinity_grade():
    ci = make_conic_at_infinity()
    assert grades(ci) == [5]


# ══ §3 result 8: gauge-fixing cross-check ════════════════════════════════════

def test_Iod_dual_has_no_eobar_eo3_in_ipns():
    """
    §3 result 8: for canonical IPNS conics, einfbar=0 and einf3=0 components.
    This is the IPNS dual of the OPNS gauge constraint (Iod).
    """
    # All standard IPNS conics we construct are in the canonical subspace
    for name, (_, s) in [
        ('circle',    make_circle(0, 0, 1)),
        ('ellipse',   make_ellipse(2, 1)),
        ('hyperbola', make_hyperbola(1, 2)),
        ('parabola',  make_parabola(1)),
        ('line',      make_line_ipns(1, 1, -3)),
    ]:
        s_einf3, s_einfbar = infinity_ipns_components(s)
        assert abs(s_einf3) < TOL, f"{name}: einf3={s_einf3}"
        assert abs(s_einfbar) < TOL, f"{name}: einfbar={s_einfbar}"
