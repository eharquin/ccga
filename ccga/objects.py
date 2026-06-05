"""
CCGA geometric object constructors.

Each constructor returns an (opns, ipns) pair where applicable, or just the
canonical form.  Grade assignments follow §3 and §5 of CLAUDE.md.

Dual convention throughout: ipns = opns * I_inv  (right-multiply).

── Object inventory ────────────────────────────────────────────────────────
  CCGA point         grade-1 OPNS  (make_point_ccga; optional radius r,
                     r=0 → null point, r≠0 → round object, p²=±r²)
  CGA round point    grade-3 OPNS  (make_round_point = P ∧ I_inf^▷; collapses
                     the two CCGA quadratic coords into the single isotropic
                     CGA radius term — §3.3 / §3.9)
  PointPair          grade-2 OPNS, grade-6 IPNS
  TangentPoint       grade-2 OPNS null (coincident limit)
  GeneralConic       grade-7 OPNS, grade-1 IPNS
  Circle             grade-7 OPNS, grade-1 IPNS  (sub-family of conic)
  Ellipse            grade-7 OPNS, grade-1 IPNS
  Hyperbola          grade-7 OPNS, grade-1 IPNS
  Parabola           grade-7 OPNS, grade-1 IPNS
  Line               grade-7 OPNS, grade-1 IPNS  (degenerate conic)
  FlatPoint          grade-4 OPNS
  IdealPoint         grade-1 OPNS  (direction at infinity)
  LineAtInfinity     grade-5 OPNS
  ConicAtInfinity    grade-5 OPNS
"""

import numpy as np
from .algebra import (
    alg, e1, e2,
    eo, einf, eo1, eo2, eo3, einf1, einf2, einf3,
    eobar, einfbar, Iod, Iinfd, Io, Iinf, Ieps, I, I_inv,
)
from .point import point as _point
from .operations import dual, meet, join, is_zero, grades, pure_grade


# ── helpers ──────────────────────────────────────────────────────────────────

def _coeff(mv, key):
    """Return float coefficient of orthogonal basis key (e.g. 'e3') in mv."""
    for k, v in mv.items():
        if alg.bin2canon.get(k, '') == key:
            return float(v)
    return 0.0


def _ipns_to_conic_coeffs(s):
    """
    Extract (A,B,C,D,E,F) from IPNS grade-1 conic s stored in orthogonal basis.

    make_conic_ipns stores (in orthogonal keys e1..e8):
      e1:D  e2:E  e3:-2A+F/4  e4:-2B+F/4  e5:-C  e6:-2A-F/4  e7:-2B-F/4
    Inverse:
      A = -(c3+c6)/4,  B = -(c4+c7)/4,  C = -c5,  D=c1, E=c2, F=2*(c3-c6)
    """
    c1=_coeff(s,'e1'); c2=_coeff(s,'e2')
    c3=_coeff(s,'e3'); c4=_coeff(s,'e4'); c5=_coeff(s,'e5')
    c6=_coeff(s,'e6'); c7=_coeff(s,'e7'); c8=_coeff(s,'e8')
    # C uses symmetric (c5+c8)/2 since eo3 and einf3 share e5,e8 (see classify).
    A = -(c3+c6)/4;  B = -(c4+c7)/4;  C = -(c5+c8)/2
    D = c1;  E = c2;  F = (c3-c6)+(c4-c7)
    return A, B, C, D, E, F


# ══ Round / point objects ════════════════════════════════════════════════════

def make_point_ccga(x, y, r=0.0, imaginary=False):
    """
    CCGA point representing (x,y), grade-1.  Radius is optional:

      r=0 (default) → null point, p²=0, p·einf=-1.
      r≠0           → round object (CGA-style circle), p²=+r² (real) or
                      p²=−r² (imaginary=True).

    This is the single CCGA base-point constructor — there is no separate
    "round point" in CCGA; the radius is just an optional argument.  The name
    "round point" is reserved for the CGA round point (see make_round_point).
    """
    return _point(x, y, r, imaginary)


def make_point_pair(p1, p2):
    """
    Grade-2 OPNS blade (point pair / dipole).

    OPNS:  P = p1 ^ p2   (grade 2)
    IPNS:  P* = dual(P)  (grade 6)
    Reality: P^2 > 0 → real pair; P^2 < 0 → imaginary pair.
    """
    opns = p1 ^ p2
    ipns = dual(opns)
    return opns, ipns


def make_conic_tripole(p1, p2, p3):
    """
    Grade-3 OPNS blade (conic tripole) — the wedge of three CCGA points.

    OPNS:  T = p1 ^ p2 ^ p3   (grade 3)
    IPNS:  T* = dual(T)        (grade 5)

    Unlike CGA's p1∧p2∧p3 (merely the circle through them), the CCGA tripole
    retains the three points: p ^ T = 0 has exactly the 3 points as solutions.
    Recover them with ccga.extract.extract_tripole (circumcircle + cubic +
    Cardano — no single ±√ exists for 3 points).
    """
    opns = p1 ^ p2 ^ p3
    ipns = dual(opns)
    return opns, ipns


def make_conic_quadpole(p1, p2, p3, p4):
    """
    Grade-4 OPNS blade (conic quadpole) — the wedge of four CCGA points.

    OPNS:  Q = p1 ^ p2 ^ p3 ^ p4   (grade 4)
    IPNS:  Q* = dual(Q)            (grade 4)

    A quadpole is a wedge of two dipoles, Q = pp_ij ∧ pp_kl, for each of the 3
    pairings.  Recover the four points with ccga.extract.extract_quadpole
    (GA pencil + resolvent cubic + two ±√ dipoles — Ferrari's quartic).
    """
    opns = p1 ^ p2 ^ p3 ^ p4
    ipns = dual(opns)
    return opns, ipns


def make_tangent_point(p, tangent_dir):
    """
    Degenerate (null) point pair: coincident-point limit.

    Constructed as p ^ t  where t is a null tangent direction (p itself works
    if we perturb by an ideal vector in the tangent direction).  The result
    satisfies (p^t)^2 = 0.

    tangent_dir should be a null vector (e.g. an ideal point direction).
    """
    opns = p ^ tangent_dir
    ipns = dual(opns)
    return opns, ipns


# ══ Conic objects ════════════════════════════════════════════════════════════

def make_conic_opns(pts):
    """
    Grade-7 OPNS conic from exactly 5 points (result 6).

      C_opns = Iod ^ p1 ^ p2 ^ p3 ^ p4 ^ p5

    Returns the grade-7 blade.
    """
    if len(pts) != 5:
        raise ValueError("Exactly 5 points required for OPNS conic")
    C = Iod
    for p in pts:
        C = C ^ p
    return C


def make_conic_ipns(A, B, C, D, E, F):
    """
    Grade-1 IPNS conic from coefficients Ax²+By²+Cxy+Dx+Ey+F=0 (result 1).

      s_{o1} = -2A,  s_{o2} = -2B,  s_{o3} = -C
      s_{e1} = D,    s_{e2} = E
      s_{inf1} = s_{inf2} = -F/2   (canonical: equal split, s_{infbar}=0)
    """
    return (-2*A)*eo1 + (-2*B)*eo2 + (-C)*eo3 + D*e1 + E*e2 + (-F/2)*(einf1 + einf2)


def conic_from_5points(pts):
    """
    Build both OPNS and IPNS representations of the conic through 5 points.

    Returns (opns_grade7, ipns_grade1).
    """
    opns = make_conic_opns(pts)
    ipns = dual(opns)
    return opns, ipns


def make_circle(cx, cy, r):
    """
    Circle centred at (cx,cy) with radius r (§3 result 3).

    IPNS:  s = p_centre - (r²/2)·einf   (grade 1)
    The circle is the special conic with A=B=-1/2, C=0 after normalization,
    i.e. x²+y²−2cx·x−2cy·y+(cx²+cy²−r²) = 0.

    Canonical IPNS: s_{eobar}=0, s_{eo3}=0  (isotropic: same scale on eo1,eo2).
    OPNS: 5-point construction with 5 points on the circle.
    """
    centre = _point(cx, cy)
    ipns = centre - (r*r/2) * einf
    opns = dual(ipns)
    return opns, ipns


def make_ellipse(a, b, cx=0.0, cy=0.0):
    """
    Axis-aligned ellipse centred at (cx,cy):  (x-cx)²/a² + (y-cy)²/b² = 1.

    Expanded: x²/a² + y²/b² - 2cx/a²·x - 2cy/b²·y + (cx²/a²+cy²/b²-1) = 0

    IPNS grade-1.  Distinct diagonal:  a≠b ↔ eobar component ≠ 0.
    """
    A = 1/(a*a);  B = 1/(b*b);  C = 0.0
    D = -2*cx*A;  E = -2*cy*B
    F = cx*cx*A + cy*cy*B - 1.0
    ipns = make_conic_ipns(A, B, C, D, E, F)
    opns = dual(ipns)
    return opns, ipns


def make_hyperbola(a, b, cx=0.0, cy=0.0):
    """
    Axis-aligned hyperbola centred at (cx,cy): (x-cx)²/a² - (y-cy)²/b² = 1.

    A = 1/a², B = -1/b²,  C = 0.
    """
    A = 1/(a*a);  B = -1/(b*b);  C = 0.0
    D = -2*cx*A;  E = -2*cy*B
    F = cx*cx*A + cy*cy*B - 1.0
    ipns = make_conic_ipns(A, B, C, D, E, F)
    opns = dual(ipns)
    return opns, ipns


def make_parabola(p_focus, axis='x'):
    """
    Axis-aligned parabola with parameter p_focus (distance focus–directrix):
      y² = 4·p·x  (axis='x')  or  x² = 4·p·y  (axis='y').

    For axis='x': A=0, B=1, C=0, D=-4p, E=0, F=0.
    Degenerate: A=0 (no x² term) — one ideal point, tangent to line at infinity.
    """
    p = p_focus
    if axis == 'x':
        A, B, C, D, E, F = 0.0, 1.0, 0.0, -4*p, 0.0, 0.0
    else:
        A, B, C, D, E, F = 1.0, 0.0, 0.0, 0.0, -4*p, 0.0
    ipns = make_conic_ipns(A, B, C, D, E, F)
    opns = dual(ipns)
    return opns, ipns


def make_tilted_ellipse(a, b, theta, cx=0.0, cy=0.0):
    """
    Ellipse rotated by angle theta (uses non-zero C = e_{o3} term).

    Ax²+By²+Cxy+Dx+Ey+F=0 via rotation of axes.
    """
    cos2 = np.cos(theta)**2;  sin2 = np.sin(theta)**2
    sc   = np.sin(theta)*np.cos(theta)
    A = cos2/(a*a) + sin2/(b*b)
    B = sin2/(a*a) + cos2/(b*b)
    C = 2*sc*(1/(a*a) - 1/(b*b))
    # translate centre
    D = -2*(A*cx + C/2*cy)
    E = -2*(B*cy + C/2*cx)
    F = A*cx*cx + B*cy*cy + C*cx*cy - 1.0
    ipns = make_conic_ipns(A, B, C, D, E, F)
    opns = dual(ipns)
    return opns, ipns


# ── Line as degenerate conic ─────────────────────────────────────────────────

def make_line_ipns(nx, ny, d):
    """
    Finite line  nx·x + ny·y + d = 0  as a degenerate IPNS conic.

    A=B=C=0 (no quadratic part) → grade-1 vector in the linear subspace.
    IPNS: s = D·e1 + E·e2 + F·einf   with D=nx, E=ny, F=d
    (using F = -(s_inf1 + s_inf2) → s_inf1=s_inf2=-d/2).
    """
    ipns = make_conic_ipns(0.0, 0.0, 0.0, nx, ny, d)
    opns = dual(ipns)
    return opns, ipns


def make_line_2points(p1, p2):
    """
    Line through two points as a degenerate OPNS conic.  IPNS via dual.

    Extracts the line equation from (x1,y1) and (x2,y2), then delegates to
    make_line_ipns — avoids the collinear 5-point degeneracy.
    """
    x1 = float((p1 | e1).e);  y1 = float((p1 | e2).e)
    x2 = float((p2 | e1).e);  y2 = float((p2 | e2).e)
    # Line: (y2-y1)*x + (x1-x2)*y + (x2*y1 - x1*y2) = 0
    nx = y2 - y1;  ny = x1 - x2;  d = x2*y1 - x1*y2
    norm = (nx*nx + ny*ny)**0.5
    if norm < 1e-12:
        raise ValueError("The two points are identical; cannot define a line.")
    return make_line_ipns(nx/norm, ny/norm, d/norm)


# ══ Flat / ideal elements ════════════════════════════════════════════════════

def make_ideal_point(x, y, r=0.0, imaginary=False):
    """
    Ideal (round) point — a round point with its homogeneous coordinate eo
    zeroed, exactly as in CGA:

        p_inf = x·e1 + y·e2 + (x²/2)·einf1 + (y²/2)·einf2 + xy·einf3 ∓ (r²/2)·einf

    (the round point p(x,y,r) minus its eo part).  Grade 1.  Because it has no
    eo component, p_inf · einf = 0 (it is not normalizable as a finite point).
    """
    sign = +1.0 if imaginary else -1.0
    base = (x*e1 + y*e2
            + (x*x/2)*einf1 + (y*y/2)*einf2 + x*y*einf3)
    if r:
        base = base + sign * (r*r/2) * einf
    return base


def make_round_point(x, y, r=0.0, imaginary=False):
    """
    CGA round point, grade-3 OPNS:  R = p ∧ I_inf^▷  (Iinfd).

    Wedging a CCGA point with the infinity-gauge blade
    I_inf^▷ = (einf1−einf2) ∧ einf3 annihilates the two conic-specific infinity
    directions (einfbar and einf3), collapsing the CCGA point's separate
    quadratic coordinates  x²/2·einf1 + y²/2·einf2 + xy·einf3  into the single
    isotropic CGA term (x²+y²)/2 (§3.3 "one isotropic radius", §3.9 GAC map).

    Equivalently  R = p_cga ∧ I_inf^▷  with the standard CGA conformal point
        p_cga = eo + x·e1 + y·e2 + (x²+y²)/2 · einf.
    With r≠0 the isotropic term becomes ((x²+y²)−r²)/2, i.e. a CGA sphere.

    Coordinate entry point for the CGA round-object family in cga.py.
    """
    from . import cga
    return cga.round_point(make_point_ccga(x, y, r, imaginary))


def make_flat_point(x, y):
    """
    Flat point: a finite point "pinned to infinity" — it records position but
    no round structure.

    P_flat = p(x,y) ^ I_inf   (grade 1+3=4).
    Represents the point (x,y) in the flat (non-round) sense.
    """
    p = _point(x, y)
    return p ^ Iinf


def make_line_at_infinity():
    """
    The line at infinity  L_∞ = I_inf  (grade-3 blade, OPNS).

    All ideal points lie in its outer-product null space.
    """
    return Iinf


def make_conic_at_infinity():
    """
    The conic at infinity built from the infinity gauge blade.

    C_∞ = Iod ^ (3 ideal points at infinity).
    In CCGA the "conic at infinity" is  Iod ^ I_inf  (grade 5).
    Points on it satisfy: they are ideal AND lie in the gauge subspace.
    """
    return Iod ^ Iinf


# ── IPNS reference constructor for conics at infinity (for classification) ───

def infinity_ipns_components(ipns):
    """
    Return the gauge-inert components of an IPNS grade-1 conic (§3 result 4).

    In canonical form (zero gauge): s_{einfbar}=0 and s_{einf3}=0.
    In orthogonal basis, eo3 and einf3 share e5,e8 keys.
    make_conic_ipns stores:  e5:-C (eo3 part), e8:-C (also eo3 part)
    A non-zero e8 or asymmetric e5/e8 signals gauge contamination.

    Returns (s_einf3, s_einfbar) in null-basis coefficients:
      s_einf3  = c_e8 - c_e5   (should be 0 in canonical gauge)
      s_einfbar = F_from_e3 - F_from_e4 = 0 when einf1 and einf2 carry equal F
                = 2*(c3-c6) - 2*(c4-c7)  (should be 0)
    """
    c3=_coeff(ipns,'e3'); c4=_coeff(ipns,'e4')
    c5=_coeff(ipns,'e5'); c6=_coeff(ipns,'e6')
    c7=_coeff(ipns,'e7'); c8=_coeff(ipns,'e8')
    s_einf3   = (c8 - c5)         # non-zero only if eo3 and einf3 have different weight
    s_einfbar = 2*(c3-c6) - 2*(c4-c7)   # difference of F from each infinity pair
    return s_einf3, s_einfbar
