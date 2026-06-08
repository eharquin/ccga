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
from .operations import dual, undual, meet, join, is_zero, grades, pure_grade


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


def make_conic_pentapole(p1, p2, p3, p4, p5):
    """
    Grade-5 OPNS blade (conic pentapole) — the bare wedge of five CCGA points.

    OPNS:  P5 = p1 ^ p2 ^ p3 ^ p4 ^ p5   (grade 5)
    IPNS:  P5* = dual(P5)                 (grade 3)

    The pentapole already IS the conic as a point-set: q ^ P5 = 0 ⟺ q lies on
    the conic through the five points.  A conic is one linear relation among the
    six Veronese coordinates (1, x, y, x²/2, y²/2, xy), so all of its points lie
    in a 5-D subspace of V6; the five points span that subspace.

    What it does NOT give cleanly is the algebraic (coefficient) form: its dual
    is the grade-3 blade

        dual(P5) = -½ · (s ∧ Iinfd),   Iinfd = (einf1 - einf2) ∧ einf3,

    i.e. the clean grade-1 conic vector s "smeared" by the infinity-gauge
    bivector.  Wedging the origin gauge blade Iod = eobar ∧ eo3 collapses this:
    pentapole_to_conic(P5) = P5 ∧ Iod is grade 7 and its dual is the clean
    grade-1 conic vector (see make_conic_opns / conic_dual_grade1).
    """
    opns = p1 ^ p2 ^ p3 ^ p4 ^ p5
    ipns = dual(opns)
    return opns, ipns


def pentapole_to_conic(P5):
    """Promote a grade-5 pentapole to the grade-7 OPNS conic:  C7 = P5 ∧ Iod."""
    return P5 ^ Iod


def conic_dual_grade1(C7):
    """Clean grade-1 IPNS conic vector s = dual(C7) of a grade-7 OPNS conic.

    Returns the vector whose components are the conic coefficients (A,B,C,D,E,F)
    via classify.ipns_to_coeffs.  (Thin, named wrapper around dual for symmetry
    with the OPNS side.)
    """
    return dual(C7)


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

      C_opns = Iod ^ p1 ^ p2 ^ p3 ^ p4 ^ p5 = Iod ^ pentapole

    The bare 5-wedge (pentapole, grade 5) already fixes the conic as a point
    locus; wedging Iod = eobar ∧ eo3 is a pure gauge fix so the dual lands at
    grade 1 (the clean conic vector s) instead of grade 3 (= -½ s ∧ Iinfd).
    See make_conic_pentapole for the distinction.

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


def make_ellipse_from_focus(center, focus, a=None, ecc=None, through=None):
    """Ellipse from its **center** and **one focus** — the inverse of conic_center
    / conic_foci — built GA-natively by a versor sandwich.

    center, focus are CCGA grade-1 points.  The pair fixes only 4 of the ellipse's
    5 d.o.f.: the center (cx,cy), the major-axis direction û (the focus lies on
    the major axis), and the focal distance c = |focus − center|.  The remaining
    size d.o.f. is **underdetermined** — center + one focus is a one-parameter
    family of confocal ellipses — so exactly ONE of the following must be given:

      a       — semi-major axis (must satisfy a > c),
      ecc     — eccentricity e = c/a   (0 < e < 1; then a = c/e),
      through — a CCGA point the ellipse must pass through (fixes a),

    with the minor semi-axis b = √(a²−c²).  The other focus (center − c·û) is
    automatic.  Picking either focus yields the same ellipse.

    Construction (pure GA, no coordinate conic formula):
      1. canonical axis-aligned ellipse s₀ = make_ellipse(a, b) at the origin
         (foci on the x-axis at ±c),
      2. place it with the versor V = T(center)·R(θ), θ = angle of û,
         s = V s₀ ~V  (apply_versor) — a grade-1 IPNS conic.

    The only scalar steps are the focal-vector readoff (GA inner products, as
    everywhere in the repo), atan2 for θ, and the √ for b — the irreducible
    root step (CCGA cheat sheet §5).  Returns (opns_grade7, ipns_grade1).
    """
    from .transform import translator, rotor, apply_versor

    def _xy(p):
        w = -float((p | einf).e)
        if abs(w) < 1e-12:
            raise ValueError("center/focus must be a finite point (p·einf ≠ 0)")
        return float((p | e1).e) / w, float((p | e2).e) / w

    cx, cy = _xy(center)
    fx, fy = _xy(focus)
    dx, dy = fx - cx, fy - cy
    c = (dx*dx + dy*dy) ** 0.5
    if c < 1e-12:
        raise ValueError("focus coincides with center; major-axis direction "
                         "undefined (use make_circle for the c→0 limit)")
    theta = np.arctan2(dy, dx)

    given = [v is not None for v in (a, ecc, through)]
    if sum(given) != 1:
        raise ValueError("give exactly one of: a (semi-major), ecc, or through")
    if ecc is not None:
        if not 0.0 < ecc < 1.0:
            raise ValueError("ellipse eccentricity must satisfy 0 < ecc < 1")
        a = c / ecc
    elif through is not None:
        qx, qy = _xy(through)
        # coords of `through` in the centered, axis-aligned frame (ξ along û)
        co, si = np.cos(theta), np.sin(theta)
        xi = (qx - cx)*co + (qy - cy)*si
        eta = -(qx - cx)*si + (qy - cy)*co
        # ξ²/a² + η²/(a²−c²) = 1  ⇒  A²−(c²+ξ²+η²)A+ξ²c² = 0,  A = a²
        bb = c*c + xi*xi + eta*eta
        disc = bb*bb - 4*c*c*xi*xi
        if disc < 0:
            raise ValueError("no real ellipse through that point with this focus")
        A2 = (bb + disc**0.5) / 2                  # larger root ⇒ a² > c² ⇒ b² > 0
        a = A2 ** 0.5
    if a <= c:
        raise ValueError(f"semi-major a={a} must exceed focal distance c={c}")
    b = (a*a - c*c) ** 0.5

    _, s0 = make_ellipse(a, b)
    ipns = apply_versor(translator(cx, cy) * rotor(theta), s0)
    return undual(ipns), ipns


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


def make_hyperbola_3points(p1, p2, p3, dir1, dir2):
    """
    Hyperbola through 3 finite points with prescribed asymptotic directions.

      H = p1 ∧ p2 ∧ p3 ∧ vinf(dir1) ∧ vinf(dir2) ∧ Iod      (grade-7 OPNS)

    The two ideal points point_at_infinity(dir1/2) ∈ I_inf become the conic's
    intersections with the line at infinity, i.e. its asymptotic directions
    (so the result is a hyperbola, Δ = C²−4AB > 0).  dir1/dir2 are (vx,vy)
    tuples; pass ((1,0),(0,1)) for a rectangular (axis-aligned) hyperbola.

    Returns (opns_grade7, ipns_grade1).
    """
    from .point import point_at_infinity
    opns = (p1 ^ p2 ^ p3 ^ point_at_infinity(*dir1)
            ^ point_at_infinity(*dir2) ^ Iod)
    return opns, dual(opns)


def make_parabola_3points(p1, p2, p3, axis_dir):
    """
    General (tilted) parabola through 3 finite points with the given axis
    direction — the double-contact-at-infinity construction:

      P = p1 ∧ p2 ∧ p3 ∧ vinf(v) ∧ vinf'(v) ∧ Iod           (grade-7 OPNS)

    where v = axis_dir, vinf = point_at_infinity, vinf' = tangent_at_infinity.
    The factor vinf(v) ∧ vinf'(v) is a *second-order* contact with the line at
    infinity (the parabola is tangent to infinity at vinf(v)), giving Δ = 0 with
    the single (double) ideal direction v.  Works for any v, not just the axes.

    Returns (opns_grade7, ipns_grade1).
    """
    from .point import point_at_infinity, tangent_at_infinity
    opns = (p1 ^ p2 ^ p3 ^ point_at_infinity(*axis_dir)
            ^ tangent_at_infinity(*axis_dir) ^ Iod)
    return opns, dual(opns)


def make_ellipse_3points(p1, p2, p3, a=2.0, b=1.0):
    """Ellipse through 3 finite points, constrained by an IMAGINARY ideal-point
    pair (so it has no real points at infinity → Δ < 0):

      E = p1 ∧ p2 ∧ p3 ∧ B ∧ Iod,   B = (a²·einf1 − b²·einf2) ∧ einf3.

    B is the real bivector of the complex-conjugate ideal directions (a, ±i·b)
    = i·(vinf(a,ib) ∧ vinf(a,−ib)) — a *weighted* Iinfd.  Its line at infinity is
    NON-SECANT to the conic-at-infinity (2 imaginary ideal points), which is
    exactly what makes the conic an ellipse.  a = b gives B = Iinfd (the circular
    points) → a circle.  Compare make_hyperbola_3points (2 real ideal points →
    secant → hyperbola) and make_parabola_3points (1 double → tangent → parabola).

    Returns (opns_grade7, ipns_grade1).
    """
    B = (a*a*einf1 - b*b*einf2) ^ einf3
    opns = p1 ^ p2 ^ p3 ^ B ^ Iod
    return opns, dual(opns)


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


def make_line_pair(line1, line2):
    """Degenerate conic = the **line pair** through two IPNS lines.

    A line pair is the *symmetric product* (symmetric square) of the two line
    covectors $\\ell_1=(a_1,b_1,c_1)$, $\\ell_2=(a_2,b_2,c_2)$ — its locus is
    $(\\ell_1\\!\\cdot q)(\\ell_2\\!\\cdot q)=0$, i.e. line1 OR line2:

        A=a1a2, B=b1b2, C=a1b2+a2b1, D=a1c2+a2c1, E=b1c2+b2c1, F=c1c2.

    There is **no single GA product** of the two line vectors that yields it
    ($\\ell_1\\ell_2$ is scalar+bivector, not the grade-1 conic) — the symmetric
    square is a Veronese/tensor operation, taken here on the coefficients.
    Returns (opns_grade7, ipns_grade1); the result is degenerate (det M3 = 0),
    a crossing pair (Δ>0) or, for parallel lines, a parallel pair (Δ=0).
    Recover the two lines with ccga.extract._lines_of.
    """
    from .classify import ipns_to_coeffs
    a1, b1, c1 = ipns_to_coeffs(line1)[3:6]
    a2, b2, c2 = ipns_to_coeffs(line2)[3:6]
    ipns = make_conic_ipns(a1*a2, b1*b2, a1*b2 + a2*b1,
                           a1*c2 + a2*c1, b1*c2 + b2*c1, c1*c2)
    return undual(ipns), ipns


def make_parallel_line_pair(E, F, G):
    """Degenerate PARALLEL line pair: the line through E,F together with the line
    through G parallel to it (both along the E→F direction).

    Construction (degenerate-hyperbola limit with two **opposite ideal points**):
    with v = E − F, use V1 = make_ideal_point(v), V2 = make_ideal_point(−v) and

        C = E ∧ F ∧ G ∧ V1 ∧ V2 ∧ Iod.

    V1, V2 are antipodal on the line at infinity but share the *same* line
    direction, so the two asymptotic directions of the "hyperbola" merge — the
    quadratic part becomes a perfect square (Δ = 0) and the conic degenerates to
    two parallel lines.  Returns (opns_grade7, ipns_grade1); recover the two
    lines with ccga.extract._lines_of.
    """
    x = float((E | e1).e) - float((F | e1).e)
    y = float((E | e2).e) - float((F | e2).e)
    V1 = make_ideal_point(x, y)
    V2 = make_ideal_point(-x, -y)
    opns = E ^ F ^ G ^ V1 ^ V2 ^ Iod
    return opns, dual(opns)


def make_secant_line_pair_through_origin(P, Q, R):
    """Degenerate SECANT (crossing) line pair: the line through the **origin and
    P**, together with the line through **Q and R**.

    Same family as make_parallel_line_pair, but the merged direction is P's
    *position* v = P (direction origin→P):

        V1 = make_ideal_point(v), V2 = make_ideal_point(−v),  v = P
        C = P ∧ Q ∧ R ∧ V1 ∧ V2 ∧ Iod.

    make_ideal_point(±v) is the ideal point of direction v, so one component is
    forced to be the line of direction v through the origin (which contains P);
    the other carries Q, R.  Δ > 0 (crossing), det M3 = 0.  Returns
    (opns_grade7, ipns_grade1); split with ccga.extract._lines_of.
    """
    x = float((P | e1).e)
    y = float((P | e2).e)
    V1 = make_ideal_point(x, y)
    V2 = make_ideal_point(-x, -y)
    opns = P ^ Q ^ R ^ V1 ^ V2 ^ Iod
    return opns, dual(opns)


# ══ Tangents / polars ════════════════════════════════════════════════════════

def polar_line(conic, q):
    """Polar line of a point q with respect to a conic, as an IPNS grade-1 line.

    Built GA-natively from the point-map differentials (the gradient of the
    conic form):

        ∂ₓp = e1 + x·einf1 + y·einf3,   ∂ᵧp = e2 + y·einf2 + x·einf3,
        nx = ½ (∂ₓp · s),  ny = ½ (∂ᵧp · s),  c = (eo + ½x·e1 + ½y·e2) · s,

    with s the IPNS conic vector and q = (x, y).  The polar is the line
    nx·x + ny·y + c = 0; when q lies on the conic it is the **tangent** at q
    (use tangent_line).  Equivalently the polar equals M₃·[x,y,1] (the pole–polar
    duality), so the center is polar_line's pole of the line at infinity.
    """
    from .classify import _conic_vector
    s = _conic_vector(conic)
    w = -float((q | einf).e)
    if abs(w) < 1e-12:
        raise ValueError("polar of an ideal point is not an affine line")
    x = float((q | e1).e) / w
    y = float((q | e2).e) / w
    tx = e1 + x*einf1 + y*einf3
    ty = e2 + y*einf2 + x*einf3
    carrier = eo + (x/2)*e1 + (y/2)*e2
    ip = lambda a: float((a | s).e)
    return make_conic_ipns(0.0, 0.0, 0.0, 0.5*ip(tx), 0.5*ip(ty), ip(carrier))


def tangent_line(conic, p, tol=1e-7):
    """Tangent line to a conic at a point p that lies on it (grade-1 IPNS line).

    The tangent is the polar of the contact point (polar_line); it meets the
    conic in the double point p.  Raises if p is not on the conic."""
    from .classify import _conic_vector
    s = _conic_vector(conic)
    if abs(float((p | s).e)) > tol:
        raise ValueError("p is not on the conic (q·s ≠ 0); use polar_line")
    return polar_line(conic, p)


def normal_line(conic, p, tol=1e-7):
    """Normal line to a conic at a point p on it (grade-1 IPNS line).

    The normal is perpendicular to the tangent and passes through p: it points
    along the gradient direction (nx, ny) of the tangent.  Raises if p ∉ conic.
    """
    from .classify import _conic_vector, ipns_to_coeffs
    s = _conic_vector(conic)
    if abs(float((p | s).e)) > tol:
        raise ValueError("p is not on the conic")
    nx, ny = ipns_to_coeffs(tangent_line(conic, p))[3:5]
    w = -float((p | einf).e)
    x = float((p | e1).e) / w
    y = float((p | e2).e) / w
    # line through (x,y) with direction (nx,ny): −ny·X + nx·Y + (ny·x − nx·y) = 0
    return make_conic_ipns(0.0, 0.0, 0.0, -ny, nx, ny*x - nx*y)


def apollonius_conic(conic, q):
    """The Apollonius (normal-foot) conic of an external point q.

    The feet of the normals dropped from q onto the conic are exactly the points
    p where (q − p) ∥ ∇F(p).  That condition is itself a conic — a **rectangular
    hyperbola** (trace 0) through q and the conic's center — so the feet are
    conic ∩ apollonius_conic (see extract.normal_feet / project_point_to_conic).
    """
    from .classify import _conic_vector, ipns_to_coeffs
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(conic))
    qw = -float((q | einf).e)
    qx = float((q | e1).e) / qw
    qy = float((q | e2).e) / qw
    return make_conic_ipns(-C, C, 2*(A - B),
                           C*qx - E - 2*A*qy,
                           2*B*qx - C*qy + D,
                           E*qx - D*qy)


def conic_from_5_tangents(lines):
    """Conic tangent to 5 given lines — the dual construction.

    Each line is an IPNS grade-1 line with homogeneous coefficients (u, v, w)
    (u·x + v·y + w = 0).  Tangent lines of a conic satisfy the dual conic
    equation lᵀ·M*·l = 0, so 5 lines pin down the dual matrix M* (a null space),
    and the point conic is M₃ = adj(M*) ∝ M*⁻¹.

    Returns (opns_grade7, ipns_grade1).
    """
    import numpy as np
    from .classify import ipns_to_coeffs
    rows = []
    for l in lines:
        _, _, _, u, v, w = ipns_to_coeffs(l)        # line coeffs (D,E,F)=(u,v,w)
        rows.append([u*u, v*v, u*v, u*w, v*w, w*w])
    if len(rows) < 5:
        raise ValueError("need at least 5 tangent lines")
    _, _, Vt = np.linalg.svd(np.array(rows))
    a, b, c, d, e, f = Vt[-1]                        # dual conic [a b c d e f]
    Mstar = np.array([[a, c/2, d/2], [c/2, b, e/2], [d/2, e/2, f]])
    if abs(np.linalg.det(Mstar)) < 1e-12:
        raise ValueError("degenerate dual conic (lines not in general position)")
    M3 = np.linalg.inv(Mstar)
    M3 = M3 / np.max(np.abs(M3))
    ipns = make_conic_ipns(M3[0, 0], M3[1, 1], 2*M3[0, 1],
                           2*M3[0, 2], 2*M3[1, 2], M3[2, 2])
    return undual(ipns), ipns


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
