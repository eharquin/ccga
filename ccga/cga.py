"""
The CGA "round" object family, recovered inside CCGA via the infinity-gauge
blade  Iinfd = (einf1 − einf2) ∧ einf3.

Wedging with Iinfd collapses the two conic-specific infinity directions
(einfbar, einf3) and lands a CCGA multivector in the CGA subalgebra (§3.3 "one
isotropic radius", §3.9 GAC map).  Each CGA object of grade k appears in CCGA at
grade k+2:

    round point   p ∧ Iinfd                 grade 3
    point pair    p1 ∧ p2 ∧ Iinfd           grade 4
    flat point    p ∧ einf ∧ Iinfd          grade 4   (≡ −(p ∧ Iinf))
    circle        p1 ∧ p2 ∧ p3 ∧ Iinfd      grade 5
    line          p1 ∧ p2 ∧ einf ∧ Iinfd    grade 5

These are *distinct* from the CCGA conic objects in objects.py (which live at
grades 1/2/7).  Constructors take CCGA points (multivectors) — compose them with
point()/make_point_ccga()/make_ideal_point() first; a point carrying a radius
(make_point_ccga(x,y,r)) turns round_point into a sphere/circle.

Reality is uniform:  reality(O) = sign( (Iod | O)² ).  The contraction Iod | O
also extracts the underlying CGA-content blade (its grade is the CGA grade,
i.e. 2 lower than O).
"""

from .algebra import Iinfd, Iod, einf, to_null_basis
from .operations import grades, is_zero

_TOL = 1e-9


# ── constructors (inputs are CCGA point multivectors) ─────────────────────────

def round_point(p):
    """CGA round point  p ∧ Iinfd  (grade 3).  Pass a radius-carrying point for a sphere."""
    return p ^ Iinfd


def point_pair(p1, p2):
    """CGA point pair  p1 ∧ p2 ∧ Iinfd  (grade 4)."""
    return p1 ^ p2 ^ Iinfd


def flat_point(p):
    """CGA flat point  p ∧ einf ∧ Iinfd  (grade 4).  Equals −(p ∧ Iinf)."""
    return p ^ einf ^ Iinfd


def circle(p1, p2, p3):
    """CGA circle through 3 points  p1 ∧ p2 ∧ p3 ∧ Iinfd  (grade 5)."""
    return p1 ^ p2 ^ p3 ^ Iinfd


def line(p1, p2):
    """CGA line through 2 points  p1 ∧ p2 ∧ einf ∧ Iinfd  (grade 5)."""
    return p1 ^ p2 ^ einf ^ Iinfd


# ── analysis helpers ──────────────────────────────────────────────────────────

def cga_blade(O):
    """The underlying CGA-content blade  Iod | O  (grade = grade(O) − 2)."""
    return Iod | O


def _scalar(mv):
    """Grade-0 part as a float (0.0 if absent)."""
    for k, v in mv.items():
        if k == 0:
            return float(v)
    return 0.0


def reality(O, tol=_TOL):
    """'real' | 'imaginary' | 'degenerate' from sign of (Iod | O)²."""
    X = Iod | O
    sq = _scalar(X * X)
    if abs(sq) < tol:
        return 'degenerate'
    return 'real' if sq > 0 else 'imaginary'


def is_finite(O, tol=_TOL):
    """True if O has an origin (eo) component → finite; False → ideal (at infinity)."""
    return any('eo' in name for name in to_null_basis(O, tol))


def _has_einf_factor(O, tol=_TOL):
    """True if O contains the einf factor (flat-point / line family)."""
    return is_zero(O ^ einf, tol)


def classify_cga(O, tol=_TOL):
    """
    Classify a CGA round object built by this module.

    Returns {'type', 'grade', 'cga_grade', 'reality', 'finite'}.
    Types: cga_round_point, cga_point_pair, cga_flat_point, cga_circle, cga_line.
    """
    gs = grades(O, tol)
    g = gs[0] if len(gs) == 1 else None
    has_einf = _has_einf_factor(O, tol)

    if g == 3:
        typ = 'cga_round_point'
    elif g == 4:
        typ = 'cga_flat_point' if has_einf else 'cga_point_pair'
    elif g == 5:
        typ = 'cga_line' if has_einf else 'cga_circle'
    else:
        return {'type': 'unknown', 'grade': gs, 'cga_grade': None,
                'reality': 'n/a', 'finite': None}

    return {'type': typ, 'grade': gs, 'cga_grade': max(g - 2, 0),
            'reality': reality(O, tol), 'finite': is_finite(O, tol)}
