"""
CCGA operations: join, meet, dual — one fixed convention throughout.

Dual convention (§4, result 7):
  C_ipns = dual(C_opns)  =  C_opns * I_inv   (right-multiply by I^{-1})
  C_opns = undual(C_ipns) = C_ipns * I        (right-multiply by I)

Meet (regressive product):
  A & B  works correctly in kingdon 1.4.0 for Algebra(5,3).
  It agrees with (A.dual()^B.dual()).dual() up to an overall sign that
  is consistent and harmless for incidence tests (both give p^meet=0).
  We use & directly as it is cleaner.

Join:
  A ^ B  (outer product) — grade adds when subspaces are disjoint.

Right complement (paper §6.1, "Conics, their pencils and intersections in GA"):
  A^c is the unique linear map with E ^ E^c = I for every null-basis blade E
  (the wedge of a subset of eo1,eo2,eo3,e1,e2,einf1,einf2,einf3), extended by
  linearity.  See right_complement() for the verified identities.
"""

import numpy as np

from .algebra import (I, I_inv, alg, _NULL_PRINT_NAMES, _NULL_BLADE_NAMES, _T,
                       to_null_basis)


def join(A, B):
    """Outer (wedge) product — span of two blades."""
    return A ^ B


def meet(A, B):
    """Regressive product — intersection of two OPNS blades."""
    return A & B


def dual(mv):
    """IPNS dual: mv_ipns = mv_opns * I^{-1}  (right-multiply by I^{-1})."""
    return mv * I_inv


def undual(mv):
    """Inverse dual: mv_opns = mv_ipns * I  (right-multiply by I)."""
    return mv * I


def inner(A, B):
    """Grade-0 (scalar) part of A | B."""
    return float((A | B).e)


def max_coeff(mv, tol=0.0):
    """Max absolute coefficient in mv (after optional threshold)."""
    vals = [abs(float(v)) for v in mv.values() if abs(float(v)) > tol]
    return max(vals) if vals else 0.0


def is_zero(mv, tol=1e-10):
    return max_coeff(mv) <= tol


def grades(mv, tol=1e-10):
    """Return sorted list of non-zero grades present in mv."""
    return sorted({bin(k).count('1') for k, v in mv.items()
                   if abs(float(v)) > tol})


def pure_grade(mv, tol=1e-10):
    """Return the single grade of mv, or raise if mixed/zero."""
    gs = grades(mv, tol)
    if len(gs) == 0:
        raise ValueError("Zero multivector has no grade")
    if len(gs) > 1:
        raise ValueError(f"Mixed grades {gs}")
    return gs[0]


# ── Right complement, norm, orthogonality (paper §6.1) ──────────────────────
#
# For a null-basis blade E = wedge of a subset S of the 8 vectors
#   NB = (eo1, eo2, eo3, e1, e2, einf1, einf2, einf3)
# (in this fixed order), define E^c = sign * wedge(NB[i] for i in complement
# of S, increasing order), with sign chosen so that E ^ E^c = I exactly.
# Extend to all multivectors by linearity via the to_null_basis() decomposition.
#
# Because E ^ F^c = 0 for basis blades E != F of the same grade (S != T with
# |S|=|T| forces S ^ T^c == 0 or != {0..7}), and E ^ E^c = I for every E, this
# gives the universal identity (verified numerically for grades 1, 2, 7):
#
#     A ^ A^c = (sum_E A_E^2) * I        ("norm" identity, paper §6.1)
#
# where A_E are A's coefficients in the null working basis.  This reproduces
# the paper's "Theorem (right complement of n-PSE)": for a 1-PSE point p with
# B_p-coordinates (a,b,c,d,e,f) (i.e. p = a*eo + b*e1 + c*e2 + d*einf1 +
# e*einf2 + f*einf3 = a*eo1 + a*eo2 + b*e1 + c*e2 + d*einf1 + e*einf2 + f*einf3
# in the null working basis), p ^ p^c = (2a^2+b^2+c^2+d^2+e^2+f^2) * I.
#
# Also: right_complement is grade-complementary (k -> 8-k), and
# (A^c)^c = -A  (verified for grade 1).

def _build_complement_table():
    n = len(_NULL_PRINT_NAMES)
    table = {}
    for name in _NULL_BLADE_NAMES:
        S = set() if name == '1' else {_NULL_PRINT_NAMES.index(f) for f in name.split('^')}
        Sc = sorted(set(range(n)) - S)
        cname = '^'.join(_NULL_PRINT_NAMES[i] for i in Sc) if Sc else '1'
        inv = sum(1 for s in S for t in Sc if t < s)
        sign = -1.0 if (inv % 2 == 0) else 1.0  # = (-1)**(inv+1), so E ^ E^c = +I
        table[name] = (cname, sign)
    return table


_COMPLEMENT_TABLE = _build_complement_table()
_NAME_TO_COL = {name: _T[:, i] for i, name in enumerate(_NULL_BLADE_NAMES)}


def right_complement(mv, tol=1e-12):
    """
    Right complement A^c (paper §6.1): E ^ E^c = I for every null-basis
    blade E, extended by linearity.  Numeric only (uses to_null_basis).
    """
    comps = to_null_basis(mv, tol)
    vec = np.zeros(256)
    for name, c in comps.items():
        cname, sign = _COMPLEMENT_TABLE[name]
        vec += (c * sign) * _NAME_TO_COL[cname]
    return alg.multivector({k: float(v) for k, v in enumerate(vec) if abs(v) > tol})


def norm2(mv, tol=1e-12):
    """||A||^2 = (A ^ A^c)^*  =  sum_E A_E^2  (paper §6.1)."""
    return float(dual(mv ^ right_complement(mv, tol)).e)


def norm(mv, tol=1e-12):
    """||A|| = sqrt((A ^ A^c)^*)  (paper §6.1).  Always real: norm2 >= 0."""
    return np.sqrt(norm2(mv, tol))


def orthogonal(A, B, tol=1e-9):
    """A, B orthogonal iff A ^ B^c = 0  (paper §6.1)."""
    return is_zero(A ^ right_complement(B), tol)


def proportional(A, B, tol=1e-7):
    """
    True iff A == ratio * B for some scalar ratio (the paper's "≡", equality
    up to overall scale, used throughout the pencil-calculus theorems).
    Returns (is_proportional, ratio).
    """
    if is_zero(B, tol):
        return is_zero(A, tol), 0.0
    bd = dict(B.items())
    ref = max(bd, key=lambda k: abs(bd[k]))
    ratio = float(dict(A.items()).get(ref, 0.0)) / bd[ref]
    scale = max(norm(A), norm(B), 1.0)
    return bool(is_zero(A - B * ratio, tol * scale)), ratio
