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
"""

from .algebra import I, I_inv


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
