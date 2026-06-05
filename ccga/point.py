"""
CCGA point embedding  R^2 → R^{5,3}.

  p(x,y) = eo + x·e1 + y·e2 + (x²/2)·einf1 + (y²/2)·einf2 + xy·einf3

A point may carry a RADIUS, exactly as a CGA round point / sphere does
(§3 result 3).  With radius r:

  p(x,y,r) = eo + x·e1 + y·e2 + (x²/2)·einf1 + (y²/2)·einf2 + xy·einf3 ∓ (r²/2)·einf

  - real radius     (imaginary=False): p² = +r²  (subtracts (r²/2)·einf)
  - imaginary radius (imaginary=True): p² = −r²  (adds (r²/2)·einf)

Properties (§2), with r = 0:
  - Null:          p² = 0
  - Normalization: p · einf = -1
  - Distance:      p(x,y) · p(x',y') = -½[(x-x')²+(y-y')²]
"""

from .algebra import (alg, e1, e2, eo, einf, einf1, einf2, einf3)


def point(x, y, r=0.0, imaginary=False):
    """
    Embed (x,y) ∈ R^2 in R^{5,3}.

    r=0 → null point (p²=0, p·einf=-1).
    r≠0 → round point / point with radius:
            p² = +r² (real, default) or p² = −r² (imaginary=True).
    """
    base = (eo + x*e1 + y*e2
            + (x*x/2)*einf1 + (y*y/2)*einf2 + x*y*einf3)
    if r:
        sign = +1.0 if imaginary else -1.0
        base = base + sign * (r*r/2) * einf
    return base


def normalize(mv):
    """Normalize a round object so that mv · einf = -1."""
    scale = float(-(mv | einf).e)
    if abs(scale) < 1e-14:
        raise ValueError("Cannot normalize: mv · einf = 0 (ideal element?)")
    return mv * (1.0 / scale)


def inner_product(p, q):
    """Scalar inner product of two multivectors (grade-0 part of p|q)."""
    return float((p | q).e)


def verify_point_properties():
    """Symbolically and numerically verify §2 properties."""
    import sympy as sp

    x, y, xp, yp = sp.symbols('x y xp yp', real=True)

    # ── symbolic null check ──────────────────────────────────────────────────
    alg_sym = alg   # reuse the same algebra with sympy coefficients
    def _pt(cx, cy):
        return (eo + cx*e1 + cy*e2
                + (cx*cx/2)*einf1 + (cy*cy/2)*einf2 + cx*cy*einf3)

    p_sym = _pt(x, y)
    p2 = (p_sym * p_sym).e
    p2_simplified = sp.simplify(p2) if not isinstance(p2, (int, float)) else p2
    assert sp.simplify(p2_simplified) == 0, f"p^2 = {p2_simplified}, expected 0"

    # ── symbolic normalization ──────────────────────────────────────────────
    norm_val = (p_sym | einf).e
    norm_simplified = sp.simplify(norm_val)
    assert sp.simplify(norm_simplified + 1) == 0, f"p·einf = {norm_simplified}, expected -1"

    # ── symbolic distance ───────────────────────────────────────────────────
    q_sym = _pt(xp, yp)
    ip = (p_sym | q_sym).e
    expected = sp.Rational(-1, 2) * ((x - xp)**2 + (y - yp)**2)
    diff = sp.expand(sp.simplify(ip - expected))
    assert diff == 0, f"distance identity off by {diff}"

    # ── numeric sanity ──────────────────────────────────────────────────────
    import numpy as np
    for (px_, py_), (qx_, qy_) in [((1, 2), (4, 6)), ((0, 0), (3, 4))]:
        p_ = point(px_, py_);  q_ = point(qx_, qy_)
        assert abs((p_ * p_).e) < 1e-12, "numeric null check failed"
        assert abs((p_ | einf).e + 1) < 1e-12, "numeric normalization failed"
        got  = (p_ | q_).e
        want = -0.5 * ((px_ - qx_)**2 + (py_ - qy_)**2)
        assert abs(got - want) < 1e-12, f"distance: got {got}, want {want}"

    print("point.py: all verifications passed.")


if __name__ == "__main__":
    verify_point_properties()
