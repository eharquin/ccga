"""
CCGA transformations as **versors** — closed-form geometric-algebra operators
acting by the sandwich product  V X ~V  uniformly on every object (points,
pairs, conics, …).  All three are genuine GA formulas (no coordinate fallback);
each is verified to act correctly on points *and* conics, and (being a versor)
to preserve incidence.

Generators (δp = [G, p] from the infinitesimal action of the embedding):

  translation along x by τ   Tₓ(τ) = (1 − ½τ e₁∧e∞₁)(1 − ½τ e₂∧e∞₃)
  translation along y by τ   T_y(τ) = (1 − ½τ e₂∧e∞₂)(1 − ½τ e₁∧e∞₃)
  rotation by α              R(α) = e^{αE} e^{αK},
                               E = −½ e₁₂                       (E² = −¼)
                               K = ē_o∧e∞₃ − e_{o3}∧ē∞          (K³ = −4K)
                               e^{αE} = cos(α/2) − sin(α/2) e₁₂
                               e^{αK} = 1 + ½sin(2α) K + ¼(1−cos2α) K²
  scaling by s (about origin) D(s) = ∏ᵢ (cosh u + sinh u · e_{oᵢ}∧e∞ᵢ),  u = ½ln s

The "2α" in the rotor is the symmetric-square action on the Veronese (quadratic)
coordinates; E and K commute, so R = e^{αE}e^{αK} = e^{α(E+K)}.
"""
import numpy as np

from .algebra import (e1, e2, eo1, eo2, eo3, einf1, einf2, einf3,
                      eobar, einfbar, einf)

# ── fixed generator blades ────────────────────────────────────────────────────
_E = -0.5 * (e1 ^ e2)
_K = (eobar ^ einf3) - (eo3 ^ einfbar)
_K2 = _K * _K
_DIL = [eo1 ^ einf1, eo2 ^ einf2, eo3 ^ einf3]
# reflection across the x-axis: the grade-3 versor e2 ∧ eo3 ∧ einf3 ( = e2·e₊₃·e₋₃);
# it negates exactly the y-odd carriers e2 (y) and eo3, einf3 (xy).
_REFLECT_X = e2 ^ eo3 ^ einf3


def apply_versor(V, X):
    """Transform any multivector X by the versor V via the sandwich V·X·~V.

    Use this (explicit `V * X * ~V`) rather than the `>>` operator (CLAUDE.md)."""
    return V * X * (~V)


# ── translation ───────────────────────────────────────────────────────────────

def translator(tx, ty=0.0):
    """Versor translating by (tx, ty).

      Tₓ(τ) = (1 − ½τ e₁∧e∞₁)(1 − ½τ e₂∧e∞₃),
      T_y(τ) = (1 − ½τ e₂∧e∞₂)(1 − ½τ e₁∧e∞₃),  T = Tₓ(tx) · T_y(ty).
    """
    Tx = (1 - 0.5*tx*(e1 ^ einf1)) * (1 - 0.5*tx*(e2 ^ einf3))
    Ty = (1 - 0.5*ty*(e2 ^ einf2)) * (1 - 0.5*ty*(e1 ^ einf3))
    return Tx * Ty


# ── rotation (about the origin) ───────────────────────────────────────────────

def rotor(alpha):
    """Versor rotating by angle α about the origin (closed form, K³ = −4K)."""
    eE = np.cos(alpha/2) - np.sin(alpha/2) * (e1 ^ e2)
    eK = 1 + (np.sin(2*alpha)/2) * _K + ((1 - np.cos(2*alpha))/4) * _K2
    return eE * eK


# ── scaling / dilation (about the origin) ─────────────────────────────────────

def dilator(s):
    """Versor scaling by factor s > 0 about the origin (closed form)."""
    if s <= 0:
        raise ValueError("scale factor must be positive")
    u = 0.5 * np.log(s)
    D = np.cosh(u) + np.sinh(u) * _DIL[0]
    for B in _DIL[1:]:
        D = D * (np.cosh(u) + np.sinh(u) * B)
    return D


# ── transformations about an arbitrary centre (conjugation) ───────────────────

def reflector(nx, ny, d=0.0):
    """Versor reflecting across the line  nx·x + ny·y + d = 0.

    NOTE — reflection across a line is NOT given by sandwiching the line's
    grade-1 IPNS vector ℓ ( ℓ X ℓ⁻¹ ).  That bare vector reflects only the linear
    part and breaks the Veronese cross-term (einf3 = xy), producing null vectors
    off the point variety and the wrong image for tilted lines (it is correct
    only for axis-aligned lines, by coincidence).  The reflector must be the
    grade-3 versor below, which carries the xy coupling via eo3∧einf3.

    Built GA-natively by conjugating the x-axis reflector _REFLECT_X
    (e₂∧eo3∧einf3): rotate it to the line's direction and translate it to the
    line's offset —  T(c)·R(θ)·_REFLECT_X·~R(θ)·T(−c),  with θ the line direction
    and c a point on the line.  An involution; acts on points and conics alike.
    """
    n = (nx*nx + ny*ny) ** 0.5
    nx, ny, d = nx/n, ny/n, d/n
    theta = np.arctan2(ny, nx) + np.pi/2          # line direction angle
    R = rotor(theta)
    Vo = R * _REFLECT_X * (~R)                     # reflect across the line through O
    cx, cy = -d*nx, -d*ny                          # a point on the line
    return translator(cx, cy) * Vo * translator(-cx, -cy)


# ── inversion & transversion — CGA ROUND-FAMILY ONLY ──────────────────────────
#
# IMPORTANT: inversion and transversion are *Möbius* maps — they preserve circles
# and lines but send a general conic to a **quartic** (verified: inversion of an
# ellipse fits a quartic, not a conic).  A versor preserves grade, hence maps a
# grade-1 conic to a grade-1 conic, so these CANNOT realise inversion/transversion
# on general CCGA conics.  They ARE versors of the **CGA round sub-family**
# (round points, circles, lines built with Iinfd, see ccga.cga): on those objects
# the sandwich gives the correct circle inversion.  Do NOT apply them to ellipses
# / hyperbolas / parabolas and expect the geometric image.

def inversion(cx=0.0, cy=0.0, r=1.0):
    """Versor for inversion in the circle of centre (cx, cy) and radius r —
    the mirror circle itself, as the (odd) grade-1 versor  σ = centre − ½r²·e∞.
    Apply with apply_versor (σ X ~σ).  **CGA round-family only** (see module note):
    correct on round points / circles / lines, not on general conics."""
    from .point import point as _pt
    return _pt(cx, cy) - 0.5*r*r*einf


def transversion(bx, by, r=1.0):
    """Versor for a transversion (special conformal map) — inversion ∘ translation
    ∘ inversion,  σ·T(b)·σ  with σ = inversion(0,0,r).  **CGA round-family only.**
    (The b here is the composed parameter; it differs from the textbook special-
    conformal b by the inversion scale.)"""
    s = inversion(0.0, 0.0, r)
    return s * translator(bx, by) * s


# NOTE: a **shear** (x,y)→(x+ky,y) is NOT a CCGA versor — it is not a conformal
# (orthogonal) transformation of R^{5,3}, so no bivector G satisfies δp=[G,p]
# (verified: best-fit residual ~12% of ‖δp‖).  Shears map conics to conics only
# as a *linear outermorphism* on the Veronese coordinates, not by a sandwich; a
# GA versor implementation does not exist.


def rotor_about(alpha, cx, cy):
    """Rotate by α about the point (cx, cy):  T(c) · R(α) · T(−c)."""
    return translator(cx, cy) * rotor(alpha) * translator(-cx, -cy)


def dilator_about(s, cx, cy):
    """Scale by s about the point (cx, cy):  T(c) · D(s) · T(−c)."""
    return translator(cx, cy) * dilator(s) * translator(-cx, -cy)
