"""
Conic ∩ line → dipole extraction: what the CCGA_CONIC_LINE_DIPOLE graph computes,
and the geometric construction that actually recovers the point pair.

Reproduces the ga-constructor graph `CCGA_CONIC_LINE_DIPOLE` (the line is built
through two of the conic's own points E,F, so the *true* intersection dipole is
exactly E∧F — a built-in ground truth) and answers:

    "how can the dipole be extracted using I2, I2_1, I2_2?"

Findings (all verified below):

1.  STRUCTURE.  With L1 = E∧F∧Iinf∧Iod (the line-as-conic) and C1 the conic,
        I6 = C1 & L1          (grade 6)   =   (E∧F) ∧ T ∧ Iod
    where T = asymptotic_dipole(C1) is the conic's pair of ideal points.  So I6
    is the 4-point Bézout object: the finite dipole E∧F wedged with the conic's
    two ideal points (T) and the Iod gauge.

2.  WHAT THE SANDWICH MEANS (the "geometric information").  The graph's
        Inv1 = e3^e4^e5 = e_{+1}∧e_{+2}∧e_{+3}      (positive-axis pseudoscalar)
        Inv2 = e6^e7^e8 = e_{-1}∧e_{-2}∧e_{-3}      (negative-axis pseudoscalar)
    and  X ⟼ Inv >>> X = Inv·X·~Inv  is the ORIGIN↔INFINITY INVERSION:
        eo_i ⟼ -2·einf_i ,   einf_i ⟼ -½·eo_i      (Euclidean e1,e2 fixed).
    That is a genuine conformal involution swapping origin and infinity — which
    is why the user sensed I2inv1/I2inv2 "carry geometric information".

3.  NEGATIVE RESULT.  Neither I2 = I6 | (E∧F∧Iinfd) nor I2_1, I2_2 (its two
    inverted re-contractions) is ∝ E∧F, and E∧F is NOT in their linear span
    (rank 3 → 4 when E∧F is appended).  Reason: I6 = (E∧F)∧T∧Iod determines the
    dipole only *modulo T*, so no fixed contraction / inversion isolates it.
    The dipole is reachable by *some* grade-2 contractor, but it is a dense,
    conic-dependent operator with no closed blade form — there is no clean
    `I6 | something` formula.

4.  THE CONSTRUCTION THAT WORKS (the user's "geometric transformation" intuition,
    done right) = the source paper's Algorithm 3 `conic_line_inter`:
    a GA ROTOR aligns the line to a coordinate axis, collapsing the intersection
    to a one-variable quadratic; solve it, rotate the two points back, and wedge
    with Iinfd to get the renderable CGA point pair  E∧F∧Iinfd.

Run:  python notebook/conic_line_dipole_extraction.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccga.algebra import (Iod, Iinf, Iinfd, ep1, ep2, ep3, em1, em2, em3,
                          eo1, einf1, eo, einf, e1, e2)
from ccga.point import point
from ccga.operations import meet, grades, proportional, is_zero, to_null_basis
from ccga.transform import rotor, apply_versor
from ccga.classify import ipns_to_coeffs, _conic_vector
from ccga.extract import asymptotic_dipole
import ccga.cga as cga


def sw(M, B):
    """Constructor's  M >>> B  = sandwich  M·B·~M  (sw with reverse, not inverse)."""
    return M * B * ~M


def coords(P):
    """Euclidean (x, y) of a finite grade-1 point."""
    w = -float((P | einf).e)
    return float((P | e1).e) / w, float((P | e2).e) / w


# ── the graph: line built through two conic points E,F  ⇒  true dipole = E∧F ──
E = point(-1.13, 0.86)
F = point(-0.74, 1.47)
G = point(-0.30, 1.16)
H = point(-0.77, 0.62)
J = point(-1.24, 1.45)

C1 = E ^ F ^ G ^ H ^ J ^ Iod          # grade-7 OPNS conic through the 5 points
L1 = E ^ F ^ Iinf ^ Iod               # grade-7 line-as-conic through E,F
I6 = C1 & L1                          # grade-6 intersection object
D_true = E ^ F                        # the ground-truth finite dipole

Inv1 = ep1 ^ ep2 ^ ep3                # e3^e4^e5
Inv2 = em1 ^ em2 ^ em3                # e6^e7^e8

I2 = I6 | (E ^ F ^ Iinfd)
I2inv1 = sw(Inv1, I2)
I2inv2 = sw(Inv2, I2)
I2_1 = I6 | (I2inv1 ^ Iinfd)
I2_2 = I6 | (I2inv2 ^ Iinfd)


def report():
    print("1) STRUCTURE  I6 = (E∧F) ∧ T ∧ Iod")
    T = asymptotic_dipole(C1)
    print("   grade(I6) =", grades(I6),
          "| I6 ∝ (E∧F)∧T∧Iod ?", proportional(I6, D_true ^ T ^ Iod)[0])

    print("\n2) SANDWICH = origin↔infinity inversion  (eo_i ↔ einf_i)")
    print("   Inv1 >>> eo1   =", {k: round(v, 4) for k, v in to_null_basis(sw(Inv1, eo1)).items()})
    print("   Inv1 >>> einf1 =", {k: round(v, 4) for k, v in to_null_basis(sw(Inv1, einf1)).items()})

    print("\n3) NEGATIVE RESULT  (the three blades do not contain E∧F)")
    for nm, X in [("I2", I2), ("I2_1", I2_1), ("I2_2", I2_2)]:
        print(f"   {nm:5} grade={grades(X)}  ∝ E∧F ? {proportional(X, D_true)[0]}")
    g2 = [k for k in range(256) if bin(k).count("1") == 2]
    vec = lambda mv: np.array([{kk: float(vv) for kk, vv in mv.items()}.get(k, 0.0) for k in g2])
    Mspan = np.array([vec(I2), vec(I2_1), vec(I2_2)]).T
    r3 = np.linalg.matrix_rank(Mspan, tol=1e-9)
    r4 = np.linalg.matrix_rank(np.c_[Mspan, vec(D_true)], tol=1e-9)
    print(f"   rank{{I2,I2_1,I2_2}} = {r3},  rank+{{E∧F}} = {r4}  → E∧F outside their span")

    print("\n4) WORKING CONSTRUCTION  (rotor aligns line → quadratic → dipole)")
    pts = rotor_line_conic(C1, E, F)
    P1, P2 = point(*pts[0]), point(*pts[1])
    print("   recovered points :", [tuple(round(v, 3) for v in p) for p in pts])
    print("   true E,F         :", sorted([coords(E), coords(F)]))
    D_cga = P1 ^ P2 ^ Iinfd
    print("   E∧F∧Iinfd  ∝ cga.point_pair(P1,P2) ?",
          proportional(D_cga, cga.point_pair(P1, P2))[0])
    print("   incidence  E∧(E∧F)=0, F∧(E∧F)=0 ?",
          is_zero(E ^ (P1 ^ P2)), is_zero(F ^ (P1 ^ P2)))


def rotor_line_conic(conic, A, B):
    """Paper Alg. 3 (conic_line_inter), GA-native.  Rotate the system by a GA
    rotor so the line A,B is horizontal, solve the resulting one-variable
    quadratic, rotate the two intersection points back.  Returns [(x,y),(x,y)].
    """
    (ax, ay), (bx, by) = coords(A), coords(B)
    theta = np.arctan2(by - ay, bx - ax)
    R, Rb = rotor(-theta), rotor(theta)          # align, and the inverse to undo
    y0 = coords(apply_versor(R, A))[1]           # rotated line is y = y0
    a, b, c, d, e_, f = ipns_to_coeffs(_conic_vector(apply_versor(R, conic)))
    # a x² + (c y0 + d) x + (b y0² + e y0 + f) = 0
    qa, qb, qc = a, c * y0 + d, b * y0 * y0 + e_ * y0 + f
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return []
    xs = [(-qb + s * np.sqrt(disc)) / (2 * qa) for s in (1, -1)]
    back = [coords(apply_versor(Rb, point(x, y0))) for x in xs]
    return sorted((round(x, 6), round(y, 6)) for x, y in back)


if __name__ == "__main__":
    report()
