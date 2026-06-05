"""
Center of a tripole via GA formulas.

The tripole's natural center is the CIRCUMCENTER (center of the circle through
p1,p2,p3).  Two GA routes, both verified:

  (A) reflect e∞ in the circumcircle:
        s   = (T ∧ Iod ∧ Iinfd) · I⁻¹        (circumcircle, IPNS grade-1 vector)
        c   = s e∞ s                          (sandwich = reflection of e∞)
        ĉ   = c / (−(c·e∞))                   (normalize the round point)
        (x,y) = (ĉ·e1, ĉ·e2)

  (B) perpendicular-bisector meet (no circle needed):
        each midpoint-normal condition is the IPNS vector  m_ij = p_i − p_j
        (a line: the perpendicular bisector of the edge).  The circumcenter is
        the finite point on all three; recovered as the flat point
        m_12 ∧ m_13 ∧ (carrier) ... — here shown via solving c·(p_i−p_j)=0.

We also give the CENTROID for contrast (trivial GA: normalize p1+p2+p3).
"""
import numpy as np

from ccga.algebra import e1, e2, einf, Iod, Iinfd, I_inv
from ccga.point import point
from ccga.operations import grades

P = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p = [point(*c) for c in P]
T = p[0] ^ p[1] ^ p[2]

# reference circumcenter (least-squares from coordinates)
ax, ay = P[0]; bx, by = P[1]; cxr, cyr = P[2]
d = 2 * (ax * (by - cyr) + bx * (cyr - ay) + cxr * (ay - by))
ux = ((ax**2 + ay**2) * (by - cyr) + (bx**2 + by**2) * (cyr - ay)
      + (cxr**2 + cyr**2) * (ay - by)) / d
uy = ((ax**2 + ay**2) * (cxr - bx) + (bx**2 + by**2) * (ax - cxr)
      + (cxr**2 + cyr**2) * (bx - ax)) / d
print(f"reference circumcenter (coords) = ({ux:.6f}, {uy:.6f})")


def coords(mv):
    return float((mv | e1).e), float((mv | e2).e)


def normalize(mv):
    return mv * (1.0 / (-float((mv | einf).e)))


# ── (A) reflect e∞ in the circumcircle ────────────────────────────────────────
print("\n(A) circumcircle s = (T∧Iod∧Iinfd)·I⁻¹ ;  center = s e∞ s")
s = (T ^ Iod ^ Iinfd) * I_inv
print("    s grade:", grades(s))
c = s * einf * s                      # reflection (sandwich) of e∞ in s
print("    c = s e∞ s grade:", grades(c))
chat = normalize(c)
print("    center (A) =", tuple(round(v, 6) for v in coords(chat)))

# also works straight from the OPNS grade-7 blade by dualizing first — same s.

# ── (B) perpendicular bisectors, GA meet ──────────────────────────────────────
# The IPNS vector (p_i − p_j) is the perpendicular bisector of edge ij:
#   q·(p_i − p_j) = q·p_i − q·p_j = −½(|q−p_i|² − |q−p_j|²) = 0  ⇔ equidistant.
# The circumcenter is the finite point lying on all three bisectors.
print("\n(B) perpendicular bisectors  b_ij = p_i − p_j  (IPNS lines)")
b12 = p[0] - p[1]
b13 = p[0] - p[2]
# meet of two lines (IPNS) → their common point (a flat point); recover (x,y)
# by solving the 2×2 linear system from the bisector equations directly:
from ccga.classify import ipns_to_coeffs
A1, B1, C1, D1, E1, F1 = ipns_to_coeffs(b12)
A2, B2, C2, D2, E2, F2 = ipns_to_coeffs(b13)
# each is a line  D x + E y + F = 0
M = np.array([[D1, E1], [D2, E2]]); rhs = -np.array([F1, F2])
xy = np.linalg.solve(M, rhs)
print("    bisector lines:  "
      f"{D1:.3f} x + {E1:.3f} y + {F1:.3f} = 0   and   "
      f"{D2:.3f} x + {E2:.3f} y + {F2:.3f} = 0")
print("    center (B) =", tuple(round(v, 6) for v in xy))

# ── centroid (for contrast) ──────────────────────────────────────────────────
g = normalize(p[0] + p[1] + p[2])
print("\ncentroid = normalize(p1+p2+p3) ->",
      tuple(round(v, 6) for v in coords(g)),
      "   (mean =", (round(sum(x for x, _ in P)/3, 6), round(sum(y for _, y in P)/3, 6)), ")")
