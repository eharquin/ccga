"""
Constructive (GA-native) tripole extraction — what the T / T* analysis buys,
and where the hard floor is.

Honest summary of the three steps and their algebraic cost:

  1. CIRCUMCIRCLE  C = T ∧ Iod ∧ Iinfd      — one wedge, closed form. GA-native.
  2. MEMBERSHIP CUBIC.  Parametrise C rationally q(t); then q(t) ∧ T is a
     grade-4 blade whose components are QUARTICS in t, all sharing one genuine
     cubic factor g(t) (the 4th root differs per component).  Isolating g needs
     to combine ≥2 components (polynomial GCD); there is *no* single GA scalar
     that yields the bare cubic — every scalar projection of q∧T is one such
     quartic.  This is the S₃-symmetry of "3 unordered points in one blade".
  3. CARDANO on g(t)  — an irreducible cube root.  Galois floor: no GA product
     expresses one of three points by radicals simpler than a cube root.
  4. FINISH.  A connecting line ℓ_ij contracts the tripole to a pure dipole
        ℓ_ij ⌋ T = p_i ∧ p_j,
     closed by the standard (pp ± √pp²)/(e∞·pp).  GA-native, square-root only.

So the analysis makes steps 1 and 4 pure GA; step 3's cube root is unavoidable.
The dual T* re-expresses the same content (membership q⌋T*=0; circumcircle by
meet T*∨Iod*∨Iinfd*) but does not lower the degree.
"""
import numpy as np

from ccga.algebra import e1, e2, einf, Iod, Iinfd, I_inv
from ccga.point import point
from ccga.operations import grades, is_zero
from ccga.classify import ipns_to_coeffs
from ccga.extract import extract_tripole

np.set_printoptions(suppress=True, precision=6)

P = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p = [point(*c) for c in P]
T = p[0] ^ p[1] ^ p[2]

# ── 1. circumcircle (closed-form GA blade) ────────────────────────────────────
A, B, C, D, E, F = ipns_to_coeffs((T ^ Iod ^ Iinfd) * I_inv)
cx, cy = -D / (2 * A), -E / (2 * B)
R = (cx * cx + cy * cy - F / A) ** 0.5
print(f"1. circumcircle = T∧Iod∧Iinfd  ->  center=({cx:.5f},{cy:.5f}) R={R:.5f}")


def q_of_t(t):
    c = (1 - t * t) / (1 + t * t); s = 2 * t / (1 + t * t)
    return point(cx + R * c, cy + R * s)


# ── 2. membership: components of q(t)∧T are quartics sharing the cubic ─────────
ts = np.linspace(-4, 4, 9)
keys = sorted({k for t in ts for k in (q_of_t(t) ^ T).keys()})
quartics = []
for k in keys:
    ys = np.array([float(dict((q_of_t(t) ^ T).items()).get(k, 0.0)) * (1 + t * t) ** 2
                   for t in ts])
    if np.max(np.abs(ys)) < 1e-9:
        continue
    c = np.polyfit(ts, ys, 4)
    quartics.append(c / c[0])
print(f"\n2. q(t)∧T is grade {grades(q_of_t(0.5) ^ T)}; "
      f"{len(quartics)} nonzero components, each a quartic in t.")


def real_roots(c, tol=1e-6):
    r = np.roots(c)
    return np.sort(r[np.abs(r.imag) < tol].real)


print("   sample component roots:", np.round(real_roots(quartics[0]), 4),
      "and", np.round(real_roots(quartics[1]), 4))


# common cubic = polynomial GCD (the 3 shared roots)
def poly_gcd(a, b, tol=1e-6):
    a, b = np.trim_zeros(a, 'f'), np.trim_zeros(b, 'f')
    while len(b) and np.max(np.abs(b)) > tol:
        _, r = np.polydiv(a, b)
        r = np.trim_zeros(np.where(np.abs(r) < 1e-6, 0.0, r), 'f')
        a, b = b, r
    return a / a[0]


g = quartics[0]
for c in quartics[1:]:
    g = poly_gcd(g, c)
    if len(g) == 4:
        break
print("3. GCD over components -> genuine cubic g(t), monic coeffs =", np.round(g, 5))


# ── 3. Cardano ────────────────────────────────────────────────────────────────
def cardano(a, b, c):
    pp = b - a * a / 3.0; qq = 2 * a ** 3 / 27.0 - a * b / 3.0 + c
    disc = (qq / 2) ** 2 + (pp / 3) ** 3
    if disc < 0:
        r = np.sqrt(-(pp ** 3) / 27.0); phi = np.arccos(np.clip(-qq / (2 * r), -1, 1))
        m = 2 * np.cbrt(r)
        return [m * np.cos((phi + 2 * np.pi * k) / 3) - a / 3 for k in range(3)]
    u = np.cbrt(-qq / 2 + np.sqrt(disc)); v = np.cbrt(-qq / 2 - np.sqrt(disc))
    return [u + v - a / 3]


pts = [(round(float((q_of_t(t) | e1).e), 4), round(float((q_of_t(t) | e2).e), 4))
       for t in cardano(g[1], g[2], g[3])]
print("   Cardano -> points :", sorted(pts))
print("   original          :", sorted((round(x, 4), round(y, 4)) for x, y in P))

# library end-to-end (same algorithm, robust φ-rotations)
print("   ccga.extract_tripole:", sorted((round(x, 4), round(y, 4))
                                          for x, y in extract_tripole(T)))

# ── 4. the GA-native dipole finish ────────────────────────────────────────────
print("\n4. connecting line contracts T to a pure dipole (square-root finish):")


def line_through(pa, pb):
    (x1, y1), (x2, y2) = pa, pb
    return (y2 - y1) * e1 + (x1 - x2) * e2 - (x2 * y1 - x1 * y2) * einf


for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    dip = line_through(P[i], P[j]) | T
    edge = p[i] ^ p[j]
    k0 = next(iter(edge.keys()))
    sc = next((float(v) for k, v in dip.items() if k == k0), 0.) / dict(edge.items())[k0]
    print(f"   ℓ_{i+1}{j+1} ⌋ T  grade {grades(dip)} == (scale)·p{i+1}∧p{j+1}? "
          f"{is_zero(dip - edge * sc)}")
