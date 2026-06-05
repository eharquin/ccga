"""
Tripole systematic operator probe.

ppp = p1 ^ p2 ^ p3   (grade-3 CCGA blade)

We hit the tripole with every special blade under the three products
  |  (inner / contraction)
  ^  (outer / wedge)
  &  (meet / regressive)
and report grade + null-basis decomposition, flagging the products that
carry recognisable geometry (circumcircle, signed area, connecting lines,
ideal directions, ...).  Numeric triangle + a few symbolic confirmations.
"""
import numpy as np
import sympy as sp

from ccga.algebra import (
    e1, e2, eo, einf, eobar, einfbar,
    eo1, eo2, eo3, einf1, einf2, einf3,
    Iod, Iinfd, Io, Iinf, Ieps, I, I_inv,
    format_null, to_null_basis,
)
from ccga.point import point
from ccga.operations import grades, is_zero
from ccga.classify import ipns_to_coeffs

np.set_printoptions(suppress=True, precision=6)

# ── a concrete, generic triangle ──────────────────────────────────────────────
P = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p = [point(*c) for c in P]
T = p[0] ^ p[1] ^ p[2]

print("=" * 78)
print("TRIPOLE  T = p1 ^ p2 ^ p3   for", P)
print("=" * 78)
print("grades(T) =", grades(T))
print("T in null basis:")
print("  ", format_null(T))
print()

# convenience grade-1 / higher-grade probe blades
G1 = {
    'eo': eo, 'einf': einf, 'eobar': eobar, 'einfbar': einfbar,
    'e1': e1, 'e2': e2,
    'eo1': eo1, 'eo2': eo2, 'eo3': eo3,
    'einf1': einf1, 'einf2': einf2, 'einf3': einf3,
}
GHI = {
    'Iod (eo▷)': Iod, 'Iinfd (einf▷)': Iinfd,
    'Io': Io, 'Iinf': Iinf, 'Ieps': Ieps, 'I': I,
}


def line(s=78):
    print("-" * s)


def show(label, mv):
    if is_zero(mv):
        print(f"  {label:18s} -> 0")
        return
    g = grades(mv)
    print(f"  {label:18s} -> grade {g}:  {format_null(mv)}")


# ── 1. INNER PRODUCT  v | T  (grade-1 probes) ─────────────────────────────────
print("\n### 1.  v | T   (contraction by grade-1 special blades) -> grade 2")
line()
for name, v in G1.items():
    show(f"{name} | T", v | T)

# ── 2. INNER PRODUCT  B | T  (higher-grade probes) ────────────────────────────
print("\n### 2.  B | T   (contraction by higher-grade blades)")
line()
for name, B in GHI.items():
    show(f"{name} | T", B | T)

# ── 3. WEDGE  T ^ x ────────────────────────────────────────────────────────────
print("\n### 3.  T ^ x   (outer product) -> grade up")
line()
for name, v in G1.items():
    show(f"T ^ {name}", T ^ v)
print()
for name, B in GHI.items():
    show(f"T ^ {name}", T ^ B)

# ── 4. MEET  T & x ──────────────────────────────────────────────────────────────
print("\n### 4.  T & x   (meet / regressive)")
line()
for name, B in GHI.items():
    show(f"T & {name}", T & B)

# ── 5. the circumcircle closed form ───────────────────────────────────────────
print("\n### 5.  circumcircle  =  T ^ Iod ^ Iinfd   (grade-7 OPNS conic)")
line()
cc = T ^ Iod ^ Iinfd
print("  grades:", grades(cc))
ipns = cc * I_inv
A, B, C, D, E, F = ipns_to_coeffs(ipns)
cx, cy = -D / (2 * A), -E / (2 * B)
R2 = cx * cx + cy * cy - F / A
print(f"  IPNS conic coeffs (A,B,C,D,E,F) = {np.array([A,B,C,D,E,F])}")
print(f"  => center ({cx:.6f}, {cy:.6f}),  R = {R2**0.5:.6f},   C≈0? {abs(C)<1e-9}, A≈B? {abs(A-B)<1e-9}")
# verify the 3 points are on it
for (x, y) in P:
    val = A*x*x + B*y*y + C*x*y + D*x + E*y + F
    print(f"     point ({x:+.2f},{y:+.2f}) on circle? residual = {val:+.2e}")

# ── 6. signed area  Σ  =  the  eo^e1^e2  coefficient ──────────────────────────
print("\n### 6.  signed area  Σ  (the eo∧e1∧e2 component of T)")
line()
Sigma_geo = (P[0][0]*(P[1][1]-P[2][1]) + P[1][0]*(P[2][1]-P[0][1])
             + P[2][0]*(P[0][1]-P[1][1]))
# extract by contracting with the reciprocal blade of eo^e1^e2.
# eo·einf=-1 so a clean scalar pick is  (einf ^ e1 ^ e2) | T  (up to sign).
sigma_pick = float(((einf ^ e1 ^ e2) | T).e)
print(f"  geometric  Σ = x1(y2-y3)+...           = {Sigma_geo:+.6f}")
print(f"  (einf ^ e1 ^ e2) | T                   = {sigma_pick:+.6f}")
print(f"  null coeff of 'eo1^e1^e2' in T         = {to_null_basis(T).get('eo1^e1^e2'):+.6f}")
print(f"  signed triangle area = Σ/2             = {Sigma_geo/2:+.6f}")

# ── 6b.  is  Iod | T  the circumcircle? ───────────────────────────────────────
print("\n### 6b.  Iod ⌋ T   vs   circumcircle  (both grade-1 IPNS?)")
line()
v = Iod | T
A2, B2, C2, D2, E2, F2 = ipns_to_coeffs(v)
print(f"  Iod ⌋ T coeffs (A,B,C,D,E,F) = {np.array([A2,B2,C2,D2,E2,F2])}")
cx2, cy2 = -D2/(2*A2), -E2/(2*B2)
R2b = cx2*cx2 + cy2*cy2 - F2/A2
print(f"     => center ({cx2:.6f},{cy2:.6f}), R={R2b**0.5:.6f}  (circumcircle was ({cx:.6f},{cy:.6f}), R={R2**0.5:.6f})")
print(f"     same circle as T∧Iod∧Iinfd? {abs(cx2-cx)<1e-6 and abs(cy2-cy)<1e-6 and abs(R2b-R2)<1e-6}")

# ── 6c.  boundary identity:  einf ⌋ T = -(p2∧p3 - p1∧p3 + p1∧p2) ──────────────
print("\n### 6c.  einf ⌋ T  is the alternating sum of edge dipoles")
line()
boundary = -((p[1]^p[2]) - (p[0]^p[2]) + (p[0]^p[1]))
print(f"  einf ⌋ T == -(pp23 - pp13 + pp12)?  {is_zero((einf|T) - boundary)}")
print(f"     (because einf·p_k = -1 for every point — verify: "
      f"{[round(float((einf|pk).e),3) for pk in p]})")
print(f"  eo  ⌋ T  weights edges by -|p_k|²/2  (eo·p_k = "
      f"{[round(float((eo|pk).e),3) for pk in p]} == -½|p_k|²)")

# ── 7. connecting-line dipole reduction:  ℓ | T  -> single dipole ─────────────
print("\n### 7.  ℓ_ij | T   (contract with line through p_i,p_j)  -> one dipole")
line()


def line_through(pa, pb):
    """IPNS grade-1 finite line through two (x,y) points (D e1+E e2+F einf form)."""
    (x1, y1), (x2, y2) = pa, pb
    nx, ny, d = (y2 - y1), (x1 - x2), (x2 * y1 - x1 * y2)
    return nx * e1 + ny * e2 - d * einf   # F=d -> s_inf=-d/2 each => -d*einf


def parallel(a, b, tol=1e-9):
    """True if 2-blades a,b are scalar multiples (same dipole up to scale)."""
    ka = [(k, float(v)) for k, v in a.items() if abs(float(v)) > tol]
    if not ka:
        return is_zero(b, tol)
    k0, v0 = ka[0]
    s = next((float(w) for kk, w in b.items() if kk == k0), 0.0) / v0
    return is_zero(a * s - b, tol)


# contract with the line through p_i,p_j: (ℓ·p_i)=(ℓ·p_j)=0 kill two terms,
# leaving the surviving dipole ∝ p_i∧p_j (the same two points the line meets).
for (i, j, k) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
    ell = line_through(P[i], P[j])
    dip = ell | T
    g = grades(dip)
    edge = p[i] ^ p[j]                       # the dipole through p_i, p_j
    on_i = is_zero(p[i] ^ dip)
    on_j = is_zero(p[j] ^ dip)
    on_k = is_zero(p[k] ^ dip)               # the third point should NOT lie on it
    print(f"  ℓ_{i+1}{j+1} ⌋ T -> grade {g};  ∝ p{i+1}∧p{j+1}? {parallel(dip, edge)};"
          f"  p{i+1},p{j+1} on it? {on_i and on_j};  p{k+1} on it? {on_k}")

# ── 8. T^2 and reality ─────────────────────────────────────────────────────────
print("\n### 8.  T * T   (square / reality / magnitude)")
line()
TT = float((T * T).e)
d2 = lambda a, b: (a[0]-b[0])**2 + (a[1]-b[1])**2
prod_d2 = d2(P[0], P[1]) * d2(P[0], P[2]) * d2(P[1], P[2])
area = Sigma_geo / 2
print(f"  T² (scalar)              = {TT:.6f}")
print(f"  ¼·(d12·d13·d23)²         = {prod_d2/4:.6f}   (product of squared edges /4)")
print(f"  Σ²·R²  = (2·area·R)²     = {Sigma_geo**2 * R2:.6f}")
print(f"  => |T| = |Σ|·R = ∏d/2,  always ≥ 0  (a real tripole never 'flips' sign)")
print()


# ── 9. SYMBOLIC confirmations of the key scalar extractions ───────────────────
print("\n### 9.  symbolic confirmation (generic x_i, y_i)")
line()
xs = sp.symbols('x1 x2 x3', real=True)
ys = sp.symbols('y1 y2 y3', real=True)


def psym(x, y):
    return (eo + x*e1 + y*e2
            + (x*x/2)*einf1 + (y*y/2)*einf2 + x*y*einf3)


Tsym = psym(xs[0], ys[0]) ^ psym(xs[1], ys[1]) ^ psym(xs[2], ys[2])
sig_sym = sp.simplify(((einf ^ e1 ^ e2) | Tsym).e)
Sigma = (xs[0]*(ys[1]-ys[2]) + xs[1]*(ys[2]-ys[0]) + xs[2]*(ys[0]-ys[1]))
print("  (einf ^ e1 ^ e2) | T  =", sig_sym)
print("  Σ (signed, x2)        =", sp.expand(Sigma))
print("  match (=Σ)?           ", sp.simplify(sig_sym - Sigma) == 0)

# gauge annihilations (the OPNS shadow of §3 results 4 & 5), symbolic:
print("\n  gauge facts (symbolic, all points):")
print("    einfbar | T == 0 ?", is_zero(einfbar | Tsym))
print("    einf3   | T == 0 ?", is_zero(einf3 | Tsym))
print("    (einf1 - einf2) | T == 0 ?", is_zero((einf1 - einf2) | Tsym))
print("    einfbar·p, einf3·p (generic) =",
      sp.simplify((einfbar | psym(xs[0], ys[0])).e),
      ",", sp.simplify((einf3 | psym(xs[0], ys[0])).e))

# Iod ⌋ T is always in the circle subfamily (A=B, C=0), for any triangle.
from ccga.algebra import alg as _alg


def _csym(mv, key):
    for kk, vv in mv.items():
        if _alg.bin2canon.get(kk, '') == key:
            return vv
    return 0


v = Iod | Tsym
c3, c4, c5 = _csym(v, 'e3'), _csym(v, 'e4'), _csym(v, 'e5')
c6, c7, c8 = _csym(v, 'e6'), _csym(v, 'e7'), _csym(v, 'e8')
As = -(c3 + c6) / 4
Bs = -(c4 + c7) / 4
Cs = -(c5 + c8) / 2
print("\n  Iod ⌋ T isotropy (symbolic):  A-B =", sp.simplify(As - Bs),
      ",  C =", sp.simplify(Cs), " => always a circle-type vector")
print("\n  done.")
