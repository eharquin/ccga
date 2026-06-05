"""
Dual tripole probe.

T  = p1 ^ p2 ^ p3            (grade 3, OPNS)
T* = T * I_inv  = dual(T)    (grade 5, IPNS)

We probe T* with the special blades under |, ^, & and confirm the duality
dictionary that ties every grade-5 IPNS product back to a grade-3 OPNS one:

    v ⌋ T* = (v ∧ T)*          (contraction  <->  wedge)
    v ∧ T* = (v ⌋ T)*          (wedge        <->  contraction)
    membership   q ∧ T = 0  <=>  q ⌋ T* = 0

so the OPNS facts (boundary einf⌋T, gauge zeros, circumcircle, magnitude)
all reappear on the IPNS side with | and ^ swapped and & made useful.
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
from ccga.operations import grades, is_zero, dual
from ccga.classify import ipns_to_coeffs

np.set_printoptions(suppress=True, precision=6)

P = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p = [point(*c) for c in P]
T = p[0] ^ p[1] ^ p[2]
Ts = dual(T)                         # T*  = T * I_inv,  grade 5

print("=" * 78)
print("DUAL TRIPOLE  T* = dual(T) = T * I_inv")
print("=" * 78)
print("grades(T*) =", grades(Ts))
print("T* in null basis:")
print("  ", format_null(Ts))
print(f"\n(T*)² = {float((Ts*Ts).e):.6f}    T² = {float((T*T).e):.6f}    "
      f"ratio = {float((Ts*Ts).e)/float((T*T).e):+.3f}  (= I² = -1)")


def line(s=78):
    print("-" * s)


def show(label, mv):
    if is_zero(mv):
        print(f"  {label:20s} -> 0")
        return
    print(f"  {label:20s} -> grade {grades(mv)}:  {format_null(mv)}")


G1 = {'eo': eo, 'einf': einf, 'eobar': eobar, 'einfbar': einfbar,
      'e1': e1, 'e2': e2, 'eo3': eo3, 'einf3': einf3}
GHI = {'Iod': Iod, 'Iinfd': Iinfd, 'Io': Io, 'Iinf': Iinf, 'Ieps': Ieps}

# ── 1. duality dictionary (the organising identity) ───────────────────────────
print("\n### 1.  duality dictionary:  v ⌋ T* = (v∧T)* ,   v ∧ T* = (v⌋T)*")
line()
for name, v in G1.items():
    ok_in = is_zero((v | Ts) - dual(v ^ T))
    ok_we = is_zero((v ^ Ts) - dual(v | T))
    print(f"  v={name:7s}:  v⌋T* == (v∧T)*  {ok_in};   v∧T* == (v⌋T)*  {ok_we}")

# ── 2. contraction  v ⌋ T*  (grade 5 -> 4) ────────────────────────────────────
print("\n### 2.  v ⌋ T*   (grade-1 contraction) -> grade 4  [= dual of v∧T]")
line()
for name, v in G1.items():
    show(f"{name} | T*", v | Ts)

# ── 3. wedge  v ∧ T*  (grade 5 -> 6) ──────────────────────────────────────────
print("\n### 3.  v ∧ T*   (grade-1 wedge) -> grade 6  [= dual of v⌋T]")
line()
for name, v in G1.items():
    show(f"T* ^ {name}", Ts ^ v)

# ── 4. gauge zeros, dualised:  einf̄ ∧ T* = 0 , einf3 ∧ T* = 0 ─────────────────
print("\n### 4.  gauge (dual of  einf̄⌋T=0, einf3⌋T=0  ->  wedge zeros)")
line()
print("  einfbar ∧ T* == 0 ?", is_zero(einfbar ^ Ts))
print("  einf3   ∧ T* == 0 ?", is_zero(einf3   ^ Ts))
print("  einf    ∧ T* (= dual of boundary einf⌋T):  grade",
      grades(einf ^ Ts))

# ── 5. IPNS membership:  q ∧ T = 0  <=>  q ⌋ T* = 0 ───────────────────────────
print("\n### 5.  IPNS membership   q ⌋ T* = 0   (grade 4)")
line()
for (x, y) in P:
    q = point(x, y)
    print(f"  p=({x:+.2f},{y:+.2f}):  q∧T==0 {is_zero(q ^ T)};  "
          f"q⌋T*==0 {is_zero(q | Ts)}")
qoff = point(0.5, 0.5)
print(f"  off-point (0.5,0.5):  q∧T==0 {is_zero(qoff ^ T)};  "
      f"q⌋T*==0 {is_zero(qoff | Ts)}")

# ── 6. circumcircle on the IPNS side:  meet of duals ──────────────────────────
print("\n### 6.  circumcircle via meet of duals:  T* ∨ Iod* ∨ Iinfd*")
line()
cc_opns = T ^ Iod ^ Iinfd                    # grade-7 OPNS (from part 1)
cc_ipns = dual(cc_opns)                       # grade-1 reference
meet_form = (Ts & dual(Iod)) & dual(Iinfd)    # (A∧B∧C)* = A*∨B*∨C*
print("  grade(T* ∨ Iod* ∨ Iinfd*) =", grades(meet_form))
print("  equals dual(T∧Iod∧Iinfd) (up to scale)? ",
      end="")
# compare up to overall scalar
ka = [(k, float(v)) for k, v in cc_ipns.items() if abs(float(v)) > 1e-9][0]
s = next((float(w) for kk, w in meet_form.items() if kk == ka[0]), 0.0) / ka[1]
print(is_zero(meet_form - cc_ipns * s), f"(scale {s:.4f})")
A, B, C, D, E, F = ipns_to_coeffs(meet_form)
cx, cy = -D/(2*A), -E/(2*B)
print(f"  -> circle center ({cx:.6f},{cy:.6f}), R={ (cx*cx+cy*cy-F/A)**0.5:.6f}")

# ── 7. meet  T* & B  (now non-trivial: 5 + grade ≥ 3 ≥ 8) ─────────────────────
print("\n### 7.  T* & B   (meet — useful now that T* is grade 5)")
line()
for name, B_ in GHI.items():
    show(f"T* & {name}", Ts & B_)

# ── 8. symbolic confirmation of the duality dictionary ────────────────────────
print("\n### 8.  symbolic confirmation")
line()
xs = sp.symbols('x1 x2 x3', real=True)
ys = sp.symbols('y1 y2 y3', real=True)


def psym(x, y):
    return (eo + x*e1 + y*e2 + (x*x/2)*einf1 + (y*y/2)*einf2 + x*y*einf3)


Tsym = psym(xs[0], ys[0]) ^ psym(xs[1], ys[1]) ^ psym(xs[2], ys[2])
Tsym_s = dual(Tsym)
for name, v in [('einf', einf), ('einfbar', einfbar), ('eo', eo)]:
    print(f"  {name}: v⌋T* == (v∧T)* ? {is_zero((v | Tsym_s) - dual(v ^ Tsym))}; "
          f" v∧T* == (v⌋T)* ? {is_zero((v ^ Tsym_s) - dual(v | Tsym))}")
print("  einfbar ∧ T* == 0 (symbolic)?", is_zero(einfbar ^ Tsym_s))
print("  einf3   ∧ T* == 0 (symbolic)?", is_zero(einf3   ^ Tsym_s))
print("\n  done.")
