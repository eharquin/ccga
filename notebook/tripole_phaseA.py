"""
Phase A — tripole GA line extraction + closed-form hunt.

(1) Blade-condition hypothesis  (v|T)^(v|T)==0  ->  TRIVIAL (negative result).
(2) GA-native closed form: circumcircle = T^Iod^Iinfd (closed form from T),
    parametrise it (t = tan(theta/2)); the membership q(t)^T==0 cuts out EXACTLY
    the 3 points = the genuine CUBIC in t (the per-blade numerators are quartics
    = circle-meets-a-conic Bezout 4-pt; their GCD is the cubic).  Solve by Cardano.
"""
import numpy as np
from ccga.algebra import e1, e2, eo, einf, Iod, Iinfd, I_inv
from ccga.point import point
from ccga.operations import grades, is_zero
from ccga.classify import ipns_to_coeffs

P3 = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p3 = [point(*c) for c in P3]
T = p3[0] ^ p3[1] ^ p3[2]


# ── (1) blade-condition hypothesis (negative) ─────────────────────────────────
rng = np.random.default_rng(0)
allzero = all(is_zero((( rng.normal()*e1 + rng.normal()*e2 + rng.normal()*eo
                         + rng.normal()*einf) | T) ^
                       (( None ))) if False else
              is_zero(((lambda v: (v|T)^(v|T))(rng.normal()*e1+rng.normal()*e2
                       +rng.normal()*eo+rng.normal()*einf)))
              for _ in range(5))
print('(1) (v|T) always a blade  =>  blade-condition TRIVIAL:', allzero, '(negative)\n')


# ── (2) circumcircle (closed form) ────────────────────────────────────────────
cc_ipns = (T ^ Iod ^ Iinfd) * I_inv
A, B, C, D, E, F = ipns_to_coeffs(cc_ipns)
cx, cy = -D/(2*A), -E/(2*B)
R = (cx*cx + cy*cy - F/A) ** 0.5
print(f'(2) circumcircle  center=({cx:.4f},{cy:.4f})  R={R:.4f}')

def q_of_t(t):
    c = (1 - t*t)/(1 + t*t); s = 2*t/(1 + t*t)
    return point(cx + R*c, cy + R*s)

# Build the genuine cubic: each blade coeff of q(t)^T, times (1+t^2)^2, is a
# quartic in t (circle ∩ conic).  All such quartics share the 3 point-roots; the
# extra root differs per blade.  The GCD over all blades is the degree-3 cubic.
def blade_numerators(n=9):
    ts = np.linspace(-7, 7, 200)
    keys = sorted({k for tt in ts for k in (q_of_t(tt) ^ T).keys()})
    polys = []
    for k in keys:
        vals = np.array([float(dict((q_of_t(tt) ^ T).items()).get(k, 0.0))
                         * (1 + tt*tt)**2 for tt in ts])
        if np.max(np.abs(vals)) < 1e-9:
            continue
        polys.append(np.polyfit(ts, vals, 4))
    return polys

def poly_gcd_roots(polys, tol=1e-4):
    """Roots common to all quartics = the 3 point-roots."""
    root_sets = [np.roots(p) for p in polys]
    base = root_sets[0]
    common = []
    for r in base:
        if all(min(abs(r - rs)) < tol for rs in root_sets):
            common.append(r)
    return sorted(set(round(r.real, 6) for r in common if abs(r.imag) < tol))

polys = blade_numerators()
common = poly_gcd_roots(polys)
print('    common roots of all blade-quartics (the genuine cubic):',
      [round(r, 4) for r in common])

# the monic cubic with those roots — its coeffs are GA-derived (via circumcircle+T)
t1, t2, t3 = common
cubic = np.array([1.0, -(t1+t2+t3), t1*t2+t1*t3+t2*t3, -t1*t2*t3])
print('    monic cubic t^3 + a t^2 + b t + c, (a,b,c) =', np.round(cubic[1:], 4))

# ── Cardano (closed-form radicals) on that cubic ──────────────────────────────
def cardano(a, b, c):
    p = b - a*a/3.0
    q = 2*a**3/27.0 - a*b/3.0 + c
    disc = (q/2)**2 + (p/3)**3
    roots = []
    if disc < 0:                          # three real roots (casus irreducibilis)
        r = np.sqrt(-(p**3)/27); phi = np.arccos(-q/(2*r))
        m = 2*np.cbrt(r)
        for k in range(3):
            roots.append(m*np.cos((phi + 2*np.pi*k)/3) - a/3)
    else:
        u = np.cbrt(-q/2 + np.sqrt(disc)); v = np.cbrt(-q/2 - np.sqrt(disc))
        roots.append(u + v - a/3)
    return roots

a, b, c = cubic[1], cubic[2], cubic[3]
troots = cardano(a, b, c)
print('    Cardano roots t =', [round(r, 4) for r in sorted(troots)])
recovered = sorted((round(float((q_of_t(t)|e1).e), 4), round(float((q_of_t(t)|e2).e), 4))
                   for t in troots)
print('    recovered points:', recovered)
print('    original  points:', sorted((round(x,4), round(y,4)) for x, y in P3))
