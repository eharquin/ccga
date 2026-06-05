"""
Phase B — quadpole GA pairing extraction (Ferrari, GA-native pencil).

Pencil of conics through the 4 points = { (Iod ^ Q ^ p5) * I_inv : p5 } — the
GA-native seed (replaces numpy.svd).  Two choices of p5 give two generators.
Resolvent cubic = degenerate members det(M(t))=0 (solvable by Cardano); each
degenerate conic splits into two lines, each line carries a dipole -> two ±√.
"""
import numpy as np
from ccga.algebra import e1, e2, eo, einf, Iod, Iinfd, I_inv
from ccga.point import point
from ccga.operations import grades, is_zero
from ccga.classify import ipns_to_coeffs

P4 = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9), (1.5, 2.3)]
p4 = [point(*c) for c in P4]
Q = p4[0] ^ p4[1] ^ p4[2] ^ p4[3]


# ── GA-native pencil (two generators from two 5th points) ─────────────────────
def conic_coeffs_through(p5):
    return np.array(ipns_to_coeffs((Iod ^ Q ^ p5) * I_inv), float)

G1 = conic_coeffs_through(einf)
G2 = conic_coeffs_through(eo)
print('pencil generators (A,B,C,D,E,F):')
print('  G1 = Iod^Q^einf :', np.round(G1, 3))
print('  G2 = Iod^Q^eo   :', np.round(G2, 3))


# ── resolvent cubic: degenerate members det(M(t)) = 0 ─────────────────────────
def conic_mat(c):
    A, B, C, D, E, F = c
    return np.array([[A, C/2, D/2], [C/2, B, E/2], [D/2, E/2, F]])

# det(M(G1 + t G2)) is a cubic in t -> 3 pairings.  Coeffs by 4 evaluations.
ts = np.array([-2.0, -1.0, 0.0, 1.0])
det_vals = [np.linalg.det(conic_mat(G1 + t*G2)) for t in ts]
cub = np.polyfit(np.append(ts, 2.0),
                 det_vals + [np.linalg.det(conic_mat(G1 + 2.0*G2))], 3)

def cardano(coef):
    a3, a2, a1, a0 = coef
    a, b, c = a2/a3, a1/a3, a0/a3
    p = b - a*a/3.0; q = 2*a**3/27.0 - a*b/3.0 + c
    disc = (q/2)**2 + (p/3)**3
    out = []
    if disc < 0:
        r = np.sqrt(-(p**3)/27); phi = np.arccos(np.clip(-q/(2*r), -1, 1))
        m = 2*np.cbrt(r)
        out = [m*np.cos((phi + 2*np.pi*k)/3) - a/3 for k in range(3)]
    else:
        u = np.cbrt(-q/2 + np.sqrt(disc)); v = np.cbrt(-q/2 - np.sqrt(disc))
        out = [u + v - a/3]
    return out

roots = cardano(cub)
print('resolvent cubic roots (Cardano) t =', [round(r, 4) for r in roots])


# ── each degenerate conic -> two lines -> two dipoles (±√) ─────────────────────
def lines_of(coef):
    M = conic_mat(coef); w, V = np.linalg.eigh(M); i = np.argsort(w)
    u = np.sqrt(max(w[i[2]], 0)) * V[:, i[2]]
    v = np.sqrt(max(-w[i[0]], 0)) * V[:, i[0]]
    return (u + v), (u - v)

KEYS = sorted((point(0.1, 0.2) ^ Q).keys())
def pQ(x, y):
    d = dict((point(x, y) ^ Q).items())
    return np.array([float(d.get(k, 0.0)) for k in KEYS])

def dipole_on_line(L):
    a, b, c = L; n2 = a*a + b*b
    base = np.array([-a*c, -b*c]) / n2; dirv = np.array([-b, a]) / np.sqrt(n2)
    L0, L1, L2 = pQ(*(base - dirv)), pQ(*base), pQ(*(base + dirv))
    A = 0.5*(L0 + L2) - L1; B = 0.5*(L2 - L0)
    j = int(np.argmax(np.abs(A))); disc = B[j]**2 - 4*A[j]*L1[j]
    return [tuple(np.round(base + r*dirv, 4))
            for r in ((-B[j] + disc**0.5)/(2*A[j]), (-B[j] - disc**0.5)/(2*A[j]))]

print('\neach pairing (resolvent root) -> two ±√ dipoles:')
for r in roots:
    La, Lb = lines_of(G1 + r*G2)
    print(f'  t={r:+.3f}:  {dipole_on_line(La)}  |  {dipole_on_line(Lb)}')

# full extraction from one pairing
r0 = roots[0]
La, Lb = lines_of(G1 + r0*G2)
got = sorted(dipole_on_line(La) + dipole_on_line(Lb))
print('\nextracted:', got)
print('original :', sorted((round(x,4), round(y,4)) for x, y in P4))

# Q = pp_ij ^ pp_kl structural identity (all 3 pairings)
print('\nstructural identity Q = ± pp_ij ^ pp_kl:')
for (i,j),(k,l) in [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]:
    blade = (p4[i]^p4[j]) ^ (p4[k]^p4[l])
    err = min(max(abs(v) for v in (Q-blade).values()), max(abs(v) for v in (Q+blade).values()))
    print(f'  pairing ({i+1}{j+1})({k+1}{l+1}):  err={err:.1e}')
