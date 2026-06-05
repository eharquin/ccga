"""Phase 0 follow-up: pin down the circumcircle (T) and the 4-point conic (Q)."""
import numpy as np
from ccga.algebra import (alg, e1, e2, eo, einf, eobar, einfbar,
                          einf1, einf2, einf3, Iod, Iinfd, Io, Iinf, Ieps, I, I_inv)
from ccga.point import point
from ccga.operations import grades, is_zero
from ccga.classify import classify, ipns_to_coeffs

P3 = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
P4 = P3 + [(1.5, 2.3)]
p3 = [point(*c) for c in P3]
p4 = [point(*c) for c in P4]
T = p3[0] ^ p3[1] ^ p3[2]
Q = p4[0] ^ p4[1] ^ p4[2] ^ p4[3]


def coeffs_norm(s):
    c = np.array(ipns_to_coeffs(s), float)
    n = np.linalg.norm(c)
    return c / n if n > 1e-12 else c


print('── TRIPOLE circumcircle candidates (IPNS grade-1) ──')
cc_ref = coeffs_norm(Iod | ((T ^ Iinfd) * I_inv))     # notebook route
for name, s in [('Iod | T', Iod | T),
                ('Ieps | T', Ieps | T),
                ('(T^Iod^Iinfd)*I_inv', (T ^ Iod ^ Iinfd) * I_inv)]:
    c = coeffs_norm(s)
    agree = min(np.linalg.norm(c - cc_ref), np.linalg.norm(c + cc_ref))
    print(f'  {name:24} (A,B,C,D,E,F)~{np.round(c,3)}  ==circumcircle: {agree<1e-9}')
# verify the 3 points lie on Iod|T
print('  incidence p_i | (Iod|T) == 0 :', [abs(float((p | (Iod | T)).e)) < 1e-9 for p in p3])

print('\n── QUADPOLE: a single grade-7 conic through the 4 points ──')
# conic through the 4 points + a 5th point, in canonical OPNS form Iod ^ p1..p5
for p5name, p5 in [('einf', einf), ('eo', eo), ('e1', e1)]:
    C = Iod ^ Q ^ p5
    inc = [is_zero(p ^ C) for p in p4]
    print(f'  Iod ^ Q ^ {p5name:5}: grade {grades(C)}  type={classify(C)["type"]:10}  4-pt incidence={inc}')

print('\n  bare gauge-blade products that gave grade-7 conics:')
for name, C in [('Q ^ Io', Q ^ Io)]:
    inc = [is_zero(p ^ C) for p in p4]
    print(f'  {name:10}: grade {grades(C)}  type={classify(C)["type"]:10}  4-pt incidence={inc}')

print('\n  user\'s stated anchor  Q ^ einf ^ Iinfd :',
      'ZERO' if is_zero(Q ^ einf ^ Iinfd) else grades(Q ^ einf ^ Iinfd))
print('  why:  einf ^ Iinfd ∝ Iinf ?',
      not is_zero(einf ^ Iinfd),
      ' and  Q ^ Iinf == 0 ?', is_zero(Q ^ Iinf))
