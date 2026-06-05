"""
Phase 0 — complementary gauge-blade analysis of the n-pole.

Probe T = p1^p2^p3 (tripole) and Q = p1^p2^p3^p4 (quadpole) against the special
gauge blades, under wedge (^), left contraction (|), dual (*I_inv) and meet (&),
and classify each non-zero result.  Surfaces which products yield recognisable
objects (circumcircle, pencil member, flats, ...) as closed-form GA expressions.

Run:  .venv/bin/python notebook/blade_analysis.py
"""
import numpy as np
from ccga.algebra import (
    alg, e1, e2, eo, einf, eobar, einfbar,
    Iod, Iinfd, Io, Iinf, Ieps, I, I_inv,
)
from ccga.point import point
from ccga.operations import grades, is_zero
from ccga.classify import classify

BLADES = [
    ('eo', eo), ('einf', einf), ('eobar', eobar), ('einfbar', einfbar),
    ('Iod', Iod), ('Iinfd', Iinfd), ('Io', Io), ('Iinf', Iinf),
    ('Ieps', Ieps), ('I', I),
]


def _type(mv):
    if is_zero(mv):
        return '0', []
    try:
        c = classify(mv)
        return c['type'], c['grade']
    except Exception as e:
        return f'?({e})', grades(mv)


def sweep(name, X):
    print(f'\n===== {name}  grades={grades(X)} =====')
    print(f'{"blade":8} {"op":4} {"grade":>6}  type')
    for bname, B in BLADES:
        for op, fn in [('^', lambda a, b: a ^ b),
                       ('|', lambda a, b: a | b),
                       ('&', lambda a, b: a & b)]:
            try:
                R = fn(X, B)
            except Exception:
                continue
            if is_zero(R):
                continue
            t, g = _type(R)
            print(f'{bname:8} {op:4} {str(g):>6}  {t}')


def anchors(T, Q):
    print('\n##### confirmed anchors #####')
    A1 = T ^ Iod ^ Iinfd
    print('T ^ Iod ^ Iinfd :', grades(A1), classify(A1)['type'], classify(A1)['reality'])

    # cross-check against the IPNS circumcircle route used in the notebook
    from ccga.classify import ipns_to_coeffs
    ipns_route = Iod | ((T ^ Iinfd) * I_inv)
    opns_route_ipns = A1 * I_inv  # dual of the OPNS grade-7 conic -> grade-1
    ca = np.array(ipns_to_coeffs(ipns_route))
    cb = np.array(ipns_to_coeffs(opns_route_ipns))
    # normalise both and compare (same conic up to scale)
    ca = ca / np.linalg.norm(ca); cb = cb / np.linalg.norm(cb)
    agree = min(np.linalg.norm(ca - cb), np.linalg.norm(ca + cb))
    print('   circumcircle OPNS vs IPNS route agree (||diff||):', round(float(agree), 12))

    A2 = Q ^ einf ^ Iinfd
    print('Q ^ einf ^ Iinfd :', grades(A2), classify(A2)['type'], classify(A2)['reality'])
    # is it a member of the pencil through the 4 points? (each p_i on it)
    on = [abs(float((p ^ A2).e if hasattr(p ^ A2, 'e') else 0)) for p in []]
    # incidence: p ^ A2 == 0 for OPNS grade-7 conic
    print('   pencil incidence  p_i ^ (Q^einf^Iinfd) == 0 :',
          [is_zero(p ^ A2) for p in PTS4])


# ── test configurations ───────────────────────────────────────────────────────
P3 = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
P4 = P3 + [(1.5, 2.3)]
PTS3 = [point(*c) for c in P3]
PTS4 = [point(*c) for c in P4]
T = PTS3[0] ^ PTS3[1] ^ PTS3[2]
Q = PTS4[0] ^ PTS4[1] ^ PTS4[2] ^ PTS4[3]

if __name__ == '__main__':
    sweep('Tripole T', T)
    sweep('Quadpole Q', Q)
    anchors(T, Q)
