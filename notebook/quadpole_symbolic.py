"""Phase C ground truth: symbolic grade-4 general form of Q = p1^p2^p3^p4 (r=0)."""
import sympy as sp, itertools
from ccga.algebra import e1, e2, eo, eo1, eo2, eo3, einf1, einf2, einf3

_recip = {'eo1': -einf1, 'eo2': -einf2, 'eo3': -einf3, 'e1': e1, 'e2': e2,
          'einf1': -eo1, 'einf2': -eo2, 'einf3': -eo3}
_names = ['eo1', 'eo2', 'eo3', 'e1', 'e2', 'einf1', 'einf2', 'einf3']


def null_coeffs_symk(mv, k):
    """Null-basis components of a grade-k multivector, as {blade_name: sympy}."""
    out = {}
    for combo in itertools.combinations(range(8), k):
        recip_blade = _recip[_names[combo[-1]]]
        for idx in reversed(combo[:-1]):
            recip_blade = _recip[_names[idx]] ^ recip_blade
        c = sp.nsimplify(sp.simplify((mv | recip_blade).e))
        if c != 0:
            out['^'.join(_names[i] for i in combo)] = sp.factor(c)
    return out


def sym_point(cx, cy):
    return eo + cx*e1 + cy*e2 + (cx**2/2)*einf1 + (cy**2/2)*einf2 + cx*cy*einf3


xs = sp.symbols('x1 x2 x3 x4', real=True)
ys = sp.symbols('y1 y2 y3 y4', real=True)
p = [sym_point(xs[i], ys[i]) for i in range(4)]
Q = p[0] ^ p[1] ^ p[2] ^ p[3]

comps = null_coeffs_symk(Q, 4)
print(f'# {len(comps)} nonzero grade-4 components\n')
for name, c in comps.items():
    print(f'{name}:\n    {c}\n')
