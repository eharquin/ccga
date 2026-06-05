"""
CCGA algebra setup: R^{5,3,0}, null basis, special blades, pseudoscalar.

Basis layout in kingdon Algebra(5,3) — positive axes first, then negative:
  e1, e2          → Euclidean directions  (square +1)
  e3, e4, e5      → e_{+1}, e_{+2}, e_{+3}  (square +1)
  e6, e7, e8      → e_{-1}, e_{-2}, e_{-3}  (square -1)

Null working basis:
  eo_i  = e_{+i} + e_{-i}      →  eo1=e3+e6, eo2=e4+e7, eo3=e5+e8
  einf_i = (e_{-i} - e_{+i})/2 → einf1=(e6-e3)/2, ...

  eo_i^2 = 0,  einf_i^2 = 0,  eo_i · einf_i = -1.

Dual convention (fixed once, §4 / result 7):
  C_ipns = C_opns * I_inv   (right-multiply by I^{-1})
"""

from kingdon import Algebra as _Algebra

# ── algebra instance ────────────────────────────────────────────────────────
alg = _Algebra(5, 3)

def _b(key):
    return alg.multivector({key: 1})

# ── orthogonal (diagonal) basis ─────────────────────────────────────────────
e1  = _b('e1');  e2  = _b('e2')          # Euclidean
ep1 = _b('e3');  ep2 = _b('e4');  ep3 = _b('e5')   # positive conformal
em1 = _b('e6');  em2 = _b('e7');  em3 = _b('e8')   # negative conformal

# ── null working basis ───────────────────────────────────────────────────────
eo1   = ep1 + em1
eo2   = ep2 + em2
eo3   = ep3 + em3
einf1 = (em1 - ep1) * 0.5
einf2 = (em2 - ep2) * 0.5
einf3 = (em3 - ep3) * 0.5

# ── special grade-1 blades ───────────────────────────────────────────────────
eo      = eo1 + eo2                        # combined origin
einf    = (einf1 + einf2) * 0.5            # combined infinity  (eo·einf = -1)
eobar   = eo1 - eo2                        # differential origin
einfbar = (einf1 - einf2) * 0.5           # differential infinity

# ── special higher-grade blades ──────────────────────────────────────────────
Iod   = eobar ^ eo3                        # grade-2  gauge blade (OPNS conic fix)
Iinfd = (einf1 - einf2) ^ einf3           # grade-2  infinity gauge
Io    = eo1 ^ eo2 ^ eo3                   # grade-3  origin 3-blade
Iinf  = einf1 ^ einf2 ^ einf3            # grade-3  infinity 3-blade
Ieps  = e1 ^ e2                           # grade-2  Euclidean pseudoscalar

# pseudoscalar  I = Ieps ∧ Iinf ∧ Io  (grade 8)
I     = Ieps ^ Iinf ^ Io

# cache I^{-1}: I^2 = -1  so  I^{-1} = -I
_I2 = float((I * I).e)                    # should be -1
I_inv = I * (1.0 / _I2)                   # = -I

# ── Gram-matrix verification ─────────────────────────────────────────────────
_NULL_BASIS = [e1, e2, eo1, einf1, eo2, einf2, eo3, einf3]
_NULL_NAMES = ['e1','e2','eo1','einf1','eo2','einf2','eo3','einf3']

def gram_matrix():
    """Return the 8x8 Gram matrix in the null basis."""
    n = len(_NULL_BASIS)
    import numpy as np
    G = np.zeros((n, n))
    for i, a in enumerate(_NULL_BASIS):
        for j, b in enumerate(_NULL_BASIS):
            G[i, j] = float((a | b).e)
    return G

def verify_gram():
    """Assert Gram matrix matches the target from §1."""
    import numpy as np
    G = gram_matrix()
    target = np.zeros((8, 8))
    target[0, 0] = 1.0   # e1·e1
    target[1, 1] = 1.0   # e2·e2
    for i, (oi, ii) in enumerate([(2,3),(4,5),(6,7)]):
        target[oi, ii] = -1.0
        target[ii, oi] = -1.0
    assert np.allclose(G, target, atol=1e-12), f"Gram matrix mismatch:\n{G}\nexpected:\n{target}"

def verify_pseudoscalar():
    """Assert I^2=-1, I*I_inv=1."""
    assert abs(_I2 + 1) < 1e-12, f"I^2 = {_I2}, expected -1"
    unity = float((I * I_inv).e)
    assert abs(unity - 1.0) < 1e-12, f"I*I_inv scalar = {unity}, expected 1"

def verify_special_blades():
    """Assert inner products of special grade-1 blades (§1)."""
    checks = [
        ("eo·einf",        float((eo    | einf   ).e), -1.0),
        ("eobar·einfbar",  float((eobar | einfbar).e), -1.0),
        ("eo·einfbar",     float((eo    | einfbar).e),  0.0),
        ("eo·eobar",       float((eo    | eobar  ).e),  0.0),
        ("einf·einfbar",   float((einf  | einfbar).e),  0.0),
        ("eo1·einf1",      float((eo1   | einf1  ).e), -1.0),
        ("eo2·einf2",      float((eo2   | einf2  ).e), -1.0),
        ("eo3·einf3",      float((eo3   | einf3  ).e), -1.0),
    ]
    for name, got, want in checks:
        assert abs(got - want) < 1e-12, f"{name} = {got}, expected {want}"

# ── null-basis pretty-printing ───────────────────────────────────────────────
# Display any multivector in the null working basis (requested order):
#   eo1 eo2 eo3 e1 e2 einf1 einf2 einf3
# kingdon stores everything in the orthogonal basis e1..e8; the null basis
# vectors are linear combinations of those, so their 2^8 wedge blades form an
# invertible change-of-basis matrix _T (columns = each null blade in diagonal
# coeffs).  null_coeffs = _T^{-1} @ diagonal_coeffs.  Built once at import.

import numpy as _np
from itertools import combinations as _combinations

_NULL_PRINT_BASIS = [eo1, eo2, eo3, e1, e2, einf1, einf2, einf3]
_NULL_PRINT_NAMES = ['eo1', 'eo2', 'eo3', 'e1', 'e2', 'einf1', 'einf2', 'einf3']


def _diag_vec(mv):
    """Length-256 dense vector of mv's diagonal-basis coefficients (key=bitmask)."""
    v = _np.zeros(256)
    for k, val in mv.items():
        v[k] = float(val)
    return v


def _build_null_change_of_basis():
    """Return (blade_names, T, T_inv) for the 256 null-basis blades."""
    names = []
    cols = []
    for k in range(9):
        for combo in _combinations(range(8), k):
            if k == 0:
                B = alg.multivector({0: 1.0})
                names.append('1')
            else:
                B = _NULL_PRINT_BASIS[combo[0]]
                for idx in combo[1:]:
                    B = B ^ _NULL_PRINT_BASIS[idx]
                names.append('^'.join(_NULL_PRINT_NAMES[i] for i in combo))
            cols.append(_diag_vec(B))
    T = _np.array(cols).T
    return names, T, _np.linalg.inv(T)


_NULL_BLADE_NAMES, _T, _T_inv = _build_null_change_of_basis()


def to_null_basis(mv, tol=1e-10):
    """
    Decompose mv into the null working basis (all grades).

    Returns an ordered dict {blade_name: coeff} of components above `tol`,
    where blade_name is a single null vector ('eo1', 'einf2', …), a '^'-joined
    composite ('eo1^eo3'), or '1' for the scalar part.
    """
    coeffs = _T_inv @ _diag_vec(mv)
    return {name: float(c)
            for name, c in zip(_NULL_BLADE_NAMES, coeffs)
            if abs(c) > tol}


def _fmt_num(x):
    """Compact number: integers without trailing '.0', else short float."""
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:g}"


def format_null(mv, tol=1e-10, multiline=False):
    """
    Readable null-basis string.

    Single line (default): 'eo1 + eo2 + 2*e1 + 4.5*einf2'.
    multiline=True: one blade per line, column-aligned, with the connecting
    sign trailing each line and the coefficient always shown, e.g.
        0.5   * eo1   ^ e1    +
        0.875 * eo1   ^ einf1 +
        1     * eo1   ^ einf3 +
        ...
    """
    comps = to_null_basis(mv, tol)
    if not comps:
        return "0"
    items = list(comps.items())

    if multiline:
        n = len(items)
        # split each term into (coef_str, [factor names], coeff)
        rows = [[_fmt_num(abs(c)), ([] if name == '1' else name.split('^')), c]
                for name, c in items]
        if rows[0][2] < 0:                     # leading sign on first term
            rows[0][0] = '-' + rows[0][0]
        # column widths: coefficient, then one per factor position
        coef_w = max(len(r[0]) for r in rows)
        max_f = max((len(r[1]) for r in rows), default=0)
        col_w = [max((len(r[1][i]) for r in rows if i < len(r[1])), default=0)
                 for i in range(max_f)]
        lines = []
        for i, (coef, factors, c) in enumerate(rows):
            body = coef.ljust(coef_w)
            if factors:
                cells = [factors[k].ljust(col_w[k]) for k in range(len(factors))]
                body += " * " + " ^ ".join(cells)
            if i < n - 1:                      # trailing op = sign of next term
                body += ' +' if rows[i + 1][2] >= 0 else ' -'
            lines.append(body)
        return '\n'.join(lines)

    terms = []
    for i, (name, c) in enumerate(items):
        sign = '-' if c < 0 else '+'
        mag = abs(c)
        if name == '1':
            body = _fmt_num(mag)
        elif abs(mag - 1.0) < tol:
            body = name
        else:
            body = f"{_fmt_num(mag)}*{name}"
        if i == 0:
            terms.append(f"-{body}" if c < 0 else body)
        else:
            terms.append(f"{sign} {body}")
    return ' '.join(terms)


def print_null(mv, tol=1e-10, multiline=False):
    """Print mv in the null working basis (multiline=True → one blade per line)."""
    print(format_null(mv, tol, multiline))


def verify_null_basis():
    """Round-trip: rebuild multivectors from null coeffs and compare."""
    name2col = {n: _T[:, i] for i, n in enumerate(_NULL_BLADE_NAMES)}
    # a point-like grade-1 vector, a grade-2 blade, the grade-8 pseudoscalar
    p = eo1 + eo2 + 2.0*e1 + 3.0*e2 + 2.0*einf1 + 4.5*einf2 + 6.0*einf3
    samples = [p, Iod, I]
    for mv in samples:
        comps = to_null_basis(mv, tol=1e-12)
        rebuilt = _np.zeros(256)
        for name, c in comps.items():
            rebuilt += c * name2col[name]
        assert _np.allclose(rebuilt, _diag_vec(mv), atol=1e-10), \
            f"null round-trip failed for {format_null(mv)}"


def run_all_verifications():
    verify_gram()
    verify_pseudoscalar()
    verify_special_blades()
    verify_null_basis()
    print("algebra.py: all verifications passed.")

if __name__ == "__main__":
    run_all_verifications()
