"""
CCGA classifier: grade extractor, geometric type, reality test.

Geometric types keyed by OPNS grade:
  1  → Point  (or IdealPoint if no eo component)
  2  → PointPair  (real/imaginary from P^2 sign)
  3  → LineAtInfinity  (= I_inf)
  4  → FlatPoint
  5  → ConicAtInfinity
  6  → (meet result, not a standalone object)
  7  → Conic (general); specialised to Circle/Ellipse/Hyperbola/Parabola/Line
  8  → Pseudoscalar

For grade-7 OPNS conics the subtype is determined from the IPNS (A,B,C,D,E,F):
  discriminant Δ = B² - 4AC
  Δ < 0  and  A≠0, B≠0  → Ellipse  (circle if A=B and C=0)
  Δ = 0                  → Parabola
  Δ > 0                  → Hyperbola
  A = B = C = 0          → Line (degenerate conic)
"""

from .algebra import einf, eo, Iod, to_null_basis
from .operations import grades, pure_grade, is_zero, dual, max_coeff

_TOL = 1e-9


def _is_cga_round_object(mv, tol=_TOL):
    """
    True if mv (grade 3/4/5) is a CGA round object from cga.py — i.e. its
    extracted CGA blade  X = Iod | mv  lives in the CGA subalgebra
    {e1,e2,eo1,eo2,einf1,einf2} (no eo3/einf3) and carries finite/Euclidean
    content.  This separates the family from the pure at-infinity blades
    (line_at_infinity has no finite content; conic_at_infinity carries eo3).
    """
    factors = {f for name in to_null_basis(Iod | mv, tol)
               for f in name.split('^')}
    if 'eo3' in factors or 'einf3' in factors:
        return False
    return bool(factors & {'e1', 'e2', 'eo1', 'eo2'})


def grade_of(mv, tol=_TOL):
    """Return the grade(s) present in mv (after noise-chopping)."""
    return grades(mv, tol)


def reality(mv):
    """
    Reality test for a round object (grade-1 or grade-2 OPNS).

    For grade-1 IPNS conics: use conic discriminant.
    For grade-2 OPNS point pairs: P^2 > 0 → real, < 0 → imaginary.
    Returns 'real', 'imaginary', or 'degenerate'.
    """
    sq = float((mv * mv).e)
    if abs(sq) < _TOL:
        return 'degenerate'
    return 'real' if sq > 0 else 'imaginary'


def conic_discriminant(A, B, C):
    """Δ = C² - 4AB  (sign determines conic type)."""
    return C*C - 4*A*B


def conic_subtype(A, B, C, D, E, F, tol=_TOL):
    """
    Classify a conic by its (A,B,C,D,E,F) coefficients.

    Returns one of: 'circle', 'ellipse', 'hyperbola', 'parabola',
                    'line', 'point', 'empty', 'degenerate'.
    """
    if abs(A) < tol and abs(B) < tol and abs(C) < tol:
        # No quadratic part
        if abs(D) < tol and abs(E) < tol:
            return 'degenerate'   # constant; F≠0 → empty
        return 'line'

    delta = conic_discriminant(A, B, C)

    if abs(delta) < tol:
        return 'parabola'
    elif delta < 0:
        if abs(A - B) < tol and abs(C) < tol:
            return 'circle'
        return 'ellipse'
    else:
        return 'hyperbola'


def ipns_to_coeffs(s):
    """
    Extract (A,B,C,D,E,F) from IPNS grade-1 conic s stored in the orthogonal
    basis of Algebra(5,3) (§3 result 1).

    kingdon stores multivectors in the diagonal basis e1..e8, where:
      eo1=e3+e6, eo2=e4+e7, eo3=e5+e8, einf1=(e6-e3)/2, einf2=(e7-e4)/2, einf3=(e8-e5)/2

    make_conic_ipns(A,B,C,D,E,F) stores (in orthogonal keys):
      e1:D  e2:E  e3:-2A+F/4  e4:-2B+F/4  e5:-C  e6:-2A-F/4  e7:-2B-F/4  e8:-C

    Inverse:
      A = -(c3+c6)/4    B = -(c4+c7)/4    C = -c5
      D = c1            E = c2            F = 2*(c3-c6)
    """
    from .algebra import alg

    def _c(key):
        for k, v in s.items():
            if alg.bin2canon.get(k, '') == key:
                return float(v)
        return 0.0

    c1 = _c('e1');  c2 = _c('e2')
    c3 = _c('e3');  c4 = _c('e4');  c5 = _c('e5')
    c6 = _c('e6');  c7 = _c('e7');  c8 = _c('e8')

    # eo_i lives in the symmetric part of (e_{+i}, e_{-i}); einf_i in the
    # antisymmetric part.  eo3 and einf3 BOTH touch e5,e8, so C must use the
    # symmetric combination (c5+c8)/2, not c5 alone.  (For canonical conics
    # c5==c8 and c4-c7==c3-c6, so this reduces to the old C=-c5, F=2(c3-c6).)
    A = -(c3 + c6) / 4
    B = -(c4 + c7) / 4
    C = -(c5 + c8) / 2
    D = c1;  E = c2
    F = (c3 - c6) + (c4 - c7)
    return A, B, C, D, E, F


def ipns_infinity_components(s):
    """
    Null-basis einf coefficients (s_inf1, s_inf2, s_inf3) of a grade-1 vector.

      einf_i = (e_{-i} - e_{+i})/2  →  s_inf_i = c_{-i} - c_{+i}
      i.e. s_inf1 = c6-c3,  s_inf2 = c7-c4,  s_inf3 = c8-c5.
    """
    from .algebra import alg

    def _c(key):
        for k, v in s.items():
            if alg.bin2canon.get(k, '') == key:
                return float(v)
        return 0.0

    return (_c('e6') - _c('e3'),    # s_inf1
            _c('e7') - _c('e4'),    # s_inf2
            _c('e8') - _c('e5'))    # s_inf3


def _conic_reality(sub, A, B, C, D, E, F, tol=_TOL):
    """
    Reality of a conic from its (A,B,C,D,E,F) coefficients.

    For circles/ellipses: use the full matrix discriminant to check if the
    locus is empty (imaginary) or non-empty (real).
    For hyperbola/parabola/line: real by default (have real points) unless
    degenerate.
    """
    import numpy as np
    if sub == 'line':
        return 'real'
    if sub == 'degenerate':
        return 'degenerate'
    if sub == 'hyperbola':
        return 'real'   # hyperbola always has real points
    if sub == 'parabola':
        return 'real'
    # ellipse or circle: check if the locus is empty
    # Matrix form: |A C/2 D/2|  det > 0 → real, det < 0 → imaginary, det=0 → degenerate
    #              |C/2 B E/2|
    #              |D/2 E/2 F|
    M = np.array([[A, C/2, D/2],
                  [C/2, B, E/2],
                  [D/2, E/2, F]])
    det3 = np.linalg.det(M)
    det2 = A*B - (C/2)**2   # = -Δ/4
    if abs(det3) < tol**0.5:
        return 'degenerate'
    # For an ellipse (det2 > 0): real iff det3/A < 0
    if det2 > tol:
        sign_check = det3 / A if abs(A) > tol else det3 / B
        return 'real' if sign_check < 0 else 'imaginary'
    return 'real'


def classify(mv, tol=_TOL):
    """
    Return a dict with keys:
      'grade'   : list of grades present
      'type'    : string name
      'reality' : 'real' | 'imaginary' | 'degenerate' | 'n/a'
      'coeffs'  : (A,B,C,D,E,F) for conics, else None
    """
    gs = grades(mv, tol)
    if not gs:
        return {'grade': [], 'type': 'zero', 'reality': 'n/a', 'coeffs': None}

    g = gs[0] if len(gs) == 1 else None

    # ── grade 1 ─────────────────────────────────────────────────────────────
    if g == 1:
        w  = -float((mv | einf).e)         # homogeneous (eo) coordinate
        sq = float((mv * mv).e)            # square → radius²/reality
        A, B, C, D, E, F = ipns_to_coeffs(mv)
        si1, si2, si3 = ipns_infinity_components(mv)

        has_shape = max(abs(A), abs(B), abs(C)) > tol

        if not has_shape:
            # No quadratic shape (A=B=C=0): either a line or an ideal point.
            # A LINE has isotropic infinity (si1≈si2, si3≈0); an IDEAL POINT
            # (eo-less round point) is anisotropic.
            isotropic = abs(si1 - si2) < tol and abs(si3) < tol
            if isotropic and (abs(D) > tol or abs(E) > tol):
                return {'grade': [1], 'type': 'line', 'reality': 'real',
                        'coeffs': (A, B, C, D, E, F)}
            # anisotropic, or no linear part → ideal (round) point
            return {'grade': [1], 'type': 'ideal_point', 'reality': 'n/a',
                    'coeffs': None}

        # Has shape → round object or general conic.
        sub = conic_subtype(A, B, C, D, E, F, tol)
        if sub == 'circle':
            # circle family: zero radius (null) → finite point; else circle
            if abs(sq) < tol:
                return {'grade': [1], 'type': 'point', 'reality': 'real',
                        'coeffs': None}
            r = 'real' if sq > 0 else 'imaginary'
            return {'grade': [1], 'type': 'circle', 'reality': r,
                    'coeffs': (A, B, C, D, E, F)}
        r = _conic_reality(sub, A, B, C, D, E, F, tol)
        return {'grade': [1], 'type': sub, 'reality': r,
                'coeffs': (A, B, C, D, E, F)}

    # ── grade 2 ─────────────────────────────────────────────────────────────
    if g == 2:
        r = reality(mv)
        return {'grade': [2], 'type': 'point_pair', 'reality': r,
                'coeffs': None}

    # ── grades 3/4/5: CGA round objects vs at-infinity objects ───────────────
    # CGA round objects (cga.py, built ∧ Iinfd) carry Euclidean e1/e2 content;
    # the pure at-infinity blades (line/conic at infinity) do not.
    if g in (3, 4, 5):
        if _is_cga_round_object(mv, tol):
            from . import cga
            c = cga.classify_cga(mv, tol)
            return {'grade': [g], 'type': c['type'], 'reality': c['reality'],
                    'coeffs': None}
        # no Euclidean content → at-infinity object
        at_inf = {3: 'line_at_infinity', 4: 'flat_point', 5: 'conic_at_infinity'}
        return {'grade': [g], 'type': at_inf[g], 'reality': 'n/a', 'coeffs': None}

    # ── grade 6 ─────────────────────────────────────────────────────────────
    if g == 6:
        return {'grade': [6], 'type': 'meet_result_grade6', 'reality': 'n/a',
                'coeffs': None}

    # ── grade 7 ─────────────────────────────────────────────────────────────
    if g == 7:
        ipns = dual(mv)
        A, B, C, D, E, F = ipns_to_coeffs(ipns)
        sub = conic_subtype(A, B, C, D, E, F, tol)
        r = _conic_reality(sub, A, B, C, D, E, F, tol)
        return {'grade': [7], 'type': sub, 'reality': r,
                'coeffs': (A, B, C, D, E, F)}

    # ── grade 8 ─────────────────────────────────────────────────────────────
    if g == 8:
        return {'grade': [8], 'type': 'pseudoscalar', 'reality': 'n/a',
                'coeffs': None}

    return {'grade': gs, 'type': 'mixed', 'reality': 'n/a', 'coeffs': None}
