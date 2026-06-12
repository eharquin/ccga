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

from .algebra import einf, eo, e1, e2, Iod, Iinf, to_null_basis
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


def _conic_vector(mv, tol=_TOL):
    """Route any conic representation to its clean grade-1 IPNS vector s.

      grade 1 → s = mv                (already the IPNS conic)
      grade 5 → s = dual(mv ∧ Iod)    (pentapole promoted to grade-7, dualized)
      grade 7 → s = dual(mv)          (OPNS conic dualized)
    """
    from .algebra import Iod
    gs = grades(mv, tol)
    g = gs[0] if len(gs) == 1 else None
    if g == 1:
        return mv
    if g == 5:
        return dual(mv ^ Iod)
    if g == 7:
        return dual(mv)
    raise ValueError(f"not a conic representation (grades {gs}); "
                     "expected grade 1 (IPNS), 5 (pentapole), or 7 (OPNS)")


def conic_type(mv, tol=_TOL):
    """Conic subtype of a grade-1/5/7 conic representation.

    Accepts the IPNS grade-1 vector, the bare grade-5 pentapole, or the grade-7
    OPNS conic, and returns one of 'circle','ellipse','hyperbola','parabola',
    'line','point','empty','degenerate' (see conic_subtype)."""
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    return conic_subtype(A, B, C, D, E, F, tol)


def asymptotic_directions(mv, tol=_TOL):
    """Real asymptotic directions of a conic = its intersection with the line at
    infinity, the null directions of the quadratic form A vx² + C vx vy + B vy².

    Accepts a grade-1/5/7 conic.  Returns a list of unit directions (vx, vy):
      2 directions → hyperbola, 1 → parabola (axis direction), 0 → ellipse.
    Each returned direction v equals the Veronese ideal point point_at_infinity(v)
    that lies on the conic.
    """
    import numpy as np
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))

    def _unit(vx, vy):
        n = (vx*vx + vy*vy) ** 0.5
        vx, vy = vx/n, vy/n
        if vx < -tol or (abs(vx) < tol and vy < 0):   # canonical sign
            vx, vy = -vx, -vy
        return (vx, vy)

    raw = []
    if abs(A) > tol:
        disc = C*C - 4*A*B
        if disc < -tol:
            return []                                  # ellipse: no real dirs
        sq = (max(disc, 0.0)) ** 0.5
        raw = [(-C + sq) / (2*A), (-C - sq) / (2*A)]
        dirs = [_unit(vx, 1.0) for vx in raw]
    else:
        # A = 0: Q = vy (C vx + B vy); vy=0 → (1,0); C vx + B vy=0 → other root
        dirs = [_unit(1.0, 0.0)]
        if abs(C) > tol:
            dirs.append(_unit(-B, C))
        elif abs(B) > tol:
            pass                                       # double root (1,0): parabola
    # dedupe near-equal directions
    uniq = []
    for d in dirs:
        if all(abs(d[0]-u[0]) + abs(d[1]-u[1]) > 1e-7 for u in uniq):
            uniq.append(d)
    return uniq


def conic_center(mv, tol=_TOL):
    """Center (cx, cy) of a central conic, read straight off the dual conic
    vector by GA inner products (ADVANCEMENT "Conics properties").

    With s the grade-1 IPNS conic (= the dual conic C*),
      s_i  = s · einf_i  (i=1,2,3),   s_{e1}=s·e1,  s_{e2}=s·e2,
      4·Δ₂ = s_1 s_2 − s_3²,
      cx = (s_3 s_{e2} − s_2 s_{e1}) / (4Δ₂),
      cy = (s_3 s_{e1} − s_1 s_{e2}) / (4Δ₂).
    (Verified against the algebraic center; the numerator sign is the inverse of
    the originally-conjectured formula.)  Raises if the conic is non-central
    (Δ₂ ≈ 0: parabola / degenerate).
    """
    from .algebra import einf1, einf2, einf3, e1, e2

    def ip(a, b):
        return float((a | b).e)

    s = _conic_vector(mv, tol)
    s1, s2, s3 = ip(s, einf1), ip(s, einf2), ip(s, einf3)
    se1, se2 = ip(s, e1), ip(s, e2)
    det2_4 = s1*s2 - s3*s3                       # = 4·Δ₂
    if abs(det2_4) < tol:
        raise ValueError("non-central conic (Δ₂ ≈ 0: parabola or degenerate)")
    cx = (s3*se2 - s2*se1) / det2_4
    cy = (s3*se1 - s1*se2) / det2_4
    return cx, cy


def _conic_lines(A, B, C, D, E, F):
    """
    Three IPNS-line duals l1,l2,l3 (grade-7 OPNS "lines") built from the conic
    coefficients (paper §7, "Computing discriminants in QC2GA"):

      l1 = dual( A·e1 + (C/2)·e2 - (D/2)·einf )
      l2 = dual( (C/2)·e1 + B·e2 - (E/2)·einf )
      l3 = dual( (D/2)·e1 + (E/2)·e2 - F·einf )
    """
    l1 = dual(A*e1 + (C/2)*e2 - (D/2)*einf)
    l2 = dual((C/2)*e1 + B*e2 - (E/2)*einf)
    l3 = dual((D/2)*e1 + (E/2)*e2 - F*einf)
    return l1, l2, l3


def _extract_ratio(mv, target, tol):
    """Coefficient of mv along target, read off a single shared null-basis
    component (target is a single basis blade up to sign/sum of two)."""
    md, td = to_null_basis(mv, tol), to_null_basis(target, tol)
    if not td:
        return 0.0
    key = next(iter(td))
    return md.get(key, 0.0) / td[key]


# Targets spanning the grade-6 "centre flat point" subspace p_c ^ Iod ^ Iinf,
# p_c = w*eo + x*e1 + y*e2  (paper §7).
_PC_TARGET_W = eo ^ Iod ^ Iinf
_PC_TARGET_X = e1 ^ Iod ^ Iinf
_PC_TARGET_Y = e2 ^ Iod ^ Iinf
# Target spanning the grade-5 "Delta_3" line: Iod ^ Iinf.
_D3_TARGET = Iod ^ Iinf


def conic_center_meet(mv, tol=_TOL):
    """
    Centre flat point p_c = w_c·eo + x_c·e1 + y_c·e2, obtained as the meet of
    two of the conic's three §7 dual lines (paper §7, "meet of three lines"):

      l1 & l2  ==  -1/2 · p_c ^ Iod ^ Iinf

      w_c = Delta_2 = AB - C²/4
      x_c = CE/4 - BD/2
      y_c = CD/4 - AE/2

    Finite center = (x_c/w_c, y_c/w_c).  w_c == 0 (parabola) -> p_c is an
    ideal point (the axis direction).  Verified to reproduce conic_center.
    """
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    l1, l2, _ = _conic_lines(A, B, C, D, E, F)
    m12 = l1 & l2
    w = _extract_ratio(m12, _PC_TARGET_W, tol)
    x = _extract_ratio(m12, _PC_TARGET_X, tol)
    y = _extract_ratio(m12, _PC_TARGET_Y, tol)
    return w, x, y


def conic_discriminant2(mv, tol=_TOL):
    """Delta_2 = AB - C²/4  (paper §7), via l1 & l2.  Equals -conic_discriminant/4.

    l1 & l2 == -1/2 * p_c ^ Iod ^ Iinf, so its w-component is -Delta_2/2;
    flip sign and scale to recover Delta_2 itself."""
    w, _, _ = conic_center_meet(mv, tol)
    return -2.0 * w


def conic_discriminant3(mv, tol=_TOL):
    """
    Delta_3 = det(M3) = ABF + (CDE - C²F - BD² - AE²)/4 (paper §7), via the
    meet of all three dual lines:

      l1 & l2 & l3  ==  -Delta_3/2 · Iod ^ Iinf

    Delta_3 == 0  <=>  the conic is degenerate (line pair / point).
    """
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    l1, l2, l3 = _conic_lines(A, B, C, D, E, F)
    m123 = l1 & l2 & l3
    return -2.0 * _extract_ratio(m123, _D3_TARGET, tol)


def conic_is_degenerate(mv, tol=1e-7):
    """True if the conic is degenerate (Delta_3 ≈ 0, paper §7) — a line pair
    (Δ>0 crossing, Δ=0 parallel/double) or a single point.  Distinguishes a
    genuine ellipse/hyperbola/parabola (Delta_3≠0) from its degenerate limits."""
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    scale = max(abs(v) for v in (A, B, C, D, E, F)) or 1.0
    return abs(conic_discriminant3(mv, tol)) < tol * scale**3


def conic_center_point(mv, tol=_TOL):
    """The center of a conic as a CCGA grade-1 point — the **pole of the line at
    infinity**, the projectively-unified notion of center.

    Built from the centre flat point p_c = w_c·eo + x_c·e1 + y_c·e2
    (conic_center_meet, paper §7, meet of three dual lines).

    For an ellipse/hyperbola w_c = Delta_2 ≠ 0 → a finite CCGA point.
    For a **parabola** w_c = Delta_2 = 0 → p_c is an **ideal point** (at
    infinity): the parabola is tangent to the line at infinity, so the pole is
    that point of tangency, lying in the axis direction.  Returned as
    point_at_infinity(axis); it equals the parabola's single (double) asymptotic
    direction and lies on the parabola itself.

    Raises for a degenerate conic (Delta_3 ≈ 0)."""
    from .point import point as _pt, point_at_infinity as _pinf
    if conic_is_degenerate(mv, tol):
        raise ValueError("degenerate conic: center (pole of L∞) undefined")
    w, x, y = conic_center_meet(mv, tol)
    if abs(w) > tol:
        return _pt(x / w, y / w)                 # finite center
    n = (x*x + y*y) ** 0.5                        # parabola: ideal point (axis dir)
    return _pinf(x / n, y / n)


def _central_conic_geometry(mv, tol=_TOL):
    """Shared geometry of a central conic (ellipse / hyperbola).

    Returns a dict:
      type, center (cx,cy), a (major/transverse semi-axis), b (minor/conjugate),
      a_dir, b_dir (unit eigenvectors), c (focal distance), eccentricity.

    Raises ValueError for non-central conics (parabola / degenerate, Δ₂ ≈ 0).
    """
    import numpy as np
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    sub = conic_subtype(A, B, C, D, E, F, tol)
    if sub not in ('ellipse', 'circle', 'hyperbola'):
        raise ValueError(f"not a central conic (type '{sub}')")
    cx, cy = conic_center(mv, tol)
    # constant term after translating the conic to its center
    Fp = A*cx*cx + B*cy*cy + C*cx*cy + D*cx + E*cy + F
    w, V = np.linalg.eigh(np.array([[A, C/2], [C/2, B]]))
    s = [-Fp/w[0], -Fp/w[1]]                       # signed semi-axis² per eigvec
    i, j = (0, 1) if s[0] >= s[1] else (1, 0)      # i = major/transverse (larger s)
    a = float(np.sqrt(abs(s[i])))
    b = float(np.sqrt(abs(s[j])))
    a_dir = tuple(float(x) for x in V[:, i])
    b_dir = tuple(float(x) for x in V[:, j])
    c = float(np.sqrt(a*a - s[j])) if s[j] < 0 else float(np.sqrt(max(a*a - b*b, 0.0)))
    return {'type': sub, 'center': (cx, cy), 'a': a, 'b': b,
            'a_dir': a_dir, 'b_dir': b_dir, 'c': c,
            'eccentricity': c / a if a > tol else float('inf')}


def conic_axes(mv, tol=_TOL):
    """Semi-axes of a central conic as ((a, a_dir), (b, b_dir)).

    a = major (ellipse) / transverse (hyperbola) semi-axis with unit direction
    a_dir; b = minor / conjugate semi-axis with direction b_dir ⟂ a_dir."""
    g = _central_conic_geometry(mv, tol)
    return (g['a'], g['a_dir']), (g['b'], g['b_dir'])


def parabola_geometry(mv, tol=_TOL):
    """Geometry of a parabola (the non-central conic, Δ₂ ≈ 0).

    Returns a dict:
      axis        — unit axis direction (toward the opening / the focus),
      vertex      — (vx, vy),
      focal_length— p (vertex-to-focus distance; |4p| = latus rectum factor),
      focus       — (fx, fy) = vertex + p·axis,
      directrix   — (point, normal): the line {X : normal·(X − point) = 0},
                    normal = axis, point = vertex − p·axis.

    Raises ValueError if mv is not a parabola.
    """
    import numpy as np
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    if conic_subtype(A, B, C, D, E, F, tol) != 'parabola':
        raise ValueError("not a parabola")
    w, V = np.linalg.eigh(np.array([[A, C/2], [C/2, B]]))
    i0 = int(np.argmin(np.abs(w)))             # ~0 eigenvalue → axis direction
    lam = w[1 - i0]
    u = V[:, i0]                               # axis direction
    n = V[:, 1 - i0]                           # perpendicular
    Dp = D*n[0] + E*n[1]
    Ep = D*u[0] + E*u[1]                       # ≠ 0 for a genuine parabola
    xi_v = -Dp / (2*lam)
    eta_v = -(lam*xi_v*xi_v + Dp*xi_v + F) / Ep
    vertex = xi_v*n + eta_v*u
    p = -Ep / (4*lam)                          # signed focal length along +u
    if p < 0:                                  # orient axis toward the focus
        u, p = -u, -p
    focus = vertex + p*u
    return {'axis': tuple(float(x) for x in u),
            'vertex': (float(vertex[0]), float(vertex[1])),
            'focal_length': float(p),
            'focus': (float(focus[0]), float(focus[1])),
            'directrix': ((float((vertex - p*u)[0]), float((vertex - p*u)[1])),
                          (float(u[0]), float(u[1])))}


def conic_eccentricity(mv, tol=_TOL):
    """Eccentricity: 0 circle, 0<e<1 ellipse, 1 parabola, e>1 hyperbola."""
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    if conic_subtype(A, B, C, D, E, F, tol) == 'parabola':
        return 1.0
    return _central_conic_geometry(mv, tol)['eccentricity']


def conic_foci(mv, tol=_TOL):
    """Foci (x, y) of a conic: the two foci of a central conic
    (center ± c·a_dir), or the single focus [(fx, fy)] of a parabola."""
    A, B, C, D, E, F = ipns_to_coeffs(_conic_vector(mv, tol))
    if conic_subtype(A, B, C, D, E, F, tol) == 'parabola':
        return [parabola_geometry(mv, tol)['focus']]
    g = _central_conic_geometry(mv, tol)
    cx, cy = g['center']
    dx, dy = g['a_dir']
    c = g['c']
    return [(cx + c*dx, cy + c*dy), (cx - c*dx, cy - c*dy)]


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
        # A grade-5 blade living in V6 is always SOME conic — the bare 5-point
        # pentapole (the construction ladder's grade-5 rung).  Read its subtype
        # via the grade-7 promotion s = dual(mv ∧ Iod).  Genuine circles/lines
        # keep their cga_* provenance labels below (a cocircular pentapole is
        # literally the same blade as cga.circle); only general ellipse/
        # hyperbola/parabola pentapoles are reported by their conic subtype.
        # (conic_at_infinity = Iod ∧ Iinf gives mv ∧ Iod = 0, so it is skipped.)
        if g == 5:
            s = dual(mv ^ Iod)
            if grades(s, tol) == [1]:
                A, B, C, D, E, F = ipns_to_coeffs(s)
                sub = conic_subtype(A, B, C, D, E, F, tol)
                if sub in ('ellipse', 'hyperbola', 'parabola'):
                    r = _conic_reality(sub, A, B, C, D, E, F, tol)
                    return {'grade': [5], 'type': sub, 'reality': r,
                            'coeffs': (A, B, C, D, E, F)}
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
