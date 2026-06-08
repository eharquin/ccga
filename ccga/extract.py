"""
CCGA n-pole point extraction — the GA-native analogues of the dipole ±√ formula.

Dipole (grade 2):    p_{1,2} = (pp ± √(pp²)) / (e∞·pp)              — square root
Tripole (grade 3):   circumcircle = T ∧ Iod ∧ Iinfd (closed GA form),
                     membership q(t)∧T=0 → a cubic in the circle parameter,
                     solved by Cardano                              — cube root
Quadpole (grade 4):  pencil through the 4 points = {(Iod∧Q∧p₅)·I⁻¹}
                     (GA-native, no SVD), resolvent cubic (Cardano) picks a
                     pairing, each line carries a dipole → two ±√  — Ferrari

There is no single-radical closed form for 3+ points (irreducible cubic/quartic);
these reduce the n-pole to dipoles via objects read off the blade by GA, then
solve the small residual by radicals.  See notebook/tripole_extraction.ipynb and
notebook/quadpole_extraction.ipynb.
"""
import numpy as np

from .algebra import e1, e2, eo, einf, einf3, einfbar, Iod, Iinfd, I_inv
from .point import point
from .operations import grades, meet
from .classify import ipns_to_coeffs, _conic_vector

_TOL = 1e-9


# ── shared helpers ────────────────────────────────────────────────────────────

def _cardano(a, b, c):
    """Real roots of the monic cubic t³ + a t² + b t + c (Cardano, trig form
    for the three-real-root case / casus irreducibilis)."""
    p = b - a*a/3.0
    q = 2*a**3/27.0 - a*b/3.0 + c
    disc = (q/2)**2 + (p/3)**3
    if disc < 0:                                   # three distinct real roots
        r = np.sqrt(-(p**3)/27.0)
        phi = np.arccos(np.clip(-q/(2*r), -1.0, 1.0))
        m = 2*np.cbrt(r)
        return [m*np.cos((phi + 2*np.pi*k)/3.0) - a/3.0 for k in range(3)]
    u = np.cbrt(-q/2.0 + np.sqrt(disc))            # one real root
    v = np.cbrt(-q/2.0 - np.sqrt(disc))
    return [u + v - a/3.0]


def _coords(mv):
    """Euclidean (x, y) of a grade-1 point multivector."""
    return float((mv | e1).e), float((mv | e2).e)


# ── tripole ───────────────────────────────────────────────────────────────────

def tripole_circumconic(T):
    """The circum-conic of a tripole as the grade-7 OPNS blade  T ∧ Iod ∧ Iinfd.

    For three finite/round points this is their **circle** (the round points,
    radius included, are IPNS-incident to it); if one point is **ideal** (at
    infinity) it degenerates to the **line** through the two finite points (a
    circle through ∞ is a line); two ideal points make it vanish.  Use
    classify()/conic_type() on the result, or circumcircle() for (cx,cy,R)."""
    return T ^ Iod ^ Iinfd


def circumcircle(T, tol=_TOL):
    """Centre and radius of the circle through the 3 points of tripole T, from
    the closed-form GA blade  T ∧ Iod ∧ Iinfd  (grade-7 OPNS conic).

    For **round** points the radius enters: r²_circum = (centre circle)² − r²
    (real radius shrinks it, imaginary grows it; the round points are incident
    to the returned circle).  Raises if the circum-conic is a line (one point at
    infinity) or degenerate (two) — use tripole_circumconic for those."""
    A, B, C, D, E, F = ipns_to_coeffs(tripole_circumconic(T) * I_inv)
    if abs(A) < tol or abs(B) < tol:
        raise ValueError("circum-conic is a line/degenerate (ideal point in the "
                         "tripole); use tripole_circumconic")
    cx, cy = -D/(2*A), -E/(2*B)
    R2 = cx*cx + cy*cy - F/A
    return cx, cy, (R2 ** 0.5 if R2 >= 0 else (abs(R2) ** 0.5) * 1j)


def _tripole_cubic(T, q, n=60):
    """The genuine cubic in the circle parameter t cut out on the circumcircle by
    the membership q(t)∧T = 0.  Each blade coefficient of q(t)∧T, cleared of its
    (1+t²)² denominator, is a quartic (circle ∩ conic, Bézout 4); they all share
    the 3 point-roots, so the quartic coefficient vectors span a 2-D space whose
    degree-reduced combination is the cubic g(t).  `q` maps t → point on circle."""
    ts = np.linspace(-7, 7, n)
    keys = sorted({k for tt in ts for k in (q(tt) ^ T).keys()})
    rows = []
    for k in keys:
        col = np.array([float(dict((q(tt) ^ T).items()).get(k, 0.0)) * (1 + tt*tt)**2
                        for tt in ts])
        if np.max(np.abs(col)) > _TOL:
            rows.append(np.polyfit(ts, col, 4))            # quartic, lead first
    M = np.array(rows)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    u, v = Vt[0], Vt[1]                                    # 2-D span of g·{1,t}
    w = v[0]*u - u[0]*v                                    # degree ≤3, ∝ g(t)
    w = w / np.max(np.abs(w))
    g = w[1:] if abs(w[0]) < 1e-6 else w                   # drop ~0 t⁴ lead
    return g / g[0]                                        # monic cubic [1,a,b,c]


def _on_tripole(q_pt, T, tol=1e-6):
    """Residual of the membership q ∧ T (0 ⇔ q is one of the three points)."""
    w = q_pt ^ T
    return max((abs(float(v)) for v in w.values()), default=0.0) < tol


def extract_tripole(T):
    """Recover the three points (x, y) of a CCGA tripole T = p1∧p2∧p3.

    GA circumcircle → membership cubic → Cardano (closed form by radicals).
    The circle is parametrised rationally (t = tan(θ/2)); a generic rotation φ of
    the parametrisation keeps any point off the t=∞ pole — we try several φ and
    validate each recovered point against q∧T = 0."""
    if grades(T) != [3]:
        raise ValueError(f"tripole must be grade 3, got grades {grades(T)}")
    cx, cy, R = circumcircle(T)
    for phi in (0.0, 0.7, 1.3, 2.1, 0.31):
        cphi, sphi = np.cos(phi), np.sin(phi)

        def q(t, cphi=cphi, sphi=sphi):
            c = (1 - t*t)/(1 + t*t); s = 2*t/(1 + t*t)
            return point(cx + R*(cphi*c - sphi*s), cy + R*(sphi*c + cphi*s))

        g = _tripole_cubic(T, q)
        pts = [(q(t), _coords(q(t))) for t in _cardano(g[1], g[2], g[3])]
        good = [xy for qp, xy in pts if _on_tripole(qp, T)]
        # dedupe (a double root from a bad φ) and require all three
        uniq = []
        for xy in good:
            if all(abs(xy[0]-u[0]) + abs(xy[1]-u[1]) > 1e-4 for u in uniq):
                uniq.append(xy)
        if len(uniq) == 3:
            return uniq
    raise ValueError("tripole extraction failed for all parametrisation rotations")


# ── quadpole ──────────────────────────────────────────────────────────────────

def pencil(Q):
    """Two independent generators (A,B,C,D,E,F) of the pencil of conics through
    the four points of quadpole Q.  Each conic comes from the GA blade
    (Iod ∧ Q ∧ p₅)·I⁻¹; the pencil is exactly 2-D, but a single p₅ can give a
    degenerate/zero generator for symmetric configurations, so we gather several
    candidate p₅ and return an orthonormal basis of their (rank-2) span."""
    candidates = [einf, eo, e1, e2, point(0.31, -0.72), point(1.0, 1.0)]
    rows = [np.array(ipns_to_coeffs((Iod ^ Q ^ p5) * I_inv), float)
            for p5 in candidates]
    M = np.array(rows)
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    if S[1] < 1e-9 * S[0]:
        raise ValueError("quadpole pencil is not 2-dimensional (degenerate Q?)")
    return Vt[0], Vt[1]


def _conic_mat(c):
    A, B, C, D, E, F = c
    return np.array([[A, C/2, D/2], [C/2, B, E/2], [D/2, E/2, F]])


def _resolvent_roots(g1, g2):
    """Roots t of det(M1 + t·M2) = 0 — the (up to 3) degenerate pencil members
    (pairings).  The 3×3 determinant is a cubic in t whose coefficients are got
    *exactly* by the column-replacement expansion (well conditioned, and roots at
    infinity simply drop out when the leading coefficient vanishes)."""
    M1, M2 = _conic_mat(g1), _conic_mat(g2)
    a, b = M1.T, M2.T                                      # columns as rows

    def d(c0, c1, c2):
        return np.linalg.det(np.array([c0, c1, c2]).T)

    c0 = d(a[0], a[1], a[2])                               # det(M1)
    c1 = d(b[0], a[1], a[2]) + d(a[0], b[1], a[2]) + d(a[0], a[1], b[2])
    c2 = d(a[0], b[1], b[2]) + d(b[0], a[1], b[2]) + d(b[0], b[1], a[2])
    c3 = d(b[0], b[1], b[2])                               # det(M2)
    coef = np.array([c3, c2, c1, c0])
    scale = np.max(np.abs(coef))
    nz = np.argmax(np.abs(coef) > 1e-12*scale)            # trim leading ~0 (roots at ∞)
    return [r.real for r in np.roots(coef[nz:]) if abs(r.imag) < 1e-6]


def _lines_of(coef):
    """Split a degenerate conic (rank-2 matrix) into its two real lines [a,b,c]."""
    M = _conic_mat(coef)
    w, V = np.linalg.eigh(M)
    i = np.argsort(w)
    u = np.sqrt(max(w[i[2]], 0.0)) * V[:, i[2]]
    v = np.sqrt(max(-w[i[0]], 0.0)) * V[:, i[0]]
    return (u + v), (u - v)


def extract_quadpole(Q):
    """Recover the four points (x, y) of a CCGA quadpole Q = p1∧p2∧p3∧p4.

    GA pencil → resolvent cubic (Cardano) picks a pairing → each line carries a
    dipole, split by the ±√ quadratic (Ferrari)."""
    if grades(Q) != [4]:
        raise ValueError(f"quadpole must be grade 4, got grades {grades(Q)}")
    g1, g2 = pencil(Q)
    keys = sorted((point(0.123, 0.234) ^ Q).keys())

    def pQ(x, y):
        d = dict((point(x, y) ^ Q).items())
        return np.array([float(d.get(k, 0.0)) for k in keys])

    def dipole_on_line(L):
        a, b, c = L
        n2 = a*a + b*b
        base = np.array([-a*c, -b*c]) / n2
        dirv = np.array([-b, a]) / np.sqrt(n2)
        L0, L1, L2 = pQ(*(base - dirv)), pQ(*base), pQ(*(base + dirv))
        A = 0.5*(L0 + L2) - L1
        B = 0.5*(L2 - L0)
        j = int(np.argmax(np.abs(A)))
        disc = B[j]**2 - 4*A[j]*L1[j]
        roots = ((-B[j] + disc**0.5)/(2*A[j]), (-B[j] - disc**0.5)/(2*A[j]))
        return [tuple(base + r*dirv) for r in roots]

    # pick a resolvent root whose degenerate conic splits into two real lines
    for t in _resolvent_roots(g1, g2):
        ev = np.linalg.eigvalsh(_conic_mat(g1 + t*g2))
        if ev[0] < -_TOL and ev[2] > _TOL:                 # signature (-,0,+)
            La, Lb = _lines_of(g1 + t*g2)
            return dipole_on_line(La) + dipole_on_line(Lb)
    raise ValueError("no real line-pair found in the pencil (complex points?)")


# ── conic ∨ conic intersection (grade-6 object) ───────────────────────────────
#
# Two conics meet in 4 points (Bézout).  The regressive product C1 ∨ C2 is the
# grade-6 blade
#
#     I4 = C1 ∨ C2  ∝  p1 ∧ p2 ∧ p3 ∧ p4 ∧ Iod  =  Q ∧ Iod,
#
# i.e. the quadpole Q of the 4 intersection points, gauge-fixed by Iod (the same
# Iod that turns 5 points into a grade-7 conic).  A point q is an intersection
# point iff  q ∧ I4 = 0.  The 4 points may be real, an imaginary conjugate pair,
# or ideal (at infinity) — e.g. two circles always share the two imaginary
# circular points at infinity, so they meet in 2 finite + 2 ideal points.

def conic_intersection(C1, C2):
    """Grade-6 intersection object  I4 = C1 ∨ C2  of two conics.

    Accepts grade-7 OPNS conics (the regressive product is taken directly).
    Equal (up to scale) to  p1 ∧ p2 ∧ p3 ∧ p4 ∧ Iod  for the 4 Bézout points;
    q ∧ I4 = 0 tests incidence.  Recover the quadpole with
    intersection_quadpole, the points with intersection_points.
    """
    return meet(C1, C2)


def intersection_quadpole(I4):
    """Recover the grade-4 quadpole Q of the 4 intersection points from the
    grade-6 intersection object:  Q = (einf3 ∧ einfbar) | (C1 ∨ C2).

    Inverts I4 = Q ∧ Iod (Iod = eobar ∧ eo3); einf3 ∧ einfbar is the reciprocal
    2-blade of Iod in the W2 = span{eobar, eo3} gauge plane."""
    return (einf3 ^ einfbar) | I4


def _coeffs_of(C):
    """(A,B,C,D,E,F) of a conic given as grade-1 IPNS, grade-5, or grade-7."""
    return ipns_to_coeffs(_conic_vector(C))


def _poly_det(M):
    """Determinant of a matrix whose entries are 1-D coefficient arrays (a
    polynomial each, ascending powers).  Returns the coefficient array of the
    determinant.  Exact polynomial arithmetic via numpy — no float-domain
    fragility (unlike sympy.resultant over RR)."""
    n = len(M)
    if n == 1:
        return np.asarray(M[0][0], dtype=float)
    total = np.zeros(1)
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        term = np.convolve(M[0][j], _poly_det(minor))    # poly multiply
        if j % 2:
            term = -term
        if len(term) > len(total):
            total = np.pad(total, (0, len(term) - len(total)))
        total[:len(term)] += term
    return total


def _intersection_roots(C1, C2):
    """The (up to 4) complex affine intersection points of two conics, as
    complex (x, y) pairs.  The resultant in y of the two conics (a quartic in x)
    is the Sylvester determinant, built with numpy polynomial arithmetic; its
    roots are the x-coordinates (missing roots ⇒ points at infinity)."""
    a1, b1, c1, d1, e1_, f1 = _coeffs_of(C1)
    a2, b2, c2, d2, e2_, f2 = _coeffs_of(C2)
    # Sylvester matrix of Q1, Q2 as quadratics in y, entries = polys in x
    # (ascending coeffs):  b_i constant, B_i = e_i + c_i·x, C_i = f_i + d_i·x + a_i·x²
    B1, C1c = np.array([e1_, c1]), np.array([f1, d1, a1])
    B2, C2c = np.array([e2_, c2]), np.array([f2, d2, a2])
    z, b1a, b2a = np.array([0.0]), np.array([b1]), np.array([b2])
    syl = [[b1a, B1,  C1c, z],
           [z,   b1a, B1,  C1c],
           [b2a, B2,  C2c, z],
           [z,   b2a, B2,  C2c]]
    res = _poly_det(syl)
    scale = np.max(np.abs(res)) if res.size else 0.0
    if scale < _TOL:
        raise ValueError("conics share a component (infinite intersection)")
    # drop near-zero top coefficients (roots that have gone to infinity)
    keep = np.where(np.abs(res) > 1e-9 * scale)[0]
    res = res[:keep[-1] + 1]
    xr = np.roots(res[::-1]) if res.size > 1 else []      # np.roots wants descending
    pts = []
    for xv in xr:
        if abs(b1) > _TOL:
            yroots = np.roots([b1, c1*xv + e1_, a1*xv*xv + d1*xv + f1])
        elif abs(c1*xv + e1_) > _TOL:
            yroots = [-(a1*xv*xv + d1*xv + f1) / (c1*xv + e1_)]
        else:
            yroots = []
        for yv in yroots:
            if abs(b2*yv*yv + (c2*xv + e2_)*yv + (a2*xv*xv + d2*xv + f2)) < 1e-5:
                if all(abs(xv - p[0]) + abs(yv - p[1]) > 1e-5 for p in pts):
                    pts.append((complex(xv), complex(yv)))
    return pts[:4]


def intersection_points(C1, C2, tol=1e-6):
    """Real finite intersection points (x, y) of two conics (0–4 of them).

    Ideal (at-infinity) and imaginary intersections are omitted; use
    intersection_reality for the full Bézout decomposition."""
    return sorted((p[0].real, p[1].real) for p in _intersection_roots(C1, C2)
                  if abs(p[0].imag) < tol and abs(p[1].imag) < tol)


def intersection_reality(C1, C2, tol=1e-6):
    """Bézout decomposition of the 4 intersection points of two conics.

    Returns {'real': r, 'imaginary': m, 'ideal': k} with r+m+k = 4:
      real      — finite real points,
      imaginary — finite complex-conjugate points,
      ideal     — points at infinity (e.g. the two circular points two circles
                  always share; or a shared real asymptotic direction).
    """
    pts = _intersection_roots(C1, C2)
    real = sum(1 for p in pts if abs(p[0].imag) < tol and abs(p[1].imag) < tol)
    return {'real': real, 'imaginary': len(pts) - real, 'ideal': 4 - len(pts)}


# ── normals & orthogonal projection of a point onto a conic ───────────────────

def normal_feet(conic, q):
    """Feet (x, y) of all real normals dropped from a point q onto a conic.

    The feet satisfy (q − p) ∥ ∇F(p), which is the Apollonius conic; so they are
    the real points of  conic ∩ apollonius_conic(conic, q)  (up to 4 — an
    interior point near the center has 4 normals, the evolute region)."""
    from .objects import apollonius_conic
    return intersection_points(conic, apollonius_conic(conic, q))


def project_point_to_conic(conic, q):
    """Orthogonal projection of q onto a conic: the nearest foot of a normal.

    Returns (x, y), the closest point of the conic to q.  Raises if no real foot
    is found (should not happen for a non-empty real conic)."""
    qx = float((q | e1).e) / -float((q | einf).e)
    qy = float((q | e2).e) / -float((q | einf).e)
    feet = normal_feet(conic, q)
    if not feet:
        raise ValueError("no real foot found (empty/complex conic?)")
    return min(feet, key=lambda p: (p[0] - qx)**2 + (p[1] - qy)**2)
