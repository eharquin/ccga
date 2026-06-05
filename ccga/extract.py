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

from .algebra import e1, e2, eo, einf, Iod, Iinfd, I_inv
from .point import point
from .operations import grades
from .classify import ipns_to_coeffs

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

def circumcircle(T):
    """Centre and radius of the circle through the 3 points of tripole T,
    from the closed-form GA blade  T ∧ Iod ∧ Iinfd  (grade-7 OPNS conic)."""
    A, B, C, D, E, F = ipns_to_coeffs((T ^ Iod ^ Iinfd) * I_inv)
    cx, cy = -D/(2*A), -E/(2*B)
    R = (cx*cx + cy*cy - F/A) ** 0.5
    return cx, cy, R


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
