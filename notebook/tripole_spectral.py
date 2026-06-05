"""
Working spectral reconstruction of a tripole — fixing §7 of
trivector_reconstruction_ga.md.

The degenerate operator (V⌋T)⌋T = T²·P_Π only finds the PLANE Π.  To separate
the three points we need an operator with DISTINCT eigenvalues = the coordinates.
We build the "multiplication-by-x" endomorphism on Π as a 3×3 matrix pencil.

Moment functionals (all GA inner products of a vector V with fixed blades):

    m_o (V) = −(V·e∞)      # homogeneous weight  w
    m_x (V) =  (V·e1)      # x·w
    m_y (V) =  (V·e2)      # y·w
    m_xx(V) = −(V·eo1)     # (x²/2)·w
    m_yy(V) = −(V·eo2)     # (y²/2)·w
    m_xy(V) = −(V·eo3)     # (xy)·w

For a CCGA point p (weight w):  (m_o,m_x,m_y,m_xx,m_yy,m_xy) = w(1,x,y,x²/2,y²/2,xy).
Hence the two linear maps  Π → R³

    L0(V) = ( m_o , m_x , m_y )           L0(p) = w(1, x, y)
    Lx(V) = ( m_x , 2·m_xx , m_xy )       Lx(p) = w(x, x², xy) = x·L0(p)

satisfy  Lx(p_i) = x_i · L0(p_i).  In any basis B={b1,b2,b3} of Π they are 3×3
matrices A0, Ax, and the generalised eigenproblem

    Ax c = x · A0 c        ( det(Ax − x A0) = (x−x1)(x−x2)(x−x3) = the CUBIC )

has eigenvalues x_i and eigenvectors c_i = coords of p_i in B.  (A y-pencil with
Ly=(m_y, m_xy, 2·m_yy) gives the y_i; here we just read y back off p_i = B c_i.)
"""
import numpy as np

from ccga.algebra import alg, e1, e2, einf, eo1, eo2, eo3
from ccga.point import point
from ccga.operations import is_zero

P = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p = [point(*c) for c in P]
T = p[0] ^ p[1] ^ p[2]

# ── 1. a basis of the plane Π = span(p1,p2,p3), purely from T ─────────────────
# projection onto Π:  P_Π(V) = (V⌋T)⌋T / T²   (the degenerate operator / T²).
basis8 = [alg.multivector({1 << i: 1.0}) for i in range(8)]


def vec8(mv):
    out = np.zeros(8)
    for k, v in mv.items():
        if k and (k & (k - 1)) == 0:
            out[k.bit_length() - 1] = float(v)
    return out


T2 = float((T * T).e)
Pproj = np.column_stack([vec8(((b | T) | T)) for b in basis8]) / T2
U, S, _ = np.linalg.svd(Pproj)
B = U[:, :3]                                  # 8×3, columns = a basis of Π
bvec = [alg.multivector({1 << i: B[i, j] for i in range(8) if abs(B[i, j]) > 1e-12})
        for j in range(3)]
print("dim Π =", int(np.sum(S > 1e-9)), " (basis from T, no points used)")


# ── 2. moment functionals → the two 3×3 matrices A0, Ax ───────────────────────
def moments(V):
    return dict(
        o=-float((V | einf).e), x=float((V | e1).e), y=float((V | e2).e),
        xx=-float((V | eo1).e), yy=-float((V | eo2).e), xy=-float((V | eo3).e),
    )


cols0, colsx = [], []
for b in bvec:
    m = moments(b)
    cols0.append([m['o'], m['x'], m['y']])
    colsx.append([m['x'], 2 * m['xx'], m['xy']])
A0 = np.array(cols0).T
Ax = np.array(colsx).T

# ── 3. generalised eigenproblem  Ax c = x A0 c  (eigenvalues = x_i) ───────────
xvals, C = np.linalg.eig(np.linalg.solve(A0, Ax))
print("\ncharacteristic polynomial det(Ax − x A0) roots = x-coordinates:")
print("  eigenvalues x_i =", np.round(np.sort(xvals.real), 6))

# ── 4. reconstruct each point  p_i = B c_i,  read (x,y), verify ───────────────
print("\nreconstructed points (p_i = B·c_i, normalised by −(p·e∞)):")
rec = []
for lam, c in zip(xvals, C.T):
    pv = B @ c.real
    mv = alg.multivector({1 << i: pv[i] for i in range(8) if abs(pv[i]) > 1e-12})
    w = -float((mv | einf).e)
    mv = mv * (1.0 / w)
    x, y = float((mv | e1).e), float((mv | e2).e)
    veronese_ok = abs(float((mv | eo1).e) + x * x / 2) < 1e-6      # einf1 == x²/2 ?
    rec.append((round(x, 4), round(y, 4)))
    print(f"  x(eig)={lam.real:+.4f} -> (x,y)=({x:+.4f},{y:+.4f})  "
          f"q∧T==0 {is_zero(mv ^ T, 1e-7)}  Veronese {veronese_ok}")

print("\nrecovered:", sorted(rec))
print("original :", sorted((round(x, 4), round(y, 4)) for x, y in P))
assert sorted(rec) == sorted((round(x, 4), round(y, 4)) for x, y in P), "mismatch!"
print("\nOK — spectral reconstruction recovers the three points "
      "(eigenvalues = x_i; the 3×3 char. poly IS the cubic).")
