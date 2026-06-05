"""
Test the spectral-reconstruction claim of trivector_reconstruction_ga.md §7:

    A(V) = (V ⌋ T) ⌋ T          (a linear endomorphism on grade-1 vectors)
    claim:  A(P_i) = λ_i P_i      (the three points are eigenvectors)

If true, the three points fall out of an 8×8 (really 3×3 on the plane Π)
eigendecomposition — no circle parametrisation.  We build the matrix of A in
the diagonal basis, diagonalise, and check the eigenvectors against p1,p2,p3.
"""
import numpy as np

from ccga.algebra import alg, e1, e2, einf, eo
from ccga.point import point
from ccga.operations import is_zero, grades

P = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p = [point(*c) for c in P]
T = p[0] ^ p[1] ^ p[2]

# diagonal basis vectors e1..e8 (keys 1,2,4,...,128)
basis = [alg.multivector({1 << i: 1.0}) for i in range(8)]


def vec8(mv):
    """grade-1 part of mv as an 8-vector in the diagonal basis."""
    out = np.zeros(8)
    for k, v in mv.items():
        if k and (k & (k - 1)) == 0:          # power of two => grade 1
            out[k.bit_length() - 1] = float(v)
    return out


def A(V):
    return (V | T) | T                        # (V ⌋ T) ⌋ T


# build the 8×8 matrix of A
Mat = np.column_stack([vec8(A(b)) for b in basis])
print("operator A(V) = (V⌋T)⌋T  — matrix rank:", np.linalg.matrix_rank(Mat, tol=1e-9))

# sanity: A maps into the plane Π = span(p1,p2,p3)?
pvecs = np.column_stack([vec8(pi) for pi in p])
print("rank[ p1 p2 p3 ] =", np.linalg.matrix_rank(pvecs, tol=1e-9),
      "; rank[ Π | A(e_k) ] =",
      np.linalg.matrix_rank(np.column_stack([pvecs, Mat]), tol=1e-9),
      " (== 3 ⇒ image(A) ⊆ Π)")

# eigendecomposition
w, Vmat = np.linalg.eig(Mat)
print("\neigenvalues:", np.round(w, 4))


def coords_from_vec(v8, tol=1e-9):
    mv = alg.multivector({1 << i: v8[i] for i in range(8) if abs(v8[i]) > 1e-12})
    scale = -float((mv | einf).e)
    if abs(scale) < tol:
        return None
    mv = mv * (1.0 / scale)
    return float((mv | e1).e), float((mv | e2).e), mv


print("\neigenvectors that are CCGA points (q∧T==0):")
found = []
for lam, col in zip(w, Vmat.T):
    if abs(lam.imag) > 1e-7:
        continue
    r = coords_from_vec(col.real)
    if r is None:
        continue
    x, y, mv = r
    on = is_zero(mv ^ T, 1e-7)
    tag = "  <-- point" if on else ""
    if on:
        found.append((round(x, 4), round(y, 4)))
    print(f"  λ={lam.real:+.4f}: (x,y)=({x:+.4f},{y:+.4f})  q∧T==0? {on}{tag}")

print("\nrecovered points:", sorted(set(found)))
print("original  points:", sorted((round(x, 4), round(y, 4)) for x, y in P))

# also report A(p_i): is it proportional to p_i?
print("\ndirect check  A(p_i) ?= λ_i p_i :")
for i, pi in enumerate(p):
    api = vec8(A(pi)); pv = vec8(pi)
    j = int(np.argmax(np.abs(pv)))
    lam = api[j] / pv[j]
    print(f"  p{i+1}:  A(p)=λp ? {np.allclose(api, lam*pv, atol=1e-7)}   (λ={lam:.4f})")
