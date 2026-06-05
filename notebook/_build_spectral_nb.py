"""Generate notebook/tripole_spectral.ipynb (spectral n-pole reconstruction)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "CCGA (.venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

C = []  # cells

C.append(new_markdown_cell(r"""# Spectral reconstruction of a tripole

Recover the three points $p_1,p_2,p_3$ from the trivector
$\mathbf T = p_1\wedge p_2\wedge p_3$ by an **eigenvalue problem**, with no
rational parametrisation of the circumcircle and no polynomial GCD.

This is the corrected version of §7 of `trivector_reconstruction_ga.md`.
The operator proposed there, $\mathcal A(V)=(V\lrcorner\mathbf T)\lrcorner\mathbf T$,
is **degenerate**: it equals $\mathbf T^2\,P_\Pi$ (a scalar multiple of the
projection onto the plane $\Pi=\operatorname{span}(p_1,p_2,p_3)$), so all three
points share the single eigenvalue $\mathbf T^2$ and the eigenvectors do *not*
isolate the points.

To separate them we build the **multiplication-by-$x$** endomorphism on $\Pi$,
whose eigenvalues are the distinct coordinates $x_1,x_2,x_3$. Its $3\times3$
characteristic polynomial is precisely the irreducible cubic (the Galois floor),
now solved by a standard eigen-routine."""))

C.append(new_code_cell(r"""import numpy as np
from ccga.algebra import alg, e1, e2, einf, eo1, eo2, eo3
from ccga.point import point
from ccga.operations import is_zero, grades

np.set_printoptions(suppress=True, precision=6)

# a generic triangle
P = [(0.3, 1.7), (2.1, -0.4), (-1.2, 0.9)]
p = [point(*c) for c in P]
T = p[0] ^ p[1] ^ p[2]
print("grades(T) =", grades(T), "   T² =", round(float((T*T).e), 6))"""))

C.append(new_markdown_cell(r"""## 1. Why the naïve operator fails

$\mathcal A(V)=(V\lrcorner\mathbf T)\lrcorner\mathbf T$ as an $8\times8$ matrix.
Projection onto a blade is $P_\Pi(V)=(V\lrcorner\mathbf T)\lrcorner\mathbf T^{-1}$,
so $\mathcal A=\mathbf T^2\,P_\Pi$: spectrum $\{\mathbf T^2(\times3),\,0(\times5)\}$,
with the whole plane $\Pi$ as one eigenspace — no preferred basis, no points."""))

C.append(new_code_cell(r"""basis8 = [alg.multivector({1 << i: 1.0}) for i in range(8)]

def vec8(mv):
    "grade-1 part of mv as an 8-vector in the diagonal basis e1..e8"
    out = np.zeros(8)
    for k, v in mv.items():
        if k and (k & (k - 1)) == 0:        # power of two => grade 1
            out[k.bit_length() - 1] = float(v)
    return out

A_naive = np.column_stack([vec8((b | T) | T) for b in basis8])
w = np.linalg.eigvals(A_naive)
print("eigenvalues of (V⌋T)⌋T :", np.round(np.sort(w.real), 4))
print("=> {T² (×3), 0 (×5)} with T² =", round(float((T*T).e), 4),
      " — degenerate on the plane, cannot separate points")"""))

C.append(new_markdown_cell(r"""## 2. A basis of the plane $\Pi$ — from $\mathbf T$ alone

The one thing the degenerate operator *is* good for: its column space is
$\Pi$. Normalise to the projection $P_\Pi=\mathcal A/\mathbf T^2$ and take an
orthonormal basis $B=\{b_1,b_2,b_3\}$ of its range (SVD). **No point
coordinates are used.**"""))

C.append(new_code_cell(r"""T2 = float((T * T).e)
Pproj = A_naive / T2
U, S, _ = np.linalg.svd(Pproj)
print("singular values:", np.round(S, 6), " -> dim Π =", int((S > 1e-9).sum()))

B = U[:, :3]                                   # 8×3, columns span Π
bvec = [alg.multivector({1 << i: B[i, j] for i in range(8) if abs(B[i, j]) > 1e-12})
        for j in range(3)]"""))

C.append(new_markdown_cell(r"""## 3. Moment functionals (all GA inner products)

For a CCGA point of weight $w$,
$p=w\big(e_o+x e_1+y e_2+\tfrac{x^2}{2}e_{\infty_1}+\tfrac{y^2}{2}e_{\infty_2}+xy\,e_{\infty_3}\big)$,
these contractions read off the Veronese coordinates $w(1,x,y,\tfrac{x^2}{2},\tfrac{y^2}{2},xy)$:

$$
m_o=-(V\!\cdot e_\infty)=w,\quad m_x=(V\!\cdot e_1)=wx,\quad m_y=(V\!\cdot e_2)=wy,
$$
$$
m_{xx}=-(V\!\cdot e_{o_1})=\tfrac{wx^2}{2},\quad
m_{yy}=-(V\!\cdot e_{o_2})=\tfrac{wy^2}{2},\quad
m_{xy}=-(V\!\cdot e_{o_3})=wxy.
$$"""))

C.append(new_code_cell(r"""def moments(V):
    return dict(
        o =-float((V | einf).e),   # w
        x = float((V | e1).e),     # w x
        y = float((V | e2).e),     # w y
        xx=-float((V | eo1).e),    # w x²/2
        yy=-float((V | eo2).e),    # w y²/2
        xy=-float((V | eo3).e),    # w xy
    )

# sanity: on a weight-1 point the moments are exactly (1, x, y, x²/2, y²/2, xy)
print(moments(point(0.3, 1.7)))"""))

C.append(new_markdown_cell(r"""## 4. The multiplication-by-$x$ pencil

Two linear maps $\Pi\to\mathbb R^3$:

$$
L_0(V)=\big(m_o,\,m_x,\,m_y\big),\qquad
L_x(V)=\big(m_x,\,2m_{xx},\,m_{xy}\big),
$$

so on points $L_0(p)=w(1,x,y)$ and $L_x(p)=w(x,x^2,xy)=x\,L_0(p)$. In the basis
$B$ they are $3\times3$ matrices $A_0,A_x$, and

$$
A_x\,c = x\,A_0\,c,\qquad \det(A_x-xA_0)=(x-x_1)(x-x_2)(x-x_3).
$$

The generalised eigenvalues are the $x$-coordinates; the eigenvectors $c_i$ are
the coordinates of $p_i$ in $B$."""))

C.append(new_code_cell(r"""cols0, colsx = [], []
for b in bvec:
    m = moments(b)
    cols0.append([m['o'],  m['x'],     m['y']])
    colsx.append([m['x'],  2*m['xx'],  m['xy']])
A0 = np.array(cols0).T
Ax = np.array(colsx).T

xvals, Cc = np.linalg.eig(np.linalg.solve(A0, Ax))   # Ax c = x A0 c
print("x-coordinates (eigenvalues) =", np.round(np.sort(xvals.real), 6))"""))

C.append(new_markdown_cell(r"""## 5. Reconstruct the points and verify

For each eigenpair $(x_i,c_i)$ form $p_i=B c_i$, normalise by $-(p\cdot e_\infty)$,
and read $(x,y)$. We check both that $p_i\wedge\mathbf T=0$ (lies in $\Pi$) **and**
the Veronese relation $-(p\cdot e_{o_1})=x^2/2$ (a genuine point, not an arbitrary
plane vector)."""))

C.append(new_code_cell(r"""def reconstruct(c):
    pv = B @ c.real
    mv = alg.multivector({1 << i: pv[i] for i in range(8) if abs(pv[i]) > 1e-12})
    mv = mv * (1.0 / (-float((mv | einf).e)))    # normalise weight to 1
    return mv

rec = []
for lam, c in zip(xvals, Cc.T):
    mv = reconstruct(c)
    x, y = float((mv | e1).e), float((mv | e2).e)
    veronese = abs(-float((mv | eo1).e) - x*x/2) < 1e-6
    rec.append((round(x, 4), round(y, 4)))
    print(f"x(eig)={lam.real:+.4f} -> ({x:+.4f}, {y:+.4f})   "
          f"q∧T==0 {is_zero(mv ^ T, 1e-7)}   Veronese {veronese}")

print("\nrecovered:", sorted(rec))
print("original :", sorted((round(x, 4), round(y, 4)) for x, y in P))"""))

C.append(new_markdown_cell(r"""## 6. Package it — `reconstruct_tripole_spectral`

Wrap the pipeline in one function. For robustness against a **vertical edge**
(two equal $x_i$, a defective pencil) we diagonalise multiplication by a generic
direction $u=\cos\theta\,x+\sin\theta\,y$ instead of $x$; its eigenvalues are the
distinct projections $u\!\cdot p_i$, and we read $(x,y)$ back off the
eigenvectors. The shift stays inside the **degree-2** moments:

$$
u\cdot(1,x,y)=\big(u,\;u x,\;u y\big)\ \longrightarrow\
\big(\cos\theta\,m_x+\sin\theta\,m_y,\ \;2\cos\theta\,m_{xx}+\sin\theta\,m_{xy},
\ \;\cos\theta\,m_{xy}+2\sin\theta\,m_{yy}\big).
$$"""))

C.append(new_code_cell(r"""def reconstruct_tripole_spectral(T, theta=0.37):
    "Recover p1,p2,p3 from T = p1^p2^p3 by the moment pencil (eigenvalues = u·p_i)."
    rev = ~T
    Amat = np.column_stack([vec8((b | T) | rev) for b in basis8]) / float((T * rev).e)
    U, S, _ = np.linalg.svd(Amat)
    if int((S > 1e-9 * S[0]).sum()) != 3:
        raise ValueError("plane is not 3-dimensional (degenerate / collinear T?)")
    Bn = U[:, :3]
    bn = [alg.multivector({1 << i: Bn[i, j] for i in range(8) if abs(Bn[i, j]) > 1e-12})
          for j in range(3)]
    cph, sph = np.cos(theta), np.sin(theta)
    M0, Mu = [], []
    for b in bn:
        m = moments(b)
        M0.append([m['o'], m['x'], m['y']])
        Mu.append([cph*m['x'] + sph*m['y'],
                   cph*2*m['xx'] + sph*m['xy'],
                   cph*m['xy'] + sph*2*m['yy']])
    _, Cn = np.linalg.eig(np.linalg.solve(np.array(M0).T, np.array(Mu).T))
    pts = []
    for c in Cn.T:
        pv = Bn @ c.real
        mv = alg.multivector({1 << i: pv[i] for i in range(8) if abs(pv[i]) > 1e-12})
        mv = mv * (1.0 / (-float((mv | einf).e)))
        pts.append((round(float((mv | e1).e), 4), round(float((mv | e2).e), 4)))
    return sorted(pts)

print("recovered:", reconstruct_tripole_spectral(T))

# robustness: random triangles, including a vertical edge
rng = np.random.default_rng(7)
ok = True
for _ in range(6):
    Pr = [tuple(rng.normal(size=2).round(3)) for _ in range(3)]
    Tr = point(*Pr[0]) ^ point(*Pr[1]) ^ point(*Pr[2])
    got = reconstruct_tripole_spectral(Tr)
    want = sorted((round(x, 3), round(y, 3)) for x, y in Pr)
    match = all(abs(a-b) < 1e-2 for g, w in zip(got, want) for a, b in zip(g, w))
    ok &= match
Pv = [(1.5, 0.2), (1.5, -1.1), (-0.3, 0.8)]          # two equal x (vertical edge)
gotv = reconstruct_tripole_spectral(point(*Pv[0]) ^ point(*Pv[1]) ^ point(*Pv[2]))
print("random triangles all OK:", ok, "  vertical-edge case:", gotv)"""))

C.append(new_markdown_cell(r"""## 7. Summary

* $(V\lrcorner\mathbf T)\lrcorner\mathbf T=\mathbf T^2\,P_\Pi$ finds only the
  **plane** $\Pi$ — degenerate, a single eigenvalue $\mathbf T^2$.
* Six **moment functionals** (GA inner products with
  $e_\infty,e_1,e_2,e_{o_1},e_{o_2},e_{o_3}$) read the Veronese coordinates of
  any vector in $\Pi$.
* The **multiplication pencil** $A_u c = (u\!\cdot\!p)\,A_0 c$ has eigenvalues the
  coordinates and its $3\times3$ characteristic polynomial **is** the irreducible
  cubic — Cardano delivered by a standard eigen-solver, no parametrisation, no GCD.
* **Why exactly three:** the available moments are degree $\le 2$, so the
  degree-1 moment (catalecticant) matrix is $3\times3$ — a perfect fit for the
  tripole. Multiplication by $x$ keeps $\{1,x,y\}$ inside the degree-2 reads
  ($x\cdot x=x^2$, $x\cdot y=xy$). For $n\ge4$ points the same closure would need
  degree-3 moments, which the degree-2 conic point does not carry; the quadpole
  is instead handled by the pencil + **resolvent cubic** in
  `ccga.extract.extract_quadpole` (Ferrari)."""))

nb["cells"] = C
nbf.write(nb, "notebook/tripole_spectral.ipynb")
print("wrote notebook/tripole_spectral.ipynb with", len(C), "cells")
