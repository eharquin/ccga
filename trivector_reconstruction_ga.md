
# Geometric Algebra Reconstruction of 3 Points from Trivector

## 1. Setting

We start from three points
\[
p_i = (x_i, y_i), \quad i=1,2,3
\]

embedded into a Veronese-type map:
\[
\Phi(x,y) = (1, x, y, x^2, y^2, xy)
\]

The associated trivector is:
\[
\mathbf{T} = P_1 \wedge P_2 \wedge P_3, \quad P_i = \Phi(p_i)
\]

---

## 2. Key scalar invariants

### Euclidean area
\[
\Sigma =
\begin{vmatrix}
x_1 & y_1 & 1\\
x_2 & y_2 & 1\\
x_3 & y_3 & 1
\end{vmatrix}
\]

Interpretation:
- oriented area of triangle
- collinearity detector

---

### Vandermonde-x
\[
V_x =
-\begin{vmatrix}
1 & x_1 & x_1^2\\
1 & x_2 & x_2^2\\
1 & x_3 & x_3^2
\end{vmatrix}
\]

Interpretation:
- 1D quadratic curvature in x-direction
- degeneracy of projection onto parabola \(x \mapsto x^2\)

---

### Vandermonde-y
\[
V_y =
-\begin{vmatrix}
1 & y_1 & y_1^2\\
1 & y_2 & y_2^2\\
1 & y_3 & y_3^2
\end{vmatrix}
\]

Interpretation:
- same for y-direction

---

## 3. Trivector decomposition

\[
\mathbf{T}
=
\Sigma \, e_o \wedge e_1 \wedge e_2
-\frac12 V_x \, e_o \wedge e_1 \wedge e_{\infty_1}
-\frac12 V_y \, e_o \wedge e_2 \wedge e_{\infty_2}
+\cdots
\]

Structure:
- Euclidean part: \(\Sigma\)
- axis-curvature parts: \(V_x, V_y\)
- mixed quadratic coupling terms
- full conic interaction terms

---

## 4. Determinant interpretation

Let the Veronese matrix be:
\[
M =
\begin{pmatrix}
1 & x_1 & y_1 & x_1^2 & y_1^2 & x_1 y_1\\
1 & x_2 & y_2 & x_2^2 & y_2^2 & x_2 y_2\\
1 & x_3 & y_3 & x_3^2 & y_3^2 & x_3 y_3
\end{pmatrix}
\]

Each coefficient of \(\mathbf{T}\) is a 3×3 minor of \(M\).

---

## 5. Geometric meaning of the trivector

\[
\mathbf{T} \in \Lambda^3(\mathbb{R}^6)
\]

It encodes:
- a 3D subspace \(\Pi = \mathrm{span}(P_1,P_2,P_3)\)
- not individual points directly

---

## 6. Reconstruction principle

### Step 1: recover plane
\[
\mathbf{T} \Rightarrow \Pi
\]

### Step 2: intersection with Veronese surface
\[
\Phi(x,y) \in \Pi
\]

This gives a quadratic system whose solutions are:
\[
\{P_1,P_2,P_3\}
\]

---

## 7. Geometric algebra construction

Define operator:
\[
\mathcal{A}(V) = (V \cdot \mathbf{T}) \lrcorner \mathbf{T}
\]

Properties:
- acts inside plane \(\Pi\)
- encodes mutual geometry of \(P_i\)

Eigenproblem:
\[
\mathcal{A}(P_i) = \lambda_i P_i
\]

---

## 8. Extraction of points

Once \(P_i\) is recovered:
\[
x_i = \frac{P_i \cdot e_1}{P_i \cdot e_o},
\quad
y_i = \frac{P_i \cdot e_2}{P_i \cdot e_o}
\]

---

## 9. Conceptual summary

The trivector is:
- Plücker coordinates of a 3-plane in Veronese space
- a complete invariant of three conic points

Reconstruction is:
- spectral decomposition inside that plane
- followed by projection back to \((x,y)\)

---
