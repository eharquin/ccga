# CCGA objects — general form

Closed-form taxonomy of the CCGA objects built by wedging points, with reality
conditions. Symbolically verified in `notebook/symbolic.ipynb`; implemented in
`ccga/objects.py` and `ccga/cga.py`.

## Definition

$$
\begin{align}\displaystyle
&e_o = e_{o_1} + e_{o_2}
& e_{\infty} =  \frac{e_{\infty_1} + e_{\infty_2}}{2}\\
&e_{\overline{o}} = e_{o_1} - e_{o_2}
&e_{\overline{\infty}}=\frac{e_{\infty_1} - e_{\infty_2}}{2} \\
&I_o^\triangleright = \left(e_{o_1} - e_{o_2}\right) \wedge e_{o_3} &I_{\infty}^\triangleright\! =\! \left(e_{\infty_1}\! -\! e_{\infty_2}\right)\! \wedge\! e_{\infty_3} \\
&I_o = e_{o_1} \wedge e_{o_2} \wedge e_{o_3}
&I_{\infty} = e_{\infty_1} \wedge e_{\infty_2} \wedge e_{\infty_3}  \\
&I_\epsilon = e_1 \wedge e_2
&I = I_\epsilon \wedge I_{\infty} \wedge I_o
\end{align}
$$

---

## Grade 1

The base point element of CCGA.

### conic point

**general conic point**

$$
p = w\,e_o + x\,e_1 + y\,e_2 + \frac{x^2}{2}\,e_{\infty_1} + \frac{y^2}{2}\,e_{\infty_2} + xy\,e_{\infty_3} \pm \frac{r^2}{2}\,e_{\infty}
$$

with (for $w=1$, $\;p^2 = \pm r^2$):

$$
\begin{align}
r^2 &> 0 \quad \text{real radius (a circle / round point)}\\
r^2 &= 0 \quad \text{zero radius (a finite point, } p^2=0)\\
r^2 &< 0 \quad \text{imaginary radius}
\end{align}
$$

If $w=0$ the point is *ideal*.

**ideal conic point**

$$
p_\infty = x\,e_1 + y\,e_2 + \frac{x^2}{2}\,e_{\infty_1} + \frac{y^2}{2}\,e_{\infty_2} + xy\,e_{\infty_3} \pm \frac{r^2}{2}\,e_{\infty}
$$

> `ccga.objects.make_point_ccga(x, y, r=0, imaginary=False)` (set
> `imaginary=True` for $-\tfrac{r^2}{2}e_\infty$); `make_ideal_point(...)` drops $e_o$.
> Note $\pm\tfrac{r^2}{2}e_\infty = \pm\tfrac{r^2}{4}(e_{\infty_1}+e_{\infty_2})$, so it
> shifts the $e_{\infty_1},e_{\infty_2}$ coefficients symmetrically — one isotropic
> radius, exactly as in CGA.

---

## Grade 2

### conic point pairs

**general conic point pair**

$$
pp = p_1 \wedge p_2
$$

Parametrise by **center** $p=(p_x,p_y)$, **unit direction** $v=(v_x,v_y)$ and
**radius** $r$ (half the separation), with endpoints the base points at
$p \pm r\,v$. Since the point map $f(t)=\text{point}(p+t\,v)$ is quadratic,
$f(t)=C+t\,W+t^2 v_\infty$, the dipole collapses to

$$
\boxed{\;pp \;=\; p_1 \wedge p_2 \;=\; -2r\,\big(C + r^2\,v_\infty\big)\wedge W\;}
$$

with the three pieces (the $0^\text{th}/1^\text{st}/2^\text{nd}$ derivatives of the point map):

$$
\begin{align}
C       &= e_o + p_x e_1 + p_y e_2 + \tfrac{p_x^2}{2}e_{\infty_1} + \tfrac{p_y^2}{2}e_{\infty_2} + p_xp_y\,e_{\infty_3}  &&\text{(center point)}\\
W       &= v_x e_1 + v_y e_2 + p_xv_x\,e_{\infty_1} + p_yv_y\,e_{\infty_2} + (p_xv_y+p_yv_x)\,e_{\infty_3}  &&\text{(tangent } \mathrm dC[v]\text{)}\\
v_\infty &= \tfrac{v_x^2}{2}e_{\infty_1} + \tfrac{v_y^2}{2}e_{\infty_2} + v_xv_y\,e_{\infty_3}  &&\text{(ideal point of } v\text{)}
\end{align}
$$

The overall $-2r$ is scale/orientation (a dipole is projective); the geometry sits
in $(C+r^2 v_\infty)\wedge W$, where $r^2$ is the **signed squared radius**:

$$
\begin{align}
r^2 &> 0 \quad \text{real pair (two real points)}\\
r^2 &= 0 \quad \text{tangent / degenerate (coincident)}\\
r^2 &< 0 \quad \text{imaginary pair (no real points)}
\end{align}
$$

Square (unit $v$): $\;pp^2 = 4r^4\;$ (from $pp^2=(p_1\!\cdot p_2)^2=\tfrac14\,\mathrm{dist}^4$, $\mathrm{dist}=2r$).

> This is the CCGA lift of the standard CGA(2) dipole
> $-2r\big[v_x e_{01}+v_y e_{02}+(p_xv_y{-}p_yv_x)e_{12}+(p\!\cdot\! v)(p_x e_{1\infty}{+}p_y e_{2\infty}{+}p_w e_{0\infty})-\tfrac{p_x^2+p_y^2\pm r^2}{2}(v_x e_{1\infty}{+}v_y e_{2\infty})\big]$
> (verified, §6 of the notebook): same $-2r$ scale and center$\wedge$direction +
> radius split, but CCGA spreads $e_o,e_\infty$ over $e_{o_1},e_{o_2}$ and
> $e_{\infty_1},e_{\infty_2},e_{\infty_3}$.
> Implemented as `ccga.objects.make_point_pair(p1, p2)` (`p1 ^ p2`).

### conic flat point

$$
fp = p \wedge e_\infty
$$

A finite point flattened to infinity (position, no round extent). General form:

$$
fp = \underbrace{e_o\wedge e_\infty + x\,e_1\wedge e_\infty + y\,e_2\wedge e_\infty}_{\text{flat-point core }(p_{\text{lin}}\wedge e_\infty)}
   \;+\; \Big(\tfrac{x^2}{2}e_{\infty_1}+\tfrac{y^2}{2}e_{\infty_2}+xy\,e_{\infty_3}\Big)\wedge e_\infty
$$

The first group is the position $(x,y)$ (with origin weight $e_o\wedge e_\infty$);
the second is the point's quadratic part wedged with $e_\infty$ (the
$e_{\infty_i}\wedge e_{\infty_j}$ terms). A flat point has no radius — it is real.

> `ccga.cga.flat_point(p)` builds the CGA flat point $p\wedge e_\infty\wedge I_\infty^\triangleright$
> (grade 4); the bare $p\wedge e_\infty$ above is its grade-2 CCGA precursor.

---

## Higher grades (forward pointers)

Built by wedging more points (and, for the CGA round family, the gauge blade
$I_\infty^\triangleright$):

- **CGA round family** (`ccga.cga`, $\wedge\,I_\infty^\triangleright$): round point
  (3), point pair (4), flat point (4), circle (5), line (5). Reality is uniform:
  $\operatorname{sign}\big((I_o^\triangleright \,\lrcorner\, O)^2\big)$.
- **General conic** — OPNS grade 7 $\;I_o^\triangleright\wedge p_1\wedge\cdots\wedge p_5$,
  IPNS dual grade 1 $\;A x^2+B y^2+C xy+D x+E y+F$ (see `OBJECTS.md`).
- **Pseudoscalar** — grade 8, $I$.

## Grade 3 / 4 — tripole and quadpole

Wedging three or four points keeps **all** the points (unlike CGA, where
$p_1\wedge p_2\wedge p_3$ is only the circle through them):

- **Tripole** $\;T=p_1\wedge p_2\wedge p_3$ (grade 3). Full component expansion in
  the null basis: see the Obsidian note `Objects/Conic Tripole.md`. Circumcircle
  read off by the closed form $T\wedge I_o^\triangleright\wedge I_\infty^\triangleright$.
- **Quadpole** $\;Q=p_1\wedge p_2\wedge p_3\wedge p_4$ (grade 4), a wedge of two
  dipoles $Q=pp_{ij}\wedge pp_{kl}$ for each of the 3 pairings. Full expansion:
  `Objects/Conic Quadpole.md`.
- **Pentapole** $\;P_5=p_1\wedge\dots\wedge p_5$ (grade 5) — the last "wedge of
  points" rung, and **already the conic** as a point locus
  ($q\wedge P_5=0\Leftrightarrow q$ on the conic; all conic points share a 5-D
  subspace of $V_6$). Its dual is grade 3, $-\tfrac12\,(s\wedge I_\infty^{\triangleright})$:
  the clean conic vector $s$ smeared by the infinity gauge. Wedging
  $I_o^{\triangleright}$ promotes it to the grade-7 conic whose dual is the clean
  grade-1 $s$ — a pure gauge/dual fix, not a geometric one. Conic **type** is set
  by the number of true ideal points `point_at_infinity` in the build (0 → ellipse,
  1 double → parabola, 2 → hyperbola, the latter being its asymptotic directions).
  See `OBJECTS.md` §1 and `notebook/conic_construction.ipynb`.

Point recovery (`ccga/extract.py`): no single $\pm\sqrt{}$ for 3+ points, but a
GA-native reduction to dipoles + a radical solve — circumcircle + Cardano cubic
(tripole); GA pencil + resolvent cubic + two $\pm\sqrt{}$ (quadpole, Ferrari).
See `OBJECTS.md` for the API.

(Say the word to expand the CGA round family / conic / pseudoscalar here too.)
