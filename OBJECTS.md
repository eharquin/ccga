# CCGA Object Catalog

Complete reference of the geometric objects, constructions, properties, and
transformations of **Conic Conformal Geometric Algebra** (CCGA, $\mathbb{R}^{5,3}$),
as implemented in this repo. Every entry is exported from `ccga` (`ccga/__init__.py`)
and covered by `tests/`. See `CLAUDE.md` for the algebra setup (§1), point embedding
(§2), and ground-truth anchors (§3).

**GA-honesty marker.** Entries are pure geometric algebra unless tagged
**[num]**, which marks a routine that falls back to coordinate / linear-algebra
solving (eigen, SVD, polynomial roots, Cardano/Ferrari). The *objects* are always
GA multivectors; only the scalar **extraction** of roots/eigenvalues is numeric.

---

## 1. Conventions & special blades

- **Diagonal basis** `e1,e2` (Euclidean, +), `e3,e4,e5 = e₊₁,e₊₂,e₊₃` (+),
  `e6,e7,e8 = e₋₁,e₋₂,e₋₃` (−). **Null basis** `eo_i = e₊ᵢ+e₋ᵢ`,
  `einf_i = (e₋ᵢ−e₊ᵢ)/2`; `eo_i²=einf_i²=0`, `eo_i·einf_i=−1`.
- **Dual** (`ccga/operations.py`): `ipns = dual(opns) = opns * I⁻¹`;
  `opns = undual(ipns) = ipns * I`. Fixed once against §3 result 7.
- **Join** = outer product `A ^ B`; **Meet** = regressive product `A & B` (= `meet`).
- **Display**: `from ccga import print_null` (null-basis order
  `eo1 eo2 eo3 e1 e2 einf1 einf2 einf3`; `multiline=True` for one blade per line).
- **Reality** (round/conic objects): `s² > 0` real, `< 0` imaginary, `≈ 0`
  degenerate — and `sign(s²)` is **scale-invariant** (`(λs)²=λ²s²`). General conics
  use $\Delta = C^2-4AB$ on $(A,B,C,D,E,F)$.

### Special blades (`ccga/algebra.py`)

| Symbol | Multivector | Grade | Role |
|---|---|---|---|
| $e_o$ | `eo1 + eo2` | 1 | combined origin |
| $e_\infty$ | `(einf1 + einf2)/2` | 1 | combined infinity ($e_o\!\cdot e_\infty=-1$) |
| $\bar e_o$ | `eo1 − eo2` | 1 | differential origin |
| $\bar e_\infty$ | `(einf1 − einf2)/2` | 1 | differential infinity |
| $I_o^{\triangleright}$ (`Iod`) | `(eo1−eo2) ∧ eo3` = $\bar e_o\wedge e_{o3}$ | 2 | **origin gauge** = the directions $W_2$ points can't reach; dual cleaner |
| $I_\infty^{\triangleright}$ (`Iinfd`) | `(einf1−einf2) ∧ einf3` | 2 | **infinity gauge** = the **circular-point pair** $i\,(I\wedge J)$ (§12) |
| $I_\infty$ (`Iinf`) | `einf1 ∧ einf2 ∧ einf3` | 3 | the line/conic at infinity |
| $I_o$ (`Io`) | `eo1 ∧ eo2 ∧ eo3` | 3 | origin 3-blade |
| $I_\epsilon$ (`Ieps`) | `e1 ∧ e2` | 2 | Euclidean pseudoscalar |
| $I$ | `Ieps ∧ Iinf ∧ Io` | 8 | pseudoscalar ($I^2=-1$, $I^{-1}=-I$) |

### Objects by grade (quick index)

OPNS / native grade of each object. Its IPNS dual sits at grade $8-g$ (so a *conic*
appears both as grade-1 IPNS and grade-7 OPNS; a *dipole* as grade-2 OPNS / grade-6 IPNS).

| Grade | Objects (native grade) | § |
|---|---|---|
| 0 | scalar | — |
| 1 | point; round point ($\pm r^2$); ideal point `point_at_infinity`; CGA ideal point `make_ideal_point`; `tangent_at_infinity`; **IPNS conics**: general / circle / ellipse / hyperbola / parabola / line / line-pair; lines & polars (`polar/tangent/normal_line`, `apollonius_conic`); blades $e_o,e_\infty,\bar e_o,\bar e_\infty$ | 2, 4, 8 |
| 2 | **twopole** ($p_1\wedge p_2$, bare; = `make_point_pair`/`twopole`); tangent point; gauge blades $I_o^{\triangleright},I_\infty^{\triangleright},I_\epsilon$ | 1, 3 |
| 3 | tripole; **pencil** $p_1\wedge I_o^{\triangleright}$ ($n{=}1$ rung); CGA round point ($p\wedge I_\infty^{\triangleright}$); line at infinity $I_\infty$; $I_o$ | 3, 6, 7 |
| 4 | quadpole; **gauged dipole** / pencil $n{=}2$ ($p_1\wedge p_2\wedge I_o^{\triangleright}$, `make_gauged_dipole`; self-dual, = conic∨line 2-pt object); flat point ($p\wedge I_\infty$); CGA point pair ($p_1\wedge p_2\wedge I_\infty^{\triangleright}$); CGA flat point | 3, 6, 7 |
| 5 | pentapole; **pencil** $n{=}3$; flat line ($p_1\wedge p_2\wedge I_\infty$ = CGA line); CGA circle; conic at infinity ($I_o^{\triangleright}\wedge I_\infty$) | 3, 6, 7 |
| 6 | **pencil** $n{=}4$ = conic ∨ conic intersection ($Q\wedge I_o^{\triangleright}$); plane / 2-flat | 6, 10 |
| 7 | **OPNS conics**: general conic ($p_1{\wedge}\dots{\wedge}p_5{\wedge}I_o^{\triangleright}$) & the named 3-point conics (ellipse / hyperbola / parabola = 3 pts $+$ ideal-point pair $+\,I_o^{\triangleright}$); line pair / parallel / secant pair | 3, 4, 5 |
| 8 | pseudoscalar $I$ | 1 |

**Construction ladders.** Every object above is a wedge of points with at most the two
gauge blades. Three ladders sort the zoo: (1) the **bare multipole ladder** $p \to
p_1{\wedge}p_2 \to \dots \to p_1{\wedge}\dots{\wedge}p_5$ (point, twopole, tripole,
quadpole, pentapole); (2) the **$I_o^{\triangleright}$-gauged conic ladder** — for $n\le4$
points $p_1{\wedge}\dots{\wedge}p_n{\wedge}I_o^{\triangleright}$ (grade $n{+}2$) is a
*pencil* (gauged dipole at $n{=}2$, conic∨conic blade at $n{=}4$); a full CCGA conic is the
grade-7 blade ending in ${\wedge}I_o^{\triangleright}$ (general $=$ 5 pts; the smooth named
conics $=$ 3 pts $+$ an ideal-point pair $+\,I_o^{\triangleright}$ — real ⇒ hyperbola,
double ⇒ parabola, imaginary ⇒ ellipse, circular points $\Rightarrow$ circle, hence on the
CGA side); (3) **CGA embedded in CCGA**, the $I_\infty^{\triangleright}$ sub-array (round
point, point pair, circle, line, plus flat versions) sitting at grade $=$ CGA grade $+\,2$.
See `tests/test_taxonomy.py`.

---

## 2. Points (grade 1)

The embedding $p = e_o + x\,e_1 + y\,e_2 + \tfrac{x^2}2 e_{\infty_1} + \tfrac{y^2}2 e_{\infty_2} + xy\,e_{\infty_3}$
(the Veronese of $(x,y)$). A point carries an optional radius $\mp\tfrac{r^2}2 e_\infty$.

| Object | Multivector / constructor | $p^2$ | $p\!\cdot e_\infty$ | meaning |
|---|---|---|---|---|
| **Point** (finite) | `point(x,y)` / `make_point_ccga(x,y)` | $0$ (null) | $-1$ | the point; as IPNS = zero-radius circle |
| **Round point — real** | `make_point_ccga(x,y,r)` = $p-\tfrac{r^2}2 e_\infty$ | $+r^2$ | $-1$ | real circle radius $r$ (IPNS) |
| **Round point — imaginary** | `make_point_ccga(x,y,r,imaginary=True)` = $p+\tfrac{r^2}2 e_\infty$ | $-r^2$ | $-1$ | imaginary circle (empty real locus) |
| **Ideal point** (true, at infinity) | $v_\infty=\tfrac{v_x^2}2 e_{\infty_1}+\tfrac{v_y^2}2 e_{\infty_2}+v_xv_y\,e_{\infty_3}$ — `point_at_infinity(vx,vy)` | $0$ (null) | $0$ | the **genuine** point at infinity: pure degree-2 (lives on the conic at infinity), $=\lim_{t\to\infty}p(t v)/t^2$; controls asymptotes/type |
| **CGA ideal point** (`eo` dropped) | $x e_1 + y e_2 + \tfrac{x^2}2 e_{\infty_1}+\tfrac{y^2}2 e_{\infty_2}+xy\,e_{\infty_3}\ (\mp\tfrac{r^2}2 e_\infty)$ — `make_ideal_point(x,y,r,imaginary)` | $x^2+y^2$ | $0$ | $p$ with the homogeneous coord zeroed — a degree-1⊕degree-2 **hybrid**, *not* a true ideal point; reads as a **line** (radius just shifts it); used only by the CGA round family (§7) |
| **Tangent at infinity** | $v_\infty'=-cs\,e_{\infty_1}+cs\,e_{\infty_2}+(c^2-s^2)e_{\infty_3}$, $(c,s)=v/\lVert v\rVert$ — `tangent_at_infinity(vx,vy)` | — | $0$ | $\partial_\theta$ of $v_\infty$; the double-contact direction (parabola) |

Notes: a point is the **isotropic, often-null member of the conic family** — as an
IPNS conic it is the zero-radius circle (`A=B`, `C=0`, `r=0`); it is **self-dual**
(`undual(p)` is the grade-7 OPNS form of that point-conic). Reality (real / imaginary
/ null) is meaningful only for **finite** points ($p\!\cdot e_\infty\neq0$); ideal
points carry no meaningful radius.

**The two ideal points are different objects.** The embedding is the degree-2 Veronese
$(1,x,y,\tfrac{x^2}2,\tfrac{y^2}2,xy)$ with three strata — degree 0 ($e_o$), degree 1
($e_1,e_2$), degree 2 ($e_{\infty}$). The *true* ideal point is the limit of a receding
point, $\lim_{t\to\infty}p(tv)/t^2$, which keeps **only the degree-2 part** (degree 0
$\sim 1/t^2$ and degree 1 $\sim 1/t$ both vanish) → `point_at_infinity(v)`. Merely
*zeroing the homogeneous coordinate* $e_o$ (degree 0) leaves the degree-1 part
$x e_1+y e_2$ behind → `make_ideal_point(v)`, a hybrid that is **not** a point at
infinity (it reads as a line and does **not** control asymptotes). In a plain linear
model $(1,x,y)$ there is no degree-2 stratum, so "drop the homogeneous coord" *would*
give the ideal point — the Veronese's extra stratum is exactly why it doesn't here.

---

## 3. The point→conic wedge ladder

Each extra point raises the grade by one; wedging $I_o^{\triangleright}$ at the end
gives the clean grade-1 dual.

| Object | OPNS | grade | IPNS dual | grade | construction |
|---|---|---|---|---|---|
| **Point** | $p$ | 1 | (self) | 1 | `point(x,y)` |
| **Dipole** (point pair) | $p_1\wedge p_2$ | 2 | grade 6 | 6 | `make_point_pair(p1,p2)` |
| **Tangent point** | $p\wedge t$ (null) | 2 | grade 6 | 6 | `make_tangent_point(p,t)` |
| **Tripole** | $p_1\wedge p_2\wedge p_3$ | 3 | grade 5 | 5 | `make_conic_tripole(...)` |
| **Quadpole** | $p_1\wedge\dots\wedge p_4$ | 4 | grade 4 | 4 | `make_conic_quadpole(...)` |
| **Pentapole** | $p_1\wedge\dots\wedge p_5$ | 5 | grade 3 | 3 | `make_conic_pentapole(...)` |
| **Conic** | $p_1\wedge\dots\wedge p_5\wedge I_o^{\triangleright}$ | 7 | grade 1 | 1 | `conic_from_5points` / `pentapole_to_conic(P5)` |

Dipole reality: $P^2>0$ real pair, $<0$ imaginary, $0$ tangent.

### grade-5 vs grade-7 — `Iod` is a dual cleaner, not geometry

The pentapole $P_5$ and the conic $C_7 = P_5\wedge I_o^{\triangleright}$ are the
**same conic as a point locus**: $q\wedge P_5 = 0 \Leftrightarrow q\wedge C_7 = 0
\Leftrightarrow q$ on the conic (a conic is one linear relation among the 6 Veronese
coords, so 5 points span its hyperplane). They differ only in the dual:

| | OPNS grade | dual | dual grade |
|---|---|---|---|
| $P_5$ | 5 | $-\tfrac12\,(s\wedge I_\infty^{\triangleright})$ | 3 |
| $C_7=P_5\wedge I_o^{\triangleright}$ | 7 | $s = (A,B,C,D,E,F)$ | 1 |

Helpers: `pentapole_to_conic`, `conic_dual_grade1`, `conic_type` (accepts grade 1/5/7).

**Does the bare $n$-wedge equal the curve?** Only if $n$ points span the curve's
hyperplane: a **conic** needs 5 (the 5-wedge *is* the conic, `Iod` only cleans the
dual); a **circle** needs 3 *plus roundness*, so the bare tripole ≠ circle — you must
wedge `Iinfd` (§5, §7).

---

## 4. Conics — algebraic (IPNS grade 1) and named constructors

An IPNS grade-1 vector **is** the general conic $Ax^2+By^2+Cxy+Dx+Ey+F=0$ via
$$s = -2A\,e_{o1} - 2B\,e_{o2} - C\,e_{o3} + D\,e_1 + E\,e_2 - \tfrac{F}{2}(e_{\infty_1}+e_{\infty_2}).$$
Shape $(A,B,C)$ lives on the **origin** side ($e_{o1},e_{o2},e_{o3}$); $D,E$ on
$e_1,e_2$; $F$ (size) on the **infinity** side $e_\infty$. (`ipns_to_coeffs` inverts this.)

| Object | construction | type test |
|---|---|---|
| **General conic** (IPNS) | `make_conic_ipns(A,B,C,D,E,F)` | $\Delta=C^2-4AB$ |
| **From 5 points** | `conic_from_5points([p1..p5])` = `make_conic_opns` ∧-dual | discriminant |
| **From 5 tangent lines** | `conic_from_5_tangents([l1..l5])` **[num]** (dual conic, $M_3=\mathrm{adj}(M^*)$) | — |
| **Circle** | `make_circle(cx,cy,r)` = $p_c-\tfrac{r^2}2 e_\infty$ | $A{=}B,\ C{=}0$; $s^2=r^2$ |
| **Ellipse** | `make_ellipse(a,b,cx,cy)` | $\Delta<0$ |
| **Hyperbola** | `make_hyperbola(a,b,cx,cy)` | $\Delta>0$ |
| **Parabola** | `make_parabola(p_focus,axis)` | $\Delta=0$ |
| **Tilted ellipse** | `make_tilted_ellipse(a,b,θ,cx,cy)` ($C\neq0$ via $e_{o3}$) | $\Delta<0$ |
| **Line** (degenerate conic) | `make_line_ipns(nx,ny,d)` / `make_line_2points(p1,p2)`; OPNS `p∧q∧Iinf∧Iod` | $A{=}B{=}C{=}0$ |
| **Line pair** (degenerate) | `make_line_pair(l1,l2)` — symmetric square of two lines | $\det M_3=0$ |
| **Parallel line pair** | `make_parallel_line_pair(E,F,G)` — line $EF$ + parallel through $G$ | $\Delta=0,\ \det M_3=0$ |
| **Secant pair through origin** | `make_secant_line_pair_through_origin(P,Q,R)` — line $OP$ + line $QR$ | $\Delta>0,\ \det M_3=0$ |

### Explicit IPNS multivectors $s = -2A\,e_{o1}-2B\,e_{o2}-C\,e_{o3}+D\,e_1+E\,e_2-\tfrac F2(e_{\infty_1}+e_{\infty_2})$

The named constructors are just this $s$ with specific $(A,B,C,D,E,F)$:

- **Circle** $(c_x,c_y,r)$: $A=B=1,\ C=0,\ D=-2c_x,\ E=-2c_y,\ F=c_x^2+c_y^2-r^2$
  (equivalently $s = e_o + c_x e_1 + c_y e_2 + \dots - \tfrac{r^2}2 e_\infty$, i.e. $p_c-\tfrac{r^2}2 e_\infty$).
- **Ellipse** $(a,b,c_x,c_y)$: $A=\tfrac1{a^2},\ B=\tfrac1{b^2},\ C=0,\ D=-\tfrac{2c_x}{a^2},\ E=-\tfrac{2c_y}{b^2},\ F=\tfrac{c_x^2}{a^2}+\tfrac{c_y^2}{b^2}-1$.
- **Hyperbola** $(a,b,c_x,c_y)$: $A=\tfrac1{a^2},\ B=-\tfrac1{b^2},\ C=0,\ D=-\tfrac{2c_x}{a^2},\ E=+\tfrac{2c_y}{b^2},\ F=\tfrac{c_x^2}{a^2}-\tfrac{c_y^2}{b^2}-1$ (i.e. $D=-2c_xA,\ E=-2c_yB$).
- **Parabola** $y^2{=}4px$: $A{=}0,B{=}1,C{=}0,D{=}-4p,E{=}0,F{=}0$; $x^2{=}4py$: $A{=}1,B{=}0,C{=}0,D{=}0,E{=}-4p,F{=}0$.
- **Tilted ellipse** $(a,b,\theta,c_x,c_y)$: $A=\tfrac{\cos^2\theta}{a^2}+\tfrac{\sin^2\theta}{b^2},\ B=\tfrac{\sin^2\theta}{a^2}+\tfrac{\cos^2\theta}{b^2},\ C=2\sin\theta\cos\theta(\tfrac1{a^2}-\tfrac1{b^2}),\ D=-2(Ac_x+\tfrac C2 c_y),\ E=-2(Bc_y+\tfrac C2 c_x),\ F=Ac_x^2+Bc_y^2+Cc_xc_y-1$.
- **Line** $n_x x+n_y y+d=0$: $A=B=C=0,\ D=n_x,\ E=n_y,\ F=d$ — i.e. $s = n_x e_1+n_y e_2-\tfrac d2(e_{\infty_1}+e_{\infty_2})$.
- **Line pair** $\ell_1\ell_2$, $\ell_i=(a_i,b_i,c_i)$: $A=a_1a_2,\ B=b_1b_2,\ C=a_1b_2+a_2b_1,\ D=a_1c_2+a_2c_1,\ E=b_1c_2+b_2c_1,\ F=c_1c_2$ (symmetric square).

A **line** is the most degenerate conic: as IPNS it has no $e_o$ part at all (only
$e_1,e_2$ = normal, $e_\infty$ = offset). A **line pair** is the symmetric square of
two line covectors (there is no single GA product for it); factor it back with
`extract._lines_of`; detect degeneracy with `conic_is_degenerate` **[num]**.

---

## 5. Constructive conic type — 3 points + an ideal-point pair

A conic's type is its **incidence with the line at infinity**. Constructively,
$C = p_1\wedge p_2\wedge p_3\wedge B\wedge I_o^{\triangleright}$ where $B$ is the
**ideal-point pair** (a bivector = a line in the plane at infinity); the type is how
$B$'s line meets the conic-at-infinity (the Veronese cone):

| $B$ (the 2 ideal points) | line vs ∞-conic | type | constructor |
|---|---|---|---|
| 2 **real** dirs $v_\infty(v_1)\wedge v_\infty(v_2)$ | secant | hyperbola | `make_hyperbola_3points(p1,p2,p3,d1,d2)` |
| 1 **double** dir | tangent | parabola | `make_parabola_3points(p1,p2,p3,axis)` |
| 2 **imaginary** dirs $(a^2 e_{\infty_1}-b^2 e_{\infty_2})\wedge e_{\infty_3}$ | non-secant | ellipse | `make_ellipse_3points(p1,p2,p3,a,b)` |
| circular points $I_\infty^{\triangleright}$ | non-secant | circle | `cga.circle(p1,p2,p3)` |
| 2 **opposite** ideal pts `make_ideal_point(±v)`, $v=E-F$ | merged → degenerate, both lines dir $v$ | **parallel line pair** | `make_parallel_line_pair(E,F,G)` |
| 2 **opposite** ideal pts `make_ideal_point(±v)`, $v=P$ (position) | one line forced to dir $v$ = origin→$P$ | **secant pair through origin** | `make_secant_line_pair_through_origin(P,Q,R)` |

The two degenerate rows are the same family: $3\text{ pts}\wedge \texttt{make\_ideal\_point}(\pm v)\wedge I_o^{\triangleright}$.
Since `make_ideal_point(v)` is the **ideal point of direction $v$** (on every line of
direction $v$), it forces one component line to have direction $v$: $v=E-F$ → the line
$EF$ and its parallel through $G$ (parallel pair); $v=P$ → the line of direction $P$
through the origin (which contains $P$) and the line through the other two (secant pair).

The 2 real directions of a hyperbola **are** its asymptotes. The parabola arises as
the merging limit of two ideal points ($\Delta\to0$). The ellipse pair is a *weighted*
$I_\infty^{\triangleright}$ ($a=b$ → circle). Reality is **not** $B^2$ (the infinity
space is null, $B^2=0$); it is the secant/non-secant nature. Read type back with
`conic_type` and directions with `asymptotic_directions`.

---

## 6. Flats & ideal elements

| Object | Multivector / constructor | grade | meaning |
|---|---|---|---|
| **Flat point** | `make_flat_point(x,y)` = $p\wedge I_\infty$ ($=-(p\wedge e_\infty\wedge I_\infty^{\triangleright})$) | 4 | point pinned to infinity — **pure position** ($=(e_o+x e_1+y e_2)\wedge I_\infty$); **not** $p\wedge e_\infty$ |
| **Line (flat)** | `point ∧ flat_point` = $p_1\wedge p_2\wedge I_\infty$ | 5 | line through 2 points (= `cga.line`) |
| **Plane (2-flat)** | `point ∧ line` | 6 | whole plane (2-D) |
| **Line at infinity** | `make_line_at_infinity()` = $I_\infty$ | 3 | $L_\infty$ |
| **Conic at infinity** | `make_conic_at_infinity()` = $I_o^{\triangleright}\wedge I_\infty$ | 5 | the Veronese conic at ∞; `conic ∨ C_∞ → grade 4 → \| Iinfd →` asymptotic dipole (`extract.asymptotic_dipole`, INTERSECTIONS §5.1) |
| **Ideal point** | `make_ideal_point(x,y,r,imaginary)` | 1 | direction (CGA-flavored; see §2) |
| **Conic's ideal points** | `extract.ideal_points(conic)` (via `asymptotic_dipole`) | 1 each | the conic ∩ $L_\infty$: 2 (hyperbola) / 1 (parabola) / 0 real (ellipse), as `point_at_infinity`; GA route to `asymptotic_directions` |

The flat point is the **join-unit** for flats: wedging a finite point raises the flat
dimension. Composition rules: `flat ∧ flat = 0` (repeated $I_\infty$),
`flat ∧ ideal = 0` (already in $I_\infty$), `flat ∧ finite point → line`. Flats are for
**joins**; use the IPNS line (§4) for **meets/intersections**.

---

## 7. CGA "round" family (via `Iinfd`)

Wedging $I_\infty^{\triangleright}$ recovers the full CGA round hierarchy inside CCGA
(`ccga/cga.py`); each CGA object of grade $k$ lands at grade $k{+}2$. Because
$I_\infty^{\triangleright}$ **is the circular-point pair** $\{I,J\}$ (§12), these are
exactly the conics/curves through the circular points (i.e. *round*).

| Object | construction | CCGA grade | CGA grade |
|---|---|---|---|
| **Round point / sphere** | `cga.round_point(p)` = $p\wedge I_\infty^{\triangleright}$ | 3 | 1 |
| **Point pair** | `cga.point_pair(p1,p2)` = $p_1\wedge p_2\wedge I_\infty^{\triangleright}$ | 4 | 2 |
| **Flat point** | `cga.flat_point(p)` = $p\wedge e_\infty\wedge I_\infty^{\triangleright}$ | 4 | 2 |
| **Circle** | `cga.circle(p1,p2,p3)` = $p_1\wedge p_2\wedge p_3\wedge I_\infty^{\triangleright}$ | 5 | 3 |
| **Line** | `cga.line(p1,p2)` = $p_1\wedge p_2\wedge e_\infty\wedge I_\infty^{\triangleright}$ | 5 | 3 |

- `cga.cga_blade(O) = Iod | O` (grade $-2$); `cga.reality(O) = sign((Iod|O)^2)`;
  `cga.is_finite(O)` (presence of $e_o$); `cga.classify_cga(O)`.
- `make_round_point(x,y,r)` (`objects.py`) is the coordinate entry point.
- **Incidence**: $q$ on OPNS object $O$ iff $q\wedge O = 0$.

---

## 8. Tangents, polars, normals, projection

| Object | construction | notes |
|---|---|---|
| **Polar line** of $q$ | `polar_line(conic,q)` = $\tfrac12(\partial_x p\!\cdot\! s)e_1+\tfrac12(\partial_y p\!\cdot\! s)e_2+(e_o+\tfrac{x}2 e_1+\tfrac{y}2 e_2)\!\cdot\! s$ | $=M_3[x,y,1]$ (pole–polar); GA via point-map differentials |
| **Tangent line** at $p\in C$ | `tangent_line(conic,p)` = `polar_line(conic,p)` (same formula, $p\in C$) | polar of a contact point; double contact |
| **Normal line** at $p\in C$ | `normal_line(conic,p)` = $-n_y e_1 + n_x e_2 - \tfrac{n_y x-n_x y}2(e_{\infty_1}+e_{\infty_2})$, $(n_x,n_y)$=tangent normal | ⟂ tangent through $p$ (gradient dir) |
| **Apollonius conic** of $q=(q_x,q_y)$ | `apollonius_conic(conic,q)` = `make_conic_ipns(`$-C,\,C,\,2(A{-}B),\ Cq_x{-}E{-}2Aq_y,\ 2Bq_x{-}Cq_y{+}D,\ Eq_x{-}Dq_y$`)` | rectangular hyperbola of normal feet, through $q$ and the center |
| **Normal feet** from $q$ | `normal_feet(conic,q)` **[num]** | $C \vee \text{Apollonius}(q)$ (up to 4) |
| **Orthogonal projection** | `project_point_to_conic(conic,q)` **[num]** | nearest foot; distance$(q,C)=\lVert q-\text{foot}\rVert$ |

Pole–polar reciprocity holds; the polar of the **center** is the line at infinity.
Build a conic from 5 tangent lines with `conic_from_5_tangents` **[num]** (§4).

---

## 9. Conic properties (read off the multivector)

| Property | function | notes |
|---|---|---|
| **Center** $(c_x,c_y)$ | `conic_center` — with $s_i=s\!\cdot e_{\infty_i}$, $\sigma_k=s\!\cdot e_k$: $c_x=\tfrac{s_3\sigma_2-s_2\sigma_1}{s_1s_2-s_3^2}$, $c_y=\tfrac{s_3\sigma_1-s_1\sigma_2}{s_1s_2-s_3^2}$ ($4\Delta_2=s_1s_2-s_3^2$) | pure-GA inner-product formula |
| **Center as a point** | `conic_center_point` = $p_c=(x_c,y_c,w_c)$ via `conic_center_meet` (pole of $L_\infty$) | finite point ($w_c\ne0$); **ideal** (point at infinity) for a parabola ($w_c=0$) |
| **Semi-axes** + directions | `conic_axes` **[num]**: $a_i^2=-F'/\lambda_i$, $\lambda_i$=eigvals of $\begin{psmallmatrix}A&C/2\\C/2&B\end{psmallmatrix}$, $F'$=conic at center; dirs=eigvecs | |
| **Eccentricity** | `conic_eccentricity` **[num]** = $c/a$, $c=\sqrt{a^2-b^2}$ (ellipse) / $\sqrt{a^2+b^2}$ (hyperbola) | $0$ circle, $<1$ ellipse, $1$ parabola, $>1$ hyperbola |
| **Foci** | `conic_foci` **[num]**: central $\text{center}\pm c\,\hat u_{\text{maj}}$; parabola $\text{vertex}+p\,\hat u_{\text{axis}}$ | |
| **Parabola geometry** | `parabola_geometry` **[num]** | vertex / axis / focal length / focus / directrix |
| **Discriminant $\Delta$** | `conic_discriminant(A,B,C)` = $C^2-4AB$ | type |
| **Discriminant $\Delta_2,\Delta_3$** | `conic_discriminant2`, `conic_discriminant3` — pure GA, via `conic_center_meet` / meet of 3 dual lines | $\Delta_2=AB-\tfrac{C^2}4=-\Delta/4$; $\Delta_3=\det M_3$ |
| **Degenerate?** | `conic_is_degenerate` = $\Delta_3\approx0$ (pure GA, via `conic_discriminant3`) | line pair / point |

The center of a parabola is at infinity (the pole of $L_\infty$ has $w_c=0$) — an ideal
point in the axis direction, lying on the parabola.

---

## 9a. Right complement, norm, discriminants via meet (paper §6.1, §7)

- **Right complement** $A^c$ (`right_complement`, `ccga/operations.py`): the unique
  linear map with $E\wedge E^c=I$ for every blade $E$ of the canonical null basis
  $\overline{\mathbb{G}}=(e_{o_1},e_{o_2},e_{o_3},e_1,e_2,e_{\infty_1},e_{\infty_2},e_{\infty_3})$
  (combinatorial complement: drop $E$'s factors from $I$, with the sign that makes
  $E\wedge E^c=+I$), extended by linearity. Grade-complementary ($k\to8-k$);
  $(A^c)^c=-A$.
- **Universal norm identity**: $A\wedge A^c=\big(\sum_E A_E^2\big)I$ for every
  multivector $A$ ($A_E$ = coordinates in $\overline{\mathbb{G}}$). Hence
  $\lVert A\rVert^2=(A\wedge A^c)^\star=\sum_E A_E^2\ge0$ (`norm2`, `norm`).
  For a 1-PSE point $p=a\,e_o+b\,e_1+c\,e_2+d\,e_{\infty_1}+e\,e_{\infty_2}+f\,e_{\infty_3}$,
  $\lVert p\rVert^2=2a^2+b^2+c^2+d^2+e^2+f^2$ (the $2a^2$ from $e_o=e_{o_1}+e_{o_2}$).
  For a canonical IPNS conic, $\lVert s\rVert^2=4A^2+4B^2+C^2+D^2+E^2+\tfrac{F^2}2$.
- **Orthogonality**: `orthogonal(A,B)` $\iff A\wedge B^c=0\iff\langle A,B\rangle_{\overline{\mathbb{G}}}=0$
  (polarization of the norm form). This is **not** the classical
  perpendicular-circles relation ($d^2=r_1^2+r_2^2$); instead it is the
  orthogonality of the $C_r,C_g,C_b$ basis a pencil decomposes into (§9b).
- **Discriminants/center via meet of three dual lines** (paper §7): for IPNS
  coefficients $(A,B,C,D,E,F)$,
  $$l_1=\mathrm{dual}\big(Ae_1+\tfrac C2e_2-\tfrac D2 e_\infty\big),\quad
    l_2=\mathrm{dual}\big(\tfrac C2e_1+Be_2-\tfrac E2 e_\infty\big),\quad
    l_3=\mathrm{dual}\big(\tfrac D2e_1+\tfrac E2e_2-Fe_\infty\big)$$
  (`_conic_lines`, all grade 7). Then
  $$l_1\vee l_2=-\tfrac12\big(\Delta_2\,e_o+x_c\,e_1+y_c\,e_2\big)\wedge I_o^{\triangleright}\wedge I_\infty
    \qquad(\texttt{conic\_center\_meet}\to(\Delta_2,x_c,y_c)),$$
  $$l_1\vee l_2\vee l_3=-\tfrac12\Delta_3\;I_o^{\triangleright}\wedge I_\infty
    \qquad(\texttt{conic\_discriminant3}\to\Delta_3=\det M_3).$$
  Center $=(x_c/\Delta_2,\,y_c/\Delta_2)$, exactly matching `conic_center`. Both
  `conic_is_degenerate` and `conic_center_point` now use `conic_discriminant3` /
  `conic_center_meet` — **no `[num]` linear-algebra fallback remains** for these.

---

## 9b. Pencil calculus (paper §8, Theorems 8–11; `tests/test_pencil_calculus.py`)

A **pencil** is a vector space of conics; an $n$-pencil has grade $8-n$ and is the
anti-outer product ($\vee$) of $n$ conics. The right complement turns pencils and
PSE into each other and lets pencils be built/constrained algebraically:

| theorem | operation | effect |
|---|---|---|
| Th 8 (add a conic) | $P\vee C$ | pencil of order $k{+}1$ generated by $P$'s conics and $C$ |
| Th 9 (add a point) | $P\wedge p$ | sub-pencil of $P$'s conics passing through $p$ |
| Th 10 (key identity) | $(A\vee B)\wedge B^c\equiv A$ | for orthogonal blades $A,B$ |
| Th 11 (remove a conic) | $P\wedge C^c$ | conics of $P$ orthogonal to $C$ |

**Worked example** ("pencil of three conics", paper §8): from 6 points $p_1,\dots,p_6$,
$$P=p_1\wedge p_2\wedge p_3\wedge I_o^{\triangleright}\ \ (\text{3-pencil}),\qquad
  C_r=P\wedge p_4\wedge p_5,\qquad
  C_g=P\wedge C_r^{c}\wedge p_6,\qquad
  C_b=P\wedge C_r^{c}\wedge C_g^{c}.$$
$C_r,C_g,C_b$ are pairwise orthogonal (`orthogonal`, Def 12) and $P\equiv C_r\vee
C_g\vee C_b$. Th 8–11 all verified on this example: $P\wedge p_4$ is a 2-pencil whose
conics (including $C_r$) pass through $p_4$ (Th 9); $(C_r\vee C_g)\wedge C_g^{c}\equiv
C_r$ (Th 10); $P\wedge C_b^{c}\equiv C_r\vee C_g$ (Th 11); and meeting $P$ with a
fifth conic $C\notin P$ gives a grade-4 (order-4) pencil containing $C_r,C_g,C_b,C$
(Th 8).

**Th 12 (remove a point), open**: $P\vee p^{c}\equiv P'$ where $P=P'\wedge p$ is *not*
verified — concretely
$(p_1\wedge p_2\wedge p_3\wedge I_o^{\triangleright}\wedge p_4)\vee p_5^{c}\not\equiv
p_1\wedge p_2\wedge p_3\wedge I_o^{\triangleright}\wedge p_4$. A grade-counting
argument shows this can never work: $p\in\mathbb{NO}(P)$ forces $gr(P')\ge2$, so
$gr(P'\wedge p^c)=gr(P')+7>8$ is *trivially* (dimensionally) zero — never because
$P'$ and $p$ have disjoint null-basis support, which is the genuine Def-12
orthogonality that Th 10's proof actually relies on.

---

## 10. Operations & intersection

- **Dual / undual**: `dual(O)=O*I⁻¹`, `undual(O)=O*I`. **Join**: `A ^ B`. **Meet**:
  `meet(A,B) = A & B`.
- **Conic ∨ conic** = grade-6 object $I_4 = C_1\vee C_2 \propto p_1\wedge p_2\wedge p_3\wedge p_4\wedge I_o^{\triangleright} = Q\wedge I_o^{\triangleright}$ (`conic_intersection`); incidence $q\wedge I_4=0$.
- **Recover the quadpole**: $Q=(e_{\infty_3}\wedge\bar e_\infty)\,\lrcorner\,I_4$ (`intersection_quadpole`).
- **Intersection points / reality**: `intersection_points` **[num]**,
  `intersection_reality` **[num]** → `{real, imaginary, ideal}` summing to 4 (Bézout).
  Two circles meet in 2 finite + 2 ideal (the circular points).
- **Pencil** of conics through the 4 points of a quadpole $Q$:
  $\{(I_o^{\triangleright}\wedge Q\wedge p_5)\cdot I^{-1}\}$ for varying $p_5$ — `pencil(Q)` **[num]** (rank-2 span).
- **n-pole extraction**: `circumcircle(T)` **[num]** and `tripole_circumconic(T)`$=T\wedge I_o^{\triangleright}\wedge I_\infty^{\triangleright}$
  (circle through 3 round points; a **line** if one is ideal); `extract_tripole`
  **[num]** (Cardano), `extract_quadpole` **[num]** (Ferrari).

---

## 11. Transformations — versors (`apply_versor(V,X) = V X ~V`)

Versors act uniformly on every object (points, pairs, conics) by the sandwich and
preserve incidence. Generators found from $\delta p=[G,p]$.

| Transformation | versor (explicit) | scope |
|---|---|---|
| **Translation** $(t_x,t_y)$ | `translator(tx,ty)` $=T_x(t_x)\,T_y(t_y)$, $T_x(\tau)=(1-\tfrac{\tau}2 e_1\wedge e_{\infty_1})(1-\tfrac{\tau}2 e_2\wedge e_{\infty_3})$, $T_y(\tau)=(1-\tfrac{\tau}2 e_2\wedge e_{\infty_2})(1-\tfrac{\tau}2 e_1\wedge e_{\infty_3})$ | all conics |
| **Rotation** $\alpha$ | `rotor(alpha)` $=e^{\alpha E}e^{\alpha K}$, $\ e^{\alpha E}=\cos\tfrac\alpha2-\sin\tfrac\alpha2\,e_{12}$, $\ e^{\alpha K}=1+\tfrac{\sin2\alpha}2 K+\tfrac{1-\cos2\alpha}4 K^2$, $\ E=-\tfrac12 e_{12}$, $K=\bar e_o\wedge e_{\infty_3}-e_{o3}\wedge\bar e_\infty$ ($K^3=-4K$) | all conics |
| **Scaling** $s$ | `dilator(s)` $=\prod_{i=1}^3(\cosh u+\sinh u\,e_{o_i}\wedge e_{\infty_i})$, $u=\tfrac12\ln s$ | all conics |
| **Reflection** across $n_xx{+}n_yy{+}d{=}0$ | `reflector(nx,ny,d)` $=T(c)\,R(\theta)\,V_x\,\tilde R(\theta)\,T(-c)$, $V_x=e_2\wedge e_{o3}\wedge e_{\infty_3}$, $\theta=\operatorname{atan2}(n_y,n_x)+\tfrac\pi2$, $c=-d(n_x,n_y)$ | all conics |
| **About a point** $c$ | `rotor_about(α,cx,cy)`, `dilator_about(s,cx,cy)` $=T(c)\,V\,T(-c)$ | all conics |
| **Inversion** in circle | `inversion(cx,cy,r)`: $\sigma X\tilde\sigma$ with $\sigma=p_c-\tfrac{r^2}2 e_\infty$ | **round family only** ⚠ |
| **Transversion** | `transversion(bx,by)` $=\sigma\,T(b)\,\sigma$, $\sigma=$`inversion(0,0,1)` | **round family only** ⚠ |
| **Shear** | — | **not a versor** (non-conformal) ⚠ |

The corrected rotor: the old $\cos\alpha+\sin\alpha(\dots)$ conformal factors are
wrong (null bivectors); the $2\alpha$ is the Veronese symmetric-square action.
Inversion/transversion are Möbius maps — they preserve circles but send a general
conic to a quartic, so they are versors only on the CGA round sub-family (§7), not on
general conics. Shear is not an orthogonal map of $\mathbb{R}^{5,3}$, so no versor exists.

---

## 12. Key structural facts

- **IPNS ↔ OPNS / where the shape lives.** Dualizing toggles grade 1 ↔ grade 7 of the
  *same* object; type is dual-invariant.

  | object | grade-1 IPNS | dual is a point? |
  |---|---|---|
  | general conic | shape on $e_{o1},e_{o2},e_{o3}$ (has $e_{o3}$ when tilted) | no — conic *covector* |
  | circle | $\text{center}-\tfrac{r^2}2 e_\infty$ | **yes** — a round point |
  | line | $e_1,e_2,e_\infty$ only, **no $e_o$** | no — direction + offset |
  | point | zero-radius circle ($A{=}B$,$C{=}0$,$r{=}0$, null) | self-dual |

  A point can never carry $e_{o3}$/$\bar e_o$; those shape components are exactly what
  makes a tilted/eccentric conic *not* a point.

- **$I_\infty^{\triangleright}$ = the circular points.** With $I=v_\infty(1,i)$,
  $J=v_\infty(1,-i)$ (both null): $I\wedge J = -i\,I_\infty^{\triangleright}$. Every
  circle passes through $I,J$ (`I·s = 0` for any center/radius); a general ellipse does
  not. Hence "wedge $I_\infty^{\triangleright}$ = pass through the circular points = be
  round," and a circle = the conic through 3 points **plus** $I,J$.

- **Reality is scale-invariant** ($\mathrm{sign}\,(\lambda p)^2 = \mathrm{sign}\,p^2$);
  negation does not flip real↔imaginary. Read it from $\mathrm{sign}(p^2)$, never from
  the raw $e_\infty$-coefficient sign. The invariant radius is $r^2=p^2/(p\!\cdot e_\infty)^2$.

- **Ideal points have no radius.** At infinity $p\!\cdot e_\infty=0$, so $r^2$ is
  undefined and $p^2$ is blind to any radius term — "round/imaginary ideal point" is
  not a meaningful object (the radius on `make_ideal_point` only shifts a dual line).

---

*Tests: `tests/test_{objects,cga,point,meet_join,conic_construction,conic_intersection,
conic_properties,conic_tangents,conic_normals_projection,transform,
inversion_transversion,line_pair,tripole_round,flat_point,circular_points}.py`.*
