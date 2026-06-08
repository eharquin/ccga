# CCGA / QC2GA cheat sheet — geometric-algebra formulas

A formula-only quick reference for **Conic Conformal GA** ($\mathbb{R}^{5,3}$, the
QC2GA/GAC conic algebra). Everything is stated as a geometric-algebra expression. Items
the repo does not yet implement in pure GA are tagged:

- ✅ **ported** — pure-GA formula, implemented and tested in `ccga/`.
- 🔶 **paper-only** — pure-GA formula exists (paper / verified here) but **not yet wrapped**
  as a repo function; see §8.
- ⛔ **open** — no pure-GA formula yet; only a coordinate / linear-algebra solve (`[num]`).

Companion docs: `CLAUDE.md` (setup), `OBJECTS.md` (catalog), `GENERAL_FORM.md` (closed
forms). Paper: Chomicki–Breuils–Biri–Nozick, *Conics, their pencils and intersections in
Geometric Algebra*.

---

## 1. Algebra setup ✅

**Signature** $\mathbb{R}^{5,3,0}$ (non-degenerate ⇒ dual, meet, join all clean).
Diagonal basis $e_1,e_2\,(+)$, $e_{+1},e_{+2},e_{+3}\,(+)$, $e_{-1},e_{-2},e_{-3}\,(-)$.

**Null basis** $\;e_{o_i}=e_{+i}+e_{-i},\quad e_{\infty_i}=\tfrac12(e_{-i}-e_{+i}),\quad i=1,2,3.$
$$e_{o_i}^2=e_{\infty_i}^2=0,\qquad e_{o_i}\!\cdot e_{\infty_i}=-1,\qquad e_1\!\cdot e_1=e_2\!\cdot e_2=1\ \text{(all other Gram entries }0).$$

**Special blades** (`ccga/algebra.py`):

| Symbol | Definition | Grade | key relation |
|---|---|---|---|
| $e_o$ | $e_{o_1}+e_{o_2}$ | 1 | $e_o\!\cdot e_\infty=-1$ |
| $e_\infty$ | $\tfrac12(e_{\infty_1}+e_{\infty_2})$ | 1 | |
| $\bar e_o$ | $e_{o_1}-e_{o_2}$ | 1 | $\bar e_o\!\cdot\bar e_\infty=-1$ |
| $\bar e_\infty$ | $\tfrac12(e_{\infty_1}-e_{\infty_2})$ | 1 | $e_o,e_\infty\perp\bar e_o,\bar e_\infty$ |
| $I_o^{\triangleright}$ | $\bar e_o\wedge e_{o_3}=(e_{o_1}-e_{o_2})\wedge e_{o_3}$ | 2 | origin gauge |
| $I_\infty^{\triangleright}$ | $(e_{\infty_1}-e_{\infty_2})\wedge e_{\infty_3}$ | 2 | infinity gauge = circular-point pair |
| $I_o$ | $e_{o_1}\wedge e_{o_2}\wedge e_{o_3}$ | 3 | |
| $I_\infty$ | $e_{\infty_1}\wedge e_{\infty_2}\wedge e_{\infty_3}$ | 3 | line/conic at infinity |
| $I_\epsilon$ | $e_1\wedge e_2$ | 2 | Euclidean pseudoscalar |
| $I$ | $I_\epsilon\wedge I_\infty\wedge I_o$ | 8 | $I^2=-1,\ I^{-1}=-I$ |

**Products / duality**

| op | formula | code |
|---|---|---|
| dual (IPNS) | $A^{*}=A\,I^{-1}$ | `dual(A)` |
| undual (OPNS) | $A\,I$ | `undual(A)` |
| join (span) | $A\wedge B$ | `A ^ B` |
| meet (intersect) | $A\vee B=(A^{*}\wedge B^{*})^{*}$ | `meet(A,B)` = `A & B` |
| right complement 🔶 | $A^{c}$ s.t. $A\wedge A^{c}=I$ (paper §6) | — (see §8) |

> **Meet sign.** `A & B` agrees with $(A^{*}\wedge B^{*})^{*}$ up to a consistent overall
> sign (harmless for incidence and for the discriminant magnitudes in §5).

---

## 2. Points & point-space elements (PSE) ✅

**Embedding** (degree-2 Veronese of $(x,y)$):
$$p \;=\; e_o + x\,e_1 + y\,e_2 + \tfrac{x^2}{2}e_{\infty_1} + \tfrac{y^2}{2}e_{\infty_2} + xy\,e_{\infty_3}\ \big(\pm\tfrac{r^2}{2}e_\infty\big).$$

| property | formula |
|---|---|
| null / reality | $p^2 = \pm r^2$ ($r{=}0\Rightarrow p^2{=}0$; $>0$ real, $<0$ imaginary circle) |
| normalization | $p\cdot e_\infty = -1$ (finite points); normalize via $p\mapsto p/(-p\!\cdot e_\infty)$ |
| distance | $p\cdot q = -\tfrac12\big[(x{-}x')^2+(y{-}y')^2\big]$ |

**True ideal point** (controls asymptotes / conic type) — pure degree-2:
$$v_\infty(v)=\lim_{t\to\infty}\frac{p(t v)}{t^2}=\tfrac{v_x^2}{2}e_{\infty_1}+\tfrac{v_y^2}{2}e_{\infty_2}+v_xv_y\,e_{\infty_3}\quad(\texttt{point\_at\_infinity}).$$
Paper Def. 2 form (cleared denominators, same projective object): $v_x^2 e_{\infty_1}+v_y^2 e_{\infty_2}+2v_xv_y e_{\infty_3}$.
**Tangent at ∞** (parabola double-contact): $v_\infty'(v)=-cs\,e_{\infty_1}+cs\,e_{\infty_2}+(c^2{-}s^2)e_{\infty_3}$, $(c,s)=v/\lVert v\rVert$.

> ⚠ `make_ideal_point(x,y)` (drop $e_o$, keep $x e_1+y e_2$) is **not** $v_\infty$: it is a
> degree-1⊕2 hybrid that reads as a *line* and does **not** control type. Only the
> pure-quadratic $v_\infty$ lies on the conic-at-infinity. (Veronese has 3 strata: deg 0
> $e_o$, deg 1 $e_1,e_2$, deg 2 $e_\infty$; the true ideal point keeps only deg 2.)

**PSE** (paper §5): basis $\mathcal B_p=(e_o,e_1,e_2,e_{\infty_1},e_{\infty_2},e_{\infty_3})$.
A $k$-PSE is a $k$-blade over $\mathcal B_p$.
- **real** $k$-PSE = wedge of $k$ points; **imaginary** = not factorable as point $\wedge$ (k−1)-PSE.
- Points at infinity are the special imaginary 1-PSE *with* geometric meaning (direction).
- The sum of two points is always an imaginary 1-PSE (line ∩ Veronese ≤ 2 pts).

---

## 3. Conics ✅

**IPNS — grade-1 vector $\Leftrightarrow$ general conic** $Ax^2+By^2+Cxy+Dx+Ey+F=0$:
$$\boxed{\,s = -2A\,e_{o_1}-2B\,e_{o_2}-C\,e_{o_3}+D\,e_1+E\,e_2-\tfrac{F}{2}(e_{\infty_1}+e_{\infty_2})\,}$$
Shape $(A,B,C)$ on the **origin** side; $(D,E)$ on $e_1,e_2$; size $F$ on the **infinity**
side ($-\tfrac F2(e_{\infty_1}+e_{\infty_2})=-F e_\infty$). Inert gauge: $e_{\infty_3},\bar e_\infty$
(canonical form sets them $0$). `make_conic_ipns(A..F)` / `ipns_to_coeffs(s)`.

**OPNS — grade-7 blade from 5 points:**
$$C = I_o^{\triangleright}\wedge p_1\wedge p_2\wedge p_3\wedge p_4\wedge p_5,\qquad \det(\mathrm P)\,I = C\wedge Q(p).$$
Incidence: $q\wedge C = 0 \;\Leftrightarrow\; q\cdot s = 0$ ($s=C^{*}$, grade 1).

**Wedge ladder** (each point +1 grade; $I_o^{\triangleright}$ only cleans the dual):

| object | OPNS | grade | dual grade |
|---|---|---|---|
| point | $p$ | 1 | 1 (self) |
| dipole | $p_1\wedge p_2$ | 2 | 6 |
| tripole | $p_1\wedge p_2\wedge p_3$ | 3 | 5 |
| quadpole | $p_1\wedge\dots\wedge p_4$ | 4 | 4 |
| pentapole | $p_1\wedge\dots\wedge p_5$ | 5 (already the conic locus) | 3, $=-\tfrac12 s\wedge I_\infty^{\triangleright}$ |
| **conic** | $\;\cdot\wedge I_o^{\triangleright}$ | 7 | 1, $=s$ |

**Named conics** = $s$ with specific $(A,B,C,D,E,F)$:

| conic | $(A,B,C,D,E,F)$ |
|---|---|
| circle $(c_x,c_y,r)$ | $(1,1,0,-2c_x,-2c_y,c_x^2{+}c_y^2{-}r^2)$, i.e. $s=p_c-\tfrac{r^2}2 e_\infty$, $s^2=r^2$ |
| ellipse $(a,b,c_x,c_y)$ | $(\tfrac1{a^2},\tfrac1{b^2},0,-\tfrac{2c_x}{a^2},-\tfrac{2c_y}{b^2},\tfrac{c_x^2}{a^2}{+}\tfrac{c_y^2}{b^2}{-}1)$ |
| hyperbola $(a,b,c_x,c_y)$ | $(\tfrac1{a^2},-\tfrac1{b^2},0,-2c_xA,-2c_yB,\tfrac{c_x^2}{a^2}{-}\tfrac{c_y^2}{b^2}{-}1)$ |
| parabola $y^2{=}4px$ | $(0,1,0,-4p,0,0)$ |
| tilted ellipse $(a,b,\theta,c_x,c_y)$ | $A{=}\tfrac{\cos^2\theta}{a^2}{+}\tfrac{\sin^2\theta}{b^2}$, $B{=}\tfrac{\sin^2\theta}{a^2}{+}\tfrac{\cos^2\theta}{b^2}$, $C{=}2\sin\theta\cos\theta(\tfrac1{a^2}{-}\tfrac1{b^2})$, $D{=}-2(Ac_x{+}\tfrac C2 c_y)$, $E{=}-2(Bc_y{+}\tfrac C2 c_x)$, $F{=}Ac_x^2{+}Bc_y^2{+}Cc_xc_y{-}1$ |
| line $n_x x{+}n_y y{+}d{=}0$ | $(0,0,0,n_x,n_y,d)$, i.e. $s=n_x e_1{+}n_y e_2-\tfrac d2(e_{\infty_1}{+}e_{\infty_2})$ |
| line pair $\ell_1\ell_2$ | $A{=}a_1a_2,B{=}b_1b_2,C{=}a_1b_2{+}a_2b_1,D{=}a_1c_2{+}a_2c_1,E{=}b_1c_2{+}b_2c_1,F{=}c_1c_2$ |

**Constructive type** (3 points + an ideal-point pair $B$, a line in the plane at ∞):
$$C=p_1\wedge p_2\wedge p_3\wedge B\wedge I_o^{\triangleright}.$$

| $B$ | $B$'s line vs ∞-conic | type |
|---|---|---|
| 2 real dirs $v_\infty(v_1)\wedge v_\infty(v_2)$ | secant | hyperbola (dirs = asymptotes) |
| 1 double dir $v_\infty\wedge v_\infty'$ | tangent | parabola |
| 2 imaginary dirs | non-secant | ellipse |
| circular points $I_\infty^{\triangleright}$ | non-secant | circle |

---

## 4. Pencils, operations & incidence

**Meet / join / dual** ✅ (§1). **Conic ∨ conic** ✅ → grade-6:
$$I_4 = C_1\vee C_2 \;\propto\; p_1\wedge p_2\wedge p_3\wedge p_4\wedge I_o^{\triangleright}=Q\wedge I_o^{\triangleright},\qquad q\wedge I_4=0\ \text{(incidence)}.$$
Quadpole recovery ✅: $Q=(e_{\infty_3}\wedge\bar e_\infty)\,\lrcorner\,I_4$. Bézout: 4 points
(real / imaginary / ideal); two circles → 2 finite + the 2 circular points.

**Pencil** (paper §6): an $n$-pencil is the vector space $\{\sum\lambda_i C_i\}$ = anti-outer
product $C_1\vee\dots\vee C_n$ = dual of an $n$-blade over $\mathcal B_C=(e_{o_1},e_{o_2},e_{o_3},e_1,e_2,e_\infty)$.
A conic is a 1-pencil; $I_o^{\triangleright}$ is the pencil of **all** conics.
$\;gr(P^{*})=n.$ All conics of a pencil share the same intersection points.

**Complement-based PSE↔pencil calculus** 🔶 (paper Th 8–12, via $A^{c}$):

| operation | formula | effect |
|---|---|---|
| add a conic | $P\vee C$ | pencil generated by $P$ and $C$ (order $+1$) |
| add a point | $P\wedge p$ | sub-pencil of conics of $P$ through $p$ |
| remove a conic | $P\wedge C^{c}$ | conics of $P$ orthogonal to $C$ |
| remove a point | $P\vee p^{c}$ | drop the $p$ constraint |
| key identity | $(A\vee B)\wedge B^{c}\equiv A$ | for orthogonal blades |

**Norm & orthogonality** 🔶 (paper §6): $\;\lVert A\rVert=\sqrt{(A\wedge A^{c})^{*}}=\sqrt{A^{*}\!\cdot A^{c}}=\sqrt{\textstyle\sum_E A_E^2}$;
$\;A\perp B \iff A\wedge B^{c}=0$. (Fixes the defective $\sqrt{\tilde A\cdot A}$, which drops the $c$/$e_{o_3}$ term.)

---

## 5. Properties (read off the multivector)

**Everything below is GA inner products / meets on the IPNS vector $s$; the only
irreducible scalar steps are $\sqrt{\ }$ and $\operatorname{atan2}$ (no eigensolver).**
Verified in kingdon against `np.linalg` (tilted ellipse $a{=}3,b{=}1.5,\theta{=}37°$ and a hyperbola).

**Coefficients as blade inner products** ✅:
$$A=\tfrac12 s\!\cdot e_{\infty_1},\quad B=\tfrac12 s\!\cdot e_{\infty_2},\quad C=s\!\cdot e_{\infty_3},\quad D=s\!\cdot e_1,\quad E=s\!\cdot e_2,\quad F=s\!\cdot e_o.$$
The shape part splits into an **isotropic** scalar and a **deviatoric** spin-2 vector:
$$\underbrace{A+B=s\!\cdot e_\infty}_{\text{trace (size)}},\qquad \underbrace{(A-B,\;C)=(s\!\cdot\bar e_\infty,\;s\!\cdot e_{\infty_3})}_{\text{anisotropy — angle }=2\theta_{\text{axis}}}.$$
The deviatoric pair is a **symmetric-square (Veronese) object**: its angle is $2\theta$, which
is exactly why the §6 rotor acts with $2\alpha$ and why $\bar e_\infty,e_{\infty_3}$ are the
tilt directions.

**Discriminant** ✅ (closed form from the above):
$$\Delta_2 = AB-\tfrac{C^2}{4}=\tfrac14\big[(s\!\cdot e_\infty)^2-(s\!\cdot\bar e_\infty)^2-(s\!\cdot e_{\infty_3})^2\big].$$

**Shape eigenvalues / axes / semi-axes** ✅ (replaces the eigensolver):
$$\lambda_\pm=\tfrac12(s\!\cdot e_\infty)\pm\tfrac12\sqrt{(s\!\cdot\bar e_\infty)^2+(s\!\cdot e_{\infty_3})^2},\qquad \theta_{\text{axis}}=\tfrac12\operatorname{atan2}(s\!\cdot e_{\infty_3},\ s\!\cdot\bar e_\infty).$$
$$F'=p_c\!\cdot s\ \ (\text{conic value at center, GA incidence}),\qquad a_i^2=-\frac{p_c\!\cdot s}{\lambda_i}.$$
**Eccentricity / foci** ✅: $c=\sqrt{|a^2\mp b^2|}$ (ellipse $-$ / hyperbola $+$),
$e=c/a$, foci $=p_c\pm c\,\hat u_{\theta}$.

**Asymptotic / ideal directions** ✅ (incidence with the conic-at-infinity): a true ideal
point lies on the conic iff
$$v_\infty(v)\!\cdot s = 0\ \Longleftrightarrow\ A v_x^2+C v_xv_y+B v_y^2=0$$
(2 real ⇒ hyperbola asymptotes, 1 double ⇒ parabola axis, 2 imaginary ⇒ ellipse) — i.e.
the meet $C\vee(\text{conic at }\infty)$.

**Center & discriminants via meet of three dual lines** 🔶 (paper §7, verified here):
$$\ell_1=(a e_1+\tfrac c2 e_2-\tfrac d2 e_\infty)^{*},\quad \ell_2=(\tfrac c2 e_1+b e_2-\tfrac e2 e_\infty)^{*},\quad \ell_3=(\tfrac d2 e_1+\tfrac e2 e_2-f e_\infty)^{*}$$
$$\ell_1\vee\ell_2 = -\tfrac12\big(\underbrace{\Delta_2}_{w_c}\,e_o + x_c\,e_1 + y_c\,e_2\big)\wedge I_o^{\triangleright}\wedge I_\infty \;\equiv\; p_c\wedge I_o^{\triangleright}\wedge I_\infty,\qquad \ell_1\vee\ell_2\vee\ell_3 = \tfrac12\Delta_3\, I_o^{\triangleright}\wedge I_\infty.$$
with $(a,b,c,d,e,f)=(A,B,C,D,E,F)$. The center is the flat point $\ell_1\vee\ell_2$
(homogeneous $w_c=\Delta_2$, so a **parabola**'s center is at infinity). Here
$\Delta_2=ab-\tfrac{c^2}4$, $\Delta_3=abf+\tfrac{cde-c^2f-bd^2-ae^2}4$ — both recovered as GA
scalars (the meet's overall sign convention can flip $\Delta_3$; only $\Delta_3{=}0$ vs
$\neq0$ matters for type).

**Center (inner-product form)** ✅ `conic_center`: with $s_i=s\!\cdot e_{\infty_i}$, $\sigma_k=s\!\cdot e_k$, $4\Delta_2=s_1s_2-s_3^2$:
$$c_x=\frac{s_3\sigma_2-s_2\sigma_1}{s_1s_2-s_3^2},\qquad c_y=\frac{s_3\sigma_1-s_1\sigma_2}{s_1s_2-s_3^2}.$$

**Reality** ✅: round/conic objects $s^2>0$ real, $<0$ imaginary, $\approx0$ degenerate
($\mathrm{sign}\,s^2$ scale-invariant). Invariant radius $r^2=s^2/(s\!\cdot e_\infty)^2$.

**Type table** (Δ₂, Δ₃):

| $\Delta_2$ | $\Delta_3$ | type |
|---|---|---|
| $+$ / $0$ / $-$ | $\neq0$ | ellipse / parabola / hyperbola |
| $+$ / $0$ / $-$ | $0$ | point / parallel lines / intersecting lines |

---

## 6. Transformations — versors ✅  (`apply_versor(V,X)=V X \tilde V`)

| transform | versor |
|---|---|
| translation $(t_x,t_y)$ | $T=T_x T_y$, $T_x(\tau)=(1-\tfrac\tau2 e_1{\wedge}e_{\infty_1})(1-\tfrac\tau2 e_2{\wedge}e_{\infty_3})$, $T_y(\tau)=(1-\tfrac\tau2 e_2{\wedge}e_{\infty_2})(1-\tfrac\tau2 e_1{\wedge}e_{\infty_3})$ |
| rotation $\alpha$ | $e^{\alpha E}e^{\alpha K}$, $E=-\tfrac12 e_{12}$, $K=\bar e_o{\wedge}e_{\infty_3}-e_{o_3}{\wedge}\bar e_\infty$ ($K^3=-4K$); the $2\alpha$ is the Veronese symmetric-square action |
| scaling $s$ | $\prod_{i=1}^3(\cosh u+\sinh u\,e_{o_i}{\wedge}e_{\infty_i})$, $u=\tfrac12\ln s$ |
| reflection | $T(c)R(\theta)V_x\tilde R(\theta)T(-c)$, $V_x=e_2{\wedge}e_{o_3}{\wedge}e_{\infty_3}$ |
| inversion / transversion | round family only ⚠ (Möbius → quartic on general conics) |

Versors act uniformly on all objects and preserve incidence. Shear is non-conformal → no versor.

---

## 7. QC2GA ↔ GAC (equivalent algebras) ✅

Same metric; isomorphic by a basis change (paper §4.3):
$$\bar n_+=e_o,\quad \bar n_-=\bar e_o,\quad \bar n_\times=e_{o_3},\qquad n_+=e_\infty,\quad n_-=\bar e_\infty,\quad n_\times=e_{\infty_3}.$$
Points and conics coincide; GAC contributes conic-reflection versors, QC2GA the simpler translator.

---

## 8. Open / numeric frontier (List B)

What is **not** yet a pure-GA routine in this repo.

**8a. Paper-GA exists, not yet ported (🔶 → make it ✅):**

| item | GA formula | status |
|---|---|---|
| Right complement $A^{c}$ | $A\wedge A^{c}=I$ (paper §6) | the operator everything below needs |
| Norm / orthogonality | $\lVert A\rVert=\sqrt{(A\wedge A^{c})^{*}}$; $A\wedge B^{c}=0$ | §4 |
| $\Delta_2,\Delta_3$, center | meet of three lines (§5) | **verified reproduces coords**; wrap as functions, drop `[num]` from `conic_discriminant`/`conic_is_degenerate`/`conic_center_point` |
| **shape eigenvalues / axes / semi-axes / eccentricity / foci** | inner-product form (§5): $\lambda_\pm$, $\theta_{\text{axis}}$, $a_i^2=-(p_c\!\cdot s)/\lambda_i$ | **verified = `eigvalsh`**; replaces the `np.linalg.eigh` in `conic_axes`/`_central_conic_geometry` (only $\surd$, $\operatorname{atan2}$ left) |
| asymptotic / ideal directions | $v_\infty\!\cdot s=0$ = $C\vee(\text{conic at }\infty)$ (§5) | **verified**; GA incidence, then a quadratic root |
| Pencil calculus | $P\vee C$, $P\wedge p$, $P\wedge C^{c}$, $P\vee p^{c}$ (Th 8–12) | §4 |
| Degenerate member of a pencil | $\Delta_3(\lambda C_a+\mu C_b)=0$ cubic (paper Alg 1) | GA discriminant + scalar cubic |

**8b. Only numeric today (⛔ — GA form still to find, or inherently scalar):**

| item | current `[num]` solve | note |
|---|---|---|
| line-pair factorization | Hessian adjugate + $\surd$ (`extract._lines_of`) | no pure-GA factorization yet — **open** |
| conic–line intersection | rotor-align then quadratic (paper Alg 3) | rotor GA; root scalar |
| intersection-point extraction | Cardano / Ferrari roots | GA part = pencil reduction; $\surd$ inherently scalar |
| conic from 5 tangents | $\mathrm{adj}(M^{*})$ | matrix adjugate |
| normal feet / projection | `[num]` | |

**Principle.** Objects are always GA multivectors; the irreducible scalar step is
**root extraction** ($\sqrt{\ }$, cubic/quartic, eigenvalues). The goal is to push every
*structural* operation (constraints, discriminants, center, classification, pencil
manipulation) into GA, leaving only that final scalar solve numeric — and to document
exactly where that boundary lies.
