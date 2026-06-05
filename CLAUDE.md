# CCGA — Completing the Object Algebra via Meet & Join (kingdon)

## Goal

Construct and verify **every geometric object** of Conic Conformal Geometric Algebra
(CCGA, the conic algebra in $\mathbb{R}^{5,3}$), in both **OPNS** (outer-product null
space, built by *join*) and **IPNS** (inner-product null space, the *dual*) form, with
their grades, geometric meaning, and reality conditions. Objects must be derived
**systematically** from points using the **join** ($\wedge$) and **meet** ($\vee$,
regressive) products, and every claim must pass **symbolic + numeric** verification in
`kingdon` (SymPy backend for symbolic, floats for numeric).

The deliverable is working code plus a generated `OBJECTS.md` taxonomy table. Do **not**
rederive the established results in §3 from scratch and trust your derivation — treat them
as ground truth and use them as regression anchors. If a computation disagrees with §3,
the computation (or a convention/normalization choice) is wrong until proven otherwise.

---

## 1. The algebra

Signature $\mathbb{R}^{5,3,0}$ (non-degenerate: 5 positive, 3 negative, 0 null axes).
Because it is non-degenerate the pseudoscalar is invertible, so **dual, meet, and join are
all clean** (unlike PGA — do not import PGA degeneracy habits).

**Diagonal (orthogonal) basis** — what you give kingdon:

$$e_1,\,e_2\;(+),\qquad e_{+1},e_{+2},e_{+3}\;(+),\qquad e_{-1},e_{-2},e_{-3}\;(-).$$

`Algebra(5, 3)` (or the signature list `[1,1,1,1,1,-1,-1,-1]`, ordering up to you — fix it
once and document it).

**Null basis (working basis)** — define as multivector combinations, then verify the Gram
matrix below. One valid convention (verify it reproduces the target):

$$e_{o_i} = e_{+i} + e_{-i},\qquad e_{\infty_i} = \tfrac12\,(e_{-i} - e_{+i}),\qquad i=1,2,3.$$

This gives $e_{o_i}^2 = e_{\infty_i}^2 = 0$ and $e_{o_i}\!\cdot e_{\infty_i} = -1$.

**Target Gram matrix** (in null basis, ordered $e_1,e_2,e_{o_1},e_{\infty_1},e_{o_2},e_{\infty_2},e_{o_3},e_{\infty_3}$):
$e_1\!\cdot e_1 = e_2\!\cdot e_2 = 1$; for each $i$, $e_{o_i}\!\cdot e_{\infty_i} = -1$; all
other entries $0$. **First verification step: build the $8\times8$ Gram matrix from your
null combos and assert it equals this.**

**Special blades** (define and verify grades/inner products):

| Symbol | Definition | Grade |
|---|---|---|
| $e_o$ | $e_{o_1} + e_{o_2}$ | 1 |
| $e_\infty$ | $\tfrac12(e_{\infty_1} + e_{\infty_2})$ | 1 |
| $e_{\bar o}$ | $e_{o_1} - e_{o_2}$ | 1 |
| $e_{\bar\infty}$ | $\tfrac12(e_{\infty_1} - e_{\infty_2})$ | 1 |
| $I_o^{\triangleright}$ | $(e_{o_1}-e_{o_2}) \wedge e_{o_3} = e_{\bar o}\wedge e_{o_3}$ | 2 |
| $I_\infty^{\triangleright}$ | $(e_{\infty_1}-e_{\infty_2}) \wedge e_{\infty_3}$ | 2 |
| $I_o$ | $e_{o_1}\wedge e_{o_2}\wedge e_{o_3}$ | 3 |
| $I_\infty$ | $e_{\infty_1}\wedge e_{\infty_2}\wedge e_{\infty_3}$ | 3 |
| $I_\epsilon$ | $e_1 \wedge e_2$ | 2 |
| $I$ (pseudoscalar) | $I_\epsilon \wedge I_\infty \wedge I_o$ | 8 |

Key relations to assert: $e_o\!\cdot e_\infty = -1$, $e_{\bar o}\!\cdot e_{\bar\infty} = -1$,
and $e_o,e_\infty \perp e_{\bar o},e_{\bar\infty}$ (the two Euclidean/conformal pairs are
mutually orthogonal). Compute $I$, $I^{-1}$, and $I^2$ once and cache them.

---

## 2. The point embedding

$$\mathbb{R}^2 \to \mathbb{R}^8:\quad
p = e_o + x\,e_1 + y\,e_2 + \tfrac{x^2}{2}\,e_{\infty_1} + \tfrac{y^2}{2}\,e_{\infty_2} + xy\,e_{\infty_3}$$

(i.e. $e_{o_1}$- and $e_{o_2}$-coefficients are both $1$, $e_{o_3}$-coefficient is $0$).

Properties to assert symbolically:
- **Null:** $p^2 = 0$.
- **Distance:** for $p\leftrightarrow(x,y)$, $q\leftrightarrow(x',y')$, $\;p\cdot q = -\tfrac12\big[(x-x')^2+(y-y')^2\big]$.
- **Normalization:** $p\cdot e_\infty = -1$. Normalize any round object by dividing by $-(\,\cdot\,)\cdot e_\infty$ when a canonical scale is wanted.

---

## 3. Established results (ground-truth anchors)

1. **IPNS grade-1 vector = general conic.** For a vector
   $s = s_{o_1}e_{o_1}+s_{o_2}e_{o_2}+s_{o_3}e_{o_3}+s_{e_1}e_1+s_{e_2}e_2+s_{\infty_i}e_{\infty_i}$,
   the locus $q\cdot s = 0$ is
   $$A x^2 + B y^2 + C xy + D x + E y + F = 0,$$
   $$A=-\tfrac{s_{o_1}}{2},\;B=-\tfrac{s_{o_2}}{2},\;C=-s_{o_3},\;D=s_{e_1},\;E=s_{e_2},\;F=-(s_{\infty_1}+s_{\infty_2}).$$
2. **Shape lives on the origin side; size on the infinity side.** $A,B,C$ come from
   $e_{o_1},e_{o_2},e_{o_3}$; $F$ from $e_{\infty_1}+e_{\infty_2}$.
3. **One isotropic radius.** A circle is $s = c - \tfrac{r^2}{2}\,e_\infty$ with $s^2 = r^2$
   ($r^2>0$ real, $r^2<0$ imaginary) — identical to CGA. There is **not** an independent
   radius per $e_{\infty_i}$.
4. **Gauge-inert directions.** $q\cdot s$ never depends on $s_{\infty_3}$ and depends on
   $s_{\infty_1},s_{\infty_2}$ only through their sum. So $e_{\infty_3}$ and
   $e_{\bar\infty}$ are inert; canonical IPNS conics live in
   $\operatorname{span}\{e_1,e_2,e_{o_1},e_{o_2},e_{o_3},e_\infty\}$.
5. **Points span only $V_6 = \operatorname{span}\{e_o,e_1,e_2,e_{\infty_1},e_{\infty_2},e_{\infty_3}\}$.** The two
   directions points cannot reach are $e_{\bar o}$ and $e_{o_3}$, i.e. $W_2 = \operatorname{span}\{e_{\bar o}, e_{o_3}\}$, whose 2-blade is $I_o^{\triangleright}$.
6. **OPNS conic = grade-7 blade**
   $$C = I_o^{\triangleright} \wedge p_1 \wedge \cdots \wedge p_5.$$
   The bare $\bigwedge_{i=1}^5 p_i$ (grade 5) already fixes the conic (5 points lie in a
   5-D hyperplane of $V_6$); wedging $I_o^{\triangleright}$ supplies the missing
   directions and gauge-fixes so the dual is clean.
7. **Dual conic = grade-1 vector** $C^\star = C\,I^{-1}$ (grade $8-7=1$), landing in
   $\operatorname{span}\{e_1,e_2,e_{o_1},e_{o_2},e_{o_3},e_\infty\}$. Its 6 components are the 6 conic
   coefficients of result 1. (Note: the 6-fold wedge $e_1\wedge e_2\wedge e_{o_1}\wedge e_{o_2}\wedge e_{o_3}\wedge e_\infty$ is the *pseudoscalar of that home subspace*, **not** the conic itself.)
8. **OPNS/IPNS cross-check:** $I_o^{\triangleright}$ (origin side, OPNS) and "$s$ has no
   $e_{\bar\infty}$ / no $e_{\infty_3}$" (infinity side, IPNS) are the same gauge-fixing
   under $\star$. Assert this on examples.
9. **GAC isomorphism** (for sanity / literature alignment): $\bar n_+ = e_o$,
   $n_+ = e_\infty$, $\bar n_- = e_{\bar o}$, $n_- = e_{\bar\infty}$, $\bar n_\times = e_{o_3}$,
   $n_\times = e_{\infty_3}$. Optional regression target.

---

## 4. Operations

- **Join** (span / OPNS construction): outer product `A ^ B`. Defined when blades share no
  common subspace; grade adds. This is how points/objects compose into bigger objects.
- **Meet** (intersection): regressive product `A & B` in kingdon (the $\vee$ product),
  equivalently $(A^\star \wedge B^\star)\,I$ up to orientation. Use it for object
  intersections.
- **Dual:** `A.dual()` or `A * I.inv()`. **Pin down the orientation convention once**
  (left vs right multiply by $I^{\pm1}$, and `.dual()` vs `.undual()`) by requiring it to
  reproduce result 7 on a known conic, then use it consistently everywhere.

---

## 5. Task — the object zoo to complete

Produce a constructor + classifier for every object below. For each: **OPNS form**,
**IPNS form**, **grade(s)**, **geometric meaning**, **reality test**, and **how it is
built** (which join/meet of which primitives). Fill the results into `OBJECTS.md`.

**Round / point objects**
- Point (grade 1) — given.
- Point pair / dipole — `p1 ^ p2` (grade 2). Reality from its square's sign.
- Tangent / degenerate pair (coincident points): the $r\to0$ / null limit.

**Conic objects**
- General conic: OPNS grade-7 `Iod ^ p1 ^ p2 ^ p3 ^ p4 ^ p5`; IPNS grade-1 dual.
- Circle (round sub-family: balanced origin, $e_{\bar o}=0$, $e_{o_3}=0$), with radius via $e_\infty$.
- Ellipse / hyperbola (axis-aligned via $e_{\bar o}$; tilted via $e_{o_3}$).
- Parabola (degenerate intersection with the line at infinity — see ideal elements).
- Line as a **degenerate conic** (no quadratic part); ideal/finite line forms.

**Flats / ideal elements**
- Flat point $P = p \wedge (\text{infinity element})$ — determine the correct infinity
  element(s); infinity here is the 3-D space $\operatorname{span}\{e_{\infty_1},e_{\infty_2},e_{\infty_3}\}$ with top blade $I_\infty$, so there is a family, not one $e_\infty$.
- Ideal point (drop origin part) = asymptotic direction.
- Line at infinity / conic at infinity (built from $I_\infty$, $I_\infty^{\triangleright}$).
- Classify each conic by its **incidence with infinity** (ellipse: no real ideal points;
  parabola: tangent / one; hyperbola: two) and tie this to $e_\infty$ vs $I_\infty^{\triangleright}$ components.

**Intersections (via meet `&`)** — fill an interaction table:
- line ∧/∨ line → point (grade?).
- conic ∨ line → up to 2 points.
- conic ∨ conic → up to 4 points (Bézout) — verify the count and recover the points.
- pencils: linear combinations $\lambda C_1 + \mu C_2$ of IPNS conics; verify they pass
  through the base points.

Also produce the **meet/join interaction table**: rows/cols = object types, cells =
resulting object type + grade, each verified on a concrete example.

---

## 6. Verification protocol

Every constructor ships with tests. Use SymPy-backed `kingdon` for exact checks and floats
for numeric ones.

- **Algebra:** Gram matrix == target (§1); $I^2$, $I\,I^{-1}=1$.
- **Point:** $p^2=0$; distance identity (§2); normalization.
- **IPNS conic:** expand `q . s` for symbolic point `q` and assert it equals
  $Ax^2+By^2+Cxy+Dx+Ey+F$ with the coefficient map of result 1.
- **Grades:** assert each constructed object is purely the claimed grade (chop numeric
  noise first, see gotchas).
- **OPNS↔IPNS round trip:** `(Iod ^ wedge5).dual()` reproduces the expected grade-1 dual
  conic coefficients for a conic fit to 5 known points; and dualizing back returns the
  grade-7 blade (up to scale).
- **Meet:** for two conics through known points, `C1* & C2*` (or the right dual
  combination) recovers exactly the shared/intersection points numerically.
- **GAC cross-check (optional):** apply the §3.9 dictionary and confirm objects match GAC
  forms.

---

## 7. Conventions & gotchas

- **Sandwich product:** use explicit `V * X * ~V`, not `>>` (the `>>` operator has bitten
  this pipeline before).
- **Float noise:** chop with a numeric threshold (e.g. `1e-10`) before classifying grades
  or asserting zero; do exact reasoning in SymPy where feasible.
- **Null normalization:** points are null, so "normalize" means fixing scale via
  $p\cdot e_\infty = -1$, not unit norm.
- **Gauge directions:** when reading off IPNS conics, project out / expect freedom in
  $e_{\bar\infty}$ and $e_{\infty_3}$ (result 4). Canonical form sets them to zero.
- **Dual orientation:** fix the $I^{\pm1}$ / `.dual()` convention against result 7 once,
  then never mix conventions.
- **Reality:** $s^2>0$ real, $s^2<0$ imaginary for round objects; for general conics use
  the standard conic discriminant on $(A,B,C,D,E,F)$, not just $s^2$.
- **Non-simple bivectors:** if you build versors/rotors as a side task, watch for
  non-simple bivector exponentials (split into commuting simple parts).
- **Don't assume CGA's grade-$k\leftrightarrow(n-k)$ duality maps OPNS to IPNS naively** —
  the grade-7/grade-1 pairing here is specific to the $I_o^{\triangleright}$ construction.

---

## 8. Suggested repo layout

```
ccga/
  algebra.py      # Algebra(5,3), null combos, special blades, I and I^{-1}; Gram assert
  point.py        # point embedding, normalization, distance
  objects.py      # constructors: point pair, conic (OPNS grade-7 + IPNS grade-1),
                  # circle/ellipse/parabola/hyperbola/line, ideal elements
  operations.py   # join, meet (regressive), dual helpers (one fixed convention)
  classify.py     # grade + geometric-type classifier, reality test
tests/
  test_algebra.py test_point.py test_objects.py test_meet_join.py
OBJECTS.md        # generated taxonomy table (the human-facing deliverable)
```

---

## 9. Acceptance criteria

- Gram matrix and all §1 special-blade relations verified.
- Point properties (§2) verified symbolically.
- Every object in §5 has OPNS + IPNS constructors (where both exist), correct grade,
  reality test, and passing tests.
- OPNS↔IPNS round trip verified for the general conic and at least one of each sub-type.
- Meet/join interaction table filled and each cell verified on an example, including
  conic ∨ conic = 4 points.
- `OBJECTS.md` generated with one row per object: name, OPNS, IPNS, grade(s), meaning,
  reality test, construction.
- All anchors in §3 reproduced (used as regression tests), and any deviation explained.
