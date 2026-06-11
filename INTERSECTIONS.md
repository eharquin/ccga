# CCGA object intersection — a complete analysis

How CCGA objects meet, grade by grade: the **meet** (regressive `&`), the
**grade-reduction chain** that strips an intersection blade down to its points, the
**shared-factor principle** that explains Bézout reductions (why line ∩ circle ≤ 2), and a
**full interaction table** over the object zoo. Every claim is reproducible with the
existing `ccga` functions — the file adds no new API, only the theory and its verification.

Companion to `OBJECTS.md` (object taxonomy) and `GENERAL_FORM.md` (closed forms). The meet
machinery used here lives in `ccga/extract.py`; the special blades in `ccga/algebra.py`.

```python
from ccga.algebra import Iinfd, Iinf, Iod, einf, einf3, einfbar
from ccga.point import point, point_at_infinity
from ccga.objects import (make_circle, make_line_2points, make_ellipse,
                          make_hyperbola, make_parabola)
from ccga.operations import grades, is_zero
from ccga.extract import (conic_intersection, intersection_quadpole,
                          extract_quadpole, intersection_points, intersection_reality)
```

---

## 0. Conventions

- **Join** `A ^ B` (outer product): the span; grades **add**. This is the construction
  ladder — points wedge up into bigger objects.
- **Meet** `A & B` (regressive product, the `∨` of `ccga.operations.meet`): the
  intersection; for blades in general position grades **subtract against the pseudoscalar**.
- A grade-1 CCGA point is `p = e_o + x e_1 + y e_2 + (x²/2)e_{∞1} + (y²/2)e_{∞2} + xy e_{∞3}`
  (`ccga.point.point`); it lives in `V₆ = span{e_o, e_1, e_2, e_{∞1}, e_{∞2}, e_{∞3}}`.
- The two **gauge bivectors** that govern everything below:

  $$I_o^{\triangleright} = \bar e_o \wedge e_{o_3}\;(=\texttt{Iod}),\qquad
    I_\infty^{\triangleright} = (e_{\infty_1}-e_{\infty_2})\wedge e_{\infty_3}\;(=\texttt{Iinfd}).$$

- The **circular points** I, J — the two complex points
  `point_at_infinity(1, ±i)` where every circle meets the line at infinity — have
  $I\wedge J \propto \texttt{Iinfd}$. This single fact drives §3–§5.

---

## 1. Meet grade arithmetic

For two blades meeting in general position in $\mathbb{R}^{5,3}$ (pseudoscalar grade 8):

$$\operatorname{grade}(A \vee B) = \operatorname{grade}(A) + \operatorname{grade}(B) - 8.$$

A meet is **geometrically non-trivial only when the grades sum to ≥ 8**; below that the
regressive product collapses to a scalar (grade 0) — which is exactly the **incidence
test**. The CCGA conic is grade-7 OPNS, so the meaningful conic meets are:

| `A ∨ B`                         | grades | result | meaning |
|---|---|---|---|
| conic ∨ conic                   | 7+7−8 = **6** | grade-6 intersection blade | the 4 Bézout points |
| conic ∨ conic-at-∞ (grade 5)    | 7+5−8 = **4** | quadpole | 4 ideal incidences |
| conic ∨ quadpole (grade 4)      | 7+4−8 = **3** | tripole | — |
| conic ∨ flat point (grade 4)    | 7+4−8 = **3** | tripole | — |
| conic ∨ line-at-∞ `Iinf` (3)    | 7+3−8 = **2** | dipole at ∞ | the conic's 2 ideal points |
| conic ∨ dipole (grade 2)        | 7+2−8 = **1** | point | — |
| conic ∨ point (grade 1)         | 7+1−8 = **0** | scalar | **incidence** `q ∨ C = 0 ⇔ q ∈ C` |

Verified with `grades(A & B)`; anchors `tests/test_meet_join.py::test_meet_grade_7_7`
(→ `[6]`) and `::test_meet_grade_7_1` (→ `[0]`).

---

## 2. The universal pipeline — conic ∨ conic (always valid)

Two conics meet in **4 points** (Bézout). The regressive product is the grade-6 blade

$$I_6 = C_1 \vee C_2 \;\propto\; p_1\wedge p_2\wedge p_3\wedge p_4 \wedge I_o^{\triangleright}
       \;=\; Q\wedge I_o^{\triangleright},$$

the **quadpole** $Q$ of the 4 intersection points, gauge-fixed by the *same* $I_o^{\triangleright}$
that turns 5 points into a grade-7 conic. Incidence: $q\wedge I_6 = 0 \iff q$ is one of the
four points.

```python
C1, _ = make_circle(0, 0, 2); C2, _ = make_circle(2, 0, 2)
I6 = conic_intersection(C1, C2)          # grades(I6) == [6]
```

**Recover the quadpole by stripping $I_o^{\triangleright}$.** The right contraction blade is
$I_\infty^{\triangleright}$ (`Iinfd`), because it is the reciprocal of $I_o^{\triangleright}$:

$$e_{\infty_3}\wedge\bar e_\infty = -\tfrac12\,I_\infty^{\triangleright},\qquad
  \bar e_o\!\cdot\bar e_\infty = -1,\quad e_{o_3}\!\cdot e_{\infty_3} = -1.$$

So $I_4 = I_6\,\lrcorner\,I_\infty^{\triangleright} \;\propto\; Q$ (grade 4) — this is exactly
`intersection_quadpole(I6)` (which contracts by $e_{\infty_3}\wedge\bar e_\infty$,
identical up to the factor $-2$):

```python
I4 = I6 | Iinfd                          # grades(I4) == [4],  ∝ intersection_quadpole(I6)
pts = extract_quadpole(intersection_quadpole(I6))   # the 4 points by GA pencil + Ferrari
```

The four points are recovered by the GA-native quadpole machinery (pencil → resolvent
cubic → two `±√` dipoles = Ferrari's quartic — see `ccga/extract.py`). Their **reality** is
the Bézout split

```python
intersection_reality(C1, C2)             # {'real': r, 'imaginary': m, 'ideal': k}, r+m+k = 4
```

- **real** — finite real points,
- **imaginary** — finite complex-conjugate pairs,
- **ideal** — points at infinity (shared asymptotic / circular points).

This route works for **every** pair of conics. §3 is a shortcut available only in the
round case.

---

## 3. The round/conformal reduction — the heart of the question

The user's conjecture: keep contracting,

$$I_6 \;\xrightarrow{\;\lrcorner\,I_\infty^{\triangleright}\;}\; I_4
      \;\xrightarrow{\;I_o^{\triangleright}\,\lrcorner\;}\; I_2\ (\text{grade 2}),$$

then read the two points with the **classical dipole** `±√`. This **works** — but only in a
precise regime, and naming that regime is the whole point.

```python
D2 = Iod | (I6 | Iinfd)                  # grade 2 when reducible
# classical dipole extraction:  p± = (D2 ± √(D2²)) / (e∞ · D2)
import math
s = math.sqrt(float((D2*D2).e)); di = (einf | D2).inv()
p_plus, p_minus = (D2 + s)*di, (D2 - s)*di
```

### Why it works, and exactly when

A **circle passes through the circular points I, J**: `make_circle` builds the grade-7 conic
as $p_1\wedge p_2\wedge p_3 \wedge I_\infty^{\triangleright}\wedge I_o^{\triangleright}$, and
that $I_\infty^{\triangleright}$ factor *is* $I\wedge J$. A **line**, as a degenerate conic,
is $\text{line}\cup L_\infty$ — it is built as
$p_1\wedge p_2\wedge I_\infty\wedge I_o^{\triangleright}$ with the **whole** line at infinity

$$I_\infty = I_\infty^{\triangleright}\wedge e_\infty,$$

so it too contains I, J (and every other ideal point). *(Both factorizations verified
proportional to `make_circle` / `make_line_2points` OPNS, and `Iinfd ^ einf == −Iinf`.)*

Therefore, when **both** objects pass through I, J — i.e. **both ∈ {circle, line}** — two of
the four Bézout points are *exactly* the circular points. Then

$$Q = D_2^{\text{finite}}\wedge I_\infty^{\triangleright},\qquad
  I_o^{\triangleright}\,\lrcorner\,Q = D_2^{\text{finite}},$$

a clean CGA point pair, and the `±√` lands the two finite points. Reality comes from the
sign of $D_2^2$:

| pair (radius 2 circles, 2 apart / unit circle ∩ `y=0`)| `D2²` | result |
|---|---|---|
| circle ∩ circle (secant) | `> 0` | 2 real, e.g. `(1, ±√3)` |
| circle ∩ circle (disjoint) | `< 0` | 2 imaginary; reality `{real:0, imag:2, ideal:2}` |
| circle ∩ line (secant) | `> 0` | 2 real, e.g. `(±1, 0)` |

The two “missing” Bézout points are always the **ideal** circular points — `intersection_reality`
shows `{... , 'ideal': 2}` in every row.

### Where it breaks — and why (the correction to the conjecture)

`Iod | I4` returns a grade-2 object **even when the reduction is illegitimate**, with the
**wrong** content. The shortcut is *not* a general 2-point detector:

| pair | true finite points | `Iod\|I4` dipole-landing | verdict |
|---|---|---|---|
| ellipse(3,2) ∩ line `y=0` | `(±3, 0)` | `(±2.353, 0)` | **wrong points** ✗ |
| hyperbola(1,1) ∩ line `y=0` | `(±1, 0)` | `D2² = 0` (degenerate) | **misfires** ✗ |
| ellipse(3,2) ∩ hyperbola(1,1) | 4 real points | `D2² < 0` ("imaginary") | **misfires** ✗ |

The reason: for a non-circular conic, its intersection with $L_\infty$ is its **own** ideal
pair — imaginary for an ellipse, a real asymptotic pair for a hyperbola — **not** the
universal circular points. $I_\infty^{\triangleright}$ and $I_o^{\triangleright}$ strip only
the circular-point direction, so $I_o^{\triangleright}\lrcorner I_4$ leaves a residue that
*looks* like a dipole but encodes the wrong (or degenerate) pair.

> **Validity rule.** The dipole shortcut `D2 = Iod | (I6 | Iinfd)` recovers the true finite
> intersection **iff both objects contain the circular points**, i.e. **both ∈ {circle,
> line}**. For every other pair, use the universal grade-4 quadpole route of §2 (or the
> coefficient resultant `intersection_points`), never the `±√` shortcut.

---

## 4. The shared-factor / Bézout-reduction principle

The mechanism behind §3 is general:

$$\#\,\text{finite intersections} \;=\; 4 \;-\; \#\,\text{forced common ideal points}.$$

Each object carries an **infinity signature** — its intersection with $L_\infty$ — fixed by a
blade in its construction. Two objects sharing ideal points spend Bézout budget there,
leaving fewer finite intersections:

| object | meets $L_\infty$ in | construction blade | reality of ideal pts |
|---|---|---|---|
| circle | the 2 **circular points** I, J | `Iinfd` (= $I\wedge J$) | imaginary (always shared by all circles) |
| ellipse | 2 ideal points | weighted `Iinfd`, `make_ellipse_3points` | imaginary |
| hyperbola | 2 **asymptotic** ideal points | `make_hyperbola_3points` | real |
| parabola | 1 **double** ideal point | double-contact, `make_parabola_3points` | real (tangent to $L_\infty$) |
| line (degenerate conic) | **all** of $L_\infty$ | `Iinf` = `Iinfd ^ einf` | contains every ideal point |

Read off consequences directly:

- **two circles** share I, J ⇒ 2 finite (the radical-axis dipole) — §3.
- **circle ∩ line** share I, J (line ⊃ $L_\infty$ ⊃ {I,J}) ⇒ 2 finite — §3.
- **circle ∩ ellipse** share *nothing* fixed (the ellipse misses I, J) ⇒ 4 finite — §2.
- **two lines** share *all* of $L_\infty$ ⇒ they overlap on a whole component; Bézout is
  infinite and `intersection_reality` correctly refuses. The genuine meet is a **single
  finite point** (§5).
- **co-asymptotic hyperbolas** share a real ideal point ⇒ 3 finite + 1 ideal
  (`tests/test_conic_intersection.py::test_intersection_reality_with_ideal_point`).

`asymptotic_directions` (in `ccga/classify.py`) returns exactly each conic's real ideal
points, making the signature inspectable.

---

## 5. Conic-family interaction table

`make_*` conics (grade-7 OPNS). “Route” = how to land the finite points: **§2** = grade-4
quadpole + `extract_quadpole`; **§3** = dipole `±√`; **incidence** = `q ∧ (A&B) = 0`. All
reality columns verified by `intersection_reality` (sum = 4), points by `intersection_points`.

| `A ∨ B` | meet grade | forced common ideal | finite pts | route | reality (example) |
|---|---|---|---|---|---|
| circle ∩ circle (secant)   | 6 | I, J (2) | 2 | §3 | `{2, 0, 2}` |
| circle ∩ circle (disjoint) | 6 | I, J (2) | 2 (imag) | §3 | `{0, 2, 2}` |
| circle ∩ line              | 6 | I, J (2) | 2 | §3 | `{2, 0, 2}` |
| circle ∩ ellipse           | 6 | none | 4 | §2 | `{0–4, …, 0}` |
| ellipse ∩ ellipse          | 6 | none | 4 | §2 | `{4, 0, 0}` |
| ellipse ∩ hyperbola        | 6 | none | 4 | §2 | `{4, 0, 0}` |
| ellipse ∩ line             | 6 | the ellipse's 2 ideal (imag) | 2 | §2 | `{2, 0, 2}` |
| hyperbola ∩ line           | 6 | the hyperbola's 2 ideal (real) | 2 | §2 | `{2, 0, 2}` |
| parabola ∩ line            | 6 | the parabola's double ideal | 1 | §2 | `{1, 0, 3}` |
| line ∩ line                | 6 | all of $L_\infty$ | 1 | **incidence** | n/a (shared component) |

Note the **line ∩ non-circular-conic** rows: the meet is still grade 6 and still leaves 2
(or 1) finite points, but the two “missing” points are the **conic's own** ideal pair, not I,
J — so they take the §2 quadpole route, not the §3 shortcut (cf. the ellipse∩line failure in
§3). `line ∩ line` is the lone case where the objects share an entire curve ($L_\infty$);
its single finite point is read off by incidence (anchor
`tests/test_meet_join.py::test_line_meet_line_gives_point`).

---

## 6. Full object-zoo meet/join interaction table

Objects and their OPNS grades: point (1), dipole (2), tripole (3), quadpole (4), pentapole
(5), conic (7), line-at-∞ `Iinf` (3), conic-at-∞ `Iod∧Iinf` (5), flat point (4), ideal point
(1), pseudoscalar (8).

**Join `A ^ B` (grades add)** — the construction ladder. Points wedge into
dipole → tripole → quadpole → pentapole; `^ Iod` promotes a pentapole to the grade-7 conic
(`make_conic_*pole`, `make_conic_opns`). Defined when the blades share no subspace; e.g.
`grades(point ^ point) == [2]`, `grades(Iod ^ p1^…^p5) == [7]`.

**Meet `A & B` (regressive)** — non-trivial only when grades sum ≥ 8. Grades of `A & B`
(`0` = scalar incidence, blank = vanishes):

| ∨ | point(1) | dipole(2) | tripole(3) | quadpole(4) | conic(7) | Iinf(3) | C∞(5) | flat(4) | ideal(1) |
|---|---|---|---|---|---|---|---|---|---|
| **conic(7)** | 0 | 1 | 2 | 3 | **6** | 2 | 4 | 3 | 0 |
| **C∞(5)**    | – | – | 0 | 1 | 4 | – | – | – | – |
| **Iinf(3)**  | – | – | – | – | 2 | – | – | – | – |

Reading the conic row (the geometrically rich one):

- conic ∨ point / ideal-point → **scalar**: incidence `q ∨ C = 0 ⇔ q ∈ C`.
- conic ∨ dipole → **point** (grade 1).
- conic ∨ tripole → grade 3; conic ∨ quadpole → grade 3 → finite points by §2 machinery.
- conic ∨ conic → **grade 6** (§2/§3, the 4 Bézout points).
- conic ∨ `Iinf` → **grade 2**: the conic's two **ideal points** (its asymptotic dipole);
  the same pair `asymptotic_directions` returns.
- conic ∨ conic-at-∞ → grade 4: the conic's incidence with the Veronese cone at infinity.

The all-`–` cells are meets whose grades sum < 8: in CCGA two low-grade point-objects do not
“intersect” via `&` (you **join** them instead). This asymmetry — **join builds, meet
intersects, and meet needs grade-sum ≥ 8** — is the structural summary of the table.

---

## 7. Reality summary

- **Round objects** (dipole, circle): reality from the square's sign —
  `P² > 0` real, `< 0` imaginary, `= 0` degenerate/tangent (`ccga.classify.reality`). In §3
  the landed dipole's `D2²` carries this directly.
- **General conics & their intersections**: the Bézout split
  `intersection_reality → {real, imaginary, ideal}` (sum 4). Intersection points are **real**
  (finite real), **imaginary** (finite conjugate pair), or **ideal** (at infinity — shared
  asymptotic or circular points).
- A **tangent** contact is the coincident-point limit: two of the Bézout points merge,
  `D2² → 0` for the round case (e.g. tangent circles), or a repeated root of the resultant
  for the general case.

---

## Reproducing every claim

All numbers above come from the existing library — no new code:

```bash
python -m pytest tests/test_conic_intersection.py tests/test_meet_join.py -q
```

and the inline snippets (§2 grade drop and `I6|Iinfd ∝ intersection_quadpole`; §3 the exact
points for circle∩circle / circle∩line **and** the three documented failures; §5 every
`intersection_reality` summing to 4; §6 each grade via `grades(...)`). The `Iod | I4`
shortcut is presented as analysis, deliberately **not** added as an API — its validity is
confined to the round subfamily of §3.
