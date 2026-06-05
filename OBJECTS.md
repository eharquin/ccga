# CCGA Object Taxonomy

Geometric objects of Conic Conformal Geometric Algebra (CCGA, $\mathbb{R}^{5,3}$),
as implemented in this repo. See `CLAUDE.md` for the algebra setup (§1), point
embedding (§2), and ground-truth anchors (§3).

This file is **generated from the constructors** — every grade/type/reality below
was read back from the code (`ccga/objects.py`, `ccga/cga.py`, `ccga/classify.py`)
and is covered by the test suite (`tests/`).

## Conventions

- **Dual** (`ccga/operations.py`): `ipns = opns * I_inv` (right-multiply by
  $I^{-1}$); `undual = opns = ipns * I`. Fixed once against §3 result 7.
- **Join** = outer product `A ^ B`; **Meet** = regressive product `A & B`.
- **Null basis** (display order): `eo1 eo2 eo3 e1 e2 einf1 einf2 einf3`. Inspect
  any multivector with `from ccga import print_null` (`multiline=True` for one
  aligned blade per line).
- **Reality** — round/conic: `s² > 0` real, `< 0` imaginary, `≈ 0` degenerate;
  general conics use the discriminant $\Delta = C^2 - 4AB$ on $(A,B,C,D,E,F)$.
- Special blades used below: `eo = eo1+eo2`, `einf = (einf1+einf2)/2`,
  `Iod = (eo1−eo2)∧eo3` (origin gauge), `Iinf = einf1∧einf2∧einf3`,
  `Iinfd = (einf1−einf2)∧einf3` (infinity gauge), `I` = grade-8 pseudoscalar.

---

## 1. CCGA points and conics

The native CCGA family: an IPNS **grade-1** vector is a general conic
$Ax^2+By^2+Cxy+Dx+Ey+F=0$ (§3 result 1); its OPNS dual is the **grade-7** blade
`Iod ∧ p1 ∧ … ∧ p5` (§3 result 6).

| Object | OPNS | IPNS | Grade(s) | Meaning | Reality test | Construction |
|---|---|---|---|---|---|---|
| **Point** | grade 1 | = OPNS (up to gauge) | 1 | point $(x,y)$; optional radius $r$ → round point/circle | $p^2 = \pm r^2$ ( $r{=}0$ → null) | `make_point_ccga(x,y,r=0,imaginary=False)` |
| **Point pair** (dipole) | grade 2 | grade 6 | 2 / 6 | two points | $P^2>0$ real, $<0$ imaginary | `make_point_pair(p1,p2)` = `p1 ^ p2` |
| **Tangent point** | grade 2 (null) | grade 6 | 2 / 6 | coincident-point limit | $P^2 = 0$ | `make_tangent_point(p,t)` = `p ^ t` |
| **Tripole** | grade 3 | grade 5 | 3 / 5 | three points | — | `make_conic_tripole(p1,p2,p3)` = `p1^p2^p3` |
| **Quadpole** | grade 4 | grade 4 | 4 / 4 | four points | — | `make_conic_quadpole(p1,p2,p3,p4)` = `p1^p2^p3^p4` |

### n-pole point extraction (`ccga/extract.py`)

Unlike the dipole's single-radical closed form
$p_{1,2} = (pp \pm \sqrt{pp^2})/(e_\infty\cdot pp)$, three+ points are an irreducible
cubic/quartic — no single $\pm\sqrt{}$, but solvable by radicals via a GA-native
reduction to dipoles:

- **`extract_tripole(T)`** — circumcircle `circumcircle(T)` = the closed-form blade
  `T ∧ Iod ∧ Iinfd` (grade-7 conic); the membership `q(t)∧T=0` cuts a **cubic** on
  it, solved by **Cardano** (cube root). A generic rotation of the rational circle
  parameter keeps any point off the $t=\infty$ pole.
- **`extract_quadpole(Q)`** — the pencil of conics through the 4 points is the
  GA-native family `pencil(Q)` = $\{(Iod∧Q∧p_5)\cdot I^{-1}\}$; its **resolvent
  cubic** (degenerate members, Cardano) picks a pairing $Q=pp_{ij}∧pp_{kl}$, and
  each line carries a dipole split by the standard $\pm\sqrt{}$ (Ferrari).
| **General conic** | grade 7 | grade 1 | 7 / 1 | $Ax^2+By^2+Cxy+Dx+Ey+F=0$ | discriminant $\Delta=C^2-4AB$ | `conic_from_5points([p1..p5])` (OPNS `Iod ^ p1 ^ … ^ p5`) |
| **Circle** | grade 7 | grade 1 | 7 / 1 | $A{=}B$, $C{=}0$; radius via `einf` | $s^2=r^2$ | `make_circle(cx,cy,r)` |
| **Ellipse** | grade 7 | grade 1 | 7 / 1 | $\Delta<0$ | matrix det (empty → imaginary) | `make_ellipse(a,b,cx,cy)` |
| **Hyperbola** | grade 7 | grade 1 | 7 / 1 | $\Delta>0$ | real | `make_hyperbola(a,b,cx,cy)` |
| **Parabola** | grade 7 | grade 1 | 7 / 1 | $\Delta=0$ | real | `make_parabola(p_focus,axis)` |
| **Tilted ellipse** | grade 7 | grade 1 | 7 / 1 | rotated ($C\neq0$ via `eo3`) | matrix det | `make_tilted_ellipse(a,b,theta,cx,cy)` |
| **Line** (degenerate conic) | grade 7 | grade 1 | 7 / 1 | $A{=}B{=}C{=}0$ | real | `make_line_ipns(nx,ny,d)` / `make_line_2points(p1,p2)` |

## 2. Flats and ideal elements

| Object | OPNS | IPNS | Grade(s) | Meaning | Reality test | Construction |
|---|---|---|---|---|---|---|
| **Ideal point** | grade 1 | — | 1 | asymptotic direction (round point, $eo$ dropped) | n/a ($p\cdot e_\infty=0$) | `make_ideal_point(x,y,r,imaginary)` |
| **Flat point** | grade 4 | — | 4 | finite point pinned to infinity | n/a | `make_flat_point(x,y)` = `p ^ Iinf` |
| **Line at infinity** | grade 3 | — | 3 | $L_\infty = I_\infty$ | n/a | `make_line_at_infinity()` = `Iinf` |
| **Conic at infinity** | grade 5 | — | 5 | $I_o^{\triangleright}\wedge I_\infty$ | n/a | `make_conic_at_infinity()` = `Iod ^ Iinf` |

---

## 3. CGA "round" object family (via `Iinfd`)

Wedging with the infinity-gauge blade `Iinfd` recovers the full CGA round
hierarchy inside CCGA (`ccga/cga.py`). It collapses the two conic-specific
infinity directions ($e_{\bar\infty}$, $e_{\infty_3}$) onto the single isotropic
CGA radius term (§3.3 / §3.9). Each CGA object of grade $k$ lands at grade
$k+2$ in CCGA. Constructors take **points** (multivectors).

- **CGA blade**: `cga.cga_blade(O) = Iod | O` recovers the underlying CGA blade
  (grade $= $ grade$(O) - 2$).
- **Reality** is uniform: `cga.reality(O) = sign((Iod | O)²)` → real / imaginary
  / degenerate. (Sign only — the magnitude carries a construction-dependent scale.)
- **Finite vs ideal**: `cga.is_finite(O)` — presence of an $eo$ component.
- `cga.classify_cga(O)` → `{type, grade, cga_grade, reality, finite}`; the main
  `classify()` also recognises these (and keeps `line_at_infinity` /
  `conic_at_infinity` for the pure at-infinity blades).

| Object | CCGA grade | CGA grade | Meaning | Reality / state | Construction |
|---|---|---|---|---|---|
| **Round point** | 3 | 1 | CGA round point / sphere | finite or ideal; real / imaginary / degenerate | `cga.round_point(p)` = `p ^ Iinfd` |
| **Point pair** | 4 | 2 | two CGA points | real / imaginary | `cga.point_pair(p1,p2)` = `p1 ^ p2 ^ Iinfd` |
| **Flat point** | 4 | 2 | flat point ($\equiv -(p\wedge I_\infty)$ = `make_flat_point`) | real | `cga.flat_point(p)` = `p ^ einf ^ Iinfd` |
| **Circle** | 5 | 3 | circle through 3 points | real / imaginary | `cga.circle(p1,p2,p3)` = `p1 ^ p2 ^ p3 ^ Iinfd` |
| **Line** | 5 | 3 | line through 2 points (flat) | real | `cga.line(p1,p2)` = `p1 ^ p2 ^ einf ^ Iinfd` |

A radius-carrying point (`make_point_ccga(x,y,r)`) turns `round_point` into a
**sphere** and `circle` into an off-radius circle; `imaginary=True` gives an
imaginary-radius (empty-locus) round object. `make_round_point(x,y,r)` in
`objects.py` is the coordinate-based entry point and delegates to
`cga.round_point`.

### Incidence

A point $q$ lies on an OPNS object $O$ iff $q \wedge O = 0$. Verified in
`tests/test_cga.py`: the 3 builder points and a 4th cocircular point lie on a
`circle`; collinear points lie on a `line`; off-object points do not.
