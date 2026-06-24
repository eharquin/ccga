# GA-native line ∩ conic intersection solver (CCGA)

Extract the two points where a line meets a general conic, using only geometric-algebra
operations — **no coordinate read-outs, no quadratic-formula on extracted coefficients**.
The whole pipeline is verified in `kingdon` and implemented as the ga-constructor graph
`saved_graphs/CCGA_LINE_CONIC_DIPOLE_GANATIVE.json`.

Conventions follow the rest of this repo (`CLAUDE.md`): signature R^{5,3}, the point
embedding of §2, `&` = meet (regressive), `^` = wedge, `|` = inner product, `!`/dual =
`* I⁻¹`, sandwich `V >>> X = V X ~V`.

---

## 1. The problem and the obstruction

A line through points `A,B` is the degenerate conic `L = A∧B∧Iinf∧Iod` (grade 7); the conic
is `C = p₁∧…∧p₅∧Iod` (grade 7). Their meet is **grade 6**, and it factors as

```
C & L  =  D ∧ T ∧ Iod                         (grade 6)
```

* `D = P₁∧P₂` — the finite intersection **dipole** we want (grade 2),
* `T` — the conic's **asymptotic dipole** (its two ideal points), and
* `Iod` — the OPNS gauge blade.

A line-as-conic is "finite line ∪ line-at-infinity", so the meet Bézout-counts to 4 points:
the 2 finite ones plus the conic's 2 ideal ones. The contamination is `T`.

**Why a naïve contraction fails.** `D` is only fixed *modulo* `T`, and `T` is
*conic-specific*. No fixed blade contracts `C&L` to `D`: any reciprocal of `T` leaks `D`'s
own infinity components (verified — the result is a degenerate `D²≈0` blade that splits to
the wrong points). This is the central difficulty.

---

## 2. The idea: align the line, then the gauge becomes universal

If we first apply a **motor** `M` that moves the line onto the x-axis, the contaminating
direction is no longer conic-specific — it is always the **vertical (y) direction**. The
contamination is canonicalised, so a **single fixed gauge blade** peels it.

```
        meet            align (motor)                 fixed-gauge contraction
 C, L  ─────▶  C & L  ─────────────▶  C' & L'  ──────────────────────────────▶  Tp
 (grade 7)    (grade 6)              (grade 6)        | (Iinfd ^ eo2 ^ eo3)      (grade 2)
                                                                                  │  ~M >>>
                                                                                  ▼
                                                                       EF = the clean dipole
                                                                                  │  ± split
                                                                                  ▼
                                                                              P₁ , P₂
```

---

## 3. The pipeline

### 3.1 Coordinate axes as GA objects

```
Xax = e1 ^ Iinf ^ Io        # the x-axis  (locus y = 0),  grade 7
Yax = e2 ^ Iinf ^ Io        # the y-axis  (locus x = 0),  grade 7
```

(The direction vector `e1`/`e2` wedged with the flat completion `Iinf∧Io`. Note the label
is by direction: `e1^Iinf^Io` is the line **along** e1, i.e. `y = 0`.)

### 3.2 Crossing point — by meet

```
Pc = (L & Xax) | (Iinfd ^ Io)        # grade-1 point where the line meets the x-axis
w  = -(Pc | einf)                    # normaliser  (Pc is NOT unit by default!)
```

`L & Xax` is the grade-6 line∩line object; contracting by `Iinfd∧Io` strips the gauge and
lands the crossing as a grade-1 point. **It must be normalised** by `w = -(Pc·e∞)` before
its coordinates are used — skipping this is the single easiest bug (it silently rescales the
translation and breaks everything downstream).

### 3.3 Line angle — by inner product

The inner product of the line with each axis reads the angle off directly:

```
L | Xax = -2 cos φ          L | Yax = -2 sin φ
phi = atan( (L|Yax) / (L|Xax) )
```

No coordinate differences, no `atan2` needed (a line angle is only defined mod π, so
single-argument `atan` is enough).

### 3.4 The motor `M = R · T₁` (rotation about the crossing)

Translate the crossing `Pc` to the origin, then rotate by `-φ` so the line lands on the
x-axis. Both factors are genuine CCGA versors (see `ccga/transform.py`).

```
ang = -phi
# rotor R(ang): Euclidean part eE × Veronese part eK   (eK handles the x² , y² , xy carriers)
eE = cos(ang/2) - sin(ang/2) * Ieps
K  = (eob ^ einf3) - (eo3 ^ einfb)
eK = 1 + 0.5*sin(2*ang)*K + 0.25*(1 - cos(2*ang))*(K*K)        # closed form, K³ = -4K
R  = eE * eK

# translator T₁ sending Pc → origin   (tx = -cx, ty = -cy)
tx = -Pc.e1 / w        ty = -Pc.e2 / w
T1 = (1 - 0.5*tx*(e1^einf1))*(1 - 0.5*tx*(e2^einf3))
   * (1 - 0.5*ty*(e2^einf2))*(1 - 0.5*ty*(e1^einf3))

M  = R * T1            Mb = ~M
```

A *plain* Euclidean rotor `cos - sin·e₁₂` is **wrong** here: it mangles the Veronese
(`einf`) part of the embedding. The `eK` factor is essential.

### 3.5 Align, then peel with the fixed gauge ⭐

```
Crot = M >>> C        Lrot = M >>> L          # line is now the x-axis (y = 0)

Tp = (Crot & Lrot) | (Iinfd ^ eo2 ^ eo3)      # grade-2 CLEAN dipole — fixed gauge
```

`Iinfd ^ eo2 ^ eo3` is a **constant grade-4 blade**. It works for every conic because, after
alignment, the contaminating asymptotic direction is always vertical.

*Derivation of the constant.* The vertical line-pair's asymptotic dipole
`(LP & (Iinf∧Iod)) | Iinfd` is conic-independent and equals `einf2∧einf3` (verified on
arbitrary conics). Sending it through the origin↔infinity inversion `Inv1 = e3∧e4∧e5`
(`Inv1 >>> einf_i = -½ eo_i`) gives `Inv1 >>> (einf2∧einf3) = ¼ eo2∧eo3`. Wedging with the
infinity gauge `Iinfd` yields the constant `Iinfd ∧ eo2 ∧ eo3`. So the line-pair / coefficient
construction collapses to this one fixed contraction.

### 3.6 Un-align and split

```
EF = Mb >>> Tp                                 # dipole back in the original frame  (∝ P₁∧P₂)
P1 = (EF - sqrt(EF**2)) / (einf | EF)          # standard point-pair split
P2 = (EF + sqrt(EF**2)) / (einf | EF)
```

`EF` is a simple 2-blade, so `EF² ` is a scalar; the `± √` is the irreducible
point-pair split (turning a dipole into its two points is intrinsically one quadratic).
Both `P₁,P₂` come out as clean grade-1 points.

> The split's division by `einf|EF` is safe in the ga-constructor: its `/` uses the
> blade-inverse shortcut `~B/(B·~B)`, robust even when the dipole has tiny magnitude.

---

## 4. Why it is GA-native (and what the √ is)

* Crossing point: a **meet**. Angle: an **inner product**. Alignment: a **versor**.
  Peel: a **fixed contraction**. None of these read coordinates off the objects.
* The only scalar arithmetic is `phi = atan(…)` (one angle) and the final `± √(EF²)`.
  The square root is **not** avoidable: extracting two points from a point pair is a
  quadratic. It is, however, the canonical GA point-pair split — the same √ a CGA dipole
  needs. Geometrically `EF`'s "centre ± radius" *are* the two points.

---

## 5. Generalisation: conic ∩ conic (Bézout 4)

The single-line case collapses the gauge to a constant; the line-pair re-appears for the
genuinely general solver:

1. Two conics `C₁, C₂` → the pencil `λC₁ + μC₂`.
2. Find a **degenerate member** — a **line-pair** — by solving `Δ₃(λC₁+μC₂) = 0` (a cubic).
   `Δ₃` is itself GA-native: it is read off the meet of three dual lines
   (`ccga/classify.py`, `tests/test_conic_meet_lines.py`).
3. Factor the line-pair into its two lines, and run **each line ∩ C₁** through §3.
   This yields the up-to-4 intersection points.

So the line-pair is the bridge object: for one line it degenerates to the fixed gauge of
§3.5; for two conics it is the degenerate pencil member that drives the whole solver. See
`ccga/extract.py` (`conic_intersection`, `pencil`, `extract_quadpole`) for the existing
pencil → resolvent-cubic → line-pair → dipole route; §3 here is the clean GA-native dipole
step it plugs into.

**A line-pair is just a degenerate conic, so §3 already handles it** — no special casing.
With `C` a line-pair (`Δ₃ = 0`, two lines) and `L` a cutting line, the same pipeline returns
the two crossings (one per line), and it works whether the pair is **secant** (lines meet)
or **parallel** (lines don't). Verified on both, plus a hyperbola, against ground truth; the
ga-constructor graph is `saved_graphs/CCGA_PAIR_SECANT_LINES_LINE_INTER.json`. This is the
exact step the conic ∩ conic solver calls after factoring its degenerate pencil member.

---

## 6. Verification

* `kingdon`: `EF ∝ P₁∧P₂` and the split returns the true points to machine precision, on
  multiple independent conic+line configurations (including the self-checking case where the
  line passes through two of the conic's defining points, so `P₁,P₂` are known).
* ga-constructor: `saved_graphs/CCGA_LINE_CONIC_DIPOLE_GANATIVE.json` (33 nodes, no
  coefficient extraction, no line-pair). Earlier iterations for reference:
  `CCGA_LINE_CONIC_EXTRACTION_fixed.json` (line-pair form),
  `ccga_conic_line_dipole_motor.json` / `ccga_conic_line_dipole_rotor.json` (motor / rotor).
* Repo analysis script: `notebook/conic_line_dipole_extraction.py`.
