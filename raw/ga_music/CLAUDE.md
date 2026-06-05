# Music Chord Geometric Algebra — Exploration Project

## Context

This project explores representing musical pitches, chords, and harmonic operations using a single Clifford algebra with a deliberately chosen mixed signature. The design encodes three musical asymmetries directly into the metric:

- **White keys are stable** → positive signature, `eᵢ² = +1`
- **Black keys are altered / unstable** → negative signature, `eᵢ² = −1`
- **Octaves are equivalence, not pitch** → null signature, `e_oct² = 0`

The algebra is **`Algebra(7, 5, 1)`** (Cl(7, 5, 1)). This commitment is fixed; the hypotheses below test what musical structure falls out of this specific choice. Alternative metrics are out of scope unless explicitly noted.

The author is a PGA researcher; GA tooling is well understood. Focus effort on the **musical content** and **numerical validation**, not on explaining GA basics.

## Stack

- Python 3.11+
- `kingdon` with `Algebra(7, 5, 1)`
- `numpy` for numerics; `sympy` only when symbolic verification is genuinely needed
- Jupyter notebooks (`.ipynb`), one per hypothesis
- `matplotlib` only when a plot clarifies a result

Avoid heavyweight music libraries (`music21`, `mido`).

## Basis layout

kingdon orders basis vectors by signature: positive first, then negative, then null.

| basis | square | pitch                |
|-------|--------|----------------------|
| `e1`  | +1     | C                    |
| `e2`  | +1     | D                    |
| `e3`  | +1     | E                    |
| `e4`  | +1     | F                    |
| `e5`  | +1     | G                    |
| `e6`  | +1     | A                    |
| `e7`  | +1     | B                    |
| `e8`  | −1     | C♯                   |
| `e9`  | −1     | D♯                   |
| `e10` | −1     | F♯                   |
| `e11` | −1     | G♯                   |
| `e12` | −1     | A♯                   |
| `e13` |  0     | octave (null) — `OCTAVE` |

The setup cell must expose a `PITCH` dict and an `OCTAVE` constant so notebooks reference pitches by name, not raw basis index.

## Design note: C-major is the reference key

The signature assignment privileges C-major. F is "white" here; in G-major it would be F♯ that should be white. This means **transposition and modulation are not automorphisms of this algebra in general** — they move pitches between the positive and negative subspaces and so genuinely alter the metric structure of a chord.

This is a feature, not a bug: the algebra knows what key it's in. Expect interesting friction when testing modulation/transposition hypotheses — pure rotors won't suffice, and operators that exchange positive and negative basis vectors will be hyperbolic (boost-like), not circular.

## Notebook structure

1. **Title + one-sentence hypothesis** (markdown).
2. **Setup cell** — algebra construction, `PITCH` dict, `OCTAVE`, helper functions.
3. **Construction cell** — build the relevant objects.
4. **Test cells** — compute predicted equalities/invariants. Print explicit `✓` / `✗` markers per claim.
5. **Discussion cell** (markdown) — held, partially held, or failed? What broke?

Keep cells short. One claim per cell when reasonable.

## Notation conventions

- **Pitches** referenced by name via `PITCH['C']`, never by raw basis index in notebook prose.
- **Chords** capitalized: `C_major`, `G_dom7`, `F_min`. Variable names should make the grade obvious.
- **Intervals** by interval class: `P5`, `m3`, `TT`, `M7`.
- **Operations**: `^` wedge, `*` geometric product, `|` inner product (kingdon defaults).
- **Regressive product**: always `(A.dual() ^ B.dual()).dual()`. The `&` operator is unreliable in kingdon.

## Code style

- Compact and idiomatic. Lambdas, dict comprehensions, small helpers over long imperative blocks.
- No premature classes. Labeled multivectors usually suffice.
- Numerical comparisons via `np.allclose` or an `is_zero(mv, tol=1e-10)` helper; never exact equality.

## Validation discipline

Every claim of the form "X equals Y" or "X is a rotor" or "this maps A to B" **must** be verified numerically before being asserted in prose. Unverified conjectures are marked:

```python
# CONJECTURE — not yet verified
```

When a hypothesis fails, **do not paper over it**. Null results are valuable — they tell us which musical phenomena this specific algebra does and does not capture.

## Target hypotheses (one notebook each)

1. **`01_algebra_setup.ipynb`** — construct `Algebra(7, 5, 1)`, define `PITCH` and `OCTAVE`, build standard triads and seventh chords. Verify that the C-major triad has positive `‖·‖²`, the C-diminished triad has reduced or negative `‖·‖²`, and the full chromatic cluster has mixed signature. **Foundational — do this first.**

2. **`02_octave_equivalence.ipynb`** — test the null-vector role. Define octave-shifted voicings of the same chord by adding multiples of `OCTAVE`. Do they collapse to the same equivalence class under chord operations? Compare with PGA's treatment of points at infinity.

3. **`03_modulation_by_fifth.ipynb`** — modulation C→G swaps exactly one pair: F (`e4`, positive) ↔ F♯ (`e10`, negative). Find the operator that performs this swap. Test whether it is a hyperbolic rotor `exp(t · e4 e10)` for some `t`, and whether it maps the C-diatonic 7-blade `e1 ∧ … ∧ e7` to the G-diatonic 7-blade `e1 ∧ e2 ∧ e3 ∧ e10 ∧ e5 ∧ e6 ∧ e7` (up to sign/orientation). **This is the cleanest test of the chosen signature.**

4. **`04_major_minor_orientation.ipynb`** — does reversing a major triad's trivector yield its negative-harmony minor partner? Test `~(C ∧ E ∧ G)` against F-minor (Levy's polar minor of C-major), allowing sign/scalar factors.

5. **`05_semitone_transposition.ipynb`** — the semitone shift maps white↔black non-uniformly. Pure rotors cannot implement it. What is the cleanest operator that does, and where does it break? This notebook is expected to find limits, not clean results.

6. **`06_tritone_substitution.ipynb`** — in the sub-algebra spanned by chord tones of `G7` and `D♭7`, does multiplication by the local pseudoscalar map one to the other (up to sign/scalar)? Use the regressive-product form for duality.

Notebook 01 is foundational. 02–04 are where the chosen signature should pay off. 05–06 are stress tests where the structure may resist; that's informative.

## Out of scope (for now)

- MIDI rendering, audio synthesis
- Tuning beyond 12-TET
- Microtonal extensions
- Score notation
- Performance optimization — clarity beats speed at this stage
- Alternative metrics (consonance-weighted Gram matrices, fully chromatic positive signatures, etc.) — considered and deferred

## When in doubt

Match the rigor used in the `pga_crossratio` project: precise notation, numerical verification before claims, willingness to report negative results.
