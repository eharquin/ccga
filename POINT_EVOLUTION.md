# The Point Embedding from CGA → ACGA → CCGA

How the conformal point grows as we add infinity directions — and why the origin
side never grows with it. Every formula below is verified symbolically in
`kingdon` (SymPy backend); the verification snippet is at the end.

---

## 0. Setup — three conformal pairs

CCGA lives in $\mathbb{R}^{5,3}$ with **three** null o/∞ pairs
$(e_{o_i}, e_{\infty_i})$, $i=1,2,3$, satisfying $e_{o_i}^2=e_{\infty_i}^2=0$ and
$e_{o_i}\!\cdot e_{\infty_i}=-1$. We group them into a *rotated* (physically
meaningful) basis:

| pair | origin | infinity | quadratic monomial | role |
|---|---|---|---|---|
| isotropic | $e_o = e_{o_1}+e_{o_2}$ | $e_\infty = \tfrac12(e_{\infty_1}+e_{\infty_2})$ | $x^2+y^2$ | **size** (radius) |
| anisotropic | $e_{\bar o} = e_{o_1}-e_{o_2}$ | $e_{\bar\infty} = \tfrac12(e_{\infty_1}-e_{\infty_2})$ | $x^2-y^2$ | **shape** (stretch) |
| shear / cross | $e_{o_3}$ | $e_{\infty_3}$ | $xy$ | **shape** (tilt) |

with $e_o\!\cdot e_\infty = e_{\bar o}\!\cdot e_{\bar\infty} = -1$, and the
isotropic pair orthogonal to the anisotropic pair. The dictionary going the
other way is

$$e_{\infty_1} = e_\infty + e_{\bar\infty},\qquad
  e_{\infty_2} = e_\infty - e_{\bar\infty}.$$

This is exactly the GAC naming ($\bar n_+=e_o,\ n_+=e_\infty,\ \bar n_-=e_{\bar o},\
n_-=e_{\bar\infty},\ \bar n_\times=e_{o_3},\ n_\times=e_{\infty_3}$, §3.9).

The point embedding maps the plane coordinates $(x,y)$ to the **quadratic
monomials** $(x^2+y^2,\ x^2-y^2,\ xy)$ on the infinity side (a Veronese map). Each
algebra in the hierarchy switches on one more monomial.

---

## 1. CGA — isotropic infinity only

$$\boxed{\,p_{\text{CGA}} = e_o + x\,e_1 + y\,e_2 + \tfrac12(x^2+y^2)\,e_\infty\,}$$

Only the trace $x^2+y^2$ is encoded, so the only "size" degree of freedom is a
single isotropic radius: **CGA reaches circles only.** $p_{\text{CGA}}^2 = 0$ ✓.

---

## 2. ACGA — add anisotropy via $e_{\bar\infty}$

Complete the CGA point by switching on the second monomial $x^2-y^2$, carried by
the anisotropic infinity $e_{\bar\infty}$:

$$\boxed{\,p_{\text{ACGA}} = e_o + x\,e_1 + y\,e_2
   + \tfrac12(x^2+y^2)\,e_\infty + \tfrac12(x^2-y^2)\,e_{\bar\infty}\,}$$

Expanding $e_\infty,e_{\bar\infty}$ in components, the two infinity terms collapse
to the clean per-axis form (verified identical, $\text{diff}^2=0$):

$$\tfrac12(x^2+y^2)e_\infty + \tfrac12(x^2-y^2)e_{\bar\infty}
  \;=\; \tfrac{x^2}{2}\,e_{\infty_1} + \tfrac{y^2}{2}\,e_{\infty_2}.$$

Now $x^2$ and $y^2$ are independent, so **ACGA reaches axis-aligned conics**
(ellipses, hyperbolas with axes along $e_1,e_2$) — but not tilted ones.
$p_{\text{ACGA}}^2 = 0$ ✓.

---

## 3. CCGA — add the cross term via $e_{\infty_3}$

Switch on the last monomial $xy$, carried by the shear/cross infinity
$e_{\infty_3}$ (GAC's $n_\times$):

$$\boxed{\,p_{\text{CCGA}} = e_o + x\,e_1 + y\,e_2
   + \tfrac12(x^2+y^2)\,e_\infty + \tfrac12(x^2-y^2)\,e_{\bar\infty}
   + xy\,e_{\infty_3}\,}$$

equivalently, in components,

$$p_{\text{CCGA}} = e_o + x\,e_1 + y\,e_2
   + \tfrac{x^2}{2}\,e_{\infty_1} + \tfrac{y^2}{2}\,e_{\infty_2} + xy\,e_{\infty_3}.$$

With all three quadratic monomials present, the full symmetric coefficient set
$(A,B,C)$ of a conic is reachable: **CCGA reaches general (tilted) conics.**
$p_{\text{CCGA}}^2 = 0$ ✓. This is the embedding used throughout the repo
(`ccga/point.py`).

---

## 4. Why the origin side never grows

Tempting "coherence" idea: also symmetrize the origin, $e_o = e_{o_1}+e_{o_2}+e_{o_3}$,
and write a fully symmetric point. **This is forbidden by the null condition.**

For a general point $p = a_1e_{o_1}+a_2e_{o_2}+a_3e_{o_3} + x\,e_1+y\,e_2
+ b_1e_{\infty_1}+b_2e_{\infty_2}+b_3e_{\infty_3}$ one gets

$$p^2 = (x^2+y^2) - 2(a_1b_1 + a_2b_2 + a_3b_3).$$

Plugging the Veronese map $b_1=\tfrac{x^2}{2},\,b_2=\tfrac{y^2}{2},\,b_3=xy$:

$$p^2 = x^2+y^2 - \big(a_1x^2 + a_2y^2 + 2a_3\,xy\big).$$

Requiring $p^2=0$ for **all** $(x,y)$ forces

$$\boxed{a_1 = 1,\quad a_2 = 1,\quad a_3 = 0.}$$

So the point's $e_{o_3}$ coefficient is *pinned to zero* — $e_o = e_{o_1}+e_{o_2}$
is not a convention, it is the unique null choice. Concretely, the "symmetric"
point ($a_3=1$) gives

$$p^2 = -2xy \neq 0,\qquad e_o\!\cdot e_\infty = -\tfrac32 \neq -1,$$

breaking both nullity and the $p\cdot e_\infty=-1$ normalization. The origin side
stays put while the infinity side grows; the asymmetry is structural.

> Footnote on the **other** symmetrization (`einf` including `einf3`): adding
> $e_{\infty_3}$ to $e_\infty$ keeps $e_o\!\cdot e_\infty=-1$ (since $e_o$ has no
> $e_{o_3}$ partner) and changes no locus, but it pushes a spurious gauge-inert
> $e_{\infty_3}$ component into every round object, violating the §4 canonical
> form ($e_{\infty_3}=0$). No gain, loss of canonicity — so $e_\infty$ stays
> $\tfrac12(e_{\infty_1}+e_{\infty_2})$.

---

## 5. The radius — one isotropic scalar, invariant across the hierarchy

A round point / circle adds a radius on the **infinity** side:

$$p \;\mapsto\; p \mp \tfrac{r^2}{2}\,e_\infty,\qquad
  \Big(p \mp \tfrac{r^2}{2}e_\infty\Big)^2 = \pm r^2$$

($-$ → real radius, $p^2=+r^2$; $+$ → imaginary radius, $p^2=-r^2$). This is
**identical in CGA, ACGA and CCGA**, because the result depends only on
$\text{base}\cdot e_\infty = -1$:

$$\big(\text{base}\mp\tfrac{r^2}{2}e_\infty\big)^2
  = \underbrace{\text{base}^2}_{0} \mp r^2(\text{base}\cdot e_\infty)
  + \underbrace{\tfrac{r^4}{4}e_\infty^2}_{0} = \pm r^2.$$

Crucially there is **one isotropic radius**, not one per direction (§3.3): the
radius rides on $e_\infty$ alone (the $x^2+y^2$ trace), so it always shifts the
constant term $F$ symmetrically and always describes circular *size*. The extra
infinity directions of ACGA/CCGA encode *shape*, not size:

$$\text{size} \leftrightarrow e_\infty,\qquad
  \text{shape} \leftrightarrow e_{\bar\infty}\ (\text{stretch}),\ e_{\infty_3}\ (\text{tilt}).$$

This is the §3.2 statement "size lives on the infinity side" read along the
hierarchy: CGA has *only* the size knob (hence circles only); ACGA and CCGA add
shape knobs without touching the radius mechanism.

> Side note: $\mp\tfrac{r^2}{2}e_{\infty_1}$ (a single direction) *also* squares
> to $\pm r^2$, but it is **not** a radius — it shifts $F$ anisotropically, i.e.
> it changes shape, not size. The isotropic $e_\infty$ is the only direction that
> gives a genuine circular radius.

---

## 6. Summary

| | infinity dirs used | point formula | radius | conics reachable |
|---|---|---|---|---|
| **CGA** | $e_\infty$ | $e_o + x e_1 + y e_2 + \tfrac12(x^2{+}y^2)e_\infty$ | $\mp\tfrac{r^2}{2}e_\infty$ | circles |
| **ACGA** | $e_\infty,\ e_{\bar\infty}$ | $\;+\ \tfrac12(x^2{-}y^2)e_{\bar\infty}$ | $\mp\tfrac{r^2}{2}e_\infty$ | axis-aligned conics |
| **CCGA** | $e_\infty,\ e_{\bar\infty},\ e_{\infty_3}$ | $\;+\ xy\,e_{\infty_3}$ | $\mp\tfrac{r^2}{2}e_\infty$ | general (tilted) conics |

Origin side: always $e_o = e_{o_1}+e_{o_2}$ (forced by $p^2=0$).
Radius: always isotropic on $e_\infty$ (one scalar, §3.3).
Growth happens only on the infinity side, one quadratic monomial at a time.

---

## Verification

```python
from ccga.algebra import eo, e1, e2, einf, einfbar, einf1, einf2, einf3, eo1, eo2, eo3
import sympy as sp
x, y, r = sp.symbols('x y r', real=True)
sq = lambda p: sp.expand((p * p).e)
z  = lambda e: sp.nsimplify(sp.expand(e), rational=True)   # float→rational, robust ==

p_cga  = eo + x*e1 + y*e2 + sp.Rational(1,2)*(x*x+y*y)*einf
p_acga = p_cga + sp.Rational(1,2)*(x*x-y*y)*einfbar
p_ccga = p_acga + x*y*einf3

assert z(sq(p_cga)) == 0 and z(sq(p_acga)) == 0 and z(sq(p_ccga)) == 0   # all null
acga_direct = eo + x*e1 + y*e2 + sp.Rational(1,2)*x*x*einf1 + sp.Rational(1,2)*y*y*einf2
assert z(sq(p_acga - acga_direct)) == 0                                  # einf/einfbar == per-axis
for p in (p_cga, p_acga, p_ccga):
    assert z(sq(p - sp.Rational(1,2)*r*r*einf)) == r**2                  # radius → +r^2 everywhere
    assert z((p | einf).e) == -1                                        # normalization

# forbidden symmetrization: a3 = 1 breaks nullity
p_sym = (eo1+eo2+eo3) + x*e1 + y*e2 + sp.Rational(1,2)*x*x*einf1 \
        + sp.Rational(1,2)*y*y*einf2 + x*y*einf3
assert z(sq(p_sym)) == -2*x*y                                           # NOT null
```
