"""
Flat point in CCGA — analysis of the correct form.

The flat point is  P_flat = p ∧ Iinf  (grade 4), NOT p ∧ einf.  Because CCGA's
infinity is the 3-D conic-at-infinity Iinf = einf1∧einf2∧einf3, wedging the FULL
infinity annihilates the point's quadratic/Veronese part (einf_i ∧ Iinf = 0),
leaving only the position:  p ∧ Iinf = (eo + x·e1 + y·e2) ∧ Iinf.  A single einf
does not — p ∧ einf keeps quadratic remnants, so it is not truly "flat".
"""
import numpy as np

from ccga.point import point
from ccga.algebra import einf, einf1, einf2, einf3, Iinf, Iinfd, eo, e1, e2, to_null_basis
from ccga.objects import make_flat_point
from ccga.operations import grades, is_zero
from ccga.transform import apply_versor, translator, rotor


def test_flat_point_is_p_wedge_Iinf_grade4():
    p = point(2, 3)
    assert grades(p ^ Iinf) == [4]
    assert grades(p ^ einf) == [2]                    # the grade-2 precursor (not flat)
    assert is_zero(make_flat_point(2, 3) - (p ^ Iinf))


def test_flat_point_is_purely_positional():
    # p ∧ Iinf strips the quadratic part: equals (eo + x e1 + y e2) ∧ Iinf
    p = point(2, 3)
    assert is_zero((p ^ Iinf) - ((eo + 2*e1 + 3*e2) ^ Iinf))
    # because each einf_i ∧ Iinf = 0
    assert all(is_zero(ei ^ Iinf) for ei in (einf1, einf2, einf3))
    # only positional blades remain
    blades = list(to_null_basis(p ^ Iinf).keys())
    assert all('eo' in b or 'e1' in b or 'e2' in b for b in blades)


def test_p_wedge_einf_is_not_flat():
    # p ∧ einf retains quadratic remnants (einf_i ∧ einf3 terms) → not flat
    terms = [n for n in to_null_basis(point(2, 3) ^ einf) if 'einf3' in n]
    assert terms                                       # non-empty: quadratic survives


def test_two_grade4_forms_coincide():
    # einf ∧ Iinfd = -Iinf  ⇒  p ∧ einf ∧ Iinfd = -(p ∧ Iinf)
    assert is_zero((einf ^ Iinfd) + Iinf)
    p = point(2, 3)
    assert is_zero((p ^ einf ^ Iinfd) + (p ^ Iinf))


def test_flat_point_builds_flats_by_join():
    # the flat point is the join-unit for flats: wedging a finite point raises
    # the flat dimension (0-flat → line → plane)
    from ccga import cga
    F = make_flat_point(1, 2)                          # 0-flat (grade 4)
    L = point(4, 0) ^ F                                # 1-flat: line (grade 5)
    assert grades(L) == [5]
    assert is_zero(L - cga.line(point(1, 2), point(4, 0))) or \
           is_zero(L + cga.line(point(1, 2), point(4, 0)))
    # genuine line through the two points
    assert is_zero(point(1, 2) ^ L) and is_zero(point(4, 0) ^ L)
    assert not is_zero(point(0, 5) ^ L)
    P = point(0, 3) ^ L                                # 2-flat: plane (grade 6)
    assert grades(P) == [6]


def test_flat_point_composition_rules():
    from ccga.point import point_at_infinity
    F = make_flat_point(1, 2)
    assert is_zero(F ^ make_flat_point(3, 4))          # flat ∧ flat = 0 (Iinf repeated)
    assert is_zero(F ^ point_at_infinity(1, 1))        # flat ∧ ideal = 0 (dir ⊂ Iinf)
    assert not is_zero(F ^ point(3, 4))                # flat ∧ finite point → line


def test_flat_point_covariant_and_incident():
    p = point(2, 3); F = p ^ Iinf
    # Iinf is translation-invariant ⇒ flat point moves like its point
    T = translator(1.5, -2.0)
    assert is_zero(apply_versor(T, Iinf) - Iinf)
    assert is_zero(apply_versor(T, F) - (point(3.5, 1.0) ^ Iinf))
    # rotation-covariant
    R = rotor(np.pi/2)
    assert is_zero(apply_versor(R, F) - (apply_versor(R, p) ^ Iinf))
    # incidence: only p lies on it
    assert is_zero(p ^ F) and not is_zero(point(5, 1) ^ F)
