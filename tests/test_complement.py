"""
Tests for the right complement A^c, norm, and orthogonality (paper §6.1,
"Conics, their pencils and intersections in GA").

  E ^ E^c = I  for every null-basis blade E (combinatorial complement
  w.r.t. the canonical algebraic basis (eo1,eo2,eo3,e1,e2,einf1,einf2,einf3)),
  extended by linearity.

  Universal identity:  A ^ A^c = (sum_E A_E^2) * I
  Norm:                ||A||^2 = (A ^ A^c)^*  =  sum_E A_E^2
  Orthogonality:       A, B orthogonal  <=>  A ^ B^c = 0
"""
import numpy as np
from ccga.algebra import e1, e2, eo, einf1, einf2, einf3, I, to_null_basis
from ccga.operations import (
    dual, undual, right_complement, norm2, norm, orthogonal, is_zero, grades,
)
from ccga.objects import make_conic_ipns
from ccga.classify import ipns_to_coeffs

TOL = 1e-9


def test_grade_complementary():
    """A^c has grade 8-k for grade-k A (verified for k=1 and k=7)."""
    p = eo + 2*e1 + 3*e2 + 4*einf1 + 5*einf2 + 6*einf3
    assert grades(p) == [1]
    assert grades(right_complement(p)) == [7]

    s = make_conic_ipns(0.5, 0.5, -0.3, -1.0, 2.0, -2.0)
    C = undual(s)
    assert grades(C) == [7]
    assert grades(right_complement(C)) == [1]


def test_involution_up_to_sign():
    """(A^c)^c == -A (verified for grades 1 and 7)."""
    p = eo + 2*e1 + 3*e2 + 4*einf1 + 5*einf2 + 6*einf3
    assert is_zero(right_complement(right_complement(p)) + p, TOL)

    s = make_conic_ipns(0.5, 0.5, -0.3, -1.0, 2.0, -2.0)
    C = undual(s)
    assert is_zero(right_complement(right_complement(C)) + C, TOL)


def test_one_pse_right_complement_identity():
    """
    Theorem (right complement of 1-PSE): for p = a*eo + b*e1 + c*e2
    + d*einf1 + e*einf2 + f*einf3 (a 1-PSE point),

      p ^ p^c = (sum_E p_E^2) * I

    where p_E ranges over the *canonical* null basis (eo1,eo2,eo3,e1,e2,
    einf1,einf2,einf3): since e_o = eo1+eo2 contributes a=p_{eo1}=p_{eo2},
    sum_E p_E^2 = 2a^2 + b^2 + c^2 + d^2 + e^2 + f^2.
    """
    a, b, c, d, e, f = 2.0, 3.0, 5.0, 7.0, 11.0, 13.0
    p = a*eo + b*e1 + c*e2 + d*einf1 + e*einf2 + f*einf3

    pc = right_complement(p)
    wedge = p ^ pc
    wd, Id = dict(wedge.items()), dict(I.items())
    key = next(k for k, v in Id.items() if abs(v) > 1e-12)
    ratio = wd.get(key, 0.0) / Id[key]

    expected = 2*a*a + b*b + c*c + d*d + e*e + f*f
    assert abs(ratio - expected) < TOL


def test_norm_matches_conic_coefficients():
    """
    ||s||^2 for the canonical IPNS conic s (e_infbar=0, e_inf3=0) equals
    4A^2 + 4B^2 + C^2 + D^2 + E^2 + F^2/2 -- the "fixed" norm that includes
    the C (xy) term dropped by a naive sqrt(s~.s).
    """
    A, B, C, D, E, F = 0.5, 0.5, -0.3, -1.0, 2.0, -2.0
    s = make_conic_ipns(A, B, C, D, E, F)
    assert ipns_to_coeffs(s) == (A, B, C, D, E, F)

    expected = 4*A*A + 4*B*B + C*C + D*D + E*E + F*F/2
    assert abs(norm2(s) - expected) < TOL
    assert abs(norm(s) - np.sqrt(expected)) < TOL

    # OPNS form (grade-7) has the same norm as its IPNS dual (grade-1).
    C7 = undual(s)
    assert abs(norm2(C7) - expected) < TOL


def test_orthogonal_basis_vectors():
    """e1 and e2 are orthogonal (A ^ B^c = 0); e1 is not orthogonal to itself."""
    assert orthogonal(e1, e2)
    assert not orthogonal(e1, e1)


def test_orthogonal_is_null_basis_dot_product():
    """A ^ B^c = 0  <=>  the null-basis coefficients of A, B are orthogonal."""
    q1 = eo + 2*e1                  # null comps: eo1=1, eo2=1, e1=2
    q2 = -0.5*e2 + einf1            # null comps: e2=-0.5, einf1=1  (disjoint support)
    dot = sum(to_null_basis(q1).get(k, 0.0) * to_null_basis(q2).get(k, 0.0)
              for k in set(to_null_basis(q1)) | set(to_null_basis(q2)))
    assert abs(dot) < TOL
    assert orthogonal(q1, q2)
