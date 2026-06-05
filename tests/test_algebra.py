"""Tests for ccga/algebra.py — §1 anchors."""
import numpy as np
import pytest
from ccga.algebra import (
    alg, e1, e2,
    eo1, eo2, eo3, einf1, einf2, einf3,
    eo, einf, eobar, einfbar,
    Iod, Iinfd, Io, Iinf, Ieps, I, I_inv,
    gram_matrix, verify_gram, verify_pseudoscalar, verify_special_blades,
)
from ccga.operations import grades, is_zero


def test_gram_matrix():
    """Gram matrix equals the §1 target (8×8 in null basis)."""
    G = gram_matrix()
    target = np.zeros((8, 8))
    target[0, 0] = 1.0  # e1·e1
    target[1, 1] = 1.0  # e2·e2
    for oi, ii in [(2, 3), (4, 5), (6, 7)]:
        target[oi, ii] = -1.0
        target[ii, oi] = -1.0
    assert np.allclose(G, target, atol=1e-12)


def test_pseudoscalar():
    """I^2 = -1 and I * I_inv = 1."""
    verify_pseudoscalar()


def test_special_blade_inner_products():
    """§1 key inner products: eo·einf=-1, eobar·einfbar=-1, cross=0."""
    verify_special_blades()


def test_null_basis_squares():
    """All eo_i and einf_i are null."""
    for v in [eo1, eo2, eo3, einf1, einf2, einf3]:
        assert abs(float((v * v).e)) < 1e-12


def test_euclidean_basis_squares():
    """e1^2 = e2^2 = 1."""
    assert abs(float((e1 * e1).e) - 1.0) < 1e-12
    assert abs(float((e2 * e2).e) - 1.0) < 1e-12


def test_special_blade_grades():
    """Each special blade has its declared grade (§1 table)."""
    blade_grades = [
        (eo,      1), (einf,    1), (eobar,  1), (einfbar, 1),
        (Iod,     2), (Iinfd,   2),
        (Io,      3), (Iinf,    3), (Ieps,   2),
        (I,       8),
    ]
    for blade, expected_grade in blade_grades:
        gs = grades(blade)
        assert gs == [expected_grade], f"Expected grade {expected_grade}, got {gs}"


def test_pseudoscalar_inverse():
    """I * I_inv = scalar 1 (up to rounding)."""
    product = I * I_inv
    assert abs(float(product.e) - 1.0) < 1e-12
    # no non-scalar components
    for k, v in product.items():
        if bin(k).count('1') > 0:
            assert abs(float(v)) < 1e-12


def test_Iod_grade():
    """Iod = eobar ^ eo3 has grade 2."""
    assert grades(Iod) == [2]


def test_I_squared():
    """I^2 = -1."""
    assert abs(float((I * I).e) + 1.0) < 1e-12


def test_eo_einf_orthogonal_to_eobar_einfbar():
    """eo and einf are orthogonal to eobar and einfbar."""
    pairs = [(eo, eobar), (eo, einfbar), (einf, eobar), (einf, einfbar)]
    for a, b in pairs:
        assert abs(float((a | b).e)) < 1e-12


def test_run_all():
    """Meta-test: run_all_verifications passes without assertion errors."""
    from ccga.algebra import run_all_verifications
    run_all_verifications()
