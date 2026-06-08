"""
CCGA transformations as versors (closed-form GA, sandwich product V·X·~V).

Translator / rotor / dilator are verified to act correctly on points and conics,
to fix the relevant centre for the "about a point" variants, and (being versors)
to preserve incidence q·s = 0.
"""
import numpy as np
import pytest

from ccga.point import point, normalize
from ccga.operations import is_zero, grades
from ccga.objects import make_ellipse, make_point_pair
from ccga.classify import conic_center, conic_axes, conic_type, _conic_vector
from ccga.transform import (
    apply_versor, translator, rotor, dilator, reflector,
    rotor_about, dilator_about,
)

TOL = 1e-9


def _pt_eq(q, x, y):
    return is_zero(normalize(q) - point(x, y))


# ── points ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tx,ty", [(1.5, 0), (0, -2), (3, 4)])
def test_translator_on_point(tx, ty):
    assert _pt_eq(apply_versor(translator(tx, ty), point(2, 3)), 2 + tx, 3 + ty)


@pytest.mark.parametrize("a", [np.pi/2, np.pi/3, np.pi, -0.7])
def test_rotor_on_point(a):
    x, y = 3.0, -2.0
    q = apply_versor(rotor(a), point(x, y))
    assert _pt_eq(q, x*np.cos(a) - y*np.sin(a), x*np.sin(a) + y*np.cos(a))


@pytest.mark.parametrize("s", [2.0, 0.5, 3.0])
def test_dilator_on_point(s):
    assert _pt_eq(apply_versor(dilator(s), point(2, 3)), 2*s, 3*s)


def test_translator_inverse():
    T = translator(1.5, -2.0)
    Tinv = translator(-1.5, 2.0)
    assert _pt_eq(apply_versor(Tinv, apply_versor(T, point(1, 1))), 1, 1)


# ── about a point ──────────────────────────────────────────────────────────────

def test_rotor_about_fixes_center():
    R = rotor_about(np.pi/2, 1.0, 2.0)
    assert _pt_eq(apply_versor(R, point(1, 2)), 1, 2)        # centre fixed
    assert _pt_eq(apply_versor(R, point(3, 2)), 1, 4)        # 90° about (1,2)


def test_dilator_about_fixes_center():
    D = dilator_about(2.0, 1.0, 2.0)
    assert _pt_eq(apply_versor(D, point(1, 2)), 1, 2)
    assert _pt_eq(apply_versor(D, point(3, 2)), 5, 2)


# ── conics ──────────────────────────────────────────────────────────────────────

def test_rotor_swaps_ellipse_axes():
    Er = apply_versor(rotor(np.pi/2), make_ellipse(3, 2)[0])
    assert conic_type(Er) == 'ellipse'
    assert sorted(round(a, 6) for a, _ in conic_axes(Er)) == [2.0, 3.0]


def test_dilator_scales_ellipse():
    Es = apply_versor(dilator(2.0), make_ellipse(3, 2, cx=1, cy=2)[0])
    assert np.allclose(conic_center(Es), (2, 4), atol=1e-7)        # centre scales too
    assert sorted(round(a, 6) for a, _ in conic_axes(Es)) == [4.0, 6.0]


def test_translator_moves_conic_center():
    Et = apply_versor(translator(3, -1), make_ellipse(3, 2)[0])
    assert np.allclose(conic_center(Et), (3, -1), atol=1e-7)


def test_tilted_ellipse_by_rotation():
    # rotate an axis-aligned ellipse to realise a tilted one
    Et = apply_versor(rotor(np.pi/6), make_ellipse(4, 2)[0])
    (amaj, dmaj), _ = conic_axes(Et)
    assert abs(amaj - 4) < 1e-7
    cos_ang = abs(dmaj[0]*np.cos(np.pi/6) + dmaj[1]*np.sin(np.pi/6))
    assert abs(cos_ang - 1) < 1e-7


# ── versor property: incidence preserved on any object ──────────────────────────

@pytest.mark.parametrize("V", [
    lambda: translator(2, -1), lambda: rotor(0.7),
    lambda: dilator(1.7), lambda: rotor_about(0.5, 2, 1),
])
def test_incidence_preserved(V):
    v = V()
    E = make_ellipse(3, 2)[0]
    q = point(3*np.cos(0.9), 2*np.sin(0.9))
    assert abs(float((q | _conic_vector(E)).e)) < 1e-9        # q on E
    qt = apply_versor(v, q)
    Et = apply_versor(v, E)
    assert abs(float((qt | _conic_vector(Et)).e)) < 1e-7      # qt on transformed E


@pytest.mark.parametrize("nx,ny,d,ex,ey", [
    (0, 1, 0, 2, -3),      # x-axis
    (1, 0, 0, -2, 3),      # y-axis
    (1, -1, 0, 3, 2),      # y = x
    (1, 0, -1, 0, 3),      # x = 1
    (0, 1, -2, 2, 1),      # y = 2
    (1, 1, -3, 0, 1),      # x + y = 3
])
def test_reflector_on_point(nx, ny, d, ex, ey):
    assert _pt_eq(apply_versor(reflector(nx, ny, d), point(2, 3)), ex, ey)


def test_reflector_is_involution():
    V = reflector(1, 2, -1)
    assert _pt_eq(apply_versor(V, apply_versor(V, point(2, 3))), 2, 3)


def test_reflector_on_conic_moves_center():
    E = make_ellipse(3, 2, cx=2, cy=1)[0]
    Er = apply_versor(reflector(0, 1, 0), E)        # reflect across x-axis
    assert conic_type(Er) == 'ellipse'
    assert np.allclose(conic_center(Er), (2, -1), atol=1e-7)


def test_reflector_handles_tilted_line_on_conic():
    # the grade-3 reflector carries the xy coupling a bare line vector cannot:
    # ellipse@(1,2) reflected across y=x must land at (2,1)
    E = make_ellipse(3, 2, cx=1, cy=2)[0]
    Er = apply_versor(reflector(1, -1, 0), E)
    assert np.allclose(conic_center(Er), (2, 1), atol=1e-7)


def test_line_vector_does_not_reflect_geometrically():
    # sandwiching by the line's grade-1 IPNS vector is NOT geometric reflection
    # (breaks the Veronese cross-term) — documents why reflector is grade-3.
    from ccga.objects import make_line_ipns
    ell = make_line_ipns(1, -1, 0)[1]               # y = x line, IPNS vector
    s2 = float((ell * ell).e)
    p = point(2, 3)
    q = ell * p * (ell * (1.0 / s2))                # ℓ p ℓ⁻¹
    # the genuine reflection of (2,3) across y=x is (3,2)
    assert not _pt_eq(q, 3, 2)


def test_point_pair_transforms_as_grade2():
    pp, _ = make_point_pair(point(0, 0), point(2, 0))
    ppt = apply_versor(rotor(np.pi/2), pp)
    assert grades(ppt) == [2]
