"""
Ellipse from center + one focus (the inverse of conic_center / conic_foci).

Center + one focus fixes only 4 of the 5 d.o.f. (center, major-axis direction,
focal distance c) — the size is a one-parameter family, so make_ellipse_from_focus
takes exactly one of {a, ecc, through}.  Built GA-natively by a versor sandwich;
verified to round-trip against the property extractors for all three modes, both
focus choices, and tilted/translated configurations.
"""
import numpy as np
import pytest

from ccga.objects import make_ellipse_from_focus, make_tilted_ellipse
from ccga.point import point
from ccga.classify import (
    conic_center, conic_foci, conic_subtype, conic_eccentricity,
    ipns_to_coeffs, _conic_vector,
)


def _coeffs(C):
    """Normalized (sign-fixed) conic coefficients, for scale-free comparison."""
    c = np.array(ipns_to_coeffs(_conic_vector(C)), float)
    c /= np.linalg.norm(c)
    return c if c[np.argmax(np.abs(c))] >= 0 else -c


def _reference(a, b, theta, cx, cy):
    _, s = make_tilted_ellipse(a, b, theta, cx, cy)
    (f1, f2) = conic_foci(s)
    ctr = conic_center(s)
    c = np.hypot(f2[0] - ctr[0], f2[1] - ctr[1])
    return s, point(*ctr), point(*f1), point(*f2), c


@pytest.mark.parametrize("a,b,theta,cx,cy", [
    (3.0, 1.8, np.radians(37), 1.2, -0.7),
    (5.0, 2.0, 0.0, 0.0, 0.0),
    (4.0, 3.5, np.radians(120), -2.0, 1.5),
])
def test_roundtrip_all_modes(a, b, theta, cx, cy):
    s, pc, f1, f2, c = _reference(a, b, theta, cx, cy)
    ref = _coeffs(s)

    # mode 1: semi-major axis
    _, r_a = make_ellipse_from_focus(pc, f2, a=a)
    # mode 2: eccentricity
    _, r_e = make_ellipse_from_focus(pc, f2, ecc=c / a)
    # mode 3: through a known on-curve point (a vertex on the major axis)
    vertex = point(cx + a*np.cos(theta), cy + a*np.sin(theta))
    _, r_t = make_ellipse_from_focus(pc, f2, through=vertex)
    # either focus must give the same ellipse
    _, r_other = make_ellipse_from_focus(pc, f1, a=a)

    for tag, r in (("a", r_a), ("ecc", r_e), ("through", r_t),
                   ("other-focus", r_other)):
        assert conic_subtype(*ipns_to_coeffs(_conic_vector(r))) == "ellipse", tag
        assert np.allclose(_coeffs(r), ref, atol=1e-7), tag


def test_confocal_family_is_one_parameter():
    """Same center + focus, different a ⇒ different ellipses sharing that focus."""
    pc, f = point(1.2, -0.7), point(3.0, 0.4)
    focus_xy = (3.0, 0.4)
    seen = set()
    for a in (2.5, 3.0, 4.0, 6.0):
        _, s = make_ellipse_from_focus(pc, f, a=a)
        foci = [tuple(round(v, 6) for v in F) for F in conic_foci(s)]
        assert any(np.allclose(F, focus_xy, atol=1e-6) for F in foci)
        seen.add(round(conic_eccentricity(s), 6))
    assert len(seen) == 4                      # genuinely distinct ellipses


def test_input_guards():
    pc, f = point(0.0, 0.0), point(2.4, 0.0)
    with pytest.raises(ValueError):            # no size parameter
        make_ellipse_from_focus(pc, f)
    with pytest.raises(ValueError):            # two size parameters
        make_ellipse_from_focus(pc, f, a=3.0, ecc=0.8)
    with pytest.raises(ValueError):            # a ≤ c
        make_ellipse_from_focus(pc, f, a=2.0)
    with pytest.raises(ValueError):            # e out of (0,1)
        make_ellipse_from_focus(pc, f, ecc=1.5)
    with pytest.raises(ValueError):            # focus on center
        make_ellipse_from_focus(pc, point(0.0, 0.0), a=3.0)
