"""
Conic property extraction from the multivector (center, axes, eccentricity, foci).

Verified against the closed-form parameters of the named constructors, including
translated and tilted conics, plus the non-central guard for parabolas.
"""
import numpy as np
import pytest

from ccga.objects import (
    make_ellipse, make_hyperbola, make_tilted_ellipse, make_circle,
    make_parabola,
)
from ccga.objects import make_parabola_3points
from ccga.point import point
from ccga.algebra import einf
from ccga.operations import is_zero
from ccga.classify import (
    conic_center, conic_center_point, conic_axes, conic_eccentricity, conic_foci,
    parabola_geometry, ipns_to_coeffs, _conic_vector,
)

TOL = 1e-9


def _axis_lengths(C):
    (a, _), (b, _) = conic_axes(C)
    return a, b


def test_ellipse_properties():
    C = make_ellipse(3, 2)[0]
    assert np.allclose(conic_center(C), (0, 0), atol=TOL)
    assert np.allclose(sorted(_axis_lengths(C)), [2, 3], atol=1e-7)
    assert abs(conic_eccentricity(C) - np.sqrt(5)/3) < 1e-7
    foci = sorted(conic_foci(C))
    assert np.allclose(foci, [(-np.sqrt(5), 0), (np.sqrt(5), 0)], atol=1e-7)


def test_translated_ellipse():
    C = make_ellipse(5, 2, cx=1.0, cy=-2.0)[0]
    assert np.allclose(conic_center(C), (1, -2), atol=1e-9)
    assert np.allclose(sorted(_axis_lengths(C)), [2, 5], atol=1e-7)
    c = np.sqrt(25 - 4)
    foci = sorted(conic_foci(C))
    assert np.allclose(foci, [(1 - c, -2), (1 + c, -2)], atol=1e-7)


def test_tilted_ellipse_axis_directions():
    theta = np.pi / 6
    C = make_tilted_ellipse(4, 2, theta)[0]
    (a, a_dir), (b, b_dir) = conic_axes(C)
    assert abs(a - 4) < 1e-7 and abs(b - 2) < 1e-7
    # major axis direction is ±(cosθ, sinθ)
    cos_ang = abs(a_dir[0]*np.cos(theta) + a_dir[1]*np.sin(theta))
    assert abs(cos_ang - 1) < 1e-7
    # axes orthogonal
    assert abs(a_dir[0]*b_dir[0] + a_dir[1]*b_dir[1]) < 1e-7


def test_circle_properties():
    C = make_circle(2, -1, 3)[0]
    assert np.allclose(conic_center(C), (2, -1), atol=1e-9)
    a, b = _axis_lengths(C)
    assert abs(a - 3) < 1e-7 and abs(b - 3) < 1e-7
    assert conic_eccentricity(C) < 1e-7          # circle: e = 0
    # both foci coincide with the center
    assert np.allclose(conic_foci(C), [(2, -1), (2, -1)], atol=1e-7)


def test_hyperbola_properties():
    C = make_hyperbola(2, 1)[0]
    (a, a_dir), (b, b_dir) = conic_axes(C)
    assert abs(a - 2) < 1e-7 and abs(b - 1) < 1e-7
    assert abs(conic_eccentricity(C) - np.sqrt(5)/2) < 1e-7   # √(a²+b²)/a
    foci = sorted(conic_foci(C))
    assert np.allclose(foci, [(-np.sqrt(5), 0), (np.sqrt(5), 0)], atol=1e-7)
    # transverse axis is horizontal for this hyperbola
    assert abs(abs(a_dir[0]) - 1) < 1e-7


def test_center_as_pole_of_line_at_infinity():
    # central conic: finite center point, normalizable (p·einf = -1)
    Ce = make_ellipse(5, 2, cx=1.0, cy=-2.0)[0]
    pc = conic_center_point(Ce)
    assert abs(float((pc | einf).e) + 1) < 1e-7        # finite point
    from ccga.classify import conic_center as _cc
    assert np.allclose(_cc(Ce), (1, -2), atol=1e-7)

    # parabola: center is an IDEAL point (at infinity), in the axis direction,
    # and it lies on the parabola itself (tangency with the line at infinity)
    C = make_parabola_3points(point(0, 0), point(2, 0.5), point(1, 3), (1, 1))[0]
    pinf = conic_center_point(C)
    assert abs(float((pinf | einf).e)) < 1e-7          # ideal: p·einf = 0
    # the ideal center lies on the parabola:  q · s = 0
    assert abs(float((pinf | _conic_vector(C)).e)) < 1e-7
    # it equals the ideal point in the parabola's axis direction
    # (point_at_infinity is even in its argument, so sign-free)
    from ccga.point import point_at_infinity
    ax = parabola_geometry(C)['axis']
    assert is_zero(pinf - point_at_infinity(*ax))


def test_parabola_has_no_center_or_axes():
    C = make_parabola(1, 'x')[0]
    with pytest.raises(ValueError):
        conic_center(C)            # Δ₂ = 0 → parabola has no center
    with pytest.raises(ValueError):
        conic_axes(C)


def test_axis_aligned_parabola_focus():
    # make_parabola(p,'x'): y² = 4 p x → vertex (0,0), focus (p,0)
    g = parabola_geometry(make_parabola(2, 'x')[0])
    assert np.allclose(g['vertex'], (0, 0), atol=1e-7)
    assert np.allclose(g['focus'], (2, 0), atol=1e-7)
    assert abs(abs(g['axis'][0]) - 1) < 1e-7
    assert conic_eccentricity(make_parabola(2, 'x')[0]) == 1.0
    assert np.allclose(conic_foci(make_parabola(2, 'x')[0])[0], (2, 0), atol=1e-7)


def test_tilted_parabola_focus_directrix_property():
    # focus-directrix: every point of a parabola is equidistant from focus and
    # directrix — the construction-independent correctness test.
    pts = [point(0, 0), point(2, 0.5), point(1, 3)]
    C = make_parabola_3points(*pts, (1, 1))[0]
    g = parabola_geometry(C)
    fx, fy = g['focus']
    d0, u = g['directrix']
    A, B, Cc, D, E, F = ipns_to_coeffs(_conic_vector(C))
    maxerr = 0.0
    for x in np.linspace(-1, 3, 30):
        for yv in np.roots([B, Cc*x + E, A*x*x + D*x + F]):
            if abs(yv.imag) < 1e-7:
                q = np.array([x, yv.real])
                d_focus = np.hypot(q[0] - fx, q[1] - fy)
                d_dir = abs(u[0]*(q[0] - d0[0]) + u[1]*(q[1] - d0[1]))
                maxerr = max(maxerr, abs(d_focus - d_dir))
    assert maxerr < 1e-9
    # the 3 builder points lie on the parabola (sanity)
    assert conic_eccentricity(C) == 1.0


def test_reality_is_scale_invariant():
    # negation / scaling must NOT change reality (p² sign is invariant: (λp)²=λ²p²)
    from ccga.objects import make_point_ccga
    from ccga.classify import reality
    from ccga.algebra import einf
    real = make_point_ccga(1, 1, 2.0)
    imag = make_point_ccga(1, 1, 2.0, imaginary=True)
    for lam in (1.0, -1.0, 2.0, -3.0, 0.5):
        assert reality(lam * real) == 'real'
        assert reality(lam * imag) == 'imaginary'
        # scale-invariant radius² = p²/(p·einf)²  is exactly r² at every scale
        lp = lam * real
        r2 = float((lp * lp).e) / float((lp | einf).e) ** 2
        assert abs(r2 - 4.0) < 1e-9
