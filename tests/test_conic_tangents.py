"""
Conic tangents / polars and the dual "conic from 5 tangents" construction.

  - tangent_line(C, p): the polar of a contact point p ∈ C, a grade-1 IPNS line
    that meets the conic in the double point p;
  - polar_line(C, q): the polar of any point, with pole–polar reciprocity
    (q2 on polar(q1) ⟺ q1 on polar(q2)) and polar(center) = line at infinity;
  - conic_from_5_tangents(lines): the dual construction, reconstructing a conic
    from 5 of its tangent lines.
"""
import numpy as np
import pytest

from ccga.point import point
from ccga.objects import (
    make_ellipse, make_tilted_ellipse, make_hyperbola,
    tangent_line, polar_line, conic_from_5_tangents,
)
from ccga.operations import grades
from ccga.classify import (
    _conic_vector, ipns_to_coeffs, conic_type, conic_center,
)

TOL = 1e-9


def _on_conic(s, x, y):
    A, B, C, D, E, F = ipns_to_coeffs(s)
    return abs(A*x*x + B*y*y + C*x*y + D*x + E*y + F)


def _ellipse_point(a, b, t, cx=0.0, cy=0.0):
    return point(cx + a*np.cos(t), cy + b*np.sin(t))


def test_tangent_line_touches_at_double_point():
    E = make_ellipse(3, 2)[0]
    s = _conic_vector(E)
    px, py = 3*np.cos(0.7), 2*np.sin(0.7)
    p = point(px, py)
    l = tangent_line(E, p)
    assert grades(l) == [1] and conic_type(l) == 'line'
    # p lies on the tangent line
    assert abs(float((p | l).e)) < TOL
    # the line meets the conic in a DOUBLE point (discriminant 0)
    A, B, C, D, E_, F = ipns_to_coeffs(s)
    nx, ny, c = ipns_to_coeffs(l)[3:6]
    dirx, diry = -ny, nx                     # direction along the line
    t = -c / (nx*nx + ny*ny)
    bx, by = nx*t, ny*t                      # a point on the line
    qa = A*dirx*dirx + B*diry*diry + C*dirx*diry
    qb = 2*A*bx*dirx + 2*B*by*diry + C*(bx*diry + by*dirx) + D*dirx + E_*diry
    qc = A*bx*bx + B*by*by + C*bx*by + D*bx + E_*by + F
    assert abs(qb*qb - 4*qa*qc) < 1e-9       # tangency


def test_tangent_requires_point_on_conic():
    E = make_ellipse(3, 2)[0]
    with pytest.raises(ValueError):
        tangent_line(E, point(5, 5))         # not on the ellipse


def test_pole_polar_reciprocity():
    E = make_ellipse(3, 2, cx=1.0, cy=-0.5)[0]
    q1, q2 = point(2.0, 1.0), point(-1.0, 0.3)
    l1 = polar_line(E, q1)
    l2 = polar_line(E, q2)
    # q2 on polar(q1)  ⟺  q1 on polar(q2)
    assert abs(float((q2 | l1).e) - float((q1 | l2).e)) < 1e-9


def test_polar_of_center_is_line_at_infinity():
    E = make_ellipse(5, 2, cx=1.0, cy=-2.0)[0]
    cx, cy = conic_center(E)
    l = polar_line(E, point(cx, cy))
    # line at infinity: no finite normal part (nx = ny = 0)
    nx, ny, c = ipns_to_coeffs(l)[3:6]
    assert abs(nx) < 1e-7 and abs(ny) < 1e-7 and abs(c) > 1e-7


@pytest.mark.parametrize("C,sampler", [
    (make_ellipse(3, 2)[0], lambda t: _ellipse_point(3, 2, t)),
    (make_tilted_ellipse(4, 2, np.pi/6, cx=1.0, cy=-0.5)[0],
     lambda t: None),   # filled below from conic roots
])
def test_conic_from_5_tangents_roundtrip(C, sampler):
    s = _conic_vector(C)
    A, B, Cc, D, E, F = ipns_to_coeffs(s)
    # gather points on the conic
    pts = []
    for x in np.linspace(-3, 5, 400):
        for yv in np.roots([B, Cc*x + E, A*x*x + D*x + F]):
            if abs(yv.imag) < 1e-9:
                pts.append((x, float(yv.real)))
    five = [pts[i] for i in np.linspace(0, len(pts) - 1, 5).astype(int)]
    lines = [tangent_line(C, point(*xy)) for xy in five]
    opns, ipns = conic_from_5_tangents(lines)
    assert grades(opns) == [7] and grades(ipns) == [1]
    assert conic_type(ipns) == conic_type(s)
    # coefficients agree up to scale
    o = np.array(ipns_to_coeffs(s))
    r = np.array(ipns_to_coeffs(ipns))
    k = np.argmax(np.abs(o))
    r = r * (o[k] / r[k])
    assert np.allclose(o, r, atol=1e-6)
    # each contact point lies on the reconstructed conic
    assert all(_on_conic(ipns, x, y) < 1e-7 for x, y in five)
