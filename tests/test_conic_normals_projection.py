"""
Conic normals and orthogonal projection of a point onto a conic.

  - normal_line(C, p): the line ⟂ tangent through p ∈ C;
  - apollonius_conic(C, q): the rectangular hyperbola of normal feet from q;
  - normal_feet / project_point_to_conic: feet of normals and the nearest one,
    computed as conic ∩ apollonius_conic.
"""
import numpy as np
import pytest

from ccga.point import point
from ccga.objects import (
    make_ellipse, make_tilted_ellipse, normal_line, apollonius_conic,
    tangent_line,
)
from ccga.operations import grades
from ccga.classify import (
    ipns_to_coeffs, _conic_vector, conic_type, conic_center, conic_discriminant,
)
from ccga.extract import normal_feet, project_point_to_conic

TOL = 1e-9


def _coeffs(C):
    return ipns_to_coeffs(_conic_vector(C))


def test_normal_perpendicular_to_tangent():
    E = make_ellipse(3, 2)[0]
    p = point(3*np.cos(0.7), 2*np.sin(0.7))
    n = normal_line(E, p)
    assert grades(n) == [1]
    assert abs(float((p | n).e)) < TOL                # p on the normal
    tx, ty = ipns_to_coeffs(tangent_line(E, p))[3:5]
    nx, ny = ipns_to_coeffs(n)[3:5]
    assert abs(tx*nx + ty*ny) < 1e-9                  # tangent ⟂ normal


def test_apollonius_is_rectangular_hyperbola_through_q_and_center():
    E = make_ellipse(3, 2, cx=1.0, cy=-0.5)[0]
    q = point(5.0, 2.0)
    g = apollonius_conic(E, q)
    A, B, C, D, Ee, F = ipns_to_coeffs(g)
    assert conic_discriminant(A, B, C) > 0            # hyperbola
    assert abs(A + B) < 1e-9                          # rectangular (trace 0)
    # passes through q and through the conic center
    assert abs(float((q | g).e)) < 1e-7
    cx, cy = conic_center(E)
    assert abs(A*cx*cx + B*cy*cy + C*cx*cy + D*cx + Ee*cy + F) < 1e-7


@pytest.mark.parametrize("qx,qy", [(5, 1), (0.5, 0.3), (-6, 2), (4, 4)])
def test_projection_is_valid_foot_and_nearest(qx, qy):
    E = make_ellipse(3, 2)[0]
    A, B, C, D, Ee, F = _coeffs(E)
    q = point(qx, qy)
    feet = normal_feet(E, q)
    assert len(feet) in (2, 4)
    for (x, y) in feet:                               # every foot is valid
        assert abs(A*x*x + B*y*y + C*x*y + D*x + Ee*y + F) < 1e-7
        grad = np.array([2*A*x + C*y + D, C*x + 2*B*y + Ee])
        assert abs((qx - x)*grad[1] - (qy - y)*grad[0]) < 1e-6   # (q−p) ∥ ∇F
    proj = project_point_to_conic(E, q)
    dmin = np.hypot(qx - proj[0], qy - proj[1])
    assert dmin <= min(np.hypot(qx - x, qy - y) for x, y in feet) + 1e-9


def test_projection_of_point_on_conic_is_itself():
    E = make_tilted_ellipse(4, 2, np.pi/5, cx=1.0, cy=2.0)[0]
    A, B, C, D, Ee, F = _coeffs(E)
    # a point on the conic
    x0 = 2.3
    ys = np.roots([B, C*x0 + Ee, A*x0*x0 + D*x0 + F])
    y0 = float(ys[np.isreal(ys)][0].real)
    proj = project_point_to_conic(E, point(x0, y0))
    assert np.allclose(proj, (x0, y0), atol=1e-6)
