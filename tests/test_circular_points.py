"""
The circular points and Iinfd.

The two imaginary circular points at infinity, I=(1:i:0), J=(1:-i:0), are the
Veronese ideal points vinf(1,±i).  Their wedge IS the infinity-gauge blade:

        I ∧ J = -i · Iinfd ,      Iinfd = (einf1 - einf2) ∧ einf3.

So Iinfd *is* the circular-point pair — the common element of all circles.  A
circle is exactly a conic through these two points, which is why wedging Iinfd
(cga.circle = p1∧p2∧p3∧Iinfd) forces roundness: the 3 points plus the 2 circular
points make 5 points → the unique conic through them = the circle.
"""
import numpy as np

from ccga.point import point
from ccga import cga
from ccga.objects import make_circle, make_ellipse
from ccga.algebra import einf1, einf2, einf3, Iinfd
from ccga.operations import dual

# circular points as Veronese ideal points with complex direction (1, ±i)
I = 0.5*einf1 - 0.5*einf2 + 1j*einf3
J = 0.5*einf1 - 0.5*einf2 - 1j*einf3


def _cmax(mv):
    return max((abs(complex(v)) for v in mv.values()), default=0.0)


def test_circular_points_wedge_to_iinfd():
    assert _cmax((I ^ J) - (-1j) * Iinfd) < 1e-12      # I ∧ J = -i · Iinfd
    assert abs(complex((I * I).e)) < 1e-12             # both null
    assert abs(complex((J * J).e)) < 1e-12


def test_circular_points_lie_on_every_circle():
    for cx, cy, r in [(0, 0, 1), (3, 2, 5), (-1, 4, 0.3)]:
        s = dual(make_circle(cx, cy, r)[0])
        assert abs(complex((I | s).e)) < 1e-9
        assert abs(complex((J | s).e)) < 1e-9


def test_general_ellipse_misses_the_circular_points():
    s = dual(make_ellipse(3, 2)[0])
    assert abs(complex((I | s).e)) > 1e-3              # not on an ellipse


def test_iinfd_circle_passes_through_circular_points():
    # cga.circle = p1∧p2∧p3∧Iinfd = conic through the 3 points AND {I,J}
    C = cga.circle(point(0, 0), point(2, 0), point(1, 2))
    assert _cmax(I ^ C) < 1e-9                          # I on the circle (OPNS)
    assert _cmax(J ^ C) < 1e-9
