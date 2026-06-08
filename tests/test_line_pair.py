"""
Line pair (degenerate conic) and its relation to flat points.

A line pair is the **symmetric square** of two line covectors — a degenerate
conic (det M3 = 0).  It is NOT built from flat points: flat points build single
lines, and flat∧flat = 0.  But its component lines (flat-point constructions) are
recovered by factoring (extract._lines_of).
"""
import numpy as np

from ccga.point import point
from ccga.objects import make_line_ipns, make_line_pair, conic_from_5points, make_flat_point
from ccga.classify import ipns_to_coeffs, conic_type, conic_is_degenerate, _conic_vector
from ccga.operations import grades, is_zero
from ccga.extract import _lines_of
from ccga.algebra import Iinf


def test_line_pair_is_degenerate_conic():
    l1 = make_line_ipns(1, -1, 0)[1]      # x - y = 0
    l2 = make_line_ipns(1, 1, -2)[1]      # x + y - 2 = 0
    opns, ipns = make_line_pair(l1, l2)
    assert grades(opns) == [7] and grades(ipns) == [1]
    assert conic_is_degenerate(ipns)                 # det M3 = 0
    # locus = union of the two lines: points on either satisfy q·s = 0
    for q in (point(0, 0), point(2, 2), point(0, 2), point(3, -1)):  # on L1 or L2
        assert abs(float((q | ipns).e)) < 1e-9
    assert abs(float((point(1, 0) | ipns).e)) > 1e-9   # off both lines


def test_line_pair_matches_five_point_conic():
    # 3 points on L1 (x=y), 2 on L2 (x+y=2): the 5-point conic is the same pair
    _, s5 = conic_from_5points([point(0, 0), point(1, 1), point(2, 2),
                                point(0, 2), point(3, -1)])
    _, slp = make_line_pair(make_line_ipns(1, -1, 0)[1], make_line_ipns(1, 1, -2)[1])
    o = np.array(ipns_to_coeffs(s5)); r = np.array(ipns_to_coeffs(slp))
    k = np.argmax(np.abs(o))
    assert np.allclose(o, r * (o[k] / r[k]), atol=1e-9)


def test_line_pair_factors_back_into_lines():
    l1 = make_line_ipns(1, -1, 0)[1]
    l2 = make_line_ipns(1, 1, -2)[1]
    _, slp = make_line_pair(l1, l2)
    La, Lb = _lines_of(ipns_to_coeffs(slp))
    got = sorted([tuple(np.round(L / np.max(np.abs(L)), 3)) for L in (La, Lb)])
    exp = sorted([tuple(np.round(np.array([1.0, -1.0, 0.0]), 3)),
                  tuple(np.round(np.array([0.5, 0.5, -1.0]), 3))])
    assert got == exp


def test_flat_points_cannot_wedge_into_a_pair():
    # flat points build single lines, not pairs: flat ∧ flat = 0
    assert is_zero(make_flat_point(0, 0) ^ make_flat_point(1, 1))
    # a genuine (non-degenerate) conic is not degenerate, for contrast
    from ccga.objects import make_ellipse
    assert not conic_is_degenerate(_conic_vector(make_ellipse(3, 2)[0]))


def test_parallel_line_pair():
    # degenerate-hyperbola limit with two opposite ideal points -> parallel lines:
    # one through E,F, one through G parallel to EF
    import numpy as np
    from ccga.objects import make_parallel_line_pair
    from ccga.classify import ipns_to_coeffs, conic_discriminant, conic_is_degenerate
    from ccga.extract import _lines_of
    E, F, G = point(-0.2, 3.2), point(3.6, 2.8), point(5.3, -4.8)
    opns, ipns = make_parallel_line_pair(E, F, G)
    assert grades(opns) == [7]
    A, B, C, D, Ee, Fc = ipns_to_coeffs(ipns)
    assert abs(conic_discriminant(A, B, C)) < 1e-7      # Δ = 0 (parallel)
    assert conic_is_degenerate(ipns)                    # det M3 = 0

    La, Lb = _lines_of((A, B, C, D, Ee, Fc))
    def on(L, p):
        x, y = float((p | __import__('ccga').algebra.e1).e), float((p | __import__('ccga').algebra.e2).e)
        return abs(L[0]*x + L[1]*y + L[2]) / np.hypot(L[0], L[1])
    # one line carries E,F; the other carries G
    ef = La if on(La, E) < 1e-7 else Lb
    g = Lb if ef is La else La
    assert on(ef, E) < 1e-7 and on(ef, F) < 1e-7 and on(g, G) < 1e-7
    # the two lines are parallel
    assert abs(La[0]*Lb[1] - La[1]*Lb[0]) < 1e-9


def test_secant_line_pair_through_origin():
    # v = position of P -> secant pair: line(origin,P) + line(Q,R), Δ>0
    import numpy as np
    from ccga.objects import make_secant_line_pair_through_origin
    from ccga.classify import ipns_to_coeffs, conic_discriminant, conic_is_degenerate
    from ccga.extract import _lines_of
    from ccga.algebra import e1, e2
    P, Q, R = point(3.6, 2.8), point(9.2, 11.9), point(17.9, 8.7)
    opns, ipns = make_secant_line_pair_through_origin(P, Q, R)
    assert grades(opns) == [7]
    A, B, C, D, Ee, Fc = ipns_to_coeffs(ipns)
    assert conic_discriminant(A, B, C) > 1e-6           # Δ > 0 (crossing)
    assert conic_is_degenerate(ipns)                    # det M3 = 0
    La, Lb = _lines_of((A, B, C, D, Ee, Fc))

    def on(L, p):
        x, y = float((p | e1).e), float((p | e2).e)
        return abs(L[0]*x + L[1]*y + L[2]) / np.hypot(L[0], L[1])
    O = point(0, 0)
    # one line through origin & P; the other through Q & R
    op = La if (on(La, O) < 1e-6 and on(La, P) < 1e-6) else Lb
    qr = Lb if op is La else La
    assert on(op, O) < 1e-6 and on(op, P) < 1e-6
    assert on(qr, Q) < 1e-6 and on(qr, R) < 1e-6
    # secant (not parallel)
    assert abs(La[0]*Lb[1] - La[1]*Lb[0]) > 1e-9
