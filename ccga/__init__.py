"""CCGA — Conic Conformal Geometric Algebra in R^{5,3} (kingdon)."""

from .algebra import to_null_basis, format_null, print_null
from .point import point, point_at_infinity, tangent_at_infinity
from .objects import (
    make_conic_tripole, make_conic_quadpole,
    make_conic_pentapole, pentapole_to_conic, conic_dual_grade1,
    make_hyperbola_3points, make_parabola_3points, make_ellipse_3points,
    make_line_pair, make_parallel_line_pair, make_secant_line_pair_through_origin,
    polar_line, tangent_line, conic_from_5_tangents,
    normal_line, apollonius_conic,
)
from .classify import (
    conic_type, asymptotic_directions, conic_center, conic_center_point,
    conic_axes, conic_eccentricity, conic_foci, parabola_geometry,
    conic_is_degenerate,
)
from .transform import (
    apply_versor, translator, rotor, dilator, reflector,
    rotor_about, dilator_about, inversion, transversion,
)
from .extract import (
    circumcircle, tripole_circumconic, extract_tripole, pencil, extract_quadpole,
    conic_intersection, intersection_quadpole, intersection_points,
    intersection_reality, normal_feet, project_point_to_conic,
)

__all__ = [
    "to_null_basis", "format_null", "print_null",
    "point", "point_at_infinity", "tangent_at_infinity",
    "make_conic_tripole", "make_conic_quadpole",
    "make_conic_pentapole", "pentapole_to_conic", "conic_dual_grade1",
    "make_hyperbola_3points", "make_parabola_3points", "make_ellipse_3points",
    "make_line_pair", "make_parallel_line_pair",
    "make_secant_line_pair_through_origin",
    "conic_is_degenerate",
    "polar_line", "tangent_line", "conic_from_5_tangents",
    "normal_line", "apollonius_conic",
    "normal_feet", "project_point_to_conic",
    "conic_type", "asymptotic_directions", "conic_center", "conic_center_point",
    "conic_axes", "conic_eccentricity", "conic_foci", "parabola_geometry",
    "apply_versor", "translator", "rotor", "dilator", "reflector",
    "rotor_about", "dilator_about", "inversion", "transversion",
    "circumcircle", "tripole_circumconic", "extract_tripole", "pencil", "extract_quadpole",
    "conic_intersection", "intersection_quadpole", "intersection_points",
    "intersection_reality",
]
