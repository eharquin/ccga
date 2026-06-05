"""CCGA — Conic Conformal Geometric Algebra in R^{5,3} (kingdon)."""

from .algebra import to_null_basis, format_null, print_null
from .objects import make_conic_tripole, make_conic_quadpole
from .extract import (
    circumcircle, extract_tripole, pencil, extract_quadpole,
)

__all__ = [
    "to_null_basis", "format_null", "print_null",
    "make_conic_tripole", "make_conic_quadpole",
    "circumcircle", "extract_tripole", "pencil", "extract_quadpole",
]
