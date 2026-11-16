"""Canonical validation and prompt formatting for loop-response descriptors."""
from collections.abc import Mapping, Sequence
import math


MECHANISM_RESPONSE_FIELDS = ("loop_release_excursion", "loop_release_progress")
MECHANISM_RESPONSE_SCHEMA = "detent-bottle-mechanism-response-v3"


def validate_mechanism_response(value):
    """Return the two normalized response values in their canonical field order.

    ``value`` may be a mapping keyed by :data:`MECHANISM_RESPONSE_FIELDS` or a
    length-two sequence in that order. Booleans, non-finite values, missing or
    extra mapping keys, and values outside [0, 1] are rejected.
    """
    if isinstance(value, Mapping):
        keys = set(value)
        expected = set(MECHANISM_RESPONSE_FIELDS)
        if keys != expected:
            raise ValueError(
                f"mechanism_response fields must be exactly {MECHANISM_RESPONSE_FIELDS}; "
                f"got {tuple(sorted(keys))}"
            )
        raw_values = [value[field] for field in MECHANISM_RESPONSE_FIELDS]
    elif isinstance(value, Sequence) or hasattr(value, "shape"):
        try:
            raw_values = list(value)
        except TypeError as exc:
            raise ValueError("mechanism_response must contain two normalized values") from exc
        if len(raw_values) != len(MECHANISM_RESPONSE_FIELDS):
            raise ValueError("mechanism_response must contain [loop_release_excursion, loop_release_progress]")
    else:
        raise ValueError("mechanism_response must be a mapping or length-two sequence")

    normalized = []
    for field, raw in zip(MECHANISM_RESPONSE_FIELDS, raw_values, strict=True):
        if isinstance(raw, bool):
            raise ValueError(f"mechanism_response.{field} must be numeric, not bool")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mechanism_response.{field} must be numeric") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"mechanism_response.{field} must be finite and within [0, 1]")
        normalized.append(number)
    return tuple(normalized)


def format_mechanism_response(value):
    """Format a validated descriptor as stable, concise task-prompt text."""
    peak, residual = validate_mechanism_response(value)
    return (
        "Mechanism response (values normalized from 0 to 1): "
        f"loop release excursion {peak:.4f}; return fraction after one tick {residual:.4f}."
    )


def format_task_prompt(prompt, mechanism_response=None):
    """Return the base task prompt, optionally conditioned on loop response."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Task prompt must be a nonempty string")
    if mechanism_response is None:
        return prompt
    return f"{prompt.strip()} {format_mechanism_response(mechanism_response)}"
