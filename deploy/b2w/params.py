"""Declaring ROS 2 parameters that accept ``1`` where ``1.0`` is expected (ROS 2 Galactic and later otherwise refuse
an integer for a parameter declared with a float default), on Foxy as on Humble."""

from __future__ import annotations


def declare(node, name: str, default):
    """Declare ``name`` with ``default`` and return its value, converted to the default's type."""
    descriptor = None
    try:
        from rcl_interfaces.msg import ParameterDescriptor

        descriptor = ParameterDescriptor(dynamic_typing=True)
    except Exception:  # noqa: BLE001 (Foxy: no dynamic typing, and no type enforcement either)
        pass
    value = (node.declare_parameter(name, default, descriptor) if descriptor is not None
             else node.declare_parameter(name, default)).value
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
    if isinstance(default, float):
        return float(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, list) and default and isinstance(default[0], float):
        return [float(v) for v in value]
    return value
