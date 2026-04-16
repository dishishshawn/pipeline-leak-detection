"""Custom Streamlit component for zero-flicker live simulator rendering."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"
_component_func = components.declare_component("live_simulator", path=str(_FRONTEND_DIR))


def live_simulator_component(data: dict, key: str = "live_sim"):
    """Render the live simulator view using the custom JS component.

    Returns the latest value posted from the frontend (or None).
    """
    return _component_func(data=data, key=key, default=None)
