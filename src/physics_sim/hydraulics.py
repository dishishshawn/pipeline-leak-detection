"""Pure physics functions: friction, Reynolds number, orifice discharge.

Every function is stateless and operates on scalars or numpy arrays.
No simulation state or config objects are referenced here.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Reynolds number
# ---------------------------------------------------------------------------

def reynolds_number(
    Q: np.ndarray | float,
    D: float,
    A: float,
    rho: float,
    mu: float,
) -> np.ndarray | float:
    """Re = rho * v * D / mu   where  v = |Q| / A."""
    return rho * np.abs(Q) * D / (mu * A)


# ---------------------------------------------------------------------------
# Darcy friction factor
# ---------------------------------------------------------------------------

def friction_factor(Re: np.ndarray | float, rel_roughness: float) -> np.ndarray | float:
    """Darcy friction factor covering laminar, transition, and turbulent flow.

    Laminar  (Re < 2 300):  f = 64 / Re
    Turbulent (Re >= 4 000): Swamee-Jain explicit approximation
    Transition:               linear blend
    """
    scalar = np.ndim(Re) == 0
    Re = np.atleast_1d(np.asarray(Re, dtype=np.float64))
    f = np.zeros_like(Re)

    # Laminar
    lam = (Re > 1.0) & (Re < 2300.0)
    f[lam] = 64.0 / Re[lam]

    # Turbulent (Swamee-Jain)
    turb = Re >= 4000.0
    if np.any(turb):
        f[turb] = 0.25 / np.log10(
            rel_roughness / 3.7 + 5.74 / Re[turb] ** 0.9
        ) ** 2

    # Transition — linear blend between laminar at Re=2300 and turbulent at Re=4000
    trans = (Re >= 2300.0) & (Re < 4000.0)
    if np.any(trans):
        f_lam = 64.0 / Re[trans]
        f_turb_ref = 0.25 / np.log10(
            rel_roughness / 3.7 + 5.74 / 4000.0 ** 0.9
        ) ** 2
        alpha = (Re[trans] - 2300.0) / 1700.0
        f[trans] = f_lam * (1.0 - alpha) + f_turb_ref * alpha

    return f.item() if scalar else f


# ---------------------------------------------------------------------------
# Orifice leak discharge
# ---------------------------------------------------------------------------

def orifice_flow(
    P_internal: float,
    P_external: float,
    Cd: float,
    A_orifice: float,
    rho: float,
) -> float:
    """Volumetric leak flow through a sharp-edged orifice (m^3/s).

    Q = Cd * A * sqrt(2 * dP / rho)

    Returns zero for non-positive differential pressure.
    """
    dP = P_internal - P_external
    if dP <= 0.0:
        return 0.0
    return Cd * A_orifice * np.sqrt(2.0 * dP / rho)


# ---------------------------------------------------------------------------
# Steady-state flow (for initial conditions)
# ---------------------------------------------------------------------------

def steady_state_flow(
    P_in: float,
    P_out: float,
    pipe,
    fluid,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """Iterative Darcy-Weisbach solution for steady uniform flow.

    Solves:  dP = f(Re) * (L/D) * rho * Q^2 / (2 * A^2)
    """
    dP = P_in - P_out
    if dP <= 0.0:
        return 0.0

    f = 0.02  # initial guess
    for _ in range(max_iter):
        Q = pipe.area * np.sqrt(
            2.0 * pipe.diameter * dP / (f * pipe.length * fluid.density)
        )
        Re = reynolds_number(Q, pipe.diameter, pipe.area, fluid.density, fluid.viscosity)
        f_new = friction_factor(Re, pipe.roughness / pipe.diameter)
        if isinstance(f_new, np.ndarray):
            f_new = f_new.item()
        if abs(f_new - f) < tol:
            break
        f = f_new
    return float(Q)
