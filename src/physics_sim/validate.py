"""Physics validation and sanity checks.

Each check returns a dict with ``name``, ``passed`` (bool), and ``detail``
(human-readable string).  The suite verifies that the numerical solution
is physically consistent: pressures decrease along the pipe, mass is
conserved, leaks create the expected flow imbalance, friction factor
responds to roughness, and no unphysical values appear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.physics_sim.config import SimConfig
from src.physics_sim.hydraulics import friction_factor


def validate_physics(
    config: SimConfig,
    raw: dict,
    measured: pd.DataFrame | None = None,
) -> list[dict]:
    """Run all physics checks against a single simulation result."""
    checks: list[dict] = []
    P = raw["P"]  # (N, n_times)
    Q = raw["Q"]  # (N+1, n_times)
    t = raw["t"]

    # 1. Pressure monotonically decreasing at t=0 (before any leak)
    P_t0 = P[:, 0]
    mono = bool(np.all(np.diff(P_t0) <= 0.0))
    checks.append({
        "name": "pressure_decreasing_initial",
        "passed": mono,
        "detail": f"P(t=0) from {P_t0[0]:.0f} to {P_t0[-1]:.0f} Pa",
    })

    # 2. No NaN / Inf in solution
    finite = bool(np.all(np.isfinite(P)) and np.all(np.isfinite(Q)))
    checks.append({
        "name": "no_nan_inf",
        "passed": finite,
        "detail": "All values finite" if finite else "Found NaN/Inf",
    })

    # 3. Initial mass balance (Q_in ≈ Q_out before leak)
    Q_in_0 = float(Q[0, 0])
    Q_out_0 = float(Q[-1, 0])
    err = abs(Q_in_0 - Q_out_0) / max(abs(Q_in_0), 1e-10)
    checks.append({
        "name": "mass_balance_initial",
        "passed": err < 0.02,
        "detail": f"Q_in={Q_in_0:.6f}, Q_out={Q_out_0:.6f}, rel_err={err:.2e}",
    })

    # 4. Leak causes Q_in > Q_out (after leak stabilises)
    if config.leak is not None:
        t_stable = config.leak.start_time + max(config.leak.onset_duration, 60.0) + 60.0
        idx = int(np.searchsorted(t, t_stable))
        if idx < len(t):
            Q_in_l = float(Q[0, idx])
            Q_out_l = float(Q[-1, idx])
            checks.append({
                "name": "leak_flow_imbalance",
                "passed": Q_in_l > Q_out_l * 1.001,
                "detail": f"Q_in={Q_in_l:.6f} > Q_out={Q_out_l:.6f}",
            })

    # 5. Pressure range physically reasonable
    P_min, P_max = float(P.min()), float(P.max())
    ok = P_min > 0.0 and P_max < 20.0e6
    checks.append({
        "name": "pressure_range",
        "passed": ok,
        "detail": f"[{P_min:.0f}, {P_max:.0f}] Pa",
    })

    # 6. Flow non-negative (normal direction)
    Q_min = float(Q.min())
    checks.append({
        "name": "flow_non_negative",
        "passed": Q_min >= -1e-4,
        "detail": f"Q_min = {Q_min:.6f} m³/s",
    })

    # 7. Roughness increases friction factor (direct physics check)
    Re_test = 100_000.0
    f_smooth = float(friction_factor(Re_test, 1e-6))
    f_rough = float(friction_factor(Re_test, 1e-3))
    checks.append({
        "name": "roughness_increases_friction",
        "passed": f_rough > f_smooth,
        "detail": f"f(e/D=1e-6)={f_smooth:.5f}, f(e/D=1e-3)={f_rough:.5f}",
    })

    # 8. No high-frequency numerical oscillation in pressure
    if P.shape[1] > 20:
        # Check last cell (most prone to outlet-boundary reflections)
        P_last = P[-1, -20:]
        diff2 = np.diff(P_last, 2)
        osc_ratio = float(np.std(diff2) / max(np.std(P_last), 1.0))
        checks.append({
            "name": "no_numerical_oscillation",
            "passed": osc_ratio < 0.5,
            "detail": f"2nd-diff std / signal std = {osc_ratio:.4f}",
        })

    return checks


def print_validation(checks: list[dict]) -> None:
    """Pretty-print validation results to stdout."""
    for c in checks:
        tag = "PASS" if c["passed"] else "FAIL"
        print(f"  [{tag}] {c['name']}: {c['detail']}")
