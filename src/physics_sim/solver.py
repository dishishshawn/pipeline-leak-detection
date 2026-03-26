"""Transient pipeline flow solver.

Assembles and integrates the coupled ODE system:

  Continuity (per cell):   dP_i/dt = (B_eff / V_cell) * (Q_i - Q_{i+1} - Q_leak_i)
  Momentum  (per face):    dQ_j/dt = (A / (rho*dx_j)) * (P_up - P_down)
                                    - f * Q_j * |Q_j| / (2*D*A)

State vector y = [P_0 .. P_{N-1},  Q_0 .. Q_N]
   N pressure cells, N+1 flow faces (including inlet and outlet).

Boundary conditions: fixed inlet pressure, time-varying outlet pressure
(demand cycle).  The Radau solver handles the stiffness introduced by
the bulk modulus coupling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.physics_sim.config import SimConfig
from src.physics_sim.hydraulics import (
    friction_factor,
    orifice_flow,
    reynolds_number,
    steady_state_flow,
)

P_ATM = 101_325.0  # Pa — atmospheric back-pressure for leak orifice


# ---------------------------------------------------------------------------
# First-order sensor lag (discrete IIR filter)
# ---------------------------------------------------------------------------

def _first_order_lag(signal: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """Causal exponential smoothing: y[k] = y[k-1] + alpha*(x[k]-y[k-1])."""
    if tau <= 0.0 or dt <= 0.0:
        return signal.copy()
    alpha = dt / (tau + dt)
    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class PipelineSimulator:
    """Physics-based transient pipeline simulator.

    Usage::

        sim = PipelineSimulator(config)
        raw = sim.run()                       # solve ODE
        df  = sim.measure(raw, "scenario_1")  # apply sensor model → DataFrame
    """

    def __init__(self, config: SimConfig) -> None:
        self.cfg = config
        pipe = config.pipe
        fluid = config.fluid

        N = pipe.n_cells
        A = pipe.area
        dx = pipe.dx
        rho = fluid.density
        B_eff = config.effective_bulk_modulus

        # Cache integers / floats used in the hot ODE RHS
        self._N = N
        self._A = A
        self._D = pipe.diameter
        self._rho = rho
        self._mu = fluid.viscosity
        self._dx = dx
        self._rel_rough = pipe.roughness / pipe.diameter

        # Pre-computed coefficients -----------------------------------------------
        # Continuity:  dP/dt = coeff_P * (Q_in - Q_out - Q_leak)
        self._coeff_P = B_eff / (A * dx)

        # Momentum: different effective length for boundary vs internal faces
        dx_face = np.full(N + 1, dx)
        dx_face[0] = dx / 2.0
        dx_face[-1] = dx / 2.0
        self._coeff_Q = A / (rho * dx_face)          # shape (N+1,)
        self._coeff_fric = 1.0 / (2.0 * pipe.diameter * A)  # scalar

        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def initial_conditions(self) -> np.ndarray:
        """Steady-state pressure profile + uniform flow."""
        bc = self.cfg.boundary
        pipe = self.cfg.pipe
        fluid = self.cfg.fluid
        N = self._N

        Q_ss = steady_state_flow(bc.inlet_pressure, bc.outlet_pressure, pipe, fluid)
        x = pipe.cell_centers
        P = bc.inlet_pressure - (bc.inlet_pressure - bc.outlet_pressure) * (x / pipe.length)
        Q = np.full(N + 1, Q_ss)
        return np.concatenate([P, Q])

    # ------------------------------------------------------------------
    # Leak model
    # ------------------------------------------------------------------

    def _leak_flow_at_cell(self, t: float, cell_idx: int, P_cell: float) -> float:
        leak = self.cfg.leak
        if leak is None or t < leak.start_time:
            return 0.0

        # Does this cell contain the leak?
        dx = self._dx
        if not (cell_idx * dx <= leak.position < (cell_idx + 1) * dx):
            return 0.0

        dt_leak = t - leak.start_time
        if leak.onset_type == "sudden":
            opening = 1.0
        elif leak.onset_type == "ramp":
            opening = min(1.0, dt_leak / max(leak.onset_duration, 1e-6))
        else:  # exponential
            opening = 1.0 - np.exp(-dt_leak / max(leak.onset_duration, 1e-6))

        A_orifice = leak.max_orifice_area * opening
        return orifice_flow(P_cell, P_ATM, leak.discharge_coefficient, A_orifice, self._rho)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _bc_pressures(self, t: float) -> tuple[float, float]:
        bc = self.cfg.boundary
        demand = 1.0 + bc.demand_amplitude * np.sin(2.0 * np.pi * t / bc.demand_period)
        P_out = bc.outlet_pressure * (2.0 - demand)
        return bc.inlet_pressure, P_out

    # ------------------------------------------------------------------
    # ODE right-hand side (hot path)
    # ------------------------------------------------------------------

    def _ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        N = self._N
        P = y[:N]
        Q = y[N:]

        P_in, P_out = self._bc_pressures(t)

        # Leak discharge per cell
        Q_leak = np.array([self._leak_flow_at_cell(t, i, P[i]) for i in range(N)])

        # --- Continuity ---
        dPdt = self._coeff_P * (Q[:-1] - Q[1:] - Q_leak)

        # --- Momentum ---
        Re = reynolds_number(Q, self._D, self._A, self._rho, self._mu)
        f = friction_factor(Re, self._rel_rough)
        if np.ndim(f) == 0:
            f = np.full_like(Q, float(f))

        # Extended pressure array: [P_in, P_0 .. P_{N-1}, P_out]
        P_ext = np.empty(N + 2)
        P_ext[0] = P_in
        P_ext[1 : N + 1] = P
        P_ext[N + 1] = P_out
        dP_face = P_ext[:-1] - P_ext[1:]  # upstream - downstream for each face

        dQdt = self._coeff_Q * dP_face - f * Q * np.abs(Q) * self._coeff_fric

        return np.concatenate([dPdt, dQdt])

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Integrate the ODE system and return raw arrays."""
        y0 = self.initial_conditions()
        dt_out = self.cfg.sensors.sampling_interval
        t_eval = np.arange(0.0, self.cfg.duration + dt_out * 0.1, dt_out)
        t_eval = t_eval[t_eval <= self.cfg.duration]

        sol = solve_ivp(
            self._ode_rhs,
            [0.0, self.cfg.duration],
            y0,
            method=self.cfg.solver_method,
            t_eval=t_eval,
            rtol=self.cfg.solver_rtol,
            atol=self.cfg.solver_atol,
            max_step=min(10.0, dt_out),
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        N = self._N
        raw: dict = {
            "t": sol.t,
            "P": sol.y[:N],                 # (N, n_times)
            "Q": sol.y[N:],                  # (N+1, n_times)
            "n_cells": N,
            "cell_centers": self.cfg.pipe.cell_centers,
            "face_positions": self.cfg.pipe.face_positions,
        }

        # Leak rate time series
        if self.cfg.leak is not None:
            leak_cell = min(int(self.cfg.leak.position / self._dx), N - 1)
            leak_rates = np.array([
                self._leak_flow_at_cell(t, leak_cell, raw["P"][leak_cell, i])
                for i, t in enumerate(sol.t)
            ])
        else:
            leak_rates = np.zeros_like(sol.t)
        raw["leak_rate"] = leak_rates

        return raw

    # ------------------------------------------------------------------
    # Sensor / measurement model
    # ------------------------------------------------------------------

    def measure(self, raw: dict, scenario_id: str = "run_0") -> pd.DataFrame:
        """Apply sensor model to raw solution -> SCADA-like DataFrame.

        Interpolates physics at sensor positions, adds Gaussian noise and
        first-order lag, then appends labels and operating-state tags.
        """
        t = raw["t"]
        P_cells = raw["P"]       # (N, n_times)
        Q_faces = raw["Q"]       # (N+1, n_times)
        leak_rates = raw["leak_rate"]
        n_t = len(t)

        cell_x = raw["cell_centers"]
        face_x = raw["face_positions"]
        scfg = self.cfg.sensors
        tcfg = self.cfg.temperature
        rng = self._rng
        dt = scfg.sampling_interval

        records: dict[str, np.ndarray] = {"time": t, "scenario_id": np.full(n_t, scenario_id)}

        # -- Pressure at each sensor ----------------------------------------
        for pos, label in zip(scfg.positions, scfg.labels):
            P_true = np.array([np.interp(pos, cell_x, P_cells[:, ti]) for ti in range(n_t)])
            noise = rng.normal(0.0, scfg.pressure_noise_std, n_t)
            records[f"P_{label}"] = _first_order_lag(P_true + noise, scfg.lag_time_constant, dt)

        # -- Flow at inlet and outlet ----------------------------------------
        Q_in_true = Q_faces[0, :]
        Q_out_true = Q_faces[-1, :]
        records["Q_inlet"] = _first_order_lag(
            Q_in_true + rng.normal(0.0, scfg.flow_noise_std, n_t),
            scfg.lag_time_constant, dt,
        )
        records["Q_outlet"] = _first_order_lag(
            Q_out_true + rng.normal(0.0, scfg.flow_noise_std, n_t),
            scfg.lag_time_constant, dt,
        )

        # -- Temperature (decoupled, flow-dependent decay length) ------------
        Q_ref = max(float(Q_faces[0, 0]), 1e-10)
        for pos, label in zip(scfg.positions, scfg.labels):
            # Local flow at sensor position for each timestep
            Q_local = np.array([np.interp(pos, face_x, Q_faces[:, ti]) for ti in range(n_t)])
            L_eff = tcfg.thermal_decay_length * np.clip(Q_local / Q_ref, 0.1, 2.0)
            T_true = tcfg.ground_temperature + (
                (tcfg.inlet_temperature - tcfg.ground_temperature) * np.exp(-pos / L_eff)
            )
            noise = rng.normal(0.0, scfg.temperature_noise_std, n_t)
            records[f"T_{label}"] = _first_order_lag(T_true + noise, scfg.lag_time_constant, dt)

        # -- Labels ----------------------------------------------------------
        records["leak_rate"] = leak_rates
        records["leak_label"] = (leak_rates > 0.0).astype(int)
        records["leak_position"] = np.full(n_t, self.cfg.leak.position if self.cfg.leak else np.nan)

        # -- Operating state -------------------------------------------------
        phase = 2.0 * np.pi * t / self.cfg.boundary.demand_period
        demand = 1.0 + self.cfg.boundary.demand_amplitude * np.sin(phase)
        state = np.where(
            t < 60.0, "startup",
            np.where(demand > 1.05, "high_demand",
            np.where(demand < 0.95, "low_demand", "normal")),
        )
        records["operating_state"] = state

        return pd.DataFrame(records)
