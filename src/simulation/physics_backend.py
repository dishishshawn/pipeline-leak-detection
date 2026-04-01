"""Physics simulator backend for the live dashboard.

Wraps PipelineSimulator (ODE-based transient flow) as a drop-in for
PipelineTelemetrySimulator, exposing the same interface the dashboard expects.

The ODE is run once per segment on initialisation (or when the buffer is
exhausted).  Output is pre-converted to dashboard-compatible column names so
the model prediction pipeline works identically for both backends.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.physics_sim.config import (
    BoundaryConfig,
    FluidConfig,
    LeakConfig,
    PipeConfig,
    SensorConfig,
    SimConfig,
    TemperatureConfig,
)
from src.physics_sim.solver import PipelineSimulator

logger = logging.getLogger(__name__)

# operating_state values produced by solver.measure() -> event_type mapping
_STATE_TO_EVENT: dict[str, str] = {
    "startup": "normal",
    "high_demand": "warning",
    "low_demand": "normal",
    "normal": "normal",
}

# Normalise leak_rate (m³/s) to 0-1 severity — 0.05 m³/s ≈ full 20 mm orifice
_LEAK_SEVERITY_SCALE = 0.05

# Simulation duration per ODE run (seconds)
_SIM_DURATION = 1800.0  # 30 minutes
# Sensor sampling interval — must match SensorConfig(sampling_interval=10.0)
_SAMPLING_INTERVAL = 10.0


# ---------------------------------------------------------------------------
# Duck-type helpers
# ---------------------------------------------------------------------------

@dataclass
class _DummyProfile:
    """Placeholder so len(config.segment_profiles) works like the real config."""
    segment_id: int


@dataclass
class _PhysicsBackendConfig:
    """Minimal config object that satisfies the dashboard's attribute access."""
    history_limit: int
    segment_profiles: list[_DummyProfile] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PhysicsSimulatorBackend:
    """Duck-typed simulator that drives PipelineSimulator ODE runs.

    Each pipeline segment gets its own independently seeded ODE run with
    slightly varied boundary conditions.  When the buffer runs out the
    simulation is re-run with a new seed and alternating leak presence,
    so the live view loops indefinitely with varied physics.

    Public interface mirrors PipelineTelemetrySimulator:
        start() / stop() / reset() / add_scenario() / advance_steps()
        .segment_ids (property)
        .step_index  (property)
        .config      (duck-typed _PhysicsBackendConfig)
    """

    def __init__(
        self,
        segment_count: int,
        history_limit: int,
        seed: int = 42,
    ) -> None:
        self._segment_count = segment_count
        self._history_limit = history_limit
        self._base_seed = seed
        self._seed_offset = 0

        profiles = [_DummyProfile(segment_id=i + 1) for i in range(segment_count)]
        self.config = _PhysicsBackendConfig(
            history_limit=history_limit,
            segment_profiles=profiles,
        )

        self._running = False
        self._cursor = 0
        self._step_index = 0
        self._elapsed_seconds = 0.0
        self._start_time = datetime(2026, 1, 1, 6, 0, 0)

        # Pre-run ODE for all segments (may raise RuntimeError on failure)
        self._buffers: list[pd.DataFrame] = []
        self._run_simulations()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, wall_clock: Any = None) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._cursor = 0
        self._step_index = 0
        self._seed_offset = 0
        self._elapsed_seconds = 0.0
        self._run_simulations()

    def add_scenario(self, scenario: Any) -> None:
        logger.warning(
            "PhysicsSimulatorBackend: add_scenario() is a no-op — "
            "manual scenario injection is not supported in physics mode."
        )

    def advance_steps(self, steps: int) -> pd.DataFrame:
        """Return the next `steps` rows per segment as a single DataFrame."""
        if not self._running or not self._buffers:
            return pd.DataFrame()

        buf_len = len(self._buffers[0])
        if self._cursor + steps > buf_len:
            # Buffer exhausted — re-run with a new seed before reading
            self._refill_buffers()

        end = min(self._cursor + steps, len(self._buffers[0]))
        chunks = [seg_df.iloc[self._cursor:end].copy() for seg_df in self._buffers]
        self._cursor = end
        self._step_index += steps

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    @property
    def segment_ids(self) -> list[int]:
        return list(range(1, self._segment_count + 1))

    @property
    def step_index(self) -> int:
        return self._step_index

    # ------------------------------------------------------------------
    # Internal ODE management
    # ------------------------------------------------------------------

    def _make_sim_config(self, segment_index: int, has_leak: bool) -> SimConfig:
        """Build a per-segment SimConfig with varied boundary conditions."""
        rng = random.Random(self._base_seed + self._seed_offset + segment_index * 100)

        # Vary inlet/outlet pressure ±15 % across segments for diversity
        p_inlet = 20.0e6 * (1.0 + rng.uniform(-0.15, 0.15))
        p_outlet = 8.0e6 * (1.0 + rng.uniform(-0.10, 0.10))
        demand_amp = rng.uniform(0.02, 0.08)

        leak_cfg: LeakConfig | None = None
        if has_leak:
            leak_cfg = LeakConfig(
                position=10_000.0 * rng.uniform(0.2, 0.8),
                max_orifice_diameter=rng.uniform(0.005, 0.020),
                discharge_coefficient=0.62,
                start_time=rng.uniform(300.0, 900.0),
                onset_type=rng.choice(["sudden", "ramp"]),
                onset_duration=rng.uniform(60.0, 180.0),
            )

        return SimConfig(
            pipe=PipeConfig(n_cells=10),
            fluid=FluidConfig(),
            boundary=BoundaryConfig(
                inlet_pressure=p_inlet,
                outlet_pressure=p_outlet,
                demand_amplitude=demand_amp,
            ),
            temperature=TemperatureConfig(),
            leak=leak_cfg,
            sensors=SensorConfig(sampling_interval=10.0),
            duration=_SIM_DURATION,
            seed=self._base_seed + self._seed_offset + segment_index,
        )

    def _run_one_segment(self, segment_index: int, has_leak: bool) -> pd.DataFrame:
        cfg = self._make_sim_config(segment_index, has_leak)
        sim = PipelineSimulator(cfg)
        raw = sim.run()
        physics_df = sim.measure(raw, scenario_id=f"seg{segment_index + 1}")
        current_start = self._start_time + timedelta(seconds=self._elapsed_seconds)
        return self._to_dashboard_format(physics_df, segment_id=segment_index + 1, start_time=current_start)

    def _run_simulations(self) -> None:
        """Run ODE for every segment. Odd cycles include a leak scenario."""
        has_leak = (self._seed_offset % 2) == 1
        self._buffers = []
        for i in range(self._segment_count):
            df = self._run_one_segment(i, has_leak=has_leak)
            self._buffers.append(df)
        self._cursor = 0

    def _refill_buffers(self) -> None:
        """Advance elapsed time, increment seed, re-run ODE.

        Add one extra sampling interval so the first timestamp of the new
        buffer does not collide with the last timestamp of the drained buffer.
        """
        self._elapsed_seconds += _SIM_DURATION + _SAMPLING_INTERVAL
        self._seed_offset += 1
        self._run_simulations()

    # ------------------------------------------------------------------
    # Column mapping: physics -> dashboard format
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dashboard_format(
        df: pd.DataFrame, segment_id: int, start_time: datetime
    ) -> pd.DataFrame:
        """Convert solver.measure() output to dashboard-compatible columns."""
        n = len(df)
        out = pd.DataFrame()

        # Timestamps — continuous from start_time
        out["timestamp"] = [
            start_time + timedelta(seconds=float(t)) for t in df["time"]
        ]
        out["segment_id"] = segment_id

        # Core SCADA columns expected by build_features()
        out["pressure"] = df["P_mid"].values
        out["flow_rate"] = df["Q_inlet"].values
        out["temperature"] = df["T_mid"].values  # measure() already converts K->C

        # Pass through full sensor columns for PhysicsModelWrapper
        for col in ("P_inlet", "P_mid", "P_outlet", "Q_inlet", "Q_outlet",
                    "T_inlet", "T_mid", "T_outlet"):
            if col in df.columns:
                out[col] = df[col].values

        # Categorical/label columns consumed by dashboard UI
        operating_state = df["operating_state"].values if "operating_state" in df.columns else np.full(n, "normal")
        out["event_type"] = [_STATE_TO_EVENT.get(str(s), "normal") for s in operating_state]

        leak_label = df["leak_label"].values if "leak_label" in df.columns else np.zeros(n)
        leak_rate = df["leak_rate"].values if "leak_rate" in df.columns else np.zeros(n)

        out["target"] = leak_label.astype(int)
        out["alarm_triggered"] = (leak_rate > 0.001).astype(int)
        out["scenario_context"] = np.where(leak_label == 1, "leak_progression", "nominal")
        out["leak_severity"] = np.clip(leak_rate / _LEAK_SEVERITY_SCALE, 0.0, 1.0)
        out["pump_efficiency"] = 1.0
        out["valve_status"] = 1
        out["pump_state"] = 1

        return out
