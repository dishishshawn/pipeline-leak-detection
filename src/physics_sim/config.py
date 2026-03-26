"""Parameter dataclasses for the physics pipeline simulator.

Every physical constant has a documented default with units.  Pipe wall
elasticity is folded into an *effective* bulk modulus so the transient
model captures realistic pressure wave speeds without requiring a
separate structural sub-model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Pipe geometry
# ---------------------------------------------------------------------------

@dataclass
class PipeConfig:
    """Pipeline segment geometry and wall properties."""

    length: float = 10_000.0        # m  (10 km)
    diameter: float = 0.3048        # m  (12 in)
    roughness: float = 4.6e-5       # m  (commercial steel)
    wall_thickness: float = 0.0095  # m  (~3/8 in)
    n_cells: int = 10               # spatial discretisation cells

    @property
    def area(self) -> float:
        """Internal cross-section area (m^2)."""
        return np.pi / 4 * self.diameter ** 2

    @property
    def dx(self) -> float:
        """Cell length (m)."""
        return self.length / self.n_cells

    @property
    def cell_centers(self) -> np.ndarray:
        """Position of each cell centre (m from inlet)."""
        return np.linspace(self.dx / 2, self.length - self.dx / 2, self.n_cells)

    @property
    def face_positions(self) -> np.ndarray:
        """Position of each cell face including inlet/outlet (m)."""
        return np.linspace(0.0, self.length, self.n_cells + 1)


# ---------------------------------------------------------------------------
# Fluid properties
# ---------------------------------------------------------------------------

@dataclass
class FluidConfig:
    """Single-phase fluid properties (assumed constant for Stage 1)."""

    name: str = "crude_oil"
    density: float = 850.0          # kg/m^3
    viscosity: float = 5.0e-3       # Pa.s  (5 cP, medium crude)
    bulk_modulus: float = 1.5e9     # Pa
    specific_heat: float = 2000.0   # J/(kg.K)


FLUID_PRESETS: dict[str, FluidConfig] = {
    "crude_oil": FluidConfig("crude_oil", 850.0, 5.0e-3, 1.5e9, 2000.0),
    "water":     FluidConfig("water",     998.0, 1.0e-3, 2.2e9, 4186.0),
    "diesel":    FluidConfig("diesel",    832.0, 2.5e-3, 1.4e9, 2100.0),
    "gasoline":  FluidConfig("gasoline",  720.0, 0.6e-3, 1.1e9, 2220.0),
}


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

@dataclass
class BoundaryConfig:
    """Inlet / outlet pressure boundaries with optional demand variation.

    ``inlet_pressure`` represents a pumping-station discharge.
    ``outlet_pressure`` represents a delivery or tank-farm back-pressure.
    Demand variation is modelled as a sinusoidal perturbation on the
    outlet pressure (higher demand -> lower outlet pressure).
    """

    inlet_pressure: float = 5.0e6   # Pa  (50 bar / ~725 psi)
    outlet_pressure: float = 2.0e6  # Pa  (20 bar / ~290 psi)
    demand_amplitude: float = 0.05  # fraction of outlet pressure
    demand_period: float = 3600.0   # s   (1-hour cycle)


# ---------------------------------------------------------------------------
# Leak specification
# ---------------------------------------------------------------------------

@dataclass
class LeakConfig:
    """Orifice-style leak with configurable onset behaviour."""

    position: float = 5000.0                     # m from inlet
    max_orifice_diameter: float = 0.010          # m  (10 mm)
    discharge_coefficient: float = 0.62          # sharp-edged orifice
    start_time: float = 600.0                    # s
    onset_type: Literal["sudden", "ramp", "exponential"] = "ramp"
    onset_duration: float = 120.0                # s  (for ramp / exponential)

    @property
    def max_orifice_area(self) -> float:
        return np.pi / 4 * self.max_orifice_diameter ** 2


# ---------------------------------------------------------------------------
# Sensor / measurement model
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """SCADA sensor placement and noise characteristics."""

    positions: list[float] = field(
        default_factory=lambda: [0.0, 5000.0, 10000.0]
    )
    labels: list[str] = field(
        default_factory=lambda: ["inlet", "mid", "outlet"]
    )
    pressure_noise_std: float = 5_000.0   # Pa  (~0.05 bar)
    flow_noise_std: float = 0.000_5       # m^3/s
    temperature_noise_std: float = 0.1    # K
    sampling_interval: float = 5.0        # s
    lag_time_constant: float = 2.0        # s  (first-order sensor lag)


# ---------------------------------------------------------------------------
# Temperature (post-processed, decoupled from hydraulics)
# ---------------------------------------------------------------------------

@dataclass
class TemperatureConfig:
    """Simplified thermal model: exponential decay from inlet to ground temp.

    ``thermal_decay_length`` is proportional to mass flow.  When flow
    drops (leak), L_eff shrinks and downstream temperatures fall —
    producing a physically meaningful detection signal.
    """

    inlet_temperature: float = 313.15       # K  (40 deg-C)
    ground_temperature: float = 288.15      # K  (15 deg-C)
    thermal_decay_length: float = 20_000.0  # m  (at nominal flow)


# ---------------------------------------------------------------------------
# Top-level simulation config
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Assembles all sub-configs for a single simulation run."""

    pipe: PipeConfig = field(default_factory=PipeConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    leak: LeakConfig | None = None
    sensors: SensorConfig = field(default_factory=SensorConfig)
    duration: float = 1800.0       # s  (30 min)
    solver_method: str = "Radau"   # implicit, A-stable — handles stiff bulk-modulus dynamics
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-8
    seed: int = 42

    @property
    def effective_bulk_modulus(self) -> float:
        """Combined fluid + pipe-wall compressibility.

        1/B_eff = 1/B_fluid + D / (E_steel * t_wall)

        For a 12-in steel pipe with medium crude this gives ~1.2 GPa,
        yielding a wave speed of ~1 190 m/s — close to the 1 000-1 400
        m/s range reported for real oil pipelines.
        """
        E_steel = 200.0e9  # Pa
        inv_fluid = 1.0 / self.fluid.bulk_modulus
        inv_wall = self.pipe.diameter / (E_steel * self.pipe.wall_thickness)
        return 1.0 / (inv_fluid + inv_wall)
