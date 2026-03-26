"""Physics-grounded pipeline transient flow simulator.

Solves 1D mass and momentum conservation on a staggered grid with
Darcy-Weisbach friction and orifice-style leak modeling.  Designed to
generate realistic SCADA-like sensor datasets for ML leak detection.
"""

from src.physics_sim.config import (
    BoundaryConfig,
    FluidConfig,
    FLUID_PRESETS,
    LeakConfig,
    PipeConfig,
    SensorConfig,
    SimConfig,
    TemperatureConfig,
)
from src.physics_sim.solver import PipelineSimulator
from src.physics_sim.scenarios import ScenarioSpec, generate_batch
from src.physics_sim.export import export_scada_csv, export_metadata, export_long_format
from src.physics_sim.validate import validate_physics, print_validation
from src.physics_sim.plots import plot_pressure_profiles, plot_timeseries_comparison

__all__ = [
    "BoundaryConfig",
    "FluidConfig",
    "FLUID_PRESETS",
    "LeakConfig",
    "PipeConfig",
    "PipelineSimulator",
    "ScenarioSpec",
    "SensorConfig",
    "SimConfig",
    "TemperatureConfig",
    "export_long_format",
    "export_metadata",
    "export_scada_csv",
    "generate_batch",
    "plot_pressure_profiles",
    "plot_timeseries_comparison",
    "print_validation",
    "validate_physics",
]
