#!/usr/bin/env python
"""Generate physics-based pipeline leak detection dataset.

Runs a batch of normal and leak scenarios through the transient
pipeline simulator, applies sensor noise, and exports:

  data/physics_sim/scada_timeseries.csv      (wide-format SCADA)
  data/physics_sim/scenario_metadata.csv      (one row per scenario)
  data/physics_sim/plots/*.png                (comparison plots)

Usage:
    python scripts/generate_physics_dataset.py
    python scripts/generate_physics_dataset.py --n-normal 50 --n-leak 50 --duration 3600
    python scripts/generate_physics_dataset.py --long-format --parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as `python scripts/generate_physics_dataset.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.physics_sim.config import SimConfig, PipeConfig, FluidConfig
from src.physics_sim.solver import PipelineSimulator
from src.physics_sim.scenarios import generate_batch
from src.physics_sim.export import export_scada_csv, export_scada_parquet, export_metadata, export_long_format
from src.physics_sim.validate import validate_physics, print_validation
from src.physics_sim.plots import plot_pressure_profiles, plot_timeseries_comparison, plot_temperature_comparison

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate physics-based pipeline dataset")
    ap.add_argument("--n-normal", type=int, default=25, help="Number of normal scenarios")
    ap.add_argument("--n-leak", type=int, default=25, help="Number of leak scenarios")
    ap.add_argument("--duration", type=float, default=1800.0, help="Scenario duration in seconds")
    ap.add_argument("--output", type=str, default="data/physics_sim", help="Output directory")
    ap.add_argument("--no-validate", action="store_true", help="Skip validation checks")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    ap.add_argument("--long-format", action="store_true", help="Also export long-format CSV")
    ap.add_argument("--parquet", action="store_true", help="Also export Parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # -- Generate scenario specs ------------------------------------------
    specs = generate_batch(
        n_normal=args.n_normal,
        n_leak=args.n_leak,
        duration=args.duration,
        seed=args.seed,
    )
    log.info("Generated %d scenario specs (%d normal, %d leak)",
             len(specs), args.n_normal, args.n_leak)

    # -- Run simulations --------------------------------------------------
    all_dfs: list[pd.DataFrame] = []
    first_normal_raw = first_normal_df = first_normal_cfg = None
    first_leak_raw = first_leak_df = first_leak_cfg = None

    # Validation tracking
    do_validate = not args.no_validate
    CRITICAL_CHECKS = {"no_nan_inf", "pressure_range", "flow_non_negative"}
    validation_pass = 0
    validation_warn = 0
    validation_fail = 0
    failed_scenarios: list[str] = []

    t_batch_start = time.time()
    for i, spec in enumerate(specs):
        t0 = time.time()
        try:
            sim = PipelineSimulator(spec.config)
            raw = sim.run()
            df = sim.measure(raw, scenario_id=spec.scenario_id)
        except Exception as exc:
            log.warning("  [%d/%d] %s FAILED: %s", i + 1, len(specs), spec.scenario_id, exc)
            validation_fail += 1
            failed_scenarios.append(f"{spec.scenario_id}: simulation error — {exc}")
            continue

        elapsed = time.time() - t0
        tag = "LEAK" if spec.leak_exists else "NORM"

        # -- Per-scenario validation -------------------------------------------
        if do_validate:
            checks = validate_physics(spec.config, raw, df)
            critical_failures = [c for c in checks if not c["passed"] and c["name"] in CRITICAL_CHECKS]
            warnings = [c for c in checks if not c["passed"] and c["name"] not in CRITICAL_CHECKS]

            if critical_failures:
                validation_fail += 1
                for c in critical_failures:
                    log.error("  [%d/%d] %s CRITICAL FAIL: %s — %s",
                              i + 1, len(specs), spec.scenario_id, c["name"], c["detail"])
                failed_scenarios.append(
                    f"{spec.scenario_id}: {', '.join(c['name'] for c in critical_failures)}")
                # Skip this scenario — data is untrustworthy
                continue

            if warnings:
                validation_warn += 1
                for c in warnings:
                    log.warning("  [%d/%d] %s WARN: %s — %s",
                                i + 1, len(specs), spec.scenario_id, c["name"], c["detail"])
            else:
                validation_pass += 1

        all_dfs.append(df)
        log.info("  [%d/%d] %s (%s) — %.1fs, %d rows",
                 i + 1, len(specs), spec.scenario_id, tag, elapsed, len(df))

        # Capture first of each type for plots
        if not spec.leak_exists and first_normal_raw is None:
            first_normal_raw, first_normal_df, first_normal_cfg = raw, df, spec.config
        if spec.leak_exists and first_leak_raw is None:
            first_leak_raw, first_leak_df, first_leak_cfg = raw, df, spec.config

    if not all_dfs:
        log.error("All scenarios failed — no output generated.")
        return

    t_batch = time.time() - t_batch_start
    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Batch complete: %d scenarios, %d rows, %.1fs total",
             len(all_dfs), len(combined), t_batch)

    # -- Export ------------------------------------------------------------
    export_scada_csv(combined, out / "scada_timeseries.csv")
    export_metadata(specs, out / "scenario_metadata.csv")
    log.info("Exported wide CSV → %s", out / "scada_timeseries.csv")

    if args.parquet:
        export_scada_parquet(combined, out / "scada_timeseries.parquet")
        log.info("Exported Parquet → %s", out / "scada_timeseries.parquet")

    if args.long_format:
        export_long_format(combined, out / "sensor_long_format.csv")
        log.info("Exported long CSV → %s", out / "sensor_long_format.csv")

    # -- Validation summary ------------------------------------------------
    if do_validate:
        log.info("=== VALIDATION SUMMARY ===")
        log.info("  Passed:   %d / %d scenarios", validation_pass, len(specs))
        log.info("  Warnings: %d", validation_warn)
        log.info("  Failed:   %d", validation_fail)
        if failed_scenarios:
            log.warning("  Failed scenarios:")
            for desc in failed_scenarios:
                log.warning("    - %s", desc)

    # -- Plots -------------------------------------------------------------
    if not args.no_plots and first_normal_df is not None and first_leak_df is not None:
        pdir = out / "plots"
        plot_pressure_profiles(first_normal_raw, first_normal_cfg,
                               save_path=pdir / "pressure_profile_normal.png")
        plot_pressure_profiles(first_leak_raw, first_leak_cfg,
                               save_path=pdir / "pressure_profile_leak.png")
        plot_timeseries_comparison(first_normal_df, first_leak_df,
                                   save_path=pdir / "normal_vs_leak.png")
        plot_temperature_comparison(first_normal_df, first_leak_df,
                                    save_path=pdir / "temperature_comparison.png")
        log.info("Plots → %s", pdir)

    # -- Summary -----------------------------------------------------------
    n_leak_rows = int(combined["leak_label"].sum())
    log.info("=== DATASET SUMMARY ===")
    log.info("  Scenarios:   %d (%d normal, %d leak)",
             len(all_dfs), args.n_normal, args.n_leak)
    log.info("  Total rows:  %d", len(combined))
    log.info("  Leak rows:   %d (%.1f%%)", n_leak_rows, 100 * n_leak_rows / len(combined))
    log.info("  Columns:     %s", list(combined.columns))
    log.info("  Output dir:  %s", out.resolve())


if __name__ == "__main__":
    main()
