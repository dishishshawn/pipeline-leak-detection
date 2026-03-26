"""Plotting utilities for physics simulator output.

All functions accept raw solution dicts or DataFrames and optionally
save to disk.  Figures are closed after saving to avoid memory leaks
during batch generation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pressure_profiles(
    raw: dict,
    config,
    times_s: list[float] | None = None,
    save_path: Path | str | None = None,
) -> None:
    """Pressure vs position at selected time snapshots."""
    P = raw["P"]
    t = raw["t"]
    x_km = raw["cell_centers"] / 1000.0

    if times_s is None:
        dur = config.duration
        times_s = [0, dur / 4, dur / 2, dur * 3 / 4, dur]

    fig, ax = plt.subplots(figsize=(10, 5))
    for ts in times_s:
        idx = int(np.argmin(np.abs(t - ts)))
        ax.plot(x_km, P[:, idx] / 1e6, marker="o", markersize=4, label=f"t = {t[idx]:.0f} s")

    ax.set_xlabel("Position along pipe (km)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title("Pressure Profile Along Pipeline")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_timeseries_comparison(
    df_normal: pd.DataFrame,
    df_leak: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """3-panel comparison: pressure, flow, and leak rate."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # -- Pressure --
    for col in sorted(c for c in df_normal.columns if c.startswith("P_")):
        axes[0].plot(df_normal["time"], df_normal[col] / 1e6,
                     linestyle="--", alpha=0.6, label=f"{col} (normal)")
        axes[0].plot(df_leak["time"], df_leak[col] / 1e6,
                     label=f"{col} (leak)")
    axes[0].set_ylabel("Pressure (MPa)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # -- Flow --
    axes[1].plot(df_normal["time"], df_normal["Q_inlet"] * 1000,
                 linestyle="--", label="Q_inlet (normal)")
    axes[1].plot(df_normal["time"], df_normal["Q_outlet"] * 1000,
                 linestyle="--", label="Q_outlet (normal)")
    axes[1].plot(df_leak["time"], df_leak["Q_inlet"] * 1000,
                 label="Q_inlet (leak)")
    axes[1].plot(df_leak["time"], df_leak["Q_outlet"] * 1000,
                 label="Q_outlet (leak)")
    axes[1].set_ylabel("Flow (L/s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # -- Leak rate --
    axes[2].plot(df_leak["time"], df_leak["leak_rate"] * 1000, color="red", label="Leak rate")
    axes[2].set_ylabel("Leak rate (L/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Normal vs Leak Scenario Comparison", fontsize=13)
    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_temperature_comparison(
    df_normal: pd.DataFrame,
    df_leak: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """Temperature at each sensor: normal vs leak."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in sorted(c for c in df_normal.columns if c.startswith("T_")):
        ax.plot(df_normal["time"], df_normal[col] - 273.15,
                linestyle="--", alpha=0.6, label=f"{col} (normal)")
        ax.plot(df_leak["time"], df_leak[col] - 273.15, label=f"{col} (leak)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature: Normal vs Leak")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_and_close(fig, save_path)


def _save_and_close(fig, path: Path | str | None) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
