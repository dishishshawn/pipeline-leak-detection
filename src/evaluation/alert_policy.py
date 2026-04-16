"""Alert policy layer: separates raw detector score from operator alerts.

Provides persistence windows, cooldowns, and multi-tick confirmation so
that a raw model score does not directly equal an operator alert.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class AlertPolicyConfig:
    threshold: float = 0.5
    persistence_ticks: int = 3
    cooldown_ticks: int = 10
    rearm_ticks: int = 10
    confirmation_k: int = 3
    confirmation_window: int = 5
    mode: str = "persistence"  # "persistence" | "confirmation"


class AlertPolicy:
    """Stateful alert filter that smooths raw scores into alert decisions."""

    def __init__(self, config: AlertPolicyConfig | None = None) -> None:
        self.config = config or AlertPolicyConfig()
        self.reset()

    def reset(self) -> None:
        self._consecutive_above = 0
        self._cooldown_remaining = 0
        self._needs_rearm = False
        self._below_since_fire = 0
        self._recent: deque[bool] = deque(maxlen=self.config.confirmation_window)

    def update(self, score: float) -> bool:
        """Process one tick. Returns True if alert should fire."""
        above = score >= self.config.threshold
        self._recent.append(above)

        # Cooldown: suppress alerts for N ticks after last alert
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        # After firing, require the score to actually drop below threshold
        # for `rearm_ticks` consecutive ticks before another alert can fire.
        # Prevents long-window features (CUSUM, expanding_dev) from keeping
        # the score elevated and causing a phantom re-alert after the leak
        # has already cleared.
        if self._needs_rearm:
            if not above:
                self._below_since_fire += 1
                if self._below_since_fire >= self.config.rearm_ticks:
                    self._needs_rearm = False
                    self._below_since_fire = 0
                    self._consecutive_above = 0
            else:
                self._below_since_fire = 0
            return False

        if self.config.mode == "persistence":
            if above:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0

            if self._consecutive_above >= self.config.persistence_ticks:
                self._cooldown_remaining = self.config.cooldown_ticks
                self._consecutive_above = 0
                self._needs_rearm = True
                self._below_since_fire = 0
                return True
            return False

        elif self.config.mode == "confirmation":
            count_above = sum(self._recent)
            if count_above >= self.config.confirmation_k:
                self._cooldown_remaining = self.config.cooldown_ticks
                self._needs_rearm = True
                self._below_since_fire = 0
                self._recent.clear()
                return True
            return False

        return False

    def process_series(self, scores: np.ndarray) -> np.ndarray:
        """Process a full series of scores. Returns boolean alert array."""
        self.reset()
        alerts = np.zeros(len(scores), dtype=bool)
        for i, score in enumerate(scores):
            alerts[i] = self.update(float(score))
        return alerts
