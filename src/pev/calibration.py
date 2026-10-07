"""Applying the calibrator that ships with the checkpoint.

Proper scoring rules get the ordering right but leave the probabilities sharp.
``calibration.json`` carries the fix: per objective, one temperature for the
choice distribution and one affine map for the improvement logit, fitted on
held-out calibration families -- never on validation, which selected the
checkpoint, and never on test. Both transforms are monotone, so they change no
ranking and no metric that depends only on order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FILENAME = "calibration.json"


@dataclass(frozen=True)
class ObjectiveCalibration:
    """One objective's transform, plus what the fit was and how well it went."""

    objective: str
    temperature: float = 1.0
    alpha: float = 1.0
    beta: float = 0.0
    #: Which family pool supplied the fit: own, borrowed, pooled, or identity.
    choice_rung: str = "identity"
    improve_rung: str = "identity"
    n_families: int = 0
    n_episodes: int = 0
    choice_nll_before: float = float("nan")
    choice_nll_after: float = float("nan")
    improve_nll_before: float = float("nan")
    improve_nll_after: float = float("nan")
    #: Parameters that landed on a bound -- a fit the data did not pin down.
    boundary: tuple[str, ...] = ()

    @property
    def unstable(self) -> bool:
        return bool(self.boundary)

    def apply_choice(self, logits: np.ndarray) -> np.ndarray:
        """Temperature-scaled softmax over the offered actions, STOP included."""
        z = np.asarray(logits, dtype=float) / self.temperature
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def apply_improve(self, logits: np.ndarray) -> np.ndarray:
        """Affine-then-sigmoid, per candidate."""
        return 1.0 / (1.0 + np.exp(-(self.alpha * np.asarray(logits, dtype=float) + self.beta)))

    @classmethod
    def from_dict(cls, d: dict) -> ObjectiveCalibration:
        return cls(
            objective=d["objective"],
            temperature=d.get("temperature", 1.0),
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 0.0),
            choice_rung=str(d.get("choice_rung", "identity")),
            improve_rung=str(d.get("improve_rung", "identity")),
            n_families=d.get("n_families", 0),
            n_episodes=d.get("n_episodes", 0),
            choice_nll_before=d.get("choice_nll_before", float("nan")),
            choice_nll_after=d.get("choice_nll_after", float("nan")),
            improve_nll_before=d.get("improve_nll_before", float("nan")),
            improve_nll_after=d.get("improve_nll_after", float("nan")),
            boundary=tuple(d.get("boundary", ())),
        )


@dataclass(frozen=True)
class StopThreshold:
    """When to stop rather than spend another assay."""

    tau: float = 0.0
    #: ``"p_stop"`` stops when the calibrated STOP probability reaches tau;
    #: ``"gain"`` when no predicted gain clears it.
    mode: str = "gain"
    fitted_on: str = "val"
    n_episodes: int = 0
    score: float = float("nan")

    def chooses_stop(
        self, predicted_gain: np.ndarray | None = None, *, p_stop: float | None = None
    ) -> bool:
        if self.mode == "p_stop":
            return bool(p_stop is not None and p_stop >= self.tau)
        gain = np.asarray(predicted_gain if predicted_gain is not None else [], dtype=float)
        return bool(gain.size == 0 or gain.max() <= self.tau)

    @classmethod
    def from_dict(cls, d: dict) -> StopThreshold:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class Calibrator:
    """The whole bundle: a transform per objective, plus the stop rule."""

    per_objective: dict[str, ObjectiveCalibration] = field(default_factory=dict)
    version: str = "identity"
    fitted_on: dict = field(default_factory=dict)
    stop_threshold: StopThreshold | None = None

    @classmethod
    def identity(cls) -> Calibrator:
        return cls()

    def for_objective(self, assay_type: str) -> ObjectiveCalibration:
        """The transform for an objective, falling back to pooled then identity."""
        if assay_type in self.per_objective:
            return self.per_objective[assay_type]
        if "pooled" in self.per_objective:
            return self.per_objective["pooled"]
        return ObjectiveCalibration(objective=assay_type)

    @classmethod
    def from_dict(cls, d: dict) -> Calibrator:
        stop = d.get("stop_threshold")
        return cls(
            per_objective={
                k: ObjectiveCalibration.from_dict(v)
                for k, v in d.get("per_objective", {}).items()
            },
            version=d.get("version", "identity"),
            fitted_on=d.get("fitted_on", {}),
            stop_threshold=StopThreshold.from_dict(stop) if stop else None,
        )

    def render(self) -> str:
        lines = [f"calibration {self.version}"]
        for name, obj in sorted(self.per_objective.items()):
            lines.append(
                f"  {name:10s} T={obj.temperature:.4f} alpha={obj.alpha:.4f} "
                f"beta={obj.beta:+.4f}"
                + ("  [unstable: " + ", ".join(obj.boundary) + "]" if obj.unstable else "")
            )
            lines.append(
                f"             choice rung {obj.choice_rung} | improve rung {obj.improve_rung}"
                f" | {obj.n_families} families / {obj.n_episodes} episodes"
            )
        if self.stop_threshold is not None:
            t = self.stop_threshold
            lines.append(f"  stop threshold tau={t.tau:.4f} ({t.mode}, fitted on {t.fitted_on})")
        return "\n".join(lines)


def load(directory: Path, *, required: bool = False) -> Calibrator:
    """The bundle's calibrator, or the identity so uncalibrated runs still work."""
    path = Path(directory) / FILENAME
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} does not exist")
        log.warning("%s has no %s; reporting uncalibrated probabilities", directory, FILENAME)
        return Calibrator.identity()
    return Calibrator.from_dict(json.loads(path.read_text()))
