"""The discovery screen: can a ranking save wells on a family never seen?

A panel is one measured reference protein and *every* measured single
substitution of it inside one assay context -- a few hundred to a few thousand
candidates, all measured, drawn from families reserved before training. Nothing
is padded: an unmeasured mutation is absent rather than a miss, because a screen
cannot be charged for a well nobody ran. The reference is the wild type, which is
what keeps the base rate low enough to be worth enriching.

STOP is out of scope here -- "no beneficial edit exists" is not a claim a
measured subset can support. The inclusion rules below are model-blind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pev import metrics
from pev.model import USABLE_RESIDUES
from pev.splits import NS_BASELINE, stable_hash

log = logging.getLogger(__name__)

#: Wells a screen may spend on one panel.
BUDGET = 10

#: Candidates a panel must offer to count as broad coverage. Below this the
#: budget stops being a selection and starts being an enumeration.
MIN_CANDIDATES = 100

#: Replicate SDs a candidate must clear on top of the meaningful effect.
MARGIN_SD = 1.0

#: What counts as a **meaningful** improvement, in the readout's own units: the
#: literature's call for a stabilizing mutation, the smallest shift a thermal
#: assay will report, and a two-fold improvement in Kd. Fixed from convention
#: before any model was scored, not set to whatever would make hits rare.
MEANINGFUL: dict[str, float] = {
    "folding_free_energy": 0.5,
    "melting_temperature": 1.0,
    "affinity": 0.3,
}

#: Readouts in source-specific arbitrary units have no convention to borrow, so
#: the meaningful effect is half a robust SD of the panel's own spread -- a
#: medium effect size in the only scale the assay offers. Computed from the
#: measurements alone, never from a prediction.
RELATIVE_SD = 0.5


@dataclass
class Panel:
    """One reference protein and every measured single substitution of it."""

    context_id: str
    family_id: str
    sequence_class: str
    assay_type: str
    readout_type: str
    source: str
    reference_id: str
    reference_sequence: str
    reference_y: float
    candidate_ids: list[str]
    edits: list[tuple[int, str]]
    #: ``y(candidate) - y(reference)``, oriented higher-is-better.
    utilities: np.ndarray
    #: Per-candidate improvement threshold: the larger of replicate noise and the
    #: prespecified meaningful effect.
    threshold: np.ndarray
    target_sequence: str | None = None
    partner_sequence: str | None = None

    def __len__(self) -> int:
        return len(self.edits)

    @property
    def labels(self) -> np.ndarray:
        return self.utilities > self.threshold

    @property
    def prevalence(self) -> float:
        return float(self.labels.mean())


def build_panels(
    variants: pd.DataFrame,
    *,
    min_candidates: int = MIN_CANDIDATES,
    margin_sd: float = MARGIN_SD,
) -> list[Panel]:
    """Every broad-coverage panel a split can supply, with its labels attached.

    The inclusion rules, fixed before scoring and all but the last blind to every
    outcome: the row is measured and uncensored; its assay context carries a
    measured zero-mutation reference (median over replicate reference rows);
    candidates are single substitutions of that reference's *own* sequence,
    recovered by diffing rather than trusting the ``mutations`` column; the edit
    sits inside the encoder's window; at least ``min_candidates`` survive; and at
    least one clears its threshold. The last rule is the only one that looks at an
    outcome, and it drops escape assays where no edit improves at all.
    """
    usable = variants[variants["censored"].eq(False) & variants["y"].notna()]
    if usable.empty:
        return []
    context_sd = (
        usable.groupby("assay_context_id")["y_sd"].median()
        if "y_sd" in usable.columns
        else pd.Series(dtype=float)
    )
    references = usable[usable["n_mut"].eq(0)]
    singles = usable[usable["n_mut"].eq(1)]

    panels: list[Panel] = []
    dropped: dict[str, int] = {"no_reference": 0, "too_narrow": 0, "no_hit": 0}
    for (context_id, family_id), group in singles.groupby(
        ["assay_context_id", "family_id"], sort=True
    ):
        reference = references[
            references["assay_context_id"].eq(context_id)
            & references["family_id"].eq(family_id)
        ]
        if reference.empty:
            dropped["no_reference"] += 1
            continue
        # A handful of contexts carry more than one zero-mutation row. Where those
        # rows are the same molecule they are replicates; where they are not (two
        # SKEMPI complexes filed under one context) only one can be the reference,
        # so the panel keeps the modal sequence and reads its value from the rows
        # that actually carry it.
        reference_sequence = str(reference["sequence"].mode().iat[0])
        reference = reference[reference["sequence"].eq(reference_sequence)]
        row = reference.iloc[0]
        reference_y = float(reference["y"].median())
        reference_sd = _reference_sd(reference, row, context_sd)

        ids, edits, utilities, noise = [], [], [], []
        for candidate in group.itertuples():
            sequence = str(candidate.sequence)
            if len(sequence) != len(reference_sequence):
                continue
            diff = [
                (i + 1, b)
                for i, (a, b) in enumerate(zip(reference_sequence, sequence, strict=True))
                if a != b
            ]
            if len(diff) != 1 or diff[0][0] > USABLE_RESIDUES:
                continue
            ids.append(str(candidate.variant_id))
            edits.append(diff[0])
            utilities.append(float(candidate.y) - reference_y)
            noise.append(_sd(candidate._asdict(), context_sd))

        if len(edits) < min_candidates:
            dropped["too_narrow"] += 1
            continue

        u = np.asarray(utilities, dtype=float)
        sd = np.sqrt(np.square(np.asarray(noise, dtype=float)) + reference_sd**2)
        readout = str(row["readout_type"])
        meaningful = MEANINGFUL.get(readout)
        if meaningful is None:
            # No convention to borrow: half a robust SD of the panel's own spread.
            meaningful = RELATIVE_SD * 1.4826 * float(np.median(np.abs(u - np.median(u))))
        threshold = np.maximum(margin_sd * sd, meaningful)

        if not (u > threshold).any():
            dropped["no_hit"] += 1
            continue

        panels.append(
            Panel(
                context_id=str(context_id),
                family_id=str(family_id),
                sequence_class=str(row["sequence_class"]),
                assay_type=str(row["assay_type"]),
                readout_type=readout,
                source=str(row["source"]),
                reference_id=str(row["variant_id"]),
                reference_sequence=reference_sequence,
                reference_y=reference_y,
                candidate_ids=ids,
                edits=edits,
                utilities=u,
                threshold=threshold,
                target_sequence=_optional(row.get("target_sequence")),
                partner_sequence=_optional(row.get("partner_sequence")),
            )
        )

    log.info(
        "discovery: %d panel(s) over %d families; dropped %d without a measured "
        "reference, %d under %d candidates, %d with no measured improvement",
        len(panels),
        len({p.family_id for p in panels}),
        dropped["no_reference"],
        dropped["too_narrow"],
        min_candidates,
        dropped["no_hit"],
    )
    return panels


def _reference_sd(reference: pd.DataFrame, row, context_sd: pd.Series) -> float:
    """How well the panel's anchor is pinned down.

    Every utility is measured against this one number, so its uncertainty is
    everyone's. One measurement can offer only its own replicate SD; several of the
    same molecule can do much worse -- AlphaSeq records the Ab-14 wild type at 4.9,
    7.3 and 8.4 pKd in one assay, where the first row's internal SD would claim
    0.02. The dispersion across replicates is the honest figure.
    """
    own = _sd(row, context_sd)
    if len(reference) < 2:
        return own
    spread = float(reference["y"].std(ddof=1)) / np.sqrt(len(reference))
    if not np.isfinite(spread):
        return own
    return max(own, spread)


def _sd(row, context_sd: pd.Series) -> float:
    """Replicate SD, imputed from the assay context when a row has none."""
    value = row["y_sd"] if isinstance(row, dict) else row.get("y_sd")
    if value is not None and not pd.isna(value):
        return float(value)
    key = row["assay_context_id"] if isinstance(row, dict) else row.get("assay_context_id")
    if len(context_sd):
        imputed = context_sd.get(key)
        if imputed is not None and not pd.isna(imputed):
            return float(imputed)
    return 0.0


def _optional(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# Rankings
# --------------------------------------------------------------------------- #


def random_rankings(panel: Panel, *, draws: int = 20, seed: int = 0) -> list[np.ndarray]:
    """Several uniform permutations, averaged over.

    A screening baseline read from one permutation of a 3,000-candidate panel is
    mostly noise. Averaging twenty gives a comparator tight enough that the
    paired interval measures the method rather than the draw.
    """
    rng = np.random.default_rng(stable_hash(panel.context_id, seed, namespace=NS_BASELINE))
    return [rng.permutation(len(panel)) for _ in range(draws)]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

#: Everything computed per panel.
HEADLINE = ("hit_rate", "enrichment", "gain", "ndcg", "auroc")

#: What the tables print. Three numbers: how often a well pays off, how much
#: better that is than screening at random, and whether the ordering is real.
#: The rest stay in the CSVs -- they agree, and five columns of agreement is not
#: five times the evidence.
PRIMARY = ("hit_rate", "enrichment", "auroc")


def score_panel(panel: Panel, ranking: np.ndarray, *, budget: int = BUDGET) -> dict:
    """Every reported metric for one panel under one ranking."""
    labels = panel.labels
    k = min(budget, len(panel))
    return {
        "context_id": panel.context_id,
        "family_id": panel.family_id,
        "sequence_class": panel.sequence_class,
        "assay_type": panel.assay_type,
        "readout_type": panel.readout_type,
        "source": panel.source,
        "n_candidates": len(panel),
        "n_hits": int(labels.sum()),
        "prevalence": panel.prevalence,
        # The claim: what fraction of the ten wells a lab would actually run
        # comes back a hit, and how that compares with running ten at random.
        "hit_rate": metrics.precision_at_k(labels, ranking, k),
        "enrichment": metrics.enrichment_at_k(labels, ranking, k),
        # How much of the best improvement in the whole scan those ten wells find.
        "gain": metrics.improvement_ratio(panel.utilities, ranking[:k]),
        # The graded version of the hit rate: finding the large win early counts
        # for more than finding ten marginal ones.
        "ndcg": metrics.ndcg_at_k(panel.utilities, ranking, k),
        # Ranking quality with no budget in it at all, kept because it is the one
        # number whose random-to-perfect span does not move with the panel.
        "auroc": metrics.panel_auroc(labels, ranking),
        "assays_to_hit": metrics.assays_to_first_hit(labels, ranking),
    }


def score(
    panels: list[Panel],
    rankings: dict[str, np.ndarray | list[np.ndarray]],
    *,
    budget: int = BUDGET,
    method: str = "",
) -> pd.DataFrame:
    """One row per panel. ``rankings`` may hold several draws per panel, averaged."""
    rows = []
    for panel in panels:
        drawn = rankings.get(panel.context_id)
        if drawn is None:
            continue
        if isinstance(drawn, np.ndarray):
            drawn = [drawn]
        scored = [score_panel(panel, r, budget=budget) for r in drawn]
        row = dict(scored[0])
        for column in (*HEADLINE, "assays_to_hit"):
            values = np.array([s[column] for s in scored], dtype=float)
            row[column] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        row["method"] = method
        rows.append(row)
    return pd.DataFrame(rows)


def perfect_rankings(panels: list[Panel]) -> dict[str, np.ndarray]:
    """The ceiling: hits first, ordered by measured effect."""
    return {
        p.context_id: np.lexsort((-p.utilities, ~p.labels)) for p in panels
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def macro(
    frame: pd.DataFrame, *, by: tuple[str, ...] = (), unit: str = "panel"
) -> pd.DataFrame:
    """Average the per-panel metrics up to a reporting group.

    ``unit="panel"`` weights every panel equally -- a panel is one screen, and this
    is the rule the reported tables use. ``unit="family"`` averages panels within a
    family first, so a family measured twice counts once. The two disagree wherever
    a family supplies more than one panel (seven peptide families supply thirteen),
    so both are written out. The bootstrap in :func:`compare` clusters by family
    under either unit.
    """
    if frame.empty:
        return frame
    if unit not in ("panel", "family"):
        raise ValueError(f"unit must be 'panel' or 'family', not {unit!r}")
    columns = [c for c in (*HEADLINE, "assays_to_hit", "prevalence") if c in frame.columns]
    per_family = (
        frame
        if unit == "panel"
        else frame.groupby(["family_id", *by], as_index=False)[columns].mean()
    )
    grouped = per_family.groupby(list(by), as_index=False) if by else None

    def block(subset: pd.DataFrame) -> dict:
        out = {c: float(subset[c].mean(skipna=True)) for c in columns}
        out["families"] = int(subset["family_id"].nunique())
        out["panels"] = int(frame[frame["family_id"].isin(subset["family_id"])].shape[0])
        return out

    if grouped is None:
        return pd.DataFrame([block(per_family)])
    rows = []
    for keys, subset in per_family.groupby(list(by), sort=True):
        label = dict(zip(by, keys if isinstance(keys, tuple) else (keys,), strict=True))
        rows.append({**label, **block(subset)})
    return pd.DataFrame(rows)


def compare(
    a: pd.DataFrame,
    b: pd.DataFrame,
    column: str,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired family-clustered bootstrap of ``a - b`` over shared panels."""
    merged = a[["context_id", "family_id", column]].merge(
        b[["context_id", column]], on="context_id", suffixes=("_a", "_b")
    )
    merged = merged.dropna()
    if merged.empty:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_families": 0}
    per_family = merged.groupby("family_id", as_index=False)[
        [f"{column}_a", f"{column}_b"]
    ].mean()
    point, lo, hi = metrics.paired_bootstrap(
        per_family[f"{column}_a"].to_numpy(),
        per_family[f"{column}_b"].to_numpy(),
        per_family["family_id"].to_numpy(),
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_families": int(per_family["family_id"].nunique()),
        "n_panels": len(merged),
        "superior": bool(np.isfinite(lo) and lo > 0),
    }
