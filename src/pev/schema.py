"""The data contract every row of the variant table satisfies.

:func:`validate` is the only place these rules live, so this module is where to
look when a column's meaning is in question. The invariant worth stating loudly:
``y`` is always in native, documented units, oriented so higher is better.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

#: The 20 residues a substitution may target.
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

#: Residues tolerated *inside* an assayed sequence but never as a mutation
#: target. Real data contains these: CR6261's light chain carries a literal X.
AMBIGUOUS_AA = "BJOUXZ"

SEQUENCE_ALPHABET = frozenset(STANDARD_AA + AMBIGUOUS_AA)


class SequenceClass(StrEnum):
    PEPTIDE = "peptide"
    DOMAIN = "domain"
    ANTIBODY = "antibody"  # includes nanobody / VHH


class AssayType(StrEnum):
    BINDING = "binding"
    STABILITY = "stability"


class ReadoutType(StrEnum):
    AFFINITY = "affinity"
    BINDING_SIGNAL = "binding_signal"
    FOLDING_FREE_ENERGY = "folding_free_energy"
    MELTING_TEMPERATURE = "melting_temperature"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    CALIB = "calib"
    TEST = "test"


#: Which objective each readout belongs to. Enforced by :func:`validate`; a
#: readout may not migrate between objectives.
READOUT_TO_ASSAY: dict[ReadoutType, AssayType] = {
    ReadoutType.AFFINITY: AssayType.BINDING,
    ReadoutType.BINDING_SIGNAL: AssayType.BINDING,
    ReadoutType.FOLDING_FREE_ENERGY: AssayType.STABILITY,
    ReadoutType.MELTING_TEMPERATURE: AssayType.STABILITY,
}

#: Documented native unit per readout, recorded so reports never guess.
READOUT_UNITS: dict[ReadoutType, str] = {
    ReadoutType.AFFINITY: "pKd = -log10(Kd / M)",
    ReadoutType.BINDING_SIGNAL: "oriented enrichment / selection score (source-specific)",
    ReadoutType.FOLDING_FREE_ENERGY: "kcal/mol, higher = more stable",
    ReadoutType.MELTING_TEMPERATURE: "degrees Celsius",
}


# --------------------------------------------------------------------------- #
# Mutations
# --------------------------------------------------------------------------- #

_MUT_RE = re.compile(rf"^([{STANDARD_AA}])(\d+)([{STANDARD_AA}])$")


@dataclass(frozen=True, slots=True)
class Mutation:
    """A single substitution, 1-based against the family's parent sequence."""

    wt: str
    pos: int
    mut: str

    def __post_init__(self) -> None:
        if self.wt not in STANDARD_AA:
            raise ValueError(f"wt residue {self.wt!r} not in standard alphabet")
        if self.mut not in STANDARD_AA:
            raise ValueError(f"mut residue {self.mut!r} not in standard alphabet")
        if self.pos < 1:
            raise ValueError(f"position must be 1-based, got {self.pos}")
        if self.wt == self.mut:
            raise ValueError(f"{self} is not a substitution")

    def __str__(self) -> str:
        return f"{self.wt}{self.pos}{self.mut}"

    @classmethod
    def parse(cls, token: str) -> Mutation:
        m = _MUT_RE.match(token.strip())
        if not m:
            raise ValueError(f"cannot parse mutation {token!r}; expected e.g. 'A12G'")
        return cls(wt=m.group(1), pos=int(m.group(2)), mut=m.group(3))
def parse_mutations(s: str | None) -> list[Mutation]:
    if not s:
        return []
    return [Mutation.parse(tok) for tok in s.split(",") if tok.strip()]
# --------------------------------------------------------------------------- #

#: (column, dtype-kind, nullable). Kind is checked loosely -- pandas dtypes vary
#: across parquet round-trips, so we test semantics rather than exact dtypes.
COLUMNS: dict[str, tuple[str, bool]] = {
    "variant_id": ("str", False),
    "source": ("str", False),
    "family_id": ("str", False),
    "assay_context_id": ("str", False),
    "sequence_class": ("str", False),
    "assay_type": ("str", False),
    "readout_type": ("str", False),
    "sequence": ("str", False),
    "parent_id": ("str", True),
    "mutations": ("str", True),
    "n_mut": ("int", False),
    "y": ("float", True),  # null iff censored
    "y_raw": ("float", True),
    "y_sd": ("float", True),
    "n_replicates": ("int", True),
    "censored": ("bool", False),
    "target_sequence": ("str", True),
    "partner_sequence": ("str", True),
    "mutable_positions": ("list", True),
    "split": ("str", True),
    "y_unit01": ("float", True),  # reporting only, never a training target
}

REQUIRED_COLUMNS = tuple(COLUMNS)


class SchemaError(ValueError):
    """Raised when a variant table violates the contract."""


def _fail(problems: list[str]) -> None:
    if problems:
        joined = "\n  - ".join(problems)
        raise SchemaError(f"variant table violates the contract:\n  - {joined}")


def validate(df: pd.DataFrame, *, strict_sequences: bool = True) -> pd.DataFrame:
    """Validate a variant table, returning it unchanged.

    Checks the structural contract only. Outcome *orientation* cannot be checked
    here -- it is source-specific and is asserted by each adapter's own test
    against a known improving mutation.
    """
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _fail([f"missing columns: {missing}"])

    if df.empty:
        return df

    for col, (_, nullable) in COLUMNS.items():
        if not nullable and df[col].isna().any():
            n = int(df[col].isna().sum())
            problems.append(f"{col!r} is non-nullable but has {n} null(s)")

    dupes = int(df["variant_id"].duplicated().sum())
    if dupes:
        problems.append(f"{dupes} duplicate variant_id(s)")

    for col, enum in (
        ("sequence_class", SequenceClass),
        ("assay_type", AssayType),
        ("readout_type", ReadoutType),
    ):
        allowed = {e.value for e in enum}
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            problems.append(f"{col!r} has values outside {sorted(allowed)}: {sorted(bad)}")

    bad_split = set(df["split"].dropna().unique()) - {s.value for s in Split}
    if bad_split:
        problems.append(f"'split' has unknown values: {sorted(bad_split)}")

    # Readout must stay inside its objective.
    known = df["readout_type"].isin(READOUT_TO_ASSAY)
    if known.any():
        expected = df.loc[known, "readout_type"].map(
            {k.value: v.value for k, v in READOUT_TO_ASSAY.items()}
        )
        mismatch = df.loc[known, "assay_type"] != expected
        if mismatch.any():
            rows = df.loc[known][mismatch][["readout_type", "assay_type"]].drop_duplicates()
            problems.append(f"readout/assay mismatch: {rows.to_dict('records')}")

    # Censoring: a censored row carries no usable outcome, and an uncensored one must.
    censored = df["censored"].astype(bool)
    if (censored & df["y"].notna()).any():
        n = int((censored & df["y"].notna()).sum())
        problems.append(f"{n} censored row(s) carry a non-null y; censored outcomes have no label")
    if (~censored & df["y"].isna()).any():
        n = int((~censored & df["y"].isna()).sum())
        problems.append(f"{n} uncensored row(s) have null y; mark them censored instead")

    if (df["n_mut"] < 0).any():
        problems.append("negative n_mut")

    n_mut_stated = df["mutations"].fillna("").map(lambda s: len(parse_mutations(s)))
    if not n_mut_stated.equals(df["n_mut"].astype(int)):
        n = int((n_mut_stated != df["n_mut"].astype(int)).sum())
        problems.append(f"{n} row(s) where n_mut disagrees with the mutations string")

    if strict_sequences:
        bad_seq = df["sequence"].map(lambda s: bool(set(str(s)) - SEQUENCE_ALPHABET))
        if bad_seq.any():
            offenders = sorted({c for s in df.loc[bad_seq, "sequence"] for c in set(str(s))} - SEQUENCE_ALPHABET)
            problems.append(f"{int(bad_seq.sum())} sequence(s) contain non-residue chars: {offenders}")

    _fail(problems)
    return df
