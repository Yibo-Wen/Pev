"""Reading the evaluation split.

Splits were made by *family* before training, and a family held out for one
property is held out for every property. Building them is not part of this
distribution; what remains is the little that reading them needs -- including the
one subtlety, that Protein G has to be excluded at the **homolog** level, since
MegaScale and ProteinGym both carry it under other names.
"""

from __future__ import annotations

import hashlib

import pandas as pd

#: Substrings that mark a family as Protein G / GB1 at the *homolog* level.
PROTEIN_G_MARKERS = ("gb1", "spg1", "spg2", "protein_g", "1pga", "1pgb", "2gb1", "5ubs")

#: Sources that are evaluation-only by construction, and excluded from the
#: discovery panels because they belong to work reported elsewhere.
EVALUATION_EXCLUDED_SOURCES: tuple[str, ...] = ("gb1",)

#: Namespace for the random-baseline permutations, so that draw stays independent
#: of any other decision keyed on the same strings.
NS_BASELINE = "baseline"


def is_protein_g(family_id: str) -> bool:
    fam = family_id.lower()
    return any(marker in fam for marker in PROTEIN_G_MARKERS)


def evaluation_variants(variants: pd.DataFrame) -> pd.DataFrame:
    """The rows the discovery panels are built from: reserved families, minus Protein G."""
    fam = variants["family_id"].astype(str)
    return variants[
        (variants["split"] == "test")
        & (~variants["source"].isin(EVALUATION_EXCLUDED_SOURCES))
        & (~fam.map(is_protein_g))
    ]


def stable_hash(text: str, seed: int, *, namespace: str = "") -> int:
    """A 32-bit hash that is stable across processes.

    Builtin ``hash()`` is salted per interpreter, so anything seeded from it differs
    between runs -- which silently voids "every method saw identical panels".
    ``namespace`` keeps two decisions over the same keys independent.
    """
    digest = hashlib.sha256(f"{namespace}:{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:4], "big")
