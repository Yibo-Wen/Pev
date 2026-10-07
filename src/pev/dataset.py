"""Reading the variant table: one row per measured variant.

Loading goes through :func:`pev.schema.validate` rather than straight to the
caller, so a parquet of the wrong shape fails at the boundary instead of
producing numbers that look fine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pev import schema


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return schema.validate(df)
