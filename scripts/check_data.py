#!/usr/bin/env python
"""Check the downloaded evaluation split before anything is scored on it.

    uv run scripts/check_data.py

Check whether these are the right files and whether they hold the right panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from pev import splits
from pev.dataset import load
from pev.experiments import discovery

ROOT = Path(__file__).resolve().parents[1]

#: How closely a rebuilt panel's prevalence has to match the pinned value. The
#: pins are rounded to six places; anything above this is a real difference.
PREVALENCE_TOL = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_files(pins: dict, skip: Path | None = None) -> list[str]:
    """Hash every pinned file. Absent or altered bytes stop the run here."""
    problems: list[str] = []
    for rel, want in pins["files"].items():
        path = ROOT / rel
        if skip is not None and path == skip:
            print(f"  --  {rel} not hashed: scoring a different file, {skip}")
            continue
        if not path.exists():
            problems.append(f"{rel} is missing -- unpack {pins['archive']} at the repository root")
            continue
        size = path.stat().st_size
        if size != want["bytes"]:
            problems.append(f"{rel} is {size:,} bytes, expected {want['bytes']:,}")
            continue
        got = sha256(path)
        if got != want["sha256"]:
            problems.append(f"{rel} hashes to {got[:16]}..., expected {want['sha256'][:16]}...")
            continue
        print(f"  ok  {rel}  {size:,} bytes  {got[:16]}...")
    return problems


def check_split(variants: pd.DataFrame, pins: dict) -> list[str]:
    """The archive is the reserved evaluation split and only that."""
    problems: list[str] = []
    other = sorted(set(variants["split"].unique()) - {"test"})
    if other:
        problems.append(f"the archive carries non-test splits: {', '.join(other)}")
    want = pins["evaluation_split"]
    got = {
        "rows": len(variants),
        "families": int(variants["family_id"].nunique()),
        "assay_contexts": int(variants["assay_context_id"].nunique()),
    }
    for key, value in want.items():
        if got[key] != value:
            problems.append(f"evaluation split has {got[key]:,} {key}, expected {value:,}")
    print(f"  ok  evaluation split: {got['rows']:,} rows, {got['families']} families, "
          f"{got['assay_contexts']} assay contexts")
    return problems


def check_panels(variants: pd.DataFrame, pins: dict) -> list[str]:
    """Rebuild the benchmark and compare it panel by panel against the pins."""
    problems: list[str] = []
    bench = pins["benchmark"]
    built = {
        p.context_id: p
        for p in discovery.build_panels(splits.evaluation_variants(variants))
    }
    pinned = {c["context_id"]: c for c in bench["contexts"]}

    for context_id, want in sorted(pinned.items()):
        panel = built.get(context_id)
        if panel is None:
            problems.append(f"panel {context_id} is absent")
            continue
        got = {
            "family_id": panel.family_id,
            "sequence_class": panel.sequence_class,
            "assay_type": panel.assay_type,
            "candidates": len(panel),
            "hits": int(panel.labels.sum()),
        }
        for key, value in got.items():
            if want[key] != value:
                problems.append(f"panel {context_id}: {key} is {value}, expected {want[key]}")
        if abs(panel.prevalence - want["prevalence"]) > PREVALENCE_TOL:
            problems.append(
                f"panel {context_id}: prevalence is {panel.prevalence:.6f}, "
                f"expected {want['prevalence']:.6f}"
            )

    # Panels the corpus can build that the benchmark does not include. Reported,
    # never a failure: the benchmark is the pinned list, and `run_discovery.py
    # --panels` holds the run to it.
    extra = sorted(set(built) - set(pinned))
    if not problems:
        frame = pd.DataFrame(bench["contexts"])
        print(f"  ok  {bench['panels']} panels, {bench['families']} families, "
              f"{bench['candidates']:,} candidates")
        for cls, group in frame.groupby("sequence_class"):
            print(f"        {cls:9s} {len(group):3d} panels  "
                  f"{group.family_id.nunique():3d} families  "
                  f"{group.candidates.sum():6,d} candidates")
    if extra:
        print(f"\n  note  {len(extra)} further panel(s) buildable but outside the benchmark:")
        for context_id in extra:
            print(f"        {context_id}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pins", type=Path, default=ROOT / "data/exp1_panels.json")
    ap.add_argument("--variants", type=Path, default=ROOT / "data/processed/variants.parquet")
    args = ap.parse_args(argv)

    pins = json.loads(args.pins.read_text())
    print(f"checking against {args.pins}\n")

    # The pins name repository-relative paths, which is where the archive
    # unpacks. Pointing --variants somewhere else -- a corpus rebuilt from the
    # original sources, which is a superset and will not hash the same -- leaves
    # the panel comparison, which is the part that still means something.
    pinned = ROOT / "data/processed/variants.parquet"
    rebuilt = args.variants.resolve() != pinned.resolve()

    problems = check_files(pins, skip=pinned if rebuilt else None)
    if problems:
        return report(problems)

    variants = load(args.variants)
    if not rebuilt:
        problems += check_split(variants, pins)
    problems += check_panels(variants, pins)
    return report(problems)


def report(problems: list[str]) -> int:
    if problems:
        print(f"\nFAILED -- {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nPASSED -- the Experiment 1 test data is the pinned one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
