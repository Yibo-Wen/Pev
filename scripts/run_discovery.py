#!/usr/bin/env python
"""The primary held-out-family test: discovery on broad single-substitution panels.

    uv run scripts/run_discovery.py --checkpoint checkpoints/pev \
      --panels data/exp1_panels.json --heads binding=choice,stability=choice

One reference protein, every measured single substitution of it, a ten-well
budget. Reports hit rate and enrichment at the budget, recovered gain, NDCG and
panel AUROC, averaged over panels -- and over families beside them, since the two
differ wherever a family supplies more than one panel.

``--panels`` holds the run to the pinned benchmark so a later corpus cannot add
or drop one. ``--heads`` replays the ranking head chosen on the calibration and
validation families, before any test panel was scored; that is the only tuning in
the experiment, and it happened on data this distribution does not ship.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pev import splits
from pev.dataset import load
from pev.experiments import discovery
from pev.loading import config_from_metadata, load_checkpoint
from pev.model import DecisionModel

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("discovery")

#: How a model turns one panel into an ordering. Both come free from one forward
#: pass; which one ranks better is an empirical question, settled on a non-test
#: split before the test panels are scored.
HEADS = ("choice", "improve")
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=Path, default=ROOT / "data/processed/variants.parquet")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "discovery")
    ap.add_argument(
        "--panels",
        type=Path,
        default=None,
        help="hold the run to a pinned panel list (data/exp1_panels.json). The "
        "benchmark is a fixed set of panels, not whatever the corpus happens to "
        "yield, so a later corpus cannot quietly add or drop one",
    )
    ap.add_argument("--budget", type=int, default=discovery.BUDGET)
    ap.add_argument("--min-candidates", type=int, default=discovery.MIN_CANDIDATES)
    ap.add_argument("--margin-sd", type=float, default=discovery.MARGIN_SD)
    ap.add_argument("--random-draws", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--head", choices=HEADS, default=None, help="skip selection, force one head")
    ap.add_argument(
        "--heads",
        default=None,
        help="skip selection and pin the per-objective rule, e.g. "
        "'binding=improve,stability=choice'. Use this to apply one already-chosen "
        "rule across seed replicates rather than re-choosing it per run.",
    )
    ap.add_argument(
        "--zero-shot",
        action="store_true",
        help="also score the untrained ESM-2 backbone, to separate what training buys "
        "from what the pretrained prior already knew",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    pd.set_option("display.width", 220, "display.max_columns", 40)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.variants.exists():
        raise SystemExit(
            f"no measurements at {args.variants}. Download pev-exp1-test.zip (see README.md), then\n"
            f"  unzip pev-exp1-test.zip && make check"
        )
    variants = load(args.variants)
    build = {"min_candidates": args.min_candidates, "margin_sd": args.margin_sd}
    model = _load(args.checkpoint, args.device)

    # The ranking head is chosen per objective on non-test splits, once, before a
    # single test panel is scored. Both heads come off the same forward pass, so
    # this costs nothing at inference and everything in credibility if it were
    # done anywhere else.
    backbone = None
    if args.zero_shot:
        from pev.baselines import load_mutate_everything

        backbone = load_mutate_everything(device=args.device)

    heads: dict[str, str] = {}
    if args.heads:
        heads = dict(pair.split("=", 1) for pair in args.heads.split(",") if pair)
    elif args.head:
        heads = dict.fromkeys(("binding", "stability"), args.head)
    heads = heads or dict.fromkeys(("binding", "stability"), "choice")
    print(f"ranking heads: {heads}")

    panels = discovery.build_panels(splits.evaluation_variants(variants), **build)
    if args.panels:
        panels = _frozen(panels, args.panels)
    if not panels:
        print("no broad-coverage panels in the evaluation split")
        return 1
    _describe(panels, args)

    cached = _logits(model, panels)
    backbone_cached = _logits(backbone, panels) if backbone is not None else None
    rankings = _order(cached, panels, lambda p: heads.get(p.assay_type, "choice"))
    scored = {
        "pev": discovery.score(panels, rankings, budget=args.budget, method="pev"),
        "random": discovery.score(
            panels,
            {p.context_id: discovery.random_rankings(p, draws=args.random_draws, seed=args.seed)
             for p in panels},
            budget=args.budget,
            method="random",
        ),
        "perfect": discovery.score(
            panels, discovery.perfect_rankings(panels), budget=args.budget, method="perfect"
        ),
    }
    for name, frame in scored.items():
        frame.to_parquet(args.out / f"panels_{name}.parquet", index=False)

    if args.zero_shot:
        scored["zero_shot"] = discovery.score(
            panels,
            _order(backbone_cached, panels, lambda _p: "choice"),
            budget=args.budget,
            method="zero_shot",
        )
        scored["zero_shot"].to_parquet(args.out / "panels_zero_shot.parquet", index=False)

    order = ["perfect", "pev", *(["zero_shot"] if args.zero_shot else []), "random"]
    overall = pd.concat(
        [discovery.macro(scored[m]).assign(method=m) for m in order], ignore_index=True
    )
    by_class = pd.concat(
        [discovery.macro(scored[m], by=("sequence_class",)).assign(method=m) for m in order],
        ignore_index=True,
    )
    by_objective = pd.concat(
        [discovery.macro(scored[m], by=("assay_type",)).assign(method=m) for m in order],
        ignore_index=True,
    )
    by_cell = pd.concat(
        [
            discovery.macro(scored[m], by=("sequence_class", "assay_type")).assign(method=m)
            for m in order
        ],
        ignore_index=True,
    )
    # The same panels under the other aggregation unit. Written out rather than
    # chosen between: the two rules disagree only where one family supplies more
    # than one panel -- seven peptide families supply thirteen panels -- and a
    # reader should be able to see by how much rather than take the headline on
    # trust. The reported tables above are the per-panel mean.
    family = {
        name: pd.concat(
            [discovery.macro(scored[m], by=by, unit="family").assign(method=m) for m in order],
            ignore_index=True,
        )
        for name, by in (("overall", ()), ("by_class", ("sequence_class",)))
    }

    for name, frame in (
        ("overall", overall),
        ("by_class", by_class),
        ("by_objective", by_objective),
        ("by_cell", by_cell),
        ("overall_by_family", family["overall"]),
        ("by_class_by_family", family["by_class"]),
    ):
        frame.to_csv(args.out / f"{name}.csv", index=False)

    shown = [*discovery.HEADLINE, "assays_to_hit", "families", "panels"]
    chosen = ", ".join(f"{o}->{h}" for o, h in sorted(heads.items()))
    print(f"\n=== primary discovery test, K = {args.budget} wells (ranking head: {chosen}) ===")
    print(overall[["method", *shown]].to_string(index=False, float_format=_f))
    print("\n=== the same panels averaged by family rather than by panel ===")
    print(family["overall"][["method", *shown]].to_string(index=False, float_format=_f))
    print("\n=== by objective ===")
    print(by_objective[["assay_type", "method", *shown]].to_string(index=False, float_format=_f))
    print("\n=== by task cell ===")
    print(
        by_cell[["sequence_class", "assay_type", "method", *shown]].to_string(
            index=False, float_format=_f
        )
    )

    comparisons: dict = {"overall": {}, "by_objective": {}}
    for against in ("random", *(["zero_shot"] if args.zero_shot else [])):
        print(f"\n=== Pev - {against}, paired family-clustered bootstrap ===")
        block = {}
        for column in discovery.HEADLINE:
            result = discovery.compare(
                scored["pev"], scored[against], column, n_resamples=args.bootstrap, seed=args.seed
            )
            block[column] = result
            star = "*" if result["superior"] else " "
            print(
                f"  {column:<12} {result['point']:+.4f} "
                f"[{result['lo']:+.4f}, {result['hi']:+.4f}]{star}"
            )
        for objective in sorted(scored["pev"]["assay_type"].unique()):
            sub = discovery.compare(
                scored["pev"][scored["pev"]["assay_type"] == objective],
                scored[against][scored[against]["assay_type"] == objective],
                "enrichment",
                n_resamples=args.bootstrap,
                seed=args.seed,
            )
            block[f"enrichment:{objective}"] = sub
            print(
                f"    {objective:<10} enrichment {sub['point']:+.3f} "
                f"[{sub['lo']:+.3f}, {sub['hi']:+.3f}]{'*' if sub['superior'] else ' '}"
            )
        comparisons["overall"][against] = block

    print("\n=== Pev - random, by sequence class ===")
    comparisons["by_class"] = {}
    for key, column_name in (("by_class", "sequence_class"), ("by_objective", "assay_type")):
        if key == "by_objective":
            print("\n=== Pev - random, by objective ===")
        for group in sorted(scored["pev"][column_name].unique()):
            left = scored["pev"][scored["pev"][column_name] == group]
            right = scored["random"][scored["random"][column_name] == group]
            block = {
                c: discovery.compare(left, right, c, n_resamples=args.bootstrap, seed=args.seed)
                for c in discovery.PRIMARY
            }
            comparisons[key][group] = block
            cells = "  ".join(
                f"{c}: {block[c]['point']:+.3f} [{block[c]['lo']:+.3f}, {block[c]['hi']:+.3f}]"
                f"{'*' if block[c]['superior'] else ' '}"
                for c in discovery.PRIMARY
            )
            print(f"  {group:<12} {cells}")
    print("  * lower bound excludes zero")

    payload = {
        "heads": heads,
        "budget": args.budget,
        "min_candidates": args.min_candidates,
        "margin_sd": args.margin_sd,
        "meaningful_improvement": discovery.MEANINGFUL,
        "relative_sd_for_unitless_readouts": discovery.RELATIVE_SD,
        "panels": len(panels),
        "families": len({p.family_id for p in panels}),
        "candidates": int(sum(len(p) for p in panels)),
        "overall": overall.to_dict("records"),
        "by_class": by_class.to_dict("records"),
        "by_objective": by_objective.to_dict("records"),
        "by_cell": by_cell.to_dict("records"),
        "comparisons": comparisons,
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, default=_jsonable))
    print(f"\nwrote {args.out}")
    return 0


def _describe(panels, args) -> None:
    frame = pd.DataFrame(
        [
            {
                "sequence_class": p.sequence_class,
                "assay_type": p.assay_type,
                "family_id": p.family_id,
                "n": len(p),
                "prevalence": p.prevalence,
                "ceiling": min(1 / p.prevalence, len(p) / args.budget),
            }
            for p in panels
        ]
    )
    print(
        f"discovery panels: {len(panels)} over {frame['family_id'].nunique()} families, "
        f"{int(frame['n'].sum()):,} measured candidates"
    )
    table = frame.groupby(["sequence_class", "assay_type"]).agg(
        panels=("n", "size"),
        families=("family_id", "nunique"),
        median_candidates=("n", "median"),
        hit_rate=("prevalence", "median"),
        enrichment_ceiling=("ceiling", "median"),
    )
    print(table.to_string(float_format=_f))


def _load(checkpoint: Path, device: str):
    import torch

    if not (checkpoint / "metadata.json").exists():
        raise SystemExit(
            f"no checkpoint at {checkpoint}. The weights are not in this repository; fetch them with\n"
            f'  .venv/bin/hf download Yibooooo/Pev --include "checkpoints/*" --local-dir .'
        )
    cfg = config_from_metadata(json.loads((checkpoint / "metadata.json").read_text()))
    model = DecisionModel(cfg)
    if cfg.lora_last_n_layers:
        model.encoder.enable_lora()
    load_checkpoint(checkpoint, model)
    model.encoder.enable_context_cache()
    model.to(torch.device(device if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def _logits(model, panels) -> dict[str, dict[str, np.ndarray | None]]:
    """One encoder forward per panel; both heads come off the same pass."""
    import torch

    out: dict[str, dict] = {}
    for i, panel in enumerate(panels, 1):
        with torch.no_grad():
            scored = model.score(
                panel.reference_sequence,
                panel.edits,
                panel.assay_type,
                panel.readout_type,
                target_sequence=panel.target_sequence,
                partner_sequence=panel.partner_sequence,
            )
        out[panel.context_id] = {
            "choice": scored.choice_logits.double().cpu().numpy()[: len(panel)],
            "improve": (
                None
                if scored.improve_logits is None
                else scored.improve_logits.double().cpu().numpy()
            ),
        }
        if i % 20 == 0 or i == len(panels):
            log.info("scored %d/%d panels", i, len(panels))
    return out


def _scores(cached: dict, panel, head: str) -> np.ndarray:
    values = cached[panel.context_id].get(head)
    if values is None:
        values = cached[panel.context_id]["choice"]
    return np.asarray(values, dtype=float)[: len(panel)]
def _order(cached: dict, panels, head_for) -> dict:
    """One ranking per panel: the chosen head's score, descending."""
    return {
        panel.context_id: np.argsort(-_scores(cached, panel, head_for(panel)))
        for panel in panels
    }


def _frozen(panels: list, pins: Path) -> list:
    """Hold the run to the pinned benchmark.

    A benchmark is a fixed list of panels. Corpora grow, so a panel set rebuilt from
    whatever is on disk is quietly not the one the published numbers used. Every
    pinned panel must be present; anything else is dropped with a note.
    """
    wanted = [c["context_id"] for c in json.loads(pins.read_text())["benchmark"]["contexts"]]
    built = {p.context_id: p for p in panels}
    missing = [c for c in wanted if c not in built]
    if missing:
        raise SystemExit(
            f"{len(missing)} pinned panel(s) absent from this corpus -- "
            f"run scripts/check_data.py:\n  " + "\n  ".join(missing)
        )
    dropped = sorted(set(built) - set(wanted))
    if dropped:
        print(f"\npinned to {pins.name}: {len(dropped)} buildable panel(s) outside "
              f"the benchmark, not scored")
        for context_id in dropped:
            print(f"  - {context_id}")
    return [built[c] for c in wanted]
def _f(value) -> str:
    return f"{value:.3f}"


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
