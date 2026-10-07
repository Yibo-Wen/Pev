"""Loading a trained checkpoint for evaluation.

The backbone is deliberately absent from a checkpoint: it is frozen, far larger
than what was trained, and reconstructible by name from Hugging Face. So
``load_checkpoint`` refuses a file with keys the model has no slot for, and
``backbone_fingerprint`` detects a backbone that is not the one the weights were
trained against.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from pev.model import DecisionModel, ModelConfig

log = logging.getLogger(__name__)


def backbone_fingerprint(model: DecisionModel) -> str:
    """A digest of the frozen backbone.

    Since the backbone is not saved, a silently different cached copy would change
    every reported number with nothing to show it. This is what makes that
    detectable.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.encoder.backbone.state_dict().items()):
        # LoRA is saved separately, so it is not part of the frozen backbone. The
        # pooler is excluded because ESM-2 ships without one and `EsmModel`
        # initialises it randomly, which would change the digest on every load.
        if "lora" in name or ".pooler." in f".{name}.":
            continue
        # Enabling LoRA renames every backbone key without moving a weight: a
        # `base_model.model.` prefix on all of them, plus a `.base_layer` segment
        # on the adapted projections. Undoing both makes the digest a property of
        # the weights, so a fingerprint recorded before LoRA was enabled still
        # matches after it is.
        canonical = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        digest.update(canonical.encode())
        digest.update(tensor.detach().cpu().to(torch.float32).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(directory: Path | str, model: DecisionModel) -> dict:
    """Load the trainable weights onto a freshly constructed backbone.

    ``strict=False`` is required because the backbone is absent from the file by
    design -- so every key the file *does* carry must match, or the load would be
    a silent no-op and evaluation would run on randomly initialised heads.
    """
    directory = Path(directory)
    state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    report = model.load_state_dict(state, strict=False)
    unexpected = set(report.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            f"{directory} carries {len(unexpected)} key(s) this model has no slot for, "
            f"e.g. {sorted(unexpected)[:3]} -- did LoRA get enabled before loading?"
        )
    meta = json.loads((directory / "metadata.json").read_text())
    saved = meta.get("backbone_fingerprint")
    if saved:
        current = backbone_fingerprint(model)
        if current != saved:
            log.warning(
                "backbone fingerprint mismatch for %s (%s != %s): the cached "
                "%s differs from the one trained against",
                directory,
                current[:12],
                saved[:12],
                meta.get("model_config", {}).get("backbone", "backbone"),
            )
    return meta


def config_from_metadata(meta: dict) -> ModelConfig:
    """The architecture a checkpoint was trained with, as recorded in its metadata."""
    return ModelConfig(**meta["model_config"])
