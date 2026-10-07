"""The untrained comparator: what ESM-2 already knows.

``MutateEverythingModel`` reads every substitution off ESM-2's own masked-LM head
in one unmasked pass over the parent -- wild-type marginals, no trained
parameters -- which separates what Pev's training bought from what the pretrained
prior brought. The random floor is not here: it needs no model, and
``discovery.random_rankings`` draws it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pev.model import AA_INDEX, Encoder, ModelConfig, ScoreOutput
from pev.schema import STANDARD_AA, AssayType, ReadoutType

# --------------------------------------------------------------------------- #


class MutateEverythingModel(nn.Module):
    """Score every offered substitution from one unmasked pass over the parent.

    Zero training: no LoRA, no heads, no calibration, just ESM-2's pretrained
    masked-LM head read at every position. One backbone pass whatever the candidate
    count, so it is a parallel-scoring comparison rather than a per-mutant one, and
    it ignores the assay conditioning a masked-LM has no notion of.

    These are **wild-type marginals**: the scored position is not masked first,
    because masking each one would cost a forward pass per position. That is the
    standard approximation (Meier et al. 2021) and generally weaker than a true
    masked-marginal score, so a gap here is partly the approximation's.
    """

    name = "mutate_everything"

    def __init__(self, backbone: str = "facebook/esm2_t33_650M_UR50D") -> None:
        super().__init__()
        cfg = ModelConfig(backbone=backbone, use_lm_features=True, lora_last_n_layers=0)
        self.encoder = Encoder(cfg)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score(
        self,
        sequence: str,
        edits: list[tuple[int, str]],
        assay: AssayType,
        readout: ReadoutType,
        *,
        target_sequence: str | None = None,
        partner_sequence: str | None = None,
    ) -> ScoreOutput:
        """One encoder forward pass over the parent; every edit is read off it.

        ``assay``, ``readout``, ``target_sequence`` and ``partner_sequence`` are
        accepted only so this matches :meth:`DecisionModel.score`'s signature and
        can stand in for it unmodified; all four are ignored.
        """
        device = next(self.parameters()).device
        states, _, _ = self.encoder([sequence])
        log_probs = self.encoder.residue_log_probs(states)  # (1, L, vocab)
        vocab = self.encoder.tokenizer.convert_tokens_to_ids(list(STANDARD_AA))
        table = torch.tensor(vocab, dtype=torch.long, device=device)

        if not edits:
            return ScoreOutput(
                choice_logits=torch.zeros(1, device=device),
                improve_logits=None,
                predicted_gain=None,
            )

        positions = torch.tensor([pos for pos, _ in edits], dtype=torch.long, device=device)
        mut_idx = torch.tensor([AA_INDEX[r] for _, r in edits], dtype=torch.long, device=device)
        wt_idx = torch.tensor(
            [AA_INDEX.get(sequence[pos - 1], 0) for pos, _ in edits],
            dtype=torch.long,
            device=device,
        )

        at_position: Tensor = log_probs[0, positions]  # (n_edits, vocab)
        mut_lp = at_position.gather(1, table[mut_idx].unsqueeze(1)).squeeze(1)
        wt_lp = at_position.gather(1, table[wt_idx].unsqueeze(1)).squeeze(1)
        scores = (mut_lp - wt_lp).float()

        choice_logits = torch.cat([scores, scores.new_zeros(1)])
        return ScoreOutput(choice_logits=choice_logits, improve_logits=None, predicted_gain=None)


def load_mutate_everything(
    backbone: str = "facebook/esm2_t33_650M_UR50D", *, device: str = "cuda"
) -> MutateEverythingModel:
    """Construct and place the zero-shot baseline -- no checkpoint to load."""
    model = MutateEverythingModel(backbone=backbone)
    resolved = device if torch.cuda.is_available() else "cpu"
    model.to(torch.device(resolved))
    model.eval()
    return model
