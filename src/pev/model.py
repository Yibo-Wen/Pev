"""The decision model: one encoding of the parent, every edit scored in parallel.

The variants are configurations of one module rather than parallel
implementations, so the comparison between them is controlled. ``feature_builder``
decides whether candidate features come from the parent's encoding
(``parent_reuse``) or from a separate encoding of each mutant (``per_mutant``);
``head_mode`` decides the loss. Backbone, LoRA, conditioning and heads are shared.

The model is **history-free**: ``forward`` sees the current sequence and its
conditioning and nothing else, so campaign state is the caller's by construction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from pev.schema import STANDARD_AA, AssayType, ReadoutType

log = logging.getLogger(__name__)

FeatureBuilder = Literal["parent_reuse", "per_mutant"]
HeadMode = Literal["joint", "choice_only", "mse"]

#: 1022 residues plus <cls> and <eos>. ESM-2's tokenizer does not truncate on
#: its own -- `model_max_length` is the integer sentinel -- so this is passed
#: explicitly at every call.
MAX_TOKENS = 1024

#: Residues the encoder can actually read. Two of ``MAX_TOKENS`` go to ``<cls>``
#: and ``<eos>``, so an edit past this has no hidden state and cannot be scored.
USABLE_RESIDUES = MAX_TOKENS - 2

#: ``log p(mut) - log p(wt)`` and ``log p(wt)`` -- the substitution's evolutionary
#: plausibility and how conserved the position is.
N_LM_FEATURES = 2

#: ``max`` and ``mean`` of each candidate head output, so STOP knows what it is
#: turning down.
N_PANEL_FEATURES = 4

AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AA)}


@dataclass
class ModelConfig:
    backbone: str = "facebook/esm2_t33_650M_UR50D"
    hidden: int = 256
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    #: LoRA is confined to the final N layers, which also lets everything below
    #: them run under no_grad.
    lora_last_n_layers: int = 4
    #: Width of the assay / readout embeddings.
    cond_dim: int = 32
    #: Width of the projected target / partner embeddings.
    context_dim: int = 128
    feature_builder: FeatureBuilder = "parent_reuse"
    head_mode: HeadMode = "joint"
    #: Weight on the improvement loss. Zero gives the choice-only ablation.
    improvement_weight: float = 1.0
    #: Mutants encoded per forward in the per-mutant builder. A single batch of
    #: 256 does not fit at 650M on a 24 GB card, so this is what keeps
    #: The largest candidate budget scored in one request. Same for every method.
    per_mutant_batch: int = 32
    #: Feed each candidate ESM-2's own masked-LM opinion of the substitution:
    #: ``log p(mut | context) - log p(wt | context)`` and the wild type's log
    #: probability. `AutoModel` discards that head, so without this the candidate
    #: MLP relearns from `h_i` what the pretrained head already encodes.
    use_lm_features: bool = False
    #: Give the candidate its *wild-type* residue as well as its replacement. The
    #: effect of a substitution is a property of the pair, and the feature vector
    #: only ever carried the incoming residue.
    use_wt_residue: bool = False
    #: Let the STOP head see what is on offer. It was `g([h_bar; q_t])` -- asked
    #: whether to stop without being shown the panel -- which is why its
    #: precision sat at 0.17-0.22 and stopping cost 7% of GB1 final fitness.
    panel_aware_stop: bool = False
    max_length: int = MAX_TOKENS
    dropout: float = 0.1

    @property
    def is_regression(self) -> bool:
        return self.head_mode == "mse"

    @property
    def lambda_improve(self) -> float:
        return 0.0 if self.head_mode == "choice_only" else self.improvement_weight


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #


#: ``<prefix>encoder.layer.<i>.attention.self.<query|value>``. The prefix varies:
#: ``AutoModel`` gives a bare ``EsmModel`` with no prefix, while the task heads
#: nest it under ``esm.``. Matching the suffix keeps both working.
_ATTENTION_RE = re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.attention\.self\.(query|value)$")


def lora_target_modules(module: nn.Module, last_n: int) -> list[str]:
    """Resolve query/value module paths in the final ``last_n`` layers.

    Resolved rather than hardcoded for two reasons: ``peft`` has no ``esm`` entry in
    its target-module mapping, and a bare ``["query", "value"]`` would match every
    layer instead of the final few, quietly adapting all 33.
    """
    found: list[tuple[int, str]] = []
    for name, _ in module.named_modules():
        if m := _ATTENTION_RE.search(name):
            found.append((int(m.group(1)), name))
    if not found:
        raise RuntimeError("no ESM attention query/value modules found to attach LoRA to")

    highest = max(idx for idx, _ in found)
    cutoff = highest - last_n + 1
    return sorted(name for idx, name in found if idx >= cutoff)


def _load_backbone(cfg: ModelConfig):
    """The encoder, and ESM-2's masked-LM head when the variant wants it.

    ``AutoModel`` drops ``lm_head``; loading the masked-LM model keeps both. The head
    is a linear map on hidden states, so it scores all 20 substitutions at a position
    from the encoding already computed -- no second pass, parent reuse untouched.
    """
    from transformers import AutoModel, AutoModelForMaskedLM

    if not cfg.use_lm_features:
        return AutoModel.from_pretrained(cfg.backbone), None
    full = AutoModelForMaskedLM.from_pretrained(cfg.backbone)
    return full.esm, full.lm_head


class Encoder(nn.Module):
    """ESM-2 with LoRA on the last few layers.

    Layers below the first LoRA layer can never receive gradients, so they run
    under ``no_grad``. That saves activation memory for *both* model variants
    equally, which matters because the two are compared on cost.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        backbone, self.lm_head = _load_backbone(cfg)
        self.hidden_size = backbone.config.hidden_size
        self.n_layers = backbone.config.num_hidden_layers

        for p in backbone.parameters():
            p.requires_grad_(False)
        if self.lm_head is not None:
            for p in self.lm_head.parameters():
                p.requires_grad_(False)

        self.backbone = backbone
        self._lora_enabled = False
        self._context_cache: dict[str, Tensor] | None = None
        self._context_cache_limit = 0

    def enable_lora(self) -> None:
        """Attach LoRA adapters. Called when fine-tuning starts, not before."""
        if self._lora_enabled:
            return
        from peft import LoraConfig, get_peft_model

        targets = lora_target_modules(self.backbone, self.cfg.lora_last_n_layers)
        peft_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=targets,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)
        self._lora_enabled = True
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        log.info("LoRA enabled on %d modules (%d trainable params)", len(targets), trainable)

    def tokenize(self, sequences: list[str], device: torch.device) -> dict[str, Tensor]:
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def forward(self, sequences: list[str]) -> tuple[Tensor, Tensor, Tensor]:
        """Encode sequences.

        Returns per-residue states ``H`` (B, L, D), the mean-pooled ``h_bar``
        (B, D), and the residue mask (B, L) with special tokens removed so that
        pooling and position lookup both address real residues.
        """
        device = next(self.parameters()).device
        batch = self.tokenize(sequences, device)
        out = self.backbone(**batch)
        states = out.last_hidden_state

        mask = batch["attention_mask"].bool().clone()
        # Drop <cls>/<eos> so index 0 of the trimmed tensor is residue 1.
        mask[:, 0] = False
        lengths = batch["attention_mask"].sum(dim=1)
        mask[torch.arange(mask.size(0), device=device), lengths - 1] = False

        weights = mask.unsqueeze(-1).to(states.dtype)
        h_bar = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return states, h_bar, mask

    def enable_context_cache(self, max_entries: int = 512) -> None:
        """Memoise frozen target/partner embeddings by sequence.

        Without it every ``score()`` re-encodes the target and partner from scratch,
        which for a long target dominates the request. Enabled identically for every
        method, and warming it is timed outside the request. Safe because these come
        from the frozen backbone under ``no_grad``; LoRA never touches them.
        """
        self._context_cache = {}
        self._context_cache_limit = max_entries

    def clear_context_cache(self) -> None:
        if self._context_cache is not None:
            self._context_cache.clear()

    @property
    def context_cache_size(self) -> int:
        return 0 if self._context_cache is None else len(self._context_cache)

    @torch.no_grad()
    def residue_log_probs(self, states: Tensor) -> Tensor:
        """``log p(residue | context)`` at every position, from the pretrained head.

        Frozen and under ``no_grad``: this is ESM-2's evolutionary prior, used as
        a feature rather than something to fine-tune.
        """
        if self.lm_head is None:
            raise RuntimeError("this encoder was built without the masked-LM head")
        return torch.log_softmax(self.lm_head(states).float(), dim=-1)

    def encode_frozen(self, sequences: list[str]) -> Tensor:
        """Mean-pooled states with no gradient -- for target/partner context."""
        cache = self._context_cache
        if cache is None:
            with torch.no_grad():
                _, h_bar, _ = self.forward(sequences)
            return h_bar.detach()

        wanted = dict.fromkeys(sequences)
        resolved = {s: cache[s] for s in wanted if s in cache}
        missing = [s for s in wanted if s not in resolved]
        if missing:
            with torch.no_grad():
                _, h_bar, _ = self.forward(missing)
            for seq, row in zip(missing, h_bar.detach(), strict=True):
                resolved[seq] = row

        # Answer from `resolved`, never from the cache: a request larger than the
        # cache limit would otherwise evict a sequence it still has to return.
        for seq in missing:
            while len(cache) >= self._context_cache_limit and cache:
                cache.pop(next(iter(cache)))
            if self._context_cache_limit:
                cache[seq] = resolved[seq]
        return torch.stack([resolved[s] for s in sequences])


# --------------------------------------------------------------------------- #
# Conditioning
# --------------------------------------------------------------------------- #


class Conditioner(nn.Module):
    """Builds ``q_t = [e_assay; e_readout; b]``.

    ``b`` holds projected, frozen target and partner embeddings together with
    their absence masks, so a context with no recorded partner is represented
    explicitly rather than as a zero vector that could be confused with a real
    embedding.
    """

    def __init__(self, cfg: ModelConfig, hidden_size: int) -> None:
        super().__init__()
        self.assay = nn.Embedding(len(AssayType), cfg.cond_dim)
        self.readout = nn.Embedding(len(ReadoutType), cfg.cond_dim)
        self.target_proj = nn.Linear(hidden_size, cfg.context_dim)
        self.partner_proj = nn.Linear(hidden_size, cfg.context_dim)
        self.out_dim = 2 * cfg.cond_dim + 2 * (cfg.context_dim + 1)

    def forward(
        self,
        assay_idx: Tensor,
        readout_idx: Tensor,
        target_emb: Tensor | None,
        target_mask: Tensor | None,
        partner_emb: Tensor | None,
        partner_mask: Tensor | None,
    ) -> Tensor:
        batch = assay_idx.size(0)
        device = assay_idx.device
        parts = [self.assay(assay_idx), self.readout(readout_idx)]

        for emb, mask, proj in (
            (target_emb, target_mask, self.target_proj),
            (partner_emb, partner_mask, self.partner_proj),
        ):
            if emb is None:
                width = proj.out_features
                parts.append(torch.zeros(batch, width, device=device))
                parts.append(torch.zeros(batch, 1, device=device))
            else:
                present = (
                    mask.to(emb.dtype).unsqueeze(-1)
                    if mask is not None
                    else torch.ones(batch, 1, device=device, dtype=emb.dtype)
                )
                parts.append(proj(emb) * present)
                parts.append(present)
        return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #


class CandidateHead(nn.Module):
    """Two-layer MLP over candidate features.

    The final linear carries **two** outputs: the choice logit ``z_a`` and the
    improvement logit ``v_a``, which reuse the same hidden features. The
    regression variant replaces it with a single scalar.
    """

    def __init__(self, in_dim: int, hidden: int, outputs: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, outputs)

    def forward(self, features: Tensor) -> Tensor:
        return self.out(self.net(features))


class StopHead(nn.Module):
    """``z_STOP = g([h_bar; q_t])`` -- a separate scalar, not one of the edits."""

    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


@dataclass(frozen=True)
class ScoreRequest:
    """One episode's inputs. The unit ``score_batch`` batches over."""

    sequence: str
    edits: tuple[tuple[int, str], ...]
    assay: AssayType
    readout: ReadoutType
    target_sequence: str | None = None
    partner_sequence: str | None = None

    def token_cost(self, *, per_mutant: bool) -> int:
        """Tokens one forward spends on this episode, padding aside.

        The parent is always encoded -- the STOP head reads its pooled state for
        both feature builders -- and ``per_mutant`` encodes every candidate on top.
        """
        return self.tokens * (1 + len(self.edits)) if per_mutant else self.tokens

    @property
    def tokens(self) -> int:
        """Encoder positions, after truncation."""
        return min(len(self.sequence) + 2, MAX_TOKENS)

    def attention_cost(self, *, per_mutant: bool) -> int:
        """Cost that grows with the *square* of the length.

        A token budget alone is the wrong guard: attention is O(L^2) while the
        feed-forward layers are O(L), so a batch that is legal on tokens can still
        exhaust the card. This corpus runs from a median parent of 63 residues to a
        99th percentile of 1,273, and bounding both keeps one setting safe across it.
        """
        sequences = 1 + len(self.edits) if per_mutant else 1
        return sequences * self.tokens * self.tokens


@dataclass
class ScoreOutput:
    """Raw, uncalibrated outputs for one episode's panel."""

    choice_logits: Tensor  # (n_candidates + 1,), STOP last
    improve_logits: Tensor | None  # (n_candidates,)
    predicted_gain: Tensor | None  # (n_candidates,), regression variant only
    candidate_ids: list[str] = field(default_factory=list)

    @property
    def stop_logit(self) -> Tensor:
        return self.choice_logits[-1]


class DecisionModel(nn.Module):
    """Scores every allowed single substitution in parallel, plus STOP."""

    def __init__(self, cfg: ModelConfig, encoder: Encoder | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder or Encoder(cfg)
        d = self.encoder.hidden_size

        self.conditioner = Conditioner(cfg, d)
        self.residue = nn.Embedding(len(STANDARD_AA), cfg.cond_dim)

        # [h_i ; h_bar ; e_r ; q_t]
        candidate_dim = 2 * d + cfg.cond_dim + self.conditioner.out_dim
        if cfg.use_wt_residue:
            candidate_dim += cfg.cond_dim
        if cfg.use_lm_features:
            candidate_dim += N_LM_FEATURES
        outputs = 1 if cfg.is_regression else 2
        self.candidate_head = CandidateHead(candidate_dim, cfg.hidden, outputs, cfg.dropout)
        stop_dim = d + self.conditioner.out_dim
        if cfg.panel_aware_stop:
            stop_dim += N_PANEL_FEATURES
        self.stop_head = StopHead(stop_dim, cfg.hidden, cfg.dropout)

    # -- feature construction ------------------------------------------------ #

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
        """Score one parent's candidate panel.

        A thin wrapper over :meth:`score_batch`, so the batched training path and
        the single-request evaluation path cannot drift apart.
        """
        return self.score_batch(
            [
                ScoreRequest(
                    sequence=sequence,
                    edits=tuple(edits),
                    assay=assay,
                    readout=readout,
                    target_sequence=target_sequence,
                    partner_sequence=partner_sequence,
                )
            ]
        )[0]

    def score_batch(self, requests: list[ScoreRequest]) -> list[ScoreOutput]:
        """Score several episodes in one encoder forward.

        One episode at a time leaves the GPU idle -- a 56-residue domain is 58 tokens.
        Batching the parent encodes gives thousands of tokens per forward at the same
        memory, which is where the speedup comes from; ``edits`` are cheap either way.

        Padding is right-side and the tokenizer's attention mask keeps it out of both
        attention and the mean pool, so a 1-based residue position indexes the token
        dimension correctly whatever the batch's longest sequence is.
        """
        if not requests:
            return []

        over = [
            (r.sequence[:8], pos)
            for r in requests
            for pos, _ in r.edits
            if pos > MAX_TOKENS - 2
        ]
        if over:
            raise ValueError(
                f"{len(over)} edit(s) sit past residue {MAX_TOKENS - 2}, which the "
                f"encoder truncates away (first: position {over[0][1]} of a sequence "
                f"starting {over[0][0]!r}). Panel construction filters these out; "
                "a request built by hand has to respect the window too."
            )

        device = next(self.parameters()).device
        counts = [len(r.edits) for r in requests]
        positions = torch.tensor(
            [p for r in requests for p, _ in r.edits], dtype=torch.long, device=device
        )
        residues = torch.tensor(
            [AA_INDEX[res] for r in requests for _, res in r.edits],
            dtype=torch.long,
            device=device,
        )
        owner = torch.repeat_interleave(
            torch.arange(len(requests), device=device),
            torch.tensor(counts, dtype=torch.long, device=device),
        )

        # The parent is encoded for both builders: the STOP head reads its pooled
        # state, which is the one thing per-mutant does not get from its mutants.
        states, h_bar, _ = self.encoder([r.sequence for r in requests])
        q_t = torch.cat(
            [
                self._conditioning(
                    r.assay, r.readout, r.target_sequence, r.partner_sequence, device
                )
                for r in requests
            ],
            dim=0,
        )

        if sum(counts) == 0:
            head_out = torch.zeros(
                (0, 1 if self.cfg.is_regression else 2), device=device, dtype=h_bar.dtype
            )
        else:
            if self.cfg.feature_builder == "parent_reuse":
                h_i, pooled = states[owner, positions], h_bar[owner]
            else:
                h_i, pooled = self._mutant_features(requests, positions)

            parts = [h_i, pooled, self.residue(residues), q_t[owner]]
            if self.cfg.use_wt_residue:
                parts.append(self.residue(self._wild_type_indices(requests, device)))
            if self.cfg.use_lm_features:
                # Always read off the *parent* encoding, for both feature
                # builders. The substitution's evolutionary plausibility is a
                # property of (context, position, wt, mut); computing it from the
                # mutant for one variant and the parent for the other would make
                # the LM head part of the controlled difference, which it is not.
                parts.append(self._lm_features(states, requests, owner, positions, residues))
            head_out = self.candidate_head(torch.cat(parts, dim=-1))

        stop_parts = [h_bar, q_t]
        if self.cfg.panel_aware_stop:
            stop_parts.append(self._panel_summary(head_out, counts, h_bar))
        stop_logits = self.stop_head(torch.cat(stop_parts, dim=-1)).reshape(-1)

        outputs: list[ScoreOutput] = []
        offset = 0
        for b, n in enumerate(counts):
            piece = head_out[offset : offset + n]
            offset += n
            stop = stop_logits[b : b + 1]
            if self.cfg.is_regression:
                gain = piece.reshape(-1)
                # A regression model has no choice distribution; expose the
                # ranking it does induce so the harness can treat it uniformly.
                outputs.append(
                    ScoreOutput(
                        choice_logits=torch.cat([gain, stop]),
                        improve_logits=None,
                        predicted_gain=gain,
                    )
                )
            else:
                z_edit, v_edit = piece.unbind(-1)
                outputs.append(
                    ScoreOutput(
                        choice_logits=torch.cat([z_edit, stop]),
                        improve_logits=v_edit,
                        predicted_gain=None,
                    )
                )
        return outputs

    def _wild_type_indices(self, requests: list[ScoreRequest], device) -> Tensor:
        """The residue each candidate replaces, flattened across the batch."""
        return torch.tensor(
            [
                AA_INDEX.get(r.sequence[pos - 1], 0)
                for r in requests
                for pos, _ in r.edits
            ],
            dtype=torch.long,
            device=device,
        )

    def _lm_features(
        self,
        states: Tensor,
        requests: list[ScoreRequest],
        owner: Tensor,
        positions: Tensor,
        residues: Tensor,
    ) -> Tensor:
        """``[log p(mut) - log p(wt), log p(wt)]`` per candidate."""
        log_probs = self.encoder.residue_log_probs(states)
        vocab = self.encoder.tokenizer.convert_tokens_to_ids(list(STANDARD_AA))
        table = torch.tensor(vocab, dtype=torch.long, device=states.device)

        wt = self._wild_type_indices(requests, states.device)
        at_position = log_probs[owner, positions]  # (candidates, vocab)
        mut_lp = at_position.gather(1, table[residues].unsqueeze(1)).squeeze(1)
        wt_lp = at_position.gather(1, table[wt].unsqueeze(1)).squeeze(1)
        return torch.stack([mut_lp - wt_lp, wt_lp], dim=-1).to(states.dtype)

    def _panel_summary(self, head_out: Tensor, counts: list[int], h_bar: Tensor) -> Tensor:
        """What each episode's panel looks like, for the STOP head.

        ``max`` and ``mean`` of the candidate head's outputs. STOP is a decision
        about whether anything on offer is worth taking, and it was previously
        made without reference to the offer.
        """
        summary = h_bar.new_zeros((len(counts), N_PANEL_FEATURES))
        offset = 0
        for b, n in enumerate(counts):
            if n == 0:
                offset += n
                continue
            block = head_out[offset : offset + n]
            offset += n
            stats = [block.max(dim=0).values, block.mean(dim=0)]
            flat = torch.cat(stats)[: N_PANEL_FEATURES]
            summary[b, : flat.numel()] = flat
        return summary

    def _mutant_features(
        self, requests: list[ScoreRequest], positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode every candidate mutant across the batch, in micro-chunks.

        Batched rather than one at a time, so the per-mutant baseline is not
        handicapped. Chunking is not a tuning knob: on a 24 GB card at 650M a batch of
        256 mutants does not fit, and ``per_mutant_batch`` applies identically wherever
        this builder runs, so the comparison stays matched.
        """
        mutants: list[str] = []
        for request in requests:
            for pos, residue in request.edits:
                chars = list(request.sequence)
                chars[pos - 1] = residue
                mutants.append("".join(chars))

        chunk = max(1, self.cfg.per_mutant_batch)
        h_i_parts: list[Tensor] = []
        pooled_parts: list[Tensor] = []
        for start in range(0, len(mutants), chunk):
            batch = mutants[start : start + chunk]
            states, pooled, _ = self.encoder(batch)
            index = positions[start : start + len(batch)]
            h_i_parts.append(states[torch.arange(states.size(0), device=states.device), index])
            pooled_parts.append(pooled)
        return torch.cat(h_i_parts, dim=0), torch.cat(pooled_parts, dim=0)

    def _conditioning(
        self,
        assay: AssayType,
        readout: ReadoutType,
        target_sequence: str | None,
        partner_sequence: str | None,
        device: torch.device,
    ) -> Tensor:
        assay_idx = torch.tensor([list(AssayType).index(assay)], device=device)
        readout_idx = torch.tensor([list(ReadoutType).index(readout)], device=device)

        def embed(seq: str | None) -> tuple[Tensor | None, Tensor | None]:
            if not seq:
                return None, None
            emb = self.encoder.encode_frozen([seq])
            return emb, torch.ones(1, device=device)

        target_emb, target_mask = embed(target_sequence)
        partner_emb, partner_mask = embed(partner_sequence)
        return self.conditioner(
            assay_idx, readout_idx, target_emb, target_mask, partner_emb, partner_mask
        )

    # -- parameter groups ---------------------------------------------------- #

    def small_module_parameters(self):
        """Heads and conditioning -- everything trained from the warm start on."""
        yield from self.conditioner.parameters()
        yield from self.residue.parameters()
        yield from self.candidate_head.parameters()
        yield from self.stop_head.parameters()

    def lora_parameters(self):
        return (p for n, p in self.encoder.named_parameters() if p.requires_grad and "lora" in n)


