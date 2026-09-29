"""Character level autoregressive transformer over the amino acid alphabet.

Deliberately small. A four layer model with a hidden width of 128 is about one
million parameters, trains in under ten minutes on a single consumer GPU, and
learns the compositional bias, length distribution and the three-to-four residue
periodicity that produces an amphipathic face, without any of that being
specified.

Chosen over a variational autoencoder for three reasons. Sequence variational
autoencoders fail in ways that cost hours to diagnose, principally posterior
collapse, where the decoder learns to ignore the latent and the model silently
degenerates into an unconditional language model. Latent space optimisation
against a learned oracle is the most reward-hackable machinery in this field.
And the organising laboratory's own reference model is a conditional variational
autoencoder, so that is not the axis to compete on.

Breadth comes from here. Optimisation comes from the explicit search in
search.py, where hard constraints can be built into the proposal distribution
rather than hoped for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import ALPHABET, MAX_LENGTH, MIN_LENGTH

BOS, EOS, PAD = 0, 1, 2
N_SPECIAL = 3
VOCAB_SIZE = N_SPECIAL + len(ALPHABET)

# Conditioning tokens appended to the vocabulary: coarse net charge buckets and
# provenance. Provenance separates catalogued antimicrobial peptides from
# validated cryptic actives found inside proteins never selected for immunity,
# which occupy a different region of sequence space and are further from the
# reference databases the novelty filter screens against.
CHARGE_BUCKETS = (-2.0, 2.0, 4.0, 6.0, 8.0, 12.0)
N_CHARGE_TOKENS = len(CHARGE_BUCKETS) + 1
N_SOURCE_TOKENS = 2
COND_OFFSET = VOCAB_SIZE
TOTAL_VOCAB = VOCAB_SIZE + N_CHARGE_TOKENS + N_SOURCE_TOKENS

AA_TO_ID = {a: i + N_SPECIAL for i, a in enumerate(ALPHABET)}
ID_TO_AA = {i + N_SPECIAL: a for i, a in enumerate(ALPHABET)}


def charge_token(charge: float) -> int:
    idx = sum(charge >= b for b in CHARGE_BUCKETS)
    return COND_OFFSET + idx


def source_token(is_cryptic: bool) -> int:
    return COND_OFFSET + N_CHARGE_TOKENS + int(is_cryptic)


def encode(seq: str, charge: float, is_cryptic: bool = False) -> list[int]:
    return (
        [charge_token(charge), source_token(is_cryptic), BOS]
        + [AA_TO_ID[a] for a in seq]
        + [EOS]
    )


def decode(ids) -> str:
    return "".join(ID_TO_AA[i] for i in ids if i in ID_TO_AA)


class PeptideLM(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int = 512, dropout: float = 0.1, max_len: int = MAX_LENGTH + 8):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token = nn.Embedding(TOTAL_VOCAB, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, TOTAL_VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.token(x) + self.position(pos)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=mask, src_key_padding_mask=(x == PAD))
        return self.head(self.norm(h))


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_every: int = 1


def _collate(batch, device):
    width = max(len(b) for b in batch)
    padded = torch.full((len(batch), width), PAD, dtype=torch.long)
    for i, ids in enumerate(batch):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded.to(device)


def train(model: PeptideLM, encoded, cfg: TrainConfig,
          checkpoint_path: str | Path | None = None) -> PeptideLM:
    """Standard next token objective. Checkpoints every epoch.

    Checkpointing is not optional: on a free tier hosted runtime a disconnect
    partway through a long session with nothing written is the failure mode that
    ends the submission.
    """
    torch.manual_seed(cfg.seed)
    model = model.to(cfg.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(encoded) / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    generator = torch.Generator().manual_seed(cfg.seed)

    for epoch in range(cfg.epochs):
        order = torch.randperm(len(encoded), generator=generator).tolist()
        running, n_batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            batch = [encoded[i] for i in order[start : start + cfg.batch_size]]
            x = _collate(batch, cfg.device)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, TOTAL_VOCAB), x[:, 1:].reshape(-1), ignore_index=PAD
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss.item())
            n_batches += 1
        if checkpoint_path and (epoch + 1) % cfg.checkpoint_every == 0:
            save(model, checkpoint_path, epoch=epoch + 1, loss=running / n_batches)
    return model


@torch.no_grad()
def sample(model: PeptideLM, n: int, seed: int = 42, temperature: float = 1.0,
           top_p: float = 0.95, min_len: int = MIN_LENGTH, max_len: int = MAX_LENGTH,
           charge_target: float | None = None, cryptic: bool = False,
           device: str | None = None, batch_size: int = 512):
    """Nucleus sampling with a fixed generator, so output is reproducible.

    Reproducibility is a co-authorship requirement: running the generation script
    twice must produce identical output.
    """
    device = device or next(model.parameters()).device.type
    model = model.to(device).eval()
    g = torch.Generator(device=device).manual_seed(seed)

    out: list[str] = []
    while len(out) < n:
        b = min(batch_size, (n - len(out)) * 2)
        charge = 6.0 if charge_target is None else charge_target
        prefix = torch.tensor(
            [charge_token(charge), source_token(cryptic), BOS], dtype=torch.long, device=device
        )
        x = prefix.unsqueeze(0).expand(b, 3).contiguous()
        done = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len + 1):
            logits = model(x)[:, -1, :]
            logits[:, COND_OFFSET:] = -float("inf")
            logits[:, PAD] = -float("inf")
            logits[:, BOS] = -float("inf")
            if x.shape[1] - 3 < min_len:
                logits[:, EOS] = -float("inf")

            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            cutoff = cumulative - sorted_probs > top_p
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            picked = torch.multinomial(sorted_probs, num_samples=1, generator=g)
            nxt = sorted_idx.gather(-1, picked)
            nxt[done] = PAD
            x = torch.cat([x, nxt], dim=1)
            done = done | (nxt.squeeze(-1) == EOS)
            if bool(done.all()):
                break

        for row in x[:, 3:].tolist():
            seq = decode(row)
            if min_len <= len(seq) <= max_len:
                out.append(seq)
            if len(out) >= n:
                break
    return out[:n]


def save(model: PeptideLM, path: str | Path, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "d_model": model.d_model,
                "max_len": model.max_len,
            },
            "meta": meta,
        },
        path,
    )


def load(path: str | Path, device: str = "cpu") -> PeptideLM:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = PeptideLM(d_model=blob["config"]["d_model"], max_len=blob["config"]["max_len"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()
