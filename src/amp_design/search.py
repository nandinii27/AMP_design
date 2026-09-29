"""Register-aware local search over de novo scaffolds.

The mutation operators know the helix geometry. An alpha helix advances one
hundred degrees per residue, so positions separated by three or four residues
land on the same face. Substituting a residue on the hydrophobic face is a
different move from substituting one on the polar face, and that distinction is
most of what rational peptide design knows.

Insertions and deletions are handled separately and kept rare. A single indel
rotates the register of every downstream residue by one hundred degrees, which
can create or destroy an amphipathic face wholesale. They are good for escaping
a local optimum and useless for fine tuning, so the objective is recomputed from
scratch after one rather than assumed local.

Search starts from language model samples, never from natural templates. The
eighty percent identity ceiling means a top-100 candidate needs at least four
edits from every known antibacterial peptide on a twenty residue sequence, so
the classical approach of scanning substitutions on a natural peptide is ruled
out by the rules themselves.

Everything is seeded and order-deterministic: two runs with the same seed must
produce byte identical output, which the organisers verify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    ALPHABET,
    FORBIDDEN_RESIDUES,
    HELIX_DEGREES_PER_RESIDUE,
    HYDROPHOBIC,
    TOPK_MAX_LENGTH,
    TOPK_MIN_LENGTH,
)

CATIONIC_CHOICES = tuple("KR")
HYDROPHOBIC_CHOICES = tuple("AILMFVW")
POLAR_CHOICES = tuple("GNQST")
ANIONIC_CHOICES = tuple("DE")
ALLOWED = tuple(a for a in ALPHABET if a not in FORBIDDEN_RESIDUES)

CLASS_CHOICES = {
    "cationic": CATIONIC_CHOICES,
    "hydrophobic": HYDROPHOBIC_CHOICES,
    "polar": POLAR_CHOICES,
    "anionic": ANIONIC_CHOICES,
}


def residue_class(aa: str) -> str:
    if aa in "KRH":
        return "cationic"
    if aa in HYDROPHOBIC:
        return "hydrophobic"
    if aa in "DE":
        return "anionic"
    return "polar"


def wheel_angle(position: int) -> float:
    return (position * HELIX_DEGREES_PER_RESIDUE) % 360.0


def estimate_face_centre(seq: str) -> float:
    """Circular mean of the hydrophobic residues' wheel angles.

    Estimated per sequence so the operator follows the face the molecule
    actually has rather than an assumed one.
    """
    angles = np.deg2rad([wheel_angle(i) for i, a in enumerate(seq) if a in HYDROPHOBIC])
    if angles.size == 0:
        return 0.0
    return float(np.rad2deg(np.angle(np.exp(1j * angles).mean())) % 360.0)


def on_face(position: int, face_centre: float, half_width: float = 90.0) -> bool:
    delta = abs((wheel_angle(position) - face_centre + 180.0) % 360.0 - 180.0)
    return delta <= half_width


@dataclass
class MutationOperator:
    """Proposal distribution with hard constraints built in.

    Infeasible sequences are never proposed rather than proposed and penalised.
    That is both faster and impossible to game.
    """

    p_class_preserving: float = 0.40
    p_face_aware: float = 0.30
    p_swap: float = 0.15
    p_indel: float = 0.08
    p_terminal: float = 0.07
    min_len: int = TOPK_MIN_LENGTH
    max_len: int = TOPK_MAX_LENGTH

    def __post_init__(self) -> None:
        w = np.array([
            self.p_class_preserving, self.p_face_aware,
            self.p_swap, self.p_indel, self.p_terminal,
        ], dtype=float)
        self._weights = w / w.sum()

    def __call__(self, seq: str, rng: np.random.Generator) -> str:
        move = int(rng.choice(5, p=self._weights))
        if move == 0:
            return self._class_preserving(seq, rng)
        if move == 1:
            return self._face_aware(seq, rng)
        if move == 2:
            return self._swap(seq, rng)
        if move == 3:
            return self._indel(seq, rng)
        return self._terminal(seq, rng)

    def _pick(self, pool, current, rng):
        choices = [c for c in pool if c != current]
        return choices[int(rng.integers(len(choices)))] if choices else current

    def _class_preserving(self, seq: str, rng) -> str:
        i = int(rng.integers(len(seq)))
        return seq[:i] + self._pick(CLASS_CHOICES[residue_class(seq[i])], seq[i], rng) + seq[i + 1:]

    def _face_aware(self, seq: str, rng) -> str:
        """Substitute according to which side of the helix the position is on.

        On the hydrophobic face, either reinforce it or deliberately break it.
        Breaking the face is the selectivity move: an interrupted hydrophobic
        surface still binds an anionic bacterial interface but cannot form the
        deep contiguous contact that insertion into a cholesterol rich mammalian
        bilayer requires.
        """
        centre = estimate_face_centre(seq)
        i = int(rng.integers(len(seq)))
        if on_face(i, centre):
            pool = HYDROPHOBIC_CHOICES if rng.random() < 0.7 else POLAR_CHOICES + CATIONIC_CHOICES
        else:
            pool = CATIONIC_CHOICES if rng.random() < 0.7 else POLAR_CHOICES
        return seq[:i] + self._pick(pool, seq[i], rng) + seq[i + 1:]

    def _swap(self, seq: str, rng) -> str:
        """Exchange two residues: arrangement changes, composition does not.

        The purest arrangement move. Net charge, mean hydrophobicity and mass are
        all invariant; only the geometry changes.
        """
        if len(seq) < 2:
            return seq
        i, j = (int(k) for k in rng.choice(len(seq), size=2, replace=False))
        chars = list(seq)
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)

    def _indel(self, seq: str, rng) -> str:
        if rng.random() < 0.5 and len(seq) > self.min_len:
            i = int(rng.integers(len(seq)))
            return seq[:i] + seq[i + 1:]
        if len(seq) < self.max_len:
            i = int(rng.integers(len(seq) + 1))
            return seq[:i] + ALLOWED[int(rng.integers(len(ALLOWED)))] + seq[i:]
        return seq

    def _terminal(self, seq: str, rng) -> str:
        if rng.random() < 0.5 and len(seq) > self.min_len:
            return seq[1:] if rng.random() < 0.5 else seq[:-1]
        if len(seq) < self.max_len:
            aa = ALLOWED[int(rng.integers(len(ALLOWED)))]
            return aa + seq if rng.random() < 0.5 else seq + aa
        return seq


@dataclass
class AnnealingConfig:
    n_steps: int = 400
    t_start: float = 0.06
    t_end: float = 0.002
    softness_start: float = 2.0
    softness_end: float = 0.4
    seed: int = 42


@dataclass
class AnnealingResult:
    best_sequence: str
    best_score: float
    accepted: int
    visited: list = field(default_factory=list)


def anneal(seed_sequence: str, score_fn, config: AnnealingConfig,
           operator: MutationOperator | None = None,
           keep_visited: bool = True) -> AnnealingResult:
    """Simulated annealing with a jointly annealed constraint softness.

    Temperature controls how often a worse move is accepted. Softness controls
    how strictly the physicochemical windows bite: starting loose lets the chain
    cross mildly infeasible ground to reach a distant basin, and tightening
    ensures the answer ends up comfortably inside the envelope.

    score_fn takes (sequence, softness_scale) and returns a float, or negative
    infinity for hard infeasibility.

    Visited sequences are retained because every candidate the search touches is
    a legitimate library member, and the library must contain the top list.
    """
    operator = operator or MutationOperator()
    rng = np.random.default_rng(config.seed)

    current = seed_sequence
    current_score = score_fn(current, config.softness_start)
    best, best_score = current, current_score
    accepted = 0
    visited: list[str] = []

    for step in range(config.n_steps):
        frac = step / max(1, config.n_steps - 1)
        temperature = config.t_start * (config.t_end / config.t_start) ** frac
        softness = config.softness_start * (config.softness_end / config.softness_start) ** frac

        proposal = operator(current, rng)
        if proposal == current:
            continue
        proposal_score = score_fn(proposal, softness)
        if keep_visited and np.isfinite(proposal_score):
            visited.append(proposal)

        delta = proposal_score - current_score
        if delta >= 0 or (
            np.isfinite(proposal_score)
            and np.isfinite(current_score)
            and rng.random() < math.exp(delta / max(temperature, 1e-9))
        ):
            current, current_score = proposal, proposal_score
            accepted += 1
            if current_score > best_score:
                best, best_score = current, current_score

    return AnnealingResult(best, float(best_score), accepted, visited)


def refine_population(seeds, score_fn, config: AnnealingConfig,
                      operator: MutationOperator | None = None,
                      keep_visited: bool = True):
    """One independent chain per seed, with a deterministic per-seed offset."""
    results = []
    for k, seed_sequence in enumerate(seeds):
        cfg = AnnealingConfig(
            n_steps=config.n_steps,
            t_start=config.t_start,
            t_end=config.t_end,
            softness_start=config.softness_start,
            softness_end=config.softness_end,
            seed=config.seed + 1000 * (k + 1),
        )
        results.append(anneal(seed_sequence, score_fn, cfg, operator, keep_visited))
    return results
