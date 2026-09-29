"""Sequence featurisation.

Four tiers of arrangement descriptors, cheapest first, so that an ablation can
attribute predictive power to each:

  tier 1  hydrophobic moment at helical and sheet periodicity, global and windowed
  tier 2  spectral purity, the share of hydrophobicity power at 100 degrees
  tier 3  circular statistics on the helical wheel, including arc width and the
          largest angular gap interrupting the hydrophobic face
  tier 4  H0 persistence of the three dimensional hydrophobic point cloud

Plus composition, Henderson-Hasselbalch net charge, Wimley-White partitioning
energies, Das-Pappu charge patterning, and synthesis liability flags.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    ANIONIC,
    BETA_BRANCHED,
    BOMAN,
    CATIONIC,
    EISENBERG,
    HELIX_DEGREES_PER_RESIDUE,
    HELIX_PROPENSITY,
    HYDROPHOBIC,
    KYTE_DOOLITTLE,
    LIABILITY_MOTIFS,
    OXIDATION_PRONE,
    PKA_CTERM,
    PKA_NTERM,
    PKA_SIDE,
    RESIDUE_MASS,
    SHEET_PROPENSITY,
    WATER_MASS,
    WW_INTERFACE,
    WW_OCTANOL,
)
from .topology import H0_FEATURE_NAMES, h0_descriptors


def molecular_weight(seq: str) -> float:
    """Monoisotopic mass in Da. Required to convert ug/mL activities to uM."""
    return sum(RESIDUE_MASS[a] for a in seq) + WATER_MASS


def net_charge(seq: str, ph: float = 7.4) -> float:
    """Net charge by Henderson-Hasselbalch, with free termini.

    Counting K+R minus D+E is wrong at physiological pH because histidine is
    partially protonated. The C terminus is always a full negative here: the
    challenge forbids amidation, so training data drawn from amidated peptides
    carries a systematic offset of about one charge unit.
    """
    q = 1.0 / (1.0 + 10.0 ** (ph - PKA_NTERM))
    q -= 1.0 / (1.0 + 10.0 ** (PKA_CTERM - ph))
    for aa in seq:
        pka = PKA_SIDE.get(aa)
        if pka is None:
            continue
        if aa in CATIONIC:
            q += 1.0 / (1.0 + 10.0 ** (ph - pka))
        elif aa in ANIONIC:
            q -= 1.0 / (1.0 + 10.0 ** (pka - ph))
    return float(q)


def isoelectric_point(seq: str, lo: float = 0.0, hi: float = 14.0) -> float:
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hydrophobic_moment(seq: str, delta_deg: float = HELIX_DEGREES_PER_RESIDUE) -> float:
    """Eisenberg hydrophobic moment: magnitude of the resultant hydrophobicity
    vector when residue i is placed at angle i * delta."""
    if not seq:
        return 0.0
    h = np.array([EISENBERG[a] for a in seq])
    ang = np.deg2rad(delta_deg) * np.arange(len(seq))
    return float(np.abs((h * np.exp(1j * ang)).sum()) / len(seq))


def windowed_moment(seq: str, window: int = 11, delta_deg: float = HELIX_DEGREES_PER_RESIDUE) -> float:
    """Maximum moment over sliding windows.

    Many natural peptides carry one amphipathic segment flanked by disordered
    tails; the global moment averages that signal away.
    """
    if len(seq) <= window:
        return hydrophobic_moment(seq, delta_deg)
    return max(
        hydrophobic_moment(seq[i : i + window], delta_deg)
        for i in range(len(seq) - window + 1)
    )


def periodicity_purity(seq: str) -> float:
    """Share of mean centred hydrophobicity power sitting at helical periodicity.

    A sequence can reach a respectable moment while being spectrally noisy. This
    separates a clean helical amphipath from an accidental one.
    """
    if len(seq) < 6:
        return 0.0
    h = np.array([EISENBERG[a] for a in seq])
    h = h - h.mean()
    if not np.any(h):
        return 0.0
    idx = np.arange(len(seq))
    angles = np.deg2rad(np.arange(2, 360, 2))
    power = np.array([np.abs((h * np.exp(1j * a * idx)).sum()) ** 2 for a in angles])
    total = power.sum()
    if total <= 0:
        return 0.0
    helical = power[int(np.argmin(np.abs(np.rad2deg(angles) - HELIX_DEGREES_PER_RESIDUE)))]
    return float(helical / total)


def wheel_statistics(seq: str) -> dict[str, float]:
    """Circular statistics of hydrophobic residues on the helical wheel.

    arc_width        angular sector spanned by the hydrophobic face. Around 180
                     degrees is maximally amphipathic and also maximally
                     haemolytic; narrower faces are associated with selectivity.
    largest_gap      biggest angular gap inside the face. This is imperfect
                     amphipathicity made quantitative, which is the classic
                     selectivity preserving modification.
    resultant_length mean resultant vector length of the hydrophobic angles
    circular_var     1 minus resultant length
    """
    angles = np.deg2rad(
        [(i * HELIX_DEGREES_PER_RESIDUE) % 360.0 for i, a in enumerate(seq) if a in HYDROPHOBIC]
    )
    if angles.size < 2:
        return {
            "wheel_arc_width": 0.0,
            "wheel_largest_gap": 0.0,
            "wheel_resultant": 0.0,
            "wheel_circular_var": 1.0,
            "wheel_n_hydrophobic": float(angles.size),
        }

    resultant = float(np.abs(np.exp(1j * angles).mean()))
    ordered = np.sort(angles % (2 * np.pi))
    gaps = np.diff(np.concatenate([ordered, [ordered[0] + 2 * np.pi]]))
    largest_gap = float(gaps.max())
    arc_width = float(2 * np.pi - largest_gap)
    interior = np.sort(gaps)[:-1]
    interior_gap = float(interior.max()) if interior.size else 0.0

    return {
        "wheel_arc_width": np.rad2deg(arc_width),
        "wheel_largest_gap": np.rad2deg(interior_gap),
        "wheel_resultant": resultant,
        "wheel_circular_var": 1.0 - resultant,
        "wheel_n_hydrophobic": float(angles.size),
    }


def _delta(sub_pos: float, sub_neg: float, whole_pos: float, whole_neg: float) -> float:
    fcr_sub = sub_pos + sub_neg
    if fcr_sub == 0:
        return (whole_pos - whole_neg) ** 2
    sigma_sub = (sub_pos - sub_neg) ** 2 / fcr_sub
    fcr_whole = whole_pos + whole_neg
    sigma_whole = (whole_pos - whole_neg) ** 2 / fcr_whole if fcr_whole else 0.0
    return (sigma_sub - sigma_whole) ** 2


def _blob_delta(charges: np.ndarray, blob: int) -> float:
    n = len(charges)
    if n < blob:
        return 0.0
    whole_pos = float((charges > 0).mean())
    whole_neg = float((charges < 0).mean())
    vals = []
    for i in range(n - blob + 1):
        w = charges[i : i + blob]
        vals.append(
            _delta(float((w > 0).mean()), float((w < 0).mean()), whole_pos, whole_neg)
        )
    return float(np.mean(vals))


def kappa(seq: str, blobs: tuple[int, int] = (5, 6)) -> float:
    """Das-Pappu charge patterning parameter.

    Measures how segregated positive and negative residues are along the chain,
    normalised against the maximally segregated permutation of the same
    composition. Zero is perfectly mixed, one is fully segregated. Standard in
    the disordered protein literature and effectively unused in peptide design,
    despite antimicrobial peptides being conditionally disordered polyampholytes.
    """
    charges = np.array([1 if a in "KR" else (-1 if a in "DE" else 0) for a in seq])
    if not np.any(charges):
        return 0.0
    observed = float(np.mean([_blob_delta(charges, b) for b in blobs]))
    pos = int((charges > 0).sum())
    neg = int((charges < 0).sum())
    neutral = len(charges) - pos - neg
    segregated = np.array([1] * pos + [0] * neutral + [-1] * neg)
    reference = float(np.mean([_blob_delta(segregated, b) for b in blobs]))
    return float(observed / reference) if reference > 0 else 0.0


def hydrophobic_blockiness(seq: str, blob: int = 5) -> float:
    """Kappa analogue over the hydrophobic/polar partition.

    The cheap scalar shadow of the H0 face fragmentation descriptor. Included so
    the topological features can be ablated against something simpler.
    """
    binary = np.array([1 if a in HYDROPHOBIC else -1 for a in seq])
    return kappa("".join("K" if b > 0 else "D" for b in binary), blobs=(blob, blob + 1))


def longest_run(seq: str, members: frozenset) -> int:
    best = run = 0
    for a in seq:
        run = run + 1 if a in members else 0
        best = max(best, run)
    return best


def partitioning_energies(seq: str) -> dict[str, float]:
    """Wimley-White transfer free energies.

    A strongly favourable interface energy with a weakly favourable octanol
    energy describes a peptide that parks at the headgroup region, which is
    sufficient to disrupt an anionic bacterial membrane. A strongly favourable
    octanol energy describes deep insertion into the hydrocarbon core, which is
    what lyses erythrocytes. The difference is the selectivity axis expressed
    thermodynamically, and it is measured rather than fitted, so it carries
    information the learned heads cannot get from biased activity databases.
    """
    dg_if = sum(WW_INTERFACE[a] for a in seq)
    dg_oct = sum(WW_OCTANOL[a] for a in seq)
    n = max(len(seq), 1)
    return {
        "dg_interface": dg_if,
        "dg_octanol": dg_oct,
        "dg_interface_per_res": dg_if / n,
        "dg_octanol_per_res": dg_oct / n,
        "interfacial_preference": dg_oct - dg_if,
    }


def synthesis_liabilities(seq: str) -> dict[str, float]:
    """Flags feeding the quality control survival term.

    A candidate that fails synthesis, is insoluble, or misses purity control is
    not retested and its slot in the experimental batch is simply lost, so this
    is an expected value term rather than a nicety.
    """
    n = max(len(seq), 1)
    return {
        "frac_hydrophobic": sum(a in HYDROPHOBIC for a in seq) / n,
        "longest_hydrophobic_run": float(longest_run(seq, HYDROPHOBIC)),
        "longest_beta_branched_run": float(longest_run(seq, BETA_BRANCHED)),
        "n_oxidation_prone": float(sum(a in OXIDATION_PRONE for a in seq)),
        "n_liability_motifs": float(sum(seq.count(m) for m in LIABILITY_MOTIFS)),
        "mean_sheet_propensity": float(np.mean([SHEET_PROPENSITY[a] for a in seq])),
        "length": float(len(seq)),
    }


def composition(seq: str) -> dict[str, float]:
    n = max(len(seq), 1)
    pos = sum(a in "KR" for a in seq)
    neg = sum(a in "DE" for a in seq)
    return {
        "fcr": (pos + neg) / n,
        "ncpr": (pos - neg) / n,
        "frac_cationic": pos / n,
        "frac_anionic": neg / n,
        "frac_aromatic": sum(a in "FWY" for a in seq) / n,
        "frac_glycine": seq.count("G") / n,
        "frac_proline": seq.count("P") / n,
    }


def featurize(seq: str, include_topology: bool = True) -> dict[str, float]:
    """Full descriptor vector for one sequence."""
    feats: dict[str, float] = {
        "net_charge": net_charge(seq),
        "charge_density": net_charge(seq) / max(len(seq), 1),
        "isoelectric_point": isoelectric_point(seq),
        "mean_hydrophobicity": float(np.mean([EISENBERG[a] for a in seq])),
        "gravy": float(np.mean([KYTE_DOOLITTLE[a] for a in seq])),
        "boman": float(np.mean([BOMAN[a] for a in seq])),
        "mean_helix_propensity": float(np.mean([HELIX_PROPENSITY[a] for a in seq])),
        "molecular_weight": molecular_weight(seq),
        "moment_helix": hydrophobic_moment(seq, 100.0),
        "moment_sheet": hydrophobic_moment(seq, 180.0),
        "moment_windowed": windowed_moment(seq),
        "periodicity_purity": periodicity_purity(seq),
        "kappa": kappa(seq),
        "hydrophobic_blockiness": hydrophobic_blockiness(seq),
    }
    feats.update(composition(seq))
    feats.update(wheel_statistics(seq))
    feats.update(partitioning_energies(seq))
    feats.update(synthesis_liabilities(seq))
    if include_topology:
        feats.update(h0_descriptors(seq))
    return feats


def featurize_many(seqs, include_topology: bool = True):
    import pandas as pd

    return pd.DataFrame([featurize(s, include_topology) for s in seqs], index=list(seqs))


def feature_names(include_topology: bool = True) -> list[str]:
    names = list(featurize("KWKLFKKIGIGKFLHSAKKF", include_topology=False).keys())
    if include_topology:
        names += list(H0_FEATURE_NAMES)
    return names


TOPOLOGY_FEATURES = list(H0_FEATURE_NAMES)
WHEEL_FEATURES = [
    "wheel_arc_width",
    "wheel_largest_gap",
    "wheel_resultant",
    "wheel_circular_var",
    "wheel_n_hydrophobic",
]
