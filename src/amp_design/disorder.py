"""Conditional-disorder descriptors.

Antimicrobial peptides are conditionally disordered: random coil in water,
amphipathic helix on contact with an anionic interface. Every arrangement
feature in features.py places residues on a perfect helix at one hundred degrees
per residue, which is an assumption about a conformation the molecule may rarely
occupy rather than a prediction.

This module models the transition instead of the assumed end state.

The distinctive claim is not that helicity predicts activity. Two crude
helicity proxies, mean Chou-Fasman propensity and the spectral purity of the
hydrophobicity periodogram, are already in the tested feature set and neither
added signal. The claim here is narrower and non-monotone: there is an optimum.
A peptide too helical in water pre-folds, self-associates and lyses erythrocytes;
one too disordered never folds productively at the interface. Only a model that
can represent a turning point can express this, which is why pairwise
correlation could not have detected it.

Helicity is estimated with an Agadir-style Lifson-Roig two-state calculation:
tractable, well-characterised, and dependent only on sequence. The environment
enters through the helix propagation weights, which are scaled to reflect the
lower dielectric and reduced hydrogen-bond competition at a membrane interface.
That scaling is a coarse device, not a simulation, and the quantity of interest
is the difference between the two environments rather than either value.
"""

from __future__ import annotations

import numpy as np

from .constants import CATIONIC, HELIX_PROPENSITY

# Lifson-Roig helix propagation weights, derived from measured helix propensities
# in host-guest peptide systems. Higher means more helix-favouring.
# Alanine is the reference at 1.0; proline and glycine are strong breakers.
LR_W = {
    "A": 1.00, "L": 0.85, "M": 0.82, "R": 0.79, "K": 0.76,
    "Q": 0.74, "E": 0.72, "I": 0.70, "W": 0.68, "F": 0.65,
    "S": 0.55, "H": 0.54, "D": 0.52, "Y": 0.51, "V": 0.50,
    "N": 0.48, "T": 0.46, "C": 0.45, "G": 0.20, "P": 0.02,
}

# Nucleation penalty. Helix formation is cooperative: starting a helix is much
# harder than extending one, which is why short peptides are disordered in water
# even when their residues are helix-favouring.
LR_V_WATER = 0.036
LR_V_INTERFACE = 0.145

# At a membrane interface the effective dielectric drops and water no longer
# competes for backbone hydrogen bonds, so helix propagation is favoured. The
# scaling below is a coarse representation of that shift, not a simulation.
INTERFACE_W_BOOST = 1.45
INTERFACE_PROLINE_RELIEF = 1.0


def _lifson_roig_helicity(seq: str, w_scale: float, v: float,
                          proline_relief: float = 1.0) -> float:
    """Fractional helicity from a Lifson-Roig transfer matrix.

    Three states per residue (coil, helix-start, helix-interior) reduce to a
    two by two transfer matrix in the standard formulation. Returns the mean
    fraction of residues in the helical state.
    """
    n = len(seq)
    if n < 4:
        return 0.0

    weights = []
    for aa in seq:
        w = LR_W.get(aa, 0.5) * w_scale
        if aa == "P":
            w *= proline_relief
        weights.append(max(w, 1e-6))

    # Transfer matrices: [[w, v], [1, 1]] per residue, in the Lifson-Roig
    # two-parameter reduction. Rescaled each step to avoid overflow on long
    # sequences; the rescaling cancels in the ratio below.
    def partition(ws) -> tuple[float, np.ndarray]:
        z = np.array([1.0, 1.0])
        log_scale = 0.0
        for w in ws:
            m = np.array([[w, v], [1.0, 1.0]])
            z = m @ z
            norm = float(z.sum())
            if norm > 0:
                z = z / norm
                log_scale += np.log(norm)
        return log_scale, z

    total_log, _ = partition(weights)

    # Helicity by finite difference: the derivative of log Z with respect to a
    # small uniform boost in w gives the mean number of helical residues.
    eps = 1e-3
    boosted = [w * (1.0 + eps) for w in weights]
    boosted_log, _ = partition(boosted)
    mean_helical = (boosted_log - total_log) / eps

    return float(np.clip(mean_helical / n, 0.0, 1.0))


def helicity_water(seq: str) -> float:
    return _lifson_roig_helicity(seq, 1.0, LR_V_WATER)


def helicity_interface(seq: str) -> float:
    return _lifson_roig_helicity(
        seq, INTERFACE_W_BOOST, LR_V_INTERFACE, INTERFACE_PROLINE_RELIEF
    )


def conditional_disorder_features(seq: str,
                                  water_optimum: float = 0.18) -> dict[str, float]:
    """Descriptors of the disorder to order transition.

    helicity_water        estimated fractional helicity in aqueous solution
    helicity_interface    the same at a membrane-mimetic interface
    helicity_gap          the difference. Large means strongly conditional:
                          disordered free, folded on binding
    helicity_ratio        the same relationship as a ratio, which is scale free
                          with respect to peptide length
    switchability         gap weighted by how close the water-state helicity is
                          to the optimum. Peaks at intermediate water helicity
                          and falls off on both sides, so it encodes a turning
                          point rather than a monotone trend
    water_optimum_dev     squared deviation from the optimum, given to the model
                          directly so a non-monotone response can be fitted
                          without relying on the tree to discover the turning
                          point itself
    prefold_risk          high water helicity combined with high hydrophobicity,
                          the aggregation-prone and haemolytic corner
    charge_helix_coupling cationic residues weighted by local helix propensity,
                          the interface-binding face's charge density
    """
    hw = helicity_water(seq)
    hi = helicity_interface(seq)
    gap = hi - hw

    from .features import net_charge

    hydrophobic_fraction = sum(a in "AILMFVWY" for a in seq) / max(len(seq), 1)
    local_helix = np.array([HELIX_PROPENSITY.get(a, 1.0) for a in seq])
    cationic_mask = np.array([a in CATIONIC for a in seq], dtype=float)

    deviation = (hw - water_optimum) ** 2
    switchability = gap * float(np.exp(-deviation / (2 * 0.10 ** 2)))

    return {
        "helicity_water": hw,
        "helicity_interface": hi,
        "helicity_gap": gap,
        "helicity_ratio": hi / max(hw, 1e-3),
        "switchability": switchability,
        "water_optimum_dev": deviation,
        "prefold_risk": hw * hydrophobic_fraction,
        "charge_helix_coupling": float((cationic_mask * local_helix).sum()) / max(len(seq), 1),
        "helicity_length_product": hi * len(seq),
    }


CONDITIONAL_DISORDER_FEATURES = (
    "helicity_water",
    "helicity_interface",
    "helicity_gap",
    "helicity_ratio",
    "switchability",
    "water_optimum_dev",
    "prefold_risk",
    "charge_helix_coupling",
    "helicity_length_product",
)
