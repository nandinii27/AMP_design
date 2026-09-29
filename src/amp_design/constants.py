"""Amino acid scales, physical constants and challenge rules.

All scales are experimental values from the literature. Nothing here is fitted
to the competition data.
"""

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(ALPHABET)}

# Challenge sequence rules.
MIN_LENGTH = 8
MAX_LENGTH = 50
TOPK_MIN_LENGTH = 12
TOPK_MAX_LENGTH = 30
MAX_IDENTITY_TO_REFERENCE = 0.80
POTENCY_THRESHOLD_UM = 16.0

# Alpha helix geometry.
HELIX_DEGREES_PER_RESIDUE = 100.0
HELIX_RISE_ANGSTROM = 1.5
HELIX_RADIUS_ANGSTROM = 2.3

# Side chain pKa values, plus free termini. The challenge forbids terminal
# modification, so the C terminus always carries a full negative charge.
PKA_SIDE = {
    "K": 10.54, "R": 12.48, "H": 6.04,
    "D": 3.90, "E": 4.07, "C": 8.18, "Y": 10.46,
}
CATIONIC = frozenset("KRH")
ANIONIC = frozenset("DECY")
PKA_NTERM = 9.69
PKA_CTERM = 2.34

# Eisenberg consensus hydrophobicity.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Wimley-White transfer free energies, kcal/mol, whole residue, neutral pH.
# Negative is favourable transfer out of water.
# wif: water to POPC bilayer interface. woct: water to n-octanol.
WW_INTERFACE = {
    "A": 0.17, "R": 0.81, "N": 0.42, "D": 1.23, "C": -0.24,
    "Q": 0.58, "E": 2.02, "G": 0.01, "H": 0.96, "I": -0.31,
    "L": -0.56, "K": 0.99, "M": -0.23, "F": -1.13, "P": 0.45,
    "S": 0.13, "T": 0.14, "W": -1.85, "Y": -0.94, "V": 0.07,
}
WW_OCTANOL = {
    "A": 0.50, "R": 1.81, "N": 0.85, "D": 3.64, "C": -0.02,
    "Q": 0.77, "E": 3.63, "G": 1.15, "H": 2.33, "I": -1.12,
    "L": -1.25, "K": 2.80, "M": -0.67, "F": -1.71, "P": 0.14,
    "S": 0.46, "T": 0.25, "W": -2.09, "Y": -0.71, "V": -0.46,
}

BOMAN = {
    "A": -1.81, "R": 14.92, "N": 6.64, "D": 8.72, "C": -1.28,
    "Q": 5.54, "E": 6.81, "G": -0.94, "H": 4.66, "I": -4.92,
    "L": -4.92, "K": 5.55, "M": -2.35, "F": -2.98, "P": 0.00,
    "S": 3.40, "T": 2.57, "W": -2.33, "Y": -0.14, "V": -4.04,
}

HELIX_PROPENSITY = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
SHEET_PROPENSITY = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Needed to convert ug/mL activities to uM. About 44 percent of the activity
# records arrive in mass units while the potency threshold is molar.
RESIDUE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886,
    "C": 103.1388, "Q": 128.1307, "E": 129.1155, "G": 57.0519,
    "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
    "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782,
    "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
WATER_MASS = 18.0153

HYDROPHOBIC = frozenset("AILMFVWY")
POLAR_NEUTRAL = frozenset("GNQSTP")
BETA_BRANCHED = frozenset("IVT")

# Cysteine is excluded outright: a free thiol dimerises, and the challenge
# forbids any modification that would cap it. Roughly a quarter of the
# reference set contains cysteine, which is a structural class this design
# does not attempt.
FORBIDDEN_RESIDUES = frozenset("C")
OXIDATION_PRONE = frozenset("MW")
LIABILITY_MOTIFS = ("NG", "DP", "DG", "NS")

# PROVISIONAL physicochemical design windows.
#
# The first pass used literature priors of charge 4 to 8 and a moment floor of
# 0.35. Measured against the reference set those were clearly too tight: median
# charge is 2.85 with only 23.5 percent inside the window, and median moment is
# 0.303.
#
# Refitting naively to that distribution would be a different error. About a
# quarter of the reference set is cysteine-rich disulfide-stabilised sheet
# peptides, a class excluded here, whose charge and moment describe molecules
# this design will never emit. The windows below are widened rather than
# recentred, and calibrate_windows() replaces them with quantiles of the correct
# subpopulation: linear, cysteine free, potent and non-haemolytic.
DEFAULT_WINDOWS = {
    "net_charge": (3.0, 9.0),
    "mean_hydrophobicity": (-0.10, 0.55),
    "max_hydrophobic_run": (0.0, 4.0),
    "boman": (0.0, 3.2),
}
MOMENT_FLOOR = 0.25
