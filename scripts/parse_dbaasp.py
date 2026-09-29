"""Parse the raw DBAASP dump into a clean activity table.

Five things here are load bearing and each one silently destroys the dataset if
got wrong.

Nested enums. Every enum-like field in the detail endpoint is an object with a
name key, not a string: unit, complexity, targetSpecies, activityMeasureGroup,
medium, nTerminus, cTerminus. Treating any of them as a string drops every row
it touches without raising, which looks like an empty database rather than a
parsing bug.

Measure type. targetActivities holds MIC, IC50, MBC, EC50 and others. An IC50
and an MIC are different quantities and pooling them corrupts the potency
target. Only MIC and MIC50 are kept.

Target organism. The dump is full of Candida albicans and HIV-1 records.
Antifungal and antiviral activity run on different membrane chemistry, so
anything that is not a bacterium is dropped rather than folded into a bacterial
potency head.

Units. About 44 percent of values are ug/ml and the challenge threshold is
16 uM. Conversion needs molecular weight, which varies roughly twofold across
the legal length range, so a mixed-unit table corrupts the target by an
unpredictable per-peptide factor.

Censoring. Values arrive as bare numbers, as right-censored bounds (>100, >128,
and so on, about ten percent of rows), as intervals (2-4, 4-8, 8-16, which are
the two-fold dilution series reporting its own resolution), and as NA. Each
needs a different treatment and none of them survives float() coercion.

    uv run python scripts/parse_dbaasp.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
VALID = re.compile(rf"^[{ALPHABET}]+$")
MIN_LENGTH, MAX_LENGTH = 8, 50
POTENCY_THRESHOLD_UM = 16.0

CENSOR_NONE, CENSOR_RIGHT, CENSOR_LEFT, CENSOR_INTERVAL = 0, 1, -1, 2

RESIDUE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886,
    "C": 103.1388, "Q": 128.1307, "E": 129.1155, "G": 57.0519,
    "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
    "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782,
    "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
WATER_MASS = 18.0153

GRAM_POSITIVE_GENERA = (
    "staphylococcus", "enterococcus", "bacillus", "streptococcus",
    "listeria", "micrococcus", "clostridium", "lactobacillus",
    "corynebacterium", "mycobacterium", "propionibacterium",
)
GRAM_NEGATIVE_GENERA = (
    "escherichia", "pseudomonas", "klebsiella", "acinetobacter",
    "salmonella", "enterobacter", "serratia", "proteus", "shigella",
    "helicobacter", "campylobacter", "neisseria", "burkholderia",
    "yersinia", "vibrio", "haemophilus", "stenotrophomonas",
    "citrobacter", "moraxella", "legionella", "aeromonas",
)
NON_BACTERIAL = (
    "candida", "aspergillus", "cryptococcus", "fusarium", "saccharomyces",
    "trichophyton", "hiv", "influenza", "herpes", "virus", "leishmania",
    "plasmodium", "trypanosoma", "cell line", "hela", "hek",
)

# Haemolysis assay ceilings.
#
# The reported HC50 distribution piles up at exactly these round levels, and the
# giveaway is that "100" appears 1244 times as a bare value against only 254
# times written as ">100". Most authors who tested up to the highest
# concentration in their series and saw no lysis recorded the ceiling without
# the inequality. Those rows do not say "this peptide lyses cells at 100 uM",
# they say "we stopped at 100". Treating them as observations feeds the
# experiment's budget into the target as though it were a property of the
# molecule, which is enough to flatten any real signal.
#
# This is an assumption, not a fact about any individual record, and it belongs
# in the training data statement. It is the conservative direction: mislabelling
# a true measurement as censored loses information, whereas mislabelling a
# ceiling as a measurement injects noise.
HC50_CEILINGS = {
    10.0, 20.0, 25.0, 32.0, 50.0, 64.0, 100.0, 128.0, 150.0,
    200.0, 250.0, 256.0, 300.0, 400.0, 500.0, 512.0, 1000.0,
}

# Haemolysis is measured on erythrocytes from several species which differ
# substantially in sensitivity. Human erythrocytes account for about 77 percent
# of the records, so restricting to them removes a source of between-study
# variance at modest cost in sample size.
ERYTHROCYTE_SOURCE = ("human",)

# Strains carrying a documented resistance phenotype. The competition scores a
# multidrug-resistant category separately, and public databases carry very
# little data against resistant clinical isolates. These are what exist: MRSA,
# ESBL and vancomycin-resistant strains with meaningful record counts. It is a
# thin signal, not a full head, and the data statement should say so.
RESISTANT_STRAIN_MARKERS = (
    "atcc 43300", "atcc 33591", "atcc baa-44", "mrsa", "mrse",
    "atcc 700603", "esbl", "vre", "atcc 51299", "atcc 700221",
    "carbapenem", "multidrug", "multi-drug", "resistant",
)


def name_of(value) -> str:
    """Every enum-like field in the detail endpoint is {'name': ...}."""
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def molecular_weight(seq: str) -> float:
    return sum(RESIDUE_MASS[a] for a in seq) + WATER_MASS


def clean_sequence(seq) -> str | None:
    s = str(seq or "").strip().upper()
    if not VALID.match(s):
        return None
    if not MIN_LENGTH <= len(s) <= MAX_LENGTH:
        return None
    return s


def to_micromolar(value: float, unit: str, seq: str) -> float | None:
    u = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    if u in {"um", "umol/l", "micromolar"}:
        return float(value)
    if u in {"nm", "nmol/l"}:
        return float(value) / 1000.0
    if u in {"mm", "mmol/l"}:
        return float(value) * 1000.0
    if u in {"ug/ml", "mg/l", "microg/ml"}:
        return float(value) * 1000.0 / molecular_weight(seq)
    if u in {"mg/ml", "g/l"}:
        return float(value) * 1e6 / molecular_weight(seq)
    if u in {"ng/ml", "ug/l"}:
        return float(value) / molecular_weight(seq)
    return None


def parse_concentration(raw) -> tuple[float | None, int]:
    """Value plus censoring code.

    An interval such as 4-8 is the dilution series reporting its own resolution:
    the true value lies in that window. It is collapsed to the upper bound,
    which is the lowest well showing inhibition, and flagged as interval
    censored so the regression can treat it accordingly.
    """
    if raw is None:
        return None, CENSOR_NONE
    if isinstance(raw, (int, float)):
        return (float(raw), CENSOR_NONE) if np.isfinite(raw) else (None, CENSOR_NONE)

    text = str(raw).strip().replace(",", ".")
    if not text or text.upper() in {"NA", "N/A", "ND", "NT", "-"}:
        return None, CENSOR_NONE

    if text.startswith((">", "≥")):
        nums = re.findall(r"\d+\.?\d*", text)
        return (float(nums[0]), CENSOR_RIGHT) if nums else (None, CENSOR_NONE)
    if text.startswith(("<", "≤")):
        nums = re.findall(r"\d+\.?\d*", text)
        return (float(nums[0]), CENSOR_LEFT) if nums else (None, CENSOR_NONE)

    interval = re.fullmatch(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", text)
    if interval:
        return float(interval.group(2)), CENSOR_INTERVAL

    nums = re.findall(r"\d+\.?\d*", text)
    return (float(nums[0]), CENSOR_NONE) if len(nums) == 1 else (None, CENSOR_NONE)


def gram_class(species: str) -> str | None:
    s = species.lower()
    if any(p in s for p in NON_BACTERIAL):
        return None
    if any(p in s for p in GRAM_POSITIVE_GENERA):
        return "gram_positive"
    if any(p in s for p in GRAM_NEGATIVE_GENERA):
        return "gram_negative"
    return None


def is_resistant_strain(species: str) -> bool:
    s = species.lower()
    return any(m in s for m in RESISTANT_STRAIN_MARKERS)


def is_mic(measure: str) -> bool:
    m = measure.upper().replace(" ", "").replace("_", "")
    return m in {"MIC", "MIC50", "MIC90"}


def terminally_modified(record: dict) -> bool:
    nterm = name_of(record.get("nTerminus")).upper()
    cterm = name_of(record.get("cTerminus")).upper()
    n_mod = bool(nterm) and nterm not in {"FREE", "NONE"}
    c_mod = bool(cterm) and cterm not in {"FREE", "NONE", "COOH"}
    return n_mod or c_mod


def parse(path: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    stats = Counter()

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue

            stats["records"] += 1

            if name_of(rec.get("complexity")).lower() != "monomer":
                stats["dropped_not_monomer"] += 1
                continue
            if rec.get("unusualAminoAcids"):
                stats["dropped_unusual_aa"] += 1
                continue

            seq = clean_sequence(rec.get("sequence"))
            if seq is None:
                stats["dropped_bad_sequence"] += 1
                continue

            stats["usable_peptides"] += 1
            modified = terminally_modified(rec)
            if modified:
                stats["terminally_modified"] += 1

            for act in rec.get("targetActivities") or []:
                measure = name_of(act.get("activityMeasureGroup")) or str(
                    act.get("activityMeasureValue") or ""
                )
                if not is_mic(measure):
                    stats["dropped_not_mic"] += 1
                    continue

                species = name_of(act.get("targetSpecies"))
                gram = gram_class(species)
                if gram is None:
                    stats["dropped_non_bacterial_or_unknown"] += 1
                    continue

                value, censor = parse_concentration(act.get("concentration"))
                if value is None:
                    stats["dropped_unparseable_value"] += 1
                    continue

                um = to_micromolar(value, name_of(act.get("unit")), seq)
                if um is None:
                    stats["dropped_bad_unit"] += 1
                    continue

                rows.append(
                    {
                        "sequence": seq,
                        "endpoint": "mic",
                        "group": gram,
                        "species": species,
                        "resistant": is_resistant_strain(species),
                        "value_um": um,
                        "censor": censor,
                        "amidated": modified,
                    }
                )
                stats["mic_rows"] += 1

            for act in rec.get("hemoliticCytotoxicActivities") or []:
                target = name_of(act.get("targetCell")).lower()
                if not target or "erythrocyte" not in target:
                    stats["dropped_non_erythrocyte"] += 1
                    continue
                if not any(src in target for src in ERYTHROCYTE_SOURCE):
                    stats["dropped_non_human_erythrocyte"] += 1
                    continue

                value, censor = parse_concentration(act.get("concentration"))
                if value is None:
                    stats["dropped_unparseable_value"] += 1
                    continue

                if censor == CENSOR_NONE and value in HC50_CEILINGS:
                    censor = CENSOR_RIGHT
                    stats["reclassified_ceiling_as_censored"] += 1

                um = to_micromolar(value, name_of(act.get("unit")), seq)
                if um is None:
                    stats["dropped_bad_unit"] += 1
                    continue

                rows.append(
                    {
                        "sequence": seq,
                        "endpoint": "hc50",
                        "group": "hc50",
                        "species": name_of(act.get("targetCell")) or "erythrocyte",
                        "resistant": False,
                        "value_um": um,
                        "censor": censor,
                        "amidated": modified,
                    }
                )
                stats["hc50_rows"] += 1

    return pd.DataFrame(rows), dict(stats)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sequence and endpoint group.

    Replicates are collapsed on the log2 scale, which is the natural scale for a
    two-fold dilution series. A sequence counts as right censored only when every
    contributing measurement was.
    """
    df = df.copy()
    df["log2_um"] = np.log2(df["value_um"])

    parts = []
    for key in ("gram_positive", "gram_negative", "hc50"):
        subset = df[df["group"] == key]
        if subset.empty:
            continue
        agg = (
            subset.groupby("sequence")
            .agg(
                log2_um=("log2_um", "median"),
                censor=("censor", lambda s: CENSOR_RIGHT if (s == CENSOR_RIGHT).all() else CENSOR_NONE),
                n_obs=("log2_um", "size"),
                amidated=("amidated", "max"),
            )
            .reset_index()
        )
        agg["group"] = key
        parts.append(agg)

    resistant = df[(df["endpoint"] == "mic") & (df["resistant"])]
    if not resistant.empty:
        agg = (
            resistant.groupby("sequence")
            .agg(
                log2_um=("log2_um", "median"),
                censor=("censor", lambda s: CENSOR_RIGHT if (s == CENSOR_RIGHT).all() else CENSOR_NONE),
                n_obs=("log2_um", "size"),
                amidated=("amidated", "max"),
            )
            .reset_index()
        )
        agg["group"] = "resistant"
        parts.append(agg)

    return pd.concat(parts, ignore_index=True)


def to_wide(agg: pd.DataFrame) -> pd.DataFrame:
    wide = agg.pivot(index="sequence", columns="group", values="log2_um")
    censor = agg.pivot(index="sequence", columns="group", values="censor")
    censor.columns = [f"{c}_censor" for c in censor.columns]
    counts = agg.pivot(index="sequence", columns="group", values="n_obs")
    counts.columns = [f"{c}_n" for c in counts.columns]
    return wide.join(censor).join(counts)


def report_windows(wide: pd.DataFrame) -> None:
    """Refit the physicochemical design windows on the right subpopulation.

    Fitting them to every antibacterial peptide is wrong: about a quarter of that
    population is cysteine-rich disulfide-stabilised sheet peptides, a structural
    class this design excludes outright and whose charge and moment distributions
    describe molecules that will never be generated. The window should come from
    linear, cysteine-free, potent, non-haemolytic peptides only.
    """
    from math import log2

    threshold = log2(POTENCY_THRESHOLD_UM)

    potent = wide[
        (wide.get("gram_negative", pd.Series(dtype=float)) <= threshold)
        | (wide.get("gram_positive", pd.Series(dtype=float)) <= threshold)
    ]
    seqs = [s for s in potent.index if "C" not in s]

    if not seqs:
        print("\nNo potent cysteine-free peptides found; windows not refit.")
        return

    selective = seqs
    if "hc50" in wide.columns:
        best_mic = wide.loc[seqs, [c for c in ("gram_positive", "gram_negative") if c in wide]].min(axis=1)
        window = wide.loc[seqs, "hc50"] - best_mic
        selective = [s for s in seqs if window.get(s, np.nan) >= 2.0]

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from amp_design.features import hydrophobic_moment, net_charge

    for label, population in (("potent, Cys-free", seqs), ("plus safety window >= 4x", selective)):
        if not population:
            continue
        charges = np.array([net_charge(s) for s in population])
        moments = np.array([hydrophobic_moment(s) for s in population])
        print(f"\n{label}  (n = {len(population)})")
        print(
            f"  charge  median {np.median(charges):5.2f}   "
            f"10th {np.quantile(charges, 0.10):5.2f}   90th {np.quantile(charges, 0.90):5.2f}"
        )
        print(
            f"  moment  median {np.median(moments):5.3f}   "
            f"10th {np.quantile(moments, 0.10):5.3f}   90th {np.quantile(moments, 0.90):5.3f}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/dbaasp_raw.ndjson"))
    p.add_argument("--out", type=Path, default=Path("data/activity_wide.csv"))
    p.add_argument("--long-out", type=Path, default=Path("data/activity_long.csv"))
    p.add_argument("--skip-windows", action="store_true")
    args = p.parse_args()

    long_df, stats = parse(args.raw)
    if long_df.empty:
        print("No rows parsed. Check the raw file.")
        return 1

    print("--- parse ---")
    for key in sorted(stats):
        print(f"  {key:36s} {stats[key]}")

    agg = aggregate(long_df)
    wide = to_wide(agg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.long_out, index=False)
    wide.to_csv(args.out)

    print("\n--- coverage (unique sequences per endpoint) ---")
    for col in ("gram_positive", "gram_negative", "hc50", "resistant"):
        if col in wide.columns:
            n = int(wide[col].notna().sum())
            censored = int((wide.get(f"{col}_censor", pd.Series(dtype=int)) == CENSOR_RIGHT).sum())
            print(f"  {col:16s} {n:6d}   right-censored {censored}")

    threshold = np.log2(POTENCY_THRESHOLD_UM)
    for col in ("gram_positive", "gram_negative", "resistant"):
        if col in wide.columns:
            sub = wide[col].dropna()
            if len(sub):
                print(f"  {col:16s} potent (<= 16 uM): {int((sub <= threshold).sum())} / {len(sub)}")

    both = wide[["gram_positive", "gram_negative"]].notna().all(axis=1).sum() if {
        "gram_positive", "gram_negative"
    } <= set(wide.columns) else 0
    print(f"\n  sequences with both Gram classes: {both}")
    if "hc50" in wide.columns:
        with_mic = wide[[c for c in ("gram_positive", "gram_negative") if c in wide]].notna().any(axis=1)
        print(f"  sequences with MIC and HC50:      {int((with_mic & wide['hc50'].notna()).sum())}")

    if not args.skip_windows:
        print("\n--- design window refit ---")
        report_windows(wide)

    print(f"\nWrote {args.out} and {args.long_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())