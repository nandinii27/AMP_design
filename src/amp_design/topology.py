"""Topological descriptors of the hydrophobic face.

The helical wheel is a projection down the helix axis and discards the axial
coordinate. Two hydrophobic residues at the same wheel angle but ten positions
apart project onto the same point yet sit roughly 15 A apart in space, so the
wheel reports a contiguous face where the molecule has two separate patches.

These functions build the real three dimensional point cloud of hydrophobic
side chain centroids and summarise its H0 persistence. H0 of a Vietoris-Rips
filtration is multiscale single linkage clustering: the death times record the
scales at which hydrophobic patches merge into one component. A face that is
genuinely contiguous in space yields one long lived component and short lived
others. A fragmented face yields several components that merge late.

H1 is deliberately not used. On a point cloud of ten to twenty points arranged
on a helix it is dominated by sampling noise and would not survive an ablation.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    HELIX_DEGREES_PER_RESIDUE,
    HELIX_RADIUS_ANGSTROM,
    HELIX_RISE_ANGSTROM,
    HYDROPHOBIC,
)

_GUDHI_AVAILABLE: bool | None = None


def _gudhi():
    global _GUDHI_AVAILABLE
    try:
        import gudhi

        _GUDHI_AVAILABLE = True
        return gudhi
    except ImportError:
        _GUDHI_AVAILABLE = False
        return None


def helix_coordinates(seq: str, radius: float = HELIX_RADIUS_ANGSTROM) -> np.ndarray:
    """Cartesian coordinates of side chain centroids on an idealised alpha helix.

    Returns an (n, 3) array. The x and y coordinates place each residue at its
    wheel angle; z advances along the helix axis.
    """
    idx = np.arange(len(seq), dtype=float)
    theta = np.deg2rad(HELIX_DEGREES_PER_RESIDUE) * idx
    return np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), HELIX_RISE_ANGSTROM * idx]
    )


def hydrophobic_point_cloud(seq: str) -> np.ndarray:
    """Coordinates of hydrophobic residues only."""
    coords = helix_coordinates(seq)
    mask = np.array([a in HYDROPHOBIC for a in seq], dtype=bool)
    return coords[mask]


def _h0_death_times(points: np.ndarray) -> np.ndarray:
    """Finite H0 death times of the Vietoris-Rips filtration.

    Falls back to a direct minimum spanning tree computation when gudhi is
    absent: for H0 the Rips death times are exactly the MST edge weights, so
    the fallback is not an approximation.
    """
    if len(points) < 2:
        return np.array([])

    gudhi = _gudhi()
    if gudhi is not None:
        max_edge = float(np.linalg.norm(points.max(0) - points.min(0))) + 1.0
        rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
        tree = rips.create_simplex_tree(max_dimension=1)
        tree.compute_persistence(persistence_dim_max=False)
        intervals = tree.persistence_intervals_in_dimension(0)
        deaths = np.array([d for _, d in intervals if np.isfinite(d)])
        return np.sort(deaths)

    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import squareform, pdist

    dist = squareform(pdist(points))
    mst = minimum_spanning_tree(dist).toarray()
    return np.sort(mst[mst > 0])


def h0_descriptors(seq: str) -> dict[str, float]:
    """Summary statistics of hydrophobic face connectivity.

    h0_n_components   number of hydrophobic residues, i.e. components at scale 0
    h0_max_death      scale at which the last two patches merge. Large means the
                      face is fragmented in space.
    h0_total_death    sum of merge scales, a global dispersion measure
    h0_mean_death     average merge scale
    h0_gap            largest jump between consecutive merge scales. A large gap
                      separates within patch merges from between patch merges and
                      is the clearest signature of a broken face.
    h0_n_patches      number of components surviving past the gap, i.e. the
                      number of distinct hydrophobic patches
    """
    points = hydrophobic_point_cloud(seq)
    n = len(points)
    empty = {
        "h0_n_components": float(n),
        "h0_max_death": 0.0,
        "h0_total_death": 0.0,
        "h0_mean_death": 0.0,
        "h0_gap": 0.0,
        "h0_n_patches": float(max(n, 1)),
    }
    if n < 3:
        return empty

    deaths = _h0_death_times(points)
    if deaths.size == 0:
        return empty

    if deaths.size > 1:
        diffs = np.diff(deaths)
        gap_idx = int(np.argmax(diffs))
        gap = float(diffs[gap_idx])
        n_patches = float(deaths.size - gap_idx)
    else:
        gap = 0.0
        n_patches = 1.0

    return {
        "h0_n_components": float(n),
        "h0_max_death": float(deaths[-1]),
        "h0_total_death": float(deaths.sum()),
        "h0_mean_death": float(deaths.mean()),
        "h0_gap": gap,
        "h0_n_patches": n_patches,
    }


H0_FEATURE_NAMES = (
    "h0_n_components",
    "h0_max_death",
    "h0_total_death",
    "h0_mean_death",
    "h0_gap",
    "h0_n_patches",
)
