"""
Generate a vector configuration from lattice points on the surface of
the convex hull of a set of random centrally-symmetric lattice points.

Central symmetry (for every sampled v, -v is also included) guarantees
the origin is strictly inside the hull, so the resulting vectors span
all of R³ and the triangulated fan is complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull

from regfans import VectorConfiguration

if TYPE_CHECKING:
    from regfans import Fan


def _surface_lattice_points(
    pts: list[list[int]],
    hull: ConvexHull,
) -> list[list[int]]:
    """
    Return every non-origin integer point that lies on the surface of
    `hull`.  For each facet plane we enumerate candidate integer points
    and keep those that satisfy all hull inequalities.
    """
    arr = np.array(pts, dtype=float)
    lo  = np.floor(arr.min(axis=0)).astype(int)
    hi  = np.ceil (arr.max(axis=0)).astype(int)

    result: set[tuple[int, ...]] = set()
    eqs = hull.equations  # (nfacets, 4): n·x + d ≤ 0 for interior/surface

    for row in eqs:
        n, d = row[:3], float(row[3])
        # Pivot on the axis with the largest |coefficient| for accuracy.
        pivot = int(np.argmax(np.abs(n)))
        o0, o1 = [i for i in range(3) if i != pivot]

        if abs(n[pivot]) < 1e-9:
            continue

        for v0 in range(lo[o0], hi[o0] + 1):
            for v1 in range(lo[o1], hi[o1] + 1):
                pv  = (-d - n[o0] * v0 - n[o1] * v1) / n[pivot]
                pvi = round(pv)
                if abs(pv - pvi) > 1e-6:
                    continue
                if not (lo[pivot] <= pvi <= hi[pivot]):
                    continue
                p        = [0, 0, 0]
                p[o0]    = v0
                p[o1]    = v1
                p[pivot] = pvi
                p_arr    = np.array(p, dtype=float)
                # Keep only if strictly inside or on all half-spaces.
                if np.all(eqs @ np.append(p_arr, 1.0) <= 1e-6):
                    pt = tuple(p)
                    if pt != (0, 0, 0):
                        result.add(pt)

    return [list(p) for p in result]


def random_vectors(
    seed:      int = 1102,
    n_pairs:   int = 6,
    max_coord: int = 3,
) -> list[list[int]]:
    """
    **Description:**
    Sample `n_pairs` random non-zero integer vectors in
    [−max_coord, max_coord]³ together with their negatives, compute
    the convex hull, and return all non-origin lattice points on the
    hull surface.

    **Arguments:**
    - `seed`: RNG seed (default 1102).
    - `n_pairs`: Number of (v, −v) pairs to seed the hull with.
    - `max_coord`: Coordinate range (inclusive).

    **Returns:**
    A list of integer 3-vectors.
    """
    rng  = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()

    while len(seen) // 2 < n_pairs:
        v  = rng.integers(-max_coord, max_coord + 1, size=3)
        if np.all(v == 0):
            continue
        vt = tuple(v.tolist())
        vm = tuple((-v).tolist())
        seen.add(vt)
        seen.add(vm)

    pts  = [list(p) for p in seen]
    hull = ConvexHull(pts)
    return _surface_lattice_points(pts, hull)


def random_vc(
    seed:      int = 1102,
    n_pairs:   int = 6,
    max_coord: int = 3,
) -> VectorConfiguration:
    """
    **Description:**
    Return the VectorConfiguration of lattice points on the convex hull
    of random centrally-symmetric lattice points.

    **Returns:**
    A VectorConfiguration.
    """
    return VectorConfiguration(random_vectors(seed, n_pairs, max_coord))


def random_fan(
    seed:      int = 1102,
    n_pairs:   int = 6,
    max_coord: int = 3,
) -> Fan:
    """
    **Description:**
    Return a triangulated fan from lattice points on the convex hull of
    random centrally-symmetric lattice points.

    **Returns:**
    A Fan.
    """
    return random_vc(seed, n_pairs, max_coord).triangulate()
