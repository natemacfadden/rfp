# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

"""
grow2d — standalone port from CYTools face_triangulations.py.

Original: cytools/ntfe/face_triangulations.py @ 1e09ca4
Backend: pure Python + NumPy (C port planned later).
"""

import time
import numpy as np

from geometry import get_bdry, intersect, triangle_area_2x


def grow2d(
    pts: np.ndarray,
    bdry: set = None,
    seed: int = None,
    verbosity: int = 0,
) -> set:
    """
    Grow a fine triangulation (FT) of a 2D lattice polygon.

    Arguments
    ---------
    pts : (n, 2) integer array of all lattice points.
    bdry : pre-computed boundary edges (set of frozensets of indices).
        Computed from pts if not provided.
    seed : RNG seed. Defaults to time-based.
    verbosity : 0=silent, 1=milestones, 2=per-edge, 3=per-vertex.

    Returns
    -------
    Set of 3-tuples (i, j, k) with i<j<k — the simplices of the FT.
    Returns a partial triangulation on failure (should be rare).
    """
    if seed is None:
        seed = time.time_ns() % (2**32)
    t0 = time.perf_counter()

    rand_gen = np.random.Generator(np.random.PCG64(seed=seed))

    pts = np.asarray(pts)
    pts_i = list(range(len(pts)))

    # boundary edges
    if bdry is None:
        if verbosity >= 1:
            print(time.perf_counter() - t0, ": Calculating boundary edges...")
        bdry = get_bdry(pts)

    # choose starting simplex: random unit-area triangle
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Choosing starting simplex...")
    while True:
        start = rand_gen.choice(pts_i, 3, replace=False)
        if triangle_area_2x(pts[start]) == 1:
            start = sorted(start)
            break

    simps = {tuple(start)}

    edges = {
        frozenset((start[0], start[1])),
        frozenset((start[0], start[2])),
        frozenset((start[1], start[2])),
    }

    choosable = edges - bdry

    # bounding boxes for intersection pre-filtering
    edges_bounds = {}
    for i in range(2):
        for j in range(i + 1, 3):
            e = frozenset((start[i], start[j]))
            edges_bounds[e] = [
                [min(pts[start[i]][0], pts[start[j]][0]),
                 min(pts[start[i]][1], pts[start[j]][1])],
                [max(pts[start[i]][0], pts[start[j]][0]),
                 max(pts[start[i]][1], pts[start[j]][1])],
            ]

    # grow
    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Growing simplices...")

    while choosable:
        edge = rand_gen.choice(list(choosable))
        edge_lis = list(edge)
        to_try = [i for i in pts_i if i not in edge]
        rand_gen.shuffle(to_try)

        if verbosity >= 2:
            print(f"Building off edge={edge}...")

        while True:
            if not to_try:
                print("Failed! Returning partial triangulation...")
                return simps

            i = to_try.pop()

            if verbosity >= 3:
                print(f"  Trying vertex {i} ({len(to_try)} left) -> ", end="")

            if tuple(sorted([*edge, i])) in simps:
                if verbosity >= 3:
                    print("existing simplex")
                continue

            area_2x = triangle_area_2x(pts[[*edge, i]])
            if area_2x != 1:
                if verbosity >= 3:
                    print(f"area={area_2x}/2, need 1/2")
                continue

            edges_new = [frozenset((e, i)) for e in edge_lis]
            edges_new_bounds = [edges_bounds.get(e, None) for e in edges_new]

            for j in range(2):
                if edges_new_bounds[j] is None:
                    ep = pts[[*edges_new[j]]]
                    edges_bounds[edges_new[j]] = [
                        np.min(ep, axis=0).tolist(),
                        np.max(ep, axis=0).tolist(),
                    ]
                    edges_new_bounds[j] = edges_bounds[edges_new[j]]

            p0i_min, p0i_max = edges_new_bounds[0]
            p1i_min, p1i_max = edges_new_bounds[1]

            any_intersect = False
            for other in edges:
                other_lis = list(other)
                po_min, po_max = edges_bounds[other]

                # bbox pre-check then exact intersection for edge (edge_lis[0], i)
                if not (
                    po_max[0] < p0i_min[0]
                    or po_max[1] < p0i_min[1]
                    or p0i_max[0] < po_min[0]
                    or p0i_max[1] < po_min[1]
                ):
                    if intersect(
                        pts[edge_lis[0]], pts[i],
                        pts[other_lis[0]], pts[other_lis[1]],
                    ):
                        if verbosity >= 3:
                            print(
                                f"edge ({edge_lis[0]},{i}) intersects "
                                f"({other_lis[0]},{other_lis[1]})"
                            )
                        any_intersect = True
                        break

                # bbox pre-check then exact intersection for edge (edge_lis[1], i)
                if not (
                    po_max[0] < p1i_min[0]
                    or po_max[1] < p1i_min[1]
                    or p1i_max[0] < po_min[0]
                    or p1i_max[1] < po_min[1]
                ):
                    if intersect(
                        pts[edge_lis[1]], pts[i],
                        pts[other_lis[0]], pts[other_lis[1]],
                    ):
                        if verbosity >= 3:
                            print(
                                f"edge ({edge_lis[1]},{i}) intersects "
                                f"({other_lis[0]},{other_lis[1]})"
                            )
                        any_intersect = True
                        break

            if not any_intersect:
                if verbosity >= 3:
                    print(f"accepted, adding simplex {sorted([*edge, i])}")
                simps.add(tuple(sorted([*edge, i])))
                choosable.remove(edge)
                edges = edges.union(edges_new)
                choosable ^= set(edges_new) - bdry
                break

    if verbosity >= 1:
        print(time.perf_counter() - t0, ": Done!")

    return simps
