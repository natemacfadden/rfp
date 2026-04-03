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
Standalone 2D geometry helpers for grow2d.

Ported from cytools/helpers/basic_geometry.py.
"""

import numpy as np
from scipy.spatial import ConvexHull


def ccw(A, B, C) -> bool:
    """True iff AC is CCW from AB."""
    return (B[0] - A[0]) * (C[1] - A[1]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D) -> bool:
    """
    True iff open segments AB and CD intersect in their strict interiors.
    Returns False if they share an endpoint.
    """
    if (
        (A[0] == C[0] and A[1] == C[1])
        or (A[0] == D[0] and A[1] == D[1])
        or (B[0] == C[0] and B[1] == C[1])
        or (B[0] == D[0] and B[1] == D[1])
    ):
        return False
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def triangle_area_2x(pts) -> float:
    """
    Twice the area of the triangle defined by the 3x2 array pts.
    Uses the shoelace formula.
    """
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    return abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))


def get_bdry(pts: np.ndarray) -> set:
    """
    Compute boundary edges of a 2D convex lattice polygon.

    Walks the convex hull and, for each hull edge, collects all input points
    that lie on that edge segment. Returns a set of frozensets of index pairs.

    Arguments
    ---------
    pts : (n, 2) integer array of all lattice points (boundary + interior).

    Returns
    -------
    Set of frozenset({i, j}) for each primitive boundary edge.
    """
    hull = ConvexHull(pts)
    hull_verts = hull.vertices  # ordered indices into pts

    bdry = set()
    n_hull = len(hull_verts)

    for k in range(n_hull):
        u_idx = hull_verts[k]
        v_idx = hull_verts[(k + 1) % n_hull]
        u = pts[u_idx].astype(float)
        v = pts[v_idx].astype(float)
        dv = v - u

        # find all input points lying on segment u->v
        dp = pts.astype(float) - u  # (n, 2)
        cross = dv[0] * dp[:, 1] - dv[1] * dp[:, 0]
        dot = dp @ dv
        len_sq = float(dv @ dv)

        on_edge = np.where((cross == 0) & (dot >= 0) & (dot <= len_sq))[0]
        params = (dp[on_edge] @ dv) / len_sq
        order = np.argsort(params)
        on_edge = on_edge[order]

        for j in range(len(on_edge) - 1):
            bdry.add(frozenset((int(on_edge[j]), int(on_edge[j + 1]))))

    return bdry
