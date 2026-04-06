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

import numpy as np
from scipy.spatial import ConvexHull


def ccw(a, b, c) -> int:
    """Twice the signed area of triangle (a, b, c). Positive = CCW."""
    return int((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def triangle_area_2x(pts) -> int:
    """Twice the area of triangle given by pts[0], pts[1], pts[2]."""
    return abs(ccw(pts[0], pts[1], pts[2]))


def intersect(a, b, c, d) -> bool:
    """
    True if open segment (a,b) properly crosses open segment (c,d).
    Shared endpoints are not considered intersections.
    """
    if set(map(tuple, [a, b])) & set(map(tuple, [c, d])):
        return False
    d1 = ccw(c, d, a)
    d2 = ccw(c, d, b)
    d3 = ccw(a, b, c)
    d4 = ccw(a, b, d)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


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
