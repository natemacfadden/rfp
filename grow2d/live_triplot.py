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
Live-updating triplot: grow a fine triangulation seed-by-seed and display each.

Usage::

    python live_triplot.py --data <file> [--n <int>] [--seed <int>]

Options
-------
--data <file>
    Input data file (required). One point per line as ``[x, y]``.
--n <int>
    Number of seeds to run (default: 100).
--seed <int>
    Starting seed (default: 0).
"""

import re
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from grow import grow2d
from geometry import get_bdry

# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    data_file : str
    n_seeds : int
    start_seed : int
    """
    data       = None
    n          = 100
    start_seed = 0

    args = sys.argv[1:]
    while args:
        if args[0] == "--data" and len(args) > 1:
            data = args[1]; args = args[2:]
        elif args[0] == "--n" and len(args) > 1:
            n = int(args[1]); args = args[2:]
        elif args[0] == "--seed" and len(args) > 1:
            start_seed = int(args[1]); args = args[2:]
        else:
            sys.exit(f"Unknown argument: {args[0]}\n{__doc__.strip()}")

    if data is None:
        sys.exit(f"--data is required\n{__doc__.strip()}")

    return data, n, start_seed

# =============================================================================
# Main
# =============================================================================

data_file, n_seeds, start_seed = parse_args()

with open(data_file) as f:
    raw = f.read().strip()

# Parse points: each entry is [c0, c1, ...] or [x, y]
pts_list = []
for m in re.finditer(r'\[(-?\d+(?:,\s*-?\d+)*)\]', raw):
    coords = [int(x) for x in m.group(1).split(',')]
    pts_list.append(coords)
pts = np.array(pts_list, dtype=float)

if pts.ndim != 2:
    sys.exit(f"Expected 2D array of points, got shape {pts.shape}")

# strip any constant column (homogenizing coordinate can be anywhere)
if pts.shape[1] > 2:
    varying = np.any(pts != pts[0], axis=0)
    pts = pts[:, varying]

if pts.shape[1] != 2:
    sys.exit(f"Expected 2D points after dehomogenization, got shape {pts.shape}")

# pre-compute boundary (shared across seeds)
bdry = get_bdry(pts.astype(int))

plt.ion()
fig, ax = plt.subplots()

for i, seed in enumerate(range(start_seed, start_seed + n_seeds)):
    simps = grow2d(pts.astype(int), bdry=bdry, seed=seed)
    simps_arr = np.array(sorted(simps))

    ax.cla()
    ax.set_aspect("auto")
    ax.set_title(f"grow2d  seed={seed}  ({i+1}/{n_seeds})")
    ax.triplot(pts[:, 0], pts[:, 1], simps_arr, color="steelblue", linewidth=0.8)
    ax.scatter(pts[:, 0], pts[:, 1], s=18, color="steelblue", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.1)

plt.ioff()
plt.show()
