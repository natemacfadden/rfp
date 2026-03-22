# Written by Claude Code (claude-sonnet-4-6)

"""Run pushing seed-by-seed on a data file and show a live-updating triplot.

Usage: python3 live_triplot.py [--data <file>] [--n <int>] [--fct <path>]
  --data <file>   Input data file (default: data/491_big2face.dat)
  --n    <int>    Number of seeds to run (default: 100)
  --fct  <path>   Path to pushing binary (default: ./pushing)
"""

import sys
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    data = None
    n    = 100
    fct  = "./pushing"
    args = sys.argv[1:]
    while args:
        if args[0] == "--data" and len(args) > 1:
            data = args[1]; args = args[2:]
        elif args[0] == "--n" and len(args) > 1:
            n = int(args[1]); args = args[2:]
        elif args[0] == "--fct" and len(args) > 1:
            fct = args[1]; args = args[2:]
        else:
            sys.exit(f"Unknown argument: {args[0]}\n{__doc__.strip()}")

    if data is None:
        print("must pass path to data...")
        sys.exit(0)

    return data, n, fct

data_file, n_seeds, fct_bin = parse_args()

with open(data_file) as f:
    raw = f.read().strip()

# parse points from the data file; drop the 0th (homogenizing) coordinate
all_pts = []
for m in re.finditer(r'\[(-?\d+(?:,\s*-?\d+)*)\]', raw):
    coords = [int(x) for x in m.group(1).split(',')]
    all_pts.append(coords[1:])  # drop first coordinate
pts = np.array(all_pts, dtype=float)

plt.ion()
fig, ax = plt.subplots()

for seed in range(n_seeds):
    result = subprocess.run(
        [fct_bin, '--seed', str(seed), raw],
        capture_output=True, text=True
    )
    line = result.stdout.strip()
    if not line:
        continue

    simps = []
    for m in re.finditer(r'\[([^\]]+)\]', line):
        verts = [int(x) for x in m.group(1).split(',') if x.strip()]
        if len(verts) == 3:
            simps.append(verts)
    if not simps:
        continue
    simps = np.array(simps)

    ax.cla()
    ax.set_aspect('auto')
    ax.set_title(f"seed={seed}")
    ax.triplot(pts[:, 0], pts[:, 1], simps, color='steelblue', linewidth=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.05)

plt.ioff()
plt.show()
