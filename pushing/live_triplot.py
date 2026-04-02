# Written by Claude Code (claude-sonnet-4-6)

"""Run pushing seed-by-seed on a data file and show a live-updating triplot.

Usage: python3 live_triplot.py [--data <file>] [--n <int>] [--fct <path>] [--random] [--fine]
  --data    <file>  Input data file (required)
  --n       <int>   Number of seeds to run (default: 100)
  --fct     <path>  Path to pushing binary (default: ./rfp)
  --random          Pass --random to the binary
  --fine            Pass --fine to the binary (implies --random)
"""

import sys
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    data   = None
    n      = 100
    fct    = "./rfp"
    random = False
    fine   = False
    args = sys.argv[1:]
    while args:
        if args[0] == "--data" and len(args) > 1:
            data = args[1]; args = args[2:]
        elif args[0] == "--n" and len(args) > 1:
            n = int(args[1]); args = args[2:]
        elif args[0] == "--fct" and len(args) > 1:
            fct = args[1]; args = args[2:]
        elif args[0] == "--random":
            random = True; args = args[1:]
        elif args[0] == "--fine":
            fine = True; args = args[1:]
        else:
            sys.exit(f"Unknown argument: {args[0]}\n{__doc__.strip()}")

    if data is None:
        sys.exit(f"--data is required\n{__doc__.strip()}")
    if fine:
        random = True  # fine implies random

    return data, n, fct, random, fine

data_file, n_seeds, fct_bin, do_random, do_fine = parse_args()

with open(data_file) as f:
    raw = f.read().strip()

# parse points from the data file; drop the 0th (homogenizing) coordinate
all_pts = []
for m in re.finditer(r'\[(-?\d+(?:,\s*-?\d+)*)\]', raw):
    coords = [int(x) for x in m.group(1).split(',')]
    all_pts.append(coords[1:])  # drop first coordinate
pts = np.array(all_pts, dtype=float)
if pts.ndim != 2 or pts.shape[1] < 2:
    sys.exit(f"Expected 2D points, got shape {pts.shape}")

plt.ion()
fig, ax = plt.subplots()

for seed in range(n_seeds):
    cmd = [fct_bin, '--seed', str(seed)]
    if do_random: cmd.append('--random')
    if do_fine:   cmd.append('--fine')
    cmd.append(raw)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        sys.exit(f"Binary not found: {fct_bin}")
    except subprocess.TimeoutExpired:
        print(f"seed={seed}: timed out, skipping")
        continue

    if result.returncode != 0:
        print(f"seed={seed}: non-zero return code {result.returncode}, skipping")
        continue

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
    mode = "rfp" if do_fine else ("rp" if do_random else "p")
    ax.set_title(f"{mode} seed={seed}")
    ax.triplot(pts[:, 0], pts[:, 1], simps, color='steelblue', linewidth=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.pause(0.05)

plt.ioff()
plt.show()
