# ntt - Nate's Triangulation Toys
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

A collection of triangulation toys for studying lattice point/vector configurations.

## pushing

A C library for constructing pushing triangulations of point/vector configurations,
with optional randomization and fineness.

A *pushing triangulation* assigns an order to the points/vectors, constructing a
simplex from the first $N$ points/vectors and then incrementally adding new ones by
connecting new points/vectors to the externally-visible facets. This also has
interpretation of assigning exponentially-spaced heights $h_i = c^i$ to the vectors, for
sufficiently large $c$. This latter interpretation shows that such triangulations are
regular.

A greedy randomized variant thus gives a cheap source of semi-random fine regular
triangulations. Gives a more diverse sampling than CYTools' `random_triangulations_fast`
while not being prohibitively expensive. See [pushing/README.md](pushing/README.md)
for details and algorithm notes.

No external dependencies for the core library — just a C compiler.

## grow2d

Random fine triangulations of 2D lattice polygons. Originally implemented by me in
[CYTools](https://github.com/LiamMcAllisterGroup/cytools);
ported here as a standalone module. C backend planned.

The algorithm starts from a random unimodular triangle, then iteratively
extends the triangulation by choosing an exterior edge and finding a compatible
vertex — one forming a unit-area triangle whose new edges don't cross any
existing edge. Repeating with different seeds samples the space of fine
triangulations.

Accepts homogenized point configurations (e.g. from `../data/`) and strips
constant coordinates automatically. Includes a live matplotlib plot. Requires
numpy and scipy.

```bash
cd grow2d
python live_triplot.py --data ../data/491_big2face.dat
```

See [grow2d/README.md](grow2d/README.md) for details.

## vcgame

An interactive terminal game built around **triangulations of 3D lattice vector
configurations**. The fan is rendered in real time using a curses-based ASCII renderer.

The player navigates a simplicial fan by moving along geodesics on the 2-sphere. Crossing
a wall between cones performs a **bistellar flip**, modifying the triangulation live. The
fan can be locked for free exploration without flipping.

Requires [regfans](https://github.com/natemacfadden/regfans) and numpy. It works best if
pynput is also installed.

```bash
cd vcgame
python main.py --shape cube --n 5
python main.py --shape trunc_oct
python main.py --shape reflexive --polytope_id 7
```

`vcgame` was built with [Claude Code](https://claude.ai/claude-code).
