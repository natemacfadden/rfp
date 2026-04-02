# ntt
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
triangulations — useful as an alternative to full flip-graph traversal. See
[pushing/README.md](pushing/README.md) for details and algorithm notes.

Example configurations (various dimensions) are provided in `data/`. Compile and run with:

```bash
clang -o rfp pushing/src/demo.c
./data/ncube 5 | ./rfp -n 1000
```

or, more interactively

```
python pushing/live_triplot.py --n 1000
```

No external dependencies — just a C compiler.

## vcgame

An interactive terminal game built around **triangulations of 3D lattice vector
configurations**. The fan is rendered in real time using a curses-based ASCII renderer.

The player navigates a simplicial fan by moving along geodesics on the 2-sphere. Crossing
a wall between cones performs a **bistellar flip**, modifying the triangulation live. The
fan can be locked for free exploration without flipping.

Requires [regfans](https://github.com/natemacfadden/regfans), numpy, and pynput.

```bash
cd vcgame
python main.py --shape cube --n 5
python main.py --shape trunc_oct
python main.py --shape reflexive --polytope_id 7
```

Built on [regfans](https://github.com/natemacfadden/regfans). Much of this project was
developed with the assistance of [Claude Code](https://claude.ai/claude-code) (Anthropic).

## grow2d *(coming soon)*

Random triangulations of lattice polygons.
