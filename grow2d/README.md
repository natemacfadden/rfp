# grow2d
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

'Grows' random fine triangulations of 2D lattice polygons.

The algorithm starts from a random unimodular triangle, then iteratively extends
the triangulation by choosing an exterior edge of the current simplicial complex and
finding a compatible vertex — one that forms a unit-area triangle and whose connecting
edges don't cross any existing edge. Repeating with different seeds samples the space
of fine triangulations.

Ported from [CYTools](https://github.com/LiamMcAllisterGroup/cytools). C backend
planned.

## Details

Uses 2D geometry (e.g., Pick's theorem) for fast computations.

## Running

For a live plot of triangulations (requires matplotlib, scipy):
```bash
python live_triplot.py --data <file>
```

Data files contain one point per line as `[x, y]`. For example:
```
[0, 0]
[1, 0]
[2, 0]
[0, 1]
[1, 1]
[0, 2]
```

Options:
```
--data <file>    Input data file (required)
--n <int>        Number of seeds to run (default: 100)
--seed <int>     Starting seed (default: 0)
```
