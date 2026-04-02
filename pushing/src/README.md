The source code for the pushing triangulation program. The library is `pushing.h`. This contains everything necessary to generate (optionally random, optionally random&fine) pushing triangulations of point/vector configurations. If you want better performance for configurations with moderate dimension, run `hardcoded_leibniz.py` which writes a `det.h` file with hardcoded Laplace expansions of the determinant. This gives significant speedups in some cases. If no `det.h` file exists, the code defaults to a cofactor expansion.

As a simple demo, see `demo.c`. This is an example application which reads vectors and then uses `pushing.h` to construct the pushing triangulation(s) of the associated configuration. For some example configurations, see `../data/`.

Also included is `ncube.c`, a small utility that prints the vertices of the $n$-dimensional unit cube $[0,1]^n$ as a vector configuration. Compile with `clang -o ncube ncube.c` and pipe into `rfp`.

A more interactive demo is in `live_triplot.py`, which calls the compiled `rfp` binary repeatedly and plots the resulting triangulations live.
