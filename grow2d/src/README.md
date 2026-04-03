# src

Source code for the grow2d library.

## grow.py

The core library. Implements `grow2d`, which grows a fine triangulation of a
2D lattice polygon by iteratively adding unit-area triangles from a random
starting simplex, checking for edge intersections at each step.

Ported from [CYTools](https://github.com/LiamMcAllisterGroup/cytools)
(`cytools/ntfe/face_triangulations.py` @ `1e09ca4`). C backend planned.

## geometry.py

Standalone 2D geometry helpers used by `grow.py`: `ccw`, `intersect`,
`triangle_area_2x`, and `get_bdry` (boundary edge detection via convex hull).

Ported from CYTools (`cytools/helpers/basic_geometry.py`).
