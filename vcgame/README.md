# vcgame

A game built around *vector configurations* and their *triangulations*.

The player navigates on a simplicial fan (a triangulation of a 3D lattice
vector configuration) by moving along geodesics on the 2-sphere. Crossing a
wall between cones performs a bistellar flip, changing the triangulation. The
fan can be locked to allow exploration without modification.

Built on [regfans](https://github.com/natemacfadden/regfans) for triangulation
computation.

## Project structure

The pipeline has three stages, each independent:

```
shapes/      stage 1 (generate integer vectors and triangulate into a fan)
renderer/    stage 2 (ASCII rendering of a fan, no game state)
game/        stage 3 (interactive game loop, player, agents)
```

Stages can be used independently. For example, stages 1 and 2 together support
generating, rendering, and saving shapes without any game logic.

## Stage 1 (shape generation)

### CLI

Print integer vectors for a named shape as JSON:

```bash
python -m shapes cube --n 3
python -m shapes cube --n 5
python -m shapes random --seed 42
python -m shapes reflexive --polytope_id 7
python -m shapes trunc_oct
```

Each command prints a JSON array of integer 3-vectors to stdout, e.g.:

```
[[-1,-1,-1], [-1,-1,0], ..., [1,1,1]]
```

### Python API

```python
from shapes import get_vectors, vectors_to_fan, load_shape

# Stage 1: integer vectors only
vectors = get_vectors("cube", n=3)          # list[list[int]]

# Stage 2: triangulate into a fan
fan = vectors_to_fan(vectors)               # regfans.Fan

# Or both in one step
fan = load_shape("cube", n=3)
```

### Available shapes

| Name | Description | Parameters |
|---|---|---|
| `cube` | Boundary lattice points of an n×n×n integer cube | `--n` (odd, ≥ 3, **required**) |
| `random` | Random centrally-symmetric lattice vectors on convex hull | `--seed` |
| `reflexive` | Lattice points of a 3D reflexive polytope (4319 available) | `--polytope_id` (0–4318) |
| `trunc_oct` | Vertices of the truncated octahedron | none |

## Running the game

```bash
python main.py
python main.py --shape trunc_oct
python main.py --shape random --seed 42
python main.py --shape reflexive --polytope_id 7
python main.py --shape cube --n 5 --color 1 --flashlight
```

## Status

Early development. Intended to be small.

## Development note

Much of this project was developed with the assistance of
[Claude Code](https://claude.ai/claude-code) (Anthropic).
