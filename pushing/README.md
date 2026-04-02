# Overview
Constructs a pushing triangulation of a point/vector configuration. Optional configurations include
- constructing a *random* pushing triangulation
- constructing a random & *fine* pushing triangulation (or at least try to).

The final call is likely the most interesting, as this is a cheap way to get semi-random fine regular triangulations of arbitrary-dimensional PCs. The tech works for VCs, too, but sometimes gets caught in deadlocks where no new vectors can be pushed while preserving fineness. This final method (specifically for PCs) was inspired by [TOPCOM](https://www.wm.uni-bayreuth.de/de/team/rambau_joerg/TOPCOM/)'s fine triangulation method.

## Details
Definitions (see [DRS](https://doi.org/10.1007/978-3-642-12971-1)):
- a *vector configuration* (VC) is a collection of labeled vectors. Its support is a a convex cone
- a *point configuration* (VC) is a collection of labeled points. Its support is a a polytope. Any PC can be treated as a VC by 'homogenization' - replace $pt \leftarrow (1,pt)$.
- a *triangulation* is decomposition of the support of the VC into simplicial sub-regions. I.e., into simplicial cones
- a *fine* triangulation 'uses' all vectors in the VC. I.e., each vector is a generator of a simplicial cone
- a *regular* triangulation can be constructed via a lifting procedure (embed the vectors in 1-higher dimension, assign heights to them, treat the lower facets as the triangulation)
- a *pushing* triangulation, for some order of the vectors, assigns heights $h_i = c^i$ for sufficiently large $c>0$. This also has a combinatorial characterization.

Pushing triangulations are interesting because they are always regular.

## Warnings
As with TOPCOM, this also applies to point configurations (PCs/polytopes) if you pass the homogenized points. I.e., put all the points on some affine hyperplane. Easiest way is to take each point and append/prepend them all, uniformly, with a component $1$. E.g., represent the PC $[[0,0], [0,1], [1,0], [1,1]]$ as the 'acyclic VC' $[[0,0,1], [0,1,1], [1,0,1], [1,1,1]]$.

The greedy method seems to always generate a fine triangulation for a PC (i.e., an acyclic VC). For general VCs, sometimes the greedy algorithm gets stuck in a situation where it cannot add more vectors due to the previously added vectors.

No duplication checking is performed...

## Compilation/running
Compile a demo with
`clang -g -o rfp src/demo.c `

Run this with (assuming you have also compile `ncube`... if not, go to the data directory and run `clang -g -o ncube ncube.c `)
`./data/ncube 5 | ./rfp -n 1000`
or even
`python live_triplot.py --n 1000`
